from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from dnn_argparser import *
# Keras stuff
import keras
from keras.models import load_model
from load_dataset import load_and_preprocess_dataset
import numpy as np
from noisy_sgd import NoisySGD
from sparse_layer import Sparse, SparseConv2D, SparseDepthwiseConv2D
from replace_dense_with_sparse import replace_dense_with_sparse
from rewiring_callback import RewiringCallback
# Import OS to deal with directories
import os
import tensorflow as tf
from keras import backend as K
import pylab as plt
from utilities import generate_filename

start_time = plt.datetime.datetime.now()
# Get number of cores reserved by the batch system
# (NSLOTS is automatically set, or use 4 otherwise)
NUMCORES = int(os.getenv("NSLOTS", 4))
print("Using", NUMCORES, "core(s)")

# Create TF session using correct number of cores
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
                                        intra_op_parallelism_threads=NUMCORES, allow_soft_placement=True,
                                        device_count={'CPU': NUMCORES}))

# Set the Keras TF session
K.set_session(sess)

# Checking directory structure exists
if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if not os.path.isdir(args.model_dir) and not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)

epochs = args.epochs or 10
# https://github.com/Zehaos/MobileNet/blob/master/train_image_classifier.py#L191-L192
# batch = 32
batch = args.batch or 32
learning_rate = args.lr  # the default is None from argparser
decay_rate = 0.8  # changed from 0.8

p_0 = args.conn_level or .10  # global connectivity level
print("Flat connectivity level", p_0)
builtin_sparsity = [p_0] * 8
alphas = [0, 10 ** -7, 10 ** -6, 10 ** -9, 0]
final_conns = np.asarray(builtin_sparsity)
conn_decay_values = None
if args.conn_decay:
    conn_decay_values = (np.log(1. / final_conns) / epochs).tolist()
    builtin_sparsity = np.ones(len(conn_decay_values)).tolist()

# Check whether the model that has been provided to argparser is .hdf5 / .h5 on
# disk or a reference to Keras Mobilenet (i.e. :mobilenet)
_is_builtin_model = False


# Add LR reduction schedule based on Inception paper

def lr_reduction_schedule(epoch, lr):
    """
    a function that takes an epoch index as input (integer, indexed from 0)
    and current learning rate and
    returns a new learning rate as output (float).
    :param epoch: epoch index (indexed from 0)
    :type epoch: int
    :param lr: current learning rate
    :type lr: float
    :return: new learning rate
    :rtype: float
    """
    if epoch % 7 == 0:
        return lr * .96


if args.continue_from_epoch != 0:
    for _previous_epochs in range(args.continue_from_epoch):
        learning_rate = lr_reduction_schedule(_previous_epochs, learning_rate)

if args.model[0] == ":":
    model_name = args.model[1:]
    _is_builtin_model = True
else:
    print("Continuing training on model", args.model)
    model_name = "mobilenet_cont"
    # Based on the model name we could infer a re-starting epoch
    # TODO infer epoch number of saved model

# Dense model
if _is_builtin_model:
    # Is a built-in model = load from keras
    if args.random_weights:
        model = keras.applications.MobileNet(weights=None)
    else:
        model = keras.applications.MobileNet()
else:
    # The model is not built-in = load from disk
    # Just in case, load our usual custom objects
    c_obj = {'Sparse': Sparse,
             'SparseConv2D': SparseConv2D,
             'SparseDepthwiseConv2D': SparseDepthwiseConv2D,
             'NoisySGD': NoisySGD}
    # Retrieve model from disk
    model = load_model(args.model, custom_objects=c_obj)

if args.sparse_layers and not args.soft_rewiring:
    if args.conn_decay:
        print("Connectivity decay rewiring enabled", conn_decay_values)
        model = replace_dense_with_sparse(
            model,
            activation=args.activation, batch_size=batch,
            builtin_sparsity=builtin_sparsity,
            reg_coeffs=alphas,
            conn_decay=conn_decay_values, no_cache=args.no_cache)
    else:
        model = replace_dense_with_sparse(
            model,
            activation=args.activation, batch_size=batch,
            builtin_sparsity=builtin_sparsity,
            reg_coeffs=alphas, no_cache=args.no_cache)
elif args.sparse_layers and args.soft_rewiring:
    print("Soft rewiring enabled", args.soft_rewiring)
    model = replace_dense_with_sparse(
        model,
        activation=args.activation, batch_size=batch,
        reg_coeffs=alphas, no_cache=args.no_cache)
model.summary()

dataset_info = load_and_preprocess_dataset(
    'imagenet', batch_size=batch, path=args.dataset_path,
    steps_per_epoch=args.steps_per_epoch,
    val_steps_per_epoch=args.val_steps_per_epoch)
train_gen = dataset_info['train']
val_gen = dataset_info['val']
# test_gen = dataset_info['test']
input_shape = dataset_info['input_shape']
num_classes = dataset_info['num_classes']
no_train = dataset_info['no_train']
no_val = dataset_info['no_val']
# no_test = dataset_info['no_test']


# set up steps_per_epoch
steps_per_epoch = args.steps_per_epoch or no_train // batch
validation_steps_per_epoch = args.val_steps_per_epoch or no_val // batch

print("Training Steps per epoch", steps_per_epoch)
print("Validation Steps per epoch", validation_steps_per_epoch)
print("Number of classes:", num_classes)
print("Number of training examples:", no_train)
print("Number of validation examples:", no_val)

activation_name = "relu"
loss_name = "crossent"

if args.optimizer.lower() == "sgd":
    if learning_rate:
        optimizer = keras.optimizers.SGD(lr=learning_rate)
    else:
        optimizer = keras.optimizers.SGD()
    optimizer_name = "sgd"

elif args.optimizer.lower() in ["ada", "adadelta"]:
    optimizer = keras.optimizers.adadelta()
    optimizer_name = "adadelta"
elif args.optimizer.lower() in ["adam"]:
    optimizer = keras.optimizers.adam()
    optimizer_name = "adam"
elif args.optimizer.lower() in ["noisy_sgd", "ns"]:
    # custom optimizer to include noise and temperature
    from noisy_sgd import NoisySGD

    if learning_rate:
        optimizer = NoisySGD(lr=learning_rate)
    else:
        optimizer = NoisySGD()
    optimizer_name = "noisy_sgd"
elif args.optimizer.lower() in ["rms", "rms_prop", "rmsprop"]:
    optimizer_name = "rms_prop"
    # https://github.com/Zehaos/MobileNet/blob/master/train_image_classifier.py#L307-L312
    optimizer = keras.optimizers.RMSprop(
        # lr=0.01,
        # decay=0.9, epsilon=1.0
    )
else:
    optimizer = args.optimizer
    optimizer_name = args.optimizer

loss = keras.losses.categorical_crossentropy

# disable rewiring with sparse layers to see the performance of the layer
# when 90% of connections are disabled and static
deep_r = RewiringCallback(fixed_conn=args.disable_rewiring,
                          soft_limit=args.soft_rewiring,
                          asserts_on=args.asserts_on)

lr_schedule = LearningRateScheduler(lr_reduction_schedule, verbose=1)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

suffix = ""
if args.suffix:
    suffix = "_" + args.suffix

output_filename = ""
if args.result_filename:
    output_filename = args.result_filename
else:
    output_filename = "results_for_" + model_name
if args.sparse_layers:
    if args.soft_rewiring:
        sparse_name = "sparse_soft"
    else:
        if args.conn_decay:
            sparse_name = "sparse_decay"
        else:
            sparse_name = "sparse_hard"
else:
    sparse_name = "dense"

__acr_filename = "models/" + generate_filename(
    optimizer_name, activation_name, sparse_name, loss_name, suffix,
    args.random_weights,
    acronym=True)
checkpoint_filename = __acr_filename + \
                      "_weights.{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint_callback = ModelCheckpoint(checkpoint_filename, period=5)

__filename = generate_filename(
    optimizer_name, activation_name, sparse_name, loss_name, suffix,
    args.random_weights)
output_filename += __filename
output_filename += ".csv"

csv_path = os.path.join(args.result_dir, output_filename)
csv_logger = keras.callbacks.CSVLogger(
    csv_path,
    separator=',',
    append=True)

callback_list = []
if args.sparse_layers:
    callback_list.append(deep_r)

if args.tensorboard:
    tb_log_filename = "./sparse_logs" if args.sparse_layers else "./dense_logs"

    tb = keras.callbacks.TensorBoard(
        log_dir=tb_log_filename,
        histogram_freq=0,  # turning this on needs validation_data in model.fit
        batch_size=batch, write_graph=True,
        write_grads=True, write_images=True,
        embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None,
        update_freq='epoch')
    callback_list.append(tb)

callback_list.append(csv_logger)
callback_list.append(checkpoint_callback)
callback_list.append(lr_schedule)

if not args.data_augmentation:

    model.fit_generator(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callback_list,
                        validation_data=val_gen,
                        validation_steps=validation_steps_per_epoch,
                        shuffle=True,
                        max_queue_size=10,
                        use_multiprocessing=True,
                        workers=1,
                        initial_epoch=args.continue_from_epoch
                        )
else:
    raise NotImplementedError("Data augmentation not currently supported for "
                              "Mobilenet trained on Imagenet")

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

print("Total time elapsed -- " + str(total_time))

score = model.evaluate_generator(val_gen,
                                 steps=validation_steps_per_epoch,
                                 max_queue_size=5,
                                 verbose=1)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

model_path = os.path.join(
    args.model_dir,
    "trained_model_of_" + model_name + __filename + ".h5")

model.save(model_path)

print("Results (csv) saved at", csv_path)
print("Model saved at", model_path)
print("Total time elapsed -- " + str(total_time))
