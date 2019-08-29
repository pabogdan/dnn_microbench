import json

from keras.callbacks import ModelCheckpoint

from dnn_argparser import *
from noisy_softplus import NoisySoftplus
# Keras stuff
import keras
from keras.models import load_model
from load_dataset import load_and_preprocess_dataset
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

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
learning_rate = 0.5
decay_rate = 0.8  # changed from 0.8

p_0 = .05  # global connectivity level
builtin_sparsity = [8 * p_0, .8 * p_0, 8 * p_0, 1]
alphas = [0, 10 ** -7, 10 ** -6, 10 ** -9, 0]
final_conns = np.asarray(builtin_sparsity)
conn_decay_values = None
if args.conn_decay:
    conn_decay_values = (np.log(1. / final_conns) / epochs).tolist()
    builtin_sparsity = np.ones(len(conn_decay_values)).tolist()

# Dense model
if args.random_weights:
    model = keras.applications.MobileNet(weights=None)
else:
    model = keras.applications.MobileNet()
if args.sparse_layers and not args.soft_rewiring:
    if args.conn_decay:
        print("Connectivity decay rewiring enabled", conn_decay_values)
        model = replace_dense_with_sparse(
            model,
            activation=args.activation, batch_size=batch,
            builtin_sparsity=builtin_sparsity,
            reg_coeffs=alphas,
            conn_decay=conn_decay_values)
    else:
        model = replace_dense_with_sparse(
            model,
            activation=args.activation, batch_size=batch,
            builtin_sparsity=builtin_sparsity,
            reg_coeffs=alphas)
elif args.sparse_layers and args.soft_rewiring:
    print("Soft rewiring enabled", args.soft_rewiring)
    model = replace_dense_with_sparse(
        model,
        activation=args.activation, batch_size=batch,
        reg_coeffs=alphas)
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
    if not args.sparse_layers:
        optimizer = keras.optimizers.SGD()
    else:
        optimizer = keras.optimizers.SGD(lr=learning_rate)
    optimizer_name = "sgd"

elif args.optimizer.lower() in ["ada", "adadelta"]:
    optimizer = keras.optimizers.adadelta()
    optimizer_name = "adadelta"
elif args.optimizer.lower() in ["noisy_sgd", "ns"]:
    # custom optimizer to include noise and temperature
    from noisy_sgd import NoisySGD

    if not args.sparse_layers:
        optimizer = NoisySGD()
    else:
        optimizer = NoisySGD(lr=learning_rate)
    optimizer_name = "noisy_sgd"
else:
    optimizer = args.optimizer
    optimizer_name = args.optimizer

loss = keras.losses.categorical_crossentropy

# disable rewiring with sparse layers to see the performance of the layer
# when 90% of connections are disabled and static
deep_r = RewiringCallback(fixed_conn=args.disable_rewiring,
                          soft_limit=args.soft_rewiring,
                          asserts_on=args.asserts_on)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

suffix = ""
if args.suffix:
    suffix = "_" + args.suffix

if args.model[0] == ":":
    model_name = args.model[1:]
else:
    model_name = args.model

output_filename = ""
if args.result_filename:
    output_filename = args.result_filename
else:
    output_filename = "results_for_" + model_name
if args.sparse_layers:
    if args.soft_rewiring:
        sparse_name = "sparse_soft"
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
checkpoint_callback = ModelCheckpoint(checkpoint_filename)

__filename = generate_filename(
    optimizer_name, activation_name, sparse_name, loss_name, suffix,
    args.random_weights)
output_filename += __filename
output_filename += ".csv"

csv_path = os.path.join(args.result_dir, output_filename)
csv_logger = keras.callbacks.CSVLogger(
    csv_path,
    separator=',',
    append=False)

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
                        workers=1
                        )
else:
    raise NotImplementedError("Data augmentation not currently supported for "
                              "Mobilenet trained on Imagenet")

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

print("Total time elapsed -- " + str(total_time))

score = model.evaluate_generator(val_gen,
                                 steps=validation_steps_per_epoch,
                                 max_queue_size=10,
                                 verbose=1)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

model_path = os.path.join(
    args.model_dir,
    "trained_model_of_" + model_name +  __filename + ".h5")

model.save(model_path)

print("Results (csv) saved at", csv_path)
print("Model saved at", model_path)
print("Total time elapsed -- " + str(total_time))
