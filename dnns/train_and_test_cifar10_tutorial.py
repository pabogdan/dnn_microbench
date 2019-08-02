from dnn_argparser import *
from noisy_softplus import NoisySoftplus
# Keras stuff
import keras
from keras.models import load_model
from load_dataset import load_and_preprocess_dataset
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from rewiring_callback import RewiringCallback
# Import OS to deal with directories
import os
# network generation imports
from cifar_tf_tutorial_model_setup import generate_cifar_tf_tutorial_model, \
    generate_sparse_cifar_tf_tutorial_model
import tensorflow as tf
from keras import backend as K
import pylab as plt

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

dataset_info = load_and_preprocess_dataset(
    'cifar10')
x_train, y_train = dataset_info['train']
x_test, y_test = dataset_info['test']
img_rows, img_cols = dataset_info['img_dims']
input_shape = dataset_info['input_shape']
num_classes = dataset_info['num_classes']

print(x_train.shape)
epochs = args.epochs or 10
batch = args.batch or 128
learning_rate = 0.5
decay_rate = 0.8  # changed from 0.8

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

p_0 = .05  # global connectivity level
builtin_sparsity = [8 * p_0, .8 * p_0, 8 * p_0, 1]
alphas = [0, 10 ** -7, 10 ** -6, 10 ** -9, 0]

if not args.sparse_layers:
    model = generate_cifar_tf_tutorial_model(
        activation=args.activation, batch_size=batch)
elif args.sparse_layers and not args.soft_rewiring:
    model = generate_sparse_cifar_tf_tutorial_model(
        activation=args.activation, batch_size=batch,
        builtin_sparsity=builtin_sparsity,
        reg_coeffs=alphas)
else:
    print("Soft rewiring enabled", args.soft_rewiring)
    model = generate_sparse_cifar_tf_tutorial_model(
        activation=args.activation, batch_size=batch,
        reg_coeffs=alphas)
model.summary()

# disable rewiring with sparse layers to see the performance of the layer
# when 90% of connections are disabled and static
deep_r = RewiringCallback(fixed_conn=args.disable_rewiring,
                          soft_limit=args.soft_rewiring)
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
activation_name = "relu"
loss_name = "crossent"
output_filename += "_" + activation_name
output_filename += "_" + loss_name
output_filename += "_" + sparse_name
output_filename += "_" + optimizer_name + suffix
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

if not args.data_augmentation:
    model.fit(x_train[:x_train.shape[0] - (x_train.shape[0] % batch)],
              y_train[:x_train.shape[0] - (x_train.shape[0] % batch)],
              batch_size=batch,
              epochs=epochs,
              verbose=1,
              callbacks=callback_list,
              validation_data=(x_test, y_test),
              shuffle=True
              # validation_split=.2
              )
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0
    )

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    steps_per_epoch = x_train.shape[0] // batch
    print("Steps per epoch", steps_per_epoch)
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train[:steps_per_epoch * batch],
                                     y_train[:steps_per_epoch * batch],
                                     batch_size=batch),
                        epochs=epochs,
                        callbacks=callback_list,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(x_test, y_test))

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

print("Total time elapsed -- " + str(total_time))

score = model.evaluate(x_test, y_test, verbose=1, batch_size=batch)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

model_path = os.path.join(
    args.model_dir,
    "trained_model_of_" + model_name + "_" + activation_name +
    "_" + loss_name + "_" + sparse_name +
    "_" + optimizer_name + suffix)

model.save(model_path)

print("Results (csv) saved at", csv_path)
print("Model saved at", model_path)
