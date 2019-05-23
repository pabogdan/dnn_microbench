from dnn_argparser import *
from noisy_softplus import NoisySoftplus
# Keras stuff
import keras
from keras import backend as K
from keras.layers import Activation
from keras import activations
from keras.models import load_model
from keras.datasets import mnist, cifar10, cifar100
import keras.utils as utils
import numpy as np
from keras.utils import np_utils

# network generation imports
from mobilenet_setup import generate_mobilenet_model

# Optimizer selection
optimizer = None


sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                           decay=decay_rate, nesterov=False)
adadelta = keras.optimizers.Adadelta()

if args.optimizer.lower() == "sgd":
    optimizer = sgd
elif args.optimizer.lower() in ["ada", "adadelta"]:
    optimizer = adadelta
# Loss selection
loss = None
cross_ent = keras.losses.categorical_crossentropy
mse = keras.losses.mean_squared_error


if args.loss.lower() == "mse":
    loss = mse
elif args.loss.lower() in ["ent", "crossent", "cross_ent"]:
    loss = cross_ent

# Dataset selection
if args.dataset.lower() == "mnist":
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
elif args.dataset.lower() == "cifar10":
    # input image dimensions
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)
    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

elif args.dataset.lower() == "cifar100":
    # input image dimensions
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)
    num_classes = 100

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

if args.model[0] == ":":
    if args.model[1:].lower() == "mobilenet":
        model = generate_mobilenet_model(input_shape, num_classes,
                                         activation=args.activation)
        args.model = args.model[1:]
else:
    # load the model from file
    model_filename = args.model
    if ".h5" not in args.model:
        model_filename = args.model + ".h5"

    model = load_model(model_filename, custom_objects={'noisy_softplus_0.17_1': NoisySoftplus()})
model.summary()
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'])
# check initial performance for the 3 cases
# Evaluating pretrained model

output_filename = ""
if args.result_filename:
    output_filename = args.result_filename
else:
    output_filename = "results_for_" + args.model
output_filename += "_" + args.activation
output_filename += "_" + args.optimizer
output_filename += ".csv"
csv_logger = keras.callbacks.CSVLogger(output_filename, separator=',',
                                       append=False)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[csv_logger])

score = model.evaluate(x_test, y_test, verbose=1)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

model.save("trained_model_of_" + args.model + "_" + args.activation +
           "_" + args.optimizer)
