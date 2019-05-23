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

# Optimizer selection
sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                           decay=decay_rate, nesterov=False)
adadelta = keras.optimizers.Adadelta()
cross_ent = keras.losses.categorical_crossentropy
mse = keras.losses.mean_squared_error


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
    (x_train, y_train), (x_test, y_test) = cifar10.load_data(label_mode='fine')

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)


    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

# load the 3 differents experimental cases
model = load_model(args.model, custom_objects={'noisy_softplus_0.17_1': NoisySoftplus()})
model.summary()
model.compile(
    optimizer=sgd,
    loss=mse,
    metrics=['accuracy'])
# check initial performance for the 3 cases
# Evaluating pretrained model

output_filename = ""
if args.result_filename:
    output_filename = args.result_filename
else:
    output_filename = "results_for_" + args.model + ".csv"

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

model.save("trained_model_of_" + args.model)
