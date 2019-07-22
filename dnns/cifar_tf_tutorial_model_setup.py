import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, \
    Conv2D, AveragePooling2D, Flatten, MaxPooling2D, BatchNormalization
from keras.utils import CustomObjectScope
from noisy_softplus import NoisySoftplus
from sparse_layer import Sparse


def generate_cifar_tf_tutorial_model(activation='relu', batch_size=128):
    '''
    Model is defined in  https://www.tensorflow.org/tutorials/images/deep_cnn
    and
    https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py

    Accuracy:
    cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
    data) as judged by cifar10_eval.py.
    Speed: With batch_size 128.
    System        | Step Time (sec/batch)  |     Accuracy
    ------------------------------------------------------------------
    1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
    1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

    :return: the architecture of the network
    :rtype: keras.models.Sequential
    '''

    # input image dimensions
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)  # 3 channels in CIFAR
    # input_shape = (1, img_rows * img_cols)
    reg_coeff = 1e-5

    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py#L47
    # batch_size = 128

    # deal with string 'nsp'
    if activation in ['nsp', 'noisysoftplus', 'noisy_softplus']:
        activation = NoisySoftplus()

    model = Sequential()
    # Conv1
    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     input_shape=input_shape,
                     batch_size=batch_size,
                     bias_initializer=keras.initializers.Zeros(),
                     strides=[1, 1],
                     padding="SAME",
                     name="conv1"
                     )
              )

    # Pool1
    model.add(MaxPooling2D(padding="SAME",
                           name="pool1"
                           )
              )
    # Norm1
    model.add(BatchNormalization(name="norm1"))

    # Conv2
    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     bias_initializer=keras.initializers.Constant(0.1),
                     name="conv2"
                     )
              )

    # Norm2 (#TODO why is this before Pool2?)
    model.add(BatchNormalization(name="norm2"))

    # Pool2
    model.add(MaxPooling2D(padding="SAME",
                           name="pool2"
                           )
              )

    # Flatten
    model.add(Flatten())
    # Local3 (FC 300)
    model.add(Dense(units=384,
                    input_shape=input_shape,
                    # use_bias=False,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.l1(reg_coeff),
                    bias_initializer=keras.initializers.Constant(0.1),
                    name="local3"
                    ))

    # Local4 (FC 100)
    model.add(Dense(units=192,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.l1(reg_coeff),
                    bias_initializer=keras.initializers.Constant(0.1),
                    name="local4"
                    ))

    # Fully-connected (FC) layer
    model.add(Dense(10, activation='softmax',
                    bias_initializer=keras.initializers.Zeros(),
                    name="output"
                    )
              )

    # Return the model
    return model


def generate_sparse_cifar_tf_tutorial_model(activation='relu'):
    '''
    Model is defined in  https://www.tensorflow.org/tutorials/images/deep_cnn
    and
    https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py

    Accuracy:
    cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
    data) as judged by cifar10_eval.py.
    Speed: With batch_size 128.
    System        | Step Time (sec/batch)  |     Accuracy
    ------------------------------------------------------------------
    1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
    1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

    :return: the architecture of the network
    :rtype: keras.models.Sequential
    '''

    # input image dimensions
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 3)  # 3 channels in CIFAR
    # input_shape = (1, img_rows * img_cols)
    reg_coeff = 1e-5

    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py#L47
    batch_size = 128

    # deal with string 'nsp'
    if activation in ['nsp', 'noisysoftplus', 'noisy_softplus']:
        activation = NoisySoftplus()

    model = Sequential()
    # Conv1
    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     input_shape=input_shape,
                     batch_size=batch_size,
                     bias_initializer=keras.initializers.Zeros,
                     strides=[1, 1, 1, 1],
                     padding="SAME"
                     )
              )

    # Pool1
    model.add(MaxPooling2D(padding="SAME"
                           )
              )
    # Norm1
    # model.add(NOr)

    # Conv2
    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     bias_initializer=keras.initializers.Constant(0.1)
                     )
              )

    # Norm2 (why is this before Pool2?)

    # Pool2
    model.add(MaxPooling2D(padding="SAME"
                           )
              )

    # Flatten
    model.add(Flatten())
    # Local3 (FC 300)
    model.add(Dense(units=384,
                    input_shape=input_shape,
                    # use_bias=False,
                    activation=activation,
                    batch_size=10,
                    kernel_regularizer=keras.regularizers.l1(reg_coeff),
                    ))

    # Local4 (FC 100)
    model.add(Dense(units=192,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.l1(reg_coeff),
                    ))

    # Fully-connected (FC) layer
    model.add(Dense(10, activation='softmax'))

    # Return the model
    return model


def save_model(model, filename):
    if ".h5" not in filename:
        filename += ".h5"
    with CustomObjectScope({'NoisySoftplus': NoisySoftplus()}):
        model.save(filename)


if __name__ == "__main__":
    pass
