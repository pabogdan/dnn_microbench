from collections import Iterable

import keras
from keras.models import Sequential
from keras.layers import Dense, \
    Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.utils import CustomObjectScope
from keras_rewiring.activations.noisy_softplus import NoisySoftplus
from keras_rewiring.sparse_layer import Sparse, SparseConv2D


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


def generate_sparse_cifar_tf_tutorial_model(activation='relu', batch_size=128,
                                            builtin_sparsity=None,
                                            reg_coeffs=1e-5,
                                            conn_decay=None):
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
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)  # 3 channels in CIFAR
    # input_shape = (1, img_rows * img_cols)
    if not isinstance(reg_coeffs, Iterable):
        reg_coeffs = [reg_coeffs] * 5

    if builtin_sparsity is None:
        builtin_sparsity = [None] * 4

    if conn_decay is None:
        conn_decay = [None] * 4

    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py#L47
    # batch_size = 128

    # deal with string 'nsp'
    if activation in ['nsp', 'noisysoftplus', 'noisy_softplus']:
        activation = NoisySoftplus()

    model = Sequential()
    # Conv1
    model.add(Conv2D(
        filters=64,
        kernel_size=(5, 5),
        input_shape=input_shape,
        batch_size=batch_size,
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.l1(reg_coeffs.pop(0)),
        strides=[1, 1],
        padding="SAME",
        name="conv1"
    )
    )

    # Pool1
    model.add(MaxPooling2D(pool_size=3,
                           strides=2,
                           padding="SAME",
                           name="pool1"
                           )
              )
    # Norm1
    model.add(BatchNormalization(name="norm1"))

    # Conv2
    model.add(SparseConv2D(
        filters=64,
        # consume the first entry in builtin_sparsity
        connectivity_level=builtin_sparsity.pop(0),
        connectivity_decay=conn_decay.pop(0) or None,
        kernel_size=(5, 5),
        bias_initializer=keras.initializers.Constant(0.1),
        kernel_regularizer=keras.regularizers.l1(reg_coeffs.pop(0)),
        name="conv2"
    ))

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
    model.add(Sparse(
        units=384,
        input_shape=input_shape,
        # consume the 2nd entry in builtin_sparsity
        connectivity_level=builtin_sparsity.pop(0),
        connectivity_decay=conn_decay.pop(0) or None,
        # use_bias=False,
        activation=activation,
        kernel_regularizer=keras.regularizers.l1(reg_coeffs.pop(0)),
        bias_initializer=keras.initializers.Constant(0.1),
        name="local3"
    ))

    # Local4 (FC 100)
    model.add(Sparse(
        units=192,
        activation=activation,
        # consume the 3rd entry in builtin_sparsity
        connectivity_level=builtin_sparsity.pop(0),
        connectivity_decay=conn_decay.pop(0) or None,
        kernel_regularizer=keras.regularizers.l1(reg_coeffs.pop(0)),
        bias_initializer=keras.initializers.Constant(0.1),
        name="local4"
    ))

    # Fully-connected (FC) layer
    model.add(Sparse(
        units=10,
        activation='softmax',
        # consume the last entry in builtin_sparsity
        connectivity_level=builtin_sparsity.pop(0),
        connectivity_decay=conn_decay.pop(0) or None,
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.l1(reg_coeffs.pop(0)),
        name="output"
    )
    )
    assert len(builtin_sparsity) == 0
    assert len(reg_coeffs) == 0
    assert len(conn_decay) == 0
    # Return the model
    return model


def save_model(model, filename):
    if ".h5" not in filename:
        filename += ".h5"
    with CustomObjectScope({'NoisySoftplus': NoisySoftplus()}):
        model.save(filename)


if __name__ == "__main__":
    pass
