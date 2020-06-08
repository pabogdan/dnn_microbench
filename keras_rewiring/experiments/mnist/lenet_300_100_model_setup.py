import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import CustomObjectScope
from keras_rewiring.activations.noisy_softplus import NoisySoftplus
from keras_rewiring.sparse_layer import Sparse
import tensorflow as tf
import numpy as np


def generate_lenet_300_100_model(activation='relu', categorical_output=True,
                                 num_classes=10):
    '''
    Model is defined in LeCun et al. 1998
    Gradient-Based Learning Applied to Document Recognition
    :return: the architecture of the network
    :rtype: keras.models.Sequential
    '''

    # input image dimensions
    img_rows, img_cols = 28, 28
    # input_shape = (img_rows, img_cols, 1)
    input_shape = (1, img_rows * img_cols)
    reg_coeff = 1e-5

    # deal with string 'nsp'
    if activation in ['nsp', 'noisysoftplus', 'noisy_softplus']:
        activation = NoisySoftplus()

    model = Sequential()
    # First (input) layer (FC 300)
    model.add(Dense(units=300,
                    input_shape=input_shape,
                    # use_bias=False,
                    activation=activation,
                    batch_size=10,
                    kernel_regularizer=keras.regularizers.l1(reg_coeff),
                    ))

    # Second layer (FC 100)
    model.add(Dense(units=100,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.l1(reg_coeff),
                    ))

    # Fully-connected (FC) layer
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',
                    kernel_constraint=keras.constraints.NonNeg()))

    # Return the model
    return model


def generate_sparse_lenet_300_100_model(activation='relu',
                                        categorical_output=True,
                                        builtin_sparsity=None,
                                        conn_decay=None,
                                        num_classes=10):
    '''
    Model is defined in LeCun et al. 1998
    Gradient-Based Learning Applied to Document Recognition
    :return: the architecture of the network
    :rtype: keras.models.Sequential
    '''

    # input image dimensions
    img_rows, img_cols = 28, 28
    # input_shape = (img_rows, img_cols, 1)
    input_shape = (1, img_rows * img_cols)
    reg_coeff = 1e-5
    if builtin_sparsity is None:
        builtin_sparsity = [None] * 3

    if conn_decay is None:
        conn_decay = [None] * 3

    # deal with string 'nsp'
    if activation in ['nsp', 'noisysoftplus', 'noisy_softplus']:
        activation = NoisySoftplus()

    model = Sequential()
    # First (input) layer (FC 300)
    model.add(Sparse(units=300,
                     # consume the first entry in builtin_sparsity
                     connectivity_level=builtin_sparsity.pop(0) or None,
                     connectivity_decay=conn_decay.pop(0) or None,
                     input_shape=input_shape,
                     # use_bias=False,
                     activation=activation,
                     batch_size=10,
                     kernel_regularizer=keras.regularizers.l1(reg_coeff))
              )
    # Second layer (FC 100)
    model.add(Sparse(units=100,
                     # consume the 2nd entry in builtin_sparsity
                     connectivity_level=builtin_sparsity.pop(0) or None,
                     connectivity_decay=conn_decay.pop(0) or None,
                     activation=activation,
                     kernel_regularizer=keras.regularizers.l1(reg_coeff)))
    # Fully-connected (FC) layer
    model.add(Flatten())
    model.add(Sparse(units=num_classes,
                     # consume the last entry in builtin_sparsity
                     connectivity_level=builtin_sparsity.pop(0) or None,
                     connectivity_decay=conn_decay.pop(0) or None,
                     kernel_constraint=keras.constraints.NonNeg(),
                     activation='softmax'))

    assert len(builtin_sparsity) == 0
    assert len(conn_decay) == 0
    # Return the model
    return model


def convert_model_to_tf(model):
    layers = model.layers

    # Clear session of any other models
    keras.backend.clear_session()

    is_model_sparse = False
    if isinstance(layers[0], Sparse):
        is_model_sparse = True

    # define weight, biases and masks outside the functions
    l300_weights = tf.constant(layers[0].get_weights()[0])
    l100_weights = tf.constant(layers[1].get_weights()[0])
    l10_weights = tf.constant(layers[-1].get_weights()[0])

    l300_biases = tf.constant(layers[0].get_weights()[1])
    l100_biases = tf.constant(layers[1].get_weights()[1])
    l10_biases = tf.constant(layers[-1].get_weights()[1])

    if is_model_sparse:
        # l300_indices = tf.constant(np.where(layers[0].get_weights()[-1] > 0.5), dtype=tf.int64)
        # l100_indices = tf.constant(np.where(layers[1].get_weights()[-1] > 0.5), dtype=tf.int64)
        # l10_indices = tf.constant(np.where(layers[-1].get_weights()[-1] > 0.5), dtype=tf.int64)
        l300_indices = tf.where(tf.not_equal(layers[0].get_weights()[-1], 0.0))
        l100_indices = tf.where(tf.not_equal(layers[1].get_weights()[-1], 0.0))
        l10_indices = tf.where(tf.not_equal(layers[-1].get_weights()[-1], 0.0))

        l300_values = tf.gather_nd(tf.constant(layers[0].get_weights()[0]), l300_indices)
        l100_values = tf.gather_nd(tf.constant(layers[1].get_weights()[0]), l100_indices)
        l10_values = tf.gather_nd(tf.constant(layers[-1].get_weights()[0]), l10_indices)

        l300_dims = tf.constant(layers[0].get_weights()[-1].shape, dtype=tf.int64)
        l100_dims = tf.constant(layers[1].get_weights()[-1].shape, dtype=tf.int64)
        l10_dims = tf.constant(layers[-1].get_weights()[-1].shape, dtype=tf.int64)

        # l300_weights = tf.sparse.SparseTensor(indices=l300_indices,
        #                                       values=l300_values,
        #                                       dense_shape=l300_dims)
        # l100_weights = tf.sparse.SparseTensor(indices=l100_indices,
        #                                       values=l100_values,
        #                                       dense_shape=l100_dims)
        # l10_weights = tf.sparse.SparseTensor(indices=l10_indices,
        #                                      values=l10_values,
        #                                      dense_shape=l10_dims)
        # One of two ways to use sparsness: kernel is a sparsetensor
        # or kernel has a lot of zeros
        l300_weights = tf.constant(layers[0].get_weights()[0]*layers[0].get_weights()[-1])
        l100_weights = tf.constant(layers[1].get_weights()[0]*layers[1].get_weights()[-1])
        l10_weights = tf.constant(layers[-1].get_weights()[0]*layers[-1].get_weights()[-1])

    tf.config.experimental_run_functions_eagerly(True)

    # Define the computational graph
    @tf.function
    def dense_forward_pass(x):
        # layer 1
        a_1 = tf.matmul(x, l300_weights) + l300_biases
        activation_1 = tf.nn.relu(a_1)
        # layer 2
        a_2 = tf.matmul(activation_1, l100_weights) + l100_biases
        activation_2 = tf.nn.relu(a_2)
        # output layer
        logits_3 = tf.matmul(activation_2, l10_weights) + l10_biases
        return tf.nn.softmax(logits_3)

    @tf.function
    def sparse_forward_pass(x):
        # layer 1
        # masked_x = tf.sparse.mask(x, np.where(layers[0].get_weights()[-1].astype(bool)))
        # masked_x = tf.boolean_mask(x, layers[0].get_weights()[-1].astype(bool))
        a_1 = tf.matmul(x, l300_weights, b_is_sparse=True) + l300_biases
        activation_1 = tf.nn.relu(a_1)
        # layer 2
        a_2 = tf.matmul(activation_1, l100_weights, b_is_sparse=True) + l100_biases
        activation_2 = tf.nn.relu(a_2)
        # output layer
        logits_3 = tf.matmul(activation_2, l10_weights, b_is_sparse=True) + l10_biases
        return tf.nn.softmax(logits_3)

    # return sparse_forward_pass if is_model_sparse else dense_forward_pass
    return dense_forward_pass


def save_model(model, filename):
    if ".h5" not in filename:
        filename += ".h5"
    with CustomObjectScope({'NoisySoftplus': NoisySoftplus()}):
        model.save(filename)


if __name__ == "__main__":
    print("Testing the script operation")
    model_relu = generate_lenet_300_100_model()
    print(model_relu.get_config())
    save_model(model_relu, "relu_mnist_lenet_300_100")
    model = generate_lenet_300_100_model(activation='softplus')
    print(model.get_config())
    save_model(model, "softplus_mnist_lenet_300_100")
    model = generate_lenet_300_100_model(activation=NoisySoftplus())
    print(model.get_config())
    save_model(model, "noisysoftplus_mnist_lenet_300_100")

    # Test accuracy of empty model
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    batch = 10

    # the data, split between train and test sets
    from keras.datasets import mnist

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
    print(x_test.shape, 'xtest shape')
    print(y_test.shape, 'ytest shape')
    # x_test = x_test.reshape(batch, x_test.shape[0]//batch, x_test.shape[1], x_test.shape[2], x_test.shape[3])

    model_relu.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adadelta(),
                       metrics=['accuracy'])
    print(model_relu.summary())
    # model_relu.fit(x_train, y_train,
    #           batch_size=batch,
    #           epochs=1,
    #           verbose=0)
    score = model_relu.evaluate(x_test, y_test, verbose=0, batch_size=batch)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
