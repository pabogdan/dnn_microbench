import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, \
    Conv2D, AveragePooling2D, Flatten
from keras.utils import CustomObjectScope
from noisy_softplus import NoisySoftplus


def generate_mnist_model(activation='relu'):
    '''
    Model is defined in Liu et al 2016
    Noisy Softplus : A Biology Inspired Activation Function
    :return: the architecture of the network
    :rtype: keras.models.Sequential
    '''

    # input image dimensions
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    # deal with string 'nsp'
    if activation in ['nsp', 'noisysoftplus', 'noisy_softplus']:
        activation = NoisySoftplus()

    model = Sequential()
    # First input convolution (6c5)
    model.add(Conv2D(filters=6,
                     kernel_size=(5, 5),
                     input_shape=input_shape,
                     use_bias=False,
                     activation=activation,
                     batch_size=50))

    # First pooling layer (average, 6p, 2s)
    model.add(AveragePooling2D(pool_size=2))

    # Second convolution (12c5)
    model.add(Conv2D(filters=12,
                     kernel_size=(5, 5),
                     use_bias=False,
                     activation=activation))

    # Second pooling layer (average, 12p, 2s)
    model.add(AveragePooling2D(pool_size=2))

    # Fully-connected (FC) layer
    model.add(Flatten())
    model.add(Dense(10))

    # Return the model
    return model


def save_model(model, filename):
    if ".h5" not in filename:
        filename += ".h5"
    with CustomObjectScope({'NoisySoftplus': NoisySoftplus()}):
        model.save(filename)


if __name__ == "__main__":
    print("Testing the script operation")
    model_relu = generate_mnist_model()
    print(model_relu.get_config())
    save_model(model_relu, "relu_mnist_liu2016")
    model = generate_mnist_model(activation='softplus')
    print(model.get_config())
    save_model(model, "softplus_mnist_liu2016")
    model = generate_mnist_model(activation=NoisySoftplus())
    print(model.get_config())
    save_model(model, "noisysoftplus_mnist_liu2016")

    # Test accuracy of empty model
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

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

    model_relu.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    score = model_relu.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])