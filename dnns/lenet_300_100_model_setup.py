import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, \
    Conv2D, AveragePooling2D, Flatten
from keras.utils import CustomObjectScope
from noisy_softplus import NoisySoftplus


def generate_lenet_300_100_model(activation='relu', categorical_output=True):
    '''
    Model is defined in Liu et al 2016
    Noisy Softplus : A Biology Inspired Activation Function
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
                    kernel_regularizer=keras.regularizers.l1(reg_coeff)))

    # Second layer (FC 100)
    model.add(Dense(units=100,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.l1(reg_coeff)))

    # Fully-connected (FC) layer
    model.add(Flatten())
    if categorical_output:
        model.add(Dense(10, activation='softmax'))
    else:
        model.add(Dense(1))

    # Return the model
    return model


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
