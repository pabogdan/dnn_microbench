import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, \
    Conv2D, AveragePooling2D, Flatten
from keras.utils import CustomObjectScope
from noisy_softplus import NoisySoftplus
from vis.utils.utils import apply_modifications


def generate_mobilenet_model(input_shape, num_classes, activation='relu',
                             custom_object=NoisySoftplus()):
    '''
    Model is defined in Howard et al (2017)
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
    Applications
    :return: the architecture of the network
    :rtype: keras.applications.mobilenet.MobileNet
    '''
    model = keras.applications.mobilenet.MobileNet(
        weights=None,
        input_shape=input_shape,
        classes=num_classes,
        include_top=True)

    if activation != 'relu':
        # Loop through layers and replace the ReLU activation with your
        # flavour of choice
        act_to_use = None
        if activation == 'softplus':
            act_to_use = keras.activations.softplus
        elif activation in ['noisysoftplus', 'noisy_softplus']:
            act_to_use = NoisySoftplus()
        elif isinstance(activation, NoisySoftplus):
            act_to_use = activation

        for layer in model.layers:
            if hasattr(layer, 'activation'):
                layer.activation = act_to_use
        with CustomObjectScope({'noisy_softplus': custom_object}):
            model = apply_modifications(model)

    # Return the model
    return model


def save_model(model, filename, custom_object=NoisySoftplus()):
    if ".h5" not in filename:
        filename += ".h5"
    with CustomObjectScope({'NoisySoftplus': custom_object}):
        model.save(filename)


if __name__ == "__main__":
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)
    num_classes = 10
    print("Testing the script operation")
    model = generate_mobilenet_model(input_shape, num_classes)
    print(model.get_config())
    save_model(model, "relu_cifar10_mobilenet")
    model = generate_mobilenet_model(input_shape, num_classes,
                                     activation='softplus')
    print(model.get_config())
    save_model(model, "softplus_cifar10_mobilenet")
    model = generate_mobilenet_model(input_shape, num_classes,
                                     activation=NoisySoftplus())
    print(model.get_config())
    save_model(model, "noisysoftplus_cifar10_mobilenet")


    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)
    num_classes = 100
    print("Testing the script operation")
    model = generate_mobilenet_model(input_shape, num_classes)
    print(model.get_config())
    save_model(model, "relu_cifar100_mobilenet")
    model = generate_mobilenet_model(input_shape, num_classes,
                                     activation='softplus')
    print(model.get_config())
    save_model(model, "softplus_cifar100_mobilenet")
    model = generate_mobilenet_model(input_shape, num_classes,
                                     activation=NoisySoftplus())
    print(model.get_config())
    save_model(model, "noisysoftplus_cifar100_mobilenet")
