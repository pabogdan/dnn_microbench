import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, \
    Conv2D, AveragePooling2D, Flatten, DepthwiseConv2D
from keras.utils import CustomObjectScope

from noisy_sgd import NoisySGD
from noisy_softplus import NoisySoftplus
from sparse_layer import Sparse, SparseConv2D, SparseDepthwiseConv2D
from keras.models import load_model
from keras import backend as K
import ntpath


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def replace_dense_with_sparse(
        model_filename, activation='relu',
        batch_size=1,
        builtin_sparsity=None,
        reg_coeffs=None,
        conn_decay=None,
        custom_object={}):
    '''
    Model is defined in Howard et al (2017)
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
    Applications
    :return: the architecture of the network
    :rtype: keras.applications.mobilenet.MobileNet
    '''
    custom_object = custom_object or {}
    model = load_model(model_filename,
                       custom_objects=custom_object)
    sparse_model = Sequential()
    act_to_use = None
    if activation != 'relu':
        # Loop through layers and replace the ReLU activation with your
        # flavour of choice
        if activation == 'softplus':
            act_to_use = keras.activations.softplus
        elif activation in ['noisysoftplus', 'noisy_softplus']:
            act_to_use = NoisySoftplus()
        elif isinstance(activation, NoisySoftplus):
            act_to_use = activation

    for i, layer in enumerate(model.layers):
        # get layer configuration
        layer_config = layer.get_config()
        # modify layer name
        layer_config['name'] = "sparse_" + layer_config['name']
        # replace with the appropriate sparse layer
        curr_sparse_layer = None
        if isinstance(layer, DepthwiseConv2D):
            curr_sparse_layer = SparseDepthwiseConv2D(**layer_config)
        elif isinstance(layer, Conv2D):
            curr_sparse_layer = SparseConv2D(**layer_config)
        elif isinstance(layer, Dense):
            curr_sparse_layer = Sparse(**layer_config)

        if curr_sparse_layer is not None:
            sparse_model.add(curr_sparse_layer)
        else:
            sparse_model.add(layer)

        # if hasattr(layer, 'activation') and act_to_use is not None:
        #     layer.activation = act_to_use

    _cache = "__pycache__"
    if not os.path.isdir(_cache):
        os.mkdir(_cache)

    custom_object.update({'Sparse': Sparse,
                          'SparseConv2D': SparseConv2D,
                          'SparseDepthwiseConv2D': SparseDepthwiseConv2D,
                          'NoisySGD': NoisySGD})

    # https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    converted_model_filename = path_leaf(model_filename) + "_converted_to_sparse"
    file_path = os.path.join(_cache, converted_model_filename)
    with CustomObjectScope(custom_object):
        sparse_model.save(file_path)

    K.clear_session()

    _model = load_model(file_path, custom_objects=custom_object)
    # Return the model
    return _model
