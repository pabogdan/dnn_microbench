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
import json
import h5py


# https://github.com/keras-team/keras/issues/10417#issuecomment-435620108
def fix_layer0(filename, batch_input_shape, dtype):
    with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def replace_dense_with_sparse(
        model, activation='relu',
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
    sparse_model = Sequential()

    model_input_shape = None

    for i, layer in enumerate(model.layers):
        # get layer configuration
        layer_config = layer.get_config()
        # modify layer name
        layer_config['name'] = "sparse_" + layer_config['name']
        # replace with the appropriate sparse layer
        curr_sparse_layer = None
        if i == 0 and 'batch_input_shape' in layer_config:
            model_input_shape = layer_config['batch_input_shape']

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
    converted_model_filename = model.name + "_converted_to_sparse"
    print("Converted filename", converted_model_filename)
    file_path = os.path.join(_cache, converted_model_filename + ".h5")
    with CustomObjectScope(custom_object):
        sparse_model.save(file_path)
    fix_layer0(file_path, model_input_shape or [None, 224, 224, 3], 'float32')

    K.clear_session()

    # Return the model
    return load_model(file_path, custom_objects=custom_object)


if __name__ == "__main__":
    model = keras.applications.MobileNet()
    sparse_model = replace_dense_with_sparse(model)
    sparse_model.summary()

    model_path = os.path.join("05-mobilenet_dwarf_v1",
                              "standard_dense_dwarf.h5")
    model = load_model(model_path)
    sparse_model = replace_dense_with_sparse(model)
    sparse_model.summary()
