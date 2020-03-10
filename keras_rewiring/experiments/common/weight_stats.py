from .imports import *

def weights_from_model(model):
    layers = model.layers
    for l in layers:
        l_n = l.name
        kernel = K.get_value(l.kernel)
        bias = K.get_value(l.bias)
        if hasattr(l,"mask"):
            mask = K.get_value(l.mask)
        else:
            mask = None
