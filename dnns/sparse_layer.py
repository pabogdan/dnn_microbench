from keras import backend as K, activations, initializers, regularizers, constraints
from keras.engine import InputSpec
from keras.engine.topology import Layer
import numpy as np
import keras


class Sparse(Layer):

    def __init__(self, units, connectivity_level, activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Sparse, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.connectivity_level = connectivity_level

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      trainable=True,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # self.signs = self.add_weight(name='sign',
        #                              shape=(input_dim, self.units),
        #                              initializer=self.kernel_initializer,
        #                              trainable=False,
        #                              regularizer=self.kernel_regularizer,
        #                              constraint=self.kernel_constraint)

        # only some of the values in kernel can be updated, based on this mask
        # self.mask = self.add_weight(name='mask',
        #                             shape=(input_dim, self.units),
        #                             initializer=keras.initializers.Zeros(),
        #                             trainable=False)



        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Set the correct initial values here
        # use K.set_value(x, value)
        total_number_of_matrix_entries = input_dim * self.units
        number_of_active_synapses = int(self.connectivity_level *
                                        total_number_of_matrix_entries)
        # https://stackoverflow.com/questions/47941079/can-i-make-random-mask-with-numpy?rq=1
        _pre_mask = np.zeros(total_number_of_matrix_entries, int)
        _pre_mask[:number_of_active_synapses] = 1

        np.random.shuffle(_pre_mask)
        _pre_mask = _pre_mask.astype(bool).reshape((input_dim, self.units))
        # set this as the mask
        # K.set_value(self.mask, _pre_mask)
        self.mask = K.variable(_pre_mask, name="mask")

        # apply mask
        self.kernel = self.kernel * self.mask

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        # Be sure to call this at the end
        super(Sparse, self).build(input_shape)



    def add_update(self, updates, inputs=None):
        super(Sparse, self).add_update(updates, inputs)

    def call(self, inputs, **kwargs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units

        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(Sparse, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
