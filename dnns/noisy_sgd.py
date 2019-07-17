from keras.legacy import interfaces
from keras.optimizers import SGD
from keras import backend as K


class NoisySGD(SGD):

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, temperature=0.,
                 gradient_noise_coeefficient=1e-6, **kwargs):
        super(NoisySGD, self).__init__(lr=lr, momentum=momentum, decay=decay,
                                       nesterov=nesterov, **kwargs)
        self.temperature = temperature
        self.gradient_noise_coeefficient = gradient_noise_coeefficient
        # with K.name_scope(self.__class__.__name__):
        #     self.noise = K.random_normal(shape=stddev=gradient_noise_coeefficient)


    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))
            noise = K.random_normal(shape=K.shape(p),
                                    stddev=self.gradient_noise_coeefficient)
            # temperature_update =
            if self.nesterov:
                new_p = p + self.momentum * v - lr * g + lr * noise
            else:
                new_p = p + v + lr * noise

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

