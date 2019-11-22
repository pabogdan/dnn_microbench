import keras


class NoisySoftplus():
    ''' The Noisy Softplus activation function
        Values of k and sigma taken from Liu et al. 2017

    Noisy Softplus fits well to the practical response
    firing rate of the LIF neuron
    with suitable calibration of k and S, see Figure 7.
    The parameter pair of (k, S) is
    curve-fitted with the triple data points of (λ, x, σ).
    The fitted parameter was set to (k, S) = (0.19, 208.76)
    for the practical response
    firing rate driven by synaptic noisy current
    with τsyn = 1 ms and was set to
    (k, S) = (0.35, 201.06) when τsyn = 10 ms.
    The calibration currently is
    conducted by linear least squares regression;
    numerical analysis is
    considered however for future work to express
    the factors with biological
    parameters of a LIF neuron.

    '''

    def __init__(self, k=0.20, sigma=.5):
        self.k = k
        self.sigma = sigma
        self.__name__ = 'noisy_softplus'

    def __call__(self, *args, **kwargs):
        return self.k * self.sigma * keras.backend.softplus(args[0] / (self.k * self.sigma))
