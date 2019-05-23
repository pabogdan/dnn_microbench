import keras

class NoisySoftplus():
    ''' The Noisy Softplus activation function
        Values of k and sigma taken from Liu et al. 2017
    
    '''
    
    def __init__(self, k=0.17, sigma=1):
        self.k = k
        self.sigma = sigma
        self.__name__ = 'noisy_softplus_{}_{}'.format(self.k,
                                                    self.sigma)
                
    def __call__ (self, *args, **kwargs):
        return self.k*self.sigma*keras.backend.softplus(args[0]/(self.k*self.sigma))
