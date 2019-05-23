# Deep Neural Network (DNN) micro-benchmark

This repository is concerned with the exploration of some neural network
architectures to transform them into spiking models to be run on 
the SpiNNaker neuromorphic platform. 

They are trained and tested on different datasets using different 
architrectures and components. The provided implementation uses keras and 
tensorflow.


## MNIST

Architecture described by [Liu et al (2017)](https://arxiv.org/abs/1706.03609) 
and, in more detail, [Liu (2018)](https://www.research.manchester.ac.uk/portal/en/theses/deep-spiking-neural-networks(336e6a37-2a0b-41ff-9ffb-cca897220d6c).html)

A comparison is performed between the same network architecture using 3 
different activation functions:

    1 - ReLU
    2 - Softplus
    3 - Noisy Softplus [Liu et al (2017)](https://arxiv.org/abs/1706.03609)
    4 - Logarithmic Data Representation [Miyashita et al (2016)](http://arxiv.org/abs/1603.01025)

These models are implemented using keras.

## CIFAR10 and CIFAR100

Architecture described by [Howard et al (2017)](https://arxiv.org/abs/1704.04861)


## Imagenet 2012

Architecture described by [Howard et al (2017)](https://arxiv.org/abs/1704.04861)

