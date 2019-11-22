# Keras Rewiring 

This repository contains the necessary Keras components to perform 
online rewiring during training. This implementation is based on the paper by 
Guillaume Bellec, 
David Kappel, 
Wolfgang Maass and
Robert Legenstein 
["Deep Rewiring: Training very sparse deep networks"](http://arxiv.org/abs/1711.05136)
at TU Graz, Austria. 

Package components:

- rewiring callback - actually performs the rewiring at the end of each batch
- sparse layers - these are required because they contain a connectivity mask 
(0 - connection missing, 1 - connections exists); this information is used by 
the rewiring callback.
  - Dense equivalent is called Sparse
  - Conv2D equivalent is called SparseConv2D
  - DepthwiseConv2D equivalent is called SparseDepthwiseConv2D
- optimizers - dubbed NoisySGD, it is an extension of Stochastic Gradient 
Descent (SGD) incorporating Gaussian, zero-meaned noise in 
its error computation.
- utilities - some utilities are present to convert networks using "dense" 
 layers to "sparse" ones which the rewiring callback can interact with. In this
 context, "dense" simply covers built-in Keras layers. An 
 ImageNet generator is also provided.
- experiment files - this is a research project, thus we require scripts that
run our experimental setups. 


# Installation

This package can be installed by running ONE of the following commands:

`python setup.py develop`

or 

`python setup.py install`

or 

`pip install .`

TODO: make this package available through PyPi

# How to use

## For predefined networks 

TODO

## In your own networks

TODO


