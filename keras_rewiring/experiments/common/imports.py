from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras_rewiring.experiments.dnn_argparser import *
# Keras stuff
import keras
from keras.models import load_model
from keras_rewiring.utilities.load_dataset import load_and_preprocess_dataset
import numpy as np
from keras_rewiring.sparse_layer import Sparse, SparseConv2D, SparseDepthwiseConv2D
from keras_rewiring.utilities.replace_dense_with_sparse import replace_dense_with_sparse
from keras_rewiring.rewiring_callback import RewiringCallback
# Import OS to deal with directories
import os
import sys
import tensorflow as tf
from keras import backend as K
import pylab as plt
# custom optimizer to include noise and temperature
from keras_rewiring.optimizers.noisy_sgd import NoisySGD

