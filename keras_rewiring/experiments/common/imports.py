from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras_rewiring.experiments.dnn_argparser import *
# Keras stuff
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from keras_rewiring.utilities.load_dataset import load_and_preprocess_dataset
import numpy as np
from keras_rewiring.sparse_layer import Sparse, SparseConv2D, SparseDepthwiseConv2D
from keras_rewiring.utilities.replace_dense_with_sparse import replace_dense_with_sparse
from keras_rewiring.rewiring_callback import RewiringCallback
# Import OS to deal with directories
import os
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
import pylab as plt
from matplotlib import cm as cm_mlib
import matplotlib as mlib
from matplotlib import animation, rc, colors
# custom optimizer to include noise and temperature
from keras_rewiring.optimizers.noisy_sgd import NoisySGD
# ensure we use viridis as the default cmap
plt.viridis()

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 22})
mlib.rcParams.update({'errorbar.capsize': 5})
# mlib.rcParams.update({'figure.autolayout': True})

# define better cyclical cmap
# https://gist.github.com/MatthewJA/5a0a6d75748bf5cb5962cb9d5572a6ce
cyclic_viridis = colors.LinearSegmentedColormap.from_list(
    'cyclic_viridis',
    [(0, cm_mlib.viridis.colors[0]),
     (0.25, cm_mlib.viridis.colors[256 // 3]),
     (0.5, cm_mlib.viridis.colors[2 * 256 // 3]),
     (0.75, cm_mlib.viridis.colors[-1]),
     (1.0, cm_mlib.viridis.colors[0])])

