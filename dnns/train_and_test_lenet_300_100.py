from dnn_argparser import *
from noisy_softplus import NoisySoftplus
# Keras stuff
import keras
from keras.models import load_model
from load_dataset import load_and_preprocess_dataset
import numpy as np
from rewiring_callback import RewiringCallback
# Import OS to deal with directories
import os
# network generation imports
from mobilenet_model_setup import generate_mobilenet_model
from mnist_model_setup import generate_mnist_model
from lenet_300_100_model_setup import generate_lenet_300_100_model

# Checking directory structure exists
if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if not os.path.isdir(args.model_dir) and not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)

is_output_categorical = True
dataset_info = load_and_preprocess_dataset(
    'mnist', categorical_output=is_output_categorical)
x_train, y_train = dataset_info['train']
x_test, y_test = dataset_info['test']
img_rows, img_cols = dataset_info['img_dims']
input_shape = dataset_info['input_shape']
num_classes = dataset_info['num_classes']

# reshape input to flatten data
x_train = x_train.reshape(x_train.shape[0], 1, np.prod(x_train.shape[1:]))
x_test = x_test.reshape(x_test.shape[0], 1, np.prod(x_test.shape[1:]))

print(x_train.shape)
epochs = 10
batch = 10
learning_rate = 0.5
decay_rate = 0.8

# TODO implement custom optimizer to include noise and temperature
optimizer = keras.optimizers.SGD(lr=learning_rate,
                                 decay=decay_rate, nesterov=False)

loss = keras.losses.categorical_crossentropy

model = generate_lenet_300_100_model(activation=args.activation,
                                     categorical_output=is_output_categorical)
model.summary()
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'])

# check initial performance for the 3 cases
# Evaluating pretrained model
suffix = ""
if args.suffix:
    suffix = "_" + args.suffix

output_filename = ""
if args.result_filename:
    output_filename = args.result_filename
else:
    output_filename = "results_for_" + args.model
activation_name = "relu"
loss_name = "crossent"
optimizer_name = "sgd"
output_filename += "_" + activation_name
output_filename += "_" + loss_name
output_filename += "_" + optimizer_name + suffix
output_filename += ".csv"

connectivity_proportion = [.01, .03, .3]

csv_logger = keras.callbacks.CSVLogger(
    os.path.join(args.result_dir, output_filename),
    separator=',',
    append=False)
tb = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=0,  # turning this on needs validation_data in model.fit
    batch_size=batch, write_graph=True,
    write_grads=True, write_images=True,
    embeddings_freq=0, embeddings_layer_names=None,
    embeddings_metadata=None, embeddings_data=None,
    update_freq='epoch')

deep_r = RewiringCallback(connectivity_proportion=connectivity_proportion)

model.fit(x_train, y_train,
          batch_size=batch,
          epochs=epochs,
          verbose=1,
          callbacks=[csv_logger, tb, deep_r],
          validation_split=0)

score = model.evaluate(x_test, y_test, verbose=1, batch_size=batch)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

model.save(os.path.join(
    args.model_dir,
    "trained_model_of_" + args.model + "_" + activation_name +
    "_" + loss_name +
    "_" + optimizer_name + suffix))
