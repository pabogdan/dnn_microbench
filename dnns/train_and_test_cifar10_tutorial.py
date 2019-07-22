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
from cifar_tf_tutorial_model_setup import generate_cifar_tf_tutorial_model, \
    generate_sparse_cifar_tf_tutorial_model
import tensorflow as tf
from keras import backend as K
import pylab as plt

start_time = plt.datetime.datetime.now()
# Get number of cores reserved by the batch system
# (NSLOTS is automatically set, or use 4 otherwise)
NUMCORES=int(os.getenv("NSLOTS", 4))
print("Using", NUMCORES, "core(s)")

# Create TF session using correct number of cores
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
   intra_op_parallelism_threads=NUMCORES, allow_soft_placement=True,
   device_count = {'CPU': NUMCORES}))

# Set the Keras TF session
K.set_session(sess)

# Checking directory structure exists
if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if not os.path.isdir(args.model_dir) and not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)

dataset_info = load_and_preprocess_dataset(
    'cifar10')
x_train, y_train = dataset_info['train']
x_test, y_test = dataset_info['test']
img_rows, img_cols = dataset_info['img_dims']
input_shape = dataset_info['input_shape']
num_classes = dataset_info['num_classes']

print(x_train.shape)
epochs = args.epochs or 10
batch = args.batch or 128
learning_rate = 0.5
decay_rate = 0.8  # changed from 0.8

connectivity_proportion = [.01, .03, .3]

if args.optimizer.lower() == "sgd":
    if not args.sparse_layers:
        optimizer = keras.optimizers.SGD()
    else:
        optimizer = keras.optimizers.SGD(lr=learning_rate)
    optimizer_name = "sgd"

elif args.optimizer.lower() in ["ada", "adadelta"]:
    optimizer = keras.optimizers.adadelta()
    optimizer_name = "adadelta"
elif args.optimizer.lower() in ["noisy_sgd", "ns"]:
    # custom optimizer to include noise and temperature
    from noisy_sgd import NoisySGD

    if not args.sparse_layers:
        optimizer = NoisySGD()
    else:
        optimizer = NoisySGD(lr=learning_rate)
    optimizer_name = "noisy_sgd"
else:
    optimizer = args.optimizer
    optimizer_name = args.optimizer

loss = keras.losses.categorical_crossentropy

if not args.sparse_layers:
    model = generate_cifar_tf_tutorial_model(
        activation=args.activation, batch_size=batch)
else:
    model = generate_sparse_cifar_tf_tutorial_model(
        activation=args.activation, batch_size=batch)
model.summary()

# disable rewiring with sparse layers to see the performance of the layer
# when 90% of connections are disabled and static
deep_r = RewiringCallback(connectivity_proportion=connectivity_proportion,
                          fixed_conn=args.disable_rewiring)
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

suffix = ""
if args.suffix:
    suffix = "_" + args.suffix

if args.model[0] == ":":
    model_name = args.model[1:]
else:
    model_name = args.model

output_filename = ""
if args.result_filename:
    output_filename = args.result_filename
else:
    output_filename = "results_for_" + model_name
activation_name = "relu"
loss_name = "crossent"
output_filename += "_" + activation_name
output_filename += "_" + loss_name
output_filename += "_" + optimizer_name + suffix
output_filename += ".csv"

csv_path = os.path.join(args.result_dir, output_filename)
csv_logger = keras.callbacks.CSVLogger(
    csv_path,
    separator=',',
    append=False)

tb_log_filename = "./sparse_logs" if args.sparse_layers else "./dense_logs"

tb = keras.callbacks.TensorBoard(
    log_dir=tb_log_filename,
    histogram_freq=0,  # turning this on needs validation_data in model.fit
    batch_size=batch, write_graph=True,
    write_grads=True, write_images=True,
    embeddings_freq=0, embeddings_layer_names=None,
    embeddings_metadata=None, embeddings_data=None,
    update_freq='epoch')

callback_list = []
if args.sparse_layers:
    callback_list.append(deep_r)

callback_list += [csv_logger, tb]
model.fit(x_train[:x_train.shape[0]-(x_train.shape[0] % batch)],
          y_train[:x_train.shape[0]-(x_train.shape[0] % batch)],
          batch_size=batch,
          epochs=epochs,
          verbose=1,
          callbacks=callback_list,
          validation_data=(x_test, y_test),
          shuffle=True
          # validation_split=.2
          )

score = model.evaluate(x_test, y_test, verbose=1, batch_size=batch)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

model_path = os.path.join(
    args.model_dir,
    "trained_model_of_" + model_name + "_" + activation_name +
    "_" + loss_name +
    "_" + optimizer_name + suffix)

model.save(model_path)

print("Results (csv) saved at", csv_path)
print("Model saved at", model_path)

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time
print("Total time elapsed -- " + str(total_time))