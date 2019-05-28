from dnn_argparser import *
from noisy_softplus import NoisySoftplus
# Keras stuff
import keras
from keras.models import load_model
from load_dataset import load_and_preprocess_dataset
import numpy as np

# network generation imports
from mobilenet_model_setup import generate_mobilenet_model
from mnist_model_setup import generate_mnist_model

# Optimizer selection
optimizer = None

sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                           decay=decay_rate, nesterov=False)
adadelta = keras.optimizers.Adadelta()

if args.optimizer.lower() == "sgd":
    optimizer = sgd
elif args.optimizer.lower() in ["ada", "adadelta"]:
    optimizer = adadelta
# Loss selection
loss = None
cross_ent = keras.losses.categorical_crossentropy
mse = keras.losses.mean_squared_error

if args.loss.lower() == "mse":
    loss = mse
elif args.loss.lower() in ["ent", "crossent", "cross_ent"]:
    loss = cross_ent

dataset_info = load_and_preprocess_dataset(args.dataset)
x_train, y_train = dataset_info['train']
x_test, y_test = dataset_info['test']
img_rows, img_cols = dataset_info['img_dims']
input_shape = dataset_info['input_shape']
num_classes = dataset_info['num_classes']

if args.model[0] == ":":
    args.model = args.model[1:]
    if args.model.lower() == "mobilenet":
        model = generate_mobilenet_model(input_shape, num_classes,
                                         activation=args.activation)
    elif args.model.lower() == "mnist":
        model = generate_mnist_model(activation=args.activation)

else:
    # load the model from file
    model_filename = args.model
    if ".h5" not in args.model:
        model_filename = args.model + ".h5"

    model = load_model(model_filename,
                       custom_objects={'noisy_softplus': NoisySoftplus(),
                                       'noisy_softplus_0.17_1': NoisySoftplus(0.17, 1)})
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
output_filename += "_" + args.activation
output_filename += "_" + args.loss
output_filename += "_" + args.optimizer + suffix
output_filename += ".csv"
csv_logger = keras.callbacks.CSVLogger(output_filename, separator=',',
                                       append=False)

model.fit(x_train, y_train,
          batch_size=args.batch,
          epochs=args.epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[csv_logger])

score = model.evaluate(x_test, y_test, verbose=1)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])


model.save("trained_model_of_" + args.model + "_" + args.activation +
            "_" + args.loss +
           "_" + args.optimizer + suffix)
