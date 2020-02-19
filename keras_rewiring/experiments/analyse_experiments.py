from keras_rewiring.experiments.analysis_argparser import *
import string
import matplotlib.pyplot as plt
from matplotlib import cm as cm_mlib
from matplotlib import animation, rc, colors
import matplotlib as mlib
from scipy import stats
from pprint import pprint as pp
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import traceback
import os
import copy
from datetime import datetime
import statsmodels.graphics.api as smg
import pandas as pd
import warnings
import ntpath
from colorama import Fore, Style, init as color_init

mlib.use('Agg')
warnings.filterwarnings("ignore", category=UserWarning)

# ensure we use viridis as the default cmap
plt.viridis()
viridis_cmap = mlib.cm.get_cmap('viridis')

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})

def color_for_index(index, size, cmap=viridis_cmap):
    return cmap(1 / (size - index + 2))

def analyse_experiment(in_file, results_dir):
    print("=" * 80)
    print("Processing ", in_file)
    print("-" * 80)
    # Check if the folders exist
    if not os.path.isdir(results_dir) and not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Based on in_file extension decide on what analysis needs to be done
    split_fname = in_file.split(".")
    ext = split_fname[-1]
    if ext in ["csv"]:
        csv_analysis(in_file, results_dir)
    elif ext in ["hdf5, hd5"]:
        raise NotImplementedError("hdf5 files not support at the moment")
    print("=" * 80)


def plot_single(data, filename, xlabel, ylabel, labels, results_dir,
                extensions=("pdf", "svg")):
    fig = plt.figure(figsize=(12, 6), dpi=300)
    size_of_data = len(data)
    for d, l, i in zip(data, labels, range(size_of_data)):
        plt.plot(d, label=l, color=color_for_index(index=i,
                                                   size=size_of_data))
    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for ext in extensions:
        plt.savefig(os.path.join(results_dir, filename + ".{}".format(ext)))
    plt.close(fig)


def csv_analysis(in_file, results_dir):
    csv = pd.read_csv(in_file)
    split_fname = ntpath.basename(in_file).split(".")
    results_dir = os.path.join(results_dir, split_fname[0])
    # Check if the folders exist
    if not os.path.isdir(results_dir) and not os.path.exists(results_dir):
        os.mkdir(results_dir)

    csv_column_names = csv.columns
    pp(csv[['epoch', 'acc', 'loss', 'val_acc', 'val_loss']].describe())
    # sum together all csv columns related to rewiring
    # select column names corresponding to some rewiring
    rewiring_columns_names = []
    for c in csv_column_names:
        if "no_rewires" in c:
            rewiring_columns_names.append(c)
    summed_rewires = csv[rewiring_columns_names].sum(axis=1)
    # plotting area
    # plot acc
    plot_single([csv.acc], filename="training_acc",
                xlabel="Epoch", ylabel="Accuracy",
                labels=["Training accuracy"],
                results_dir=results_dir)
    # plot loss
    plot_single([csv.loss], filename="loss",
                xlabel="Epoch", ylabel="Loss",
                labels=["Training loss"],
                results_dir=results_dir)
    # plot val_acc
    plot_single([csv.val_acc], filename="val_acc",
                xlabel="Epoch", ylabel="Accuracy",
                labels=["Validation accuracy"],
                results_dir=results_dir)
    # plot val_loss
    plot_single([csv.val_loss], filename="val_loss",
                xlabel="Epoch", ylabel="Loss",
                labels=["Validation loss"],
                results_dir=results_dir)
    # plot number_of_rewires
    plot_single([summed_rewires], filename="summed_rewires",
                xlabel="Epoch", ylabel="# of rewires",
                labels=["# of rewires"],
                results_dir=results_dir)


if __name__ == "__main__":
    if (analysis_args.input and len(analysis_args.input) > 0 and
            not analysis_args.compare):
        # Generate results for individual files
        for in_file in analysis_args.input:
            analyse_experiment(in_file, analysis_args.results_dir)
    elif (analysis_args.input and len(analysis_args.input) > 0 and
            analysis_args.compare):
        # Generate comparison results for files
        pass
    mnist_static_results = "mnist/results/"
    mnet_static_results = "mobilenet/results/"
    cifar_static_results = "cifar10/results/"
    roots_bloody_roots = [mnist_static_results, cifar_static_results, mnet_static_results]
    for root_dir in roots_bloody_roots:
        for file in os.listdir(root_dir):
            try:
                analyse_experiment(os.path.join(root_dir, file),
                                   analysis_args.results_dir)
            except Exception as e:
                traceback.print_exc()
