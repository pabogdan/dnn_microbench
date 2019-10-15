from dnns.utilities import *
import numpy as np
import matplotlib as mlib
import matplotlib.pyplot as plt
from dnns.load_dataset import load_and_preprocess_dataset

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 22})
mlib.rcParams.update({'errorbar.capsize': 5})


def compute_mean_img(generator, batch_size, num_examples, num_classes):
    # set up steps_per_epoch
    steps_per_epoch = num_examples // batch_size
    # validation_steps_per_epoch = no_val // batch_size

    print("Training Steps per epoch", steps_per_epoch)
    print("Number of classes:", num_classes)
    print("Number of training examples:", num_examples)

    mean_image = None
    classes = None
    print("=" * 50)
    print("Computing the mean ImageNet image...")
    progbar = Progbar(steps_per_epoch, interval=1)
    for i in range(steps_per_epoch):
        img, cls = generator.__next__()
        if i == 0:
            mean_image = np.mean(img, axis=0)
            classes = np.sum(cls, axis=0)
        else:
            curr_mean = np.mean(img, axis=0)
            mean_image = (mean_image + curr_mean) / 2.
            classes += np.sum(cls, axis=0)
        progbar.update(i)
    print("=" * 50)
    return {
        'mean_image': mean_image,
        'classes': classes
    }


if __name__ == "__main__":
    # check if the figures folder exist
    fig_folder = 'figures/'
    if not os.path.isdir(fig_folder) and not os.path.exists(fig_folder):
        os.mkdir(fig_folder)
    dataset_path = "F:/ILSVRC"
    batch_size = 100
    dataset_info = load_and_preprocess_dataset(
        'imagenet', batch_size=batch_size, path=dataset_path)
    train_gen = dataset_info['train']
    val_gen = dataset_info['val']
    # TODO Implement test generator to compare values from train and val
    # test_gen = dataset_info['test']
    input_shape = dataset_info['input_shape']
    num_classes = dataset_info['num_classes']
    no_train = dataset_info['no_train']
    no_val = dataset_info['no_val']

    res_dict = compute_mean_img(train_gen,
                     num_classes=num_classes,
                     num_examples=no_train,
                     batch_size=batch_size)
    mean_image = res_dict['mean_image']
    classes = res_dict['classes']

    fig = plt.figure(figsize=(8, 8), dpi=600)
    plt.imshow(mean_image + 1)
    plt.savefig(fig_folder + "mean_image.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 16), dpi=600)
    plt.bar(np.arange(classes.size), classes)
    plt.xlabel("Class")
    plt.ylabel("Occurances in data")
    plt.savefig(fig_folder + "mean_image.png")
    plt.close(fig)

    res_dict_val = compute_mean_img(val_gen,
                     num_classes=num_classes,
                     num_examples=no_val,
                     batch_size=batch_size)
    mean_image_val = res_dict['mean_image']
    classes_val = res_dict['classes']


    fig = plt.figure(figsize=(8, 8), dpi=600)
    plt.imshow(mean_image_val + 1)
    plt.savefig(fig_folder + "mean_image_val.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 16), dpi=600)
    plt.bar(np.arange(classes_val.size), classes_val)
    plt.xlabel("Class")
    plt.ylabel("Occurances in data")
    plt.savefig(fig_folder + "mean_image_val.png")
    plt.close(fig)

    np.savez("mean_imagenet_image",
             mean_image=mean_image,
             classes=classes,
             mean_image_val=mean_image_val,
             classes_val=classes_val)
