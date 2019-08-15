import os
import json
import xml.etree.ElementTree as ET
import xml
from keras.applications import imagenet_utils as iu
from keras_preprocessing.image.utils import load_img, img_to_array
import numpy as np
import ntpath

all_imagenet_classes = None
class2index = None
number_of_samples = {}

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def _imagenet_class_lookup(cls):
    global all_imagenet_classes, class2index

    if all_imagenet_classes is None:
        with open("imagenet_class_index.json") as f:
            all_imagenet_classes = json.load(f)
        class2index = {value[0]: int(key) for key, value in
                       all_imagenet_classes.items()}

    return class2index[cls]

def imagenet_number_of_samples(mode=None):
    global number_of_samples
    if mode is None:
        return number_of_samples
    return number_of_samples[mode]



def _path_management(mode, root_path):
    '''
    Handle data (img) and class path info
    :param mode:
    :type mode:
    :param root_path:
    :type root_path:
    :return:
    :rtype:
    '''
    global number_of_samples
    _cls_loc = "CLS-LOC"
    img_additional_path = os.path.join(
        root_path, "Data", _cls_loc, mode)
    cls_additional_path = os.path.join(
        root_path, "Annotations", _cls_loc, mode)

    # print("Img dirs for", mode.capitalize(), img_dirs)
    # print("Cls dirs for", mode.capitalize(), cls_dirs)

    images = []
    img_dict = {}
    classes = []
    cls_dict = {}
    cls_to_label = {}

    if mode == "train":
        img_dirs = os.listdir(img_additional_path)
        cls_dirs = os.listdir(cls_additional_path)
        for dir in img_dirs:
            p = os.path.join(
                img_additional_path, dir)
            _imgs = os.listdir(p)
            mini_img = []
            for i in _imgs:
                mini_img.append(os.path.join(p, i))
            images += mini_img
            img_dict[dir] = mini_img
        for dir in cls_dirs:
            c = os.path.join(
                cls_additional_path, dir)
            _cls = os.listdir(c)
            mini_cls = []
            for i in _cls:
                mini_cls.append(os.path.join(c, i))
            classes += mini_cls
            cls_dict[dir] = mini_cls
        number_of_samples["train"] = len(images)
    elif mode == "val":
        img_dirs = os.listdir(img_additional_path)
        cls_dirs = os.listdir(cls_additional_path)
        for _i in img_dirs:
            _ip = os.path.join(
                img_additional_path, _i)
            images.append(_ip)
            img_dict[_i] = _ip
        for _c in cls_dirs:
            _cp = os.path.join(
                cls_additional_path, _c)
            classes.append(_cp)
            cls_dict[_c] = _cp
        number_of_samples["val"] = len(images)
    elif mode == "test":
        img_dirs = os.listdir(img_additional_path)
        for _i in img_dirs:
            _ip = os.path.join(
                img_additional_path, _i)
            img_dict[_i] = _ip
            images.append(_ip)
        number_of_samples["test"] = len(images)
    else:
        raise ValueError("Invalid mode selected {}".format(mode))

    # Check if we have the same number of images as classes
    print("=" * 50, "\n", mode.capitalize(), "generator")
    if len(img_dict.keys()) < len(cls_dict.keys()):
        print("=" * 50, "\nYou have fewer image classes than total classes")
        print("If you expected this, disregard this message")
    print("=" * 50, "\n")


    cls_to_label = {}
    for cls in classes:
        pl = path_leaf(cls)[:-4]
        try:
            xml_tree = ET.parse(cls)
            root = xml_tree.getroot()
            for o in root.iter('object'):
                index = _imagenet_class_lookup(o[0].text)
                # set the required label
                label = np.zeros(1000)
                label[index] = 1
                cls_to_label[pl] = label
                break
        except xml.etree.ElementTree.ParseError as e:
            print("XML corruption occured. This XML is empty", cls)


    if mode == "train":
        # Imagenet is inconsistent. Some JPEGs don't have XML equivalents
        for img in images:
            pl = path_leaf(img)[:-5]
            if pl not in cls_to_label.keys():
                split_pl = pl.split("_")[0]
                index = _imagenet_class_lookup(split_pl)
                # set the required label
                label = np.zeros(1000)
                label[index] = 1
                cls_to_label[pl] = label

    return np.asarray(images), np.asarray(classes), cls_to_label


def imagenet_generator(mode, batch, root_path,
                       img_size=(224, 224),
                       shuffle=True):
    image_paths, class_paths, cls_dict = \
        _path_management(mode, root_path)
    if len(image_paths) == 0:
        raise ValueError("Why do you not have image paths?")
    indices = np.arange(len(image_paths))
    while True:
        _index = 0

        # if shuffle, shuffle indices
        if shuffle:
            np.random.shuffle(indices)
        # another loop keep track of generations for current epoch
        while _index < len(image_paths):
            # images and labels (classes) for the current batch
            images_to_yield = []
            labels_to_yield = []

            # indices for current batch
            indices_to_yield = indices[_index: _index + batch]

            # assemble batch of images
            _curr_img_paths = image_paths[indices_to_yield]
            for _cip in _curr_img_paths:
                x = img_to_array(load_img(_cip, target_size=img_size))
                # preprocess Imagenet data
                pre_processed_img = iu.preprocess_input(
                    np.asarray(x), mode="tf")
                images_to_yield.append(pre_processed_img)

            # assemble batch of labels if there are any classes
            if len(class_paths) > 0:
                for _cip in _curr_img_paths:
                    pl = path_leaf(_cip)[:-5]
                    labels_to_yield.append(cls_dict[pl])

            _index += batch
            yield (np.asarray(images_to_yield), np.asarray(labels_to_yield))


if __name__ == "__main__":
    ilsvrc_path = "F:\ILSVRC"
    batch_size = 10
    print("Train generator")
    gen = imagenet_generator("train", batch_size, ilsvrc_path)
    img, cls = gen.__next__()
    print("train_img1", img.shape, cls.shape)
    img, cls = gen.__next__()
    print("train_img2", img.shape, cls.shape)

    print("Val generator")
    val_gen = imagenet_generator("val", batch_size, ilsvrc_path)
    img, cls = val_gen.__next__()
    print("val_img", img.shape, cls.shape)

    print("Test generator")
    test_gen = imagenet_generator("test", batch_size, ilsvrc_path)
    img, cls = test_gen.__next__()
    print("tst_img", img.shape, cls.shape)

    img, cls = gen.__next__()
    print("train_img3", img.shape, cls.shape)

