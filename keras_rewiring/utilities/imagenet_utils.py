import os
import json
import xml.etree.ElementTree as ET
import xml
from keras.applications import imagenet_utils as iu
from keras.utils import Progbar
from keras_preprocessing.image.utils import load_img, img_to_array
import numpy as np
import ntpath

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


class ImagenetDataGenerator(object):
    def __init__(self, mode, batch, root_path,
                 img_size=(224, 224),
                 shuffle=True, steps_per_epoch=None):
        self.mode = mode
        self.batch = batch
        self.root_path = root_path
        self.img_size = img_size
        self.shuffle = shuffle

        self.all_imagenet_classes = None
        self.class2index = None
        self.number_of_samples = 0

        self.steps_per_epoch = steps_per_epoch

        self.no_classes = 0

        self.image_paths, self.cls_dict = \
            self._path_management()

    def __call__(self):
        return self.imagenet_generator()

    def _imagenet_class_lookup(self, cls):
        if self.all_imagenet_classes is None:
            with open(os.path.join(__location__,
                                   "imagenet_class_index.json")) as f:
                self.all_imagenet_classes = json.load(f)
            self.class2index = {value[0]: int(key) for key, value in
                                self.all_imagenet_classes.items()}

        return self.class2index[cls]

    def imagenet_number_of_samples(self):
        return self.number_of_samples

    def _path_management(self):
        '''
        Handle data (img) and class path info
        :param mode:
        :type mode:
        :param root_path:
        :type root_path:
        :return:
        :rtype:
        '''
        _cls_loc = "CLS-LOC"
        img_additional_path = os.path.join(
            self.root_path, "Data", _cls_loc, self.mode)
        cls_additional_path = os.path.join(
            self.root_path, "Annotations", _cls_loc, self.mode)

        # print("Img dirs for", mode.capitalize(), img_dirs)
        # print("Cls dirs for", mode.capitalize(), cls_dirs)

        images = []
        img_dict = {}
        classes = []
        cls_dict = {}

        if self.mode == "train":
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
        elif self.mode == "val":
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
        elif self.mode == "test":
            img_dirs = os.listdir(img_additional_path)
            for _i in img_dirs:
                _ip = os.path.join(
                    img_additional_path, _i)
                img_dict[_i] = _ip
                images.append(_ip)
        else:
            raise ValueError("Invalid mode selected {}".format(self.mode))

        self.number_of_samples = len(images)
        if not self.steps_per_epoch:
            self.steps_per_epoch = self.number_of_samples // self.batch
        images = np.asarray(images)
        classes = np.asarray(classes)
        img_names = np.copy(images)
        for i, name in enumerate(img_names):
            img_names[i] = path_leaf(name).split(".")[0]

        cls_dict = dict.fromkeys(img_names)
        # Check if we have the same number of images as classes
        print("=" * 50, "\n", self.mode.capitalize(), "generator")
        if len(img_dict.keys()) < len(cls_dict.keys()):
            print("=" * 50, "\nYou have fewer image classes than total classes")
            print("If you expected this, disregard this message")
        print("=" * 50, "\n")

        cls_to_label = {}
        if self.mode != "train":
            progbar_no_updates = len(classes)
            progbar = Progbar(progbar_no_updates, interval=2)
            for index, cls in enumerate(classes):
                progbar.update(index)
                pl = path_leaf(cls)[:-4]
                try:
                    xml_tree = ET.parse(cls)
                    root = xml_tree.getroot()
                    for o in root.iter('object'):
                        index = self._imagenet_class_lookup(o[0].text)
                        # set the required label
                        label = np.zeros(1000)
                        label[index] = 1
                        cls_to_label[pl] = label
                        break
                except xml.etree.ElementTree.ParseError as e:
                    print("XML corruption occured. This XML is empty", cls)

        if self.mode == "train":
            # Imagenet is inconsistent. Some JPEGs don't have XML equivalents
            progbar_no_updates = len(images)
            progbar = Progbar(progbar_no_updates, interval=2)
            for index, img in enumerate(images):
                progbar.update(index)
                pl = path_leaf(img)[:-5]
                if pl not in cls_to_label.keys():
                    split_pl = pl.split("_")[0]
                    index = self._imagenet_class_lookup(split_pl)
                    # set the required label
                    label = np.zeros(1000)
                    label[index] = 1
                    cls_to_label[pl] = label

        self.no_classes = len(classes)
        return np.asarray(images), cls_to_label

    def imagenet_generator(self):
        if len(self.image_paths) == 0:
            raise ValueError("Why do you not have image paths?")
        indices = np.arange(len(self.image_paths))
        while True:
            _index = 0

            # if shuffle, shuffle indices
            if self.shuffle:
                np.random.shuffle(indices)
            # another loop keep track of generations for current epoch
            while _index < len(self.image_paths) and \
                    _index < self.steps_per_epoch * self.batch:
                # images and labels (classes) for the current batch
                images_to_yield = []
                labels_to_yield = []

                # indices for current batch
                indices_to_yield = indices[_index: _index + self.batch]

                # assemble batch of images
                _curr_img_paths = self.image_paths[indices_to_yield]
                for _cip in _curr_img_paths:
                    x = img_to_array(load_img(_cip, target_size=self.img_size))
                    # preprocess Imagenet data
                    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py#L75-L84
                    pre_processed_img = iu.preprocess_input(
                        np.asarray(x), mode="tf")
                    images_to_yield.append(pre_processed_img)

                # assemble batch of labels if there are any classes
                if self.no_classes > 0:
                    for _cip in _curr_img_paths:
                        pl = path_leaf(_cip)[:-5]
                        labels_to_yield.append(self.cls_dict[pl])

                _index += self.batch
                yield (np.asarray(images_to_yield), np.asarray(labels_to_yield))


if __name__ == "__main__":
    ilsvrc_path = "F:\ILSVRC"
    batch_size = 10
    print("Train generator")
    train_obj = ImagenetDataGenerator("train", batch_size, ilsvrc_path)
    gen = train_obj()
    print(train_obj.number_of_samples)
    img, cls = gen.__next__()
    print("train_img1", img.shape, cls.shape)
    img, cls = gen.__next__()
    print("train_img2", img.shape, cls.shape)

    print("Val generator")
    val_obj = ImagenetDataGenerator("val", batch_size, ilsvrc_path)
    val_gen = val_obj.imagenet_generator()
    print(val_obj.number_of_samples)
    img, cls = val_gen.__next__()
    print("val_img", img.shape, cls.shape)

    print("Test generator")
    test_obj = ImagenetDataGenerator("test", batch_size, ilsvrc_path)
    test_gen = test_obj.imagenet_generator()
    print(test_obj.number_of_samples)
    img, cls = test_gen.__next__()
    print("tst_img", img.shape, cls.shape)

    img, cls = gen.__next__()
    print("train_img3", img.shape, cls.shape)
