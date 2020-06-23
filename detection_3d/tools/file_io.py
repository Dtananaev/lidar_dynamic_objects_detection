#!/usr/bin/env python
__copyright__ = """
Copyright (c) 2020 Tananaev Denis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from detection_3d.data_preprocessing.pandaset_tools.helpers import labels


def load_and_resize_image(image_filename, resize=None, data_type=tf.float32):
    """
    Load png image to tensor and resize if necessary
    Arguments:
       image_filename: image file to load
       resize: tensor [new_width, new_height] or None
    Return:
       img: tensor of the size [1, H, W, 3]
    """

    img = tf.io.read_file(image_filename)
    img = tf.image.decode_png(img)
    # Add batch dim
    img = tf.expand_dims(img, axis=0)

    if resize is not None:
        img = tf.compat.v1.image.resize_nearest_neighbor(img, resize)

    img = tf.cast(img, data_type)
    return img


def save_plot_to_image(file_to_save, figure):
    """
    Save matplotlib figure to image and close
    """
    plt.savefig(file_to_save)
    plt.close(figure)


def read_json(json_filename):
    with open(json_filename) as json_file:
        data = json.load(json_file)
        return data


def save_bboxes_to_file(filename, centroid, width, length, height, alpha, label):

    if centroid is not None:
        with open(filename, "w") as the_file:
            for c, w, l, h, a, lbl in zip(
                centroid, width, length, height, alpha, label
            ):
                data = "{};{};{};{};{};{};{};{}\n".format(
                    c[0], c[1], c[2], l, w, h, a, lbl
                )
                the_file.write(data)


def load_bboxes(label_filename):
    # returns the array with [num_boxes, (bbox_parm)]
    with open(label_filename) as f:
        bboxes = np.asarray([line.rstrip().split(";") for line in f])
        # Convert labels to numbers
        bboxes[:, -1] = [labels[label] for label in bboxes[:, -1]]
        bboxes = np.asarray(bboxes, dtype=float)
        return bboxes


def load_lidar(lidar_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(lidar_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


def save_lidar(lidar_filename, scan):
    scan = scan.reshape((-1))
    scan.tofile(lidar_filename)


def save_to_json(json_filename, dict_to_save):
    """
    Save to json file
    """
    with open(json_filename, "w") as f:
        json.dump(dict_to_save, f, indent=2)


def save_dataset_list(dataset_file, data_list):
    """
    Saves dataset list to file.
    """
    with open(dataset_file, "w") as f:
        for item in data_list:
            f.write("%s\n" % item)


def load_dataset_list(dataset_dir, dataset_file, delimiter=";"):
    """
    The function loads list of data from dataset
    file.
    Args:
     dataset_file: path to the .dataset file.
    Returns:
     dataset_list: list of data.
    """

    file_path = os.path.join(dataset_dir, dataset_file)
    dataset_list = []
    with open(file_path) as f:
        dataset_list = f.readlines()
    dataset_list = [x.strip().split(delimiter) for x in dataset_list]
    return dataset_list
