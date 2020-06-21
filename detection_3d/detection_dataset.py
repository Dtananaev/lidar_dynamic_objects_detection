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
import numpy as np
import argparse
from tqdm import tqdm
from detection_3d.parameters import Parameters
from detection_3d.tools.file_io import load_dataset_list, load_lidar, load_bboxes
from detection_3d.tools.detection_helpers import (
    make_top_view_image,
    make_eight_points_boxes,
    get_bboxes_grid,
)
from detection_3d.tools.visualization_tools import visualize_2d_boxes_on_top_image
from PIL import Image


class DetectionDataset:
    """
    This is dataset layer for 3d detection experiment
    Arguments:
        param_settings: parameters of experiment
        dataset_file: name of .dataset file
        shuffle: shuffle the data True/False
    """

    def __init__(self, param_settings, dataset_file, shuffle=False):
        # Private methods
        self.seed = param_settings["seed"]

        self.param_settings = param_settings
        self.dataset_file = dataset_file
        self.inputs_list = load_dataset_list(
            self.param_settings["dataset_dir"], dataset_file
        )
        self.num_samples = len(self.inputs_list)
        self.num_it_per_epoch = int(
            self.num_samples / self.param_settings["batch_size"]
        )
        self.output_types = [tf.float32, tf.float32]

        ds = tf.data.Dataset.from_tensor_slices(self.inputs_list)

        if shuffle:
            ds = ds.shuffle(self.num_samples)
        ds = ds.map(
            map_func=lambda x: tf.py_function(
                self.load_data, [x], Tout=self.output_types
            ),
            num_parallel_calls=12,
        )
        ds = ds.batch(self.param_settings["batch_size"])
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.dataset = ds

    def load_data(self, data_input):
        """
        Loads image and semseg and resizes it
        Note: This is numpy function.
        """
        lidar_file, bboxes_file = np.asarray(data_input).astype("U")

        lidar = load_lidar(lidar_file)
        bboxes = load_bboxes(bboxes_file)
        labels = bboxes[:, -1]
        lidar_corners_3d, _ = make_eight_points_boxes(bboxes[:, :-1])
        # # Shift lidar coordinate to positive quadrant
        lidar_coord = np.asarray(self.param_settings["lidar_offset"], dtype=np.float32)
        lidar = lidar + lidar_coord
        lidar_corners_3d = lidar_corners_3d + lidar_coord[:3]
        # Process data
        top_view = make_top_view_image(
            lidar, self.param_settings["grid_meters"], self.param_settings["voxel_size"]
        )
        box_grid = get_bboxes_grid(
            labels,
            lidar_corners_3d,
            self.param_settings["grid_meters"],
            self.param_settings["bbox_voxel_size"],
        )
        return top_view, box_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DatasetLayer.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        help="creates .dataset file",
        default="train.datatxt",
    )
    args = parser.parse_args()

    param_settings = Parameters().settings
    train_dataset = DetectionDataset(param_settings, args.dataset_file)

    bbox_voxel_size = np.asarray(param_settings["bbox_voxel_size"], dtype=np.float32)
    grid_meters = np.array(param_settings["grid_meters"], dtype=np.float32)

    lidar_coord = np.array(param_settings["lidar_offset"], dtype=np.float32)

    for samples in tqdm(train_dataset.dataset, total=train_dataset.num_it_per_epoch):
        top_images, boxes_grid = samples
        print(f"lidar {top_images.shape}, boxes {boxes_grid.shape}")

        top_view = (
            visualize_2d_boxes_on_top_image(
                boxes_grid, top_images, grid_meters, bbox_voxel_size
            )
            * 255
        )
        img = Image.fromarray(top_view[0].astype("uint8"))
        img.save("result.png")
        input()
