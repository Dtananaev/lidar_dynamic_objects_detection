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

import argparse
import os
import numpy as np
import tensorflow as tf
from detection_3d.parameters import Parameters
from detection_3d.tools.training_helpers import setup_gpu
from detection_3d.detection_dataset import DetectionDataset
from detection_3d.tools.visualization_tools import visualize_2d_boxes_on_top_image
from detection_3d.tools.file_io import save_bboxes_to_file
from detection_3d.tools.detection_helpers import (
    make_eight_points_boxes,
    get_boxes_from_box_grid,
    get_bboxes_parameters_from_points,
)
from PIL import Image
from tqdm import tqdm
import timeit


def validation_inference(param_settings, dataset_file, model_dir, output_dir):
    setup_gpu()

    # Load model
    model = tf.keras.models.load_model(model_dir)
    bbox_voxel_size = np.asarray(param_settings["bbox_voxel_size"], dtype=np.float32)
    lidar_coord = np.array(param_settings["lidar_offset"], dtype=np.float32)
    grid_meters = param_settings["grid_meters"]

    val_dataset = DetectionDataset(param_settings, dataset_file, shuffle=False)
    param_settings["val_size"] = val_dataset.num_samples
    for val_samples in tqdm(
        val_dataset.dataset, desc=f"val_inference", total=val_dataset.num_it_per_epoch,
    ):
        top_view, gt_boxes, lidar_filenames = val_samples
        predictions = model(top_view, training=False)
        for image, predict, gt, filename in zip(
            top_view, predictions, gt_boxes, lidar_filenames
        ):
            filename = str(filename.numpy())
            seq_folder = filename.split("/")[-3]
            name = os.path.splitext(os.path.basename(filename))[0]
            # Ensure that output dir exists or create it
            top_view_dir = os.path.join(output_dir, "top_view", seq_folder)
            bboxes_dir = os.path.join(output_dir, "bboxes", seq_folder)
            os.makedirs(top_view_dir, exist_ok=True)
            os.makedirs(bboxes_dir, exist_ok=True)
            p_top_view = (
                visualize_2d_boxes_on_top_image(
                    [predict], [image], grid_meters, bbox_voxel_size, prediction=True,
                )
                * 255
            )
            gt_top_view = (
                visualize_2d_boxes_on_top_image(
                    [gt], [image], grid_meters, bbox_voxel_size, prediction=False,
                )
                * 255
            )
            result = np.vstack((p_top_view[0], gt_top_view[0]))
            file_to_save = os.path.join(top_view_dir, name + ".png")
            img = Image.fromarray(result.astype("uint8"))
            img.save(file_to_save)

            box, labels, _ = get_boxes_from_box_grid(predict, bbox_voxel_size)
            box = box.numpy()
            box, _ = make_eight_points_boxes(box)
            if len(box) > 0:
                box = box - lidar_coord[:3]
                labels = np.argmax(labels, axis=-1)
                (
                    centroid,
                    width,
                    length,
                    height,
                    yaw,
                ) = get_bboxes_parameters_from_points(box)
                bboxes_name = os.path.join(bboxes_dir, name + ".txt")
                save_bboxes_to_file(
                    bboxes_name, centroid, width, length, height, yaw, labels
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference  validation set.")
    parser.add_argument(
        "--dataset_file", default="val.datatxt",
    )

    parser.add_argument("--output_dir", default="inference")

    parser.add_argument(
        "--model_dir", default="YoloV3_Lidar-0085",
    )
    args = parser.parse_args()

    param_settings = Parameters().settings
    validation_inference(
        param_settings, args.dataset_file, args.model_dir, args.output_dir
    )
