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
from detection_3d.tools.visualization_tools import visualize_2d_boxes_on_top_image


def train_summaries(train_out, optimizer, param_settings, learning_rate):
    """
    Visualizes  the train outputs in tensorboards
    """

    writer = tf.summary.create_file_writer(param_settings["train_summaries"])
    with writer.as_default():
        # Losses
        (
            obj_loss,
            label_loss,
            z_loss,
            delta_xy_loss,
            width_loss,
            height_loss,
            delta_orient_loss,
        ) = train_out["losses"]

        # Show learning rate given scheduler
        if param_settings["scheduler"]["name"] != "no_scheduler":
            with tf.name_scope("Optimizer info"):
                step = float(
                    optimizer.iterations.numpy()
                )  # triangular_scheduler learning rate needs float dtype
                tf.summary.scalar(
                    "learning_rate", learning_rate(step), step=optimizer.iterations
                )
        with tf.name_scope("Training losses"):
            tf.summary.scalar(
                "1.Total loss", train_out["total_loss"], step=optimizer.iterations
            )
            tf.summary.scalar("2.obj loss", obj_loss, step=optimizer.iterations)
            tf.summary.scalar("3.label_loss", label_loss, step=optimizer.iterations)
            tf.summary.scalar("4. z_loss", z_loss, step=optimizer.iterations)
            tf.summary.scalar(
                "5. delta_xy_loss", delta_xy_loss, step=optimizer.iterations
            )
            tf.summary.scalar("6. width_loss", width_loss, step=optimizer.iterations)
            tf.summary.scalar("8. height_loss", height_loss, step=optimizer.iterations)
            tf.summary.scalar(
                "9. delta_orient_loss", delta_orient_loss, step=optimizer.iterations
            )

        if (
            param_settings["step_summaries"] is not None
            and optimizer.iterations % param_settings["step_summaries"] == 0
        ):
            bbox_voxel_size = np.asarray(
                param_settings["bbox_voxel_size"], dtype=np.float32
            )
            lidar_coord = np.array(param_settings["lidar_offset"], dtype=np.float32)
            gt_bboxes = train_out["box_grid"]
            p_bboxes = train_out["predictions"]
            grid_meters = param_settings["grid_meters"]
            top_view = train_out["top_view"]
            gt_top_view = visualize_2d_boxes_on_top_image(
                gt_bboxes, top_view, grid_meters, bbox_voxel_size,
            )

            p_top_view = visualize_2d_boxes_on_top_image(
                p_bboxes, top_view, grid_meters, bbox_voxel_size, prediction=True,
            )

            # Show GT
            with tf.name_scope("1-Ground truth bounding boxes"):
                tf.summary.image("Top view", gt_top_view, step=optimizer.iterations)

            with tf.name_scope("2-Predicted bounding boxes"):
                tf.summary.image(
                    "Predicted top view", p_top_view, step=optimizer.iterations
                )


def epoch_metrics_summaries(param_settings, epoch_metrics, epoch):
    """
    Visualizes epoch metrics
    """
    # Train results
    writer = tf.summary.create_file_writer(param_settings["train_summaries"])
    with writer.as_default():
        # Show epoch metrics for train
        with tf.name_scope("Epoch metrics"):
            tf.summary.scalar(
                "1. Loss", epoch_metrics.train_loss.result().numpy(), step=epoch
            )

    # Val results
    writer = tf.summary.create_file_writer(param_settings["eval_summaries"])
    with writer.as_default():
        # Show epoch metrics for train
        with tf.name_scope("Epoch metrics"):
            tf.summary.scalar(
                "1. Loss", epoch_metrics.val_loss.result().numpy(), step=epoch
            )
