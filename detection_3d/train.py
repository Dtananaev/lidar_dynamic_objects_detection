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
import tensorflow as tf
from detection_3d.parameters import Parameters
from detection_3d.detection_dataset import DetectionDataset
from detection_3d.tools.detection_helpers import get_voxels_grid
from detection_3d.model import YoloV3_Lidar
from detection_3d.tools.training_helpers import (
    setup_gpu,
    initialize_model,
    load_model,
    get_optimizer,
)
from detection_3d.losses import detection_loss
from detection_3d.tools.summary_helpers import train_summaries
from tqdm import tqdm


@tf.function
def train_step(param_settings, train_samples, model, optimizer, epoch_metrics=None):

    with tf.GradientTape() as tape:
        top_view, box_grid = train_samples
        predictions = model(top_view, training=True)
        (
            obj_loss,
            label_loss,
            z_loss,
            delta_xy_loss,
            width_loss,
            height_loss,
            delta_orient_loss,
        ) = detection_loss(box_grid, predictions)
        losses = [
            obj_loss,
            label_loss,
            z_loss,
            delta_xy_loss,
            width_loss,
            height_loss,
            delta_orient_loss,
        ]
        total_detection_loss = tf.reduce_sum(losses)
        # Get L2 losses for weight decay
        total_loss = total_detection_loss + tf.add_n(model.losses)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch_metrics is not None:
        epoch_metrics.train_loss(total_detection_loss)

    train_outputs = {
        "total_loss": total_loss,
        "losses": losses,
        "box_grid": box_grid,
        "predictions": predictions,
        "top_view": top_view,
    }

    return train_outputs


# @tf.function
# def val_step(samples, model, epoch_metrics=None):

#     _, top_images, bboxes, _, _ = samples
#     predictions = model(top_images, training=False)
#     (
#         obj_loss,
#         label_loss,
#         z_loss,
#         delta_xy_loss,
#         width_loss,
#         height_loss,
#         delta_orient_loss,
#     ) = pc_net_loss(bboxes, predictions)
#     losses = [
#         obj_loss,
#         label_loss,
#         z_loss,
#         delta_xy_loss,
#         width_loss,
#         height_loss,
#         delta_orient_loss,
#     ]
#     detection_loss = tf.reduce_sum(losses)

#     if epoch_metrics is not None:
#         epoch_metrics.val_loss(detection_loss)


def train(resume=False):
    setup_gpu()
    # General parameters
    param = Parameters()

    # Init label colors and label names
    tf.random.set_seed(param.settings["seed"])

    train_dataset = DetectionDataset(param.settings, "train.datatxt", shuffle=True,)

    param.settings["train_size"] = train_dataset.num_samples
    val_dataset = DetectionDataset(param.settings, "val.datatxt", shuffle=False)
    param.settings["val_size"] = val_dataset.num_samples

    model = YoloV3_Lidar(weight_decay=param.settings["weight_decay"])
    voxels_grid = get_voxels_grid(
        param.settings["voxel_size"], param.settings["grid_meters"]
    )
    input_shape = [1, voxels_grid[0], voxels_grid[1], 2]
    initialize_model(model, input_shape)
    model.summary()
    start_epoch, model = load_model(param.settings["checkpoints_dir"], model, resume)
    model_path = os.path.join(param.settings["checkpoints_dir"], "{model}-{epoch:04d}")

    optimizer = get_optimizer(
        param.settings["optimizer"], param.settings["learning_rate"]
    )
    # epoch_metrics = EpochMetrics()

    for epoch in range(start_epoch, param.settings["max_epochs"]):
        save_dir = model_path.format(model=model.name, epoch=epoch)
        # epoch_metrics.reset()
        for train_samples in tqdm(
            train_dataset.dataset,
            desc=f"Epoch {epoch}",
            total=train_dataset.num_it_per_epoch,
        ):
            train_outputs = train_step(
                param.settings, train_samples, model, optimizer, epoch_metrics=None
            )
            train_summaries(train_outputs, optimizer, param.settings)
        # for val_samples in tqdm(
        #     val_dataset.dataset, desc="Validation", total=val_dataset.num_it_per_epoch
        # ):
        #     val_step(val_samples, model, epoch_metrics)
        # epoch_metrics_summaries(param.settings, epoch_metrics, epoch)
        # epoch_metrics.print_metrics()
        # # Save all
        # param.save_to_json(save_dir)
        # epoch_metrics.save_to_json(save_dir)
        model.save(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN.")
    parser.add_argument(
        "--resume",
        type=lambda x: x,
        nargs="?",
        const=True,
        default=False,
        help="Activate nice mode.",
    )
    args = parser.parse_args()
    train(resume=args.resume)
