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
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy


def detection_loss(gt_bboxes, pred_bboxes, num_classes=26):

    # [2, 280, 160, 7]
    # (x_grid, y_grid, (objectness, min_delta_x, min_delta_y, max_delta_x, max_delta_y, z, label))
    (
        gt_objectness,
        gt_delta_xy,
        gt_orient_xy,
        gt_z_coord,
        gt_width,
        gt_height,
        gt_label,
    ) = tf.split(gt_bboxes, (1, 2, 2, 1, 1, 1, 1), axis=-1)

    (
        p_objectness,
        p_delta_xy,
        p_orient_xy,
        p_z_coord,
        p_width,
        p_height,
        p_label,
    ) = tf.split(pred_bboxes, (1, 2, 2, 1, 1, 1, num_classes), axis=-1)

    # Objectness
    p_objectness = tf.sigmoid(p_objectness)
    obj_loss = binary_crossentropy(gt_objectness, p_objectness)

    # Evaluate regression only for non-zero ground truth objects
    obj_mask = tf.squeeze(gt_objectness, -1)

    # Evaluate other 6 parameters of the bboxes
    label_loss = obj_mask * sparse_categorical_crossentropy(
        gt_label, p_label, from_logits=True
    )

    delta_xy_loss = obj_mask * tf.reduce_sum(tf.abs(gt_delta_xy - p_delta_xy), axis=-1)
    delta_orient_loss = obj_mask * tf.reduce_sum(
        tf.abs(gt_orient_xy - p_orient_xy), axis=-1
    )

    z_loss = obj_mask * tf.squeeze(tf.abs(gt_z_coord - p_z_coord), -1)
    width_loss = obj_mask * tf.squeeze(tf.abs(gt_width - p_width), -1)
    height_loss = obj_mask * tf.squeeze(tf.abs(gt_height - p_height), -1)

    obj_loss = tf.reduce_sum(obj_loss)
    label_loss = tf.reduce_sum(label_loss)
    z_loss = tf.reduce_sum(z_loss)
    delta_xy_loss = tf.reduce_sum(delta_xy_loss)
    width_loss = tf.reduce_sum(width_loss)
    height_loss = tf.reduce_sum(height_loss)
    delta_orient_loss = tf.reduce_sum(delta_orient_loss)

    return (
        obj_loss,
        label_loss,
        z_loss,
        delta_xy_loss,
        width_loss,
        height_loss,
        delta_orient_loss,
    )

