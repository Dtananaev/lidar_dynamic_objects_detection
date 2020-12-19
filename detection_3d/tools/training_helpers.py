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
import os
import glob
import tensorflow as tf
import copy
from detection_3d.tools.statics import NO_SCHEDULER, RESTARTS_SCHEDULER, ADAM


def setup_gpu():
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        # Will not allocate all memory but only necessary amount
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def initialize_model(model, input_shape):
    """
    Helper tf2 specific model initialization (need for saving mechanism)
    """
    sample = tf.zeros(input_shape, tf.float32)
    model.predict(sample)


def load_model(checkpoints_dir, model, resume):
    """
    Resume model from given checkpoint
    """
    start_epoch = 0
    if resume:
        search_string = os.path.join(checkpoints_dir, "*")
        checkpoints_list = sorted(glob.glob(search_string))
        if len(checkpoints_list) > 0:
            current_epoch = int(os.path.split(checkpoints_list[-1])[-1].split("-")[-1])
            model = tf.keras.models.load_model(checkpoints_list[-1])
            start_epoch = current_epoch + 1  # we should continue from the next epoch
            print(f"RESUME TRAINING FROM CHECKPOINT: {checkpoints_list[-1]}.")
        else:
            print(f"CAN'T RESUME TRAINING! NO CHECKPOINT FOUND! START NEW TRAINING!")
    return start_epoch, model


def get_optimizer(optimizer_name, scheduler, num_iter_per_epoch):
    if scheduler["name"] == NO_SCHEDULER:
        learning_rate = scheduler["initial_learning_rate"]
    elif scheduler["name"] == RESTARTS_SCHEDULER:
        tmp_scheduler = copy.deepcopy(scheduler)
        tmp_scheduler["first_decay_steps"] = (
            scheduler["first_decay_steps"] * num_iter_per_epoch
        )
        learning_rate = tf.keras.experimental.CosineDecayRestarts(**tmp_scheduler)

    if optimizer_name == ADAM:
        optimizer_type = tf.keras.optimizers.Adam(learning_rate)
    else:
        ValueError("Error: Unknow optimizer {}".format(optimizer_name))

    return learning_rate, optimizer_type
