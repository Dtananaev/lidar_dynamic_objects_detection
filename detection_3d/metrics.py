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
from perception.tools.io_file import save_to_json
import numpy as np
import os


class EpochMetrics:
    """
    The class computes the loss
    for train and validation step
    """

    def __init__(self):
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")

    def reset(self):
        """
        Reset all metrics to zero (need to do each epoch)
        """
        self.train_loss.reset_states()
        self.val_loss.reset_states()

    def save_to_json(self, dir_to_save):
        """
        Save all metrics to the json file
        """

        # Check that folder is exitsts or create it
        os.makedirs(dir_to_save, exist_ok=True)
        json_filename = os.path.join(dir_to_save, "epoch_metrics.json")
        # fill the dict
        metrics_dict = {
            "train_loss": str(self.train_loss.result().numpy()),
            "val_loss": str(self.val_loss.result().numpy()),
        }
        save_to_json(json_filename, metrics_dict)

    def print_metrics(self):
        """
        Print all metrics
        """
        train_loss = np.around(self.train_loss.result().numpy(), decimals=2)
        val_loss = np.around(self.val_loss.result().numpy(), decimals=2)

        template = "train_loss {}, val_loss {}".format(train_loss, val_loss)
        print(template)
