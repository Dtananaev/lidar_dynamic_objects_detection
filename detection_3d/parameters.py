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
from detection_3d.tools.file_io import save_to_json
from detection_3d.tools.statics import NO_SCHEDULER, RESTARTS_SCHEDULER, ADAM


class Parameters(object):
    """
    The class contains experiment parameters.
    """

    def __init__(self):
        self.settings = {
            # The directory of the dataset
            "dataset_dir": "dataset",
            "batch_size": 4,
            # The checkpoint related
            "checkpoints_dir": "log/checkpoints",
            "train_summaries": "log/summaries/train",
            "eval_summaries": "log/summaries/val",
            # Update tensorboard train images each step_summaries iterations
            "step_summaries": 100,  # to turn off make it None
            # General settings
            "seed": 2020,
            "max_epochs": 1000,
            "weight_decay": 1,
        }

        # Set special parameters
        self.settings["optimizer"] = ADAM
        self.settings["scheduler"] = SchedulerSettings.no_scheduler()
        self.settings["augmentation"] = True
        # Detection related
        self.settings["grid_meters"] = [52.0, 104.0, 8.0]  # [x,y,z ] in meters
        # [x,y,z, intensity] offset to shift all lidar points in positive coordinate quadrant
        # (all x,y,z coords >=0)
        self.settings["lidar_offset"] = [26.0, 52.0, 2.5, 0.0]
        # [x,y,z] voxel size in meters
        self.settings["voxel_size"] = [0.125, 0.125, 8.0]
        # [x,y,z] voxel size in meters
        self.settings["bbox_voxel_size"] = [0.25, 0.25, 1.0]

        # Automatically defined during training parameters
        self.settings["train_size"] = None  # the size of train set
        self.settings["val_size"] = None  # the size of val set

    def save_to_json(self, dir_to_save):
        """
        Save parameters to .json
        """
        # Check that folder is exitsts or create it
        os.makedirs(dir_to_save, exist_ok=True)
        json_filename = os.path.join(dir_to_save, "parameters.json")
        save_to_json(json_filename, self.settings)


class SchedulerSettings:
    """
    The class contains parameters for different schedulers.
    """

    def __init__(self):
        pass

    # Supported schedulers
    @staticmethod
    def no_scheduler():
        """
        Constant learning rate scheduler.
        """
        scheduler = {
            "name": NO_SCHEDULER,
            "initial_learning_rate": 1e-5,
        }
        return scheduler

    @staticmethod
    def restarts_scheduler():
        """
        The warm restarts scheduler.
        See: https://arxiv.org/abs/1608.03983
        """
        scheduler = {
            "name": RESTARTS_SCHEDULER,
            "initial_learning_rate": 1e-4,  # 2e-3
            "first_decay_steps": 80,  # Important: convertable param from epoch to iteration
            "t_mul": 2.0,
            "m_mul": 1.0,
            "alpha": 1e-6,
        }
        return scheduler
