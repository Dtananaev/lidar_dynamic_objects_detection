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

import numpy as np
from detection_3d.data_preprocessing.pandaset_tools.transform import (
    eulerAnglesToRotationMatrix,
)


def random_rotate_lidar_boxes(
    lidar, lidar_corners_3d, min_angle=-np.pi / 4, max_angle=np.pi / 4
):
    yaw = np.random.uniform(min_angle, max_angle)
    R = eulerAnglesToRotationMatrix([0, 0, yaw])
    lidar = np.transpose(lidar)
    lidar_corners_3d = np.transpose(lidar_corners_3d, (0, 2, 1))

    lidar[:3] = np.matmul(R, lidar[:3])
    lidar_corners_3d = np.matmul(R, lidar_corners_3d)

    lidar_corners_3d = np.transpose(lidar_corners_3d, (0, 2, 1))
    lidar = np.transpose(lidar)
    return lidar, lidar_corners_3d


def random_flip_x_lidar_boxes(lidar, lidar_corners_3d):
    lidar[:, 0] = -lidar[:, 0]
    lidar_corners_3d[:, :, 0] = -lidar_corners_3d[:, :, 0]
    return lidar, lidar_corners_3d


def random_flip_y_lidar_boxes(lidar, lidar_corners_3d):
    lidar[:, 1] = -lidar[:, 1]
    lidar_corners_3d[:, :, 1] = -lidar_corners_3d[:, :, 1]
    return lidar, lidar_corners_3d
