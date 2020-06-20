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
from detection_3d.data_preprocessing.pandaset_tools.transform import rot_z


def make_xzyhwly(bboxes):
    """
    Get raw data from bboxes and return xyzwlhy
    """
    label = bboxes[:, 1]
    yaw = bboxes[:, 2]
    c_x = bboxes[:, 5]
    c_y = bboxes[:, 6]
    c_z = bboxes[:, 7]
    length = bboxes[:, 8]
    width = bboxes[:, 9]
    height = bboxes[:, 10]
    new_boxes = np.asarray([c_x, c_y, c_z, length, width, height, yaw], dtype=np.float)
    return label, np.transpose(new_boxes)


def make_eight_points_boxes(bboxes_xyzlwhy):

    l = bboxes_xyzlwhy[:, 3] / 2.0
    w = bboxes_xyzlwhy[:, 4] / 2.0
    h = bboxes_xyzlwhy[:, 5] / 2.0
    # 3d bounding box corners
    x_corners = np.asarray([l, l, -l, -l, l, l, -l, -l])
    y_corners = np.asarray([w, -w, -w, w, w, -w, -w, w])
    z_corners = np.asarray([-h, -h, -h, -h, h, h, h, h])
    corners_3d = np.concatenate(([x_corners], [y_corners], [z_corners]), axis=0)
    yaw = np.asarray(bboxes_xyzlwhy[:, -1], dtype=np.float)
    corners_3d = np.transpose(corners_3d, (2, 0, 1))
    R = np.transpose(rot_z(yaw), (2, 0, 1))

    corners_3d = np.matmul(R, corners_3d)

    centroid = bboxes_xyzlwhy[:, :3]
    corners_3d += centroid[:, :, None]
    orient_p = (corners_3d[:, :, 0] + corners_3d[:, :, 5]) / 2.0
    # orient_p = (corners_3d[:, :, 0] + corners_3d[:, :, 7]) / 2.0
    orientation_3d = np.concatenate(
        (centroid[:, :, None], orient_p[:, :, None]), axis=-1
    )
    corners_3d = np.transpose(corners_3d, (0, 2, 1))
    orientation_3d = np.transpose(orientation_3d, (0, 2, 1))

    return corners_3d, orientation_3d


def get_bboxes_parameters_from_points(lidar_corners_3d):
    """
    The function returns 7 parameters of box [x, y, z, w, l, h, yaw]

    Arguments:
        lidar_corners_3d: [num_ponts, 8, 3]
    """
    centroid = (lidar_corners_3d[:, -2, :] + lidar_corners_3d[:, 0, :]) / 2.0
    delta_l = lidar_corners_3d[:, 0, :2] - lidar_corners_3d[:, 1, :2]
    delta_w = lidar_corners_3d[:, 1, :2] - lidar_corners_3d[:, 2, :2]
    width = np.linalg.norm(delta_w, axis=-1)
    length = np.linalg.norm(delta_l, axis=-1)

    height = lidar_corners_3d[:, -1, -1] - lidar_corners_3d[:, 0, -1]
    yaw = np.arctan2(delta_l[:, 1], delta_l[:, 0])

    return centroid, width, length, height, yaw


def filter_boxes(labels, bboxes_3d, orient_3d, lidar, treshold=20):
    labels_res = []
    box_res = []
    orient_res = []
    for idx, box in enumerate(bboxes_3d):
        min_x = np.min(box[:, 0])
        max_x = np.max(box[:, 0])
        min_y = np.min(box[:, 1])
        max_y = np.max(box[:, 1])
        min_z = np.min(box[:, 2])
        max_z = np.max(box[:, 2])
        mask_x = (lidar[:, 0] >= min_x) & (lidar[:, 0] <= max_x)
        mask_y = (lidar[:, 1] >= min_y) & (lidar[:, 1] <= max_y)
        mask_z = (lidar[:, 2] >= min_z) & (lidar[:, 2] <= max_z)
        mask = mask_x & mask_y & mask_z
        result = np.sum(mask.astype(float))
        if result > treshold:
            box_res.append(box)
            orient_res.append(orient_3d[idx])
            labels_res.append(labels[idx])
    return np.asarray(labels_res), np.asarray(box_res), np.asarray(orient_res)
