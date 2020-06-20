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


def rot_z(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    ones = np.ones_like(c)
    zeros = np.zeros_like(c)
    return np.asarray([[c, -s, zeros], [s, c, zeros], [zeros, zeros, ones]])


def make_xzyhwly(bboxes):
    """
    Get raw data from bboxes and return xyzwlhy
    """
    label = bboxes[:, 1]
    yaw = bboxes[:, 2]
    c_x = bboxes[:, 5]
    c_y = bboxes[:, 6]
    c_z = bboxes[:, 7]
    width = bboxes[:, 8]
    length = bboxes[:, 9]
    height = bboxes[:, 10]
    new_boxes = np.asarray([c_x, c_y, c_z, width, length, height, yaw], dtype=np.float)
    return label, np.transpose(new_boxes)


def make_eight_points_boxes(bboxes_xyzwhly):

    l = bboxes_xyzwhly[:, 3] / 2.0
    w = bboxes_xyzwhly[:, 4] / 2.0
    h = bboxes_xyzwhly[:, 5] / 2.0
    # 3d bounding box corners
    x_corners = np.asarray([l, l, -l, -l, l, l, -l, -l])
    y_corners = np.asarray([w, -w, -w, w, w, -w, -w, w])
    z_corners = np.asarray([-h, -h, -h, -h, h, h, h, h])
    corners_3d = np.concatenate(([x_corners], [y_corners], [z_corners]), axis=0)
    yaw = np.asarray(bboxes_xyzwhly[:, -1], dtype=np.float)
    corners_3d = np.transpose(corners_3d, (2, 0, 1))
    R = np.transpose(rot_z(yaw), (2, 0, 1))

    corners_3d = np.matmul(R, corners_3d)

    centroid = bboxes_xyzwhly[:, :3]
    corners_3d += centroid[:, :, None]

    orient_p = (corners_3d[:, :, 0] + corners_3d[:, :, 7]) / 2.0
    orientation_3d = np.concatenate(
        (centroid[:, :, None], orient_p[:, :, None]), axis=-1
    )
    corners_3d = np.transpose(corners_3d, (0, 2, 1))
    orientation_3d = np.transpose(orientation_3d, (0, 2, 1))

    return corners_3d, orientation_3d
