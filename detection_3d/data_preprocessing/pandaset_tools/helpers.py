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

labels = {
    "Cones": 0,
    "Towed Object": 1,
    "Semi-truck": 2,
    "Train": 3,
    "Temporary Construction Barriers": 4,
    "Rolling Containers": 5,
    "Animals - Other": 6,
    "Pylons": 7,
    "Emergency Vehicle": 8,
    "Motorcycle": 9,
    "Construction Signs": 10,
    "Medium-sized Truck": 11,
    "Other Vehicle - Uncommon": 12,
    "Tram / Subway": 13,
    "Road Barriers": 14,
    "Bus": 15,
    "Pedestrian with Object": 16,
    "Personal Mobility Device": 17,
    "Signs": 18,
    "Other Vehicle - Pedicab": 19,
    "Pedestrian": 20,
    "Car": 21,
    "Other Vehicle - Construction Vehicle": 22,
    "Bicycle": 23,
    "Motorized Scooter": 24,
    "Pickup Truck": 25,
}


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
