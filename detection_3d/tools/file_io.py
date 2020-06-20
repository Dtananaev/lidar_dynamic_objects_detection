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

import json
import numpy as np


def read_json(json_filename):
    with open(json_filename) as json_file:
        data = json.load(json_file)
        return data


def save_bboxes_to_file(filename, centroid, width, length, height, alpha, label):

    if centroid is not None:
        with open(filename, "w") as the_file:
            for c, w, l, h, a, lbl in zip(
                centroid, width, length, height, alpha, label
            ):
                data = "{} {} {} {} {} {} {} {}\n".format(
                    c[0], c[1], c[2], w, l, h, a, lbl
                )
                the_file.write(data)


def load_lidar(lidar_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(lidar_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


def save_lidar(lidar_filename, scan):
    scan = scan.reshape((-1))
    scan.tofile(lidar_filename)
