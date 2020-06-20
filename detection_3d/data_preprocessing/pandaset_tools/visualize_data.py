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
import numpy as np
import os
import glob
import pandas as pd
from detection_3d.tools.visualization_tools import visualize_lidar, visualize_bboxes_3d
from detection_3d.data_preprocessing.pandaset_tools.helpers import (
    make_xzyhwly,
    make_eight_points_boxes,
)
import mayavi.mlab as mlab


def show_data(dataset_dir):
    """
    The function visualizes data from pandaset.
    Arguments:
        dataset_dir: directory with  Pandaset data
    """
    # Get list of data samples
    search_string = os.path.join(dataset_dir, "*", "lidar", "*.pkl.gz")
    lidar_list = sorted(glob.glob(search_string))

    for lidar_path in lidar_list:
        # Get respective bboxes
        bboxe_path = lidar_path.split("/")
        bboxe_path[-2] = "annotations/cuboids"
        bboxe_path = os.path.join(*bboxe_path)

        # Load data
        lidar = np.asarray(pd.read_pickle(lidar_path))
        bboxes = np.asarray(pd.read_pickle(bboxe_path))
        _, bboxes = make_xzyhwly(bboxes)
        bboxes, orientation_3d = make_eight_points_boxes(bboxes)
        # Visualize only lidar 0 (there is also lidar 1)
        lidar = lidar[lidar[:, -1] == 0]
        figure = visualize_bboxes_3d(bboxes, None, orientation_3d)
        figure = visualize_lidar(lidar, figure)
        mlab.show(1)
        input()
        mlab.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D pandaset.")
    parser.add_argument("--dataset_dir", default="../../dataset")
    args = parser.parse_args()
    show_data(args.dataset_dir)
