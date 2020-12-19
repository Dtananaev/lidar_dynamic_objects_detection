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
from detection_3d.data_preprocessing.pandaset_tools.helpers import (
    make_xzyhwly,
    filter_boxes,
)
from detection_3d.tools.detection_helpers import (
    make_eight_points_boxes,
    get_bboxes_parameters_from_points,
)
import mayavi.mlab as mlab
from tqdm import tqdm
from detection_3d.tools.file_io import read_json
from detection_3d.data_preprocessing.pandaset_tools.transform import (
    quaternion_to_euler,
    to_transform_matrix,
    transform_lidar_box_3d,
)
from detection_3d.tools.visualization_tools import visualize_lidar, visualize_bboxes_3d


def preprocess_data(dataset_dir):
    """
    The function visualizes data from pandaset.
    Arguments:
        dataset_dir: directory with  Pandaset data
    """
    shift_lidar = [
        25,
        50,
        2.5,
    ]  # The lidar coordinates is in the middle of point cloud we shift them to left top corner of the top view image
    # the top view image applied to the area of 50x100 meters around the car, where the most dense lidar point cloud
    # Get list of data samples
    search_string = os.path.join(dataset_dir, "*")
    seq_list = sorted(glob.glob(search_string))
    for seq in tqdm(seq_list, desc="Process sequences", total=len(seq_list)):
        search_string = os.path.join(seq, "lidar", "*.pkl.gz")
        lidar_list = sorted(glob.glob(search_string))
        lidar_pose_path = os.path.join(seq, "lidar", "poses.json")
        lidar_pose = read_json(lidar_pose_path)
        for idx, lidar_path in enumerate(lidar_list):
            # Get pose of the lidar
            translation = lidar_pose[idx]["position"]
            translation = np.asarray([translation[key] for key in translation])
            rotation = lidar_pose[idx]["heading"]
            rotation = np.asarray([rotation[key] for key in rotation])
            rotation = quaternion_to_euler(*rotation)
            Rt = to_transform_matrix(translation, rotation)

            # Get respective bboxes
            bbox_path = lidar_path.split("/")
            bbox_path[-2] = "annotations/cuboids"
            bbox_path = os.path.join(*bbox_path)

            # Load data
            lidar = np.asarray(pd.read_pickle(lidar_path))
            # Get only lidar 0 (there is also lidar 1)
            lidar = lidar[lidar[:, -1] == 0]
            intensity = lidar[:, 3]
            lidar = transform_lidar_box_3d(lidar, Rt)
            # add intensity
            lidar = np.concatenate((lidar, intensity[:, None]), axis=-1)

            # Load bboxes
            bboxes = np.asarray(pd.read_pickle(bbox_path))
            labels, bboxes = make_xzyhwly(bboxes)
            corners_3d, orientation_3d = make_eight_points_boxes(bboxes)
            corners_3d = np.asarray(
                [transform_lidar_box_3d(box, Rt) for box in corners_3d]
            )
            orientation_3d = np.asarray(
                [transform_lidar_box_3d(box, Rt) for box in orientation_3d]
            )
            labels, corners_3d, orientation_3d = filter_boxes(
                labels, corners_3d, orientation_3d, lidar
            )
            centroid, width, length, height, yaw = get_bboxes_parameters_from_points(
                corners_3d
            )

            boxes_new = np.concatenate(
                (
                    centroid,
                    length[:, None],
                    width[:, None],
                    height[:, None],
                    yaw[:, None],
                ),
                axis=-1,
            )
            lidar[:, :3] = lidar[:, :3] + shift_lidar

            corners_3d, orientation_3d = make_eight_points_boxes(boxes_new)
            corners_3d = corners_3d + shift_lidar
            orientation_3d = orientation_3d + shift_lidar
            figure = visualize_bboxes_3d(corners_3d, None, orientation_3d)
            figure = visualize_lidar(lidar, figure)
            mlab.show(1)
            input()
            mlab.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 3D pandaset.")
    parser.add_argument("--dataset_dir", default="../../dataset")
    args = parser.parse_args()
    preprocess_data(args.dataset_dir)
