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
from detection_3d.tools.file_io import read_json, load_bboxes, load_lidar
from detection_3d.data_preprocessing.pandaset_tools.transform import (
    to_transform_matrix,
    quaternion_to_euler,
)
from detection_3d.tools.visualization_tools import visualize_lidar, visualize_bboxes_3d
import mayavi.mlab as mlab

from detection_3d.tools.detection_helpers import make_eight_points_boxes
from detection_3d.tools.visualization_tools import visualize_bboxes_on_image
from PIL import Image


def pose_to_matrix(pose):
    translation = pose["position"]
    translation = np.asarray([translation[key] for key in translation])
    rotation = pose["heading"]
    rotation = np.asarray([rotation[key] for key in rotation])
    rotation = quaternion_to_euler(*rotation)
    Rt = to_transform_matrix(translation, rotation)
    return Rt


def intrinscs_to_matrix(fx, fy, cx, cy):
    intrinsics = np.eye(3)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    return intrinsics


def get_extrinsic(lidar_pose, image_pose):
    world_T_lidar = pose_to_matrix(lidar_pose)
    world_T_cam = pose_to_matrix(image_pose)
    cam_T_world = np.linalg.inv(world_T_cam)
    cam_T_lidar = np.dot(cam_T_world, world_T_lidar)
    return cam_T_lidar


def filter_boxes(box):

    mask_x = box[:, 0] >= 0
    mask_y = box[:, 1] >= 0
    mask_z = box[:, 2] >= 0
    mask = (mask_x & mask_y & mask_z).all()
    # print(f"box o {box[:, 0] } b1 {box[:, 1] } b2 {box[:, 2] }")
    # input()
    mask = (mask_z).all()
    return mask


def project_boxes_to_cam(bboxes, extrinsics):
    bboxes = np.transpose(bboxes, (0, 2, 1))
    ones = np.ones_like(bboxes[:, 0, :])
    bbox_hom = np.concatenate((bboxes, ones[:, None, :]), axis=1)
    cam_box = np.matmul(extrinsics, bbox_hom)
    cam_box = cam_box[:, :3, :]
    cam_box = np.transpose(cam_box, (0, 2, 1))
    res = []

    for box in cam_box:
        if (box[:, 2] > 0).any():
            res.append(box)
    res = np.asarray(res)
    # # # print(f"cam_box {cam_box.shape}, res {res.shape}")
    # cam_box = res / np.expand_dims(res[:, 2, :], axis=1)

    # box_2d = np.matmul(intrinsics, cam_box)
    # box_2d = np.transpose(box_2d, (0, 2, 1))

    return res  # box_2d[:, :, :2]


def project_to_image(box, intrinsics):

    if len(box) > 0:
        box = box / np.expand_dims(box[:, :, 2], axis=-1)
        box = np.transpose(box, (0, 2, 1))
        box = np.matmul(intrinsics, box)
        box = np.transpose(box, (0, 2, 1))
        box = box[:, :, :2]

    return box


def project_lidar_to_cam(lidar, extrinsics):
    lidar = np.transpose(lidar)
    ones = np.ones_like(lidar[0])
    lidar_hom = np.concatenate((lidar, ones[None, :]), axis=0)
    lidar_cam = np.matmul(extrinsics, lidar_hom)
    lidar_cam = np.transpose(lidar_cam)
    return lidar_cam


def get_orient(lidar_corners_3d):

    orient_p1 = (lidar_corners_3d[:, 3, :] + lidar_corners_3d[:, 6, :]) / 2.0

    orient_p2 = (lidar_corners_3d[:, 0, :] + lidar_corners_3d[:, 5, :]) / 2.0
    centroid = (orient_p1 + orient_p2) / 2.0

    orient_3d = np.concatenate((centroid[:, None, :], orient_p2[:, None, :]), axis=1)

    return orient_3d


def reproject_and_save(dataset_dir, inference_dir, camera_name, seq):

    seq_number = seq.split("/")[-1]
    bbox_list = sorted(glob.glob(seq + "/*.txt"))
    image_f_dir = os.path.join(dataset_dir, seq_number, "camera", camera_name)
    intrinsics_f = read_json(image_f_dir + "/intrinsics.json")
    intr_f = intrinscs_to_matrix(**intrinsics_f)
    poses_f = read_json(image_f_dir + "/poses.json")
    lidar_poses = read_json(
        os.path.join(dataset_dir, seq_number, "lidar", "poses.json")
    )
    output_dir = os.path.join(inference_dir, "image_bboxes", seq_number, camera_name)
    os.makedirs(output_dir, exist_ok=True)
    for idx, bbox_path in enumerate(bbox_list):
        name = os.path.splitext(os.path.basename(bbox_path))[0]
        cam_T_lidar = get_extrinsic(lidar_poses[idx], poses_f[idx])

        # lidar_file = os.path.join(
        #    dataset_dir, seq_number, "lidar_processed", name + ".bin"
        # )
        # lidar = load_lidar(lidar_file)
        # lidar = lidar[:, :3]
        # Load boxes

        bboxes = load_bboxes(bbox_path, label_string=False)
        labels = bboxes[:, -1]

        lidar_corners_3d, _ = make_eight_points_boxes(bboxes[:, :-1])
        orient_3d = get_orient(lidar_corners_3d)

        lidar_corners_3d = project_boxes_to_cam(lidar_corners_3d, cam_T_lidar)
        orient_3d = project_boxes_to_cam(orient_3d, cam_T_lidar)

        # lidar = project_lidar_to_cam(lidar, cam_T_lidar)
        # figure = visualize_lidar(lidar)
        # figure = visualize_bboxes_3d(lidar_corners_3d, figure)
        # project to image
        image_path = os.path.join(image_f_dir, name + ".jpg")
        image_f = np.asarray(Image.open(image_path)) / 255
        box_2d = project_to_image(lidar_corners_3d, intr_f)
        orient_2d = project_to_image(orient_3d, intr_f)
        image = visualize_bboxes_on_image(image_f, box_2d, labels, orient_2d) * 255

        image_name = os.path.join(output_dir, name + ".png")
        img = Image.fromarray(image.astype("uint8"))
        img.save(image_name)


def reproject(inference_dir, dataset_dir):
    bbox_dir = os.path.join(inference_dir, "bboxes")

    sequences = sorted(glob.glob(bbox_dir + "/*"))
    for seq in sequences:
        reproject_and_save(dataset_dir, inference_dir, "back_camera", seq)
        reproject_and_save(dataset_dir, inference_dir, "front_camera", seq)
        reproject_and_save(dataset_dir, inference_dir, "front_left_camera", seq)
        reproject_and_save(dataset_dir, inference_dir, "front_right_camera", seq)
        reproject_and_save(dataset_dir, inference_dir, "right_camera", seq)
        reproject_and_save(dataset_dir, inference_dir, "left_camera", seq)
        print(f"sequence {seq} is done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproject bboxes to images set.")
    parser.add_argument(
        "--inference_dir", default="inference",
    )

    parser.add_argument("--dataset_dir", default="dataset")
    args = parser.parse_args()
    reproject(args.inference_dir, args.dataset_dir)
