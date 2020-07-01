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
import tensorflow as tf


def rot_z(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    ones = np.ones_like(c)
    zeros = np.zeros_like(c)
    return np.asarray([[c, -s, zeros], [s, c, zeros], [zeros, zeros, ones]])


def make_eight_points_boxes(bboxes_xyzlwhy):
    bboxes_xyzlwhy = np.asarray(bboxes_xyzlwhy)
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
    orient_p = (corners_3d[:, :, 0] + corners_3d[:, :, 7]) / 2.0
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


def get_voxels_grid(voxel_size, grid_meters):
    voxel_size = np.asarray(voxel_size, np.float32)
    grid_size_meters = np.asarray(grid_meters, np.float32)
    voxels_grid = np.asarray(grid_size_meters / voxel_size, np.int32)
    return voxels_grid


def get_bboxes_grid(bbox_labels, lidar_corners_3d, grid_meters, bbox_voxel_size):
    """
        The function transform lidar_corners_3d (8 points of bboxes) to
        parametrized version of bbox.
    """
    voxels_grid = get_voxels_grid(bbox_voxel_size, grid_meters)
    # Find box parameters
    centroid, width, length, height, _ = get_bboxes_parameters_from_points(
        lidar_corners_3d
    )
    # find the vector of orientation [centroid, orient_point]
    orient_point = (lidar_corners_3d[:, 1] + lidar_corners_3d[:, 2]) / 2.0

    voxel_coordinates = np.asarray(
        np.floor(centroid[:, :2] / bbox_voxel_size[:2]), np.int32
    )
    # Filter bboxes not fall in the grid
    bound_x = (voxel_coordinates[:, 0] >= 0) & (
        voxel_coordinates[:, 0] < voxels_grid[0]
    )
    bound_y = (voxel_coordinates[:, 1] >= 0) & (
        voxel_coordinates[:, 1] < voxels_grid[1]
    )
    mask = bound_x & bound_y
    # Filter all non related bboxes
    centroid = centroid[mask]
    orient_point = orient_point[mask]
    width = width[mask]
    length = length[mask]
    height = height[mask]
    bbox_labels = bbox_labels[mask]
    voxel_coordinates = voxel_coordinates[mask]
    # Confidence
    confidence = np.ones_like(width)

    # Voxels close corners to the coordinate system origin (0,0,0)
    voxels_close_corners = (
        np.asarray(voxel_coordinates, np.float32) * bbox_voxel_size[:2]
    )
    # Get x,y, coordinate
    delta_xy = centroid[:, :2] - voxels_close_corners
    orient_xy = orient_point[:, :2] - voxels_close_corners
    z_coord = centroid[:, -1]

    # print(
    #     f"confidence {confidence.shape}, delta_xy {delta_xy.shape}, orient_xy {orient_xy.shape}, z_coord {z_coord.shape}, width {width.shape}, height {height.shape}, bbox_labels {bbox_labels.shape}"
    # )
    # (x_grid, y_grid, (objectness, min_delta_x, min_delta_y, max_delta_x, max_delta_y, z, label))
    # objectness means 1 if box exists for this grid cell else 0
    output_tensor = np.zeros((voxels_grid[0], voxels_grid[1], 9), np.float32)
    if len(bbox_labels) > 0:
        data = np.concatenate(
            (
                confidence[:, None],
                delta_xy,
                orient_xy,
                z_coord[:, None],
                width[:, None],
                height[:, None],
                bbox_labels[:, None],
            ),
            axis=-1,
        )
        output_tensor[voxel_coordinates[:, 0], voxel_coordinates[:, 1]] = data
    return output_tensor


def get_boxes_from_box_grid(box_grid, bbox_voxel_size, conf_trhld=0.0):

    # Get non-zero voxels
    objectness, delta_xy, orient_xy, z_coord, width, height, label = tf.split(
        box_grid, (1, 2, 2, 1, 1, 1, -1), axis=-1
    )

    mask = box_grid[:, :, 0] > conf_trhld
    valid_idx = tf.where(mask)

    z_coord = tf.gather_nd(z_coord, valid_idx)
    width = tf.gather_nd(width, valid_idx)
    height = tf.gather_nd(height, valid_idx)
    objectness = tf.gather_nd(objectness, valid_idx)
    label = tf.gather_nd(label, valid_idx)
    delta_xy = tf.gather_nd(delta_xy, valid_idx)
    orient_xy = tf.gather_nd(orient_xy, valid_idx)
    voxels_close_corners = tf.cast(valid_idx, tf.float32) * bbox_voxel_size[None, :2]
    xy_coord = delta_xy + voxels_close_corners
    xy_orient = orient_xy + voxels_close_corners

    delta = xy_orient[:, :2] - xy_coord[:, :2]
    length = 2 * tf.norm(delta, axis=-1, keepdims=True)
    yaw = tf.expand_dims(tf.atan2(delta[:, 1], delta[:, 0]), axis=-1)

    bbox = tf.concat([xy_coord, z_coord, length, width, height, yaw], axis=-1,)
    return bbox, label, objectness


def make_top_view_image(lidar, grid_meters, voxels_size, channels=3):
    """
    The function makes top view image from lidar
    Arguments:
        lidar: lidar array of the shape [num_points, 3]
        width: width of the top view image
        height: height of the top view image
        channels: number of channels of the top view image
    """
    mask_x = (lidar[:, 0] >= 0) & (lidar[:, 0] < grid_meters[0])
    mask_y = (lidar[:, 1] >= 0) & (lidar[:, 1] < grid_meters[1])
    mask_z = (lidar[:, 2] >= 0) & (lidar[:, 2] < grid_meters[2])
    mask = mask_x & mask_y & mask_z
    lidar = lidar[mask]
    voxel_grid = get_voxels_grid(voxels_size, grid_meters)
    voxels = np.asarray(np.floor(lidar[:, :3] / voxels_size), np.int32)
    top_view = np.zeros((voxel_grid[0], voxel_grid[1], 2), np.float32)
    top_view[voxels[:, 0], voxels[:, 1], 0] = lidar[:, 2]  # z values
    top_view[voxels[:, 0], voxels[:, 1], 1] = lidar[:, 3]  # intensity values

    return top_view
