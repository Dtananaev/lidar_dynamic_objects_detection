import numpy as np

from detection_3d.tools.file_io import load_bboxes, load_lidar
from detection_3d.tools.detection_helpers import (
    get_bboxes_grid,
    make_eight_points_boxes,
    make_top_view_image,
    get_bboxes_parameters_from_points,
)

from detection_3d.parameters import Parameters
from detection_3d.tools.visualization_tools import visualize_lidar, visualize_bboxes_3d
import mayavi.mlab as mlab
import numpy as np
from detection_3d.tools.file_io import read_json

param_settings = Parameters().settings

from detection_3d.reproject_to_image import intrinscs_to_matrix, get_extrinsic

bboxes_file = "/home/denis/lidar_dynamic_objects_detection/detection_3d/dataset/001/bbox_processed/00.txt"
lidar_file = "/home/denis/Pandaset/PandaSet/001/lidar_processed/00.bin"
pose_lidar = "/home/denis/Pandaset/PandaSet/001/lidar/poses.json"
pose_camera = "/home/denis/Pandaset/PandaSet/001/camera/front_camera/poses.json"
intrinsics = "/home/denis/Pandaset/PandaSet/001/camera/front_camera/intrinsics.json"

pose_l = read_json(pose_lidar)
pose_c = read_json(pose_camera)
intr_c = read_json(intrinsics)

cam_T_lid = get_extrinsic(pose_l[0], pose_c[0])

lidar = load_lidar(lidar_file)
lidar = lidar[:, :3]
lidar = np.transpose(lidar)
ones = np.ones_like(lidar[0])
lidar_hom = np.concatenate((lidar, ones[None, :]), axis=0)
lidar_cam = np.matmul(cam_T_lid, lidar_hom)
lidar_cam = np.transpose(lidar_cam)
lidar = lidar_cam[:, :3]
lidar = lidar[lidar[:, 2] >= 0]
bboxes = load_bboxes(bboxes_file)
labels = bboxes[:, -1]


print(f"bboxes[:, 1:] {bboxes.shape}")
lidar_corners_3d, orient = make_eight_points_boxes(bboxes[:, :-1])
(centroid, width, length, height, yaw,) = get_bboxes_parameters_from_points(
    lidar_corners_3d
)
print(f"shape {centroid.shape}, height {height.shape} yaw {yaw.shape}")
box_xyzlwhy = np.concatenate(
    (centroid, length[:, None], width[:, None], height[:, None], yaw[:, None]), axis=-1
)
lidar_corners_3d, orient = make_eight_points_boxes(box_xyzlwhy)

# # Shift lidar coordinate to positive quadrant
lidar_coord = np.asarray(param_settings["lidar_offset"], dtype=np.float32)
# lidar = lidar + lidar_coord
# lidar_corners_3d = lidar_corners_3d + lidar_coord[:3]
##orient = orient + lidar_coord[:3]
#  Process data

figure = visualize_lidar(lidar)
figure = visualize_bboxes_3d(lidar_corners_3d, figure, orient)
mlab.show(1)
input()
mlab.close(figure)

