# Dynamic objects detection in LiDAR

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Dtananaev/lidar_dynamic_objects_detection/blob/master/LICENSE.md) 

The method

The lidar point cloud represented as top view image where each pixel of the image corresponds to 12.5x12.5 cm. For each grid cell
we project radom point and write height and intensity
<p align="center">
  <img src="https://github.com/Dtananaev/lidar_dynamic_objects_detection/blob/master/pictures/topview.png" width="900"/>
</p>
We are doing direct regression of the 3D boxes, thus for each pixel of the image we regress confidence between 0 and 1, 7 parameters for box (dx_centroid, dy_centroid, z_centroid, width, height, dx_front, dy_front) and classes.
<p align="center">
  <img src="https://github.com/Dtananaev/lidar_dynamic_objects_detection/blob/master/pictures/box_parametrization.png" width="1500"/>
</p>
We apply binary cross entrophy for confidence loss, l1 loss for all box parameters regression and softmax loss for classes regression.
The confidence map computed from ground truth boxes. We assign the closest to the box centroid cell as confidence 1.0 (green on the image above)
and 0 otherwise. We apply confidence loss for all the pixels. Other losses  applied only for those pixels where we have confidence ground truth 1.0. 

The result of network (click on the image below)
[![result](https://github.com/Dtananaev/lidar_dynamic_objects_detection/blob/master/pictures/result.png)](https://youtu.be/f_HZg9Cq-h4)
The network weights could be loaded [weight](https://drive.google.com/file/d/1m8N5m2WXATgFNw88BRqEbUieiyV7p3S0/view?usp=sharing).

