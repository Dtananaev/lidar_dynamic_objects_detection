# lidar_dynamic_objects_detection

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Dtananaev/lidar_dynamic_objects_detection/blob/master/LICENSE.md) 

The method

The lidar point cloud represented as top view image where each pixel of the image corresponds to 12.5x12.5 cm. For each grid cell
we project radom point and write height and intensity
<p align="center">
  <img src="https://github.com/Dtananaev/lidar_dynamic_objects_detection/blob/master/pictures/topview.png" width="700"/>
</p>
We are doing direct regression of the 3D boxes, thus for each pixel of the image we regress confidence between 0 and 1, 7 parameters for box (dx_centroid, dy_centroid, z_centroid, width, height, dx_front, dy_front) and classes.
<p align="center">
  <img src="https://github.com/Dtananaev/lidar_dynamic_objects_detection/blob/master/pictures/box_parametrization.png" width="700"/>
</p>
The result of network (click on the image below)
[![result](https://github.com/Dtananaev/lidar_dynamic_objects_detection/blob/master/pictures/result.png)](https://youtu.be/f_HZg9Cq-h4)
The network weights could be loaded [weight](https://drive.google.com/file/d/1m8N5m2WXATgFNw88BRqEbUieiyV7p3S0/view?usp=sharing).

