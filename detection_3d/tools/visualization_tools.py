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
import mayavi.mlab as mlab
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm


def visualize_lidar(lidar, figure=None):
    """ 
    Draw lidar points
    Args:
        lidar: numpy array (n,3) of XYZ
        figure: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """

    if figure is None:
        figure = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )

    color = lidar[:, 2]
    mlab.points3d(
        lidar[:, 0],
        lidar[:, 1],
        lidar[:, 2],
        color,
        mode="point",
        scale_factor=0.3,
        figure=figure,
    )

    # draw origin
    mlab.points3d(
        0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2, figure=figure
    )
    # draw axis
    mlab.plot3d(
        [0, 2], [0, 0], [0, 0], color=(1, 0, 0), tube_radius=None, figure=figure
    )
    mlab.plot3d(
        [0, 0], [0, 2], [0, 0], color=(0, 1, 0), tube_radius=None, figure=figure
    )
    mlab.plot3d(
        [0, 0], [0, 0], [0, 2], color=(0, 0, 1), tube_radius=None, figure=figure
    )
    return figure


def visualize_bboxes_3d(lidar_corners_3d, figure=None, orientation=None):
    if figure is None:
        figure = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )

    for b in tqdm(lidar_corners_3d, desc=f"Add bboxes", total=len(lidar_corners_3d)):
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=(1, 1, 1),
                tube_radius=None,
                line_width=1,
                figure=figure,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=(1, 1, 1),
                tube_radius=None,
                line_width=1,
                figure=figure,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=(1, 1, 1),
                tube_radius=None,
                line_width=1,
                figure=figure,
            )
    if orientation is not None:
        for o in orientation:
            mlab.plot3d(
                [o[0, 0], o[1, 0]],
                [o[0, 1], o[1, 1]],
                [o[0, 2], o[1, 2]],
                color=(1, 1, 1),
                tube_radius=None,
                line_width=1,
                figure=figure,
            )
    print(f"Done")
    return figure
