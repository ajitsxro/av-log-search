"""
project.py

math for moving LiDAR points into a camera image.

points are numpy arrays of shape (3, N): row 0 is x, row 1 is y, row 2 is z,
one column per point, in meters. Camera frame: x right, y down, z forward.

kept free of nuScenes and database code so it can be unit tested with
hand-computed numbers.
"""

import numpy as np


def transform_points(points, rotation_matrix, translation):
    return rotation_matrix @ points + translation.reshape(3, 1)


def inverse_transform_points(points, rotation_matrix, translation):
    return rotation_matrix.T @ (points - translation.reshape(3, 1))


def project_to_image(points_camera, intrinsic, image_width, image_height, min_depth=1.0):
    depths = points_camera[2, :]
    in_front = depths > min_depth
    points_in_front = points_camera[:, in_front]
    depths_in_front = depths[in_front]

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    u = fx * points_in_front[0, :] / depths_in_front + cx
    v = fy * points_in_front[1, :] / depths_in_front + cy

    inside_image = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

    return u[inside_image], v[inside_image], depths_in_front[inside_image]