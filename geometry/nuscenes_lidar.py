"""
nuscenes_lidar.py

loads one nuScenes LiDAR sweep and moves its points into a camera's frame.

the 4 hops:
    1. LiDAR -> car, at LiDAR time      (LiDAR calibrated_sensor, forward)
    2. car -> world, at LiDAR time      (LiDAR ego_pose, forward)
    3. world -> car, at camera time     (camera ego_pose, inverse)
    4. car -> camera                    (camera calibrated_sensor, inverse)

going through the world frame corrects for the car moving between the
LiDAR and camera timestamps.
"""

import os

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from geometry.project import inverse_transform_points, transform_points


def rotation_and_translation(record):
    rotation_matrix = Quaternion(record["rotation"]).rotation_matrix
    translation = np.array(record["translation"])
    return rotation_matrix, translation


def lidar_points_in_camera(nusc, lidar_token, camera_token):
    lidar_data = nusc.get("sample_data", lidar_token)
    camera_data = nusc.get("sample_data", camera_token)

    cloud = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data["filename"]))
    points_lidar = cloud.points[:3, :]

    lidar_calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    camera_pose = nusc.get("ego_pose", camera_data["ego_pose_token"])
    camera_calib = nusc.get("calibrated_sensor", camera_data["calibrated_sensor_token"])

    rotation, translation = rotation_and_translation(lidar_calib)
    points_car_at_lidar_time = transform_points(points_lidar, rotation, translation)

    rotation, translation = rotation_and_translation(lidar_pose)
    points_world = transform_points(points_car_at_lidar_time, rotation, translation)

    rotation, translation = rotation_and_translation(camera_pose)
    points_car_at_camera_time = inverse_transform_points(points_world, rotation, translation)

    rotation, translation = rotation_and_translation(camera_calib)
    points_camera = inverse_transform_points(points_car_at_camera_time, rotation, translation)

    intrinsic = np.array(camera_calib["camera_intrinsic"])
    time_gap_ms = (camera_data["timestamp"] - lidar_data["timestamp"]) / 1000

    return points_camera, intrinsic, time_gap_ms