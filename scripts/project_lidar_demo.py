"""
project_lidar_demo.py

projects the LiDAR sweep for the first keyframe of scene 0 onto its
front-camera image, colors each point by depth, and saves the plot to
output/lidar_on_image.png.

"""

import os

import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from PIL import Image

from geometry.nuscenes_lidar import lidar_points_in_camera
from geometry.project import project_to_image
from config import DATAROOT, NUSCENES_VERSION


OUTPUT_PATH = "output/lidar_on_image.png"


def main():
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=DATAROOT, verbose=False)

    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])
    lidar_token = sample["data"]["LIDAR_TOP"]
    camera_token = sample["data"]["CAM_FRONT"]

    points_camera, intrinsic, time_gap_ms = lidar_points_in_camera(nusc, lidar_token, camera_token)

    camera_data = nusc.get("sample_data", camera_token)
    image = Image.open(os.path.join(DATAROOT, camera_data["filename"]))
    image_width, image_height = image.size

    u, v, depths, ranges = project_to_image(points_camera, intrinsic, image_width, image_height)

    print("intrinsic matrix K:")
    print(intrinsic.round(1))
    print("camera minus lidar time gap:", round(time_gap_ms, 1), "ms")
    print("lidar points total:", points_camera.shape[1])
    print("points landing in the image:", len(depths))
    print("depth range in image:", round(depths.min(), 1), "to", round(depths.max(), 1), "m")

    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(16, 9))
    plt.imshow(image)
    plt.scatter(u, v, c=depths, s=2, cmap="viridis")
    plt.colorbar(label="depth (m)")
    plt.axis("off")
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=100)
    print("saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()