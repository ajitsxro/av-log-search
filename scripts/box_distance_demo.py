"""
box_distance_demo.py

for one frame, estimates the distance of every stored YOLO detection with
confidence >= 0.3 using the LiDAR points inside its box. Prints a table and
saves an image with each box labeled by class and distance.

"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import psycopg
from nuscenes.nuscenes import NuScenes
from PIL import Image

from geometry.distance import box_distance
from geometry.nuscenes_lidar import lidar_points_in_camera
from geometry.project import project_to_image
from config import DATABASE_URL, DATAROOT, NUSCENES_VERSION

FRAME_ID = 1
DISPLAY_THRESHOLD = 0.3
OUTPUT_PATH = "output/box_distances.png"


def main():
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=DATAROOT, verbose=False)

    with psycopg.connect(DATABASE_URL) as conn:
        frame = conn.execute(
            "SELECT sample_token, camera_token, image_path FROM frames WHERE frame_id = %s",
            (FRAME_ID,),
        ).fetchone()
        detections = conn.execute(
            "SELECT class_name, confidence, x1, y1, x2, y2 FROM detections "
            "WHERE frame_id = %s AND confidence >= %s ORDER BY confidence DESC",
            (FRAME_ID, DISPLAY_THRESHOLD),
        ).fetchall()

    sample_token = frame[0]
    camera_token = frame[1]
    image_path = frame[2]

    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]

    points_camera, intrinsic, time_gap_ms = lidar_points_in_camera(nusc, lidar_token, camera_token)

    image = Image.open(os.path.join(DATAROOT, image_path))
    image_width, image_height = image.size
    u, v, depths, ranges = project_to_image(points_camera, intrinsic, image_width, image_height)

    plt.figure(figsize=(16, 9))
    plt.imshow(image)
    plt.axis("off")

    print("class          conf  points  distance")
    for detection in detections:
        class_name, confidence, x1, y1, x2, y2 = detection
        distance, num_points = box_distance(u, v, ranges, x1, y1, x2, y2)

        if distance is None:
            distance_text = "unknown"
        else:
            distance_text = str(round(distance, 1)) + " m"

        print(f"{class_name:14} {confidence:.2f}  {num_points:6}  {distance_text}")

        rectangle = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2)
        plt.gca().add_patch(rectangle)
        plt.text(x1, y1 - 6, class_name + " " + distance_text, color="yellow", fontsize=10, backgroundcolor="black")

    os.makedirs("output", exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=100)
    print("saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()