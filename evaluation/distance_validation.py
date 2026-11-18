"""
distance_validation.py

measures how accurate the LiDAR distance estimates are, using nuScenes
ground-truth 3D boxes.

for every frame:
1. project each labeled 3D box into the front camera (devkit helpers) to get
   its 2D box and its true distance: camera to box center, in meters.
2. match each YOLO detection (conf >= 0.3) to the unmatched labeled box of
   the same class with the highest IoU, if IoU >= 0.5. One-to-one.
3. estimate the detection's distance from LiDAR (median range inside the box,
   minimum 1 point) and compare it to the true distance.

labels measure to the box center and LiDAR hits the nearest surface, so
estimates should read short. Signed error (estimate minus truth) shows that.

writes every match to output/distance_validation.csv and prints median error
by true distance and by number of LiDAR points in the box.

"""

import csv
import os
import time

import numpy as np
import psycopg
from nuscenes.nuscenes import NuScenes

from config import DATABASE_URL, DATAROOT, IMAGE_HEIGHT, IMAGE_WIDTH, NUSCENES_VERSION
from evaluation.labels import labeled_boxes_in_image, match_detections
from geometry.distance import box_distance
from geometry.nuscenes_lidar import lidar_points_in_camera
from geometry.project import project_to_image

CONFIDENCE_THRESHOLD = 0.3
KEEP_FRACTION = 0.75
TRUTH = "surface"
OUTPUT_PATH = "output/distance_validation.csv"

HELD_OUT_ONLY = True

DISTANCE_BUCKETS = [(0, 10), (10, 30), (30, 1000)]
POINT_BUCKETS = [(1, 2), (3, 4), (5, 9), (10, 1000000)]

CATEGORY_TO_CLASS = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
    "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer": "person",
}

DISTANCE_BUCKETS = [(0, 10), (10, 30), (30, 1000)]
POINT_BUCKETS = [(1, 2), (3, 4), (5, 9), (10, 1000000)]


def median_or_none(values):
    if len(values) == 0:
        return None
    return float(np.median(values))


def print_bucket_table(title, rows, key, buckets):
    print(title)
    print("  range        n    median |error|   median signed error")
    for low, high in buckets:
        absolute_errors = []
        signed_errors = []
        for row in rows:
            if low <= row[key] < high + (1 if key == "num_points" else 0):
                absolute_errors.append(abs(row["error"]))
                signed_errors.append(row["error"])

        median_abs = median_or_none(absolute_errors)
        median_signed = median_or_none(signed_errors)
        abs_text = "-" if median_abs is None else f"{median_abs:.2f} m"
        signed_text = "-" if median_signed is None else f"{median_signed:+.2f} m"
        print(f"  {low}-{high:<9} {len(absolute_errors):4}    {abs_text:>10}        {signed_text:>10}")
    print()


def main():
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=DATAROOT, verbose=False)
    tuning_scenes = set()
    if HELD_OUT_ONLY:
        mini = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
        for scene in mini.scene:
            tuning_scenes.add(scene["name"])

    with psycopg.connect(DATABASE_URL) as conn:
        frames = conn.execute(
            "SELECT frame_id, sample_token, camera_token, scene_name FROM frames ORDER BY frame_id"
        ).fetchall()
        detection_rows = conn.execute(
            "SELECT frame_id, class_name, confidence, x1, y1, x2, y2 FROM detections "
            "WHERE confidence >= %s ORDER BY frame_id, confidence DESC",
            (CONFIDENCE_THRESHOLD,),
        ).fetchall()

    detections_by_frame = {}
    for row in detection_rows:
        frame_id = row[0]
        if frame_id not in detections_by_frame:
            detections_by_frame[frame_id] = []
        detections_by_frame[frame_id].append(row[1:])

    results = []
    total_detections = 0
    total_matched = 0
    matched_with_no_points = 0

    frames_checked = 0
    start_time = time.perf_counter()
    for frame_id, sample_token, camera_token, scene_name in frames:
        if HELD_OUT_ONLY and scene_name in tuning_scenes:
            continue

        frames_checked += 1
        if frames_checked % 2000 == 0:
            print("frames checked:", frames_checked, "|", round(time.perf_counter() - start_time), "s")

        detections = detections_by_frame.get(frame_id, [])
        total_detections += len(detections)
        if len(detections) == 0:
            continue
            

        sample = nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        points_camera, intrinsic, time_gap_ms = lidar_points_in_camera(nusc, lidar_token, camera_token)
        u, v, depths, ranges = project_to_image(points_camera, intrinsic, IMAGE_WIDTH, IMAGE_HEIGHT)

        labeled = labeled_boxes_in_image(nusc, camera_token)
        matches = match_detections(detections, labeled)
        total_matched += len(matches)

        for detection, label, overlap in matches:
            class_name, confidence, x1, y1, x2, y2 = detection
            estimate, num_points = box_distance(u, v, ranges, x1, y1, x2, y2, min_points=1, keep_fraction=KEEP_FRACTION)
            if estimate is None:
                matched_with_no_points += 1
                continue

            if TRUTH == "surface":
                true_distance = label["surface_distance"]
            else:
                true_distance = label["center_distance"]

            results.append({
                "frame_id": frame_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "iou": round(overlap, 3),
                "num_points": num_points,
                "true_distance": round(true_distance, 2),
                "estimate": round(estimate, 2),
                "error": round(estimate - true_distance, 2),
            })

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    all_abs = []
    all_signed = []
    for row in results:
        all_abs.append(abs(row["error"]))
        all_signed.append(row["error"])

    print("keep fraction:", KEEP_FRACTION)
    print("truth:", TRUTH)
    print("held out only:", HELD_OUT_ONLY, "| frames checked:", frames_checked)
    # “HELD_OUT_ONLY skips the 10 mini scenes used to choose KEEP_FRACTION and MIN_POINTS.”
    print("detections (conf >= 0.3):", total_detections)
    print("matched to a labeled box:", total_matched)
    print("matched but 0 LiDAR points:", matched_with_no_points)
    print("evaluated:", len(results))
    print("overall median |error|:", round(median_or_none(all_abs), 2), "m")
    print("overall median signed error:", round(median_or_none(all_signed), 2), "m")
    print()
    print_bucket_table("by true distance (m)", results, "true_distance", DISTANCE_BUCKETS)
    print_bucket_table("by LiDAR points in box", results, "num_points", POINT_BUCKETS)
    print("saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()