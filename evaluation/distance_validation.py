"""
distance_validation.py

Measures LiDAR distance accuracy against nuScenes labels and compares
estimate settings.

For every frame in the chosen official nuScenes split ("train" or "val"):
1. Get labeled boxes in the front camera with their true distance
   (nearest surface by default, or box center).
2. Match each YOLO detection (conf >= 0.3) to the same-class labeled box
   with the highest IoU (>= 0.5), one-to-one.
3. Estimate each matched detection's distance at several percentiles of the
   LiDAR ranges in the central KEEP_FRACTION of its box (at least 3 points),
   and compare each estimate to the true distance.

Tune on "train" (700 scenes), then report the chosen setting on "val"
(150 scenes), so the reported error comes from scenes never used to choose it.

Reports per percentile: median absolute error, 90th-percentile absolute
error (the large misses that range queries pick up), and median signed
error (bias), overall, by class and by true distance.

Run from the repo root with:
    python -m evaluation.distance_validation
"""

import csv
import os
import time

import numpy as np
import psycopg
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from config import DATABASE_URL, DATAROOT, IMAGE_HEIGHT, IMAGE_WIDTH, NUSCENES_VERSION
from evaluation.labels import labeled_boxes_in_image, match_detections
from geometry.distance import box_distance
from geometry.nuscenes_lidar import lidar_points_in_camera
from geometry.project import project_to_image

SPLIT = "val"
PERCENTILES = [50, 25]
CONFIDENCE_THRESHOLD = 0.3
KEEP_FRACTION = 0.75
TRUTH = "surface"
OUTPUT_PATH = "output/distance_validation.csv"
DISTANCE_BUCKETS = [(0, 10), (10, 30), (30, 1000)]
REPORT_CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]


def error_summary(errors):
    if len(errors) == 0:
        return None

    absolute_errors = []
    for error in errors:
        absolute_errors.append(abs(error))

    return {
        "n": len(errors),
        "median_abs": float(np.median(absolute_errors)),
        "p90_abs": float(np.percentile(absolute_errors, 90)),
        "median_signed": float(np.median(errors)),
    }


def errors_for(rows, percentile, class_name=None, distance_range=None):
    errors = []
    for row in rows:
        if class_name is not None and row["class_name"] != class_name:
            continue
        if distance_range is not None:
            low, high = distance_range
            if not (low <= row["true_distance"] < high):
                continue
        errors.append(row[f"est_p{percentile}"] - row["true_distance"])
    return errors


def format_cell(summary):
    if summary is None:
        return "      -      "
    return f"{summary['median_abs']:5.2f} / {summary['p90_abs']:4.1f}"


def main():
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=DATAROOT, verbose=False)
    split_scenes = set(create_splits_scenes()[SPLIT])

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

    rows = []
    frames_checked = 0
    total_detections = 0
    total_matched = 0
    unknown = 0
    start_time = time.perf_counter()

    for frame_id, sample_token, camera_token, scene_name in frames:
        if scene_name not in split_scenes:
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

            if TRUTH == "surface":
                true_distance = label["surface_distance"]
            else:
                true_distance = label["center_distance"]

            row = {"frame_id": frame_id, "class_name": class_name, "true_distance": round(true_distance, 2)}
            known = True
            for percentile in PERCENTILES:
                estimate, num_points = box_distance(
                    u, v, ranges, x1, y1, x2, y2, keep_fraction=KEEP_FRACTION, percentile=percentile
                )
                if estimate is None:
                    known = False
                    break
                row[f"est_p{percentile}"] = round(estimate, 2)
            row["num_points"] = num_points

            if known:
                rows.append(row)
            else:
                unknown += 1

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print()
    print(f"split: {SPLIT} ({len(split_scenes)} scenes) | frames: {frames_checked} | truth: {TRUTH} | keep: {KEEP_FRACTION}")
    print(f"detections (conf >= {CONFIDENCE_THRESHOLD}): {total_detections} | matched: {total_matched} | "
          f"unknown (< 3 points): {unknown} | evaluated: {len(rows)}")

    print()
    print("percentile | median |err| | p90 |err| | median signed")
    for percentile in PERCENTILES:
        summary = error_summary(errors_for(rows, percentile))
        print(f"   p{percentile:<7} | {summary['median_abs']:9.2f} m | {summary['p90_abs']:7.2f} m | {summary['median_signed']:+10.2f} m")

    header = "".join(f"     p{percentile:<9}" for percentile in PERCENTILES)

    print()
    print("by class (median |err| / p90 |err|, meters)")
    print(f"  class          n  {header}")
    for class_name in REPORT_CLASSES:
        cells = []
        count = 0
        for percentile in PERCENTILES:
            summary = error_summary(errors_for(rows, percentile, class_name=class_name))
            if summary is not None:
                count = summary["n"]
            cells.append(format_cell(summary))
        print(f"  {class_name:10} {count:6}  " + "  ".join(cells))

    print()
    print("by true distance (median |err| / p90 |err|, meters)")
    print(f"  range          n  {header}")
    for low, high in DISTANCE_BUCKETS:
        cells = []
        count = 0
        for percentile in PERCENTILES:
            summary = error_summary(errors_for(rows, percentile, distance_range=(low, high)))
            if summary is not None:
                count = summary["n"]
            cells.append(format_cell(summary))
        print(f"  {low}-{high:<8} {count:6}  " + "  ".join(cells))

    print()
    print("saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()