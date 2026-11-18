"""
backfill_distances.py

fills distance_m and lidar_points for every detection not processed yet,
using the LiDAR sweep from the same keyframe.

distance_m: median straight-line distance (meters, from the camera) of the
LiDAR points in the central 75% of the box; NULL if fewer than 3 points.
lidar_points: number of points in that central region.

lidar_points IS NULL means "not processed yet", so reruns skip finished rows.
one transaction per frame, so a crash loses at most one frame of work.
covers all stored detections (conf >= 0.1); search filters by confidence.

"""

import psycopg
import time
from nuscenes.nuscenes import NuScenes

from geometry.distance import box_distance
from geometry.nuscenes_lidar import lidar_points_in_camera
from geometry.project import project_to_image
from config import DATABASE_URL, DATAROOT, IMAGE_HEIGHT, IMAGE_WIDTH, NUSCENES_VERSION, SCHEMA_PATH

SELECT_TODO_SQL = """
    SELECT d.detection_id, d.frame_id, f.sample_token, f.camera_token, d.x1, d.y1, d.x2, d.y2
    FROM detections d
    JOIN frames f ON f.frame_id = d.frame_id
    WHERE d.lidar_points IS NULL
    ORDER BY d.frame_id
"""
UPDATE_SQL = """
    UPDATE detections SET distance_m = %s, lidar_points = %s WHERE detection_id = %s
"""


def main():
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=DATAROOT, verbose=False)

    with open(SCHEMA_PATH) as schema_file:
        schema_sql = schema_file.read()

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(schema_sql)
        rows = conn.execute(SELECT_TODO_SQL).fetchall()
        print("detections to process:", len(rows))

        boxes_by_frame = {}
        tokens_by_frame = {}
        for row in rows:
            detection_id, frame_id, sample_token, camera_token, x1, y1, x2, y2 = row
            if frame_id not in boxes_by_frame:
                boxes_by_frame[frame_id] = []
                tokens_by_frame[frame_id] = (sample_token, camera_token)
            boxes_by_frame[frame_id].append((detection_id, x1, y1, x2, y2))

        known = 0
        unknown = 0
        start_time = time.perf_counter()
        frames_done = 0
        for frame_id in boxes_by_frame:
            sample_token, camera_token = tokens_by_frame[frame_id]
            sample = nusc.get("sample", sample_token)
            lidar_token = sample["data"]["LIDAR_TOP"]

            points_camera, intrinsic, time_gap_ms = lidar_points_in_camera(nusc, lidar_token, camera_token)
            u, v, depths, ranges = project_to_image(points_camera, intrinsic, IMAGE_WIDTH, IMAGE_HEIGHT)

            with conn.transaction():
                for detection_id, x1, y1, x2, y2 in boxes_by_frame[frame_id]:
                    distance, num_points = box_distance(u, v, ranges, x1, y1, x2, y2)
                    conn.execute(UPDATE_SQL, (distance, num_points, detection_id))
                    if distance is None:
                        unknown += 1
                    else:
                        known += 1

            frames_done += 1
            if frames_done % 1000 == 0:
                elapsed = time.perf_counter() - start_time
                print("frames:", frames_done, "of", len(boxes_by_frame), "|", round(elapsed), "s elapsed")

        total_seconds = time.perf_counter() - start_time

    print("frames processed:", len(boxes_by_frame))
    print("distance known:", known)
    print("distance unknown (< 3 points):", unknown)
    if frames_done > 0:
        print("time:", round(total_seconds), "s =", round(total_seconds / frames_done * 1000, 1), "ms/frame")

    print("frames processed:", len(boxes_by_frame))
    print("distance known:", known)
    print("distance unknown (< 3 points):", unknown)


if __name__ == "__main__":
    main()