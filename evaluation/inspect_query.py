"""
inspect_query.py

Shows why one eval query's results were judged correct or wrong. For each
returned frame it prints the detections of the asked classes (confidence,
estimated distance) next to the labeled objects of those classes (surface
distance, visibility).

Run from the repo root with:
    python -m evaluation.inspect_query 15
where 15 is the query's position in eval_queries.json, counting from 0.
"""

import json
import sys

import psycopg
from nuscenes.nuscenes import NuScenes

from api.filters import SearchFilter, filter_to_sql
from config import DATABASE_URL, DATAROOT, NUSCENES_VERSION
from evaluation.labels import labeled_boxes_in_image
from evaluation.precision import QUERIES_PATH, STRICT_VISIBILITY, frame_is_correct


def main():
    query_index = int(sys.argv[1])
    with open(QUERIES_PATH) as queries_file:
        queries = json.load(queries_file)

    item = queries[query_index]
    search_filter = SearchFilter.model_validate(item["expected"])
    print("query:", item["query"])
    print("filter:", search_filter.model_dump(exclude_defaults=True))
    print()

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=DATAROOT, verbose=False)
    sql, params = filter_to_sql(search_filter)

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(sql, params).fetchall()

        for row in rows:
            frame_id = row[0]
            camera_token = conn.execute(
                "SELECT camera_token FROM frames WHERE frame_id = %s", (frame_id,)
            ).fetchone()[0]
            detections = conn.execute(
                "SELECT class_name, confidence, distance_m FROM detections "
                "WHERE frame_id = %s AND class_name = ANY(%s) AND confidence >= %s "
                "ORDER BY confidence DESC",
                (frame_id, search_filter.classes, search_filter.min_confidence),
            ).fetchall()
            labeled = labeled_boxes_in_image(nusc, camera_token)
            correct = frame_is_correct(labeled, search_filter, STRICT_VISIBILITY)

            print("frame", frame_id, "|", row[1], "|", "CORRECT" if correct else "WRONG")

            print("   detections:")
            for class_name, confidence, distance_m in detections:
                distance_text = "unknown" if distance_m is None else f"{distance_m:.1f} m"
                print(f"      {class_name:10} conf {confidence:.2f}   est {distance_text}")

            print("   labels of asked classes:")
            found_label = False
            for label in labeled:
                if label["class_name"] in search_filter.classes:
                    found_label = True
                    print(
                        f"      {label['class_name']:10} surface {label['surface_distance']:.1f} m"
                        f"   visibility {label['visibility']}"
                    )
            if not found_label:
                print("      (none)")
            print()


if __name__ == "__main__":
    main()