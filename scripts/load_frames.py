"""
load_frames.py

create the frames table (if missing) and inserts one row per front-camera
keyframe in nuScenes mini.

time_of_day and weather come from keyword matching on the scene description
("night", "rain"), so they are set per scene, not per frame.

safe to rerun: camera_token is UNIQUE and rows that already exist are skipped.
"""

import psycopg
from nuscenes.nuscenes import NuScenes
from config import CAMERA, DATABASE_URL, DATAROOT, NUSCENES_VERSION, SCHEMA_PATH


INSERT_SQL = """
    INSERT INTO frames
        (sample_token, camera_token, scene_name, timestamp_us, image_path, time_of_day, weather)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (camera_token) DO NOTHING
"""


def tags_from_description(description):
    text = description.lower()

    time_of_day = "day"
    if "night" in text:
        time_of_day = "night"

    weather = "clear"
    if "rain" in text:
        weather = "rain"

    return time_of_day, weather


def collect_frames(nusc):
    frames = []
    for scene in nusc.scene:
        time_of_day, weather = tags_from_description(scene["description"])

        sample_token = scene["first_sample_token"]
        while sample_token != "":
            sample = nusc.get("sample", sample_token)
            camera_token = sample["data"][CAMERA]
            camera_data = nusc.get("sample_data", camera_token)

            frame = (
                sample_token,
                camera_token,
                scene["name"],
                sample["timestamp"],
                camera_data["filename"],
                time_of_day,
                weather,
            )
            frames.append(frame)

            sample_token = sample["next"]
    return frames


def main():
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=DATAROOT, verbose=False)
    frames = collect_frames(nusc)
    print("collected", len(frames), "frames from nuScenes")

    with open(SCHEMA_PATH) as schema_file:
        schema_sql = schema_file.read()

    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(schema_sql)

        inserted = 0
        with conn.cursor() as cur:
            for frame in frames:
                cur.execute(INSERT_SQL, frame)
                inserted += cur.rowcount

        total = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    print("inserted", inserted, "new rows")
    print("frames table now has", total, "rows")


if __name__ == "__main__":
    main()