"""
run_detections.py

runs YOLO on every frame not yet processed by this model and stores each
detected box in the detections table.

images come from shard files on the Modal volume (up to 512 images per tar
file, built by scripts/build_shards.py), so each GPU call does one large
read instead of hundreds of small ones. one GPU call handles the
unprocessed frames of one shard.

shards run in parallel on several T4 containers (Modal .map); each finished
shard is saved as it arrives. read, decode and detection time are measured
separately; detection is timed after one untimed warmup batch.

how reruns stay safe:
- processed_frames records (frame_id, model_name) for every finished frame,
  including frames with zero detections.
- Each shard's detections and processed_frames rows commit in one transaction.
- Only frames missing from processed_frames are sent to the GPU.
boxes are stored at confidence >= 0.1; search filters to a higher threshold.

"""

import time

import modal

from config import DATABASE_URL, DRIVING_CLASSES, MODEL_NAME, SCHEMA_PATH, shard_filename
from modal_setup import image, volume

CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 16

SELECT_PENDING_SQL = """
    SELECT f.frame_id, f.shard_id
    FROM frames f
    WHERE f.shard_id IS NOT NULL
      AND NOT EXISTS (
        SELECT 1 FROM processed_frames p
        WHERE p.frame_id = f.frame_id AND p.model_name = %s
      )
    ORDER BY f.shard_id, f.frame_id
"""
COUNT_UNSHARDED_SQL = "SELECT COUNT(*) FROM frames WHERE shard_id IS NULL"
INSERT_DETECTION_SQL = """
    INSERT INTO detections (frame_id, model_name, class_name, confidence, x1, y1, x2, y2)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""
INSERT_PROCESSED_SQL = """
    INSERT INTO processed_frames (frame_id, model_name) VALUES (%s, %s)
"""

app = modal.App("av-log-search-detections", image=image)


@app.function(gpu="T4", volumes={"/data": volume}, memory=8192, timeout=1800)
def detect_shard(shard_id, frame_ids):
    import tarfile

    import cv2
    import numpy as np
    from ultralytics import YOLO

    wanted = set(frame_ids)

    read_start = time.perf_counter()
    jpeg_bytes_by_frame = {}
    with tarfile.open("/data/shards/" + shard_filename(shard_id), "r") as tar:
        for member in tar.getmembers():
            frame_id = int(member.name.split(".")[0])
            if frame_id in wanted:
                jpeg_bytes_by_frame[frame_id] = tar.extractfile(member).read()
    read_seconds = time.perf_counter() - read_start

    decode_start = time.perf_counter()
    images = []
    for frame_id in frame_ids:
        buffer = np.frombuffer(jpeg_bytes_by_frame[frame_id], dtype=np.uint8)
        images.append(cv2.imdecode(buffer, cv2.IMREAD_COLOR))
    decode_seconds = time.perf_counter() - decode_start

    model = YOLO(MODEL_NAME)
    class_ids = list(DRIVING_CLASSES.keys())

    model(images[:BATCH_SIZE], conf=CONFIDENCE_THRESHOLD, classes=class_ids,
          device=0, agnostic_nms=True, verbose=False)

    detect_start = time.perf_counter()
    output = []
    start_index = 0
    while start_index < len(frame_ids):
        batch_frame_ids = frame_ids[start_index:start_index + BATCH_SIZE]
        batch_images = images[start_index:start_index + BATCH_SIZE]

        results = model(batch_images, conf=CONFIDENCE_THRESHOLD, classes=class_ids,
                        device=0, agnostic_nms=True, verbose=False)

        for i in range(len(batch_frame_ids)):
            detections = []
            for box in results[i].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = DRIVING_CLASSES[int(box.cls[0])]
                confidence = float(box.conf[0])
                detections.append((class_name, confidence, x1, y1, x2, y2))
            output.append({"frame_id": batch_frame_ids[i], "detections": detections})

        start_index += BATCH_SIZE
    detect_seconds = time.perf_counter() - detect_start

    return {
        "frames": output,
        "read_seconds": read_seconds,
        "decode_seconds": decode_seconds,
        "detect_seconds": detect_seconds,
    }


def save_frames(conn, frame_results):
    inserted = 0
    with conn.transaction():
        with conn.cursor() as cur:
            for frame_result in frame_results:
                frame_id = frame_result["frame_id"]
                for detection in frame_result["detections"]:
                    class_name, confidence, x1, y1, x2, y2 = detection
                    cur.execute(
                        INSERT_DETECTION_SQL,
                        (frame_id, MODEL_NAME, class_name, confidence, x1, y1, x2, y2),
                    )
                    inserted += 1
                cur.execute(INSERT_PROCESSED_SQL, (frame_id, MODEL_NAME))
    return inserted


@app.local_entrypoint()
def main(max_shards: int = 0):
    import psycopg

    with open(SCHEMA_PATH) as schema_file:
        schema_sql = schema_file.read()

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(schema_sql)

        unsharded = conn.execute(COUNT_UNSHARDED_SQL).fetchone()[0]
        if unsharded > 0:
            print("warning:", unsharded, "frames have no shard yet; run scripts.build_shards first")

        pending = conn.execute(SELECT_PENDING_SQL, (MODEL_NAME,)).fetchall()
        print("frames left to process:", len(pending))

        frames_by_shard = {}
        for frame_id, shard_id in pending:
            if shard_id not in frames_by_shard:
                frames_by_shard[shard_id] = []
            frames_by_shard[shard_id].append(frame_id)

        shard_ids = list(frames_by_shard.keys())
        if max_shards > 0:
            shard_ids = shard_ids[:max_shards]

        frame_id_lists = []
        for shard_id in shard_ids:
            frame_id_lists.append(frames_by_shard[shard_id])
        print("shards to run:", len(shard_ids))

        wall_start = time.perf_counter()
        shards_done = 0
        frames_done = 0
        run_inserted = 0
        total_read = 0.0
        total_decode = 0.0
        total_detect = 0.0

        for result in detect_shard.map(shard_ids, frame_id_lists):
            inserted = save_frames(conn, result["frames"])

            shards_done += 1
            frames_done += len(result["frames"])
            run_inserted += inserted
            total_read += result["read_seconds"]
            total_decode += result["decode_seconds"]
            total_detect += result["detect_seconds"]

            print(
                f"shard {shards_done}/{len(shard_ids)} | {len(result['frames'])} frames | "
                f"{inserted} detections | read {result['read_seconds']:.1f}s | "
                f"decode {result['decode_seconds']:.1f}s | detect {result['detect_seconds']:.1f}s"
            )

        wall_seconds = time.perf_counter() - wall_start

        processed_count = conn.execute(
            "SELECT COUNT(*) FROM processed_frames WHERE model_name = %s", (MODEL_NAME,)
        ).fetchone()[0]
        detection_count = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE model_name = %s", (MODEL_NAME,)
        ).fetchone()[0]

    print("this run saved", run_inserted, "detections")
    print("processed frames in table:", processed_count)
    print("detections in table:", detection_count)

    if frames_done > 0:
        print("wall time:", round(wall_seconds, 1), "s =", round(frames_done / wall_seconds, 1), "frames/sec end to end")
        print("shard read:", round(total_read / frames_done * 1000, 1), "ms/frame")
        print("jpeg decode:", round(total_decode / frames_done * 1000, 1), "ms/frame")
        print("detection:", round(frames_done / total_detect, 1), "frames/sec per GPU")