"""
run_detections.py

runs YOLO on every frame not yet processed by this model and stores each
detected box in the detections table.

how reruns stay safe:
- processed_frames records (frame_id, model_name) for every finished frame,
  including frames with zero detections.
- each chunk of frames is saved in one transaction: its detections and its
  processed_frames rows commit together or not at all.
- only frames missing from processed_frames are sent to the GPU, so a rerun
  after a crash continues where it stopped, with no duplicates.

boxes are stored at confidence >= 0.1; search filters to a higher threshold.

"""

import modal

from config import DATABASE_URL, DRIVING_CLASSES, MODEL_NAME, SCHEMA_PATH
from modal_setup import image, volume

MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 16
CHUNK_SIZE = 64


SELECT_TODO_SQL = """
    SELECT f.frame_id, f.image_path
    FROM frames f
    WHERE NOT EXISTS (
        SELECT 1 FROM processed_frames p
        WHERE p.frame_id = f.frame_id AND p.model_name = %s
    )
    ORDER BY f.frame_id
"""
INSERT_DETECTION_SQL = """
    INSERT INTO detections (frame_id, model_name, class_name, confidence, x1, y1, x2, y2)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""
INSERT_PROCESSED_SQL = """
    INSERT INTO processed_frames (frame_id, model_name) VALUES (%s, %s)
"""


app = modal.App("av-log-search-detections", image=image)


@app.function(gpu="T4", volumes={"/data": volume}, memory=4096, timeout=1200)
def detect_chunk(frames):
    import cv2
    from ultralytics import YOLO

    model = YOLO(MODEL_NAME)
    class_ids = list(DRIVING_CLASSES.keys())

    output = []
    start_index = 0
    while start_index < len(frames):
        batch_frames = frames[start_index:start_index + BATCH_SIZE]

        images = []
        for frame in batch_frames:
            images.append(cv2.imread("/data/" + frame[1]))

        results = model(
            images,
            conf=CONFIDENCE_THRESHOLD,
            classes=class_ids,
            device=0,
            agnostic_nms=True,
            verbose=False,
        )

        for i in range(len(batch_frames)):
            detections = []
            for box in results[i].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = DRIVING_CLASSES[int(box.cls[0])]
                confidence = float(box.conf[0])
                detections.append((class_name, confidence, x1, y1, x2, y2))
            output.append({"frame_id": batch_frames[i][0], "detections": detections})

        start_index += BATCH_SIZE

    return output


def save_chunk(conn, chunk_output):
    inserted = 0
    with conn.transaction():
        with conn.cursor() as cur:
            for frame_result in chunk_output:
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
def main(max_chunks: int = 0):
    import psycopg

    with open(SCHEMA_PATH) as schema_file:
        schema_sql = schema_file.read()

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(schema_sql)
        todo = conn.execute(SELECT_TODO_SQL, (MODEL_NAME,)).fetchall()
        print("frames left to process:", len(todo))

        chunks_done = 0
        run_inserted = 0
        start_index = 0
        while start_index < len(todo):
            if max_chunks > 0 and chunks_done == max_chunks:
                print("stopping early after", chunks_done, "chunks")
                break

            chunk = todo[start_index:start_index + CHUNK_SIZE]
            chunk_output = detect_chunk.remote(chunk)
            inserted = save_chunk(conn, chunk_output)

            chunks_done += 1
            run_inserted += inserted
            print("chunk", chunks_done, "|", len(chunk), "frames |", inserted, "detections saved")
            start_index += CHUNK_SIZE

        processed_count = conn.execute(
            "SELECT COUNT(*) FROM processed_frames WHERE model_name = %s", (MODEL_NAME,)
        ).fetchone()[0]
        detection_count = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE model_name = %s", (MODEL_NAME,)
        ).fetchone()[0]

    print("this run saved", run_inserted, "detections")
    print("processed frames in table:", processed_count)
    print("detections in table:", detection_count)