"""
detect_gpu.py

runs YOLO on the first few frames from the frames table on a Modal T4 GPU
and prints the detections. images come from the "nuscenes-mini" Modal
Volume, mounted at /data, so image paths from the database work unchanged.

"""

import modal

DATABASE_URL = "postgresql://av:av@localhost:5432/avsearch"
VOLUME_NAME = "nuscenes-mini"
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.3
NUM_FRAMES = 4
DRIVING_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics")
)
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("av-log-search-detect", image=image)


@app.function(gpu="T4", volumes={"/data": volume})
def detect_frames(image_paths):
    from ultralytics import YOLO

    model = YOLO(MODEL_NAME)
    class_ids = list(DRIVING_CLASSES.keys())

    full_paths = []
    for image_path in image_paths:
        full_paths.append("/data/" + image_path)

    results = model(
        full_paths,
        conf=CONFIDENCE_THRESHOLD,
        classes=class_ids,
        device=0,
        agnostic_nms=True,
        verbose=False,
    )

    output = []
    for i in range(len(image_paths)):
        detections = []
        for box in results[i].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_name": DRIVING_CLASSES[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 2),
                "box": [round(x1), round(y1), round(x2), round(y2)],
            })
        output.append({"image_path": image_paths[i], "detections": detections})
    return output


@app.local_entrypoint()
def main():
    import psycopg

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT frame_id, image_path FROM frames ORDER BY frame_id LIMIT %s",
            (NUM_FRAMES,),
        ).fetchall()

    image_paths = []
    for row in rows:
        image_paths.append(row[1])

    output = detect_frames.remote(image_paths)

    for i in range(len(rows)):
        print("frame", rows[i][0], output[i]["image_path"])
        for detection in output[i]["detections"]:
            print("  ", detection["class_name"], detection["confidence"], detection["box"])

