"""
detect_one.py

runs a pretrained YOLO detector on the first frame in the frames table,
on the CPU. prints each detected object and saves a copy of the image
with boxes drawn. a first check that detection works before moving to a GPU.

only driving-relevant COCO classes are kept.
"""

import os

import psycopg
from ultralytics import YOLO

DATAROOT = "data/nuscenes"
DATABASE_URL = "postgresql://av:av@localhost:5432/avsearch"
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.3
OUTPUT_PATH = "output/detect_one.jpg"
DRIVING_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}


def main():
    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT frame_id, image_path FROM frames ORDER BY frame_id LIMIT 1"
        ).fetchone()
    frame_id = row[0]
    image_path = row[1]
    full_path = os.path.join(DATAROOT, image_path)

    model = YOLO(MODEL_NAME)
    class_ids = list(DRIVING_CLASSES.keys())
    results = model(full_path, conf=CONFIDENCE_THRESHOLD, classes=class_ids, device="cpu", verbose=False)
    result = results[0]

    print("frame", frame_id, image_path)
    for box in result.boxes:
        class_name = DRIVING_CLASSES[int(box.cls[0])]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(class_name, round(confidence, 2), [round(x1), round(y1), round(x2), round(y2)])

    os.makedirs("output", exist_ok=True)
    result.save(filename=OUTPUT_PATH)
    print("saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()