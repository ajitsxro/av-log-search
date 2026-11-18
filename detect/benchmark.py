"""
benchmark.py

measures YOLO detection throughput (frames per second) on a Modal T4 GPU
for batch sizes 1, 8, 16 and 32, using every frame in the frames table.

what is timed: YOLO preprocessing + forward pass + NMS on images already
in memory. reading the JPEGs from disk is timed separately so disk speed
does not distort the GPU numbers.

each batch size gets one warmup batch, then 3 timed passes over all frames.
the median of the 3 is reported.

cost per driving hour assumes front-camera keyframes at 2 Hz
(7,200 frames per hour) and counts GPU time only, at Modal's T4 price.

"""

import time

import modal

from config import DATABASE_URL, DRIVING_CLASSES, MODEL_NAME
from modal_setup import image, volume

CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZES = [1, 8, 16, 32]
REPEATS = 3
FRAMES_PER_DRIVING_HOUR = 7200
T4_PRICE_PER_HOUR = 0.59
BENCHMARK_FRAMES = 404


app = modal.App("av-log-search-benchmark", image=image)


def run_detector(model, batch):
    model(
        batch,
        conf=CONFIDENCE_THRESHOLD,
        classes=list(DRIVING_CLASSES.keys()),
        device=0,
        agnostic_nms=True,
        verbose=False,
    )


def timed_pass(model, images, batch_size):
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()

    start_index = 0
    while start_index < len(images):
        batch = images[start_index:start_index + batch_size]
        run_detector(model, batch)
        start_index += batch_size

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return len(images) / elapsed


@app.function(gpu="T4", volumes={"/data": volume}, memory=4096, timeout=1200)
def benchmark(image_paths):
    import statistics

    import cv2
    import torch
    from ultralytics import YOLO

    load_start = time.perf_counter()
    images = []
    for image_path in image_paths:
        images.append(cv2.imread("/data/" + image_path))
    load_seconds = time.perf_counter() - load_start

    model = YOLO(MODEL_NAME)

    results = {}
    for batch_size in BATCH_SIZES:
        run_detector(model, images[:batch_size])

        fps_runs = []
        for repeat in range(REPEATS):
            fps_runs.append(timed_pass(model, images, batch_size))

        results[batch_size] = {
            "median_fps": statistics.median(fps_runs),
            "runs": fps_runs,
        }

    return {
        "gpu": torch.cuda.get_device_name(0),
        "num_frames": len(images),
        "load_seconds": load_seconds,
        "results": results,
    }


@app.local_entrypoint()
def main():
    import psycopg

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT image_path FROM frames ORDER BY frame_id LIMIT %s", (BENCHMARK_FRAMES,)
        ).fetchall()
    image_paths = []
    for row in rows:
        image_paths.append(row[0])

    report = benchmark.remote(image_paths)

    num_frames = report["num_frames"]
    load_ms_per_frame = report["load_seconds"] / num_frames * 1000
    print("gpu:", report["gpu"])
    print("frames:", num_frames)
    print("jpeg load:", round(report["load_seconds"], 2), "s total,", round(load_ms_per_frame, 1), "ms per frame")
    print()
    print("batch | median fps | runs                  | GPU $ per driving hour")

    for batch_size in BATCH_SIZES:
        result = report["results"][batch_size]
        median_fps = result["median_fps"]

        runs_text = []
        for fps in result["runs"]:
            runs_text.append(str(round(fps, 1)))

        gpu_seconds = FRAMES_PER_DRIVING_HOUR / median_fps
        cost = gpu_seconds / 3600 * T4_PRICE_PER_HOUR

        print(f"{batch_size:5} | {median_fps:10.1f} | {', '.join(runs_text):21} | ${cost:.4f}")