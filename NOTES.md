# NOTES

## Session 1
- Set up repo, Python 3.12 venv, nuScenes devkit, Postgres 16 in Docker.
- Bugs hit:
  - Downloaded the panoptic add-on instead of v1.0-mini (no images or metadata).
  - venv was on Python 3.14; the devkit supports 3.9 and 3.12, so I rebuilt on 3.12.
  - Docker daemon wasn't running, so docker compose couldn't reach the socket.
- frames table: 404 rows (all front-camera keyframes in mini). Reruns inserted 0 (idempotent).
- time_of_day / weather split: day+clear 283, night+clear 81, night+rain 40.

## Session 2
- detect_one on frame 1 (CPU, yolo11n, conf 0.3): 1 truck, 4 cars, conf 0.37 to 0.76.
- Missed vs labels:
  - Unsupported class: barriers (no COCO class).
  - Real miss: visible pedestrian in red shirt at truck's right rear, partly occluded.
  - More distant or occluded cars than YOLO found.
  - Some labeled objects are fully hidden (pedestrian behind the truck), so ground truth must filter by visibility.
- Box count at conf 0.1 / 0.3 / 0.5: 10 / 5 / 3.
  - 0.5 drops 2 real cars (0.47, 0.37).
  - 0.1 finds the missed pedestrian (person 0.13) but adds a duplicate box on one far car (IoU 0.43, under YOLO's 0.7 merge threshold).
  - Decision: batch job stores all boxes >= 0.1 with confidence; search filters at a threshold tuned by precision@10.
- Modal GPU check: T4 visible from Modal container (cuda available: True).
- Bug: modal client too old for the account's image builder version (2026.10); fixed with pip install --upgrade modal and running via python -m modal.
- agnostic_nms=True: frame 3 went 7 -> 6 detections, frame 4 went 6 -> 5 (duplicate bus boxes removed, trucks kept). Nothing else changed.
- Throughput benchmark (T4, yolo11n, FP32, conf 0.1, 404 frames, median of 3):
  | batch | fps   |
  | 1     | 95.1  |
  | 8     | 228.2 |
  | 16    | 233.7 |
  | 32    | 200.6 |
  Best: batch 16. Gains flatten after 8; batch 32 is slower (hypothesis: CPU preprocessing and memory copies).
- JPEG loading from Modal Volume: 248 ms/frame (100 s for 404) vs 4.3 ms/frame detection. Loading is the bottleneck.
- Rerun with cpu=4:
  | batch | fps   |
  | 1     | 111.1 |
  | 8     | 242.3 |
  | 16    | 239.6 |
  | 32    | 207.3 |
  JPEG load 8.9 ms/frame (was 248). Batch-32 drop persists, so CPU was not its cause.
  Load speedup is confounded: cpu=4 vs. volume cache warm on second read. Control run pending.
- Sequential end-to-end estimate: ~13 ms/frame (~76 fps). Overlapping load and inference is a future optimization.
- Control run without cpu=4: JPEG load 9.2 ms/frame (vs 8.9 with cpu=4). Load speedup was cache warmth, not CPU.
  Detection without cpu=4: batch 1 107.2, batch 8 241.3, batch 16 236.4, batch 32 204.3. cpu=4 removed.
- Headline: ~240 fps at batch 8-16 (batch 16 across 3 runs: 233.7, 239.6, 236.4). T4, yolo11n, FP32, images in memory.
- Cold reads from the Modal Volume (248 ms/frame) are the real bottleneck for new data. Fix at scale-up: parallel reads or sharded files.
- Detection job (yolo11n, conf >= 0.1, batch 16, chunks of 64): 404 frames, 4988 detections.
- Rerun after completion: 0 frames left, 0 inserted.
- Resume test: TRUNCATE, run --max-chunks 2 (128 frames, 2083 detections, matching chunks 1-2 of the full run), then rerun (276 frames). Final total 4988, same as the uninterrupted run.
- Resume run sat several minutes before the first chunk with no container output; likely waiting for a T4 (not confirmed).
- Detections by class (conf >= 0.1): car 3329, traffic light 733, person 626, truck 145, bicycle 62, bus 60, motorcycle 33. Total 4988 (~12 per frame).
- Checks: 404 processed frames, 0 exact duplicate boxes, 0 frames with no detections.
- Caveat: exact-duplicate check does not catch near-duplicate boxes below the 0.7 NMS overlap threshold.


