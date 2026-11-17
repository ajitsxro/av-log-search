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