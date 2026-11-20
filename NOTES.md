# NOTES

Log of measurements, decisions and bugs. Every number here is from a real run.

## Setup
- Python 3.12 venv (the nuScenes devkit doesn't support 3.14), Postgres 16 in Docker.
- Started on v1.0-mini: 404 front-camera keyframes. Loader is idempotent (rerun inserts 0).

## Detection (YOLO11n, COCO, 7 driving classes)
- Confidence: at 0.5, frame 1 lost 2 real cars (0.47, 0.37). At 0.1 it found a missed pedestrian (0.13) but added a duplicate car box. Decision: store everything >= 0.1, filter at 0.3 at query time.
- Same object boxed as both truck and bus. NMS is per class by default; agnostic_nms=True removed the duplicates and changed nothing else.
- Batch size (T4, FP32, 404 frames in memory, median of 3): batch 1 95.1, 8 228.2, 16 233.7, 32 200.6 fps. Using 16.
- Batch 16 across 3 runs: 233.7, 239.6, 236.4 fps.
- Loading JPEGs from the Modal Volume: 248 ms/frame vs 4.3 ms for detection.
- Adding cpu=4 seemed to cut loading to 8.9 ms. Control run without it: 9.2 ms. The speedup was the volume cache warming up, not the CPUs. Removed cpu=4.
- Resume test: truncate, run 2 chunks, rerun the rest. Final total 4,988, same as an uninterrupted run.
- Bug: Modal client too old for the account's image builder (2026.10). Fixed by upgrading and running via python -m modal.

## Scale-up (full trainval)
- 850 scenes, 34,149 front-camera frames. 99 scenes tagged night, 165 rain (keywords in scene descriptions).
- frame_id has gaps because ON CONFLICT DO NOTHING still uses up SERIAL values. Shard membership is stored in frames.shard_id instead of computed from frame_id.
- Small files were the bottleneck: 197 ms/frame to read even with 32 threads, ~5 frames/sec end to end.
- Packed images into 67 tar shards (512 each, 4.8 GB). Reads dropped to 3.9 ms/frame.
- Full run (Modal .map over T4s): ~32,100 frames in 71.2 s = 451 frames/sec end to end. Per frame: read 3.3 ms, decode 6.1 ms, detect 4.9 ms (204 fps/GPU).
- JPEG decode (6.1 ms) is now slower than detection (4.9 ms). Not fixed.
- Totals: 34,149 frames processed, 299,170 detections.
- GPU cost: TODO (Modal usage page).

## Distance from LiDAR
- 4 hops: LiDAR -> car (LiDAR time) -> world -> car (camera time) -> camera. On frame 1 the camera fired 35.5 ms before the LiDAR timestamp.
- Frame 1: 3,067 of 34,688 points land in CAM_FRONT. A car at 35 m got 16 points; LiDAR spacing predicts ~18.
- First validation compared estimates to label box centers, but LiDAR measures the nearest surface. Switching the truth to nearest-surface distance: 1.52 m -> 1.27 m median (mini, full box).
- Box edges catch background points. Keeping the central 75% of the box: 0.74 m median on mini. MIN_POINTS = 3 (3-4 point error matched 5-9 points; 1-2 points was clearly worse).
- Held-out check on the 840 non-mini scenes: 0.76 m median. Settings generalized.
- The median hid a tail. Tuned the percentile on the train split (700 scenes, 47,745 objects), median / p90:
  - p50: 0.75 / 6.36 m
  - p25: 0.52 / 4.49 m
  - p15: 0.47 / 5.06 m (tail grows again from ground points)
  - Chose p25. Person p90 went 3.9 -> 0.8 m.
- Val split (150 scenes, 10,157 objects), p50 -> p25: median 0.62 -> 0.46 m, p90 4.52 -> 3.22 m. Every class and range improved. 30+ m p90 is still 10.9 m.
- 135,897 of 299,170 detections have a distance. The rest have fewer than 3 points.
- Sanity check, frame 39045: a 379.5 px tall person at 6.3 m implies 1.89 m height. Box and distance agree.
- Traffic light distances are unverified (no traffic light labels in nuScenes).
- Test check: flipping + cx to - cx in project.py made the projection tests fail.

## Database
- Headline query (person <= 10 m, night, conf >= 0.3), EXPLAIN ANALYZE after VACUUM ANALYZE:
  - Before: parallel seq scan over ~299K rows + sort, 8.36 ms.
  - With an index on (class_name, distance_m): index scan, no sort, 0.96 ms.
- Index size: 8.6 MB (class, distance) and 2.7 MB (frame_id) on a 30 MB table.

## Search
- LLM parser: Claude Haiku, structured outputs, 3 retries, keyword fallback.
- Bug: "a cyclist close to the car" added car as a target. "The car" meant the camera car. Fixed with a prompt rule.
- "40 feet" -> 12.192 m. Vague "close to" left distance empty instead of making one up.
- Precision@10 (20 hand-written queries, checked against labels):
  - p50 distances: 0.82 strict, 0.85 lenient.
  - p25 distances: 0.86 strict, 0.89 lenient. Parse accuracy 20/20.
  - Person 10-20 m went 0.30 -> 0.80. Background points behind thin pedestrians were pushing their distance up.
- Weak queries, from inspecting the wrong frames:
  - Truck < 8 m (0.50): 3 of 5 wrong were partly hidden trucks at ~11-12 m reading ~7.5 m, because points hit the object in front. 2 of 5 had no truck label nearby (likely a van or construction vehicle).
  - >= 2 cyclists (0.50): 4 of 5 wrong were one bicycle with 2-3 overlapping boxes, below the 0.7 NMS overlap.
  - Bus/truck <= 15 m at night 0.60, bus <= 10 m 0.70.

## Latency (through the API)
- Structured search, 1,000 requests: DB p50 4.5 ms, p95 11.4 ms. Slowest is "at least 5 cars" (p95 41.4 ms, GROUP BY over many car rows).
- Text search, 200 requests: p50 762 ms, p95 1.24 s. LLM is 97% of the median. 0 fallbacks, 0 retries.
- Bug: uvicorn resolved to a global Python 3.13. Fixed with python -m uvicorn.

## Deployment
- New AWS accounts are pinned to a home region (us-east-2 here) by an AWS-managed SCP, so CreateBucket in us-west-2 was denied.
- Private S3 bucket with all 34,149 images. The API redirects to presigned URLs signed by the EC2 instance role (s3:GetObject on that bucket only). No AWS keys on the server.
- EC2 t3.small, Docker Compose: Postgres (no public port) + API on port 80. Database moved with pg_dump/pg_restore; counts match local.
- Secrets only in .env on the server. HTTP only. Instance is stopped when not demoing.
