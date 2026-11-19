# NOTES

## Part 1
- Set up repo, Python 3.12 venv, nuScenes devkit, Postgres 16 in Docker.
- Bugs hit:
  - Downloaded the panoptic add-on instead of v1.0-mini (no images or metadata).
  - venv was on Python 3.14; the devkit supports 3.9 and 3.12, so I rebuilt on 3.12.
  - Docker daemon wasn't running, so docker compose couldn't reach the socket.
- frames table: 404 rows (all front-camera keyframes in mini). Reruns inserted 0 (idempotent).
- time_of_day / weather split: day+clear 283, night+clear 81, night+rain 40.

## Part 2
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


## Part 3
- LiDAR projection on scene 0, frame 1: 3,067 of 34,688 points (8.8%) land in CAM_FRONT. Depth 4.5 to 98.1 m. Overlay lines up with truck, road, barriers.
- CAM_FRONT K: fx = fy = 1266.4, cx = 816.3, cy = 491.5 (principal point ~16 px right, ~42 px below image center).
- Camera fired 35.5 ms before LiDAR timestamp; 4-hop chain through world frame corrects for car motion.
- Far objects get few points; box edges catch background points (reason for median and a min-points rule).
- Box distances, frame 1 (conf >= 0.3): truck 655 pts 13.0 m; cars 16 pts 35.5 m, 5 pts 38.1 m, 4 pts unknown, 3 pts unknown.
- Expected points on a car at 35 m from LiDAR spacing (0.20 m horizontal, 0.79 m vertical): ~18. Measured 16.
- Dark car got 3 points in a box similar in size to the 16-point car. Hypothesis: low reflectivity. Not confirmed.
- MIN_POINTS = 5 decides 3 of 4 cars here; set it from validation error, not by guess.
- Distance validation (conf >= 0.3, IoU >= 0.5, same class, all 404 frames):
  2364 detections, 1246 matched, 100 matched with 0 points, 1146 evaluated.
  Median |error| 1.52 m, signed -0.79 m (reads short: surface vs center).
  By true distance: 0-10 m 1.14 (+0.61), 10-30 m 1.05 (-0.67), 30+ m 2.29 (-1.65).
  By points: 1-2 3.44, 3-4 2.21, 5-9 2.06, 10+ 1.17.
- MIN_POINTS 5 -> 3: 3-4 points error (2.21 m) ~ 5-9 (2.06 m); 1-2 points clearly worse (3.44 m).
- Hypothesis: box edges catch background (close range, reads long) and road in front (far range, reads short).
- Keep-fraction experiment (center truth), median |error| by range 0-10 / 10-30 / 30+:
  1.0: 1.14 / 1.05 / 2.29, unknown (0-2 pts) 15%
  0.75: 0.90 / 1.29 / 2.12, unknown 24%
  0.5: 0.18 / 1.44 / 2.12, unknown 38%
- Finding: at 1.0, background (reads long) and surface-vs-center (reads short) partly cancelled. Cropping removes background; remaining error is a consistent short bias.
- Fix: validate against label's nearest-surface distance (what LiDAR measures, and what "within X m" means). Keep center-based numbers reported too.
- Surface-truth validation (label nearest-surface distance), median |error| by range 0-10 / 10-30 / 30+:
  1.0: 1.53 / 1.20 / 1.29, overall 1.27, unknown 15%
  0.75: 0.78 / 0.69 / 0.86, overall 0.74, unknown 24%
  0.5: 0.47 / 0.55 / 0.84, overall 0.57, unknown 38%
- Decision: KEEP_FRACTION 0.75, MIN_POINTS 3. Meets <1 m in all ranges; 0.5 is more accurate but leaves 38% of objects without distance.
- Estimates read +0.4 to +0.8 m long (median of visible surface vs. single nearest point).
- Caveat: fraction chosen on the same data the error is reported on. Re-measure on new scenes at scale-up.
- Center-truth numbers reported too (overall 1.52 m at full box).
- Backfill: 4988 detections, 2403 with distance, 2585 unknown. Rerun processed 0.
- At conf >= 0.3, with distance: person 288/297, truck 96/98, bus 46/46, bicycle 15/15, motorcycle 9/10, car 893/1522, traffic light 81/376.
  Above 0.3: 40% unknown. Below 0.3: 63% unknown.
- Traffic lights: low coverage (LiDAR max +10° elevation). Distances unverified (no traffic light labels in nuScenes).
- First hand-written search, person < 10 m at night, conf >= 0.3: 3 frames (336, 353, 355), all scene-1094, 8.9 to 9.7 m. Likely one person across frames.
- Unit tests: 9 passed. Changing + cx to - cx in project.py made the projection tests fail, so the tests catch real bugs.




- Post-cleanup distance validation identical (0.74 m overall, all buckets match).



## Scale-up
- Downloaded trainval metadata (0.43 GB) and keyframes part 1 (4.2 GB); extracted only CAM_FRONT and LIDAR_TOP.
- load_frames on v1.0-trainval: 3741 frames, 94 scenes (one mini scene was also in part 1). 3337 new rows, 0 duplicate images, so mini and trainval share tokens.
- Part 1 added 0 night and 0 rain frames: day/clear 3620, night/clear 81, night/rain 40.
- All 10 trainval keyframe parts downloaded; extracted CAM_FRONT and LIDAR_TOP only.
- load_frames: 34149 frames, 0 scenes skipped, 30408 new rows inserted (3741 + 30408 = 34149).
- trainval metadata: 850 scenes, 99 with "night", 165 with "rain" in the description.
- Shards: 67 tar files (512 images each, last 357), 4.8 GB total (~73 MB/shard, ~140 KB/image). frames.shard_id records membership.
- frame_ids have gaps: INSERT ... ON CONFLICT DO NOTHING still consumes SERIAL values on skipped rows, so shards are assigned by position and stored, not computed from frame_id.
- Pre-shard test (2 chunks, 32 read threads, small files): 197 ms/frame read, 5.0 frames/sec end to end.
- Shard test (2 shards, 620 frames): read 3.9 ms/frame (was 197 with small files + 32 threads, ~50x faster), decode 7.5 ms/frame, detection 204.9 fps/GPU (warmup excluded). Wall 29.9 s, mostly container startup.
- Next bottleneck: single-threaded JPEG decode (7.5 ms) > detection (~4.9 ms/frame). Not fixed yet.
- Full detection run (sharded, Modal .map): ~32,100 frames in 71.2 s = 451 frames/sec end to end across parallel T4s.
  Per frame: read 3.3 ms, decode 6.1 ms, detect 4.9 ms (204 fps/GPU). ~70 fps/GPU end to end.
- Totals: 34,149 processed frames, 299,170 detections (~8.8 per frame).
- vs. pre-shard: ~5 frames/sec end to end. Actual run cost: <check Modal usage page>.
- Distance backfill at scale: 294,182 detections over 31,756 frames in 84 s (2.6 ms/frame, local). Known 133,494 (45%), unknown 160,688 (55%).
- 32,160 of 34,149 frames have at least one detection.
- Held-out distance validation (840 non-mini scenes, conf >= 0.3, keep 0.75, surface truth):
  137,938 detections, 74,965 matched, 10,029 with 0 points, 64,936 evaluated.
  Median |error| 0.76 m (mini: 0.74), signed +0.61 m.
  By range: 0-10 m 0.86, 10-30 m 0.68, 30+ m 1.01.
  By points: 1-2 1.78, 3-4 0.58, 5-9 0.56, 10+ 0.82.
- Settings generalize. 30+ m slightly above the 1 m target.

## Part 4
- VACUUM ANALYZE before measuring (backfill had updated every detection row).
- Query: person, distance < 10 m, conf >= 0.3, night, ORDER BY distance LIMIT 10 (warm, 3rd run).
  Before: Parallel Seq Scan over ~299K rows + sort, 8.36 ms.
  After (detections (class_name, distance_m)): Index Scan, no sort, ~419 index entries read, 0.96 ms. 8.7x faster.
- Index cost: table 30 MB, class/distance index 8.6 MB, frame_id index 2.7 MB.
- Remaining work: night check on 320 joined frames; not worth denormalizing at ~1 ms.
- SearchFilter (Pydantic, extra=forbid) + filter_to_sql (parameterized): 11 unit tests, 20 total passing.
- Headline query on full data (person <= 10 m, night, conf >= 0.3): 10 frames, conf 0.79-0.87, nearest 5.5-9.4 m (mini had 3 results at conf 0.30-0.50).
- Issue: top 10 spans only 6 scenes (consecutive frames of the same moment). Consider max_per_scene; decide with the precision eval.
- API (FastAPI + psycopg pool): POST /search, GET /frame/{id}, GET /image/{id}.
- Headline query via API: same 10 frames as direct SQL, db_ms 9.73 (single request). Slower than the 0.96 ms EXPLAIN because ranking by confidence + GROUP BY prevents the early stop that ORDER BY distance allowed.
- Invalid class returns 422 with allowed values (will feed LLM retries).
- Sanity check frame 39045: 379.5 px tall person at 6.3 m implies ~1.89 m height (fx 1266.4). Box and distance agree.
- Bug: `uvicorn` resolved to a global Python 3.13 install; fixed with python -m uvicorn.
- LLM parser (claude-haiku-5-5, structured outputs, 3 retries, keyword fallback). 5 test queries, all parsed by LLM on attempt 1.
  Latency: first call 2909 ms (connection setup), then 753-1460 ms. DB is ~10 ms, so LLM dominates text-search latency.
- Failure: "a cyclist close to the car" added class car (meant the ego car). Fixed with a prompt rule.
- Limitation: classes are OR'd with one shared min_count, so "4 cars and a bus" cannot be expressed exactly. LLM silently picks the closest filter.
- Correct: 40 feet -> 12.192 m; vague "close to" left distance null instead of inventing one.
- /search/text: with no API key in the server env, keyword fallback answered both queries correctly and reported the reason (parse 0.5 ms vs ~1 s for LLM). Unknown-object query returns 422.
- First request paid ~720 ms to import the Anthropic SDK (one time).
- Key now loaded from .env (gitignored) via python-dotenv.
- Precision@10 (20 hand-written queries, labels as truth, surface distance):
  search strict 0.82, lenient 0.85, end-to-end 0.82, parse accuracy 20/20.
  Worst: person 10-20 m 0.30, person > 30 m 0.40, truck < 8 m day 0.40, >= 2 cyclists 0.50, bus/truck <= 15 m night 0.60, bus <= 10 m 0.70.
  "bicycle at night" returned 1 frame. "pedestrians in rain at night": 10 results from 3 scenes.
- Hypotheses (to verify): background points inflate distance on thin pedestrians + confidence ranking favors close people; truck/bus/car class confusion; duplicate boxes on one bicycle.
- Percentile experiment, train split (700 scenes, 47,745 objects), median / p90 |error|:
  overall p50 0.75/6.36, p35 0.59/4.56, p25 0.52/4.49, p15 0.47/5.06.
  person p50 0.49/3.9 -> p25 0.40/0.8 (tail fixed). car 0.81/8.3 -> 0.55/6.2. bicycle 2.56/9.4 -> 0.84/3.2.
  p15 lowers medians but grows the tail (ground/foreground points). Chose p25.
- Remaining tails: car p90 6.2 m (likely occlusion or glass, unverified); 30+ m p90 13.4 m.
- Val split (150 scenes, 10,157 objects), p50 -> p25: median |error| 0.62 -> 0.46 m, p90 4.52 -> 3.22 m, signed +0.54 -> +0.38.
  person p90 4.0 -> 0.8 m. All classes and ranges improved. 30+ m p90 12.7 -> 10.9 m (still a tail).
- PERCENTILE = 25 chosen on train, confirmed on val. Replaces the earlier 0.76 m (median, mini-tuned) as the reported number.
- Recomputed all distances with p25 (299,170 boxes, 87 s). Known = 135,897, unchanged as expected.
- Precision@10 after p25: strict 0.82 -> 0.86, lenient 0.85 -> 0.89, end-to-end 0.86, parse 20/20.
  person 10-20 m 0.30 -> 0.80, person > 30 m 0.40 -> 0.70, truck < 8 m 0.40 -> 0.50, cars > 40 m rain 0.90 -> 0.80.
- Remaining weak: truck < 8 m 0.50, >= 2 cyclists 0.50, bus/truck <= 15 m night 0.60, bus <= 10 m 0.70.

