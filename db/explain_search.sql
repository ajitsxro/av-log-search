EXPLAIN ANALYZE
SELECT f.frame_id, f.scene_name, d.confidence, d.distance_m
FROM detections d
JOIN frames f ON f.frame_id = d.frame_id
WHERE d.class_name = 'person'
  AND d.distance_m < 10
  AND d.confidence >= 0.3
  AND f.time_of_day = 'night'
ORDER BY d.distance_m
LIMIT 10;