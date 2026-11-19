CREATE TABLE IF NOT EXISTS frames (
    frame_id      SERIAL PRIMARY KEY,
    sample_token  TEXT   NOT NULL,
    camera_token  TEXT   NOT NULL UNIQUE,
    scene_name    TEXT   NOT NULL,
    timestamp_us  BIGINT NOT NULL,
    image_path    TEXT   NOT NULL,
    time_of_day   TEXT   NOT NULL CHECK (time_of_day IN ('day', 'night')),
    weather       TEXT   NOT NULL CHECK (weather IN ('clear', 'rain')),
    shard_id      INTEGER
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id  SERIAL  PRIMARY KEY,
    frame_id      INTEGER NOT NULL REFERENCES frames (frame_id),
    model_name    TEXT    NOT NULL,
    class_name    TEXT    NOT NULL,
    confidence    REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    x1            REAL    NOT NULL,
    y1            REAL    NOT NULL,
    x2            REAL    NOT NULL,
    y2            REAL    NOT NULL,
    distance_m    REAL,
    lidar_points  INTEGER
);

CREATE TABLE IF NOT EXISTS processed_frames (
    frame_id      INTEGER     NOT NULL REFERENCES frames (frame_id),
    model_name    TEXT        NOT NULL,
    processed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (frame_id, model_name)

);

ALTER TABLE detections ADD COLUMN IF NOT EXISTS distance_m REAL;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS lidar_points INTEGER;
ALTER TABLE frames ADD COLUMN IF NOT EXISTS shard_id INTEGER;

CREATE INDEX IF NOT EXISTS detections_class_distance_idx ON detections (class_name, distance_m);
CREATE INDEX IF NOT EXISTS detections_frame_id_idx ON detections (frame_id);

