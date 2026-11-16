CREATE TABLE IF NOT EXISTS frames (
    frame_id      SERIAL PRIMARY KEY,
    sample_token  TEXT   NOT NULL,
    camera_token  TEXT   NOT NULL UNIQUE,
    scene_name    TEXT   NOT NULL,
    timestamp_us  BIGINT NOT NULL,
    image_path    TEXT   NOT NULL,
    time_of_day   TEXT   NOT NULL CHECK (time_of_day IN ('day', 'night')),
    weather       TEXT   NOT NULL CHECK (weather IN ('clear', 'rain'))
);