"""
config.py

settings shared across the project. DATABASE_URL can be overridden with an
environment variable, so the same code runs locally and when deployed.
"""

import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://av:av@localhost:5432/avsearch")
DATAROOT = "data/nuscenes"
NUSCENES_VERSION = "v1.0-mini"
CAMERA = "CAM_FRONT"
SCHEMA_PATH = "db/schema.sql"
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900

MODEL_NAME = "yolo11n.pt"
DRIVING_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}