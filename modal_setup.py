"""
modal_setup.py

shared Modal pieces for the GPU jobs: the container image (Python 3.12,
OpenCV system libraries, ultralytics, and this project's config) and the
volume holding the front-camera images.
"""

import modal

VOLUME_NAME = "nuscenes-mini"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics")
    .add_local_python_source("config", "modal_setup")
)
volume = modal.Volume.from_name(VOLUME_NAME)