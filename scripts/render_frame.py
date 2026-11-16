"""
render_frame.py

draws the ground-truth 3D boxes for one front-camera keyframe onto the
image and saves it to output/frame_with_boxes.png. a visual check that the
dataset loads and the labels line up with the image.
"""

import os

from nuscenes.nuscenes import NuScenes

DATAROOT = "data/nuscenes"
VERSION = "v1.0-mini"
CAMERA = "CAM_FRONT"
OUTPUT_PATH = "output/frame_with_boxes.png"


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])
    camera_token = sample["data"][CAMERA]

    os.makedirs("output", exist_ok=True)
    nusc.render_sample_data(camera_token, out_path=OUTPUT_PATH, verbose=False)
    print("saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()