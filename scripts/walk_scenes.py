"""
walk_scenes.py
lists every front-camera keyframe in nuScenes mini
for each scene, follows its keyframes in time order (each keyframe stores
the ID of the next one) and prints:
    scene name | timestamp | front camera image path | scene description

at the end, checks that the number of frames printed matches the number
nuScenes says exist, so a bug can't silently skip frames

"""



from nuscenes.nuscenes import NuScenes

DATAROOT = "data/nuscenes"
VERSION = "v1.0-mini"
CAMERA = "CAM_FRONT"


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    printed_count = 0
    expected_count = 0

    for scene in nusc.scene:
        expected_count += scene["nbr_samples"]

        sample_token = scene["first_sample_token"]
        while sample_token != "":
            sample = nusc.get("sample", sample_token)

            camera_token = sample["data"][CAMERA]
            camera_data = nusc.get("sample_data", camera_token)

            line = scene["name"] + " | " + str(sample["timestamp"]) + " | "
            line += camera_data["filename"] + " | " + scene["description"]
            print(line)
            printed_count += 1

            sample_token = sample["next"]

    print("printed", printed_count, "frames, expected", expected_count)
    if printed_count != expected_count:
        raise SystemExit("count mismatch")


if __name__ == "__main__":
    main()