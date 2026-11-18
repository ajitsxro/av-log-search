"""
labels.py

ground-truth helpers shared by the evaluation scripts: turns nuScenes 3D
labels into front-camera 2D boxes with true distances, and matches YOLO
detections to them one-to-one by class and IoU.
"""

import numpy as np
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

from config import IMAGE_HEIGHT, IMAGE_WIDTH

IOU_THRESHOLD = 0.5

CATEGORY_TO_CLASS = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
    "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer": "person",
}


def distance_to_nearest_surface(box):
    rotation = box.orientation.rotation_matrix
    camera_in_box_frame = rotation.T @ (np.zeros(3) - box.center)

    half_length = box.wlh[1] / 2
    half_width = box.wlh[0] / 2
    half_height = box.wlh[2] / 2

    nearest_x = min(max(camera_in_box_frame[0], -half_length), half_length)
    nearest_y = min(max(camera_in_box_frame[1], -half_width), half_width)
    nearest_z = min(max(camera_in_box_frame[2], -half_height), half_height)
    nearest_point = np.array([nearest_x, nearest_y, nearest_z])

    return float(np.linalg.norm(camera_in_box_frame - nearest_point))


def labeled_boxes_in_image(nusc, camera_token):
    _, boxes, intrinsic = nusc.get_sample_data(camera_token, box_vis_level=BoxVisibility.ANY)

    labeled = []
    for box in boxes:
        if box.name not in CATEGORY_TO_CLASS:
            continue

        corners = box.corners()
        if corners[2, :].min() <= 0.1:
            continue

        corners_2d = view_points(corners, intrinsic, normalize=True)
        x1 = max(0.0, float(corners_2d[0, :].min()))
        y1 = max(0.0, float(corners_2d[1, :].min()))
        x2 = min(float(IMAGE_WIDTH), float(corners_2d[0, :].max()))
        y2 = min(float(IMAGE_HEIGHT), float(corners_2d[1, :].max()))

        labeled.append({
            "class_name": CATEGORY_TO_CLASS[box.name],
            "box": (x1, y1, x2, y2),
            "center_distance": float(np.linalg.norm(box.center)),
            "surface_distance": distance_to_nearest_surface(box),
        })
    return labeled


def iou(box_a, box_b):
    overlap_x1 = max(box_a[0], box_b[0])
    overlap_y1 = max(box_a[1], box_b[1])
    overlap_x2 = min(box_a[2], box_b[2])
    overlap_y2 = min(box_a[3], box_b[3])

    overlap_width = max(0.0, overlap_x2 - overlap_x1)
    overlap_height = max(0.0, overlap_y2 - overlap_y1)
    overlap_area = overlap_width * overlap_height

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - overlap_area

    if union_area <= 0:
        return 0.0
    return overlap_area / union_area


def match_detections(detections, labeled, iou_threshold=IOU_THRESHOLD):
    matches = []
    used_labels = set()

    for detection in detections:
        class_name, confidence, x1, y1, x2, y2 = detection

        best_index = None
        best_iou = 0.0
        for index in range(len(labeled)):
            if index in used_labels:
                continue
            if labeled[index]["class_name"] != class_name:
                continue
            overlap = iou((x1, y1, x2, y2), labeled[index]["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            used_labels.add(best_index)
            matches.append((detection, labeled[best_index], best_iou))

    return matches