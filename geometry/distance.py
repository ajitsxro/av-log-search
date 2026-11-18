"""
distance.py

estimates how far away a detected object is: takes the projected LiDAR
points that land inside its 2D box and returns the median of their ranges
(straight-line distance from the camera, in meters).

median, not mean: points that slip past the object's edges hit the
background and would drag a mean far off.

boxes with fewer than MIN_POINTS points return None (distance unknown).
"""

import numpy as np

MIN_POINTS = 3
KEEP_FRACTION = 0.75


def box_distance(u, v, ranges, x1, y1, x2, y2, min_points=MIN_POINTS, keep_fraction=KEEP_FRACTION):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    half_width = (x2 - x1) * keep_fraction / 2
    half_height = (y2 - y1) * keep_fraction / 2

    inner_x1 = center_x - half_width
    inner_x2 = center_x + half_width
    inner_y1 = center_y - half_height
    inner_y2 = center_y + half_height

    inside = (u >= inner_x1) & (u <= inner_x2) & (v >= inner_y1) & (v <= inner_y2)
    ranges_inside = ranges[inside]
    num_points = len(ranges_inside)

    if num_points < min_points:
        return None, num_points

    return float(np.median(ranges_inside)), num_points