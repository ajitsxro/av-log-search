"""
distance.py

Estimates how far away a detected object is from the projected LiDAR
points that land inside the central KEEP_FRACTION of its 2D box.

The estimate is a percentile of those points' ranges (straight-line
distance from the camera, meters). 50 is the median. Lower percentiles
ignore more of the points that slip past thin objects and hit the
background behind them.

Boxes with fewer than MIN_POINTS points in the central region return
None (distance unknown).
"""

import numpy as np

MIN_POINTS = 3
KEEP_FRACTION = 0.75
PERCENTILE = 25


def box_distance(u, v, ranges, x1, y1, x2, y2,
                 min_points=MIN_POINTS, keep_fraction=KEEP_FRACTION, percentile=PERCENTILE):
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

    return float(np.percentile(ranges_inside, percentile)), num_points