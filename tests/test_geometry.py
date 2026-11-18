"""
test_geometry.py

unit tests for the projection and distance math, checked against values
computed by hand. No dataset or database needed.
 
"""

import numpy as np
import pytest

from geometry.distance import box_distance
from geometry.project import inverse_transform_points, project_to_image, transform_points

SIMPLE_K = np.array([
    [1000.0, 0.0, 800.0],
    [0.0, 1000.0, 450.0],
    [0.0, 0.0, 1.0],
])
REAL_FRONT_K = np.array([
    [1266.4, 0.0, 816.3],
    [0.0, 1266.4, 491.5],
    [0.0, 0.0, 1.0],
])
ROTATE_90_ABOUT_Z = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900


def test_projects_point_to_hand_computed_pixel():
    """(2, 1, 10): u = 1000*2/10 + 800 = 1000, v = 1000*1/10 + 450 = 550, range = sqrt(105)."""
    points = np.array([[2.0], [1.0], [10.0]])
    u, v, depths, ranges = project_to_image(points, SIMPLE_K, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert u[0] == pytest.approx(1000.0)
    assert v[0] == pytest.approx(550.0)
    assert depths[0] == pytest.approx(10.0)
    assert ranges[0] == pytest.approx(np.sqrt(105.0))


def test_projects_point_with_real_front_camera_intrinsics():
    """u = 1266.4*2/10 + 816.3 = 1069.58, v = 1266.4*1/10 + 491.5 = 618.14."""
    points = np.array([[2.0], [1.0], [10.0]])
    u, v, depths, ranges = project_to_image(points, REAL_FRONT_K, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert u[0] == pytest.approx(1069.58)
    assert v[0] == pytest.approx(618.14)


def test_drops_points_behind_or_too_close_to_camera():
    """z = -5 is behind the camera; z = 0.5 is under the 1 m minimum."""
    points = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [-5.0, 0.5],
    ])
    u, v, depths, ranges = project_to_image(points, SIMPLE_K, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert len(u) == 0


def test_drops_points_outside_image_and_keeps_the_rest():
    """(20, 0, 10) lands at u = 1000*20/10 + 800 = 2800, past the 1600 px edge."""
    points = np.array([
        [2.0, 20.0],
        [1.0, 0.0],
        [10.0, 10.0],
    ])
    u, v, depths, ranges = project_to_image(points, SIMPLE_K, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert len(u) == 1
    assert u[0] == pytest.approx(1000.0)


def test_transform_matches_hand_computation():
    """R turns (1, 2, 3) into (-2, 1, 3); adding t = (10, 0, 0) gives (8, 1, 3)."""
    points = np.array([[1.0], [2.0], [3.0]])
    moved = transform_points(points, ROTATE_90_ABOUT_Z, np.array([10.0, 0.0, 0.0]))

    np.testing.assert_allclose(moved[:, 0], [8.0, 1.0, 3.0])


def test_inverse_transform_undoes_transform():
    """R^T (p' - t) must give back the original points."""
    points = np.array([
        [1.0, -4.0],
        [2.0, 0.5],
        [3.0, 7.0],
    ])
    translation = np.array([10.0, -2.0, 1.5])

    moved = transform_points(points, ROTATE_90_ABOUT_Z, translation)
    restored = inverse_transform_points(moved, ROTATE_90_ABOUT_Z, translation)

    np.testing.assert_allclose(restored, points)


def test_median_ignores_background_points():
    """8 points on a car at 10 m, 2 background points at 40 m: median 10 m (mean would be 16 m)."""
    u = np.full(10, 50.0)
    v = np.full(10, 50.0)
    ranges = np.array([10.0] * 8 + [40.0] * 2)

    distance, num_points = box_distance(u, v, ranges, 0, 0, 100, 100)

    assert distance == pytest.approx(10.0)
    assert num_points == 10


def test_too_few_points_gives_unknown():
    """2 points is under the 3-point minimum, so distance is unknown (None)."""
    u = np.array([50.0, 50.0])
    v = np.array([50.0, 50.0])
    ranges = np.array([10.0, 10.0])

    distance, num_points = box_distance(u, v, ranges, 0, 0, 100, 100)

    assert distance is None
    assert num_points == 2


def test_keep_fraction_ignores_points_near_box_edge():
    """Box 0-100 at keep_fraction 0.75 keeps 12.5-87.5, so the 2 points at u = 5 are left out."""
    u = np.array([50.0, 50.0, 50.0, 5.0, 5.0])
    v = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
    ranges = np.array([10.0, 10.0, 10.0, 40.0, 40.0])

    distance, num_points = box_distance(u, v, ranges, 0, 0, 100, 100, keep_fraction=0.75)

    assert num_points == 3
    assert distance == pytest.approx(10.0)