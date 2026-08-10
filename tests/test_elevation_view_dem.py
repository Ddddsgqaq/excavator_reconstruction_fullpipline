"""Regression coverage for the DEM contract shared by elevation-view and M4."""

import numpy as np

from elevation_plane import build_elevation_view_grid
from streaming.pipeline import _build_grid_fixed_bounds


def test_elevation_view_grid_uses_two_percent_padding_and_marks_holes():
    all_pts = np.array([
        [0.0, 0.0, 0.0], [10.0, 2.0, 0.0], [0.0, 1.0, 10.0], [10.0, 3.0, 10.0],
    ])
    ground_pts = all_pts[:3]

    _xx, _zz, elev, has_data, x_bounds, z_bounds = build_elevation_view_grid(
        ground_pts, all_pts, grid_resolution=9
    )

    assert x_bounds == (-0.2, 10.2)
    assert z_bounds == (-0.2, 10.2)
    assert elev.shape == (9, 9)
    assert np.isfinite(elev).all()
    assert not has_data.all()


def test_m4_fixed_footprint_preserves_elevation_view_bounds():
    points = np.array([
        [0.0, 0.0, 0.0], [10.0, 2.0, 0.0], [0.0, 1.0, 10.0], [10.0, 3.0, 10.0],
    ])
    _xx, _zz, _elev, _valid, x_bounds, z_bounds = build_elevation_view_grid(
        points, points, grid_resolution=9
    )
    _xx, _zz, elev, _valid, fixed_x_bounds, fixed_z_bounds = _build_grid_fixed_bounds(
        points, x_bounds, z_bounds, grid_resolution=9
    )

    assert fixed_x_bounds == x_bounds
    assert fixed_z_bounds == z_bounds
    assert np.isfinite(elev).all()
