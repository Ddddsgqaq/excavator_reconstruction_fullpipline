"""
elevation_export.py — Export a DEM elevation grid to the JSON format consumed
by the Unity terrain loader (see TERRAIN_ELEVATION_FORMAT.md).

The Unity side (`ElevationMsg` / `HandleElevationMap`) expects a single tile:
an int16, row-major (`data[y*width + x]`) array where the real height in metres
is `data[i] * height_resolution`, plus a small metadata block. NODATA is the
int16 minimum (-32768); Unity re-scans the array for min/max and normalises, so
the absolute offset of the integers does not affect the final terrain shape.

Two DEM producers feed this:
  * elevation_plane / elevation_viewer_data  -> the interpolated `elev` grid
  * terrain_analysis.rasterize_bev            -> the `H_top` surface grid

Both hand us a float (H, W) grid in gravity-aligned world units (Y = up) plus
the (X, Z) horizontal bounds. This module turns that into the JSON dict/file.
"""

from __future__ import annotations

import json
import numpy as np

INT16_NODATA = -32768
INT16_MIN = -32768
INT16_MAX = 32767


def dem_to_elevation_msg(
    elev: np.ndarray,
    x_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    *,
    has_data: np.ndarray | None = None,
    scale_factor: float = 1.0,
    horizontal_scale: float | None = None,
    vertical_scale: float | None = None,
    height_resolution: float = 0.01,
    tile_x: int = 0,
    tile_y: int = 0,
    tile_size_meters: float | None = None,
    timestamp: float | None = None,
    zone_map: np.ndarray | None = None,
    legend: list | None = None,
) -> dict:
    """Convert a float elevation grid (aligned world units) into an ElevationMsg dict.

    Args:
        elev: (H, W) float grid. Row index = Z (`data_order` row-major, data[y*W+x]),
              column index = X. May contain NaN for holes.
        x_bounds, z_bounds: (min, max) horizontal extent of the grid in world units.
        has_data: optional (H, W) validity mask (truthy = valid). Cells that are
              falsy here, or NaN in `elev`, become NODATA.
        scale_factor: backwards-compatible isotropic metres-per-unit scale.
        horizontal_scale, vertical_scale: optional anisotropic overrides; when set,
            X/Z use horizontal_scale and elevation uses vertical_scale.
        height_resolution: int16 quantisation step in metres. Real height (m) =
              data[i] * height_resolution.
        tile_x, tile_y: tile indices for multi-tile scenes.
        tile_size_meters: physical tile edge length (m). Defaults to the X span
              times scale_factor (matches Unity's square terrain).
        timestamp: Unix seconds, recorded only (Unity does not consume it).
        zone_map: optional (H, W) int grid of work-zone codes co-registered with
              `elev` (same shape, same row-major order). When provided, a
              `semantic` block is appended to the message. Empty cells use
              ZONE_EMPTY (-1). See terrain_analysis.zone_legend().
        legend: optional list of {code, name, diggable, color} dicts describing
              the zone codes in `zone_map` (from terrain_analysis.zone_legend()).

    Returns:
        A dict matching the ElevationMsg JSON schema.
    """
    elev = np.asarray(elev, dtype=np.float64)
    if elev.ndim != 2:
        raise ValueError(f"elev must be 2D (H, W), got shape {elev.shape}")
    height, width = elev.shape  # rows = Z, cols = X

    # Valid cells: finite AND (if provided) flagged by has_data.
    valid = np.isfinite(elev)
    if has_data is not None:
        has_data = np.asarray(has_data)
        if has_data.shape != elev.shape:
            raise ValueError(
                f"has_data shape {has_data.shape} != elev shape {elev.shape}")
        valid &= has_data.astype(bool)
    if not valid.any():
        raise ValueError("No valid cells in the elevation grid.")

    # Convert elevation and map extent independently when local-plane calibration
    # found different horizontal and vertical VGGT scales.
    horizontal_m_per_unit = float(horizontal_scale) if horizontal_scale is not None else float(scale_factor)
    vertical_m_per_unit = float(vertical_scale) if vertical_scale is not None else float(scale_factor)
    if horizontal_m_per_unit <= 0 or vertical_m_per_unit <= 0:
        raise ValueError("calibration scales must be positive")
    # Heights in absolute metres.
    elev_m = elev * vertical_m_per_unit
    valid_m = elev_m[valid]
    emin = float(valid_m.min())
    emax = float(valid_m.max())

    # Quantise to int16, offset so the minimum maps to 0 (keeps values positive
    # and small). Invalid cells become NODATA.
    raw = np.full(elev.shape, INT16_NODATA, dtype=np.int64)
    q = np.rint((elev_m[valid] - emin) / float(height_resolution))
    raw[valid] = q.astype(np.int64)

    # Guard the int16 range; clip and flag if the requested step overflows.
    overflow = int(np.count_nonzero((raw > INT16_MAX) | (raw < INT16_MIN)))
    if overflow:
        clip_lo = valid & (raw < INT16_MIN)
        raw[clip_lo] = INT16_MIN
        clip_hi = valid & (raw > INT16_MAX)
        raw[clip_hi] = INT16_MAX
    raw = raw.astype(np.int16)

    # Horizontal metadata (X span drives the square terrain size by default).
    x_span = (float(x_bounds[1]) - float(x_bounds[0])) * horizontal_m_per_unit
    z_span = (float(z_bounds[1]) - float(z_bounds[0])) * horizontal_m_per_unit
    if tile_size_meters is None:
        tile_size_meters = x_span
    resolution = tile_size_meters / width if width else 0.0

    metadata = {
        "width": int(width),
        "height": int(height),
        "resolution": float(resolution),
        "height_resolution": float(height_resolution),
        "horizontal_m_per_vggt_unit": horizontal_m_per_unit,
        "vertical_m_per_vggt_unit": vertical_m_per_unit,
        "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
        "coordinate_system": "local_enu",
        "min_elevation": emin,
        "max_elevation": emax,
        "tile_x": int(tile_x),
        "tile_y": int(tile_y),
        "tile_size_meters": float(tile_size_meters),
        # Extra diagnostics (ignored by Unity, useful for humans):
        "x_span_meters": x_span,
        "z_span_meters": z_span,
        "nodata_value": INT16_NODATA,
        "nodata_count": int((~valid).sum()),
        "overflow_clipped": overflow,
    }

    msg = {
        "metadata": metadata,
        "data_type": "int16",
        "data": raw.reshape(-1).tolist(),  # row-major: data[y*width + x]
        "data_order": "row_major",
    }
    if timestamp is not None:
        msg["timestamp"] = float(timestamp)

    # Optional semantic layer, co-registered with the elevation grid.
    if zone_map is not None:
        zone_map = np.asarray(zone_map)
        if zone_map.shape != elev.shape:
            raise ValueError(
                f"zone_map shape {zone_map.shape} != elev shape {elev.shape}")
        msg["semantic"] = {
            "layer_type": "zone",
            "width": int(width),
            "height": int(height),
            # row-major, data[y*width + x]; empty cells = ZONE_EMPTY (-1)
            "data": zone_map.astype(np.int32).reshape(-1).tolist(),
            "legend": legend if legend is not None else [],
        }
    return msg


def write_elevation_json(path: str, msg: dict) -> str:
    """Serialise an ElevationMsg dict to `path`. Returns the path."""
    with open(path, "w") as f:
        json.dump(msg, f)
    return path
