"""
elevation_plane.py — DEM helpers for the elevation viewer and streaming path.

These helpers operate in the gravity-aligned frame (Y = elevation):
  * _extract_points_with_conf   — pull points + ground mask from predictions
  * _select_ground_aligned(_mask) — pick ground candidates after alignment
  * build_elevation_view_grid   — interpolate the DEM used by /elevation_viewer_data

Consumers: vggt_service.py (/elevation_viewer_data) and streaming/pipeline.py.
"""

import numpy as np

from scipy.interpolate import griddata


# ── Point extraction ─────────────────────────────────────────────────────────

def _extract_points_with_conf(predictions: dict,
                               conf_thres: float,
                               prediction_mode: str,
                               return_keep_mask: bool = False):
    """
    Returns (points (N,3), confidence (N,), per-pixel ground mask (N,) or None)
    in the original (un-aligned) VGGT world frame.
    """
    if prediction_mode == "Pointmap Branch":
        pts = predictions.get("world_points")
        conf = predictions.get("world_points_conf")
    else:
        pts = predictions.get("world_points_from_depth")
        conf = predictions.get("depth_conf")

    if pts is None:
        raise ValueError("No world points found in predictions.")

    pts = np.asarray(pts)
    if pts.ndim != 4:
        raise ValueError(f"Expected (S, H, W, 3) point map, got shape {pts.shape}")
    S, H, W, _ = pts.shape
    pts_flat = pts.reshape(-1, 3)

    if conf is None:
        conf_flat = np.ones(pts_flat.shape[0], dtype=np.float32)
    else:
        conf_flat = np.asarray(conf).reshape(-1).astype(np.float32)

    sem = predictions.get("semantic_masks")
    ground_flat = None
    if sem is not None:
        sem = np.asarray(sem)
        if sem.shape == (S, H, W):
            ground_flat = (sem == 1).reshape(-1)

    # Match vggt_service.elevation_viewer_data exactly: conf_thres is a percentile,
    # not a fraction of the maximum confidence.
    finite = np.isfinite(pts_flat).all(axis=1)
    conf_thres_val = np.percentile(conf_flat, conf_thres) if conf_thres > 0 else 0.0
    keep = finite & np.isfinite(conf_flat) & (conf_flat >= conf_thres_val)
    result = (
        pts_flat[keep], conf_flat[keep],
        ground_flat[keep] if ground_flat is not None else None,
    )
    return (*result, keep) if return_keep_mask else result


# ── Ground selection in the aligned frame ────────────────────────────────────

def _select_ground_aligned_mask(points_aligned: np.ndarray,
                                 ground_mask: np.ndarray | None,
                                 ground_percentile: float,
                                 band: float = 0.05) -> np.ndarray:
    """
    Boolean mask version of `_select_ground_aligned`.

    This keeps the selection logic in one place so callers that need to
    visualize before/after filtering can identify which source points survived.
    """
    y = points_aligned[:, 1]
    y_range = float(np.percentile(y, 98) - np.percentile(y, 2)) or 1.0

    if ground_mask is not None and ground_mask.any():
        ground_mask = np.asarray(ground_mask, dtype=bool)
        ground_idx = np.flatnonzero(ground_mask)
        gpts = points_aligned[ground_idx]
        if gpts.shape[0] >= 50:
            y_med = float(np.median(gpts[:, 1]))
            tol = band * y_range
            tight_local = np.abs(gpts[:, 1] - y_med) <= tol
            out = np.zeros(points_aligned.shape[0], dtype=bool)
            if int(tight_local.sum()) >= 50:
                out[ground_idx[tight_local]] = True
                return out
            out[ground_idx] = True
            return out

    thresh = np.percentile(y, ground_percentile)
    return y <= thresh


def _select_ground_aligned(points_aligned: np.ndarray,
                            ground_mask: np.ndarray | None,
                            ground_percentile: float,
                            band: float = 0.05) -> np.ndarray:
    """
    Pick ground points after gravity alignment. Y is now elevation.

    Strategy:
      * If a per-point ground mask is available, keep masked points whose Y
        is within ±`band`·(Y range) of the masked-points Y-mode (median).
      * Otherwise, take the lowest `ground_percentile`% by Y.
    """
    keep = _select_ground_aligned_mask(
        points_aligned, ground_mask, ground_percentile, band
    )
    return points_aligned[keep]


# ── DEM grid interpolation ───────────────────────────────────────────────────

def build_elevation_view_grid(ground_pts: np.ndarray,
                              all_pts: np.ndarray,
                              grid_resolution: int):
    """Build the DEM used by ``/elevation_viewer_data``.

    The elevation viewer deliberately uses a 2% horizontal padding, rather than
    the 5% padding used by the legacy GLB-export DEM above. Keep this as a named
    helper so consumers that promise the elevation-view DEM (including M4) share
    the same bounds, interpolation, and ``has_data`` rules.
    """
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()
    x_pad = (x_max - x_min) * 0.02
    z_pad = (z_max - z_min) * 0.02
    x_min -= x_pad; x_max += x_pad
    z_min -= z_pad; z_max += z_pad

    xi = np.linspace(x_min, x_max, grid_resolution)
    zi = np.linspace(z_min, z_max, grid_resolution)
    xx, zz = np.meshgrid(xi, zi)

    src_xz = ground_pts[:, [0, 2]]
    src_y = ground_pts[:, 1]
    elev_linear = griddata(src_xz, src_y, (xx, zz), method="linear")
    elev_nearest = griddata(src_xz, src_y, (xx, zz), method="nearest")
    elev = np.where(np.isnan(elev_linear), elev_nearest, elev_linear)
    has_data = ~np.isnan(elev_linear)  # nearest-filled cells are viewer NODATA
    return xx, zz, elev, has_data, (x_min, x_max), (z_min, z_max)


def fill_elevation_view_holes(elev: np.ndarray,
                              has_data: np.ndarray,
                              max_passes: int = 20):
    """Fill a DEM exactly like elevation_viewer buildDEMMesh.

    The browser implementation scans row-major and updates cells in place using the
    mean of currently valid 8-neighbours.  Keeping the same scan order matters: newly
    filled cells may feed later cells in the same pass.  The returned validity mask
    marks every finite vertex because the filled viewer meshes every finite cell.
    """
    filled = np.asarray(elev, dtype=np.float64).copy()
    filled_mask = np.asarray(has_data, dtype=bool).copy() & np.isfinite(filled)
    if filled.ndim != 2 or filled_mask.shape != filled.shape:
        raise ValueError(
            f"elev and has_data must be matching 2D arrays, got "
            f"{filled.shape} and {filled_mask.shape}"
        )
    rows, cols = filled.shape
    for _pass in range(max(0, int(max_passes))):
        changed = False
        for i in range(rows):
            for j in range(cols):
                if filled_mask[i, j]:
                    continue
                total = 0.0
                count = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if (0 <= ni < rows and 0 <= nj < cols
                                and filled_mask[ni, nj]
                                and np.isfinite(filled[ni, nj])):
                            total += float(filled[ni, nj])
                            count += 1
                if count:
                    filled[i, j] = total / count
                    filled_mask[i, j] = True
                    changed = True
        if not changed:
            break
    # elevation_viewer receives nearest-filled values even where has_data is false.
    # Fusion tiles can contain actual NaNs, so supply the same nearest fallback for any
    # cells not reached within 20 passes; they become real mesh vertices in both outputs.
    remaining = ~np.isfinite(filled)
    finite = np.isfinite(filled)
    if remaining.any() and finite.any():
        from scipy.ndimage import distance_transform_edt
        nearest = distance_transform_edt(~finite, return_distances=False, return_indices=True)
        filled[remaining] = filled[tuple(nearest[:, remaining])]
    return filled, np.isfinite(filled)
