"""
elevation_plane.py — Elevation plane fitting and GLB generation.

Pipeline:
  1. Estimate gravity direction (gravity_alignment.estimate_gravity)
       — primary: trajectory plane on camera centers
       — fallback: YOLOe ground-mask RANSAC
       — last resort: whole-cloud RANSAC
  2. Rotate the entire scene so gravity is +Y.
  3. Extract ground-candidate points in the aligned frame.
  4. Interpolate a regular elevation grid (DEM) over (X, Z).
  5. Build a colored trimesh and export as GLB.

Public API:
    fit_elevation_to_glb(predictions, working_dir, ...) -> dict
"""

import os
import json
import numpy as np
import trimesh

from scipy.interpolate import griddata
from matplotlib import colormaps

from gravity_alignment import (
    estimate_gravity,
    apply_alignment_to_points,
)


# ── Point extraction ─────────────────────────────────────────────────────────

def _extract_points_with_conf(predictions: dict,
                               conf_thres: float,
                               prediction_mode: str):
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

    finite = np.isfinite(pts_flat).all(axis=1)
    keep = finite & (conf_flat >= (conf_thres / 100.0) * conf_flat.max())
    return pts_flat[keep], conf_flat[keep], (ground_flat[keep] if ground_flat is not None else None)


# ── Ground selection in the aligned frame ────────────────────────────────────

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
    y = points_aligned[:, 1]
    y_range = float(np.percentile(y, 98) - np.percentile(y, 2)) or 1.0

    if ground_mask is not None and ground_mask.any():
        gpts = points_aligned[ground_mask]
        if gpts.shape[0] >= 50:
            y_med = float(np.median(gpts[:, 1]))
            tol = band * y_range
            tight = gpts[np.abs(gpts[:, 1] - y_med) <= tol]
            if tight.shape[0] >= 50:
                return tight
            return gpts

    thresh = np.percentile(y, ground_percentile)
    return points_aligned[y <= thresh]


# ── DEM grid interpolation ───────────────────────────────────────────────────

def _build_elevation_grid(ground_pts: np.ndarray,
                           all_pts: np.ndarray,
                           grid_resolution: int):
    """Interpolate Y over the (X, Z) plane in the aligned frame."""
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()
    x_pad = (x_max - x_min) * 0.05
    z_pad = (z_max - z_min) * 0.05
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
    valid = ~np.isnan(elev_linear)
    return xx, zz, elev, valid, (x_min, x_max), (z_min, z_max)


# ── Mesh construction ────────────────────────────────────────────────────────

def _elevation_to_mesh(xx, zz, elev,
                        colormap_name: str = "terrain",
                        elev_min=None, elev_max=None) -> trimesh.Trimesh:
    R, C = xx.shape
    verts = np.column_stack([xx.ravel(), elev.ravel(), zz.ravel()]).astype(np.float32)

    rs, cs = np.meshgrid(np.arange(R - 1), np.arange(C - 1), indexing="ij")
    i00 = (rs * C + cs).ravel()
    i10 = ((rs + 1) * C + cs).ravel()
    i01 = (rs * C + (cs + 1)).ravel()
    i11 = ((rs + 1) * C + (cs + 1)).ravel()
    faces = np.empty((i00.size * 2, 3), dtype=np.int32)
    faces[0::2] = np.stack([i00, i10, i01], axis=1)
    faces[1::2] = np.stack([i10, i11, i01], axis=1)

    y_vals = elev.ravel()
    lo = elev_min if elev_min is not None else float(np.nanpercentile(y_vals, 2))
    hi = elev_max if elev_max is not None else float(np.nanpercentile(y_vals, 98))
    norm = np.clip((y_vals - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    cmap = colormaps.get_cmap(colormap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)

    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, process=False)


# ── GLB helpers ──────────────────────────────────────────────────────────────

def _scene_with(mesh: trimesh.Trimesh) -> trimesh.Scene:
    s = trimesh.Scene()
    s.add_geometry(mesh, node_name="elevation_plane")
    return s


def _merge_with_existing(source_glb_path: str, mesh: trimesh.Trimesh,
                          R_align: np.ndarray, scale: float) -> trimesh.Scene:
    """
    Load an existing GLB (raw VGGT cloud, viewer-display frame) and overlay the
    aligned elevation mesh. The source GLB has already had VGGT's display
    alignment baked in; we apply our gravity rotation on top so the mesh sits
    on the cloud correctly.
    """
    base = trimesh.load(source_glb_path, force="scene")
    merged = base.copy()
    overlay = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R_align * scale
    overlay.apply_transform(T)
    merged.add_geometry(overlay, node_name="elevation_plane")
    return merged


# ── Public API ───────────────────────────────────────────────────────────────

def fit_elevation_to_glb(
    predictions: dict,
    working_dir: str,
    source_glb_path: str = "",
    grid_resolution: int = 128,
    colormap: str = "terrain",
    ground_percentile: float = 20.0,
    use_ransac: bool = True,                  # kept for API compatibility, unused
    conf_thres: float = 50.0,
    prediction_mode: str = "Depthmap and Camera Branch",
    scale_factor: float = 1.0,                # multiplier for absolute scale calibration
) -> dict:
    """
    Fit a gravity-aligned elevation plane and export GLBs.

    Returns a dict with:
        elev_only_path, merged_path, gravity_source, n_grav, R_align,
        scale_factor, warnings, log
    """
    del use_ransac  # superseded by gravity-aware estimator

    # 1. Pull points + per-point ground membership (un-aligned VGGT world).
    pts_world, conf_world, ground_world = _extract_points_with_conf(
        predictions, conf_thres, prediction_mode
    )
    if pts_world.shape[0] < 100:
        raise ValueError(f"Too few valid points ({pts_world.shape[0]}).")

    # 2. Estimate gravity & rotation. Pass full 4-D arrays so the estimator
    #    can do its own confidence filtering and shape-aware logic.
    extrinsic = predictions["extrinsic"]
    raw_pts = predictions.get("world_points_from_depth")
    raw_conf = predictions.get("depth_conf")
    sem = predictions.get("semantic_masks")
    gmask_3d = (np.asarray(sem) == 1) if sem is not None else None

    grav = estimate_gravity(
        extrinsic=extrinsic,
        world_points=raw_pts,
        ground_mask=gmask_3d,
        conf=raw_conf,
        conf_thres=conf_thres / 100.0,
    )

    # 3. Rotate points into the aligned frame and apply scale.
    pts_aligned = apply_alignment_to_points(pts_world, grav.R_align) * scale_factor

    # 4. Pick ground points in the aligned frame.
    ground_pts = _select_ground_aligned(pts_aligned, ground_world, ground_percentile)
    if ground_pts.shape[0] < 50:
        raise ValueError(f"Too few ground candidates after alignment ({ground_pts.shape[0]}).")

    # 5. DEM grid + mesh.
    xx, zz, elev, _valid, _, _ = _build_elevation_grid(ground_pts, pts_aligned, grid_resolution)
    elev_mesh = _elevation_to_mesh(xx, zz, elev, colormap)

    # 6. Export.
    tag = f"elev_r{grid_resolution}_{colormap}_aligned"
    elev_only_path = os.path.join(working_dir, f"{tag}_only.glb")
    _scene_with(elev_mesh).export(file_obj=elev_only_path)

    merged_path = os.path.join(working_dir, f"{tag}_merged.glb")
    if source_glb_path and os.path.exists(source_glb_path):
        _merge_with_existing(source_glb_path, elev_mesh, grav.R_align, scale_factor).export(file_obj=merged_path)
    else:
        _scene_with(elev_mesh).export(file_obj=merged_path)

    # 7. Persist alignment metadata so callers can reuse the same frame.
    meta = {
        "gravity_source": grav.source,
        "gravity_inliers": grav.inlier_count,
        "n_grav": grav.n_grav.tolist(),
        "R_align": grav.R_align.tolist(),
        "scale_factor": float(scale_factor),
        "warnings": grav.warnings,
        "debug": grav.debug,
    }
    with open(os.path.join(working_dir, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log_lines = [
        f"Gravity: {grav.source} (inliers={grav.inlier_count})",
        f"DEM: {grid_resolution}x{grid_resolution}, colormap={colormap}, scale={scale_factor}",
    ]
    log_lines.extend(grav.warnings)

    return {
        "elev_only_path": elev_only_path,
        "merged_path": merged_path,
        "gravity_source": grav.source,
        "n_grav": grav.n_grav.tolist(),
        "R_align": grav.R_align.tolist(),
        "scale_factor": float(scale_factor),
        "warnings": grav.warnings,
        "log": " | ".join(log_lines),
    }
