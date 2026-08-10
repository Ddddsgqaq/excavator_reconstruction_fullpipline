"""pipeline.py — reconstruct a set of in-memory frames into a DEM + ElevationMsg.

Milestone 1: extract the offline "frames → DEM" flow into ONE reusable function the
streaming loop can call repeatedly, WITHOUT changing offline behavior.

Design choice — faithful reuse over reimplementation:
  `vggt.utils.load_fn.load_and_preprocess_images` only accepts file PATHS (it does
  `Image.open` internally). Rather than fork the preprocessing, we write the in-memory
  frames to a temporary `images/` dir and call the *existing* `vggt_service._run_inference`,
  then the *existing* gravity/DEM/export helpers. The streaming DEM therefore walks the
  same code as the elevation viewer (`/elevation_viewer_data`), so results match.

The heavy imports (torch, VGGT model, vggt_service) are deferred into the function body so
this module can be imported cheaply and so nothing here loads the model unless actually run.
Intended to run inside the vggt_service process to reuse its resident `_model`.

Coordinate conventions (from the offline pipeline):
  * VGGT world (un-aligned): X-right, Y-down, Z-forward.
  * After gravity alignment: Y = up (elevation); DEM rows = Z, cols = X (row-major).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

# Defaults mirror vggt_service.py's elevation-view DEM settings so a streaming DEM uses
# the same source grid shown by the elevation viewer.
DEFAULT_CONF_THRES = 50.0                 # percentile gate on confidence (0-100)
DEFAULT_PREDICTION_MODE = "Depthmap and Camera Branch"
DEFAULT_GROUND_PERCENTILE = 20.0          # lowest N% by Y used as ground fallback
DEFAULT_GRID_RESOLUTION = 128             # DEM grid NxN (matches Unity's 128 tile)
DEFAULT_SCALE_FACTOR = 1.0                # aligned unit -> metres
DEFAULT_HEIGHT_RESOLUTION = 0.01          # int16 quantisation step (m)


@dataclass
class DemResult:
    """Everything the streaming loop needs from one reconstruction pass.

    `elev` is the float DEM (rows=Z, cols=X); `has_data` marks interpolated cells.
    `R_align` / `scale_factor` / `gravity_source` are the world-frame anchor pieces the
    coordinate-stability milestone (M4) freezes and reuses across passes.

    `ground_xyz` holds the aligned+scaled ground points (in the anchor's registered
    frame, when an anchor was applied) so the loop can build/refresh the anchor and so
    M4.2 can register the next pass against them. `registered` / `registration_rmse` /
    `registration_yaw_deg` report the M4.2 cross-pass alignment that was applied.
    """
    elev: np.ndarray                      # (grid, grid) float64, NaN in holes
    has_data: np.ndarray                  # (grid, grid) bool
    x_bounds: tuple[float, float]         # aligned X extent (world units)
    z_bounds: tuple[float, float]         # aligned Z extent (world units)
    R_align: np.ndarray                   # (3, 3) VGGT-world -> gravity-aligned
    scale_factor: float
    gravity_source: str                   # "trajectory" | "ground_mask" | "cloud_ransac" | "anchor"
    n_frames: int
    n_points: int
    ground_xyz: np.ndarray | None = None  # (M, 3) aligned+scaled ground pts (registered frame)
    points_aligned: np.ndarray | None = None  # full confidence-filtered cloud, like elevation_viewer
    point_colors: np.ndarray | None = None    # RGB uint8 aligned with points_aligned
    registered: bool = False              # True if an M4.2 cross-pass transform was applied
    registration_rmse: float | None = None
    registration_yaw_deg: float | None = None
    warnings: list = field(default_factory=list)


@dataclass
class Anchor:
    """Frozen world-frame reference for cross-pass stability (M4).

    Passing a previous pass's Anchor lets later passes reuse the same gravity rotation,
    scale AND horizontal footprint so the Unity terrain doesn't jump/rescale/pan each
    update. In M1 this is unused (prev_anchor=None) — each pass estimates its own gravity
    and footprint, matching the offline path.

      * R_align / scale_factor  — frozen gravity rotation + metric scale (M4.1).
      * x_bounds / z_bounds     — frozen DEM footprint; every pass rasterises onto the
                                  SAME grid extent, so the tile no longer pans/zooms (M4.1).
      * ref_ground_xyz          — reference ground points the next pass registers against
                                  via register_horizontal (M4.2). None → registration off.
    """
    R_align: np.ndarray
    scale_factor: float
    x_bounds: tuple[float, float] | None = None
    z_bounds: tuple[float, float] | None = None
    ref_ground_xyz: np.ndarray | None = None

    @classmethod
    def from_result(cls, res: "DemResult", *, keep_reference: bool = True) -> "Anchor":
        """Freeze an anchor from the first successful pass.

        keep_reference=True stores that pass's ground points so later passes can register
        against them (M4.2). Set False to freeze only gravity/scale/footprint (M4.1 only).
        """
        return cls(
            R_align=np.asarray(res.R_align, dtype=np.float64).copy(),
            scale_factor=float(res.scale_factor),
            x_bounds=tuple(res.x_bounds),
            z_bounds=tuple(res.z_bounds),
            ref_ground_xyz=(res.ground_xyz.copy()
                            if keep_reference and res.ground_xyz is not None else None),
        )


def _write_frames_to_images_dir(frames, working_dir: str) -> int:
    """Dump in-memory frames as PNGs into `<working_dir>/images/` for _run_inference.

    frames: list of HxWx3 uint8 (or float 0-1) arrays, or a single (S,H,W,3) array.
    Returns the number of frames written. Uses zero-padded names so sorted() order
    matches capture order (the offline path relies on sorted glob order).
    """
    from PIL import Image  # local import; PIL is already a project dep

    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        frames = list(frames)
    if len(frames) == 0:
        raise ValueError("reconstruct_frames_to_dem got 0 frames")

    images_dir = os.path.join(working_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for i, fr in enumerate(frames):
        arr = np.asarray(fr)
        if arr.dtype != np.uint8:
            # accept float 0-1 or 0-255; normalise to uint8 without changing content
            if arr.max() <= 1.0 + 1e-6:
                arr = (arr * 255.0).clip(0, 255)
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        Image.fromarray(arr[..., :3]).save(os.path.join(images_dir, f"{i:06d}.png"))
    return len(frames)


def _build_grid_fixed_bounds(ground_pts: np.ndarray,
                             x_bounds: tuple[float, float],
                             z_bounds: tuple[float, float],
                             grid_resolution: int):
    """Interpolate Y onto a FIXED (X, Z) footprint (M4.1 frozen tile).

    Mirrors elevation_plane.build_elevation_view_grid's interpolation exactly, but uses the
    anchor's frozen bounds instead of recomputing extent from the current cloud — so
    every pass rasterises onto the same grid and the Unity tile stops panning/zooming.
    The offline helper is left untouched; this is the streaming-only fixed-footprint twin.
    """
    from scipy.interpolate import griddata

    (x_min, x_max), (z_min, z_max) = x_bounds, z_bounds
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


def reconstruct_frames_to_dem(
    frames,
    *,
    prev_anchor: "Anchor | None" = None,
    register: bool = True,
    conf_thres: float = DEFAULT_CONF_THRES,
    prediction_mode: str = DEFAULT_PREDICTION_MODE,
    ground_percentile: float = DEFAULT_GROUND_PERCENTILE,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    fixed_footprint: bool = True,
    work_root: str | None = None,
    glb_out: str | None = None,
) -> DemResult:
    """Reconstruct in-memory frames into a DEM by reusing the offline pipeline.

    Mirrors the /elevation_viewer_data DEM path:
      _run_inference → _extract_points_with_conf → estimate_gravity →
      apply_alignment_to_points → _select_ground_aligned → build_elevation_view_grid.

    prev_anchor: if given (M4), reuse its R_align + scale (M4.1) instead of re-estimating
                 gravity, rasterise onto its frozen footprint (M4.1), and — when it carries
                 reference ground points and register=True — horizontally register this
                 pass onto it (M4.2). In M1 this is None, so each pass is self-contained
                 like the offline path.
    register:    enable the M4.2 cross-pass horizontal (yaw + XZ) alignment. Only has an
                 effect when prev_anchor.ref_ground_xyz is present.
    fixed_footprint: reuse the first-pass XZ bounds. Disable in fusion mode so the per-pass
                 Elevation Viewer mesh follows the complete current aligned point cloud;
                 the persistent GlobalDem supplies cross-pass spatial stability instead.
    work_root:   parent dir for the temporary images/ workspace. Defaults to a system temp
                 dir; the caller may point it at a workspaces/ subdir to keep artifacts.
    glb_out:     if given, export this pass's raw VGGT point cloud to that .glb path (for
                 offline supervision of the live reconstruction). Purely diagnostic — it
                 never affects the DEM. Export failures are recorded in warnings, not raised.

    Returns a DemResult. Raises ValueError if no valid points/cells survive.
    """
    # Deferred heavy imports — reuse the offline modules verbatim.
    import vggt_service as vs
    from gravity_alignment import (
        estimate_gravity,
        apply_alignment_to_points,
    )
    from elevation_plane import (
        _extract_points_with_conf,
        _select_ground_aligned,
        build_elevation_view_grid,
    )

    warnings: list = []

    # 1. Frames -> temp images/ dir -> existing _run_inference (same as offline).
    tmp_ctx = tempfile.TemporaryDirectory(prefix="stream_recon_", dir=work_root)
    with tmp_ctx as working_dir:
        n_frames = _write_frames_to_images_dir(frames, working_dir)
        predictions = vs._run_inference(working_dir)

        # 1b. Optional diagnostic GLB of the raw pass (before the temp dir is cleaned up).
        #     Reuses the offline exporter so the cloud matches the /reconstruct viewer.
        if glb_out:
            try:
                from visual_util import predictions_to_glb

                os.makedirs(os.path.dirname(glb_out) or ".", exist_ok=True)
                scene = predictions_to_glb(
                    predictions,
                    conf_thres=conf_thres,
                    prediction_mode=prediction_mode,
                    target_dir=working_dir,
                )
                scene.export(file_obj=glb_out)
            except Exception as exc:  # diagnostic only — never fail the pass
                warnings.append(f"glb export failed: {exc}")

        # 2. Un-aligned points + confidence + optional ground mask (offline helper).
        pts_world, conf_world, ground_world, keep_flat = _extract_points_with_conf(
            predictions, conf_thres, prediction_mode, return_keep_mask=True
        )
        if pts_world.shape[0] == 0:
            raise ValueError("No points survived confidence gate.")

        # RGB colors use the same kept pixels as the original elevation viewer.
        images = predictions.get("images")
        if images is not None:
            images = np.asarray(images)
            if images.ndim == 4 and images.shape[1] == 3:
                images = np.transpose(images, (0, 2, 3, 1))
            colors_flat = images.reshape(-1, 3)
            if np.issubdtype(colors_flat.dtype, np.floating) and colors_flat.max() <= 1.0 + 1e-6:
                colors_flat = colors_flat * 255.0
            point_colors = np.clip(colors_flat, 0, 255).astype(np.uint8)[keep_flat]
        else:
            point_colors = np.full((pts_world.shape[0], 3), 180, dtype=np.uint8)

        # 3. Gravity: reuse frozen anchor (M4) or estimate per-pass (M1, like offline).
        if prev_anchor is not None:
            R_align = np.asarray(prev_anchor.R_align, dtype=np.float64)
            scale_factor = float(prev_anchor.scale_factor)
            gravity_source = "anchor"
        else:
            grav = estimate_gravity(
                predictions["extrinsic"],
                predictions.get("world_points_from_depth", pts_world),
                ground_mask=None,
                conf=predictions.get("depth_conf"),
                conf_thres=conf_thres / 100.0,   # offline passes the /100 fraction
            )
            R_align = grav.R_align
            gravity_source = grav.source
            warnings.extend(getattr(grav, "warnings", []) or [])

        # 4. Align + scale to absolute units (offline: pts @ R_align.T * scale_factor).
        pts_aligned = apply_alignment_to_points(pts_world, R_align) * float(scale_factor)

        # 5. Ground candidates (offline helper, identical math).
        ground_pts = _select_ground_aligned(pts_aligned, ground_world, ground_percentile)
        if ground_pts.shape[0] < 3:
            raise ValueError("Too few ground points to interpolate a DEM.")

        # 5b. M4.2 — horizontally register this pass onto the anchor's reference ground.
        #     Gravity/scale are already frozen (M4.1), so only yaw + XZ translation remain.
        registered = False
        reg_rmse = None
        reg_yaw = None
        if (prev_anchor is not None and register
                and getattr(prev_anchor, "ref_ground_xyz", None) is not None):
            from .registration import register_horizontal
            tf = register_horizontal(ground_pts, prev_anchor.ref_ground_xyz)
            if tf.converged:
                ground_pts = tf.apply_to_points(ground_pts)
                pts_aligned = tf.apply_to_points(pts_aligned)
                registered = True
                reg_rmse = float(tf.rmse)
                reg_yaw = float(tf.yaw_deg)
            else:
                warnings.append("cross-pass registration did not converge; using anchor frame only")

        # 6. DEM grid. Match elevation_viewer with Use ground filter for DEM unchecked:
        #    rasterise every confidence-kept aligned point. ground_pts remains reserved for
        #    gravity and cross-pass registration, so display fidelity does not weaken M4.
        if (fixed_footprint and prev_anchor is not None
                and getattr(prev_anchor, "x_bounds", None) is not None):
            _xx, _zz, elev, valid, x_bounds, z_bounds = _build_grid_fixed_bounds(
                pts_aligned, prev_anchor.x_bounds, prev_anchor.z_bounds, grid_resolution
            )
        else:
            _xx, _zz, elev, valid, x_bounds, z_bounds = build_elevation_view_grid(
                pts_aligned, pts_aligned, grid_resolution
            )

    if not np.isfinite(elev).any():
        raise ValueError("DEM grid has no finite cells.")

    return DemResult(
        elev=elev,
        has_data=valid,
        x_bounds=(float(x_bounds[0]), float(x_bounds[1])),
        z_bounds=(float(z_bounds[0]), float(z_bounds[1])),
        R_align=np.asarray(R_align, dtype=np.float64),
        scale_factor=float(scale_factor),
        gravity_source=gravity_source,
        n_frames=n_frames,
        n_points=int(pts_aligned.shape[0]),
        ground_xyz=ground_pts,
        points_aligned=pts_aligned,
        point_colors=point_colors,
        registered=registered,
        registration_rmse=reg_rmse,
        registration_yaw_deg=reg_yaw,
        warnings=warnings,
    )


def dem_result_to_msg(
    res: DemResult,
    *,
    height_resolution: float = DEFAULT_HEIGHT_RESOLUTION,
    tile_x: int = 0,
    tile_y: int = 0,
    tile_size_meters: float | None = None,
    timestamp: float | None = None,
) -> dict:
    """Adapt a DemResult into a Unity ElevationMsg via the offline exporter (unchanged)."""
    from elevation_export import dem_to_elevation_msg
    from elevation_plane import fill_elevation_view_holes

    filled_elev, filled_valid = fill_elevation_view_holes(res.elev, res.has_data)
    msg = dem_to_elevation_msg(
        filled_elev,
        res.x_bounds,
        res.z_bounds,
        has_data=filled_valid,
        # res.elev/x_bounds/z_bounds already come from pts_aligned after metric scaling.
        # Applying res.scale_factor again here would distort both relief and footprint.
        scale_factor=1.0,
        height_resolution=height_resolution,
        tile_x=tile_x,
        tile_y=tile_y,
        tile_size_meters=tile_size_meters,
        timestamp=timestamp,
    )
    msg["metadata"]["dem_preprocessing"] = "elevation_viewer_fill_20"
    msg["metadata"]["source_nodata_count"] = int((~np.asarray(res.has_data, dtype=bool)).sum())
    msg["source_valid"] = np.asarray(res.has_data, dtype=np.uint8).reshape(-1).tolist()
    return msg
