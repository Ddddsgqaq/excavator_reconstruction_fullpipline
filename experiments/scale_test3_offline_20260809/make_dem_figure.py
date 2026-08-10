"""Gridded DEM elevation figure for scale_test3.

Unlike ``reconstruction_3d_elevation.png`` (a per-point scatter coloured by
height), this reproduces the *gridded* DEM that the main service builds in
``/elevation_viewer_data``: the same point source, the same gravity alignment
call, the same P50 confidence gate, the same semantic ground filter, and the
same ``build_elevation_view_grid`` helper (GRID_RES 128, 2% padding, linear +
nearest griddata).  Heights are then put in metres with the 15 cm ruler scale
and referenced to the tabletop, so the surface is a metric DEM you can read a
grid off, not just a coloured cloud.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gravity_alignment as ga
import terrain_analysis as ta
from elevation_plane import build_elevation_view_grid, _select_ground_aligned_mask

WS = Path(__file__).resolve().parent
VIZ = WS / "visualizations"
VIZ.mkdir(exist_ok=True)

# Main-program defaults from ElevationViewerRequest / elevation_viewer_data.
CONF_PERCENTILE = 50.0          # req.conf_thres
GROUND_PERCENTILE = 20.0        # req.ground_percentile
GRID_RES = 128                  # GRID_RES in elevation_viewer_data
OBJECT_IDS = (3, 4)             # box + bottle: excavator-equivalent objects to keep off the terrain


def build_metric_dem():
    z = np.load(WS / "predictions.npz")
    pred = {k: np.asarray(z[k]) for k in z.files}
    results = json.loads((WS / "dimension_results.json").read_text(encoding="utf-8"))
    scale = float(results["scale_calibration"]["scale_m_per_vggt_unit"])

    points = pred["world_points_from_depth"]        # (S,H,W,3), same source as the service
    conf = pred["depth_conf"]
    masks = pred["semantic_masks"]

    # 1) Gravity alignment — identical call to elevation_viewer_data.
    grav = ga.estimate_gravity(
        pred["extrinsic"], points, masks == 1, conf, conf_thres=CONF_PERCENTILE / 100.0
    )
    R_align = grav.R_align

    # 2) P50 confidence keep on finite points (service's `keep`).
    pts_flat = points.reshape(-1, 3)
    conf_flat = conf.reshape(-1)
    sem_flat = masks.reshape(-1)
    finite = np.isfinite(pts_flat).all(axis=1) & np.isfinite(conf_flat)
    thr = np.percentile(conf_flat[np.isfinite(conf_flat)], CONF_PERCENTILE)
    keep = finite & (conf_flat >= thr)

    pts_kept = pts_flat[keep]
    sem_kept = sem_flat[keep]
    pts_aligned = pts_kept @ R_align.T             # Y = up

    # 3) Semantic ground filter, then drop object points off the terrain
    #    (mirrors dem_source_mask & ~object_mask).
    ground_kept = (sem_kept == 1)
    ground_mask = _select_ground_aligned_mask(pts_aligned, ground_kept, GROUND_PERCENTILE)
    object_mask = np.isin(sem_kept, OBJECT_IDS)
    dem_source_mask = ground_mask & ~object_mask
    ground_pts = pts_aligned[dem_source_mask]

    # 4) Exact terrain DEM grid used by /elevation_viewer_data (objects removed).
    xx, zz, elev, has_data, xb, zb = build_elevation_view_grid(
        ground_pts, pts_aligned, GRID_RES
    )

    # 4b) Surface-top DEM via terrain_analysis.rasterize_bev H_top — the other
    #     documented DEM producer (export_elevation_json dem_source='htop').
    #     This one KEEPS objects, so the box/bottle stand out of the grid.
    bounds = (xb[0], xb[1], zb[0], zb[1])
    rast = ta.rasterize_bev(
        pts_aligned, sem_kept, ground_kept,
        grid_res=GRID_RES, top_percentile=90.0, bounds=bounds,
    )
    H_top = rast["H_top"]

    # 5) Put both DEMs in metres and reference to the tabletop (Y=0 at table).
    zero = float(np.median(ground_pts[:, 1]))
    elev_m = (elev - zero) * scale
    htop_m = (H_top - zero) * scale
    xx_m, zz_m = (xx - xx.min()) * scale, (zz - zz.min()) * scale
    dem_m = np.where(has_data, elev_m, np.nan)
    htop_dem = np.where(np.isfinite(H_top), htop_m, np.nan)
    return {
        "xx": xx_m, "zz": zz_m, "elev": elev_m, "dem": dem_m,
        "htop": htop_dem, "has_data": has_data, "scale": scale,
        "x_span": float((xb[1] - xb[0]) * scale),
        "z_span": float((zb[1] - zb[0]) * scale),
        "cell_cm": float((xb[1] - xb[0]) * scale / GRID_RES * 100),
        "valid_cells": int(has_data.sum()),
        "htop_cells": int(np.isfinite(H_top).sum()),
    }


def _draw_surface(ax, xx, zz, grid, dem, vmin, vmax, title):
    ls = LightSource(azdeg=315, altdeg=45)
    surf = ax.plot_surface(
        xx, zz, grid, cmap="turbo", vmin=vmin, vmax=vmax,
        rstride=1, cstride=1, linewidth=0.12, edgecolors="k",
        antialiased=True, shade=True, lightsource=ls, alpha=0.98,
    )
    surf.set_edgecolor((0, 0, 0, 0.18))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("height (m)")
    ax.set_title(title)
    ax.view_init(elev=32, azim=-60)
    ax.set_box_aspect((dem["x_span"], dem["z_span"], max(vmax - vmin, 0.15)))
    return surf


def render(dem: dict) -> Path:
    xx, zz, elev_m, dem_m, htop = dem["xx"], dem["zz"], dem["elev"], dem["dem"], dem["htop"]
    # Shared height range driven by the surface-top DEM so objects are visible.
    vmax = float(np.nanpercentile(htop, 99.5))
    vmin = float(np.nanpercentile(dem_m, 1.0))

    fig = plt.figure(figsize=(20, 6.2), dpi=170)

    # ── Panel 1: gridded terrain DEM (objects removed — same as viewer) ──────
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_surface(ax, xx, zz, np.where(np.isfinite(dem_m), elev_m, np.nan), dem,
                  vmin, vmax,
                  f"Terrain DEM · {GRID_RES}×{GRID_RES} (~{dem['cell_cm']:.2f} cm/cell)\n"
                  f"objects removed (matches /elevation_viewer_data)")

    # ── Panel 2: gridded surface-top DEM (objects kept — rasterize_bev H_top) ─
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    htop_grid = np.where(np.isfinite(htop), htop, np.nanmin(htop))
    surf = _draw_surface(ax2, xx, zz, htop_grid, dem, vmin, vmax,
                         f"Surface-top DEM · rasterize_bev H_top (P90)\n"
                         f"objects kept — box & bottle rise from the grid")

    # ── Panel 3: top-down surface-top heatmap + contours ─────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.pcolormesh(xx, zz, htop, cmap="turbo", vmin=vmin, vmax=vmax, shading="auto")
    levels = np.linspace(vmin, vmax, 12)
    cs = ax3.contour(xx, zz, np.where(np.isfinite(htop), htop, vmin),
                     levels=levels, colors="k", linewidths=0.4, alpha=0.5)
    ax3.clabel(cs, cs.levels[::3], fmt="%.2f", fontsize=6)
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)")
    ax3.set_aspect("equal")
    ax3.set_title(f"Top-down surface-top DEM · {dem['htop_cells']} filled cells")

    cb = fig.colorbar(surf, ax=[ax, ax2, ax3], shrink=0.62, pad=0.03)
    cb.set_label("height above tabletop (m)")
    fig.suptitle("VGGT gridded DEM after 15 cm ruler scaling · terrain-only vs surface-top "
                 "(both from the main pipeline's DEM producers)")
    out = VIZ / "reconstruction_3d_dem_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    dem = build_metric_dem()
    out = render(dem)
    manifest = {
        "figure": str(out),
        "source": "world_points_from_depth (Depthmap and Camera Branch)",
        "matches": "/elevation_viewer_data DEM: estimate_gravity + P50 keep + "
                   "semantic ground filter + build_elevation_view_grid (128, 2% pad)",
        "grid_resolution": GRID_RES,
        "approx_cell_size_cm": round(dem["cell_cm"], 3),
        "terrain_valid_cells": dem["valid_cells"],
        "surface_top_filled_cells": dem["htop_cells"],
        "surface_top_source": "terrain_analysis.rasterize_bev H_top P90 (export_elevation_json dem_source='htop')",
        "x_span_m": round(dem["x_span"], 3),
        "z_span_m": round(dem["z_span"], 3),
        "scale_m_per_vggt_unit": dem["scale"],
    }
    (VIZ / "dem_grid_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
