"""Build the REAL VGGT terrain base for E3 from the demo session.

This grounds E3 in actual VGGT output instead of a synthetic grid:
  * base DEM        — rasterize the session's real `world_points_from_depth` ground
    points (terrain_analysis.rasterize_bev), in a gravity-aligned metric frame.
  * per-cell point counts — how many REAL VGGT points fall in each DEM cell; this is
    the cost weight (a strategy that re-estimates a cell pays for its real points).
  * bucket anchors  — the real 3D bucket centroid per frame, mapped into DEM cells.

Only the terrain *change* (digging) is injected later (run_e3.py); the surface, the
cost units, and the interaction cue are all real. Saves e3_real_base.npz.

Env: ~/miniconda3/envs/vggt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.random import default_rng

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import terrain_analysis as ta

WS = Path(__file__).resolve().parent
SESSION = ROOT / "workspaces" / "session_20260629_100116_092814"
GRID = 96
GROUND_BAND_M = 0.02          # points below this aligned height are terrain (not the arm)


def ground_normal(P, iters=400, thr=0.01):
    rng = default_rng(0)
    best = (0, None, None)
    s = P[rng.choice(P.shape[0], 20000, replace=False)]
    for _ in range(iters):
        i = rng.choice(s.shape[0], 3, replace=False)
        p0, p1, p2 = s[i]
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        o = -n.dot(p0)
        inl = (np.abs(s.dot(n) + o) < thr).sum()
        if inl > best[0]:
            best = (inl, n, o)
    return best[1], best[2]


def main():
    d = np.load(SESSION / "predictions.npz")
    wp = d["world_points_from_depth"]
    conf = d["depth_conf"]
    N, H, W = wp.shape[:3]

    # gravity/ground frame from a mid frame (reuse arm_motion_state convention)
    ng, _ = ground_normal(wp[5].reshape(-1, 3))
    off = -np.median(wp[5].reshape(-1, 3).dot(ng))
    h5 = wp[5].reshape(-1, 3).dot(ng) + off
    if abs(np.percentile(h5, 1)) > abs(np.percentile(h5, 99)):
        ng, off = -ng, -off
    up = ng / np.linalg.norm(ng)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(up.dot(ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    ax_u = np.cross(up, ref); ax_u /= np.linalg.norm(ax_u)
    ax_v = np.cross(up, ax_u)

    def to_uvh(P):
        return np.stack([P.dot(ax_u), P.dot(ng) + off, P.dot(ax_v)], axis=1)  # [U, Hup, V]

    # aggregate all frames' real points, keep finite ground band
    P = wp.reshape(-1, 3)
    C = conf.reshape(-1)
    keep = np.isfinite(P).all(1) & np.isfinite(C)
    P = P[keep]
    uvh = to_uvh(P)                       # columns [U, Hup, V] == [X, Y(up), Z]
    ground = uvh[uvh[:, 1] < GROUND_BAND_M]

    # bounds from ground extent (+2% pad) so the DEM frame matches the real worksite
    u = ground[:, 0]; v = ground[:, 2]
    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    pu = (umax - umin) * 0.02; pv = (vmax - vmin) * 0.02
    bounds = (umin - pu, umax + pu, vmin - pv, vmax + pv)

    # rasterize the REAL ground points → real VGGT base DEM (H_top) + per-cell counts
    rast = ta.rasterize_bev(ground, None, None, grid_res=GRID,
                            top_percentile=70.0, bounds=bounds)
    base = rast["H_top"]                 # (GRID,GRID) real VGGT surface, NaN where empty
    counts = rast["count"].astype(np.int64)

    # fill NaN cells by nearest finite (so the injected-change map is dense) — records
    # a validity mask so we never claim data where VGGT saw none.
    valid = np.isfinite(base)
    filled = base.copy()
    if (~valid).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = base[tuple(idx)]

    # per-frame real bucket 3D anchor → DEM cell (row=V, col=U)
    import cv2
    from scipy.ndimage import label
    x_min, x_max, z_min, z_max = bounds
    def cell_of(uu, vv):
        col = int(np.clip((uu - x_min) / (x_max - x_min) * (GRID - 1), 0, GRID - 1))
        row = int(np.clip((vv - z_min) / (z_max - z_min) * (GRID - 1), 0, GRID - 1))
        return row, col

    anchors = []
    for f in range(N):
        Pf = wp[f].reshape(-1, 3)
        hh = (Pf.dot(ng) + off).reshape(H, W)
        mask = (hh > GROUND_BAND_M).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        lab, nl = label(mask)
        if nl > 1:
            sizes = [(lab == i).sum() for i in range(1, nl + 1)]
            mask = (lab == (np.argmax(sizes) + 1)).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        thr_x = np.percentile(xs, 12)
        sel = xs < thr_x
        bx, by = xs[sel], ys[sel]
        bp = Pf[by * W + bx].mean(axis=0)
        anchors.append(cell_of(float(bp.dot(ax_u)), float(bp.dot(ax_v))))
    anchors = np.array(anchors)

    cell_m = float((x_max - x_min) / GRID)
    np.savez_compressed(
        WS / "e3_real_base.npz",
        base_dem=filled, base_valid=valid, cell_counts=counts,
        anchors_cell=anchors, bounds=np.array(bounds),
        cell_m=cell_m, grid=GRID,
    )
    print(f"real VGGT base DEM: {GRID}x{GRID}, cell {cell_m*100:.2f} cm, "
          f"valid cells {int(valid.sum())}/{GRID*GRID}, "
          f"total real points {int(counts.sum()):,}, "
          f"height p2..p98 {np.round(np.nanpercentile(base[valid],[2,50,98]),3)}")
    print("bucket anchor cells (row,col):", anchors.tolist())


if __name__ == "__main__":
    main()
