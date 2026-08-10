"""
exp_dyn0_probe.py — E-DYN-0 offline analysis (see EXP_DYN0_PLAN.md).

NOTE: This script does NOT run VGGT or YOLOe. It consumes the artifacts they
produce (predictions.npz + semantic_masks.npz under a workspace session dir)
and quantifies the four probes P1-P4 that decide whether the L2 agent layer is
feasible on a dynamic, articulated excavator.

Inputs expected in --session:
    predictions.npz   : keys world_points (S,H,W,3), world_points_conf (S,H,W),
                        extrinsic (S,3,4), images (S,3,H,W)
    semantic_masks.npz: key semantic_masks (S,Hm,Wm) uint8, per-class ids,
                        background = 0. (Hm,Wm may differ from VGGT's H,W —
                        we nearest-resize.)

Usage:
    python exp_dyn0_probe.py --session workspaces/<session> \
        --machine-ids 1,2,3 --ground-id 4 --out output/exp_dyn0

Class ids: read the YOLOe sem_id_map printed by the service / orchestrator
(e.g. {"excavator":1,"excavator arm":2,"bucket":3,"ground":4}) and pass the
machine ids and the ground id explicitly.
"""

import os
import json
import argparse
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _nearest_resize_mask(mask, H, W):
    """Nearest-neighbour resize an (Hm,Wm) integer mask to (H,W)."""
    Hm, Wm = mask.shape
    if (Hm, Wm) == (H, W):
        return mask
    ys = (np.linspace(0, Hm - 1, H)).round().astype(int)
    xs = (np.linspace(0, Wm - 1, W)).round().astype(int)
    return mask[np.ix_(ys, xs)]


def _fit_plane_rms(pts):
    """RMS distance of pts (N,3) to their best-fit plane (PCA, smallest axis)."""
    if pts.shape[0] < 50:
        return float("nan"), 0
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    normal = vt[-1]
    d = (pts - c) @ normal
    return float(np.sqrt((d ** 2).mean())), int(pts.shape[0])


def _cam_centers(extrinsic):
    """World-frame camera centers from (S,3,4) [R|t] world->cam extrinsics."""
    C = np.zeros((extrinsic.shape[0], 3), dtype=np.float64)
    for i, E in enumerate(extrinsic):
        R, t = E[:, :3], E[:, 3]
        C[i] = -R.T @ t
    return C


# ── main analysis ────────────────────────────────────────────────────────────

def run(session, machine_ids, ground_id, conf_pct, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    preds = np.load(os.path.join(session, "predictions.npz"))
    wp = np.asarray(preds["world_points"])          # (S,H,W,3)
    conf = np.asarray(preds["world_points_conf"])   # (S,H,W)
    extr = np.asarray(preds["extrinsic"])           # (S,3,4)
    S, H, W, _ = wp.shape

    sm = np.load(os.path.join(session, "semantic_masks.npz"))["semantic_masks"]
    sm = np.asarray(sm)                             # (S,Hm,Wm)

    cam_C = _cam_centers(extr)
    scene_scale = float(np.linalg.norm(cam_C - cam_C.mean(0), axis=1).mean() + 1e-9)

    conf_thr = (conf_pct / 100.0) * float(conf.max())

    machine_centroids = []   # per-frame machine centroid (P2/P4)
    machine_pts_all = []     # all machine pts pooled (P2)
    terrain_rms_near, terrain_rms_far = [], []  # (P1)
    hit = {cid: 0 for cid in machine_ids}        # per-class frames-with-mask (P3)
    px_counts = {cid: [] for cid in machine_ids} # mask pixel counts (P3)

    for s in range(S):
        m = _nearest_resize_mask(sm[s], H, W)
        pts_s = wp[s].reshape(-1, 3)
        conf_s = conf[s].reshape(-1)
        m_flat = m.reshape(-1)
        good = np.isfinite(pts_s).all(1) & (conf_s >= conf_thr)

        # P3 per-class presence + pixel counts
        for cid in machine_ids:
            cnt = int((m == cid).sum())
            px_counts[cid].append(cnt)
            if cnt > 0:
                hit[cid] += 1

        # machine points (union of machine ids)
        mach_sel = good & np.isin(m_flat, machine_ids)
        if mach_sel.sum() >= 30:
            mp = pts_s[mach_sel]
            machine_pts_all.append(mp)
            machine_centroids.append(mp.mean(0))
        else:
            machine_centroids.append(np.array([np.nan] * 3))

        # P1 terrain: ground points, split near/far from machine image region
        if ground_id is not None:
            gsel = good & (m_flat == ground_id)
            gpts = pts_s[gsel]
            if gpts.shape[0] >= 50:
                # near = within median machine depth band; coarse proxy: split by
                # distance to this frame's machine centroid in world space.
                mc = machine_centroids[-1]
                if np.all(np.isfinite(mc)):
                    dist = np.linalg.norm(gpts - mc, axis=1)
                    near = gpts[dist < np.median(dist)]
                    far = gpts[dist >= np.median(dist)]
                    r_n, _ = _fit_plane_rms(near)
                    r_f, _ = _fit_plane_rms(far)
                    if np.isfinite(r_n):
                        terrain_rms_near.append(r_n)
                    if np.isfinite(r_f):
                        terrain_rms_far.append(r_f)

    machine_centroids = np.asarray(machine_centroids)
    valid_c = machine_centroids[np.isfinite(machine_centroids).all(1)]

    # P2 dispersion
    if len(machine_pts_all):
        pooled = np.concatenate(machine_pts_all, 0)
        _, ss, _ = np.linalg.svd(pooled - pooled.mean(0), full_matrices=False)
        principal_len = float(ss[0] / np.sqrt(max(len(pooled) - 1, 1)))
    else:
        principal_len = float("nan")
    centroid_spread = float(np.linalg.norm(valid_c - valid_c.mean(0), axis=1).mean()) \
        if len(valid_c) > 1 else float("nan")

    # P4 trajectory jumps
    jumps = np.linalg.norm(np.diff(valid_c, axis=0), axis=1) if len(valid_c) > 1 \
        else np.array([])
    max_jump = float(jumps.max()) if jumps.size else float("nan")

    report = {
        "session": session,
        "n_frames": S, "vggt_HW": [H, W],
        "scene_scale_ref": scene_scale,
        "conf_threshold": conf_thr,
        "P1_terrain": {
            "rms_near_machine_mean": float(np.mean(terrain_rms_near)) if terrain_rms_near else None,
            "rms_far_machine_mean": float(np.mean(terrain_rms_far)) if terrain_rms_far else None,
            "ratio_near_over_far": (float(np.mean(terrain_rms_near) / np.mean(terrain_rms_far))
                                    if terrain_rms_near and terrain_rms_far else None),
            "verdict_hint": "<=1.5 => terrain not significantly corrupted",
        },
        "P2_machine_shape": {
            "principal_axis_len": principal_len,
            "centroid_spread": centroid_spread,
            "spread_over_scene_scale": (centroid_spread / scene_scale) if np.isfinite(centroid_spread) else None,
            "verdict_hint": "spread << machine size => coherent blob; >> => ghosting",
        },
        "P3_yoloe": {
            "hit_rate": {str(cid): hit[cid] / S for cid in machine_ids},
            "px_cv": {str(cid): (float(np.std(px_counts[cid]) / (np.mean(px_counts[cid]) + 1e-9)))
                      for cid in machine_ids},
            "verdict_hint": "excavator hit_rate>=0.9 => body node usable",
        },
        "P4_trajectory": {
            "n_valid_centroid_frames": int(len(valid_c)),
            "max_consecutive_jump": max_jump,
            "max_jump_over_scene_scale": (max_jump / scene_scale) if np.isfinite(max_jump) else None,
            "verdict_hint": "no jump >> machine size => continuous track",
        },
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # optional plots (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(valid_c) > 1:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(valid_c[:, 0], valid_c[:, 1], valid_c[:, 2], "-o", ms=3)
            ax.set_title("P4 machine centroid trajectory (world frame)")
            fig.savefig(os.path.join(out_dir, "centroid_traj.png"), dpi=120)
        if terrain_rms_near or terrain_rms_far:
            plt.figure()
            if terrain_rms_near:
                plt.hist(terrain_rms_near, bins=20, alpha=0.6, label="near machine")
            if terrain_rms_far:
                plt.hist(terrain_rms_far, bins=20, alpha=0.6, label="far machine")
            plt.legend(); plt.title("P1 terrain plane-fit RMS")
            plt.savefig(os.path.join(out_dir, "terrain_residual_hist.png"), dpi=120)
    except Exception as e:  # noqa: BLE001
        print(f"[plot skipped] {e}")

    print(json.dumps(report, indent=2))
    return report


def _parse_ids(s):
    return [int(x) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E-DYN-0 offline probe (P1-P4).")
    ap.add_argument("--session", required=True, help="workspaces/<session> dir")
    ap.add_argument("--machine-ids", required=True,
                    help="comma sep sem ids for excavator/arm/bucket, e.g. 1,2,3")
    ap.add_argument("--ground-id", type=int, default=None,
                    help="sem id of ground/terrain class (for P1)")
    ap.add_argument("--conf-pct", type=float, default=50.0,
                    help="confidence percentile threshold (matches viewer default 50)")
    ap.add_argument("--out", default="output/exp_dyn0")
    a = ap.parse_args()
    run(a.session, _parse_ids(a.machine_ids), a.ground_id, a.conf_pct, a.out)
