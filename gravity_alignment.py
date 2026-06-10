"""
gravity_alignment.py — Estimate gravity direction and build a rotation
that maps it to +Y.

VGGT's world frame coincides with frame-0's OpenCV camera (X-right, Y-down,
Z-forward). The first frame is generally not level, so the raw point cloud
is tilted. This module estimates a gravity direction `n_grav` (a unit
vector in VGGT world space pointing "up") and returns a rotation R_align
such that R_align @ n_grav == (0, 1, 0). After applying R_align to all
points, Y becomes elevation and (X, Z) is the horizontal plane.

Strategy (in order, with fallbacks):
    1. Trajectory plane  — PCA on camera centers. Primary source.
    2. Ground-mask plane — RANSAC on YOLOe-labeled ground points.
    3. Whole-cloud plane — RANSAC on the full point cloud, biased toward
       the trajectory normal as a prior.

Sanity check: when both trajectory and ground-mask are available, compare
their normals; warn if the angle exceeds GRAVITY_DISAGREE_DEG, but still
use the trajectory normal.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


GRAVITY_DISAGREE_DEG = 10.0
TRAJ_DEGENERATE_RATIO = 0.1   # second/first PCA eigenvalue below this → degenerate
RANSAC_ITERS = 400
RANSAC_DIST = 0.03            # in VGGT world units (relative scale)


@dataclass
class GravityResult:
    R_align: np.ndarray                  # (3, 3) world → aligned-world
    n_grav: np.ndarray                   # (3,) gravity-up direction in VGGT world
    source: str                          # "trajectory" | "ground_mask" | "cloud_ransac"
    inlier_count: int
    warnings: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    """extrinsic: (S, 3, 4) world-to-camera → returns (S, 3) camera centers in world."""
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    # C = -R^T t
    return -np.einsum("sij,sj->si", R.transpose(0, 2, 1), t)


def _shortest_arc_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation that sends unit vector `src` to unit vector `dst` along the shortest arc."""
    src = src / (np.linalg.norm(src) + 1e-12)
    dst = dst / (np.linalg.norm(dst) + 1e-12)
    v = np.cross(src, dst)
    s = np.linalg.norm(v)
    c = float(np.dot(src, dst))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180° flip — pick any perpendicular axis
        axis = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - axis.dot(src) * src
        axis /= np.linalg.norm(axis)
        K = _skew(axis)
        return np.eye(3) + 2.0 * K @ K
    K = _skew(v / s)
    return np.eye(3) + s * K + (1.0 - c) * K @ K


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def _orient_normal_up(n: np.ndarray, points_above: np.ndarray, plane_anchor: np.ndarray) -> np.ndarray:
    """
    Flip `n` so that the majority of `points_above` lies on the +n side of the plane
    through `plane_anchor`. For trajectory: points_above = scene point cloud (cameras
    are above ground, so most cloud points are below the trajectory plane → we want n
    to point away from the cloud, which is "up").
    """
    rel = points_above - plane_anchor
    s = np.dot(rel, n)
    # We want most points on the *negative* side (cloud is below).
    if np.mean(s > 0) > 0.5:
        return -n
    return n


# ── Plane estimators ─────────────────────────────────────────────────────────

def estimate_from_trajectory(
    extrinsic: np.ndarray,
    cloud_for_orientation: np.ndarray,
) -> Optional[tuple]:
    """
    PCA on camera centers. Returns (normal, anchor, inlier_count, debug)
    or None if the trajectory is too degenerate to define a plane.
    """
    centers = _camera_centers(extrinsic)
    S = centers.shape[0]
    if S < 3:
        return None

    centroid = centers.mean(axis=0)
    X = centers - centroid
    # Use SVD for numerical stability.
    _, sv, Vt = np.linalg.svd(X, full_matrices=False)
    # sv: (3,) descending. Plane normal = direction of smallest variance.
    eig = sv ** 2 / max(S - 1, 1)
    ratio = eig[1] / max(eig[0], 1e-12)
    normal = Vt[-1]
    debug = {
        "pca_eigenvalues": eig.tolist(),
        "second_first_ratio": float(ratio),
        "n_cameras": int(S),
    }
    if ratio < TRAJ_DEGENERATE_RATIO:
        # Trajectory collapses to a line / point — plane is undefined.
        debug["degenerate"] = True
        return None

    normal = normal / (np.linalg.norm(normal) + 1e-12)
    normal = _orient_normal_up(normal, cloud_for_orientation, centroid)
    return normal, centroid, S, debug


def _ransac_plane(points: np.ndarray,
                  n_iter: int = RANSAC_ITERS,
                  dist_thresh: float = RANSAC_DIST,
                  prior_normal: Optional[np.ndarray] = None,
                  prior_cos: float = 0.5) -> Optional[tuple]:
    """
    Generic plane RANSAC. If `prior_normal` is given, only accept candidate
    planes whose |normal · prior_normal| > prior_cos (i.e. roughly parallel
    to the prior). Returns (normal, anchor, inlier_count, mask) or None.
    """
    n = points.shape[0]
    if n < 50:
        return None
    rng = np.random.default_rng(0xC0FFEE)
    best = None
    best_count = 0
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-8:
            continue
        normal = normal / nrm
        if prior_normal is not None and abs(float(np.dot(normal, prior_normal))) < prior_cos:
            continue
        dist = np.abs((points - p0) @ normal)
        mask = dist < dist_thresh
        cnt = int(mask.sum())
        if cnt > best_count:
            best_count = cnt
            best = (normal, p0, cnt, mask)
    if best is None:
        return None
    # Refit on inliers via SVD for a tighter normal.
    normal, anchor, cnt, mask = best
    inliers = points[mask]
    centroid = inliers.mean(axis=0)
    _, _, Vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = Vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return normal, centroid, cnt, mask


def estimate_from_ground_mask(
    world_points: np.ndarray,           # (S, H, W, 3) or (N, 3)
    ground_mask: np.ndarray,            # (S, H, W) bool / 0-1
    conf: Optional[np.ndarray] = None,  # (S, H, W) confidence
    conf_thres: float = 0.1,
    prior_normal: Optional[np.ndarray] = None,
) -> Optional[tuple]:
    """
    RANSAC on ground-labeled points. Returns (normal, anchor, inlier_count, debug).
    """
    pts = world_points.reshape(-1, 3)
    gm = ground_mask.reshape(-1).astype(bool)
    keep = gm & np.isfinite(pts).all(axis=1)
    if conf is not None:
        c = conf.reshape(-1)
        keep &= c >= (conf_thres * c.max())
    pts = pts[keep]
    if pts.shape[0] < 50:
        return None
    out = _ransac_plane(pts, prior_normal=prior_normal, prior_cos=0.5)
    if out is None:
        return None
    normal, anchor, cnt, _mask = out
    # Orient: ground normal should point toward the cameras (up).
    # We don't have cameras here — use the prior if available; otherwise flip
    # so that the larger remaining cloud sits on the +normal side.
    if prior_normal is not None and float(np.dot(normal, prior_normal)) < 0:
        normal = -normal
    debug = {"ground_pts_total": int(keep.sum()), "ground_inliers": cnt}
    return normal, anchor, cnt, debug


def estimate_from_cloud(
    world_points: np.ndarray,
    conf: Optional[np.ndarray] = None,
    conf_thres: float = 0.1,
    prior_normal: Optional[np.ndarray] = None,
) -> Optional[tuple]:
    """Whole-cloud RANSAC, optionally biased by `prior_normal`."""
    pts = world_points.reshape(-1, 3)
    keep = np.isfinite(pts).all(axis=1)
    if conf is not None:
        c = conf.reshape(-1)
        keep &= c >= (conf_thres * c.max())
    pts = pts[keep]
    if pts.shape[0] < 200:
        return None
    # Subsample for speed.
    if pts.shape[0] > 200_000:
        idx = np.random.default_rng(7).choice(pts.shape[0], 200_000, replace=False)
        pts = pts[idx]
    out = _ransac_plane(pts, prior_normal=prior_normal, prior_cos=0.6 if prior_normal is not None else 0.0)
    if out is None:
        return None
    normal, anchor, cnt, _mask = out
    if prior_normal is not None and float(np.dot(normal, prior_normal)) < 0:
        normal = -normal
    else:
        # No prior: orient so most points are on the *negative* side (normal points "up",
        # away from the bulk of the cloud which sits below the ground plane).
        normal = _orient_normal_up(normal, pts, anchor)
    return normal, anchor, cnt, {"cloud_inliers": cnt}


# ── Top-level orchestration ─────────────────────────────────────────────────

def estimate_gravity(
    extrinsic: np.ndarray,
    world_points: np.ndarray,
    ground_mask: Optional[np.ndarray] = None,
    conf: Optional[np.ndarray] = None,
    conf_thres: float = 0.1,
) -> GravityResult:
    """
    Run the trajectory-first cascade. Returns a GravityResult with the
    rotation that aligns gravity to +Y.
    """
    warnings: list = []
    debug: dict = {}

    cloud_full = world_points.reshape(-1, 3)
    finite = np.isfinite(cloud_full).all(axis=1)
    cloud_full = cloud_full[finite]

    # 1. Trajectory plane (primary).
    traj = estimate_from_trajectory(extrinsic, cloud_full)
    debug["trajectory"] = traj[3] if traj is not None else {"degenerate": True}

    chosen = None
    if traj is not None:
        n_traj, anchor_t, cnt_t, _ = traj
        chosen = ("trajectory", n_traj, anchor_t, cnt_t)

    # 2. Ground-mask sanity check / fallback.
    n_ground = None
    if ground_mask is not None and ground_mask.any():
        prior = chosen[1] if chosen is not None else None
        gm = estimate_from_ground_mask(world_points, ground_mask, conf, conf_thres, prior_normal=prior)
        if gm is not None:
            n_ground, anchor_g, cnt_g, dbg_g = gm
            debug["ground_mask"] = dbg_g
            if chosen is None:
                chosen = ("ground_mask", n_ground, anchor_g, cnt_g)
            else:
                # Sanity check vs trajectory normal.
                cosang = float(np.clip(np.dot(chosen[1], n_ground), -1.0, 1.0))
                ang_deg = float(np.degrees(np.arccos(abs(cosang))))
                debug["traj_vs_ground_deg"] = ang_deg
                if ang_deg > GRAVITY_DISAGREE_DEG:
                    warnings.append(
                        f"Trajectory normal disagrees with ground-mask normal by "
                        f"{ang_deg:.1f}° (>{GRAVITY_DISAGREE_DEG}°). Using trajectory."
                    )

    # 3. Whole-cloud fallback.
    if chosen is None:
        cl = estimate_from_cloud(world_points, conf, conf_thres)
        if cl is not None:
            n_c, anchor_c, cnt_c, dbg_c = cl
            debug["cloud"] = dbg_c
            chosen = ("cloud_ransac", n_c, anchor_c, cnt_c)
            warnings.append("Falling back to whole-cloud RANSAC for gravity estimate.")

    if chosen is None:
        raise RuntimeError("Could not estimate gravity from any source.")

    source, n_grav, _anchor, cnt = chosen
    R_align = _shortest_arc_rotation(n_grav, np.array([0.0, 1.0, 0.0]))

    return GravityResult(
        R_align=R_align,
        n_grav=n_grav,
        source=source,
        inlier_count=int(cnt),
        warnings=warnings,
        debug=debug,
    )


# ── Convenience: apply alignment to points / extrinsics ─────────────────────

def apply_alignment_to_points(points: np.ndarray, R_align: np.ndarray) -> np.ndarray:
    """Rotate point cloud (any leading shape, last dim = 3) into the aligned frame."""
    flat = points.reshape(-1, 3)
    out = flat @ R_align.T
    return out.reshape(points.shape)


def apply_alignment_to_extrinsics(extrinsic: np.ndarray, R_align: np.ndarray) -> np.ndarray:
    """
    Update world-to-camera extrinsics so they refer to the aligned world frame.
    Original:  X_cam = R X_world + t
    New world: X_world' = R_align X_world  →  X_world = R_align^T X_world'
    Therefore X_cam = R R_align^T X_world' + t
    i.e. R' = R @ R_align^T,  t' = t  (translation unchanged).
    """
    out = extrinsic.copy()
    out[:, :3, :3] = extrinsic[:, :3, :3] @ R_align.T
    return out
