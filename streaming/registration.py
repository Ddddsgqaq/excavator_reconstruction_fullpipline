"""registration.py — cross-pass horizontal alignment for streaming DEMs (M4.2).

After gravity alignment (M4.1 freezes the same `R_align` + `scale_factor` for every
pass), two consecutive passes already share the up-axis (+Y) and the metric scale. What
still drifts between passes is the *horizontal* pose: VGGT's world frame is frame-0's
camera, and each streaming window has a different frame-0, so the ground is rotated about
Y (yaw) and shifted in (X, Z) from window to window. Left uncorrected, the Unity tile
pans/rotates every update even though the terrain is static.

We estimate that residual as a yaw (about the gravity-up axis) + horizontal translation
mapping the current pass onto the anchor's reference cloud.

Key design point — WHY correspondences are found in 3D but the solve is 2D:
  The horizontal (X, Z) *positions* of ground samples are just the observed footprint —
  roughly uniform, with little distinctive structure — so matching in (X, Z) alone leaves
  yaw badly under-constrained. The terrain's distinctive structure lives in the elevation
  (Y). So we match nearest neighbours in full 3D (X, Y, Z) — relief pins the correspondence
  — but constrain the recovered transform to yaw + horizontal translation (Y is already
  fixed by the anchor). A coarse yaw multi-start escapes the ICP local minimum that a large
  inter-window yaw would otherwise cause. On genuinely flat ground translation along the
  flat direction is fundamentally unobservable, which is fine (flat terrain looks the same
  shifted).

Pure numpy/scipy; no torch, no GPU. Unit-testable with synthetic clouds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Rigid2D:
    """A yaw + horizontal-translation transform: only (X, Z) move, Y is untouched.

    `R` is a 2x2 rotation (yaw about gravity-up), `t` a length-2 (X, Z) translation.
    `identity()` is the no-op used when registration is disabled or unavailable.
    """
    R: np.ndarray                       # (2, 2) rotation in the (X, Z) plane
    t: np.ndarray                       # (2,) translation in (X, Z)
    rmse: float = 0.0                   # inlier 3D RMSE after the final iteration (world units)
    inlier_frac: float = 1.0            # fraction of source points kept as inliers
    yaw_deg: float = 0.0                # recovered yaw magnitude, for logging
    converged: bool = True

    @staticmethod
    def identity() -> "Rigid2D":
        return Rigid2D(R=np.eye(2), t=np.zeros(2), rmse=0.0, inlier_frac=1.0,
                       yaw_deg=0.0, converged=True)

    def apply_xz(self, xz: np.ndarray) -> np.ndarray:
        """Apply to an (N, 2) array of horizontal (X, Z) coordinates."""
        xz = np.asarray(xz, dtype=np.float64)
        return xz @ self.R.T + self.t

    def apply_to_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply to (N, 3) aligned points, transforming only the (X, Z) columns.

        Y (elevation) is left untouched — the anchor already fixed the vertical frame.
        """
        pts = np.asarray(pts, dtype=np.float64).copy()
        pts[:, [0, 2]] = self.apply_xz(pts[:, [0, 2]])
        return pts


def _kabsch_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares 2D rigid fit: find R (2x2 rotation), t so that R@src + t ≈ dst.

    src, dst: paired (N, 2) arrays. Standard Kabsch/Umeyama without scaling, with a
    reflection guard so the result is a proper rotation (det = +1).
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = (src - mu_s).T @ (dst - mu_d)          # (2, 2) cross-covariance
    U, _, Vt = np.linalg.svd(S)
    d = np.sign(np.linalg.det(Vt.T @ U.T))     # reflection guard → proper rotation
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def _subsample(pts: np.ndarray, max_pts: int, seed: int) -> np.ndarray:
    """Deterministically subsample rows of an (N, k) array to at most max_pts."""
    n = pts.shape[0]
    if n <= max_pts:
        return pts
    idx = np.random.default_rng(seed).choice(n, max_pts, replace=False)
    return pts[idx]


def _yaw_matrix(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s], [s, c]])


def _icp_once(
    src: np.ndarray,           # (N, 3) source points (X, Y, Z)
    dst: np.ndarray,           # (M, 3) target points
    tree,                      # cKDTree over dst
    R0: np.ndarray,            # (2, 2) initial yaw
    t0: np.ndarray,            # (2,) initial translation
    *,
    max_iters: int,
    n_keep: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Trimmed ICP from one initialisation. Correspondences in 3D, solve yaw+XZ only."""
    R, t = R0.copy(), t0.copy()
    prev_rmse = np.inf
    rmse = np.inf
    converged = False
    for _ in range(max_iters):
        cur = src.copy()
        cur[:, [0, 2]] = src[:, [0, 2]] @ R.T + t       # move only X,Z; keep Y
        dists, nn = tree.query(cur, k=1)                # nearest in full 3D
        order = np.argsort(dists)[:n_keep]              # trim to closest inliers
        R, t = _kabsch_2d(src[order][:, [0, 2]], dst[nn[order]][:, [0, 2]])
        rmse = float(np.sqrt(np.mean(dists[order] ** 2)))
        if abs(prev_rmse - rmse) < tol:
            converged = True
            break
        prev_rmse = rmse
    return R, t, rmse, converged


def _coarse_yaw_score(src, dst, tree, mu_s, mu_d, n_keep, yaw_deg):
    """One-shot trimmed RMSE for a centroid-aligned yaw guess (no ICP iteration).

    Cheap enough to sweep densely: a single 3D nearest-neighbour query. Used to locate
    ICP's convergence basin — the basin is only ~10° wide, so 45°-spaced seeds miss most
    yaws, but a dense one-shot scan reliably finds the right neighbourhood.
    """
    R0 = _yaw_matrix(yaw_deg)
    t0 = mu_d - R0 @ mu_s
    cur = src.copy()
    cur[:, [0, 2]] = src[:, [0, 2]] @ R0.T + t0
    dists, _ = tree.query(cur, k=1)
    kept = np.partition(dists, n_keep - 1)[:n_keep]
    return float(np.sqrt(np.mean(kept ** 2)))


def register_horizontal(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    max_iters: int = 40,
    trim_frac: float = 0.7,
    max_pts: int = 4000,
    tol: float = 1e-5,
    coarse_step_deg: float = 6.0,
    n_refine: int = 3,
    seed: int = 0,
) -> Rigid2D:
    """Estimate the yaw + horizontal translation aligning `src_pts` onto `dst_pts`.

    src_pts: current pass's ground points, (N, 3) aligned+scaled (X, Y, Z).
    dst_pts: anchor's reference ground points, (M, 3) in the same frame.
    trim_frac: fraction of closest correspondences kept each iteration (trimmed ICP →
               robust to partial overlap from scene change / digging).
    coarse_step_deg: resolution of the dense one-shot yaw scan that seeds ICP.
    n_refine: how many best coarse yaws to run full ICP from (the best result wins).

    Two stages: (1) a dense coarse yaw scan over the full circle locates ICP's basin
    (ICP alone only converges from within ~10° of the true yaw); (2) full trimmed ICP
    refines from the best few coarse seeds. Returns a Rigid2D mapping src → dst
    (transforming only X, Z). On degenerate input returns identity, converged=False.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3 or dst.shape[1] != 3:
        raise ValueError("register_horizontal expects (N, 3) point arrays")
    if src.shape[0] < 10 or dst.shape[0] < 10:
        out = Rigid2D.identity()
        out.converged = False
        return out

    src = _subsample(src, max_pts, seed)
    dst = _subsample(dst, max_pts, seed + 1)
    tree = cKDTree(dst)

    # Common centroid translation as the base offset for every yaw seed.
    mu_s = src[:, [0, 2]].mean(axis=0)
    mu_d = dst[:, [0, 2]].mean(axis=0)
    n_keep = max(3, int(round(src.shape[0] * trim_frac)))

    # Stage 1 — dense one-shot coarse yaw scan; keep the n_refine lowest-RMSE angles.
    yaws = np.arange(0.0, 360.0, coarse_step_deg)
    scores = [_coarse_yaw_score(src, dst, tree, mu_s, mu_d, n_keep, y) for y in yaws]
    seed_yaws = yaws[np.argsort(scores)[:max(1, n_refine)]]

    # Stage 2 — full ICP refinement from each seed; best RMSE wins.
    best = None
    for yaw in seed_yaws:
        R0 = _yaw_matrix(float(yaw))
        t0 = mu_d - R0 @ mu_s
        R, t, rmse, converged = _icp_once(
            src, dst, tree, R0, t0,
            max_iters=max_iters, n_keep=n_keep, tol=tol,
        )
        if best is None or rmse < best[2]:
            best = (R, t, rmse, converged)

    R, t, rmse, converged = best
    yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return Rigid2D(
        R=R, t=t, rmse=rmse,
        inlier_frac=float(n_keep) / float(src.shape[0]),
        yaw_deg=yaw_deg, converged=converged,
    )
