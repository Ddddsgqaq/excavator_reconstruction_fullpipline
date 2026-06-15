"""
analyze_anchors.py — Compare scale-anchor robustness:
  (A) two hand-picked endpoints (naive geometric anchoring), vs
  (B) semantic-mask-constrained measurement (all masked 3D points, extent
      measured along the gravity-up axis with robust percentiles).

The point is to quantify how much the YOLOe mask reduces the scale variance
that background-bleed and pick error inject into the endpoint method.

Stability is measured by bootstrapping (resampling pixels / jittering picks)
and reporting the relative spread of the recovered VGGT length. Because a
reconstructed volume scales as length**3, the volume-level spread is ~3x the
length-level spread — reported alongside.
"""
from __future__ import annotations
import numpy as np
import scale_calibration as sc
import gravity_alignment as ga


def gravity_R(preds: dict) -> np.ndarray:
    res = ga.estimate_gravity(
        preds["extrinsic"], preds["world_points"],
        conf=preds.get("world_points_conf"),
    )
    return res.R_align


def masked_points_aligned(preds, frame, mask, R_align, conf_frac=0.1):
    """Confidence-filtered, gravity-aligned 3D points under a boolean mask."""
    wp = preds["world_points"][frame]                 # (H,W,3)
    conf = preds["world_points_conf"][frame]          # (H,W)
    m = mask & np.isfinite(wp).all(axis=2)
    if conf is not None:
        m = m & (conf >= conf_frac * float(conf[mask].max() + 1e-9))
    pts = wp[m]
    return sc.np.asarray(ga.apply_alignment_to_points(pts, R_align))


def robust_extent(pts_aligned, axis=1, lo=2, hi=98):
    """Extent along an axis using percentiles (default Y = gravity-up)."""
    v = pts_aligned[:, axis]
    return float(np.percentile(v, hi) - np.percentile(v, lo))


def bootstrap_mask_extent(pts_aligned, axis=1, n=300, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    N = len(pts_aligned)
    k = max(10, int(N * frac))
    out = []
    for _ in range(n):
        idx = rng.choice(N, k, replace=True)
        out.append(robust_extent(pts_aligned[idx], axis=axis))
    return np.array(out)


def bootstrap_endpoint(preds, frame, p0, p1, n=300, jitter=2, win=3, seed=0):
    """Re-pick endpoints under +-jitter px to mimic human pick error."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = (frame, p0[0] + rng.integers(-jitter, jitter + 1), p0[1] + rng.integers(-jitter, jitter + 1))
        b = (frame, p1[0] + rng.integers(-jitter, jitter + 1), p1[1] + rng.integers(-jitter, jitter + 1))
        w0 = sc.pixel_to_world(preds, *a, win=win)
        w1 = sc.pixel_to_world(preds, *b, win=win)
        if w0 is None or w1 is None:
            continue
        out.append(float(np.linalg.norm(w1 - w0)))
    return np.array(out)


def summarize(name, samples):
    med = float(np.median(samples))
    spread = (np.percentile(samples, 95) - np.percentile(samples, 5)) / (med + 1e-12)
    return f"{name:28s} median_len={med:.4f}  p5-p95_rel_spread={spread*100:5.1f}%  (vol ~{spread*300:.0f}%)"


if __name__ == "__main__":
    W = "workspaces/session_20260611_162643_869764"
    preds = sc.load_predictions(W)
    masks = np.load(f"{W}/masks_f0.npz")
    R = gravity_R(preds)
    print("gravity R_align estimated.\n")

    # (A) endpoint method on the person (the unstable baseline)
    ep = bootstrap_endpoint(preds, 0, (341, 46), (340, 79), jitter=2, win=3)
    print(summarize("A) person endpoints", ep))

    # (B) mask-constrained extent along gravity-up
    for cls in ("person", "excavator"):
        if cls not in masks.files:
            continue
        pts = masked_points_aligned(preds, 0, masks[cls], R)
        if len(pts) < 10:
            print(f"B) {cls}: too few masked points ({len(pts)})"); continue
        bs = bootstrap_mask_extent(pts, axis=1)
        print(summarize(f"B) {cls} mask (Y-extent)", bs) + f"   [{len(pts)} pts]")
