"""
scale_calibration.py — Recover an absolute metric scale for a VGGT
reconstruction from a known-size reference in the scene.

VGGT (like any monocular feed-forward reconstructor) returns geometry up to
an unknown global scale: `world_points` live in arbitrary "VGGT units". Every
downstream measurement — elevation, cut/fill volume — is therefore only
relative until we pin one real-world length. This module pins it.

Method (the contribution layer, independent of the VGGT backbone):
    1. The user names a reference of known metric size (e.g. a person ~1.70 m,
       an excavator track of known length, a calibration board).
    2. We pick the reference's two endpoints (two pixels, optionally across
       different frames) and read their robust 3D positions from world_points.
    3. scale = known_metric_length / vggt_length.
    4. We cross-check scale across multiple anchors / frames and report the
       spread — a self-consistency signal that needs no external ground truth.

Key sensitivity fact used throughout the analysis:
    A reconstruction volume scales as scale**3, so a relative scale error
    `e` propagates to a *volume* error of ~3e (first order). Pinning scale
    well is therefore 3x as important for volume as for length.

Robustness: a pixel's 3D point is sampled as the confidence-filtered median
over a small window, with a fallback to `world_points_from_depth`, so a
single noisy ray cannot dominate the estimate.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ── Data loading ─────────────────────────────────────────────────────────────

PRED_PRIMARY = "world_points"
PRED_FALLBACK = "world_points_from_depth"


def load_predictions(workspace: str) -> dict:
    """Load predictions.npz from a workspace directory (or a direct .npz path)."""
    path = workspace
    if os.path.isdir(workspace):
        path = os.path.join(workspace, "predictions.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"predictions.npz not found at {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _display_frame(preds: dict, frame: int) -> np.ndarray:
    """Return an (H, W, 3) uint8 image for a frame, robust to the stored range."""
    img = preds["images"][frame]            # (3, H, W), float
    img = np.transpose(img, (1, 2, 0))      # (H, W, 3)
    lo, hi = float(img.min()), float(img.max())
    if hi <= 1.0 + 1e-3 and lo >= -1e-3:    # already [0, 1]
        img = img * 255.0
    else:                                   # arbitrary range → min-max stretch
        img = (img - lo) / (hi - lo + 1e-9) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


# ── Robust pixel → world sampling ────────────────────────────────────────────

def pixel_to_world(
    preds: dict,
    frame: int,
    u: int,
    v: int,
    win: int = 3,
    conf_frac: float = 0.1,
    source: str = "auto",
) -> Optional[np.ndarray]:
    """
    Robust 3D position of pixel (u, v) in frame `frame`, as the
    confidence-filtered median over a (2*win+1) window.

    u indexes width, v indexes height. Returns None if no valid point.
    """
    keys = [PRED_PRIMARY, PRED_FALLBACK] if source == "auto" else [source]
    H, W = preds[PRED_PRIMARY].shape[1:3]
    v0, v1 = max(0, v - win), min(H, v + win + 1)
    u0, u1 = max(0, u - win), min(W, u + win + 1)

    conf = None
    if "world_points_conf" in preds:
        conf = preds["world_points_conf"][frame, v0:v1, u0:u1].reshape(-1)

    for key in keys:
        if key not in preds:
            continue
        patch = preds[key][frame, v0:v1, u0:u1].reshape(-1, 3).astype(np.float64)
        finite = np.isfinite(patch).all(axis=1)
        keep = finite
        if conf is not None and conf.shape[0] == patch.shape[0]:
            thr = conf_frac * float(conf.max() + 1e-9)
            keep = finite & (conf >= thr)
            if keep.sum() < 3:               # relax if confidence is too strict
                keep = finite
        if keep.sum() == 0:
            continue
        return np.median(patch[keep], axis=0)
    return None


# ── Scale estimation ─────────────────────────────────────────────────────────

@dataclass
class Anchor:
    """One known-size measurement: endpoints p0=(frame,u,v), p1=(frame,u,v)."""
    name: str
    p0: tuple                 # (frame, u, v)
    p1: tuple                 # (frame, u, v)
    metric_length: float      # real-world length between the endpoints, meters


@dataclass
class ScaleEstimate:
    anchor: str
    metric_length: float
    vggt_length: float
    scale: float              # meters per VGGT unit
    endpoints_world: list = field(default_factory=list)


def scale_from_anchor(preds: dict, anchor: Anchor, win: int = 3) -> ScaleEstimate:
    w0 = pixel_to_world(preds, anchor.p0[0], anchor.p0[1], anchor.p0[2], win=win)
    w1 = pixel_to_world(preds, anchor.p1[0], anchor.p1[1], anchor.p1[2], win=win)
    if w0 is None or w1 is None:
        raise ValueError(f"anchor '{anchor.name}': no valid 3D point at an endpoint")
    d = float(np.linalg.norm(w1 - w0))
    if d < 1e-9:
        raise ValueError(f"anchor '{anchor.name}': endpoints coincide in VGGT space")
    return ScaleEstimate(
        anchor=anchor.name,
        metric_length=anchor.metric_length,
        vggt_length=d,
        scale=anchor.metric_length / d,
        endpoints_world=[w0.tolist(), w1.tolist()],
    )


def aggregate_scales(estimates: list) -> dict:
    """Combine multiple anchor estimates; report spread as a self-consistency signal."""
    scales = np.array([e.scale for e in estimates], dtype=np.float64)
    median = float(np.median(scales))
    spread = float(scales.max() - scales.min()) if len(scales) > 1 else 0.0
    rel_spread = spread / (median + 1e-12)
    # First-order volume-error implication of the scale disagreement.
    vol_rel_spread = 3.0 * rel_spread
    return {
        "n_anchors": len(estimates),
        "scale_median": median,
        "scale_mean": float(scales.mean()),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "scale_rel_spread": rel_spread,          # (max-min)/median, length-level
        "implied_volume_rel_spread": vol_rel_spread,   # ~3x, volume-level
        "per_anchor": [asdict(e) for e in estimates],
    }


# ── Metric scene summary ─────────────────────────────────────────────────────

def metric_scene_summary(preds: dict, scale: float, conf_frac: float = 0.1) -> dict:
    """Report the scene's metric extent after applying `scale`."""
    wp = preds[PRED_PRIMARY].reshape(-1, 3).astype(np.float64)
    finite = np.isfinite(wp).all(axis=1)
    wp = wp[finite]
    size_vggt = wp.max(0) - wp.min(0)
    size_m = size_vggt * scale
    return {
        "scale_m_per_unit": scale,
        "bbox_vggt": size_vggt.tolist(),
        "bbox_meters": size_m.tolist(),
        "diag_meters": float(np.linalg.norm(size_m)),
    }


# ── Interactive endpoint picker (optional) ───────────────────────────────────

def pick_anchor(preds: dict, frame: int):
    """
    Open a matplotlib window on the model-resolution frame; click the two
    endpoints of the known-size reference. Returns [(u,v),(u,v)].
    Coordinates index world_points directly (no resize mapping needed).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    img = _display_frame(preds, frame)
    fig, ax = plt.subplots(figsize=(10, img.shape[0] / img.shape[1] * 10))
    ax.imshow(img)
    ax.set_title(f"frame {frame}: click the TWO endpoints of the known-size object")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    return [(int(round(u)), int(round(v))) for (u, v) in pts]


def save_gridded_frame(preds: dict, frame: int, out_path: str, step: int = 50):
    """
    Save a model-resolution frame annotated with a pixel grid so endpoint
    coordinates can be read off without a GUI. Coordinates index world_points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = _display_frame(preds, frame)
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(W / 60, H / 60), dpi=120)
    ax.imshow(img)
    for x in range(0, W, step):
        ax.axvline(x, color="cyan", lw=0.4, alpha=0.6)
        ax.text(x + 1, 8, str(x), color="cyan", fontsize=6)
    for y in range(0, H, step):
        ax.axhline(y, color="cyan", lw=0.4, alpha=0.6)
        ax.text(1, y + 8, str(y), color="cyan", fontsize=6)
    ax.set_title(f"frame {frame}  ({W}x{H})  u=horizontal, v=vertical")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_anchor(spec: str) -> Anchor:
    """
    Parse 'name:f0,u0,v0:f1,u1,v1:length_m' into an Anchor.
    Example: person:0,402,70:0,402,150:1.70
    """
    name, e0, e1, L = spec.split(":")
    p0 = tuple(int(x) for x in e0.split(","))
    p1 = tuple(int(x) for x in e1.split(","))
    return Anchor(name=name, p0=p0, p1=p1, metric_length=float(L))


def main():
    ap = argparse.ArgumentParser(description="Metric scale calibration from a known-size reference.")
    ap.add_argument("workspace", help="workspace dir or predictions.npz path")
    ap.add_argument("--anchor", action="append", default=[],
                    help="name:f,u,v:f,u,v:length_m  (repeatable for consistency check)")
    ap.add_argument("--win", type=int, default=3, help="half-window for robust pixel sampling")
    ap.add_argument("--grid-frame", type=int, default=None,
                    help="save a gridded frame PNG for reading endpoint pixels, then exit")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    args = ap.parse_args()

    preds = load_predictions(args.workspace)

    if args.grid_frame is not None:
        ws = args.workspace if os.path.isdir(args.workspace) else os.path.dirname(args.workspace)
        out = os.path.join(ws, f"grid_frame_{args.grid_frame}.png")
        save_gridded_frame(preds, args.grid_frame, out)
        print(f"wrote {out}")
        return

    if not args.anchor:
        ap.error("provide at least one --anchor (or use --grid-frame to read pixels first)")

    estimates = [scale_from_anchor(preds, _parse_anchor(s), win=args.win) for s in args.anchor]
    agg = aggregate_scales(estimates)
    summary = metric_scene_summary(preds, agg["scale_median"])
    result = {"workspace": args.workspace, "aggregate": agg, "metric_scene": summary}

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n── interpretation ─────────────────────────────")
    print(f"scale  = {agg['scale_median']:.5f} m / VGGT-unit  ({agg['n_anchors']} anchor(s))")
    print(f"scene  ≈ {summary['bbox_meters'][0]:.2f} x {summary['bbox_meters'][1]:.2f} "
          f"x {summary['bbox_meters'][2]:.2f} m")
    if agg["n_anchors"] > 1:
        print(f"scale spread = {agg['scale_rel_spread']*100:.1f}%  "
              f"→ implied volume spread ≈ {agg['implied_volume_rel_spread']*100:.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
