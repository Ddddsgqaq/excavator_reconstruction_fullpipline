"""
diagnose_flattening.py — Root-cause analysis for the vertical under-reconstruction
found by vertical_fidelity_study.py. Three decisive probes on the excavator
(the reliable large/high-conf object):

  P1. WHICH HEAD: aspect_HW from `world_points` (pointmap head, direct regression)
      vs `world_points_from_depth` (depth head back-projected through the camera).
      If one is much flatter, that head owns the compression.

  P2. IS IT ALONG DEPTH: transform the object's masked points into the frame-0
      camera frame and compare extent along the camera DEPTH axis (Z) vs the
      image-plane axes (X,Y). In an oblique aerial view, world-up projects
      heavily onto camera-Z, so if depth-extent is suppressed relative to
      image-plane extent, the compression is a depth-range (relief-smoothing)
      effect, not a segmentation/gravity artifact.

  P3. CONFIDENCE/RANGE: split the object's points by confidence quartile and by
      depth, and report vertical extent per bucket — rules out a low-confidence
      filtering artifact (if even the top-confidence, nearest points are flat,
      the effect is intrinsic).

Run in `yoloe` env from WS/yoloe dir (needs mobileclip_blt.pt).
"""
from __future__ import annotations
import os, sys, json
import numpy as np

VYDIR = "/home/maomaoyu/WS/vggt_yoloe"
sys.path.insert(0, VYDIR)
import scale_calibration as sc
import gravity_alignment as ga
from vertical_fidelity_study import yoloe_masks


def aligned_aspect(pts_world, R, conf=None, conf_idx=None):
    p = ga.apply_alignment_to_points(pts_world, R)
    lo = np.percentile(p, 2, 0); hi = np.percentile(p, 98, 0)
    ext = hi - lo
    horiz = sorted([float(ext[0]), float(ext[2])])
    return float(ext[1]) / (horiz[0] + 1e-9), ext


def camera_frame_extents(pts_world, extr_f):
    """Extent of points along camera axes (X=right, Y=down, Z=depth)."""
    R = extr_f[:3, :3]; t = extr_f[:3, 3]
    cam = pts_world @ R.T + t
    lo = np.percentile(cam, 2, 0); hi = np.percentile(cam, 98, 0)
    return (hi - lo)  # [imgX, imgY, depthZ]


def diagnose(model, ws, conf=0.25):
    preds = sc.load_predictions(ws)
    g = ga.estimate_gravity(preds["extrinsic"], preds["world_points"],
                            conf=preds.get("world_points_conf"))
    masks = yoloe_masks(model, preds, 0, ["excavator", "person"], conf=conf)
    name = os.path.basename(ws.rstrip("/"))[:22]
    for cls in ("excavator", "person"):
        if cls not in masks:
            continue
        cc, m = masks[cls]
        wp = preds["world_points"][0]
        wpd = preds["world_points_from_depth"][0]
        cfd = preds["world_points_conf"][0]
        valid = m & np.isfinite(wp).all(2) & np.isfinite(wpd).all(2)
        if valid.sum() < 30:
            continue
        confv = cfd[valid]
        ptsP = wp[valid]; ptsD = wpd[valid]

        # P1: which head
        aP, _ = aligned_aspect(ptsP, g.R_align)
        aD, _ = aligned_aspect(ptsD, g.R_align)

        # P2: along-depth vs image-plane (pointmap head, frame-0 camera)
        ce = camera_frame_extents(ptsP, preds["extrinsic"][0])
        imgplane = np.hypot(ce[0], ce[1]); depth = ce[2]
        depth_ratio = depth / (imgplane + 1e-9)

        # P3: vertical extent for top-confidence quartile only
        thr = np.percentile(confv, 75)
        hi_idx = confv >= thr
        aP_hi, _ = aligned_aspect(ptsP[hi_idx], g.R_align) if hi_idx.sum() > 20 else (float('nan'), None)

        print(f"{name:22s} {cls:9s} conf={cc:.2f} n={int(valid.sum()):5d}")
        print(f"   P1 aspect_HW: pointmap={aP:.2f}  depth_branch={aD:.2f}   (canonical {'1.3' if cls=='excavator' else '3.8'})")
        print(f"   P2 cam-extent imgplane={imgplane:.3f} depth={depth:.3f}  depth/imgplane={depth_ratio:.2f}")
        print(f"   P3 aspect_HW top-25%-conf={aP_hi:.2f}  (vs all {aP:.2f})")


if __name__ == "__main__":
    from ultralytics import YOLOE
    from huggingface_hub import hf_hub_download
    model = YOLOE(hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg.pt"))
    wss = sys.argv[1:] or [f"{VYDIR}/workspaces/session_20260611_162643_869764"]
    for ws in wss:
        try:
            diagnose(model, ws)
        except Exception as e:
            print(f"{ws}: ERROR {e}")
