"""
vertical_fidelity_study.py — Test whether VGGT systematically under-reconstructs
VERTICAL relief, across many scenes, using a scale-independent, terrain-independent
probe: the reconstructed aspect ratio of known-canonical-aspect objects.

Why this probe: a low full-cloud vertical/horizontal ratio could just mean the
ground is genuinely flat. But an *upright person* has a fixed real aspect
(height/width ~ 3.5-4). If, across scenes, segmented people reconstruct to
aspect << that (i.e. reconstructed lying flat), vertical compression is real
and universal — independent of the unknown global scale and of the terrain.

Per detected object we report:
  Y_ext   = robust vertical extent (gravity-up axis, p2-p98)
  W_ext   = robust extent along the SMALLER horizontal axis (true-width proxy)
  L_ext   = robust extent along the LARGER horizontal axis (smear direction)
  aspect_HW = Y_ext / W_ext      (upright person ~3.8; truck/car lower)
Plus per-scene gravity source + trajectory degeneracy, so bad-gravity scenes
can be separated from genuine compression.

Run in the `yoloe` conda env, from the WS/yoloe dir (needs mobileclip_blt.pt).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

VYDIR = "/home/maomaoyu/WS/vggt_yoloe"
sys.path.insert(0, VYDIR)
import scale_calibration as sc
import gravity_alignment as ga

# canonical upright aspect height/width (for reference, not used to scale)
CANONICAL_HW = {"person": 3.8, "excavator": 1.3, "car": 0.45, "truck": 0.5}


def yoloe_masks(model, preds, frame, classes, conf=0.25):
    img = sc._display_frame(preds, frame)
    model.set_classes(classes, model.get_text_pe(classes))
    r = model.predict(source=img[:, :, ::-1], imgsz=518, conf=conf,
                      retina_masks=True, verbose=False)[0]
    out = {}
    if r.masks is None:
        return out
    md = r.masks.data.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    cf = r.boxes.conf.cpu().numpy()
    for i, (c, cc) in enumerate(zip(cls, cf)):
        name = classes[c]
        m = md[i] > 0.5
        if name not in out or m.sum() > out[name][1].sum():
            out[name] = (float(cc), m)
    return out


def object_extents(preds, frame, mask, R):
    wp = preds["world_points"][frame]
    cfd = preds["world_points_conf"][frame]
    m = mask & np.isfinite(wp).all(2) & (cfd >= 0.1 * float(cfd[mask].max() + 1e-9))
    if m.sum() < 10:
        return None
    p = ga.apply_alignment_to_points(wp[m], R)
    lo = np.percentile(p, 2, 0); hi = np.percentile(p, 98, 0)
    ext = hi - lo
    Y = float(ext[1])
    horiz = sorted([float(ext[0]), float(ext[2])])
    return {"Y_ext": Y, "W_ext": horiz[0], "L_ext": horiz[1], "n": int(m.sum())}


def study_scene(model, ws, classes, conf=0.25):
    preds = sc.load_predictions(ws)
    g = ga.estimate_gravity(preds["extrinsic"], preds["world_points"],
                            conf=preds.get("world_points_conf"))
    traj = g.debug.get("trajectory", {})
    masks = yoloe_masks(model, preds, 0, classes, conf=conf)
    rows = []
    for name, (cc, m) in masks.items():
        ext = object_extents(preds, 0, m, g.R_align)
        if ext is None:
            continue
        aspect = ext["Y_ext"] / (ext["W_ext"] + 1e-9)
        rows.append({
            "scene": os.path.basename(ws.rstrip("/")), "class": name, "conf": round(cc, 2),
            "mask_px": int(m.sum()), **{k: round(v, 4) for k, v in ext.items() if k != "n"},
            "n_pts": ext["n"], "aspect_HW": round(aspect, 3),
            "canonical_HW": CANONICAL_HW.get(name),
            "grav_src": g.source,
            "traj_ratio": round(traj.get("second_first_ratio", -1), 3) if isinstance(traj, dict) else -1,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspaces", nargs="+", help="workspace dirs to study")
    ap.add_argument("--classes", default="person,excavator,car,truck")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=os.path.join(VYDIR, "vertical_fidelity_results.json"))
    args = ap.parse_args()

    from ultralytics import YOLOE
    from huggingface_hub import hf_hub_download
    model = YOLOE(hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg.pt"))
    classes = args.classes.split(",")

    all_rows = []
    for ws in args.workspaces:
        try:
            rows = study_scene(model, ws, classes, conf=args.conf)
            all_rows.extend(rows)
            for r in rows:
                print(f"{r['scene'][:24]:24s} {r['class']:9s} conf={r['conf']:.2f} px={r['mask_px']:5d} "
                      f"Y={r['Y_ext']:.3f} W={r['W_ext']:.3f} L={r['L_ext']:.3f} "
                      f"aspectHW={r['aspect_HW']:.2f} (canon {r['canonical_HW']}) grav={r['grav_src']}")
        except Exception as e:
            print(f"{os.path.basename(ws.rstrip('/'))}: ERROR {e}")
    with open(args.out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n{len(all_rows)} object rows → {args.out}")


if __name__ == "__main__":
    main()
