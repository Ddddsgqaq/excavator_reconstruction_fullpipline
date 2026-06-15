"""
vertical_gauge_test.py  (experiment E0)

Decisive test of WHAT KIND of distortion the ~8x vertical compression is, so we
know whether the correction is "multiply by one scalar" (global affine gauge) or
"learn/fit a field" (depth-dependent projective). For each known-height control
object (excavator ~3.0 m, person ~1.7 m) — both VERTICAL in reality — we measure
how VGGT reconstructs that vertical extent in the gravity-aligned frame:

  * principal axis of the object's hi-conf points, and its TIP ANGLE from vertical
  * the angle between that tip direction and the camera VIEWING RAY
    (foreshortening hypothesis: vertical structure gets laid down ALONG the ray)
  * apparent vertical V_y = extent * |axis_Y|  (what a DEM would see as height)
  * recovery factor k = real_height / V_y
  * depth-dependence: split the excavator into near/far halves (by camera depth)
    and recompute k — if k differs, the gauge is projective (depth-varying), not affine.

Outputs: vertical_gauge_test.png, vertical_gauge_result.json.  Run in `vggt` env.
"""
import sys, os, json
sys.path.insert(0, "/home/maomaoyu/WS/vggt_yoloe")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scale_calibration as sc, gravity_alignment as ga

W = sys.argv[1] if len(sys.argv) > 1 else "workspaces/session_20260611_162643_869764"
CONTROLS = {"excavator": 3.0, "person": 1.7}   # real heights (m), provisional

preds = sc.load_predictions(W)
conf = preds["world_points_conf"]
scale = json.load(open(f"{W}/scale_result.json"))["metric_scene"]["scale_m_per_unit"]
g = ga.estimate_gravity(preds["extrinsic"], preds["world_points"], conf=conf)
al = ga.apply_alignment_to_points(preds["world_points"], g.R_align)   # Y=up
masks = np.load(f"{W}/masks_f0.npz")

# camera-0 viewing ray (world forward) in the aligned frame
R_wc0 = preds["extrinsic"][0, :3, :3].T            # cam->world rotation
ray_world = R_wc0 @ np.array([0.0, 0.0, 1.0])      # camera looks +Z in its own frame
ray_al = g.R_align @ ray_world
ray_al = ray_al / np.linalg.norm(ray_al)
cam_depth_dir_world = ray_world / np.linalg.norm(ray_world)


def principal(pts):
    c = pts.mean(0)
    _, sv, Vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = Vt[0]
    if ax[1] < 0:
        ax = -ax                       # orient "up-ish"
    # extent along axis = robust span of projection
    proj = (pts - c) @ ax
    extent = np.percentile(proj, 98) - np.percentile(proj, 2)
    return c, ax, extent, sv / sv[0]


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(u, v)), 0, 1))))


results = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

for i, (cls, Hreal) in enumerate(CONTROLS.items()):
    m = masks[cls] & np.isfinite(al[0]).all(2)
    P = al[0][m]; C = conf[0][m]
    hi = C >= np.percentile(C, 75)
    p = P[hi]
    c, axis, extent_u, svr = principal(p)
    extent_m = extent_u * scale
    tip = ang(axis, np.array([0, 1.0, 0]))            # deg from vertical
    ray_ang = ang(axis, ray_al)                        # deg between axis and viewing ray
    # apparent vertical = DIRECT Y-extent (what a DEM sees), robust to PCA picking a horizontal axis
    Vy_m = (np.percentile(p[:, 1], 98) - np.percentile(p[:, 1], 2)) * scale
    k_vert = Hreal / max(Vy_m, 1e-6)
    k_len = Hreal / max(extent_m, 1e-6)                # if whole length were the height
    results[cls] = dict(
        real_height_m=Hreal, n_pts=int(hi.sum()),
        recon_axis_aligned=[round(float(x), 3) for x in axis],
        recon_extent_m=round(extent_m, 3),
        apparent_vertical_Vy_m=round(Vy_m, 3),
        tip_from_vertical_deg=round(tip, 1),
        axis_vs_cameraray_deg=round(ray_ang, 1),
        k_apparent_vertical=round(k_vert, 2),
        k_if_full_length=round(k_len, 2),
        svd_ratio=[round(float(x), 2) for x in svr],
    )

    # side-view panel: Y vs along-ray-horizontal; draw real-vertical, recon-axis, ray
    zc = (p[:, 2] - c[2]) * scale; yc = (p[:, 1] - c[1]) * scale
    axes[i].scatter(zc, yc, s=6, c=C[hi], cmap="viridis", alpha=.7)
    axes[i].arrow(0, 0, 0, Hreal*0.5, color="red", width=0.01, head_width=0.06,
                  length_includes_head=True, label="real vertical")
    a2 = axis * extent_m * 0.5
    axes[i].arrow(0, 0, a2[2], a2[1], color="black", width=0.01, head_width=0.06,
                  length_includes_head=True, label="recon principal axis")
    r2 = ray_al * extent_m * 0.5
    axes[i].arrow(0, 0, r2[2], r2[1], color="orange", width=0.008, head_width=0.05,
                  ls="--", length_includes_head=True, label="camera ray")
    axes[i].set_aspect("equal"); axes[i].set_xlabel("Z aligned (m)"); axes[i].set_ylabel("Y up (m)")
    axes[i].set_title(f"{cls}: tip {tip:.0f}° from vertical · axis vs ray {ray_ang:.0f}°\n"
                      f"apparent height {Vy_m:.2f}m vs real {Hreal}m (k={k_vert:.1f}×)", fontsize=10)
    axes[i].legend(fontsize=7, loc="upper left")

# depth-dependence on the excavator: near vs far by camera depth (world_points_from_depth z in cam frame proxy = aligned along ray)
m = masks["excavator"] & np.isfinite(al[0]).all(2)
P = al[0][m]; C = conf[0][m]; hi = C >= np.percentile(C, 75)
p = P[hi]
along_ray = (p - p.mean(0)) @ ray_al
near = along_ray < np.median(along_ray); far = ~near
depthdep = {}
for tag, sel in [("near", near), ("far", far)]:
    if sel.sum() < 30:
        depthdep[tag] = None; continue
    ps = p[sel]
    Vy = (np.percentile(ps[:, 1], 98) - np.percentile(ps[:, 1], 2)) * scale   # direct Y-extent
    depthdep[tag] = dict(k=round(3.0 / max(Vy, 1e-6), 2), Vy_m=round(Vy, 3), n=int(sel.sum()))

kvals = [results[c]["k_apparent_vertical"] for c in CONTROLS]
# panel 3: k comparison — cross-object + excavator near/far depth halves (English labels)
bar_lbl = ["excavator", "person", "exc-near", "exc-far"]
bar_val = [kvals[0], kvals[1],
           depthdep["near"]["k"] if depthdep.get("near") else 0,
           depthdep["far"]["k"] if depthdep.get("far") else 0]
bar_col = ["#2a9d8f", "#e9c46a", "#3a86b8", "#264653"]
axes[2].bar(bar_lbl, bar_val, color=bar_col)
for j, v in enumerate(bar_val):
    axes[2].text(j, v + 0.2, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
axes[2].set_ylabel("k = real height / apparent height")
axes[2].set_title("near vs far almost equal -> depth-independent (affine)\n"
                  "differs across objects -> single scalar insufficient", fontsize=10)
axes[2].axhline(np.mean(kvals), color="gray", ls=":", lw=1)
consistency = abs(kvals[0]-kvals[1]) / np.mean(kvals)
verdict = ("AFFINE-like (single ~k may suffice)" if consistency < 0.25 else
           "k differs across objects → not a single scalar")
dd = depthdep.get("near"), depthdep.get("far")
if dd[0] and dd[1]:
    dr = abs(dd[0]["k"]-dd[1]["k"])/np.mean([dd[0]["k"], dd[1]["k"]])
    depth_verdict = ("depth-INDEPENDENT k (affine)" if dr < 0.25 else
                     "depth-DEPENDENT k (projective → needs field/SL(4))")
else:
    dr = None; depth_verdict = "insufficient points for depth split"

fig.suptitle(f"E0 vertical-gauge test — k(exc)={kvals[0]} k(person)={kvals[1]} "
             f"[{verdict}] · depth: {depth_verdict}", fontsize=12)
fig.tight_layout()
out_png = f"{W}/vertical_gauge_test.png"; fig.savefig(out_png, dpi=115); print("wrote", out_png)

result = dict(workspace=W, scale_m_per_unit=scale,
              camera_ray_aligned=[round(float(x), 3) for x in ray_al],
              controls=results,
              cross_object_k_consistency=round(float(consistency), 3),
              cross_object_verdict=verdict,
              excavator_depth_dependence=depthdep,
              depth_dependence_rel=None if dr is None else round(float(dr), 3),
              depth_verdict=depth_verdict)
json.dump(result, open(f"{W}/vertical_gauge_result.json", "w"), indent=1)
print(json.dumps(result, indent=1, ensure_ascii=False))
