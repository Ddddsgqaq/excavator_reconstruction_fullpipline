"""
viz_halo.py — Visualize the low-confidence horizontal halo around an object,
to confirm it (a) lives at the object silhouette / depth discontinuities in
image space and (b) sprays horizontally onto the ground plane in 3D.
Outputs a 4-panel PNG.
"""
import sys, os
sys.path.insert(0, "/home/maomaoyu/WS/vggt_yoloe")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scale_calibration as sc, gravity_alignment as ga

W = sys.argv[1] if len(sys.argv) > 1 else "workspaces/session_20260611_162643_869764"
preds = sc.load_predictions(W)
masks = np.load(f"{W}/masks_f0.npz")
g = ga.estimate_gravity(preds["extrinsic"], preds["world_points"], conf=preds.get("world_points_conf"))
cls = "excavator"
m = masks[cls]
img = sc._display_frame(preds, 0)
wp = preds["world_points"][0]; cfd = preds["world_points_conf"][0]
depth = preds["depth"][0, :, :, 0]

valid = m & np.isfinite(wp).all(2)
vs, us = np.where(valid)
conf = cfd[valid]
pts = ga.apply_alignment_to_points(wp[valid], g.R_align)
cthr = np.percentile(conf, 50)                 # split hi/lo conf
lo = conf < cthr

# depth-gradient at each masked pixel
gy, gx = np.gradient(depth)
dgrad = np.hypot(gx, gy)[valid]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# A: image space — low-conf pixels in red over the crop
u0, u1, v0, v1 = us.min()-10, us.max()+10, vs.min()-10, vs.max()+10
ax[0,0].imshow(img[v0:v1, u0:u1])
ax[0,0].scatter(us[lo]-u0, vs[lo]-v0, s=3, c="red", alpha=0.6, label="low-conf (bottom 50%)")
ax[0,0].scatter(us[~lo]-u0, vs[~lo]-v0, s=2, c="lime", alpha=0.3, label="high-conf")
ax[0,0].set_title("A. image space: low-conf pixels hug the silhouette"); ax[0,0].legend(loc="upper right", fontsize=8)

# B: 3D side view (horizontal X vs vertical Y), colored by conf
sc_b = ax[0,1].scatter(pts[:,0], pts[:,1], s=3, c=conf, cmap="viridis")
ax[0,1].scatter(pts[lo,0], pts[lo,1], s=4, facecolors="none", edgecolors="red", alpha=0.25)
ax[0,1].set_xlabel("X aligned (horizontal)"); ax[0,1].set_ylabel("Y aligned (UP)")
ax[0,1].set_aspect("equal"); ax[0,1].set_title("B. side view: low-conf (red rings) skirt out horizontally near ground")
plt.colorbar(sc_b, ax=ax[0,1], label="conf")

# C: 3D top view (X vs Z), colored by conf
sc_c = ax[1,0].scatter(pts[:,0], pts[:,2], s=3, c=conf, cmap="viridis")
ax[1,0].scatter(pts[lo,0], pts[lo,2], s=4, facecolors="none", edgecolors="red", alpha=0.25)
ax[1,0].set_xlabel("X aligned"); ax[1,0].set_ylabel("Z aligned"); ax[1,0].set_aspect("equal")
ax[1,0].set_title("C. top view: low-conf points spread the horizontal footprint")

# D: confidence vs depth-gradient
ax[1,1].scatter(dgrad, conf, s=3, alpha=0.3)
ax[1,1].set_xlabel("local depth gradient |∇depth|"); ax[1,1].set_ylabel("confidence")
r = np.corrcoef(dgrad, conf)[0,1]
ax[1,1].set_title(f"D. low conf ↔ high depth-gradient (corr={r:.2f})")

fig.suptitle(f"Low-confidence horizontal halo — {cls} — {os.path.basename(W.rstrip('/'))}", fontsize=13)
fig.tight_layout()
out = f"{W}/halo_viz.png"; fig.savefig(out, dpi=110); print("wrote", out)
# numeric: vertical extent contribution of lo vs hi conf
print(f"hi-conf Y-extent(p2-98)={np.percentile(pts[~lo,1],98)-np.percentile(pts[~lo,1],2):.3f}  "
      f"X-extent={np.percentile(pts[~lo,0],98)-np.percentile(pts[~lo,0],2):.3f}")
print(f"lo-conf Y-extent={np.percentile(pts[lo,1],98)-np.percentile(pts[lo,1],2):.3f}  "
      f"X-extent={np.percentile(pts[lo,0],98)-np.percentile(pts[lo,0],2):.3f}")
print(f"depth-grad: lo-conf median={np.median(dgrad[lo]):.4f}  hi-conf median={np.median(dgrad[~lo]):.4f}")
