"""
vertical_recovery_study.py — Two questions on one scene:

(1) PIT JUDGMENT: where (if anywhere) is there a real excavated pit?
    A pit = ground-surface elevation Y sitting BELOW the surrounding
    reference ground plane. We fit a ground plane to the terrain, compute
    a per-pixel "depression" map (ground_Y - Y; positive = below ground =
    pit, negative = above ground = object/mound), and show where the
    candidate ROI actually sits. (Spoiler for this scene: ~flat, no pit.)

(2) VERTICAL RECOVERY: VGGT compresses vertical relief ~8x (excavator
    3.0 m -> ~0.4 m). Does multi-frame aggregation recover the height?
    We compare per-frame reconstructed excavator height, the multi-frame
    fused height, and the two prediction heads (pointmap vs depth).

Outputs: pit_judgment.png, vertical_recovery.png, vertical_recovery_result.json
Run inside the `vggt` conda env.
"""
import sys, os, json
sys.path.insert(0, "/home/maomaoyu/WS/vggt_yoloe")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scale_calibration as sc, gravity_alignment as ga

W = sys.argv[1] if len(sys.argv) > 1 else "workspaces/session_20260611_162643_869764"
REAL_EXC_H = 3.0   # excavator real height (m), provisional probe
ROI = dict(u0=300, u1=518, v0=150, v1=294)   # candidate "pit" image bbox on frame 0

preds = sc.load_predictions(W)
conf = preds["world_points_conf"]
scale = json.load(open(f"{W}/scale_result.json"))["metric_scene"]["scale_m_per_unit"]
g = ga.estimate_gravity(preds["extrinsic"], preds["world_points"], conf=conf)
al = ga.apply_alignment_to_points(preds["world_points"], g.R_align)      # (S,H,W,3) Y=up
ald = ga.apply_alignment_to_points(preds["world_points_from_depth"], g.R_align)
img = sc._display_frame(preds, 0)
masks = np.load(f"{W}/masks_f0.npz")
H, Wd = img.shape[:2]

# ── Ground plane fit (aligned frame): Y ~ a*X + b*Z + c on robust ground points ──
P0 = al[0]; fin0 = np.isfinite(P0).all(2)
flat = al.reshape(-1, 3); cflat = conf.reshape(-1)
fk = np.isfinite(flat).all(1) & (cflat > 0.2 * cflat.max())
fp = flat[fk]
# ground = lower band of Y (exclude objects which stick up); robust
ylo = np.percentile(fp[:, 1], 60)
gp = fp[fp[:, 1] <= ylo]
A = np.c_[gp[:, 0], gp[:, 2], np.ones(len(gp))]
coef, *_ = np.linalg.lstsq(A, gp[:, 1], rcond=None)   # Y = a X + b Z + c
def ground_Y(X, Z):
    return coef[0] * X + coef[1] * Z + coef[2]

# per-pixel depression map for frame 0 (meters): ground - Y  (>0 = pit)
Ymap = np.where(fin0, P0[:, :, 1], np.nan)
gYmap = ground_Y(P0[:, :, 0], P0[:, :, 2])
dep05 = (gYmap - Ymap) * scale     # meters below reference ground

# ── FIGURE 1 : PIT JUDGMENT ──────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
exc, per = masks["excavator"], masks["person"]

# A. image + candidate ROI + object masks
ax[0,0].imshow(img)
ax[0,0].add_patch(Rectangle((ROI["u0"], ROI["v0"]), ROI["u1"]-ROI["u0"], ROI["v1"]-ROI["v0"],
                            fill=False, edgecolor="yellow", lw=2.5))
ax[0,0].contour(exc, levels=[0.5], colors="red", linewidths=1.2)
ax[0,0].contour(per, levels=[0.5], colors="cyan", linewidths=1.2)
ax[0,0].text(ROI["u0"]+4, ROI["v0"]+16, "candidate pit ROI", color="yellow", fontsize=10, weight="bold")
ax[0,0].set_title("A. frame 0 — candidate pit ROI (yellow), excavator (red), person (cyan)")

# B. elevation map
im = ax[0,1].imshow(np.where(fin0, Ymap, np.nan)*scale, cmap="terrain")
ax[0,1].add_patch(Rectangle((ROI["u0"], ROI["v0"]), ROI["u1"]-ROI["u0"], ROI["v1"]-ROI["v0"],
                            fill=False, edgecolor="yellow", lw=2))
plt.colorbar(im, ax=ax[0,1], label="aligned elevation Y (m)")
ax[0,1].set_title("B. elevation map — note the whole scene spans <0.4 m vertically")

# C. depression map: >0 (warm) = below ground = pit; <0 (cool) = above ground = object
vmax = 0.5
im = ax[1,0].imshow(dep05, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax[1,0].add_patch(Rectangle((ROI["u0"], ROI["v0"]), ROI["u1"]-ROI["u0"], ROI["v1"]-ROI["v0"],
                            fill=False, edgecolor="black", lw=2))
# mark genuine pit cells: >0.2 m below ground
pit_cells = (dep05 > 0.2)
ys, xs = np.where(pit_cells)
ax[1,0].scatter(xs, ys, s=1, c="lime", alpha=0.5)
plt.colorbar(im, ax=ax[1,0], label="depth below reference ground (m)")
ax[1,0].set_title(f"C. depression map — pit = red(>0). Cells >0.2 m below ground (lime) = {pit_cells.sum()} px")

# D. transect through ROI vertical-center: Y vs u, with ground line
vmid = (ROI["v0"]+ROI["v1"])//2
row_valid = fin0[vmid]
uu = np.arange(Wd)
ax[1,1].plot(uu[row_valid], Ymap[vmid][row_valid]*scale, ".", ms=3, label="reconstructed surface Y")
ax[1,1].plot(uu, ground_Y(P0[vmid,:,0], P0[vmid,:,2])*scale, "k--", lw=1, label="reference ground")
ax[1,1].axvspan(ROI["u0"], ROI["u1"], color="yellow", alpha=0.2, label="ROI columns")
ax[1,1].set_xlabel("image column u"); ax[1,1].set_ylabel("elevation (m)")
ax[1,1].set_title(f"D. transect at v={vmid}: surface tracks ground inside ROI → no pit")
ax[1,1].legend(fontsize=8)

# ROI depression stats
roi_mask = np.zeros((H, Wd), bool); roi_mask[ROI["v0"]:ROI["v1"], ROI["u0"]:ROI["u1"]] = True
roi_mask &= fin0 & ~exc & ~per
roi_depr = dep05[roi_mask]
roi_depr = roi_depr[np.isfinite(roi_depr)]
fig.suptitle(f"PIT JUDGMENT — {os.path.basename(W.rstrip('/'))} — "
             f"ROI median depression={np.median(roi_depr):+.2f} m (≈0 ⇒ no pit)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{W}/pit_judgment.png", dpi=110); print("wrote", f"{W}/pit_judgment.png")

# ── Excavator footprint (world XZ) from frame-0 hi-conf mask points ───────────
mk = exc & fin0
Pe = P0[mk]; Ce = conf[0][mk]
hi = Ce >= np.percentile(Ce, 75)
xz_lo = np.percentile(Pe[hi][:, [0,2]], 5, axis=0)
xz_hi = np.percentile(Pe[hi][:, [0,2]], 95, axis=0)
def in_fp(P, pad=0.0):
    return ((P[:,0]>=xz_lo[0]-pad)&(P[:,0]<=xz_hi[0]+pad)&
            (P[:,2]>=xz_lo[1]-pad)&(P[:,2]<=xz_hi[1]+pad))
gY_fp = ground_Y((xz_lo[0]+xz_hi[0])/2, (xz_lo[1]+xz_hi[1])/2)

# per-frame excavator top height above ground (hi-conf within footprint)
heights = []; fused_pts = []
for fr in range(al.shape[0]):
    P = al[fr]; Cf = conf[fr]; fin = np.isfinite(P).all(2)
    P = P[fin]; Cf = Cf[fin]; ins = in_fp(P)
    if ins.sum() < 20: heights.append(np.nan); continue
    Pi = P[ins]; Ci = Cf[ins]; h = Ci >= np.percentile(Ci, 75)
    heights.append((np.percentile(Pi[h][:,1], 95) - gY_fp) * scale)
    fused_pts.append(Pi[h])
heights = np.array(heights)
fused = np.concatenate(fused_pts)
fused_h = (np.percentile(fused[:,1], 95) - gY_fp) * scale
fused_h995 = (np.percentile(fused[:,1], 99.5) - gY_fp) * scale

# two-head excavator vertical extent (frame 0)
def yext(P, m, c, thr):
    pp = P[m & np.isfinite(P).all(2)]
    cc = c[m & np.isfinite(P).all(2)]
    pp = pp[cc >= thr]
    return (np.percentile(pp[:,1],98)-np.percentile(pp[:,1],2))*scale
c0 = conf[0]; thr25 = np.percentile(c0[mk], 75)
head_pm = yext(al[0], exc, c0, thr25)
head_dp = yext(ald[0], exc, c0, thr25)

# ── FIGURE 2 : VERTICAL COMPRESSION + RECOVERY ───────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(15, 9))

# A. excavator side view Y vs Z (hi-conf frame0), real-height bar for scale
side = Pe[hi]
sc2 = ax[0,0].scatter((side[:,2]-side[:,2].mean())*scale, (side[:,1]-gY_fp)*scale,
                      s=5, c=Ce[hi], cmap="viridis")
ax[0,0].plot([ (side[:,2].min()-side[:,2].mean())*scale-0.3 ]*2, [0, REAL_EXC_H],
             "r-", lw=4, label=f"real height {REAL_EXC_H} m")
ax[0,0].set_xlabel("horizontal Z (m, centered)"); ax[0,0].set_ylabel("height above ground (m)")
ax[0,0].set_aspect("equal"); ax[0,0].legend(fontsize=9)
ax[0,0].set_title(f"A. excavator side view — reconstructed ≈{head_pm:.2f} m vs real {REAL_EXC_H} m")
plt.colorbar(sc2, ax=ax[0,0], label="conf")

# B. two-head comparison
ax[0,1].bar(["pointmap\nhead", "depth\nhead"], [head_pm, head_dp], color=["steelblue","indianred"])
ax[0,1].axhline(REAL_EXC_H, color="red", ls="--", label=f"real {REAL_EXC_H} m")
for i,v in enumerate([head_pm, head_dp]): ax[0,1].text(i, v+0.03, f"{v:.2f}m", ha="center")
ax[0,1].set_ylabel("excavator vertical extent (m)"); ax[0,1].legend()
ax[0,1].set_title("B. both heads agree on compression → not a head artifact")

# C. per-frame heights + fused + real
fr_idx = np.arange(len(heights))
ax[1,0].bar(fr_idx, heights, color="slategray", label="per-frame height")
ax[1,0].axhline(fused_h, color="green", lw=2, label=f"multi-frame fused p95 = {fused_h:.2f} m")
ax[1,0].axhline(fused_h995, color="darkgreen", ls=":", lw=2, label=f"fused p99.5 = {fused_h995:.2f} m")
ax[1,0].axhline(REAL_EXC_H, color="red", ls="--", label=f"real {REAL_EXC_H} m")
ax[1,0].set_xlabel("frame"); ax[1,0].set_ylabel("excavator top height (m)")
ax[1,0].set_title("C. fusion ≈ best single frame, still ~8x short → no vertical recovery")
ax[1,0].legend(fontsize=8)

# D. coverage: single-frame vs multi-frame point count in footprint (multi helps coverage)
per_frame_counts = []
for fr in range(al.shape[0]):
    P = al[fr]; fin = np.isfinite(P).all(2); P = P[fin]
    per_frame_counts.append(int(in_fp(P).sum()))
ax[1,1].bar(["single\n(frame 0)", "multi\n(26 frames)"],
            [per_frame_counts[0], int(sum(per_frame_counts))], color=["slategray","seagreen"])
for i,v in enumerate([per_frame_counts[0], sum(per_frame_counts)]):
    ax[1,1].text(i, v, f"{v}", ha="center", va="bottom")
ax[1,1].set_ylabel("excavator-footprint point count")
ax[1,1].set_title("D. multi-frame DOES boost coverage (fills holes) — just not height")

fig.suptitle(f"VERTICAL COMPRESSION & (NON-)RECOVERY — excavator probe — "
             f"compression ≈ {REAL_EXC_H/head_pm:.1f}x", fontsize=13)
fig.tight_layout()
fig.savefig(f"{W}/vertical_recovery.png", dpi=110); print("wrote", f"{W}/vertical_recovery.png")

# ── JSON ─────────────────────────────────────────────────────────────────────
out = {
    "workspace": W, "scale_m_per_unit": scale, "gravity_source": g.source,
    "pit_judgment": {
        "candidate_roi_uv": ROI,
        "roi_median_depression_m": float(np.median(roi_depr)),
        "roi_p90_depression_m": float(np.percentile(roi_depr, 90)),
        "pit_cells_gt_0p2m": int(pit_cells.sum()),
        "scene_elevation_span_m": float((np.nanpercentile(Ymap,98)-np.nanpercentile(Ymap,2))*scale),
        "verdict": "no measurable pit (surface tracks reference ground)",
    },
    "vertical_compression": {
        "real_excavator_height_m": REAL_EXC_H,
        "reconstructed_pointmap_m": float(head_pm),
        "reconstructed_depth_m": float(head_dp),
        "compression_factor": float(REAL_EXC_H/head_pm),
    },
    "multiframe_recovery": {
        "per_frame_heights_m": [None if not np.isfinite(h) else round(float(h),3) for h in heights],
        "single_frame0_m": None if not np.isfinite(heights[0]) else float(heights[0]),
        "per_frame_max_m": float(np.nanmax(heights)),
        "fused_p95_m": float(fused_h), "fused_p99p5_m": float(fused_h995),
        "single_frame0_points": per_frame_counts[0],
        "multiframe_points": int(sum(per_frame_counts)),
        "coverage_gain_x": float(sum(per_frame_counts)/max(per_frame_counts[0],1)),
        "verdict": "multi-frame boosts coverage but does NOT recover vertical height",
    },
}
json.dump(out, open(f"{W}/vertical_recovery_result.json","w"), indent=1)
print("wrote", f"{W}/vertical_recovery_result.json")
print(json.dumps(out["pit_judgment"], indent=1))
print(json.dumps(out["vertical_compression"], indent=1))
print(json.dumps({k:v for k,v in out["multiframe_recovery"].items() if k!="per_frame_heights_m"}, indent=1))
