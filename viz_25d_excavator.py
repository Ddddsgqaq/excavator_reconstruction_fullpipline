#!/usr/bin/env python3
"""viz_25d_excavator.py — 2.5D terrain + real segmented excavator overlay.

Same presentation style as viz_25d.py, but:
  * the terrain SURFACE is built from the ground only (excavator excluded),
    so the machine no longer inflates the surface into a blob;
  * the excavator is drawn as its actual YOLOe-segmented point cloud, in its
    original RGB, floating on top of the terrain at the correct height.

Outputs:
  viz_25d_excavator.png        — hero 3D perspective
  viz_25d_excavator_combo.png  — triptych: BEV + two 3D angles

Usage:
  python viz_25d_excavator.py [workspace_dir] [--scale N] [--vexag N]
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

# ── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("ws", nargs="?",
                default="workspaces/session_20260617_172521_478306")
ap.add_argument("--scale", type=float, default=28.0, help="1 VGGT unit = N metres")
ap.add_argument("--vexag", type=float, default=3.0, help="vertical exaggeration")
ap.add_argument("--grid", type=int, default=128)
ap.add_argument("--exc-id", type=int, default=1, help="excavator semantic id")
ap.add_argument("--max-exc-pts", type=int, default=40000,
                help="max excavator points to scatter (perf)")
ap.add_argument("--point-size", type=float, default=8.0)
ap.add_argument("--no-clean", action="store_true",
                help="disable outlier removal + largest-cluster filtering")
ap.add_argument("--sor-k", type=int, default=16,
                help="statistical-outlier-removal: neighbours per point")
ap.add_argument("--sor-std", type=float, default=2.0,
                help="statistical-outlier-removal: std ratio (lower = stricter)")
ap.add_argument("--cluster-frac", type=float, default=0.03,
                help="largest-cluster flood-fill radius as fraction of body span")
args = ap.parse_args()

WS, SCALE, V_EXG, GRID = args.ws, args.scale, args.vexag, args.grid
EXC_ID = args.exc_id

BG, FG, ACCENT = "#0d1117", "#e6edf3", "#58a6ff"

from gravity_alignment import estimate_gravity, apply_alignment_to_points
from terrain_analysis import (
    analyze_terrain, ZONE_COLORS, ZONE_NAMES,
    ZONE_FLAT, ZONE_DIG, ZONE_DUMP, ZONE_PILE, ZONE_HAZARD, ZONE_OBSTACLE,
)

# ── load & gravity-align ──────────────────────────────────────────────────────
pred_d = np.load(os.path.join(WS, "predictions.npz"))
pred   = {k: np.array(pred_d[k]) for k in pred_d.files}
sem    = np.load(os.path.join(WS, "semantic_masks_fused.npz"))["semantic_masks"]

pts_raw   = pred["world_points_from_depth"]         # (S,H,W,3)
conf_raw  = pred["depth_conf"]                      # (S,H,W)
images    = pred["images"]                          # (S,3,H,W) or (S,H,W,3)
if images.ndim == 4 and images.shape[1] == 3:
    images = np.transpose(images, (0, 2, 3, 1))     # → (S,H,W,3)
rgb_flat  = images.reshape(-1, 3).astype(np.float32)
if rgb_flat.max() > 1.5:
    rgb_flat = rgb_flat / 255.0

pts_flat  = pts_raw.reshape(-1, 3)
conf_flat = conf_raw.reshape(-1).astype(np.float32)
conf_thres = 0.5 * conf_flat.max()

# Ground for gravity/terrain = everything that is NOT the excavator.
ground_3d = (sem != EXC_ID)

grav = estimate_gravity(pred["extrinsic"], pts_raw,
                        ground_mask=ground_3d, conf=conf_raw,
                        conf_thres=conf_thres / conf_flat.max())

pts_a = apply_alignment_to_points(pts_flat, grav.R_align)
keep  = np.isfinite(pts_a).all(axis=1) & (conf_flat >= conf_thres)
pts_k = pts_a[keep]
sem_k = sem.reshape(-1)[keep].astype(np.int32)
rgb_k = rgb_flat[keep]

exc_k = (sem_k == EXC_ID)                # excavator points
gnd_k = ~exc_k                           # terrain points (ground candidates)

# Terrain analysis on GROUND-ONLY points → surface has no excavator blob.
result = analyze_terrain(pts_k[gnd_k], sem_k[gnd_k], np.ones(int(gnd_k.sum()), bool),
                         id_to_name={}, grid_res=GRID, top_percentile=90.0)

g, L, ws = result["grid"], result["layers"], result["worksite"]

# ── metric grids ───────────────────────────────────────────────────────────
def _arr(key):
    return np.array([[np.nan if v is None else v for v in row]
                     for row in L[key]], dtype=np.float64)

H_top = _arr("H_top")    * SCALE
H_gnd = _arr("H_ground") * SCALE
zone  = np.array(L["zone_map"], dtype=np.int32)

xi = np.linspace(g["x_min"] * SCALE, g["x_max"] * SCALE, GRID)
zi = np.linspace(g["z_min"] * SCALE, g["z_max"] * SCALE, GRID)
XX, ZZ = np.meshgrid(xi, zi)

H_fill = np.where(np.isfinite(H_top), H_top, H_gnd)
finite = H_fill[np.isfinite(H_fill)]
lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
H_fill = np.clip(H_fill, lo, hi)
H_smooth = gaussian_filter(H_fill, sigma=1.6)
H_exag   = H_smooth * V_EXG

dx = (xi[-1] - xi[0]) / (GRID - 1)
dz = (zi[-1] - zi[0]) / (GRID - 1)

# ── zone RGBA + hillshade ─────────────────────────────────────────────────────
ZONE_ORDER = [ZONE_FLAT, ZONE_DIG, ZONE_DUMP, ZONE_PILE, ZONE_HAZARD, ZONE_OBSTACLE]
ZONE_ALPHA = {ZONE_FLAT:0.72, ZONE_DIG:0.95, ZONE_DUMP:0.88,
              ZONE_PILE:0.92, ZONE_HAZARD:0.95, ZONE_OBSTACLE:0.80}
present = [z for z in ZONE_ORDER if (zone == z).any()]

rgb_base = np.zeros((GRID, GRID, 3))
alpha    = np.full((GRID, GRID), 0.25)
for code in present:
    m = zone == code
    rgb_base[m] = ZONE_COLORS[code][:3]
    alpha[m]    = ZONE_ALPHA[code]

ls = LightSource(azdeg=225, altdeg=50)
intensity = ls.hillshade(H_fill, vert_exag=V_EXG, dx=dx, dy=dz)
rgb_shaded = np.clip(rgb_base * (0.40 + 0.60 * intensity[:, :, None]), 0, 1)
rgba = np.dstack([rgb_shaded, alpha])

# ── excavator point cloud (real RGB), in the same metric+exaggerated frame ────
exc_xyz = pts_k[exc_k].copy()
exc_rgb = np.clip(rgb_k[exc_k], 0, 1)
n0 = len(exc_xyz)

# (1) Statistical outlier removal — kills the mixed-color "ghost" dots that
#     depth-bleed off the silhouette edges into empty space. For each point,
#     measure mean distance to its k nearest neighbours; drop points whose
#     mean-kNN distance exceeds mean + std_ratio·std of the whole set.
def statistical_outlier_mask(xyz, k=16, std_ratio=2.0):
    from scipy.spatial import cKDTree
    if len(xyz) <= k:
        return np.ones(len(xyz), bool)
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1)          # +1: first neighbour is self (0)
    md = d[:, 1:].mean(axis=1)
    thr = md.mean() + std_ratio * md.std()
    return md <= thr

if len(exc_xyz) and not args.no_clean:
    keep_so = statistical_outlier_mask(exc_xyz, k=args.sor_k, std_ratio=args.sor_std)
    exc_xyz, exc_rgb = exc_xyz[keep_so], exc_rgb[keep_so]

# (2) Keep only the largest connected cluster (the machine body), discarding
#     detached floaters, via a coarse voxel flood-fill on a KD-tree.
def largest_cluster_mask(xyz, radius):
    from scipy.spatial import cKDTree
    n = len(xyz)
    if n == 0:
        return np.zeros(0, bool)
    tree = cKDTree(xyz)
    seen = np.zeros(n, bool)
    best = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True; comp = []
        while stack:
            i = stack.pop(); comp.append(i)
            for j in tree.query_ball_point(xyz[i], radius):
                if not seen[j]:
                    seen[j] = True; stack.append(j)
        if len(comp) > len(best):
            best = comp
    out = np.zeros(n, bool); out[best] = True
    return out

if len(exc_xyz) and not args.no_clean:
    # radius ~ a few voxel widths relative to the machine's extent
    span = np.linalg.norm(exc_xyz.max(0) - exc_xyz.min(0))
    keep_cl = largest_cluster_mask(exc_xyz, radius=span * args.cluster_frac)
    exc_xyz, exc_rgb = exc_xyz[keep_cl], exc_rgb[keep_cl]

print(f"excavator cleanup: {n0} → {len(exc_xyz)} pts "
      f"(removed {n0 - len(exc_xyz)} ghost/outlier)")

# subsample for render performance
if len(exc_xyz) > args.max_exc_pts:
    idx = np.random.default_rng(0).choice(len(exc_xyz), args.max_exc_pts, replace=False)
    exc_xyz, exc_rgb = exc_xyz[idx], exc_rgb[idx]
EX = exc_xyz[:, 0] * SCALE
EZ = exc_xyz[:, 2] * SCALE
EY = exc_xyz[:, 1] * SCALE * V_EXG
print(f"excavator points drawn: {len(exc_xyz)}  "
      f"(height {exc_xyz[:,1].min()*SCALE:.1f}..{exc_xyz[:,1].max()*SCALE:.1f} m)")

# ── draw ──────────────────────────────────────────────────────────────────────
def draw_terrain(ax, azim=-135, elev=35, show_labels=True):
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(BG)
    ax.plot_surface(XX, ZZ, H_exag, facecolors=rgba, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)
    z_floor = H_exag.min() - (H_exag.max() - H_exag.min()) * 0.15
    ax.contourf(XX, ZZ, H_fill, levels=10, zdir='z', offset=z_floor,
                cmap='terrain', alpha=0.28)

    # the actual segmented excavator, in real RGB
    ax.scatter(EX, EZ, EY, c=exc_rgb, s=args.point_size, marker="o",
               depthshade=False, edgecolors="none", zorder=12)
    if show_labels and len(EX):
        ax.text(float(EX.mean()), float(EZ.mean()), float(EY.max()) + 0.05 * (H_exag.max()-H_exag.min()),
                "Excavator", color="#ffd400", fontsize=9, fontweight="bold",
                ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ns = ws.get("next_scoop")
    if ns:
        px, pz = ns["xz"][0] * SCALE, ns["xz"][1] * SCALE
        col = int(np.argmin(np.abs(xi - px))); row = int(np.argmin(np.abs(zi - pz)))
        py = H_exag[row, col]
        ax.scatter([px], [pz], [py], color=FG, s=260, marker="*",
                   zorder=13, depthshade=False)
        if show_labels:
            ax.text(px, pz, py + 0.12*(H_exag.max()-H_exag.min()), "NEXT\nSCOOP",
                    color=FG, fontsize=8, fontweight="bold", ha="center", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_xlabel("X (m)", color=FG, labelpad=3, fontsize=8)
    ax.set_ylabel("Z (m)", color=FG, labelpad=3, fontsize=8)
    ax.set_zlabel(f"Height ×{V_EXG:.0f} (m)", color=FG, labelpad=3, fontsize=8)
    ax.grid(color="#30363d", linewidth=0.35)


def draw_excavator_closeup(ax, azim=-125, elev=18):
    """Excavator-only close-up in real RGB (NO vertical exaggeration, tight
    zoom) so the machine is actually visible next to the wide site view."""
    ax.set_facecolor("#0d1117")
    ax.view_init(elev=elev, azim=azim)
    # true metric height (undo the ×V_EXG used on the site surface)
    EYc = EY / V_EXG
    ax.scatter(EX, EZ, EYc, c=exc_rgb, s=max(args.point_size, 9),
               marker="o", depthshade=False, edgecolors="none")
    ax.set_box_aspect((1, 1, 0.9))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#30363d")
    ax.tick_params(colors="#6e7681", labelsize=6)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.grid(color="#30363d", linewidth=0.3)
    ax.set_title("Segmented excavator  ·  real RGB  (true scale)",
                 color="#ffd400", fontsize=9, pad=2)

# ── Figure 1: hero ─────────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(18, 10), facecolor=BG)
fig1.suptitle(f"Excavation Site  ·  Terrain + Segmented Excavator  ·  {os.path.basename(WS)}"
              f"  [1 unit = {SCALE:.0f} m,  vertical ×{V_EXG:.0f}]",
              color=FG, fontsize=13, y=0.975)
ax3d = fig1.add_axes([0.02, 0.03, 0.60, 0.92], projection="3d")
draw_terrain(ax3d, azim=-135, elev=35)
# zoomed excavator inset (top-right), so the machine is clearly visible
ax_ins = fig1.add_axes([0.60, 0.42, 0.30, 0.46], projection="3d")
draw_excavator_closeup(ax_ins)
handles = [Patch(facecolor=ZONE_COLORS[z], label=ZONE_NAMES[z], edgecolor="#555", linewidth=0.8)
           for z in present] + [Patch(facecolor="#c0c0c0", label="Excavator (RGB pts)", edgecolor="#555")]
leg = ax3d.legend(handles=handles, loc="upper left", fontsize=9, title="Legend",
                  facecolor="#161b22", edgecolor="#30363d", labelcolor=FG, title_fontsize=9)
leg.get_title().set_color(ACCENT)
out1 = os.path.join(WS, "viz_25d_excavator.png")
fig1.savefig(out1, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig1); print(f"saved: {out1}")

# ── Figure 2: triptych ─────────────────────────────────────────────────────────
from terrain_analysis import render_worksite_bev
fig2 = plt.figure(figsize=(24, 8), facecolor=BG)
fig2.suptitle(f"Terrain + Segmented Excavator  ·  {os.path.basename(WS)}   [1 unit = {SCALE:.0f} m]",
              color=FG, fontsize=13, y=0.98)
_tmp = os.path.join(WS, "_tmp_bev_exc.png")
render_worksite_bev(result, _tmp, title="", scale_factor=SCALE)
ax_bev = fig2.add_axes([0.01, 0.04, 0.30, 0.88]); ax_bev.imshow(plt.imread(_tmp))
ax_bev.axis("off"); ax_bev.set_title("2D BEV Zone Map", color=FG, fontsize=11, pad=6)
os.remove(_tmp)
ax_b = fig2.add_axes([0.32, 0.04, 0.32, 0.88], projection="3d")
draw_terrain(ax_b, azim=-45, elev=55); ax_b.set_title("3D · overhead", color=FG, fontsize=10, pad=6)
ax_c = fig2.add_axes([0.65, 0.04, 0.34, 0.88], projection="3d")
draw_terrain(ax_c, azim=135, elev=22); ax_c.set_title("3D · ground-level", color=FG, fontsize=10, pad=6)
handles2 = [Patch(facecolor=ZONE_COLORS[z], label=ZONE_NAMES[z], edgecolor="#555", linewidth=0.8)
            for z in present] + [Patch(facecolor="#c0c0c0", label="Excavator", edgecolor="#555")]
fig2.legend(handles=handles2, loc="lower center", ncol=len(present)+1, fontsize=9,
            facecolor="#161b22", edgecolor="#30363d", labelcolor=FG, bbox_to_anchor=(0.5, 0.0))
out2 = os.path.join(WS, "viz_25d_excavator_combo.png")
fig2.savefig(out2, dpi=130, facecolor=BG, bbox_inches="tight")
plt.close(fig2); print(f"saved: {out2}")
