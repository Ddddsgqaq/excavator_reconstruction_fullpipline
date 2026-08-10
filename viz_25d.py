#!/usr/bin/env python3
"""viz_25d.py — 2.5D presentation-quality terrain visualization.

Reads predictions.npz + semantic_masks_fused.npz from a workspace,
runs the terrain analysis pipeline, and outputs:
  viz_25d.png        — hero 3D perspective view (dark, 16:9)
  viz_25d_combo.png  — triptych: 2D BEV + two 3D angles (slide layout)

Usage:
  python viz_25d.py [workspace_dir] [--scale N]
  Default workspace: workspaces/session_20260617_172521_478306
  Default scale:     28   (1 VGGT unit = 28 m; read from worksite_bev.png title)
"""

import sys, os, math, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – registers 3d projection
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("ws", nargs="?",
                default="workspaces/session_20260617_172521_478306")
ap.add_argument("--scale", type=float, default=28.0,
                help="1 VGGT unit = N metres")
ap.add_argument("--vexag", type=float, default=3.0,
                help="vertical exaggeration for the 3D view")
ap.add_argument("--grid", type=int, default=128)
args = ap.parse_args()

WS    = args.ws
SCALE = args.scale
V_EXG = args.vexag
GRID  = args.grid

BG = "#0d1117"
FG = "#e6edf3"
ACCENT = "#58a6ff"

# ── load & gravity-align points ──────────────────────────────────────────────
from gravity_alignment import estimate_gravity, apply_alignment_to_points
from terrain_analysis import (
    analyze_terrain,
    ZONE_COLORS, ZONE_NAMES,
    ZONE_FLAT, ZONE_DIG, ZONE_DUMP, ZONE_PILE, ZONE_HAZARD, ZONE_OBSTACLE, ZONE_EMPTY,
)

pred_d = np.load(os.path.join(WS, "predictions.npz"))
pred   = {k: np.array(pred_d[k]) for k in pred_d.files}
sem    = np.load(os.path.join(WS, "semantic_masks_fused.npz"))["semantic_masks"]

pts_raw   = pred["world_points_from_depth"]   # (S,H,W,3)
conf_raw  = pred["depth_conf"]                # (S,H,W)
pts_flat  = pts_raw.reshape(-1, 3)
conf_flat = conf_raw.reshape(-1).astype(np.float32)
conf_thres = 0.5 * conf_flat.max()

grav = estimate_gravity(pred["extrinsic"], pts_raw,
                        ground_mask=(sem == 1), conf=conf_raw,
                        conf_thres=conf_thres / conf_flat.max())

pts_a = apply_alignment_to_points(pts_flat, grav.R_align)
keep  = np.isfinite(pts_a).all(axis=1) & (conf_flat >= conf_thres)
pts_k = pts_a[keep]
sem_k = sem.reshape(-1)[keep].astype(np.int32)
gnd_k = (sem.reshape(-1)[keep] == 1)

result = analyze_terrain(pts_k, sem_k, gnd_k, id_to_name={},
                         grid_res=GRID, top_percentile=90.0)

g   = result["grid"]
L   = result["layers"]
ws  = result["worksite"]

# ── build metric grids ───────────────────────────────────────────────────────
def _arr(key):
    return np.array([[np.nan if v is None else v for v in row]
                     for row in L[key]], dtype=np.float64)

H_top  = _arr("H_top")   * SCALE
H_gnd  = _arr("H_ground") * SCALE
zone   = np.array(L["zone_map"], dtype=np.int32)

xi = np.linspace(g["x_min"] * SCALE, g["x_max"] * SCALE, GRID)
zi = np.linspace(g["z_min"] * SCALE, g["z_max"] * SCALE, GRID)
XX, ZZ = np.meshgrid(xi, zi)   # XX[row,col]=xi[col], ZZ[row,col]=zi[row]

H_fill = np.where(np.isfinite(H_top), H_top, H_gnd)
# clip extreme outlier cells before smoothing (keep p2–p98)
finite = H_fill[np.isfinite(H_fill)]
lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
H_fill = np.clip(H_fill, lo, hi)
from scipy.ndimage import gaussian_filter
H_smooth = gaussian_filter(H_fill, sigma=1.6)
H_exag = H_smooth * V_EXG

dx = (xi[-1] - xi[0]) / (GRID - 1)
dz = (zi[-1] - zi[0]) / (GRID - 1)

# ── zone color RGBA with hillshade ───────────────────────────────────────────
ZONE_ORDER  = [ZONE_FLAT, ZONE_DIG, ZONE_DUMP, ZONE_PILE, ZONE_HAZARD, ZONE_OBSTACLE]
ZONE_ALPHA  = {ZONE_FLAT:0.72, ZONE_DIG:0.95, ZONE_DUMP:0.88,
               ZONE_PILE:0.92, ZONE_HAZARD:0.95, ZONE_OBSTACLE:0.80}
present = [z for z in ZONE_ORDER if (zone == z).any()]

rgb_base = np.zeros((GRID, GRID, 3), dtype=np.float64)
alpha    = np.full((GRID, GRID), 0.25)          # empty cells: near-transparent
for code in present:
    m = zone == code
    rgb_base[m] = ZONE_COLORS[code][:3]
    alpha[m]    = ZONE_ALPHA[code]

ls        = LightSource(azdeg=225, altdeg=50)
intensity = ls.hillshade(H_fill, vert_exag=V_EXG, dx=dx, dy=dz)

# ambient 40 % + diffuse 60 %, then clamp
rgb_shaded = np.clip(rgb_base * (0.40 + 0.60 * intensity[:, :, None]), 0, 1)
rgba       = np.dstack([rgb_shaded, alpha])

# ── helper: nearest grid index ───────────────────────────────────────────────
def _grid_idx(world_x, world_z):
    """Return (row, col) of the grid cell nearest to (world_x, world_z)."""
    col = int(np.argmin(np.abs(xi - world_x)))
    row = int(np.argmin(np.abs(zi - world_z)))
    return row, col


# ── shared draw function ─────────────────────────────────────────────────────
def draw_terrain(ax, azim=-135, elev=35, show_labels=True):
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(BG)

    # --- main surface ---
    ax.plot_surface(XX, ZZ, H_exag,
                    facecolors=rgba, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

    # --- contour shadow at floor ---
    z_floor = H_exag.min() - (H_exag.max() - H_exag.min()) * 0.15
    ax.contourf(XX, ZZ, H_fill,
                levels=10, zdir='z', offset=z_floor,
                cmap='terrain', alpha=0.28)

    # --- machines: vertical flagpole + label ---
    pole_h = (H_exag.max() - H_exag.min()) * 0.28
    for mc in ws.get("machines", []):
        mx = mc["centroid"][0] * SCALE
        mz = mc["centroid"][1] * SCALE
        row, col = _grid_idx(mx, mz)
        my_surf  = H_exag[row, col]
        ax.plot([mx, mx], [mz, mz], [my_surf, my_surf + pole_h],
                color="#ffd400", lw=2.5, zorder=10)
        ax.scatter([mx], [mz], [my_surf + pole_h],
                   color="#ffd400", s=100, zorder=11, depthshade=False)
        if show_labels:
            ax.text(mx, mz, my_surf + pole_h * 1.2,
                    mc.get("label", "Excavator"),
                    color="#ffd400", fontsize=8, fontweight="bold",
                    ha="center", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # --- next scoop: dashed drop-line + star ---
    ns = ws.get("next_scoop")
    if ns:
        px = ns["xz"][0] * SCALE
        pz = ns["xz"][1] * SCALE
        row, col = _grid_idx(px, pz)
        py_surf  = H_exag[row, col]
        drop_top = py_surf + pole_h * 1.4
        ax.plot([px, px], [pz, pz], [drop_top, py_surf + 0.01],
                color=FG, lw=1.8, linestyle="--", zorder=12)
        ax.scatter([px], [pz], [py_surf],
                   color=FG, s=280, marker="*", zorder=13, depthshade=False)
        if show_labels:
            ax.text(px, pz, drop_top * 1.06,
                    "NEXT\nSCOOP", color=FG, fontsize=8, fontweight="bold",
                    ha="center", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # --- axes style ---
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_xlabel("X (m)", color=FG, labelpad=3, fontsize=8)
    ax.set_ylabel("Z (m)", color=FG, labelpad=3, fontsize=8)
    ax.set_zlabel(f"Height ×{V_EXG:.0f} (m)", color=FG, labelpad=3, fontsize=8)
    ax.grid(color="#30363d", linewidth=0.35)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Hero 3D view
# ════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(18, 10), facecolor=BG)
fig1.suptitle(
    f"Excavation Site  ·  3D Semantic Terrain  ·  {os.path.basename(WS)}"
    f"  [1 unit = {SCALE:.0f} m,  vertical ×{V_EXG:.0f}]",
    color=FG, fontsize=13, y=0.975,
)

ax3d = fig1.add_axes([0.01, 0.03, 0.72, 0.92], projection="3d")
draw_terrain(ax3d, azim=-135, elev=35)

# legend on the 3D axes
handles = [Patch(facecolor=ZONE_COLORS[z], label=ZONE_NAMES[z],
                 edgecolor="#555", linewidth=0.8)
           for z in present]
leg = ax3d.legend(handles=handles,
                  loc="upper left", fontsize=9, title="Zones",
                  facecolor="#161b22", edgecolor="#30363d", labelcolor=FG,
                  title_fontsize=9)
leg.get_title().set_color(ACCENT)

# info panel (right strip)
ax_info = fig1.add_axes([0.75, 0.08, 0.23, 0.84])
ax_info.set_facecolor("#0d1117")
ax_info.set_xticks([]); ax_info.set_yticks([])
for sp in ax_info.spines.values():
    sp.set_edgecolor("#21262d")

def _txt(ax, x, y, s, color=FG, size=10, bold=False):
    kw = dict(color=color, fontsize=size, va="top", transform=ax.transAxes,
              fontfamily="monospace")
    if bold:
        kw["fontweight"] = "bold"
    ax.text(x, y, s, **kw)

y = 0.97
_txt(ax_info, 0.05, y, "ZONE AREAS", color=ACCENT, size=11, bold=True); y -= 0.07
for zinfo in sorted(ws["zones"], key=lambda d: -d["area"]):
    zcode = next((k for k, n in ZONE_NAMES.items() if n == zinfo["name"]), None)
    col   = ZONE_COLORS.get(zcode, [0.8]*3)
    color_hex = "#{:02x}{:02x}{:02x}".format(
        int(col[0]*255), int(col[1]*255), int(col[2]*255))
    _txt(ax_info, 0.05, y,
         f"  {zinfo['name']:10s} {zinfo['area']*SCALE*SCALE:>8.1f} m²",
         color=color_hex, size=10); y -= 0.057

y -= 0.03
ns = ws.get("next_scoop")
if ns:
    _txt(ax_info, 0.05, y, "NEXT SCOOP", color=ACCENT, size=11, bold=True); y -= 0.07
    _txt(ax_info, 0.05, y, f"  pos    ({ns['xz'][0]*SCALE:+.1f}, {ns['xz'][1]*SCALE:+.1f}) m"); y -= 0.057
    _txt(ax_info, 0.05, y, f"  zone   {ns['zone']}"); y -= 0.057
    _txt(ax_info, 0.05, y, f"  depth  {ns['depth']*SCALE:.2f} m"); y -= 0.057
    if ns.get("heading_deg") is not None:
        _txt(ax_info, 0.05, y, f"  hdg    {ns['heading_deg']:.0f}°"); y -= 0.057

y -= 0.05
mc_list = ws.get("machines", [])
if mc_list:
    _txt(ax_info, 0.05, y, "MACHINES", color=ACCENT, size=11, bold=True); y -= 0.07
    for mc in mc_list:
        _txt(ax_info, 0.05, y,
             f"  {mc.get('label','?'):<14s} {mc['area']*SCALE*SCALE:.0f} m²"); y -= 0.057

out1 = os.path.join(WS, "viz_25d.png")
fig1.savefig(out1, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig1)
print(f"saved: {out1}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Triptych: BEV + two 3D angles
# ════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(24, 8), facecolor=BG)
fig2.suptitle(
    f"Excavation Site Terrain Analysis  ·  {os.path.basename(WS)}   [1 unit = {SCALE:.0f} m]",
    color=FG, fontsize=13, y=0.98,
)

# --- Panel A: 2D BEV zone map (replicate render_worksite_bev inline) ---
from terrain_analysis import render_worksite_bev, ZONE_EMPTY as _ZE
_tmp_bev = os.path.join(WS, "_tmp_bev_combo.png")
render_worksite_bev(result, _tmp_bev, title="", scale_factor=SCALE)

bev_img = plt.imread(_tmp_bev)
ax_bev = fig2.add_axes([0.01, 0.04, 0.30, 0.88])
ax_bev.imshow(bev_img)
ax_bev.axis("off")
ax_bev.set_title("2D BEV Zone Map", color=FG, fontsize=11, pad=6)
os.remove(_tmp_bev)

# --- Panel B: 3D bird's-eye perspective ---
ax_b = fig2.add_axes([0.32, 0.04, 0.32, 0.88], projection="3d")
draw_terrain(ax_b, azim=-45, elev=55, show_labels=True)
ax_b.set_title("3D  ·  overhead perspective", color=FG, fontsize=10, pad=6)

# --- Panel C: 3D ground-level perspective ---
ax_c = fig2.add_axes([0.65, 0.04, 0.34, 0.88], projection="3d")
draw_terrain(ax_c, azim=135, elev=22, show_labels=True)
ax_c.set_title("3D  ·  ground-level perspective", color=FG, fontsize=10, pad=6)

# shared legend below
handles2 = [Patch(facecolor=ZONE_COLORS[z], label=ZONE_NAMES[z],
                  edgecolor="#555", linewidth=0.8)
            for z in present]
fig2.legend(handles=handles2,
            loc="lower center", ncol=len(present),
            fontsize=9, facecolor="#161b22", edgecolor="#30363d",
            labelcolor=FG, bbox_to_anchor=(0.5, 0.0))

out2 = os.path.join(WS, "viz_25d_combo.png")
fig2.savefig(out2, dpi=130, facecolor=BG, bbox_inches="tight")
plt.close(fig2)
print(f"saved: {out2}")
