"""
Visualization of HSV histogram keyframe selection.
For each keyframe, shows: prev frame | keyframe | next frame
with their H×S histograms and similarity scores.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

IMAGES_DIR = Path("/home/maomaoyu/WS/vggt_yoloe/workspaces/session_20260615_165859_875219/images")
SIM_THRESH = 0.92

# ── helpers ───────────────────────────────────────────────────────────────────
def load_sig(img_path):
    img = cv2.resize(cv2.imread(str(img_path)), (64, 64))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.astype(np.float32)

def load_rgb(img_path):
    return cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

def corr(a, b):
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))

# ── run keyframe selection ────────────────────────────────────────────────────
all_frames = sorted(IMAGES_DIR.glob("*.png"))
N = len(all_frames)
sigs = [load_sig(p) for p in all_frames]

keyframes = [0]
last_sig = sigs[0]
for i in range(1, N):
    if corr(last_sig, sigs[i]) < SIM_THRESH:
        keyframes.append(i)
        last_sig = sigs[i]

# skip frame 0 (no "prev") for cleaner demo — use KF1 onward
demo_kfs = keyframes[1:]  # [8, 14, 23]
n_kf = len(demo_kfs)

# ── layout constants ─────────────────────────────────────────────────────────
# For each keyframe: 3 columns (prev | KF | next), 3 rows (frame | hist | scores)
# Plus a bottom row with the full timeline

BG      = "#1a1a2e"
AX_BG   = "#16213e"
TC      = "#e0e0e0"        # text colour
C_KF    = "#00ff88"        # keyframe green
C_PREV  = "#4fc3f7"        # prev frame blue
C_NEXT  = "#ffb74d"        # next frame orange
C_SKIP  = "#ff4466"        # skipped red
C_THRESH = "#ff4466"

fig = plt.figure(figsize=(6 * n_kf + 1, 18))
fig.patch.set_facecolor(BG)

# outer grid: n_kf groups side by side + 1 bottom timeline row
outer = gridspec.GridSpec(2, n_kf, figure=fig,
                          top=0.93, bottom=0.08,
                          hspace=0.55, wspace=0.18,
                          height_ratios=[3.5, 1])

fig.text(0.5, 0.965,
         "HSV Histogram Similarity — Keyframe Selection Detail",
         ha="center", fontsize=15, color=TC, fontweight="bold")
fig.text(0.5, 0.945,
         "Each panel: PREV frame  →  KEYFRAME  →  NEXT frame   |   dashed line = threshold 0.92",
         ha="center", fontsize=9, color="#aaaacc")

# ── per-keyframe panels ───────────────────────────────────────────────────────
for col, kf in enumerate(demo_kfs):
    prev_idx = kf - 1
    next_idx = min(kf + 1, N - 1)

    trio_idx   = [prev_idx, kf, next_idx]
    trio_color = [C_PREV, C_KF, C_NEXT]
    trio_label = [f"F{prev_idx}\n(prev)", f"F{kf}\nKEYFRAME", f"F{next_idx}\n(next)"]

    # corr of each frame vs the keyframe
    kf_sig   = sigs[kf]
    prev_sig = sigs[prev_idx]
    next_sig = sigs[next_idx]
    corr_prev_kf = corr(prev_sig, kf_sig)
    corr_next_kf = corr(next_sig, kf_sig)
    # corr of prev vs last keyframe (this is what triggered selection)
    last_kf = keyframes[keyframes.index(kf) - 1]
    corr_trigger = corr(sigs[last_kf], kf_sig)

    # inner grid: 3 rows × 3 cols inside each outer cell
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 3, subplot_spec=outer[0, col],
        hspace=0.45, wspace=0.12,
        height_ratios=[2.5, 2, 1.2]
    )

    for j, (idx, color, label) in enumerate(zip(trio_idx, trio_color, trio_label)):
        # row 0: raw frame
        ax_img = fig.add_subplot(inner[0, j])
        ax_img.imshow(load_rgb(all_frames[idx]))
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for sp in ax_img.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)
        ax_img.set_title(label, fontsize=8, color=color, pad=3)

        # row 1: H×S histogram
        ax_hist = fig.add_subplot(inner[1, j])
        ax_hist.set_facecolor(AX_BG)
        ax_hist.imshow(sigs[idx].reshape(32, 32), aspect="auto", origin="lower",
                       extent=[0, 256, 0, 180], cmap="plasma", vmin=0)
        ax_hist.set_xlabel("Saturation", fontsize=6, color=TC)
        if j == 0:
            ax_hist.set_ylabel("Hue", fontsize=6, color=TC)
        ax_hist.tick_params(colors=TC, labelsize=5)
        for sp in ax_hist.spines.values():
            sp.set_edgecolor("#444466")

    # row 2: similarity bar — prev vs KF, next vs KF
    ax_bar = fig.add_subplot(inner[2, :])   # span all 3 cols
    ax_bar.set_facecolor(AX_BG)

    vals   = [corr_prev_kf, 1.0, corr_next_kf]
    colors = [C_PREV,       C_KF, C_NEXT]
    xlabs  = [f"prev↔KF\n{corr_prev_kf:.3f}",
              "KF↔KF\n1.000",
              f"next↔KF\n{corr_next_kf:.3f}"]
    bars = ax_bar.bar([0, 1, 2], vals, color=colors, width=0.55, zorder=3)
    ax_bar.axhline(SIM_THRESH, color=C_THRESH, linewidth=1.5,
                   linestyle="--", zorder=4, label=f"thresh {SIM_THRESH}")
    ax_bar.set_ylim(0.7, 1.08)
    ax_bar.set_xticks([0, 1, 2]); ax_bar.set_xticklabels(xlabs, fontsize=7, color=TC)
    ax_bar.tick_params(colors=TC, labelsize=7)
    ax_bar.set_ylabel("Correlation", fontsize=7, color=TC)
    ax_bar.set_title(f"KF{kf} selected: corr vs KF{last_kf} = {corr_trigger:.3f} < {SIM_THRESH}",
                     fontsize=8, color=C_KF, pad=3)
    # shade above threshold
    ax_bar.axhspan(SIM_THRESH, 1.08, color="#ff4466", alpha=0.07, zorder=1)
    ax_bar.axhspan(0.7, SIM_THRESH, color="#00ff88", alpha=0.07, zorder=1)
    ax_bar.text(2.6, SIM_THRESH + 0.005, "skip zone", fontsize=6,
                color=C_THRESH, va="bottom", ha="right")
    ax_bar.text(2.6, SIM_THRESH - 0.012, "keep zone", fontsize=6,
                color=C_KF, va="top", ha="right")
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#444466")

# ── bottom: full timeline ─────────────────────────────────────────────────────
ax_tl = fig.add_subplot(outer[1, :])
ax_tl.set_facecolor(AX_BG)

# compute sequential corr for all frames
all_corrs = [1.0]
ref_sig = sigs[0]; ref_idx = 0
running_corrs = []   # corr vs current last keyframe (triggers selection)
for i in range(N):
    c = corr(sigs[ref_idx], sigs[i])
    running_corrs.append(c)
    if i in keyframes:
        ref_idx = i

seq_corrs = [1.0] + [corr(sigs[i-1], sigs[i]) for i in range(1, N)]

ax_tl.plot(range(N), seq_corrs, color="#4fc3f7", linewidth=1.5,
           marker="o", markersize=4, label="corr vs prev frame", zorder=3)
ax_tl.plot(range(N), running_corrs, color="#ffb74d", linewidth=1.5,
           linestyle="--", marker="s", markersize=3,
           label="corr vs last keyframe", zorder=3)
ax_tl.axhline(SIM_THRESH, color=C_THRESH, linewidth=1.5,
              linestyle=":", label=f"threshold {SIM_THRESH}", zorder=4)

for kf in keyframes:
    ax_tl.axvline(kf, color=C_KF, linewidth=1.2, alpha=0.6, zorder=2)
    ax_tl.scatter([kf], [running_corrs[kf]], color=C_KF, s=80, zorder=5)
    ax_tl.text(kf, 0.72, f"KF{kf}", fontsize=7, color=C_KF,
               ha="center", va="bottom")

ax_tl.set_xlim(-0.5, N - 0.5)
ax_tl.set_ylim(0.68, 1.08)
ax_tl.set_xlabel("Frame index", color=TC, fontsize=9)
ax_tl.set_ylabel("Correlation", color=TC, fontsize=9)
ax_tl.set_title("Full Timeline — Sequential Correlation & Keyframe Triggers",
                color=TC, fontsize=10)
ax_tl.legend(fontsize=8, facecolor="#2a2a4a", labelcolor=TC, loc="lower right")
ax_tl.tick_params(colors=TC)
for sp in ax_tl.spines.values():
    sp.set_edgecolor("#444466")

out = "/home/maomaoyu/WS/vggt_yoloe/hsv_similarity_viz.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved:", out)
plt.close()
