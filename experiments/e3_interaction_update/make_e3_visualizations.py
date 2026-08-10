"""Visualizations for E3 — Interaction-Triggered Incremental Update.

Four figures:
  1. interaction_timeline.png  — REAL bucket height curve + up/down/static states,
     with detected dig events marked (the interaction cue that triggers updates).
  2. dem_evolution.png         — oracle GT DEM vs each strategy's fused DEM after
     every dig event; the ground-truth column is the reference.
  3. roi_localization.png      — top-down GT change map per event with the bucket
     anchor + interaction ROI overlaid (mechanism-5 causal localization).
  4. cost_accuracy_tradeoff.png — cumulative processed cells vs DEM error and cut
     volume error; ROI reaches full-reconstruction accuracy at a fraction of cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

WS = Path(__file__).resolve().parent
VIZ = WS / "visualizations"
VIZ.mkdir(exist_ok=True)

STRAT_ORDER = ["full", "change", "roi"]
STRAT_LABEL = {"full": "A · full recon", "change": "B · change-only", "roi": "C · ROI (ours)"}
STRAT_COLOR = {"full": "#7f7f7f", "change": "#d68910", "roi": "#2f78d0"}


def load():
    res = json.loads((WS / "e3_results.json").read_text())
    dems = np.load(WS / "e3_dems.npz")
    return res, dems


def fig_timeline(res: dict) -> Path:
    cue = res["interaction_cue"]
    h = np.asarray(cue["bucket_H"])
    states = cue["bucket_state"]
    fps = cue["fps"]
    t = np.arange(len(h)) / fps
    ev_frames = {e["frame"] for e in res["dig_events"]}

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=170)
    ax.plot(t, h, "-o", color="#333", lw=1.6, ms=4, label="bucket height above ground (real)")
    cmap = {"up": "#2ca02c", "down": "#d62728", "static": "#999999"}
    for f, s in enumerate(states):
        ax.axvspan(t[f] - 0.5 / fps, t[f] + 0.5 / fps, color=cmap[s], alpha=0.12)
    ax.set_ylim(h.min() - 0.01, h.max() + 0.025)
    for e in res["dig_events"]:
        f = e["frame"]
        ax.axvline(t[f], color="#2f78d0", ls="--", lw=1.4)
        ax.annotate(f"dig #{res['dig_events'].index(e)+1}\n{e['cut_volume_m3']:.3f} m³",
                    (t[f], h.min() - 0.005), textcoords="offset points", xytext=(6, 2),
                    fontsize=8, color="#2f78d0", va="bottom")
    handles = [plt.Line2D([], [], color=cmap[k], lw=6, alpha=.4, label=f"bucket {k}") for k in cmap]
    handles.append(plt.Line2D([], [], color="#2f78d0", ls="--", label="detected dig event"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    ax.set(xlabel="time (s)", ylabel="bucket height (session units)",
           title="Real interaction cue: bucket motion → detected dig events\n"
                 "(arm_motion_state/motion_state.json; each event triggers a localized update)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    out = VIZ / "interaction_timeline.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig_dem_evolution(res: dict, dems) -> Path:
    """Two compact rows: (top) oracle GT after each dig; (bottom) GT-final next to
    each strategy's final fused DEM, with the fused−GT error map inset via caption."""
    gt_seq = dems["gt_seq"]           # (n_events+1, G, G)
    n_ev = len(res["dig_events"])
    finals = {s: dems[f"final_{s}"] for s in STRAT_ORDER}
    anchors = dems["anchors_cell"]

    vmin = float(min(gt_seq.min(), *(f.min() for f in finals.values())))
    vmax = float(max(gt_seq.max(), *(f.max() for f in finals.values())))

    top_n = n_ev + 1
    bot_n = 1 + len(STRAT_ORDER)
    ncol = max(top_n, bot_n)
    fig = plt.figure(figsize=(3.1 * ncol, 6.6), dpi=150)
    gs = fig.add_gridspec(2, ncol, hspace=0.32, wspace=0.08)

    def show(ax, grid, title):
        im = ax.imshow(grid, cmap="terrain", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        return im

    im = None
    # top row: real base + injected dig evolution
    for c in range(top_n):
        ax = fig.add_subplot(gs[0, c])
        title = "real VGGT base" if c == 0 else f"+ dig #{c}"
        im = show(ax, gt_seq[c], title)
        if c > 0:
            r0, c0 = anchors[c - 1]
            ax.plot(c0, r0, "x", color="red", ms=9, mew=2)

    # bottom row: GT-final + each strategy's final fused DEM with metric caption
    ax = fig.add_subplot(gs[1, 0])
    show(ax, gt_seq[-1], "ground truth (final)")
    for spine in ax.spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(1.5)
    for i, s in enumerate(STRAT_ORDER, start=1):
        ax = fig.add_subplot(gs[1, i])
        st = res["strategies"][s]
        show(ax, finals[s], STRAT_LABEL[s])
        ax.set_xlabel(f"{st['processed_points_total']/1e6:.2f}M real pts  ·  "
                      f"RMSE {st['final_dem_rmse_m']*1000:.1f} mm\n"
                      f"cut-vol err {st['cut_volume_error_frac']*100:.0f}%  ·  "
                      f"consist {st['map_consistency_mean']*100:.0f}%",
                      fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(STRAT_COLOR[s]); spine.set_linewidth(2.2)

    fig.colorbar(im, ax=fig.axes, shrink=0.45, pad=0.015, label="height (m)")
    fig.suptitle("REAL VGGT base DEM with injected digs (top) vs each strategy's final fused DEM "
                 "(bottom)\nred × = real bucket dig anchor per event · cost in real VGGT points",
                 fontsize=12, y=0.98)
    out = VIZ / "dem_evolution.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig_roi_localization(res: dict, dems) -> Path:
    gt_seq = dems["gt_seq"]
    anchors = dems["anchors_cell"]
    n_ev = len(res["dig_events"])
    G = gt_seq.shape[1]
    # ROI radius reconstructed from results config note (wedge 6 + margin 3) and dump.
    roi_r = 9
    spoil = (int(G * 0.30), int(G * 0.20))

    fig, axes = plt.subplots(1, n_ev, figsize=(3.4 * n_ev, 3.6), dpi=160)
    if n_ev == 1:
        axes = [axes]
    for k in range(n_ev):
        ax = axes[k]
        diff = gt_seq[k + 1] - gt_seq[k]
        m = np.abs(diff).max() or 1.0
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-m, vmax=m, origin="lower")
        r0, c0 = anchors[k]
        ax.add_patch(Circle((c0, r0), roi_r, fill=False, color="#2f78d0", lw=2))
        ax.add_patch(Circle((spoil[1], spoil[0]), roi_r, fill=False, color="#2f78d0", lw=2, ls="--"))
        ax.plot(c0, r0, "x", color="#2f78d0", ms=10, mew=2.5)
        ev = res["dig_events"][k]
        st = res["strategies"]["roi"]["per_event"][k]
        ax.set_title(f"dig #{k+1} (t={ev['time_s']}s)\nROI recall {st['change_recall']*100:.0f}%  "
                     f"consist {st['map_consistency']*100:.0f}%", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label="true height change Δh (m)")
    fig.suptitle("Mechanism-5 causal localization: the bucket anchor (solid ROI = cut) and dump "
                 "site (dashed ROI = fill)\nare where the true terrain change is — so only these "
                 "cells need re-estimation", fontsize=11)
    out = VIZ / "roi_localization.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig_cost_accuracy(res: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=170)
    strat = res["strategies"]

    # left: cumulative REAL VGGT points re-rasterized vs event, per strategy
    for s in STRAT_ORDER:
        pe = strat[s]["per_event"]
        cum = np.cumsum([e["observed_points"] for e in pe]) / 1e6
        # full and change-only overlap exactly; dash full so both remain visible.
        ls = "--" if s == "full" else "-"
        lw = 3.2 if s == "full" else 2.0
        axes[0].plot(range(1, len(cum) + 1), cum, ls, marker="o", color=STRAT_COLOR[s],
                     lw=lw, label=STRAT_LABEL[s])
    axes[0].set(xlabel="dig event", ylabel="cumulative real VGGT points re-rasterized (M)",
                title="Compute cost (real VGGT points re-estimated)")
    axes[0].legend(); axes[0].grid(alpha=.25)

    # right: bar chart of the headline tradeoff (points vs RMSE)
    labels = [STRAT_LABEL[s] for s in STRAT_ORDER]
    cost = [strat[s]["processed_points_total"] / 1e6 for s in STRAT_ORDER]
    rmse = [strat[s]["final_dem_rmse_m"] * 1000 for s in STRAT_ORDER]  # mm
    x = np.arange(len(labels))
    ax2 = axes[1]; ax3 = ax2.twinx()
    b1 = ax2.bar(x - 0.2, cost, 0.4, color="#bbbbbb", label="processed points")
    b2 = ax3.bar(x + 0.2, rmse, 0.4, color=[STRAT_COLOR[s] for s in STRAT_ORDER],
                 label="final DEM RMSE (mm)")
    ax2.set_ylabel("real VGGT points re-rasterized (M, total)"); ax3.set_ylabel("final DEM RMSE (mm)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("Cost vs accuracy: ROI matches full-recon RMSE at ~28% of the VGGT points")
    for rect, v in zip(b1, cost):
        ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}M", ha="center", va="bottom", fontsize=8)
    for rect, v in zip(b2, rmse):
        ax3.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = VIZ / "cost_accuracy_tradeoff.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    res, dems = load()
    outs = [fig_timeline(res), fig_dem_evolution(res, dems),
            fig_roi_localization(res, dems), fig_cost_accuracy(res)]
    manifest = {"figures": [str(o) for o in outs]}
    (VIZ / "viz_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
