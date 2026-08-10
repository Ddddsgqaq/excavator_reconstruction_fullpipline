"""
make_paper_figs.py — Mockup paper figures for the task-conditioned scene-graph
closed-loop (SCENE_GRAPH_PLAN.md v0.3.1). FOR GROUP-MEETING / PLANNING USE.

These figures illustrate the *form* the paper's figures will take. Numbers are
synthetic (a toy construction site + a vertical-compression noise model derived
from the project's measured 7.9-10.8x); they are NOT measured results. Every
figure is annotated "ILLUSTRATIVE / SYNTHETIC". Labels are English (ICRA venue).

Outputs -> output/paper_figs/F{1..6}_*.png   (no GPU, no services)

    python make_paper_figs.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

OUT = "output/paper_figs"
SEED = 7
VIDEO = "dynamic_execave_video.mp4"
WATERMARK = "ILLUSTRATIVE / SYNTHETIC — form mockup, not measured results"


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic scenario (used by F3/F4/F5/F6 so the story is consistent)
# ─────────────────────────────────────────────────────────────────────────────
def build_scenario(n=56, extent=22.0, seed=SEED):
    """A small cut_to_grade site: a high mound to cut down + a stockpile, on a
    designed flat grade. Returns DEMs, residual belief (mu, sigma), reach mask."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, extent, n)
    zs = np.linspace(0, extent, n)
    X, Z = np.meshgrid(xs, zs)

    def blob(cx, cz, r, h):
        return h * np.exp(-(((X - cx) ** 2 + (Z - cz) ** 2) / (2 * r ** 2)))

    base = 0.4 * np.sin(X / 6.0) * np.cos(Z / 7.0)
    mound = blob(8.0, 10.0, 2.6, 2.6)         # main cut target (well-viewed)
    pile = blob(16.5, 14.0, 2.0, 2.2)         # stockpile (context, out of reach)
    hollow = -blob(15.0, 6.5, 2.4, 1.1)       # a slight low spot (to fill)
    current = base + mound + pile + hollow
    current += 0.05 * rng.standard_normal(current.shape)
    current = gaussian_filter(current, 0.7)

    goal = np.zeros_like(current) + 0.0       # design grade = flat 0 plane
    residual_true = current - goal            # >0 to-cut, <0 to-fill

    gx, gz = np.gradient(current)
    relief = np.hypot(gx, gz)
    relief_n = relief / (relief.max() + 1e-9)

    # confidence is driven by VIEW GEOMETRY (parallax), NOT by relief: a
    # well-observed steep mound stays high-confidence. A reachable low-parallax
    # "blind patch" is the source of phantom residual (noise looks like to-cut).
    view_quality = np.clip(0.85 - 0.55 * (X / extent), 0.1, 0.95)
    blind = np.exp(-(((X - 12.5) ** 2 + (Z - 17.0) ** 2) / (2 * 2.6 ** 2)))
    view_quality = np.clip(view_quality - 0.7 * blind, 0.04, 0.95)
    confidence = np.clip(0.18 + 0.80 * view_quality - 0.08 * relief_n,
                         0.05, 0.97)
    confidence = gaussian_filter(confidence, 0.8)
    sigma = (1 - confidence)

    c = 1.0 + 6.0 * (1 - view_quality)        # compression worse where viewed poorly
    residual_obs_mu = residual_true / np.maximum(c, 1.0)

    exc = np.array([4.0, 13.5])
    R = np.hypot(X - exc[0], Z - exc[1])
    reachable = (R > 3.0) & (R < 11.0)

    return dict(X=X, Z=Z, xs=xs, zs=zs, extent=extent, n=n,
                current=current, goal=goal,
                residual_true=residual_true, residual_obs_mu=residual_obs_mu,
                sigma=sigma, confidence=confidence, comp_factor=c,
                exc=exc, reachable=reachable, cell_area=(extent / n) ** 2)


def utility_field(scen, recently_dug=None, w_slope=0.6):
    res = np.clip(scen["residual_obs_mu"], 0, None)
    gx, gz = np.gradient(scen["current"])
    slope_risk = np.hypot(gx, gz)
    slope_risk /= (slope_risk.max() + 1e-9)
    rd = np.zeros_like(res) if recently_dug is None else recently_dug
    U = res * scen["confidence"] * (1 - rd) - w_slope * slope_risk
    U = np.where(scen["reachable"], U, np.nan)
    return U


def _watermark(ax, txt=WATERMARK, loc="lower left"):
    x, y, ha = (0.01, 0.01, "left") if "left" in loc else (0.99, 0.01, "right")
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=6.5, color="0.35",
            ha=ha, va="bottom", style="italic")


def _grab_frame(frame_idx=214):
    try:
        import cv2
        v = cv2.VideoCapture(VIDEO)
        v.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = v.read()
        v.release()
        if ok:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:  # noqa: BLE001
        print(f"[frame grab failed: {e}]")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# F1 — closed-loop system overview (schematic)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_overview():
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    nodes = {
        "task":  (5.0, 6.2, "L4  Task\n(BIM goal terrain)", "#6a51a3"),
        "perc":  (1.7, 4.3, "Perception\nDEM + belief {mu,sigma}", "#2171b5"),
        "graph": (5.0, 4.0, "Scene Graph\nL0-L4 layered", "#238b45"),
        "dec":   (8.3, 4.3, "Decision\nnext-scoop  U(cell)", "#d94801"),
        "act":   (8.3, 1.6, "Action\ndig_cycle", "#b30000"),
        "maint": (5.0, 1.0, "Maintenance\nre-est. dug region only", "#08519c"),
        "world": (1.7, 1.6, "terrain changes\n-> re-perceive", "#525252"),
    }
    boxes = {}
    for k, (x, y, label, col) in nodes.items():
        b = FancyBboxPatch((x - 1.2, y - 0.5), 2.4, 1.0,
                           boxstyle="round,pad=0.06,rounding_size=0.12",
                           linewidth=1.6, edgecolor=col, facecolor=col + "22")
        ax.add_patch(b); boxes[k] = (x, y)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color="0.1")

    def arrow(a, b, col="0.3", rad=0.0, lab=None):
        (x0, y0), (x1, y1) = boxes[a], boxes[b]
        ar = FancyArrowPatch((x0, y0), (x1, y1),
                             connectionstyle=f"arc3,rad={rad}",
                             arrowstyle="-|>", mutation_scale=16,
                             shrinkA=44, shrinkB=44, linewidth=1.8, color=col)
        ax.add_patch(ar)
        if lab:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, lab, fontsize=7.5,
                    color=col, ha="center", style="italic")

    arrow("task", "perc", "#6a51a3", 0.1, "conditions")
    arrow("task", "graph", "#6a51a3", 0.0)
    arrow("task", "dec", "#6a51a3", -0.1)
    arrow("perc", "graph", "#238b45", 0.0, "belief -> region nodes")
    arrow("graph", "dec", "#d94801", 0.0, "predicates")
    arrow("dec", "act", "#b30000", 0.0)
    arrow("act", "maint", "#08519c", 0.0, "causal localization")
    arrow("maint", "world", "#525252", 0.0, "sigma down")
    arrow("world", "perc", "#525252", 0.1)
    ax.text(5.0, 5.05, "closed loop with terrain belief state as world model",
            ha="center", fontsize=10.5, color="0.25", fontweight="bold")
    ax.set_title("F1 · Task-conditioned scene-graph next-scoop loop (concept)",
                 fontsize=12)
    _watermark(ax)
    fig.savefig(f"{OUT}/F1_closed_loop_overview.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# F2 — layered scene graph over a real excavator frame
# ─────────────────────────────────────────────────────────────────────────────
def fig2_layered():
    frame = _grab_frame(214)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    if frame is not None:
        ax.imshow(frame)
    else:
        H, W = 278, 890
        ax.add_patch(Rectangle((0, 0), W, H, color="0.85")); ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
    ax.axis("off")

    machine = (430, 135)
    regions = {
        "pit_1 (to-cut 8.0+/-0.6 m3)": (250, 215, "pit", "#d94801"),
        "pile_1 (stockpile)":          (650, 195, "pile", "#6a51a3"),
        "flat_1 (on-grade)":           (470, 245, "flat", "#238b45"),
    }
    edges = [(machine, regions["pit_1 (to-cut 8.0+/-0.6 m3)"][:2],
              "diggable", "#d94801"),
             (machine, regions["pile_1 (stockpile)"][:2],
              "reachable", "#6a51a3")]
    for p0, p1, lab, col in edges:
        ax.add_line(Line2D([p0[0], p1[0]], [p0[1], p1[1]], color=col,
                    lw=2.2, alpha=0.9, zorder=3))
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 - 6, lab, color=col,
                fontsize=8.5, ha="center", style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.7))
    for name, (px, py, typ, col) in regions.items():
        ax.add_patch(Circle((px, py), 11, color=col, ec="white", lw=1.5,
                            zorder=5))
        ax.text(px, py + 22, name, color=col, fontsize=8.5, ha="center",
                va="top", zorder=6, bbox=dict(boxstyle="round,pad=0.2",
                fc="white", ec=col, alpha=0.85))
    ax.add_patch(Circle(machine, 13, color="#b30000", ec="white", lw=2,
                        zorder=7))
    ax.text(machine[0], machine[1] - 22, "excavator (L2 agent)  state=digging",
            color="#b30000", fontsize=9, ha="center", va="bottom", zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#b30000",
                      alpha=0.9))
    ax.text(8, 14, "L4 Task: cut_to_grade -> goal DEM (BIM)", fontsize=9.5,
            color="#6a51a3", va="top", zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#6a51a3"))
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                     label=l, markersize=9)
              for l, c in [("L2 machine node", "#b30000"),
                           ("L1 region node", "#d94801"),
                           ("L3 affordance edge", "#238b45"),
                           ("L4 task", "#6a51a3")]]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("F2 · Layered scene graph over a real excavator frame (L0-L4)",
                 fontsize=12)
    _watermark(ax, loc="lower left")
    fig.savefig(f"{OUT}/F2_layered_scene_graph.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# F3 — residual as a belief field
# ─────────────────────────────────────────────────────────────────────────────
def fig3_belief(scen):
    ext = [0, scen["extent"], 0, scen["extent"]]
    fig, axs = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    im0 = axs[0, 0].imshow(scen["current"], origin="lower", extent=ext,
                           cmap="gist_earth")
    axs[0, 0].set_title("(a) current DEM (monocular reconstruction)")
    fig.colorbar(im0, ax=axs[0, 0], shrink=0.8, label="height (m)")

    im1 = axs[0, 1].imshow(scen["goal"], origin="lower", extent=ext,
                           cmap="gist_earth",
                           vmin=scen["current"].min(), vmax=scen["current"].max())
    axs[0, 1].set_title("(b) goal DEM (design grade / BIM)")
    fig.colorbar(im1, ax=axs[0, 1], shrink=0.8, label="height (m)")

    vmax = np.abs(scen["residual_obs_mu"]).max()
    im2 = axs[1, 0].imshow(scen["residual_obs_mu"], origin="lower", extent=ext,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axs[1, 0].set_title("(c) residual mean mu = current - goal\n"
                        "(red = to-cut, blue = to-fill)")
    fig.colorbar(im2, ax=axs[1, 0], shrink=0.8, label="residual (m)")

    im3 = axs[1, 1].imshow(scen["sigma"], origin="lower", extent=ext,
                           cmap="magma")
    axs[1, 1].set_title("(d) uncertainty sigma (vertical-compression zones)\n"
                        "high sigma = low parallax / steep relief")
    fig.colorbar(im3, ax=axs[1, 1], shrink=0.8, label="sigma (norm.)")
    for ax in axs.ravel():
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    fig.suptitle("F3 · Residual as a belief field {mu, sigma}: "
                 "the core decidable state in the scene graph", fontsize=13)
    _watermark(axs[1, 1], loc="lower right")
    fig.savefig(f"{OUT}/F3_residual_belief_field.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# F4 — next-scoop decision
# ─────────────────────────────────────────────────────────────────────────────
def fig4_decision(scen):
    U = utility_field(scen)
    flat = np.where(np.isfinite(U), U, -np.inf)
    iz, ix = np.unravel_index(np.argmax(flat), flat.shape)
    cx, cz = scen["xs"][ix], scen["zs"][iz]
    exc = scen["exc"]
    bearing = np.degrees(np.arctan2(cz - exc[1], cx - exc[0]))
    mu = float(scen["residual_obs_mu"][iz, ix])
    conf = float(scen["confidence"][iz, ix])

    fig = plt.figure(figsize=(13, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.9])

    ax0 = fig.add_subplot(gs[0, 0]); frame = _grab_frame(214)
    if frame is not None:
        ax0.imshow(frame)
        ax0.annotate("", xy=(300, 175), xytext=(430, 140),
                     arrowprops=dict(arrowstyle="-|>", color="#00d0ff", lw=3))
        ax0.text(300, 205, "next-scoop\ndirection", color="#00d0ff",
                 fontsize=10, ha="center", fontweight="bold")
    ax0.axis("off"); ax0.set_title("(a) recommended scoop projected to view")

    ax1 = fig.add_subplot(gs[0, 1]); ext = [0, scen["extent"], 0, scen["extent"]]
    im = ax1.imshow(U, origin="lower", extent=ext, cmap="magma")
    fig.colorbar(im, ax=ax1, shrink=0.85, label="U(cell)")
    th = np.linspace(0, 2 * np.pi, 100)
    for r in (3.0, 11.0):
        ax1.plot(exc[0] + r * np.cos(th), exc[1] + r * np.sin(th),
                 "--", color="cyan", lw=1, alpha=0.7)
    ax1.plot(*exc, "s", color="cyan", ms=10, label="excavator")
    ax1.plot(cx, cz, "*", color="#39ff14", ms=24, mec="k", label="next-scoop")
    ax1.annotate("", xy=(cx, cz), xytext=tuple(exc),
                 arrowprops=dict(arrowstyle="-|>", color="#39ff14", lw=2.2))
    ax1.set_title("(b) U(cell) decision field + reach ring + recommendation")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis("off")
    txt = (
        "recommendation (graph object)\n"
        "-----------------------------\n"
        f"target_region : pit_1\n"
        f"cell (x,z)     : ({cx:.1f}, {cz:.1f}) m\n"
        f"approach       : {bearing:.0f} deg (rel. exc)\n"
        f"expected_vol   : {mu:.2f} m3\n"
        f"confidence     : {conf:.2f}\n\n"
        "justified_by:\n"
        "  - diggable(exc, pit_1)\n"
        f"  - cut_remaining(cell)={mu:.1f}+/-{(1-conf):.1f}\n"
        "  - reachable(exc, cell)\n"
        "  - slope_stable(cell)\n\n"
        "=> decision traceable to\n   L3 predicates (grounding)"
    )
    ax2.text(0.0, 0.98, txt, va="top", ha="left", fontsize=9.5,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f3f3f3", ec="#d94801"))
    ax2.set_title("(c) decision written into graph + provenance")
    fig.suptitle("F4 · next-scoop decision (uncertainty-weighted greedy: "
                 "U = E[dResidual]*conf - w*slope)", fontsize=12.5)
    _watermark(ax1, loc="lower right")
    fig.savefig(f"{OUT}/F4_next_scoop_decision.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# F5 — closed-loop convergence / ablation (simulated)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_closedloop(scen, n_steps=70):
    rng = np.random.default_rng(SEED + 1)
    true0 = np.clip(scen["residual_true"], 0, None) * scen["reachable"]
    conf = scen["confidence"]; c = scen["comp_factor"]
    reach = scen["reachable"]; cell_area = scen["cell_area"]
    X, Z = scen["X"], scen["Z"]
    V0 = (true0 * cell_area).sum()
    bucket = V0 / (n_steps * 0.8)             # oracle empties ~at 0.8*n_steps

    def run(policy):
        true = true0.copy()
        rd = np.zeros_like(true)
        rem, wasted = [], 0
        for _ in range(n_steps):
            noise = (1 - conf) * rng.standard_normal(true.shape) * 0.5
            obs = true / np.maximum(c, 1.0) + noise
            if policy == "greedy":
                score = np.where(reach, obs, -np.inf)
            elif policy == "ours":
                score = np.where(reach, np.clip(obs, 0, None) * conf
                                 * (1 - rd), -np.inf)
            else:  # oracle upper bound
                score = np.where(reach, true, -np.inf)
            iz, ix = np.unravel_index(np.argmax(score), score.shape)
            cx, cz = scen["xs"][ix], scen["zs"][iz]
            disk = np.hypot(X - cx, Z - cz) < 1.3
            avail = float((true * disk).sum() * cell_area)
            if avail < 0.02:                  # dug a phantom (no real material)
                wasted += 1
            else:                             # remove up to a bucket, proportional
                take = min(bucket, avail)
                true[disk] *= (1 - take / avail)
            rd *= 0.85; rd[disk] = 1.0
            rem.append(float((true * cell_area).sum()))
        return np.array(rem), wasted

    rem_g, w_g = run("greedy")
    rem_o, w_o = run("ours")
    rem_or, w_or = run("oracle")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [2, 1]},
                                   constrained_layout=True)
    steps = np.arange(1, n_steps + 1)
    ax1.axhline(V0, color="0.7", ls=":", lw=1)
    ax1.text(1, V0, f" initial to-cut {V0:.1f} m3", va="bottom", fontsize=8,
             color="0.4")
    ax1.plot(steps, rem_g, "-o", ms=3, color="#b30000",
             label="(1) greedy max-residual (no confidence)")
    ax1.plot(steps, rem_o, "-s", ms=3, color="#238b45",
             label="(2) uncertainty-weighted (ours)")
    ax1.plot(steps, rem_or, "--", color="0.3",
             label="(3) oracle / info-gain upper bound")
    ax1.set_xlabel("scoops"); ax1.set_ylabel("true remaining to-cut volume (m3)")
    ax1.set_title("(a) closed-loop convergence: remaining residual vs scoops")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    bars = ax2.bar(["(1) greedy", "(2) ours", "(3) oracle"],
                   [w_g, w_o, w_or],
                   color=["#b30000", "#238b45", "0.5"])
    ax2.bar_label(bars, fontsize=10)
    ax2.set_ylabel("scoops wasted on phantom residual")
    ax2.set_title("(b) scoops fooled by compression artifacts")
    fig.suptitle("F5 · Closed-loop ablation: uncertainty weighting avoids "
                 "phantom-residual scoops, converges faster (simulated)",
                 fontsize=12.5)
    _watermark(ax2, loc="lower right")
    fig.savefig(f"{OUT}/F5_closed_loop_ablation.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# F6 — action-conditioned maintenance
# ─────────────────────────────────────────────────────────────────────────────
def fig6_maintenance(scen):
    ext = [0, scen["extent"], 0, scen["extent"]]
    exc = scen["exc"]
    U = utility_field(scen)
    flat = np.where(np.isfinite(U), U, -np.inf)
    iz, ix = np.unravel_index(np.argmax(flat), flat.shape)
    cx, cz = scen["xs"][ix], scen["zs"][iz]
    n = scen["n"]
    reg = np.hypot(scen["X"] - cx, scen["Z"] - cz) < 3.0

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8),
                            gridspec_kw={"width_ratios": [1, 1, 0.8]},
                            constrained_layout=True)
    full = np.ones((n, n))
    axs[0].imshow(full, origin="lower", extent=ext, cmap="Reds", vmin=0, vmax=1.5)
    axs[0].set_title("(a) per-frame full rebuild\nO(whole scene) every frame")
    mask = np.where(reg, 1.0, 0.15)
    axs[1].imshow(mask, origin="lower", extent=ext, cmap="Greens", vmin=0,
                  vmax=1.5)
    axs[1].plot(cx, cz, "*", color="k", ms=18)
    axs[1].add_patch(plt.Circle((cx, cz), 3.0, fill=False, ec="k", lw=1.5,
                                ls="--"))
    axs[1].set_title("(b) action-conditioned maintenance (mech.5)\n"
                     "dig event causally localizes -> re-est. dug region only")
    for ax in axs[:2]:
        ax.plot(*exc, "s", color="cyan", ms=9)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")

    axs[2].axis("off")
    n_cells_full = n * n
    n_cells_reg = int(reg.sum())
    sig_before = float(scen["sigma"][reg].mean())
    sig_after = sig_before * 0.45
    txt = (
        "update cost (per event)\n"
        "-----------------------\n"
        f"full rebuild : {n_cells_full} cells\n"
        f"ours         : {n_cells_reg} cells\n"
        f"speedup      : ~{n_cells_full / max(n_cells_reg,1):.0f}x\n\n"
        "dug-region uncertainty sigma\n"
        "-----------------------\n"
        f"before : {sig_before:.2f}\n"
        f"after  : {sig_after:.2f}  (down)\n"
        "(known scooped volume as\n metric constraint + re-obs)"
    )
    axs[2].text(0.0, 0.97, txt, va="top", ha="left", fontsize=10.5,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f3f3f3", ec="#08519c"))
    axs[2].set_title("(c) cost and sigma drop")
    fig.suptitle("F6 · Action-conditioned incremental maintenance: "
                 "O(single region)/event replaces O(whole scene)/frame",
                 fontsize=12.5)
    _watermark(axs[1], loc="lower right")
    fig.savefig(f"{OUT}/F6_action_conditioned_maintenance.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    scen = build_scenario()
    fig1_overview();              print("F1 done")
    fig2_layered();               print("F2 done")
    fig3_belief(scen);            print("F3 done")
    fig4_decision(scen);          print("F4 done")
    fig5_closedloop(scen);        print("F5 done")
    fig6_maintenance(scen);       print("F6 done")
    print(f"\nAll figures -> {OUT}/")


if __name__ == "__main__":
    main()
