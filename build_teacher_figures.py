#!/usr/bin/env python3
"""Build presentation-ready composite figures from existing experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "teacher_figures"
OUT.mkdir(exist_ok=True)

FONT_REGULAR_PATH = Path("/mnt/c/Windows/Fonts/simhei.ttf")
FONT_BOLD_PATH = Path("/mnt/c/Windows/Fonts/simhei.ttf")
FONT = FontProperties(fname=str(FONT_REGULAR_PATH)) if FONT_REGULAR_PATH.exists() else None
FONT_BOLD = FontProperties(fname=str(FONT_BOLD_PATH)) if FONT_BOLD_PATH.exists() else FONT

NAVY = "#15324B"
BLUE = "#2878B5"
CYAN = "#4EA8DE"
ORANGE = "#F28E2B"
GREEN = "#2E9D65"
RED = "#D84A4A"
GRAY = "#667784"
LIGHT = "#F4F7F9"
GRID = "#D8E1E7"
DARK = "#1F2D38"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.edgecolor": GRID,
        "axes.labelcolor": DARK,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def text(ax, x, y, s, *, size=14, color=DARK, weight="normal", ha="left", va="center", **kwargs):
    fp = FONT_BOLD if weight == "bold" else FONT
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        fontproperties=fp,
        ha=ha,
        va=va,
        **kwargs,
    )


def fig_text(fig, x, y, s, *, size=14, color=DARK, weight="normal", ha="left", va="center", **kwargs):
    fp = FONT_BOLD if weight == "bold" else FONT
    return fig.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        fontproperties=fp,
        ha=ha,
        va=va,
        **kwargs,
    )


def title_block(fig, title, subtitle, figure_no):
    fig_text(fig, 0.035, 0.965, f"FIGURE {figure_no}", size=11, color=BLUE, weight="bold")
    fig_text(fig, 0.035, 0.925, title, size=28, color=NAVY, weight="bold")
    fig_text(fig, 0.035, 0.885, subtitle, size=13, color=GRAY)
    fig.add_artist(
        mpl.lines.Line2D([0.035, 0.965], [0.86, 0.86], transform=fig.transFigure, color=GRID, lw=1.2)
    )


def panel_label(ax, letter, title):
    text(
        ax,
        0.0,
        1.04,
        letter,
        size=14,
        color="white",
        weight="bold",
        ha="center",
        va="center",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": NAVY, "edgecolor": NAVY},
        clip_on=False,
    )
    text(ax, 0.045, 1.04, title, size=15, color=NAVY, weight="bold", transform=ax.transAxes, clip_on=False)


def card(ax, xy, width, height, *, face="white", edge=GRID, radius=0.025, lw=1.2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=BLUE, lw=2.2, mutation=16, style="-|>"):
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        transform=ax.transAxes,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)
    return a


def load_image(path):
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, "white")
        bg.alpha_composite(rgba)
        return np.asarray(bg.convert("RGB"))


def show_image(ax, path, *, label=None):
    ax.imshow(load_image(ROOT / path))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if label:
        text(
            ax,
            0.02,
            0.97,
            label,
            size=11,
            color="white",
            weight="bold",
            va="top",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": NAVY, "edgecolor": "none", "alpha": 0.92},
        )


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=180, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / f"{stem}.pdf", dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def draw_camera_frames(ax, x, y, w, h):
    for i, offset in enumerate([0.0, 0.018, 0.036]):
        xx = x + offset
        yy = y + offset * 0.6
        ax.add_patch(
            FancyBboxPatch(
                (xx, yy),
                w,
                h,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor="#DDECF6" if i < 2 else "#B9D9ED",
                edgecolor=BLUE,
                linewidth=1.3,
                transform=ax.transAxes,
            )
        )
        ax.add_patch(
            Polygon(
                [(xx + 0.08 * w, yy + 0.12 * h), (xx + 0.43 * w, yy + 0.68 * h), (xx + 0.7 * w, yy + 0.26 * h)],
                closed=True,
                facecolor=GREEN,
                alpha=0.75,
                edgecolor="none",
                transform=ax.transAxes,
            )
        )


def draw_point_cloud(ax, cx, cy, rx, ry, seed=7):
    rng = np.random.default_rng(seed)
    n = 130
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(0, 1, n))
    xs = cx + rx * r * np.cos(theta)
    ys = cy + ry * r * np.sin(theta) + 0.05 * np.sin(xs * 50)
    colors = np.where(xs < cx, ORANGE, BLUE)
    ax.scatter(xs, ys, s=7, c=colors, alpha=0.75, transform=ax.transAxes, clip_on=False)


def draw_tile_grid(ax, x, y, w, h, values=None, semantic=False):
    values = values if values is not None else np.arange(9).reshape(3, 3)
    rows, cols = values.shape
    if semantic:
        palette = [LIGHT, ORANGE, BLUE, GREEN, RED, "#6E6E6E"]
        for r in range(rows):
            for c in range(cols):
                v = int(values[r, c])
                color = "#E8EEF2" if v < 0 else palette[v % len(palette)]
                ax.add_patch(
                    Rectangle(
                        (x + c * w / cols, y + (rows - 1 - r) * h / rows),
                        w / cols,
                        h / rows,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.7,
                        transform=ax.transAxes,
                    )
                )
    else:
        norm = mpl.colors.Normalize(np.nanmin(values), np.nanmax(values))
        cmap = plt.get_cmap("terrain")
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(
                    Rectangle(
                        (x + c * w / cols, y + (rows - 1 - r) * h / rows),
                        w / cols,
                        h / rows,
                        facecolor=cmap(norm(values[r, c])),
                        edgecolor="white",
                        linewidth=0.7,
                        transform=ax.transAxes,
                    )
                )
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=NAVY, linewidth=1.2, transform=ax.transAxes))


def figure_system_overview():
    fig = plt.figure(figsize=(18, 10))
    title_block(
        fig,
        "从单目视频到可决策施工地形：近期系统能力总览",
        "共享世界坐标、持续融合、语义作业分区与下游数据接口被整合为一条完整链路",
        "01",
    )
    ax = fig.add_axes([0.035, 0.11, 0.93, 0.72])
    ax.set_axis_off()

    xs = [0.015, 0.215, 0.415, 0.615, 0.815]
    titles = ["真实视频输入", "关键帧与局部重建", "稳定配准与全局 DEM", "语义地形理解", "决策与应用接口"]
    subtitles = [
        "航拍 / USB / RTSP / HTTP",
        "去重采样 · VGGT 推理",
        "冻结锚点 · 持久融合 · 多 Tile",
        "YOLOe × 高程 × 坡度",
        "下一铲 · VLM · Unity JSON",
    ]
    stage_colors = ["#E9F3F9", "#EEF3FA", "#EDF7F2", "#FFF3E6", "#F1EEF8"]

    for i, (x, t, st, fc) in enumerate(zip(xs, titles, subtitles, stage_colors)):
        card(ax, (x, 0.48), 0.17, 0.43, face=fc, edge=GRID, radius=0.02)
        text(ax, x + 0.018, 0.865, f"0{i+1}", size=11, color=BLUE, weight="bold", transform=ax.transAxes)
        text(ax, x + 0.018, 0.81, t, size=15, color=NAVY, weight="bold", transform=ax.transAxes)
        text(ax, x + 0.018, 0.755, st, size=10.5, color=GRAY, transform=ax.transAxes)
        if i < 4:
            arrow(ax, (x + 0.176, 0.69), (xs[i + 1] - 0.008, 0.69), color=BLUE)

    draw_camera_frames(ax, xs[0] + 0.028, 0.55, 0.09, 0.13)
    text(ax, xs[0] + 0.085, 0.525, "连续取流", size=10.5, color=GRAY, ha="center", transform=ax.transAxes)

    draw_camera_frames(ax, xs[1] + 0.02, 0.61, 0.055, 0.085)
    draw_point_cloud(ax, xs[1] + 0.115, 0.61, 0.045, 0.065)
    text(ax, xs[1] + 0.085, 0.525, "关键帧 → 局部点云", size=10.5, color=GRAY, ha="center", transform=ax.transAxes)

    vals = np.array([[0.2, 0.35, 0.5], [0.3, 0.65, 0.72], [0.15, 0.42, 0.6]])
    draw_tile_grid(ax, xs[2] + 0.035, 0.55, 0.1, 0.14, vals)
    text(ax, xs[2] + 0.085, 0.525, "全局地图持续长大", size=10.5, color=GRAY, ha="center", transform=ax.transAxes)

    sem = np.array([[1, 1, 0, 2], [1, 4, 0, 2], [3, 3, 5, 0], [3, 0, 0, 0]])
    draw_tile_grid(ax, xs[3] + 0.035, 0.55, 0.1, 0.14, sem, semantic=True)
    ax.scatter([xs[3] + 0.079], [0.598], marker="*", s=130, color="white", edgecolor=NAVY, linewidth=0.9, transform=ax.transAxes, zorder=5)
    text(ax, xs[3] + 0.085, 0.525, "分区 + 障碍 + 下一铲", size=10.5, color=GRAY, ha="center", transform=ax.transAxes)

    draw_tile_grid(ax, xs[4] + 0.02, 0.57, 0.065, 0.11, vals)
    arrow(ax, (xs[4] + 0.09, 0.625), (xs[4] + 0.13, 0.625), color=ORANGE, mutation=13)
    draw_tile_grid(ax, xs[4] + 0.135, 0.57, 0.03, 0.11, sem[:, :2], semantic=True)
    text(ax, xs[4] + 0.085, 0.525, "高程与语义共享栅格", size=10.5, color=GRAY, ha="center", transform=ax.transAxes)

    card(ax, (0.015, 0.06), 0.97, 0.29, face="white", edge=GRID, radius=0.018)
    text(ax, 0.035, 0.31, "两阶段可信更新机制", size=15, color=NAVY, weight="bold", transform=ax.transAxes)
    states = [
        ("初始化", "较大关键帧集"),
        ("质量审核", "覆盖率 / 点数 / 高程跨度"),
        ("READY", "冻结基准地图"),
        ("可信更新", "配准 + 变化门控"),
        ("版本推进", "只提交通过的 Tile"),
    ]
    sx = [0.04, 0.23, 0.42, 0.61, 0.80]
    for i, ((name, desc), xx) in enumerate(zip(states, sx)):
        card(ax, (xx, 0.12), 0.15, 0.115, face=LIGHT if i != 2 else "#E5F4EA", edge=GRID, radius=0.015)
        text(ax, xx + 0.075, 0.19, name, size=12, color=NAVY, weight="bold", ha="center", transform=ax.transAxes)
        text(ax, xx + 0.075, 0.145, desc, size=9.2, color=GRAY, ha="center", transform=ax.transAxes)
        if i < 4:
            arrow(ax, (xx + 0.155, 0.177), (sx[i + 1] - 0.006, 0.177), color=GREEN, mutation=13)
    text(
        ax,
        0.5,
        0.075,
        "更新失败时保留上一版本；连续拒绝进入 DEGRADED / REINIT_REQUIRED",
        size=10,
        color=RED,
        ha="center",
        transform=ax.transAxes,
    )
    save(fig, "01_system_overview")


def figure_dem_growth():
    fig = plt.figure(figsize=(18, 10))
    title_block(
        fig,
        "全局 DEM 在视频观测中持续生长",
        "同一航拍视频的真实 GPU 冒烟验证：持久融合使已观测地形逐轮增加，并稳定发布多个 Tile",
        "02",
    )
    gs = fig.add_gridspec(2, 5, left=0.04, right=0.97, bottom=0.08, top=0.82, height_ratios=[1.12, 0.72], hspace=0.22, wspace=0.08)
    cells = np.array([11770, 16633, 19279, 21007, 25666])
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        show_image(ax, f"verify_m5_viz/global_dem_pass{i+1}.png", label=f"PASS {i+1}")
        text(
            ax,
            0.5,
            -0.03,
            f"{cells[i]:,} 个已观测格",
            size=11,
            color=NAVY,
            weight="bold",
            ha="center",
            va="top",
            transform=ax.transAxes,
            clip_on=False,
        )
        if i < 4:
            ax.annotate(
                "",
                xy=(1.08, 0.5),
                xytext=(1.0, 0.5),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 2.0},
                annotation_clip=False,
            )

    ax_line = fig.add_subplot(gs[1, :3])
    panel_label(ax_line, "A", "覆盖增长曲线")
    x = np.arange(1, 6)
    ax_line.plot(x, cells, color=BLUE, marker="o", lw=3, ms=8)
    ax_line.fill_between(x, cells, cells.min() * 0.9, color=CYAN, alpha=0.14)
    for xi, yi in zip(x, cells):
        text(ax_line, xi, yi + 650, f"{yi:,}", size=10.5, color=NAVY, weight="bold", ha="center")
    ax_line.set_xticks(x)
    ax_line.set_xticklabels([f"Pass {i}" for i in x], fontproperties=FONT)
    ax_line.set_ylabel("observed cells", fontproperties=FONT)
    ax_line.set_ylim(9500, 28500)
    ax_line.grid(axis="y", color=GRID, lw=0.8)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    text(
        ax_line,
        0.02,
        0.9,
        "相对 Pass 1：+118%",
        size=13,
        color=GREEN,
        weight="bold",
        transform=ax_line.transAxes,
    )

    ax_tile = fig.add_subplot(gs[1, 3:])
    ax_tile.set_axis_off()
    panel_label(ax_tile, "B", "持久融合与 Tile 发布")
    card(ax_tile, (0.02, 0.08), 0.96, 0.82, face=LIGHT, edge=GRID, radius=0.02)
    draw_tile_grid(ax_tile, 0.08, 0.25, 0.35, 0.47, np.array([[0.1, 0.2, 0.3], [0.3, 0.7, 0.9], [0.2, 0.5, 0.6]]))
    observed = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    for r, c in observed:
        ax_tile.add_patch(
            Rectangle(
                (0.08 + c * 0.35 / 3, 0.25 + (2 - r) * 0.47 / 3),
                0.35 / 3,
                0.47 / 3,
                fill=False,
                edgecolor="white",
                linewidth=3,
                transform=ax_tile.transAxes,
            )
        )
    text(ax_tile, 0.57, 0.68, "6", size=32, color=BLUE, weight="bold", ha="center", transform=ax_tile.transAxes)
    text(ax_tile, 0.57, 0.56, "个不同 Tile", size=12, color=NAVY, weight="bold", ha="center", transform=ax_tile.transAxes)
    text(ax_tile, 0.81, 0.68, "21", size=32, color=ORANGE, weight="bold", ha="center", transform=ax_tile.transAxes)
    text(ax_tile, 0.81, 0.56, "次累计更新", size=12, color=NAVY, weight="bold", ha="center", transform=ax_tile.transAxes)
    text(
        ax_tile,
        0.57,
        0.31,
        "changed tiles / round\n6, 6, 3, 4, 4, 4, 4, 4",
        size=11,
        color=GRAY,
        ha="center",
        transform=ax_tile.transAxes,
    )
    text(
        ax_tile,
        0.81,
        0.31,
        "变化检测逐步收敛\n地图仍持续累积",
        size=11,
        color=GRAY,
        ha="center",
        transform=ax_tile.transAxes,
    )
    save(fig, "02_dem_growth")


def figure_registration_ablation():
    fig = plt.figure(figsize=(18, 10))
    title_block(
        fig,
        "轮间配准降低跨轮地图漂移：消融对照",
        "冻结 footprint 保证坐标框架；重叠足够时进一步执行水平配准，不足时安全回退到冻结锚点",
        "03",
    )
    gs = fig.add_gridspec(3, 2, left=0.045, right=0.97, bottom=0.08, top=0.82, height_ratios=[0.82, 0.82, 0.9], hspace=0.26, wspace=0.14)

    ax_on = fig.add_subplot(gs[0, :])
    show_image(ax_on, "verify_m4_viz/m4_ON_registration.png", label="Registration ON")
    panel_label(ax_on, "A", "冻结锚点 + 轮间配准")

    ax_off = fig.add_subplot(gs[1, :])
    show_image(ax_off, "verify_m4_viz/m4_OFF_footprint_only.png", label="Registration OFF")
    panel_label(ax_off, "B", "仅冻结 footprint")

    ax_bar = fig.add_subplot(gs[2, 0])
    panel_label(ax_bar, "C", "跨轮 DEM RMS")
    vals = [0.0077, 0.0056]
    bars = ax_bar.bar(["仅固定 footprint", "增加配准"], vals, color=[GRAY, BLUE], width=0.58)
    for b, v in zip(bars, vals):
        text(ax_bar, b.get_x() + b.get_width() / 2, v + 0.00022, f"{v:.4f} m", size=12, color=NAVY, weight="bold", ha="center")
    ax_bar.set_ylim(0, 0.0094)
    ax_bar.set_ylabel("RMS drift (m)", fontproperties=FONT)
    ax_bar.set_xticklabels(["仅固定 footprint", "增加配准"], fontproperties=FONT)
    ax_bar.grid(axis="y", color=GRID, lw=0.8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    text(
        ax_bar,
        0.7,
        0.82,
        "↓ 27%",
        size=24,
        color=GREEN,
        weight="bold",
        ha="center",
        transform=ax_bar.transAxes,
    )

    ax_mech = fig.add_subplot(gs[2, 1])
    ax_mech.set_axis_off()
    panel_label(ax_mech, "D", "稳健的“配准—回退”机制")
    card(ax_mech, (0.02, 0.05), 0.96, 0.85, face=LIGHT, edge=GRID, radius=0.02)
    boxes = [
        (0.07, "新一轮\n局部 DEM", "#EAF3F9"),
        (0.37, "重叠充分？", "#FFF3E6"),
        (0.70, "融合到\n全局地图", "#E8F4EC"),
    ]
    for x, label, fc in boxes:
        card(ax_mech, (x, 0.39), 0.2, 0.26, face=fc, edge=GRID, radius=0.018)
        text(ax_mech, x + 0.1, 0.52, label, size=13, color=NAVY, weight="bold", ha="center", transform=ax_mech.transAxes)
    arrow(ax_mech, (0.275, 0.52), (0.36, 0.52), color=BLUE)
    arrow(ax_mech, (0.575, 0.56), (0.69, 0.56), color=GREEN)
    arrow(ax_mech, (0.475, 0.385), (0.69, 0.28), color=GRAY)
    text(ax_mech, 0.63, 0.63, "是：估计 yaw / 平移", size=10.5, color=GREEN, ha="center", transform=ax_mech.transAxes)
    text(ax_mech, 0.55, 0.26, "否：复用冻结锚点", size=10.5, color=GRAY, ha="center", transform=ax_mech.transAxes)
    text(
        ax_mech,
        0.50,
        0.13,
        "收敛轮：yaw≈60.6°，RMSE=0.0149 m；未收敛轮不强行配准",
        size=10.2,
        color=DARK,
        ha="center",
        transform=ax_mech.transAxes,
    )
    save(fig, "03_registration_ablation")


def figure_semantic_decision():
    session = "session_20260617_172521_478306"
    decision_path = ROOT / "vlm_report" / session / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(18, 10))
    title_block(
        fig,
        "从几何地形到作业决策：语义、风险与可执行数据",
        "几何模块负责可信坐标和区域计算；VLM负责高层语义、风险解释与作业顺序",
        "04",
    )
    gs = fig.add_gridspec(2, 3, left=0.04, right=0.97, bottom=0.07, top=0.82, height_ratios=[1.1, 0.68], hspace=0.18, wspace=0.1)

    ax_bev = fig.add_subplot(gs[0, 0])
    show_image(ax_bev, f"vlm_report/{session}/bev_for_vlm.png")
    panel_label(ax_bev, "A", "语义作业地图")

    ax_overlay = fig.add_subplot(gs[0, 1])
    show_image(ax_overlay, f"vlm_report/{session}/decision_overlay.png")
    panel_label(ax_overlay, "B", "决策叠加结果")

    ax_packet = fig.add_subplot(gs[0, 2])
    ax_packet.set_axis_off()
    panel_label(ax_packet, "C", "同栅格高程 + 语义数据包")
    card(ax_packet, (0.04, 0.04), 0.92, 0.9, face=LIGHT, edge=GRID, radius=0.02)
    vals = np.array([[0.1, 0.3, 0.4, 0.5], [0.2, 0.6, 0.7, 0.5], [0.3, 0.8, 0.9, 0.4], [0.2, 0.4, 0.5, 0.3]])
    sem = np.array([[1, 1, 0, 2], [1, 4, 0, 2], [3, 3, 5, 0], [3, 0, 0, 0]])
    draw_tile_grid(ax_packet, 0.10, 0.53, 0.30, 0.27, vals)
    draw_tile_grid(ax_packet, 0.60, 0.53, 0.30, 0.27, sem, semantic=True)
    text(ax_packet, 0.25, 0.84, "Elevation int16", size=11, color=NAVY, weight="bold", ha="center", transform=ax_packet.transAxes)
    text(ax_packet, 0.75, 0.84, "Semantic zone", size=11, color=NAVY, weight="bold", ha="center", transform=ax_packet.transAxes)
    arrow(ax_packet, (0.42, 0.665), (0.58, 0.665), color=ORANGE)
    text(ax_packet, 0.5, 0.72, "逐格对齐", size=10, color=ORANGE, weight="bold", ha="center", transform=ax_packet.transAxes)
    code = (
        '{\n'
        '  "tile_x": -1, "tile_y": -2,\n'
        '  "width": 128, "height": 128,\n'
        '  "height_resolution": 0.01,\n'
        '  "data_type": "int16",\n'
        '  "semantic": {"layer_type": "zone"}\n'
        '}'
    )
    text(
        ax_packet,
        0.11,
        0.37,
        code,
        size=9.2,
        color=DARK,
        family="monospace",
        va="top",
        transform=ax_packet.transAxes,
    )
    text(
        ax_packet,
        0.63,
        0.31,
        "下游可切换\n高程 / 语义 / 网格\n并读取可挖掩码",
        size=11,
        color=NAVY,
        weight="bold",
        ha="center",
        transform=ax_packet.transAxes,
    )

    ax_logic = fig.add_subplot(gs[1, :2])
    ax_logic.set_axis_off()
    panel_label(ax_logic, "D", "职责分工：数值交给几何，策略交给 VLM")
    card(ax_logic, (0.01, 0.08), 0.47, 0.78, face="#EAF3F9", edge=GRID, radius=0.02)
    card(ax_logic, (0.52, 0.08), 0.47, 0.78, face="#FFF3E6", edge=GRID, radius=0.02)
    text(ax_logic, 0.245, 0.74, "几何模块（确定性）", size=15, color=BLUE, weight="bold", ha="center", transform=ax_logic.transAxes)
    text(
        ax_logic,
        0.245,
        0.46,
        "— 高程、坡度、粗糙度\n— 区域边界与连通域\n— 机械位置与 next-scoop 坐标\n— 数值越界检查",
        size=12,
        color=DARK,
        ha="center",
        transform=ax_logic.transAxes,
        linespacing=1.6,
    )
    text(ax_logic, 0.755, 0.74, "VLM（解释与排序）", size=15, color=ORANGE, weight="bold", ha="center", transform=ax_logic.transAxes)
    text(
        ax_logic,
        0.755,
        0.46,
        "— 场景摘要与区域理解\n— 风险说明与优先级\n— 作业顺序和动作语义\n— 不直接承担精确坐标计算",
        size=12,
        color=DARK,
        ha="center",
        transform=ax_logic.transAxes,
        linespacing=1.6,
    )

    ax_dec = fig.add_subplot(gs[1, 2])
    ax_dec.set_axis_off()
    panel_label(ax_dec, "E", "该场景的结构化输出")
    card(ax_dec, (0.02, 0.08), 0.96, 0.78, face=LIGHT, edge=GRID, radius=0.02)
    action = decision["next_action"]["action"].upper()
    target = decision["next_action"]["target_xz"]
    text(ax_dec, 0.08, 0.72, action, size=23, color=GREEN, weight="bold", transform=ax_dec.transAxes)
    text(ax_dec, 0.08, 0.58, f"目标：({target[0]:.3f}, {target[1]:.3f})", size=12, color=NAVY, weight="bold", transform=ax_dec.transAxes)
    text(ax_dec, 0.08, 0.45, "置信度：HIGH", size=11, color=GRAY, transform=ax_dec.transAxes)
    reason = "最深可及坑缘，位于主开挖轴线上；\n当前姿态无需大幅转向。"
    text(ax_dec, 0.08, 0.27, reason, size=11, color=DARK, va="center", transform=ax_dec.transAxes, linespacing=1.5)
    save(fig, "04_semantic_to_decision")


def figure_overview_sheet():
    stems = [
        ("01_system_overview.png", "01  系统总览"),
        ("02_dem_growth.png", "02  DEM持续生长"),
        ("03_registration_ablation.png", "03  配准消融"),
        ("04_semantic_to_decision.png", "04  语义到决策"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.035, top=0.92, wspace=0.05, hspace=0.15)
    fig_text(fig, 0.03, 0.965, "老师汇报大图总览", size=25, color=NAVY, weight="bold")
    fig_text(fig, 0.03, 0.93, "建议顺序：系统故事 → 核心结果 → 方法证据 → 决策落地", size=12, color=GRAY)
    for ax, (stem, caption) in zip(axes.flat, stems):
        show_image(ax, f"teacher_figures/{stem}")
        text(
            ax,
            0.02,
            -0.04,
            caption,
            size=13,
            color=NAVY,
            weight="bold",
            va="top",
            transform=ax.transAxes,
            clip_on=False,
        )
    fig.savefig(OUT / "00_teacher_figures_overview.png", dpi=160, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main():
    figure_system_overview()
    figure_dem_growth()
    figure_registration_ablation()
    figure_semantic_decision()
    figure_overview_sheet()
    print(f"Generated figures in: {OUT}")


if __name__ == "__main__":
    main()
