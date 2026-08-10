"""Controlled ablation of metric size/volume measurement on the frozen scale-test.

The script deliberately does *not* rerun VGGT or YOLOE.  It consumes the same
20-frame reconstruction and the same frozen instance masks for every variant,
so changes in the table are attributable to the measurement stack rather than
to stochastic inference.  The only physical ground truth in this capture is
the 15 cm ruler; boxes therefore use cross-view repeatability, not fabricated
absolute accuracy.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gravity_alignment as ga
from experiments.scale_test_offline_20260804 import analyze_scale_volume as az


WS = Path(__file__).resolve().parent
OUT = WS / "ablation"
PRED = WS / "predictions.npz"
MASKS = WS / "semantic_masks_combined.npz"
KNOWN_RULER_M = 0.15
OBJECT_IDS = (3, 4, 5)
OBJECT_LABELS = {3: "立放红盒", 4: "红色扁盒", 5: "纸巾盒"}
OBJECT_PLOT_LABELS = {3: "Upright red box", 4: "Flat red box", 5: "Tissue box"}


def robust_relative_mad(values: list[float]) -> float:
    """Median absolute deviation normalized by the median, in percent."""
    x = np.asarray(values, dtype=float)
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) / max(abs(med), 1e-12) * 100.0)


def percentile_bbox(points: np.ndarray, up: np.ndarray) -> dict:
    """Initial-VGGT measurement: raw p2--p98 3-D bounding cuboid."""
    h0, h1 = az.horizontal_basis(up)
    coord = np.column_stack([points @ h0, points @ h1, points @ up])
    extent = np.percentile(coord, 98, axis=0) - np.percentile(coord, 2, axis=0)
    length, width = sorted((float(extent[0]), float(extent[1])), reverse=True)
    height = float(extent[2])
    return {
        "length_units": length,
        "width_units": width,
        "height_units": height,
        "volume_units3": float(length * width * height),
        "observed_cell_fraction": 1.0,
        "table_rmse_units": None,
    }


def local_surface(points: np.ndarray, confidence: np.ndarray, table_mask: np.ndarray,
                  object_mask: np.ndarray, up: np.ndarray, conf_threshold: float) -> dict | None:
    """Final estimator: local semantic plane + confidence-filtered top surface."""
    keep = object_mask & np.isfinite(points).all(axis=2) & (confidence >= conf_threshold)
    q = points[keep]
    if len(q) < 100:
        return None
    try:
        center, local_up, residual = az.fit_frame_table_plane(
            points, confidence, table_mask, up)
    except ValueError:
        return None
    heights = (q - center) @ local_up
    if float(np.mean(heights > 0)) < 0.5:
        return None
    lower = -max(float(np.percentile(np.abs(residual), 90)), 1e-4)
    upper = float(np.percentile(heights, 99.5))
    keep_geometry = (heights >= lower) & (heights <= upper)
    q, heights = q[keep_geometry], heights[keep_geometry]
    if len(q) < 100:
        return None
    h0, h1 = az.horizontal_basis(local_up)
    uv = np.column_stack([q @ h0, q @ h1])
    try:
        row = az.elevation_grid_volume(uv, heights)
    except ValueError:
        return None
    row["table_rmse_units"] = float(np.sqrt(np.mean(residual ** 2)))
    return row


def select_object_mask(mask: np.ndarray, semantic_id: int) -> np.ndarray | None:
    comp = az.select_component(mask == semantic_id, semantic_id)
    if comp is None or comp["area"] < 500:
        return None
    x, y, w, h = comp["bbox"]
    H, W = mask.shape
    if x < 2 or y < 2 or x + w > W - 2 or y + h > H - 2:
        return None
    return az.component_mask(comp)


def one_configuration(name: str, points: np.ndarray, confidence: np.ndarray,
                      masks: np.ndarray, up: np.ndarray, scale: float,
                      use_confidence: bool, use_surface: bool) -> dict:
    threshold = float(np.percentile(confidence, 50)) if use_confidence else -np.inf
    objects: dict[str, list[dict]] = {str(i): [] for i in OBJECT_IDS}
    for frame in range(len(masks)):
        for semantic_id in OBJECT_IDS:
            obj_mask = select_object_mask(masks[frame], semantic_id)
            if obj_mask is None:
                continue
            if use_surface:
                row = local_surface(points[frame], confidence[frame], masks[frame] == 1,
                                    obj_mask, up, threshold)
            else:
                keep = obj_mask & np.isfinite(points[frame]).all(axis=2)
                if use_confidence:
                    keep &= confidence[frame] >= threshold
                q = points[frame][keep]
                row = percentile_bbox(q, up) if len(q) >= 100 else None
            if row is None:
                continue
            row = dict(row)
            row.update({
                "frame": int(frame),
                "length_cm": row["length_units"] * scale * 100.0,
                "width_cm": row["width_units"] * scale * 100.0,
                "height_cm": row["height_units"] * scale * 100.0,
                "volume_ml": row["volume_units3"] * scale ** 3 * 1e6,
            })
            objects[str(semantic_id)].append(row)

    summaries = {}
    dim_rmads, vol_rmads = [], []
    for semantic_id, rows in objects.items():
        if not rows:
            continue
        stats = {}
        for key in ("length_cm", "width_cm", "height_cm", "volume_ml"):
            x = [r[key] for r in rows]
            stats[key] = {
                "median": float(np.median(x)),
                "q25": float(np.percentile(x, 25)),
                "q75": float(np.percentile(x, 75)),
                "rmad_percent": robust_relative_mad(x),
            }
        dim_rmad = float(np.mean([stats[k]["rmad_percent"]
                                  for k in ("length_cm", "width_cm", "height_cm")]))
        vol_rmad = stats["volume_ml"]["rmad_percent"]
        dim_rmads.append(dim_rmad)
        vol_rmads.append(vol_rmad)
        summaries[semantic_id] = {
            "label": OBJECT_LABELS[int(semantic_id)], "n_frames": len(rows),
            "dimensions": stats, "dimension_rmad_percent": dim_rmad,
            "volume_rmad_percent": vol_rmad,
        }
    return {
        "name": name, "objects": objects, "summaries": summaries,
        "aggregate": {
            "mean_dimension_rmad_percent": float(np.mean(dim_rmads)),
            "mean_volume_rmad_percent": float(np.mean(vol_rmads)),
            "total_valid_object_frames": int(sum(len(v) for v in objects.values())),
        },
    }


def ruler_leave_one_out(points: np.ndarray, masks: np.ndarray) -> dict:
    """Independent, leakage-free scale check: each ruler anchor is held out."""
    anchors = []
    for frame in range(len(masks)):
        comp = az.select_component(masks[frame] == 2, 2, strict_ruler=True)
        if comp is None:
            continue
        measured = az.ruler_endpoint_length(points[frame], az.component_mask(comp))
        if measured is None:
            continue
        length, _, _ = measured
        anchors.append({"frame": int(frame), "length_units": float(length),
                        "scale": float(KNOWN_RULER_M / length)})
    rows = []
    for i, held in enumerate(anchors):
        train_scale = float(np.median([a["scale"] for j, a in enumerate(anchors) if j != i]))
        estimate_cm = held["length_units"] * train_scale * 100.0
        rows.append({"frame": held["frame"], "train_scale_m_per_unit": train_scale,
                     "heldout_length_cm": estimate_cm,
                     "absolute_error_cm": abs(estimate_cm - 15.0)})
    errors = [r["absolute_error_cm"] for r in rows]
    return {"n_anchors": len(anchors), "rows": rows,
            "mae_cm": float(np.mean(errors)), "rmse_cm": float(np.sqrt(np.mean(np.square(errors))))}


def quality_fusion(final: dict) -> dict:
    """Pre-registered multi-view consensus; gates are fixed before inspecting results."""
    fused = {}
    for obj_id, rows in final["objects"].items():
        eligible = [r for r in rows if r["observed_cell_fraction"] >= .70 and
                    r["table_rmse_units"] is not None and r["table_rmse_units"] <= .022]
        if not eligible:
            eligible = rows
        value = {k: float(np.median([r[k] for r in eligible]))
                 for k in ("length_cm", "width_cm", "height_cm", "volume_ml")}
        fused[obj_id] = {"label": OBJECT_LABELS[int(obj_id)], "n_selected": len(eligible),
                         "n_total": len(rows), **value}
    return fused


def make_figures(configs: list[dict], fusion: dict, loo: dict) -> list[str]:
    OUT.mkdir(exist_ok=True)
    labels = ["M0: raw VGGT", "M1: + confidence", "M2: + semantic ground", "M3: + local surface"]
    dim = [c["aggregate"]["mean_dimension_rmad_percent"] for c in configs]
    vol = [c["aggregate"]["mean_volume_rmad_percent"] for c in configs]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
    width = .36
    ax.bar(x - width / 2, dim, width, label="dimension repeatability rMAD↓", color="#3569b0")
    ax.bar(x + width / 2, vol, width, label="volume repeatability rMAD↓", color="#e07a3f")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("frame-to-frame relative MAD (%)")
    ax.set_title("Controlled ablation: same VGGT points and frozen instance masks")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    for offset, values in ((-width / 2, dim), (width / 2, vol)):
        for i, value in enumerate(values):
            ax.text(i + offset, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p1 = OUT / "ablation_repeatability.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    names = [OBJECT_PLOT_LABELS[int(key)] for key in fusion]
    volumes = [v["volume_ml"] for v in fusion.values()]
    selected = [f"{v['n_selected']}/{v['n_total']} frames" for v in fusion.values()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)
    axes[0].bar(names, volumes, color=["#8e44ad", "#d64541", "#18a999"])
    axes[0].set(ylabel="fused volume (mL)", title="Final local-surface + multi-view consensus")
    axes[0].grid(axis="y", alpha=.25)
    for i, (v, s) in enumerate(zip(volumes, selected)):
        axes[0].text(i, v, f"{v:.1f}\n{s}", ha="center", va="bottom", fontsize=9)
    held = [r["heldout_length_cm"] for r in loo["rows"]]
    frames = [r["frame"] for r in loo["rows"]]
    axes[1].plot(frames, held, "o-", color="#3569b0", label="leave-one-out")
    axes[1].axhline(15, color="#d64541", linestyle="--", label="physical 15 cm")
    axes[1].set(xlabel="held-out ruler frame", ylabel="recovered length (cm)",
                title=f"Metric-scale validation: MAE {loo['mae_cm']:.2f} cm")
    axes[1].grid(alpha=.25); axes[1].legend()
    fig.tight_layout()
    p2 = OUT / "ablation_metric_and_fusion.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    return [str(p1), str(p2)]


def write_report(result: dict) -> Path:
    c = result["configs"]
    base, final = c[0], c[-1]
    d0, d1 = base["aggregate"]["mean_dimension_rmad_percent"], final["aggregate"]["mean_dimension_rmad_percent"]
    v0, v1 = base["aggregate"]["mean_volume_rmad_percent"], final["aggregate"]["mean_volume_rmad_percent"]
    dim_gain = (1 - d1 / d0) * 100
    vol_gain = (1 - v1 / v0) * 100
    lines = [
        "# 尺寸与体积测量模块消融实验报告", "",
        "## 结论", "",
        f"在固定 20 帧 VGGT 重建和固定实例掩码的控制实验中，最终测量链相较于初始 VGGT 几何基线，将三类静态物体的平均尺寸重复性 rMAD 从 **{d0:.1f}%** 降至 **{d1:.1f}%**（降低 **{dim_gain:.1f}%**），体积重复性 rMAD 从 **{v0:.1f}%** 降至 **{v1:.1f}%**（降低 **{vol_gain:.1f}%**）。", "",
        f"尺度模块在三帧严格尺子锚点的留一验证中达到 **{result['ruler_leave_one_out']['mae_cm']:.2f} cm MAE / {result['ruler_leave_one_out']['rmse_cm']:.2f} cm RMSE**。", "",
        "重要边界：该采集没有盒子的实测长宽高或排水法/规格书体积真值。因此盒子的数值只能论证**跨视角重复性**，不能声称绝对尺寸/体积准确度。初始 VGGT 的全局尺度本身不可辨识，故绝对公制误差须先通过独立尺子标定后才有定义。", "",
        "## 实验设计", "",
        "- 数据：冻结的 `scale_test_offline_20260804`，20 帧、同一 `predictions.npz`、同一语义实例掩码；不重跑模型。", "- 控制变量：所有变体均用同一 15 cm 标尺获得的全局公制换算；这只是共同的评测量纲，不计入几何消融。", "- 尺度独立验证：对每个严格尺子锚点，用其余锚点的中位尺度复原该帧尺长（leave-one-out）。", "- 盒子指标：每个静态盒子跨有效帧的相对中位绝对偏差 rMAD，分别对 L/W/H 取均值（尺寸）及对体积计算；越小越好。", "- 公平性：固定目标 ROI/实例掩码，避免将分割波动误归因于几何模块；因此这是测量栈消融，不是端到端检测精度消融。", "",
        "## 消融配置与结果", "",
        "| 配置 | 新增模块 | 有效物体-帧 | 尺寸 rMAD↓ | 体积 rMAD↓ | 相对初始体积重复性提升 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cfg in c:
        agg = cfg["aggregate"]
        gain = (1 - agg["mean_volume_rmad_percent"] / v0) * 100
        detail = {"M0 初始 VGGT": "轨迹重力 + 原始 3D p2–p98 包围盒",
                  "M1 + 点置信度": "M0 + 50% 点置信度门控",
                  "M2 + 语义地面": "M1 + 语义桌面 RANSAC 重力",
                  "M3 + 局部表面": "M2 + 局部桌面平面 + 顶面高程栅格"}[cfg["name"]]
        lines.append(f"| {cfg['name']} | {detail} | {agg['total_valid_object_frames']} | {agg['mean_dimension_rmad_percent']:.1f}% | {agg['mean_volume_rmad_percent']:.1f}% | {gain:.1f}% |")
    lines += ["", "![消融重复性对比](ablation_repeatability.png)", "",
              "## 尺度模块：留一尺子验证", "",
              "| 留出帧 | 其余锚点尺度 (m/unit) | 复原尺长 | 绝对误差 |",
              "|---:|---:|---:|---:|"]
    for row in result["ruler_leave_one_out"]["rows"]:
        lines.append(f"| {row['frame']} | {row['train_scale_m_per_unit']:.4f} | {row['heldout_length_cm']:.2f} cm | {row['absolute_error_cm']:.2f} cm |")
    lines += ["", "![尺度留一验证与多视角融合](ablation_metric_and_fusion.png)", "",
              "## 最终多视角融合输出", "",
              "在 M3 的逐帧局部表面估计上，预先固定质量门：可观测栅格比例 ≥70%、局部桌面 RMSE ≤0.022 VGGT unit；对入选帧取中位数共识。", "",
              "| 物体 | 入选/总有效帧 | L × W × H (cm) | 体积 (mL) |",
              "|---|---:|---:|---:|"]
    for row in result["fusion"].values():
        lines.append(f"| {row['label']} | {row['n_selected']}/{row['n_total']} | {row['length_cm']:.1f} × {row['width_cm']:.1f} × {row['height_cm']:.1f} | {row['volume_ml']:.1f} |")
    lines += ["", "## 论文可用表述与限制", "",
              "可写为：语义地面、局部平面和高程积分显著改善了静态目标的跨视角测量重复性；尺度模块在独立留一锚点下报告了长度闭合误差。不可写为盒子绝对体积误差改善，除非补录每个盒子的卡尺 L/W/H 与实物容积真值。", "",
              "建议补采：每个对象至少 3 次卡尺测量（L/W/H），规则长方体以 L×W×H 生成体积真值，不规则盒以排水法或制造商规格交叉验证；以对象为独立样本、视频片段为重复采集，报告 MAE、MAPE、RMSE 和 95% bootstrap CI。", "",
              "## 可复现性", "",
              "运行：`/home/maomaoyu/miniconda3/envs/vggt50/bin/python experiments/scale_test_offline_20260804/run_measurement_ablation.py`。数值全量见 `ablation_results.json`，逐帧数据见 `ablation_per_frame.csv`。",
    ]
    path = OUT / "ABLATION_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pred_file = np.load(PRED)
    points = pred_file["world_points"]
    confidence = pred_file["world_points_conf"]
    extrinsic = pred_file["extrinsic"]
    masks = np.load(MASKS)["semantic_masks"]
    scale = az.calibrate_scale(points, masks)["scale_m_per_vggt_unit"]
    cloud = points.reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    trajectory = ga.estimate_from_trajectory(extrinsic, cloud)
    if trajectory is None:
        raise RuntimeError("trajectory gravity unexpectedly degenerate")
    gravity = ga.estimate_gravity(extrinsic, points, masks == 1, confidence, conf_thres=.5)
    configs = [
        one_configuration("M0 初始 VGGT", points, confidence, masks, trajectory[0], scale, False, False),
        one_configuration("M1 + 点置信度", points, confidence, masks, trajectory[0], scale, True, False),
        one_configuration("M2 + 语义地面", points, confidence, masks, gravity.n_grav, scale, True, False),
        one_configuration("M3 + 局部表面", points, confidence, masks, gravity.n_grav, scale, True, True),
    ]
    result = {
        "protocol": {"dataset": str(WS), "n_frames": int(len(masks)),
                     "fixed_scale_m_per_vggt_unit": scale,
                     "fixed_instance_masks": str(MASKS),
                     "trajectory_gravity": trajectory[0].tolist(),
                     "semantic_gravity": gravity.n_grav.tolist(),
                     "semantic_gravity_source": gravity.source,
                     "ground_vs_trajectory_deg": gravity.debug.get("traj_vs_ground_deg")},
        "ruler_leave_one_out": ruler_leave_one_out(points, masks),
        "configs": configs,
        "fusion": quality_fusion(configs[-1]),
    }
    result["figures"] = make_figures(configs, result["fusion"], result["ruler_leave_one_out"])
    with (OUT / "ablation_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    fields = ["config", "object", "frame", "length_cm", "width_cm", "height_cm", "volume_ml",
              "observed_cell_fraction", "table_rmse_units"]
    with (OUT / "ablation_per_frame.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cfg in configs:
            for obj_id, rows in cfg["objects"].items():
                for row in rows:
                    writer.writerow({"config": cfg["name"], "object": OBJECT_LABELS[int(obj_id)],
                                     **{key: row.get(key) for key in fields if key not in {"config", "object"}}})
    report = write_report(result)
    print(json.dumps({"report": str(report), "results": str(OUT / "ablation_results.json"),
                      "figures": result["figures"], "aggregate": [c["aggregate"] for c in configs]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
