"""Compare offline box measurements with user-supplied physical dimensions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from resource_profiler import ResourceProfiler


WS = Path(__file__).resolve().parent
VIZ = WS / "visualizations"
DIMENSION_KEYS = ("length_m", "width_m", "height_m")
DIMENSION_LABELS = ("length", "width", "height")

# User measurements, interpreted as length × width × height in centimetres.
TRUE_DIMENSIONS_CM = {
    3: {"name": "upright red box", "dimensions_cm": (3.9, 2.7, 8.5)},
    4: {"name": "rounded red flat box", "dimensions_cm": (11.0, 7.5, 4.4)},
    5: {"name": "blue box", "dimensions_cm": (15.0, 9.5, 6.0)},
}


def signed_error(estimated: float, truth: float) -> dict:
    error = float(estimated - truth)
    return {
        "estimated": float(estimated),
        "true": float(truth),
        "signed_error": error,
        "absolute_error": abs(error),
        "relative_error_percent": float(error / truth * 100.0),
        "absolute_relative_error_percent": float(abs(error) / truth * 100.0),
    }


def evaluate_object(semantic_id: int, result: dict, truth: dict) -> dict:
    true_dims = np.asarray(truth["dimensions_cm"], dtype=np.float64)
    estimated_dims = np.asarray([
        result[key]["median"] * 100.0 for key in DIMENSION_KEYS
    ])
    dimensions = {}
    for label, key, estimated, true_value in zip(
            DIMENSION_LABELS, DIMENSION_KEYS, estimated_dims, true_dims):
        metric = signed_error(estimated, true_value)
        q25 = float(result[key]["q25"] * 100.0)
        q75 = float(result[key]["q75"] * 100.0)
        metric.update({
            "frame_iqr": [q25, q75],
            "true_within_frame_iqr": bool(q25 <= true_value <= q75),
        })
        dimensions[label] = metric

    true_envelope_ml = float(np.prod(true_dims))
    estimated_bbox_ml = float(np.prod(estimated_dims))
    estimated_integrated_ml = float(result["volume_ml"]["median"])
    volume_q25 = float(result["volume_ml"]["q25"])
    volume_q75 = float(result["volume_ml"]["q75"])
    return {
        "semantic_id": semantic_id,
        "name": truth["name"],
        "n_valid_frames": int(result["n_valid_frames"]),
        "dimension_order": list(DIMENSION_LABELS),
        "dimensions_cm": dimensions,
        "dimension_mape_percent": float(np.mean([
            dimensions[label]["absolute_relative_error_percent"]
            for label in DIMENSION_LABELS
        ])),
        "volume": {
            "true_envelope_ml": true_envelope_ml,
            "estimated_bbox_product_ml": estimated_bbox_ml,
            "elevation_integrated": signed_error(
                estimated_integrated_ml, true_envelope_ml),
            "bbox_product": signed_error(estimated_bbox_ml, true_envelope_ml),
            "integrated_frame_iqr_ml": [volume_q25, volume_q75],
            "true_envelope_within_integrated_frame_iqr": bool(
                volume_q25 <= true_envelope_ml <= volume_q75),
        },
    }


def evaluate_all(results: dict) -> dict:
    objects = {
        str(semantic_id): evaluate_object(
            semantic_id, results["objects"][str(semantic_id)], truth)
        for semantic_id, truth in TRUE_DIMENSIONS_CM.items()
    }
    integrated_errors = np.asarray([
        obj["volume"]["elevation_integrated"]["absolute_relative_error_percent"]
        for obj in objects.values()
    ])
    all_dimension_errors = np.asarray([
        obj["dimensions_cm"][label]["absolute_relative_error_percent"]
        for obj in objects.values() for label in DIMENSION_LABELS
    ])
    true_total = sum(
        obj["volume"]["true_envelope_ml"] for obj in objects.values())
    estimated_total = sum(
        obj["volume"]["elevation_integrated"]["estimated"]
        for obj in objects.values())
    return {
        "ground_truth_source": "user-supplied dimensions on 2026-08-08",
        "dimension_interpretation": "length × width × height, centimetres",
        "volume_reference": (
            "axis-aligned outer-envelope product L×W×H; rounded corners, cavities, "
            "wall thickness, and usable internal capacity are not modelled"
        ),
        "objects": objects,
        "summary": {
            "dimension_mape_all_9_values_percent": float(
                np.mean(all_dimension_errors)),
            "integrated_volume_mape_percent": float(
                np.mean(integrated_errors)),
            "true_total_envelope_ml": float(true_total),
            "estimated_total_integrated_ml": float(estimated_total),
            "aggregate_integrated_volume_bias_percent": float(
                (estimated_total - true_total) / true_total * 100.0),
            "objects_with_true_volume_inside_frame_iqr": int(sum(
                obj["volume"]["true_envelope_within_integrated_frame_iqr"]
                for obj in objects.values()
            )),
        },
    }


def save_csv(evaluation: dict) -> Path:
    path = WS / "box_ground_truth_evaluation.csv"
    fields = [
        "semantic_id", "name", "n_valid_frames",
        "true_length_cm", "estimated_length_cm", "length_error_cm",
        "length_error_percent", "true_width_cm", "estimated_width_cm",
        "width_error_cm", "width_error_percent", "true_height_cm",
        "estimated_height_cm", "height_error_cm", "height_error_percent",
        "dimension_mape_percent", "true_envelope_ml",
        "estimated_bbox_product_ml", "bbox_volume_error_percent",
        "estimated_integrated_ml", "integrated_volume_error_ml",
        "integrated_volume_error_percent", "integrated_volume_iqr_q25_ml",
        "integrated_volume_iqr_q75_ml", "true_volume_within_integrated_iqr",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for obj in evaluation["objects"].values():
            row = {
                "semantic_id": obj["semantic_id"],
                "name": obj["name"],
                "n_valid_frames": obj["n_valid_frames"],
                "dimension_mape_percent": obj["dimension_mape_percent"],
                "true_envelope_ml": obj["volume"]["true_envelope_ml"],
                "estimated_bbox_product_ml": obj["volume"][
                    "estimated_bbox_product_ml"],
                "bbox_volume_error_percent": obj["volume"]["bbox_product"][
                    "relative_error_percent"],
                "estimated_integrated_ml": obj["volume"][
                    "elevation_integrated"]["estimated"],
                "integrated_volume_error_ml": obj["volume"][
                    "elevation_integrated"]["signed_error"],
                "integrated_volume_error_percent": obj["volume"][
                    "elevation_integrated"]["relative_error_percent"],
                "integrated_volume_iqr_q25_ml": obj["volume"][
                    "integrated_frame_iqr_ml"][0],
                "integrated_volume_iqr_q75_ml": obj["volume"][
                    "integrated_frame_iqr_ml"][1],
                "true_volume_within_integrated_iqr": obj["volume"][
                    "true_envelope_within_integrated_frame_iqr"],
            }
            for label in DIMENSION_LABELS:
                metric = obj["dimensions_cm"][label]
                row.update({
                    f"true_{label}_cm": metric["true"],
                    f"estimated_{label}_cm": metric["estimated"],
                    f"{label}_error_cm": metric["signed_error"],
                    f"{label}_error_percent": metric["relative_error_percent"],
                })
            writer.writerow(row)
    return path


def render_figure(evaluation: dict) -> Path:
    objects = list(evaluation["objects"].values())
    names = [obj["name"] for obj in objects]
    colors = ["#8e44ad", "#d64541", "#18a999"]
    relative_errors = np.asarray([
        [obj["dimensions_cm"][label]["relative_error_percent"]
         for label in DIMENSION_LABELS]
        for obj in objects
    ])
    ratios = 1.0 + relative_errors / 100.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), dpi=180)

    ax = axes[0]
    limit = max(100.0, float(np.ceil(np.max(np.abs(relative_errors)) / 10) * 10))
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(relative_errors, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(3), DIMENSION_LABELS)
    ax.set_yticks(range(3), names)
    ax.set_title("Signed dimension error")
    for row in range(3):
        for col in range(3):
            value = relative_errors[row, col]
            ax.text(col, row, f"{value:+.1f}%", ha="center", va="center",
                    color="white" if abs(value) > .45 * limit else "black",
                    fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=.72, label="(estimated - true) / true (%)")

    ax = axes[1]
    x = np.arange(3)
    for row, (name, color) in enumerate(zip(names, colors)):
        ax.plot(x, ratios[row], "o-", color=color, label=name, linewidth=2)
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.2)
    ax.set_xticks(x, DIMENSION_LABELS)
    ax.set_ylabel("estimated / true dimension")
    ax.set_title("Dimension ratio · 1.0 is exact")
    ax.grid(alpha=.22)
    ax.legend(fontsize=8)

    ax = axes[2]
    x = np.arange(3)
    width = .25
    true_volume = np.asarray([
        obj["volume"]["true_envelope_ml"] for obj in objects])
    bbox_volume = np.asarray([
        obj["volume"]["estimated_bbox_product_ml"] for obj in objects])
    integrated_volume = np.asarray([
        obj["volume"]["elevation_integrated"]["estimated"] for obj in objects])
    ax.bar(x - width, true_volume, width, label="true outer envelope", color="#2f78d0")
    ax.bar(x, bbox_volume, width, label="estimated L×W×H", color="#f39c12")
    bars = ax.bar(x + width, integrated_volume, width,
                  label="elevation-integrated", color="#18a999")
    for bar, obj in zip(bars, objects):
        error = obj["volume"]["elevation_integrated"]["relative_error_percent"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{error:+.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names, rotation=12, ha="right")
    ax.set_ylabel("volume (mL = cm³)")
    ax.set_title("Outer-envelope reference vs reconstruction")
    ax.grid(axis="y", alpha=.22)
    ax.legend(fontsize=8)

    fig.suptitle(
        "scale_test.mp4 · reconstructed boxes versus supplied physical dimensions",
        fontsize=14, y=.995,
    )
    fig.tight_layout()
    path = VIZ / "box_ground_truth_error.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    VIZ.mkdir(exist_ok=True)
    profiler = ResourceProfiler("box_ground_truth_evaluation", WS)
    with profiler.stage("load_offline_measurement_results"):
        with (WS / "experiment_results.json").open(encoding="utf-8") as f:
            results = json.load(f)
    with profiler.stage("compute_dimension_and_volume_errors"):
        evaluation = evaluate_all(results)
    with profiler.stage("write_json_and_csv"):
        json_path = WS / "box_ground_truth_evaluation.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        csv_path = save_csv(evaluation)
    with profiler.stage("render_ground_truth_comparison"):
        figure_path = render_figure(evaluation)
    profile_path = profiler.finish(metadata={"objects": len(evaluation["objects"])})
    manifest = {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "figure": str(figure_path.resolve()),
        "resource_profile": profile_path,
    }
    manifest_path = WS / "box_ground_truth_evaluation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
