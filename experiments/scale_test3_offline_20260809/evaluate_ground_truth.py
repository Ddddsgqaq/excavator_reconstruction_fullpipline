"""Evaluate scale_test3 dimension estimates against user-provided truth."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WS = Path(__file__).resolve().parent
TRUTH_CM = {
    "3": {"name": "red rectangular box", "dimensions_cm": [11.0, 7.9, 4.5]},
    "4": {"name": "red bottle", "dimensions_cm": [8.5, 7.0, 21.0]},
}
AXES = ("length", "width", "height")


def main() -> None:
    result = json.loads((WS / "dimension_results.json").read_text(encoding="utf-8"))
    evaluation = {
        "truth_source": "user-provided dimensions on 2026-08-09",
        "error_definitions": {
            "signed_error_cm": "estimate - truth",
            "relative_error_percent": "(estimate - truth) / truth * 100",
            "axis_accuracy_percent": "100 - absolute(relative_error_percent)",
            "mape_percent": "mean absolute relative error across three axes",
        },
        "objects": {},
    }
    all_signed, all_abs_rel = [], []
    csv_rows = []
    for sid, truth_info in TRUTH_CM.items():
        measured = result["objects"][sid]
        estimate = np.asarray([
            measured["dimensions_cm"][axis]["median"] for axis in AXES
        ], dtype=float)
        truth = np.asarray(truth_info["dimensions_cm"], dtype=float)
        signed = estimate - truth
        absolute = np.abs(signed)
        relative = signed / truth * 100.0
        accuracy = 100.0 - np.abs(relative)
        axes = {}
        for idx, axis in enumerate(AXES):
            axes[axis] = {
                "truth_cm": float(truth[idx]),
                "estimate_cm": float(estimate[idx]),
                "signed_error_cm": float(signed[idx]),
                "absolute_error_cm": float(absolute[idx]),
                "relative_error_percent": float(relative[idx]),
                "absolute_relative_error_percent": float(abs(relative[idx])),
                "axis_accuracy_percent": float(accuracy[idx]),
                "estimate_iqr_cm": [
                    measured["dimensions_cm"][axis]["q25"],
                    measured["dimensions_cm"][axis]["q75"],
                ],
                "truth_inside_estimate_iqr": bool(
                    measured["dimensions_cm"][axis]["q25"] <= truth[idx]
                    <= measured["dimensions_cm"][axis]["q75"]
                ),
            }
            csv_rows.append({
                "object_id": sid, "object": truth_info["name"], "axis": axis,
                **{k: axes[axis][k] for k in (
                    "truth_cm", "estimate_cm", "signed_error_cm",
                    "absolute_error_cm", "relative_error_percent",
                    "absolute_relative_error_percent", "axis_accuracy_percent",
                )},
            })
        true_bbox = float(np.prod(truth))
        estimate_bbox = float(np.prod(estimate))
        evaluation["objects"][sid] = {
            "name": truth_info["name"],
            "truth_dimensions_cm": truth.tolist(),
            "estimate_dimensions_cm": estimate.tolist(),
            "axes": axes,
            "summary": {
                "mae_cm": float(np.mean(absolute)),
                "rmse_cm": float(np.sqrt(np.mean(signed ** 2))),
                "mean_signed_error_cm": float(np.mean(signed)),
                "mape_percent": float(np.mean(np.abs(relative))),
                "mean_axis_accuracy_percent": float(np.mean(accuracy)),
                "max_absolute_error_cm": float(np.max(absolute)),
                "max_absolute_relative_error_percent": float(np.max(np.abs(relative))),
            },
            "bounding_box_product": {
                "truth_cm3": true_bbox,
                "estimate_cm3": estimate_bbox,
                "signed_error_cm3": estimate_bbox - true_bbox,
                "relative_error_percent": (estimate_bbox - true_bbox) / true_bbox * 100.0,
                "note": "outer bounding-box product only; not physical bottle volume",
            },
        }
        all_signed.extend(signed.tolist())
        all_abs_rel.extend(np.abs(relative).tolist())

    all_signed = np.asarray(all_signed)
    evaluation["overall_six_axes"] = {
        "mae_cm": float(np.mean(np.abs(all_signed))),
        "rmse_cm": float(np.sqrt(np.mean(all_signed ** 2))),
        "mape_percent": float(np.mean(all_abs_rel)),
        "mean_axis_accuracy_percent": float(100.0 - np.mean(all_abs_rel)),
        "max_absolute_error_cm": float(np.max(np.abs(all_signed))),
        "axis_count": 6,
    }
    out = WS / "ground_truth_accuracy.json"
    out.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    with (WS / "ground_truth_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    x = np.arange(3)
    width = .34
    for ax, (sid, obj) in zip(axes, evaluation["objects"].items()):
        truth = np.asarray(obj["truth_dimensions_cm"])
        estimate = np.asarray(obj["estimate_dimensions_cm"])
        ax.bar(x - width/2, truth, width, label="ground truth", color="#666666")
        ax.bar(x + width/2, estimate, width, label="estimate", color="#2f78d0")
        ax.set_xticks(x, AXES)
        ax.set_ylabel("outer dimension (cm)")
        ax.set_title(
            f"{obj['name']} · MAPE {obj['summary']['mape_percent']:.2f}%"
        )
        ax.grid(axis="y", alpha=.25)
        ax.legend()
        for idx, err in enumerate([
            obj["axes"][axis]["relative_error_percent"] for axis in AXES
        ]):
            ax.text(idx + width/2, estimate[idx], f"{err:+.1f}%",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(WS / "visualizations" / "ground_truth_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
