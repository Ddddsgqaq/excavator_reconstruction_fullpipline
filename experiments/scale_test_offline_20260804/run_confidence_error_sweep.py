"""Measure offline reconstruction error proxies across VGGT confidence levels.

The experiment deliberately reuses the frozen RGB reconstruction in
``predictions.npz``.  It does not call any online/video path and it never
reconstructs a different point cloud for the semantic measurements.
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
from resource_profiler import ResourceProfiler
from experiments.scale_test_offline_20260804 import analyze_scale_volume as analysis


WS = Path(__file__).resolve().parent
VIZ = WS / "visualizations"
PERCENTILES = (0, 76, 78, 80, 82, 84, 86, 88, 90)
KNOWN_RULER_M = 0.15
MIN_CENTER_POINTS = 30
MIN_AXIS_COVERAGE = 0.25


def ruler_measurement_at_confidence(
        points: np.ndarray, mask: np.ndarray, confidence: np.ndarray,
        threshold: float, tail_percent: float = 2.0):
    """Estimate fixed mask endpoints from confidence-filtered 3-D support.

    The full semantic component defines one fixed 2-D ruler axis and its 2–98%
    endpoints.  A robust affine mapping from axis coordinate to XYZ is fitted
    only on points surviving the threshold, then evaluated at those fixed
    endpoints.  This prevents high confidence from silently shortening the
    ruler while still allowing reliable interior points to support its length.
    """
    yy, xx = np.where(mask)
    if len(yy) < 2:
        return None
    xy = np.column_stack([xx, yy]).astype(np.float64)
    _, _, axes = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
    along = (xy - xy.mean(axis=0)) @ axes[0]
    full_lo, full_hi = np.percentile(
        along, [tail_percent, 100.0 - tail_percent])
    selected = confidence[yy, xx] >= threshold
    selected_points = points[yy[selected], xx[selected]]
    selected_along = along[selected]
    finite = np.isfinite(selected_points).all(axis=1)
    selected_points = selected_points[finite]
    selected_along = selected_along[finite]
    if len(selected_points) < MIN_CENTER_POINTS:
        return None
    support_lo, support_hi = np.percentile(selected_along, [2, 98])
    axis_coverage = float(
        (support_hi - support_lo) / max(full_hi - full_lo, 1e-9))
    if axis_coverage < MIN_AXIS_COVERAGE:
        return None

    fit = np.ones(len(selected_along), dtype=bool)
    coefficients = None
    for _ in range(4):
        design = np.column_stack([selected_along[fit], np.ones(fit.sum())])
        coefficients = np.linalg.lstsq(
            design, selected_points[fit], rcond=None)[0]
        prediction = np.column_stack([
            selected_along, np.ones(len(selected_along))
        ]) @ coefficients
        residual = np.linalg.norm(selected_points - prediction, axis=1)
        cutoff = float(np.percentile(residual, 90))
        new_fit = residual <= cutoff
        if new_fit.sum() < MIN_CENTER_POINTS or np.array_equal(new_fit, fit):
            break
        fit = new_fit
    assert coefficients is not None
    w0 = np.array([full_lo, 1.0]) @ coefficients
    w1 = np.array([full_hi, 1.0]) @ coefficients
    center = np.array([(full_lo + full_hi) * 0.5, 1.0]) @ coefficients
    return {
        "length_units": float(np.linalg.norm(w1 - w0)),
        "center_world": center.tolist(),
        "selected_points": int(len(selected_points)),
        "fit_points": int(fit.sum()),
        "axis_coverage_fraction": axis_coverage,
        "fixed_axis_endpoints": [float(full_lo), float(full_hi)],
    }


def leave_one_out_ruler_errors(lengths_units: np.ndarray) -> dict:
    """Cross-validate metric scale without calibrating and testing one frame."""
    lengths = np.asarray(lengths_units, dtype=np.float64)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    if len(lengths) < 3:
        return {
            "n_anchors": int(len(lengths)),
            "scale_m_per_unit_median": None,
            "scale_cv_percent": None,
            "loo_mae_cm": None,
            "loo_rmse_cm": None,
            "loo_bias_cm": None,
            "loo_predictions_cm": [],
        }
    scales = KNOWN_RULER_M / lengths
    predicted_cm = []
    for idx, length in enumerate(lengths):
        train_scale = float(np.median(np.delete(scales, idx)))
        predicted_cm.append(float(length * train_scale * 100.0))
    predicted_cm = np.asarray(predicted_cm)
    errors = predicted_cm - KNOWN_RULER_M * 100.0
    return {
        "n_anchors": int(len(lengths)),
        "scale_m_per_unit_median": float(np.median(scales)),
        "scale_cv_percent": float(
            np.std(scales, ddof=1) / np.mean(scales) * 100.0),
        "loo_mae_cm": float(np.mean(np.abs(errors))),
        "loo_rmse_cm": float(np.sqrt(np.mean(errors ** 2))),
        "loo_bias_cm": float(np.mean(errors)),
        "loo_predictions_cm": predicted_cm.tolist(),
    }


def table_errors(
        points: np.ndarray, confidence: np.ndarray, masks: np.ndarray,
        threshold: float, up: np.ndarray, scale: float) -> dict:
    per_frame = []
    for frame in range(len(points)):
        keep = (
            (masks[frame] == 1)
            & np.isfinite(points[frame]).all(axis=2)
            & (confidence[frame] >= threshold)
        )
        q = points[frame][keep]
        if len(q) < 100:
            continue
        residual = q @ up
        residual -= np.median(residual)
        per_frame.append({
            "frame": frame,
            "points": int(len(q)),
            "rmse_mm": float(np.sqrt(np.mean(residual ** 2)) * scale * 1000),
            "mae_mm": float(np.mean(np.abs(residual)) * scale * 1000),
            "p2_p98_span_mm": float(
                (np.percentile(residual, 98) - np.percentile(residual, 2))
                * scale * 1000
            ),
        })
    return {
        "n_frames": len(per_frame),
        "rmse_median_mm": float(np.median([x["rmse_mm"] for x in per_frame])),
        "mae_median_mm": float(np.median([x["mae_mm"] for x in per_frame])),
        "p2_p98_span_median_mm": float(
            np.median([x["p2_p98_span_mm"] for x in per_frame])),
        "per_frame": per_frame,
    }


def ruler_errors(
        points: np.ndarray, confidence: np.ndarray, masks: np.ndarray,
        threshold: float, fixed_scale: float) -> dict:
    strict = []
    visible_centers = []
    visible_frames = []
    for frame in range(len(points)):
        visible = analysis.select_component(
            masks[frame] == 2, 2, strict_ruler=False)
        if visible is not None:
            measurement = ruler_measurement_at_confidence(
                points[frame], analysis.component_mask(visible),
                confidence[frame], threshold)
            if measurement is not None:
                visible_centers.append(measurement["center_world"])
                visible_frames.append(frame)

        anchor = analysis.select_component(
            masks[frame] == 2, 2, strict_ruler=True)
        if anchor is None:
            continue
        measurement = ruler_measurement_at_confidence(
            points[frame], analysis.component_mask(anchor),
            confidence[frame], threshold)
        if measurement is not None:
            measurement["frame"] = frame
            strict.append(measurement)

    cross_validation = leave_one_out_ruler_errors(
        np.asarray([x["length_units"] for x in strict]))
    if len(visible_centers) >= 3:
        centers = np.asarray(visible_centers)
        center_ref = np.median(centers, axis=0)
        drift_cm = np.linalg.norm(centers - center_ref, axis=1) * fixed_scale * 100
        registration = {
            "n_frames": len(visible_frames),
            "frames": visible_frames,
            "center_drift_median_cm": float(np.median(drift_cm)),
            "center_drift_p95_cm": float(np.percentile(drift_cm, 95)),
        }
    else:
        registration = {
            "n_frames": len(visible_frames),
            "frames": visible_frames,
            "center_drift_median_cm": None,
            "center_drift_p95_cm": None,
        }
    return {
        "strict_anchor_measurements": strict,
        "cross_validated_scale": cross_validation,
        "registration_repeatability": registration,
    }


def confidence_level(
        percentile: int, points: np.ndarray, confidence: np.ndarray,
        masks: np.ndarray, up: np.ndarray, scale: float) -> dict:
    threshold = float(np.percentile(confidence, percentile))
    finite = np.isfinite(points).all(axis=3)
    selected = finite & (confidence >= threshold)
    table_all = finite & (masks == 1)
    ruler_all = finite & (masks == 2)
    coverage = {
        "scene_points": int(selected.sum()),
        "scene_retained_percent": float(selected.sum() / finite.sum() * 100),
        "table_points": int((selected & table_all).sum()),
        "table_retained_percent": float(
            (selected & table_all).sum() / max(table_all.sum(), 1) * 100),
        "ruler_points": int((selected & ruler_all).sum()),
        "ruler_retained_percent": float(
            (selected & ruler_all).sum() / max(ruler_all.sum(), 1) * 100),
    }
    return {
        "confidence_percentile": percentile,
        "confidence_threshold": threshold,
        "coverage": coverage,
        "table_elevation": table_errors(
            points, confidence, masks, threshold, up, scale),
        "ruler": ruler_errors(
            points, confidence, masks, threshold, scale),
    }


def save_csv(levels: list[dict]) -> Path:
    path = WS / "confidence_error_sweep.csv"
    fields = [
        "confidence_percentile", "confidence_threshold",
        "scene_retained_percent", "scene_points",
        "table_retained_percent", "table_points",
        "ruler_retained_percent", "ruler_points",
        "table_rmse_median_mm", "table_mae_median_mm",
        "table_p2_p98_span_median_mm", "table_frames",
        "ruler_anchor_frames", "ruler_scale_cv_percent",
        "ruler_loo_mae_cm", "ruler_loo_rmse_cm", "ruler_loo_bias_cm",
        "ruler_registration_frames", "ruler_center_drift_median_cm",
        "ruler_center_drift_p95_cm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for level in levels:
            coverage = level["coverage"]
            table = level["table_elevation"]
            ruler = level["ruler"]
            cv = ruler["cross_validated_scale"]
            registration = ruler["registration_repeatability"]
            writer.writerow({
                "confidence_percentile": level["confidence_percentile"],
                "confidence_threshold": level["confidence_threshold"],
                "scene_retained_percent": coverage["scene_retained_percent"],
                "scene_points": coverage["scene_points"],
                "table_retained_percent": coverage["table_retained_percent"],
                "table_points": coverage["table_points"],
                "ruler_retained_percent": coverage["ruler_retained_percent"],
                "ruler_points": coverage["ruler_points"],
                "table_rmse_median_mm": table["rmse_median_mm"],
                "table_mae_median_mm": table["mae_median_mm"],
                "table_p2_p98_span_median_mm": table["p2_p98_span_median_mm"],
                "table_frames": table["n_frames"],
                "ruler_anchor_frames": cv["n_anchors"],
                "ruler_scale_cv_percent": cv["scale_cv_percent"],
                "ruler_loo_mae_cm": cv["loo_mae_cm"],
                "ruler_loo_rmse_cm": cv["loo_rmse_cm"],
                "ruler_loo_bias_cm": cv["loo_bias_cm"],
                "ruler_registration_frames": registration["n_frames"],
                "ruler_center_drift_median_cm": registration[
                    "center_drift_median_cm"],
                "ruler_center_drift_p95_cm": registration[
                    "center_drift_p95_cm"],
            })
    return path


def _values(levels: list[dict], *keys) -> np.ndarray:
    values = []
    for level in levels:
        value = level
        for key in keys:
            value = value[key]
        values.append(np.nan if value is None else value)
    return np.asarray(values, dtype=np.float64)


def render_error_curves(levels: list[dict]) -> Path:
    labels = [
        ("all" if x["confidence_percentile"] == 0 else
         f"P{x['confidence_percentile']}")
        + f"\n{x['coverage']['scene_retained_percent']:.1f}%"
        for x in levels
    ]
    x = np.arange(len(levels))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)

    ax = axes[0, 0]
    ax.plot(x, _values(levels, "table_elevation", "rmse_median_mm"),
            "o-", label="RMSE", color="#d64541")
    ax.plot(x, _values(levels, "table_elevation", "p2_p98_span_median_mm"),
            "o-", label="p2-p98 span", color="#2f78d0")
    ax.set(ylabel="table elevation error proxy (mm)",
           title="Fixed semantic-ground normal · median across frames")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, _values(levels, "ruler", "cross_validated_scale", "loo_mae_cm"),
            "o-", label="leave-one-out MAE", color="#8e44ad")
    ax.plot(x, _values(levels, "ruler", "cross_validated_scale", "loo_rmse_cm"),
            "o-", label="leave-one-out RMSE", color="#e67e22")
    anchors = _values(levels, "ruler", "cross_validated_scale", "n_anchors")
    for xi, count in zip(x, anchors):
        ax.text(xi, .98, f"n={int(count)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8)
    ax.set(ylabel="15 cm ruler error (cm)",
           title="Cross-frame scale validation · no self-calibration")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, _values(levels, "ruler", "registration_repeatability",
                       "center_drift_median_cm"),
            "o-", label="median drift", color="#18a999")
    ax.plot(x, _values(levels, "ruler", "registration_repeatability",
                       "center_drift_p95_cm"),
            "o-", label="P95 drift", color="#34495e")
    frames = _values(levels, "ruler", "registration_repeatability", "n_frames")
    for xi, count in zip(x, frames):
        ax.text(xi, .98, f"n={int(count)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8)
    ax.set(ylabel="ruler center drift (cm)",
           title="Cross-frame registration repeatability")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(x, _values(levels, "coverage", "scene_retained_percent"),
            "o-", label="all geometry", color="#555555")
    ax.plot(x, _values(levels, "coverage", "table_retained_percent"),
            "o-", label="tabletop", color="#7f8c8d")
    ax.plot(x, _values(levels, "coverage", "ruler_retained_percent"),
            "o-", label="ruler", color="#2f78d0")
    ax.set_yscale("log")
    ax.set_ylim(.05, 120)
    ax.set(ylabel="retained target points (%)",
           title="Coverage cost of confidence filtering")
    ax.legend()

    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlabel("confidence percentile / actual scene retention")
        ax.grid(alpha=.22)
    fig.suptitle(
        "scale_test.mp4 · offline VGGT confidence versus error proxies",
        fontsize=15, y=.995,
    )
    fig.tight_layout()
    path = VIZ / "confidence_error_sweep.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_confidence_coverage(
        pred: dict, masks: np.ndarray, results: dict, levels: list[dict]) -> Path:
    points = pred["world_points_from_depth"]
    confidence = pred["depth_conf"]
    gravity = ga.estimate_gravity(
        pred["extrinsic"], points, masks == 1, confidence, conf_thres=0.5)
    aligned = ga.apply_alignment_to_points(points, gravity.R_align)
    scale = results["scale_calibration"]["scale_m_per_vggt_unit"]
    metric = aligned * scale
    finite = np.isfinite(metric).all(axis=3)
    flat_indices = np.flatnonzero(finite)
    if len(flat_indices) > 120000:
        flat_indices = np.random.default_rng(7608).choice(
            flat_indices, 120000, replace=False)
    xyz = metric.reshape(-1, 3)[flat_indices]
    conf = confidence.reshape(-1)[flat_indices]
    rgb_images = np.transpose(pred["images"], (0, 2, 3, 1))
    rgb = np.clip(rgb_images.reshape(-1, 3)[flat_indices], 0, 1)
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)

    selected_percentiles = (0, 78, 84, 88)
    lookup = {x["confidence_percentile"]: x for x in levels}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=180)
    for ax, percentile in zip(axes.flat, selected_percentiles):
        level = lookup[percentile]
        keep = conf >= level["confidence_threshold"]
        ax.scatter(
            xyz[keep, 0], xyz[keep, 2], c=rgb[keep],
            s=.22, alpha=.52, linewidths=0, rasterized=True,
        )
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[2], hi[2])
        ax.set_aspect("equal")
        ax.set_title(
            ("all confidence" if percentile == 0 else f"confidence ≥ P{percentile}")
            + f" · scene kept {level['coverage']['scene_retained_percent']:.1f}%\n"
            + f"same baseline sample: {int(keep.sum()):,}/{len(keep):,} points"
        )
        ax.set_xlabel("aligned X (m)")
        ax.set_ylabel("aligned Z (m)")
        ax.grid(alpha=.15)
    fig.suptitle(
        "Spatial coverage under confidence filtering · identical baseline indices",
        fontsize=14, y=.995,
    )
    fig.tight_layout()
    path = VIZ / "confidence_spatial_coverage.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    VIZ.mkdir(exist_ok=True)
    profiler = ResourceProfiler("confidence_error_sweep", WS)
    with profiler.stage("load_frozen_offline_artifacts"):
        loaded = np.load(WS / "predictions.npz")
        pred = {key: np.asarray(loaded[key]) for key in loaded.files}
        masks = np.load(WS / "semantic_masks_combined.npz")["semantic_masks"]
        with (WS / "experiment_results.json").open(encoding="utf-8") as f:
            results = json.load(f)
        points = pred["world_points_from_depth"]
        confidence = pred["depth_conf"]
        scale = results["scale_calibration"]["scale_m_per_vggt_unit"]

    with profiler.stage("estimate_fixed_semantic_ground_normal"):
        gravity = ga.estimate_gravity(
            pred["extrinsic"], points, masks == 1, confidence, conf_thres=0.5)

    with profiler.stage("sweep_confidence_levels"):
        levels = [
            confidence_level(q, points, confidence, masks, gravity.n_grav, scale)
            for q in PERCENTILES
        ]

    with profiler.stage("write_json_and_csv"):
        csv_path = save_csv(levels)
        payload = {
            "experiment": "VGGT offline confidence versus error proxies",
            "source_predictions": str((WS / "predictions.npz").resolve()),
            "point_branch": "world_points_from_depth (RGB-standard offline branch)",
            "scale_m_per_vggt_unit": scale,
            "gravity_source": gravity.source,
            "confidence_floor": float(np.min(confidence)),
            "confidence_floor_fraction_percent": float(
                np.mean(confidence == np.min(confidence)) * 100),
            "percentiles": list(PERCENTILES),
            "metric_notes": {
                "table": (
                    "Per-frame semantic tabletop residual to one fixed semantic-ground "
                    "normal; median RMSE/span across frames. This is a planarity/elevation "
                    "error proxy, not surveyed absolute height error."
                ),
                "ruler": (
                    "15 cm length tested by leave-one-frame-out scale calibration; "
                    "the tested frame never supplies its own scale. Only unclipped strict "
                    "ruler masks are used. Full-mask 2D endpoints stay fixed while a robust "
                    "axis-to-XYZ model is fitted on confidence-filtered support; support "
                    "covering less than 25% of the ruler axis is rejected."
                ),
                "registration": (
                    "Ruler center distance to the cross-frame median; repeatability only, "
                    "not external absolute position error."
                ),
            },
            "levels": levels,
            "csv_path": str(csv_path.resolve()),
        }
        json_path = WS / "confidence_error_sweep.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    with profiler.stage("render_error_curves"):
        curve_path = render_error_curves(levels)
    with profiler.stage("render_spatial_coverage"):
        coverage_path = render_confidence_coverage(pred, masks, results, levels)

    profile_path = profiler.finish(metadata={
        "confidence_levels": len(levels),
        "source_points": int(np.isfinite(points).all(axis=3).sum()),
    })
    manifest = {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "error_curves": str(curve_path.resolve()),
        "spatial_coverage": str(coverage_path.resolve()),
        "resource_profile": profile_path,
    }
    with (WS / "confidence_error_sweep_manifest.json").open(
            "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
