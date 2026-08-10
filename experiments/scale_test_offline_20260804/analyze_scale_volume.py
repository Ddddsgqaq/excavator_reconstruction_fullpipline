"""Reproducible metric analysis for scale_test.mp4.

Consumes only artifacts produced by the repository's offline YOLOE + VGGT
pipeline.  It combines the text masks with the instance-level visual prompt
for the upright box, calibrates metric scale from the 15 cm ruler, compares
legacy trajectory gravity with robust semantic-ground gravity, and estimates
box volumes by integrating an object-specific elevation grid.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gravity_alignment as ga
from resource_profiler import ResourceProfiler


WORKSPACE = Path(__file__).resolve().parent
PREDICTIONS = WORKSPACE / "predictions.npz"
TEXT_RESPONSE = WORKSPACE / "text_segmentation_ground1_response.json"
UPRIGHT_RESPONSE = WORKSPACE / "upright_visual_response.json"
KNOWN_RULER_M = 0.15
GRID_RES = 64

CLASS_NAMES = {
    1: "tabletop",
    2: "15 cm ruler",
    3: "upright red box",
    4: "red flat box",
    5: "tissue box",
}
OBJECT_IDS = (3, 4, 5)
COLORS = {
    1: "#a7a7a7",
    2: "#2f78d0",
    3: "#8e44ad",
    4: "#d64541",
    5: "#18a999",
}


def _load_response(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _resize_masks(mask_path: str, shape: tuple[int, int]) -> np.ndarray:
    masks = np.load(mask_path)["semantic_masks"]
    if masks.shape[1:] == shape:
        return masks.astype(np.uint8)
    out = np.zeros((len(masks), *shape), dtype=np.uint8)
    for i, mask in enumerate(masks):
        out[i] = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return out


def build_combined_masks(shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    text = _load_response(TEXT_RESPONSE)
    masks = _resize_masks(text["semantic_masks_path"], shape)

    visual = _load_response(UPRIGHT_RESPONSE)
    upright = _resize_masks(visual["semantic_masks_path"], shape)
    masks[upright > 0] = 3

    out_path = WORKSPACE / "semantic_masks_combined.npz"
    np.savez_compressed(out_path, semantic_masks=masks)
    meta = {
        "semantic_masks_path": str(out_path),
        "semantic_id_map": {
            "white tabletop": 1,
            "transparent ruler": 2,
            "upright red box": 3,
            "red plastic storage box": 4,
            "blue tissue box": 5,
        },
        "text_run": text["run_dir"],
        "upright_visual_run": visual["run_dir"],
        "upright_visual_detections": visual["total_detections"],
    }
    with (WORKSPACE / "semantic_masks_combined_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return masks, meta


def connected_components(mask: np.ndarray) -> list[dict]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components = []
    for cid in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[cid]]
        if area < 20:
            continue
        yy, xx = np.where(labels == cid)
        xy = np.column_stack([xx, yy]).astype(np.float64)
        _, singular, _ = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
        aspect = float(singular[0] / max(singular[1], 1e-9))
        rect_w, rect_h = cv2.minAreaRect(xy.astype(np.float32))[1]
        long_px = float(max(rect_w, rect_h))
        components.append({
            "cid": cid,
            "area": area,
            "bbox": [x, y, w, h],
            "aspect": aspect,
            "long_px": long_px,
            "labels": labels,
        })
    return components


def select_component(mask: np.ndarray, semantic_id: int, strict_ruler: bool = False):
    components = connected_components(mask)
    if not components:
        return None
    height, width = mask.shape
    if semantic_id == 2:
        candidates = []
        for comp in components:
            x, y, w, h = comp["bbox"]
            not_clipped = x >= 3 and y >= 3 and x + w <= width - 3 and y + h <= height - 3
            ok = (
                comp["area"] >= (3000 if strict_ruler else 300)
                and comp["aspect"] >= (3.5 if strict_ruler else 3.0)
                and comp["long_px"] >= (150.0 if strict_ruler else 50.0)
                and (not strict_ruler or not_clipped)
            )
            if ok:
                candidates.append(comp)
        if not candidates:
            return None
        return max(candidates, key=lambda c: c["area"] * c["aspect"])
    return max(components, key=lambda c: c["area"])


def component_mask(comp: dict) -> np.ndarray:
    return comp["labels"] == comp["cid"]


def ruler_endpoint_length(points: np.ndarray, mask: np.ndarray, tail_percent: float = 2.0):
    yy, xx = np.where(mask)
    xy = np.column_stack([xx, yy]).astype(np.float64)
    _, _, axes = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
    along = (xy - xy.mean(axis=0)) @ axes[0]
    lo = along <= np.percentile(along, tail_percent)
    hi = along >= np.percentile(along, 100.0 - tail_percent)
    p0 = points[yy[lo], xx[lo]]
    p1 = points[yy[hi], xx[hi]]
    p0 = p0[np.isfinite(p0).all(axis=1)]
    p1 = p1[np.isfinite(p1).all(axis=1)]
    if len(p0) < 5 or len(p1) < 5:
        return None
    w0 = np.median(p0, axis=0)
    w1 = np.median(p1, axis=0)
    return float(np.linalg.norm(w1 - w0)), w0, w1


def calibrate_scale(points: np.ndarray, masks: np.ndarray) -> dict:
    anchors = []
    for frame in range(len(masks)):
        comp = select_component(masks[frame] == 2, 2, strict_ruler=True)
        if comp is None:
            continue
        measurement = ruler_endpoint_length(points[frame], component_mask(comp))
        if measurement is None:
            continue
        length_u, p0, p1 = measurement
        anchors.append({
            "frame": frame,
            "time_s": (frame + 1) * 0.5,
            "mask_area_px_model": comp["area"],
            "mask_aspect": comp["aspect"],
            "mask_long_px_model": comp["long_px"],
            "length_vggt_units": length_u,
            "scale_m_per_unit": KNOWN_RULER_M / length_u,
            "endpoint_world": [p0.tolist(), p1.tolist()],
        })
    if len(anchors) < 3:
        raise RuntimeError(f"Only {len(anchors)} unclipped ruler anchors survived quality gates.")

    scales = np.array([a["scale_m_per_unit"] for a in anchors], dtype=np.float64)
    scale = float(np.median(scales))
    reconstructed_cm = np.array([a["length_vggt_units"] * scale * 100 for a in anchors])
    for anchor, length_cm in zip(anchors, reconstructed_cm):
        anchor["length_after_global_scale_cm"] = float(length_cm)
        anchor["closure_error_cm"] = float(length_cm - KNOWN_RULER_M * 100)

    rel_spread = float((scales.max() - scales.min()) / scale)
    return {
        "known_length_m": KNOWN_RULER_M,
        "method": "unclipped elongated semantic component; median 3D endpoints from 2% mask tails",
        "n_anchors": len(anchors),
        "scale_m_per_vggt_unit": scale,
        "scale_mean": float(scales.mean()),
        "scale_std": float(scales.std(ddof=1)),
        "scale_cv": float(scales.std(ddof=1) / scales.mean()),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "scale_relative_range": rel_spread,
        "implied_volume_relative_range_first_order": float(3.0 * rel_spread),
        "ruler_closure_mae_cm": float(np.mean(np.abs(reconstructed_cm - 15.0))),
        "ruler_closure_rmse_cm": float(np.sqrt(np.mean((reconstructed_cm - 15.0) ** 2))),
        "anchors": anchors,
    }


def horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    axis_u = np.cross(up, ref)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(up, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    return axis_u, axis_v


def fit_frame_table_plane(points: np.ndarray, confidence: np.ndarray, table_mask: np.ndarray,
                          up_prior: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table_conf = confidence[table_mask & np.isfinite(confidence)]
    if len(table_conf) < 100:
        raise ValueError("too few table confidence samples")
    conf_threshold = float(np.percentile(table_conf, 70))
    keep = table_mask & np.isfinite(points).all(axis=2) & (confidence >= conf_threshold)
    q = points[keep]
    if len(q) < 100:
        raise ValueError("too few table points")
    if len(q) > 20000:
        q = q[np.random.default_rng(2718).choice(len(q), 20000, replace=False)]

    fit = q
    center = np.median(fit, axis=0)
    normal = up_prior.copy()
    for _ in range(4):
        center = fit.mean(axis=0)
        _, _, axes = np.linalg.svd(fit - center, full_matrices=False)
        candidate = axes[-1]
        if float(np.dot(candidate, up_prior)) < 0:
            candidate = -candidate
        # A semantic tabletop fit should refine, not overturn, global gravity.
        if abs(float(np.dot(candidate, up_prior))) < np.cos(np.deg2rad(35.0)):
            candidate = up_prior.copy()
        normal = candidate / np.linalg.norm(candidate)
        residual = (q - center) @ normal
        cutoff = max(float(np.percentile(np.abs(residual), 70)), 1e-5)
        fit = q[np.abs(residual) <= cutoff]
        if len(fit) < 100:
            fit = q
            break

    center = fit.mean(axis=0)
    _, _, axes = np.linalg.svd(fit - center, full_matrices=False)
    normal = axes[-1]
    if float(np.dot(normal, up_prior)) < 0:
        normal = -normal
    if abs(float(np.dot(normal, up_prior))) < np.cos(np.deg2rad(35.0)):
        normal = up_prior.copy()
    normal /= np.linalg.norm(normal)
    residual = (fit - center) @ normal
    return center, normal, residual


def elevation_grid_volume(uv: np.ndarray, height: np.ndarray, grid_res: int = GRID_RES) -> dict:
    good = np.isfinite(uv).all(axis=1) & np.isfinite(height)
    uv = uv[good]
    height = np.clip(height[good], 0.0, None)
    if len(uv) < 100:
        raise ValueError("too few object points")

    center = np.median(uv, axis=0)
    _, _, axes = np.linalg.svd(uv - center, full_matrices=False)
    local = (uv - center) @ axes.T
    lo = np.percentile(local, 2, axis=0)
    hi = np.percentile(local, 98, axis=0)
    size = hi - lo
    if np.any(size <= 1e-8):
        raise ValueError("degenerate footprint")
    inside = np.all((local >= lo) & (local <= hi), axis=1)
    local = local[inside]
    height = height[inside]

    col = np.clip(((local[:, 0] - lo[0]) / size[0] * (grid_res - 1)).round().astype(int),
                  0, grid_res - 1)
    row = np.clip(((local[:, 1] - lo[1]) / size[1] * (grid_res - 1)).round().astype(int),
                  0, grid_res - 1)
    flat = row * grid_res + col
    top = np.full(grid_res * grid_res, np.nan, dtype=np.float64)
    for cell in np.unique(flat):
        top[cell] = np.percentile(height[flat == cell], 90)
    top = top.reshape(grid_res, grid_res)

    pixels = np.column_stack([col, row]).astype(np.int32)
    hull = cv2.convexHull(pixels)
    support = np.zeros((grid_res, grid_res), dtype=np.uint8)
    cv2.fillConvexPoly(support, hull, 1)
    support = support.astype(bool)

    valid = np.isfinite(top)
    if not valid.any():
        raise ValueError("empty elevation grid")
    nearest = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled = top[tuple(nearest)]
    filled = np.clip(filled, 0.0, np.percentile(height, 99))
    cell_area = (size[0] / grid_res) * (size[1] / grid_res)
    volume = float(filled[support].sum() * cell_area)
    return {
        "volume_units3": volume,
        "length_units": float(max(size)),
        "width_units": float(min(size)),
        "height_units": float(np.percentile(height, 95)),
        "footprint_area_units2": float(support.sum() * cell_area),
        "observed_cell_fraction": float((valid & support).sum() / max(support.sum(), 1)),
        "grid_res": grid_res,
    }


def object_measurements(points: np.ndarray, confidence: np.ndarray, masks: np.ndarray,
                        up: np.ndarray, scale: float) -> dict:
    conf_threshold = float(np.percentile(confidence, 50))
    results = {}
    for semantic_id in OBJECT_IDS:
        per_frame = []
        for frame in range(len(masks)):
            comp = select_component(masks[frame] == semantic_id, semantic_id)
            if comp is None or comp["area"] < 500:
                continue
            x, y, w, h = comp["bbox"]
            H, W = masks[frame].shape
            clipped = x < 2 or y < 2 or x + w > W - 2 or y + h > H - 2
            if clipped:
                continue
            obj_mask = component_mask(comp)
            keep = obj_mask & np.isfinite(points[frame]).all(axis=2) & (
                confidence[frame] >= conf_threshold
            )
            q = points[frame][keep]
            if len(q) < 100:
                continue
            try:
                table_center, local_up, table_residual = fit_frame_table_plane(
                    points[frame], confidence[frame], masks[frame] == 1, up
                )
            except ValueError:
                continue

            axis_u, axis_v = horizontal_basis(local_up)
            heights = (q - table_center) @ local_up
            positive_fraction = float(np.mean(heights > 0))
            if positive_fraction < 0.5:
                continue
            lower = -max(float(np.percentile(np.abs(table_residual), 90)), 1e-4)
            upper = float(np.percentile(heights, 99.5))
            geometry_keep = (heights >= lower) & (heights <= upper)
            q = q[geometry_keep]
            heights = heights[geometry_keep]
            uv = np.column_stack([q @ axis_u, q @ axis_v])
            try:
                measure = elevation_grid_volume(uv, heights)
            except ValueError:
                continue
            measure.update({
                "frame": frame,
                "time_s": (frame + 1) * 0.5,
                "semantic_pixels_model": comp["area"],
                "point_count": int(len(q)),
                "positive_height_fraction": positive_fraction,
                "table_residual_rmse_units": float(np.sqrt(np.mean(table_residual ** 2))),
                "length_m": measure["length_units"] * scale,
                "width_m": measure["width_units"] * scale,
                "height_m": measure["height_units"] * scale,
                "volume_m3": measure["volume_units3"] * scale ** 3,
                "volume_ml": measure["volume_units3"] * scale ** 3 * 1e6,
            })
            per_frame.append(measure)

        if not per_frame:
            raise RuntimeError(f"No valid measurements for semantic id {semantic_id}")

        def summary(key):
            values = np.array([row[key] for row in per_frame], dtype=np.float64)
            return {
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "min": float(values.min()),
                "max": float(values.max()),
                "cv": float(values.std(ddof=1) / values.mean()) if len(values) > 1 else 0.0,
            }

        results[str(semantic_id)] = {
            "name": CLASS_NAMES[semantic_id],
            "n_valid_frames": len(per_frame),
            "length_m": summary("length_m"),
            "width_m": summary("width_m"),
            "height_m": summary("height_m"),
            "volume_m3": summary("volume_m3"),
            "volume_ml": summary("volume_ml"),
            "observed_cell_fraction": summary("observed_cell_fraction"),
            "per_frame": per_frame,
        }
    return results


def alignment_diagnostics(points: np.ndarray, confidence: np.ndarray, masks: np.ndarray,
                          extrinsic: np.ndarray, scale: float) -> tuple[dict, np.ndarray]:
    gravity = ga.estimate_gravity(
        extrinsic, points, masks == 1, confidence, conf_thres=0.5
    )
    cloud = points.reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    trajectory = ga.estimate_from_trajectory(extrinsic, cloud)
    if trajectory is None:
        raise RuntimeError("trajectory unexpectedly degenerate")
    trajectory_normal = trajectory[0]
    angle = float(np.degrees(np.arccos(np.clip(
        abs(float(np.dot(trajectory_normal, gravity.n_grav))), -1.0, 1.0
    ))))

    conf_threshold = float(np.percentile(confidence, 50))
    legacy_spans = []
    robust_spans = []
    robust_rmse = []
    local_spans = []
    local_rmse = []
    per_frame = []
    for frame in range(len(masks)):
        keep = (
            (masks[frame] == 1)
            & np.isfinite(points[frame]).all(axis=2)
            & (confidence[frame] >= conf_threshold)
        )
        q = points[frame][keep]
        if len(q) < 100:
            continue
        legacy_y = q @ trajectory_normal
        robust_y = q @ gravity.n_grav
        legacy_y -= np.median(legacy_y)
        robust_y -= np.median(robust_y)
        legacy_span = float(np.percentile(legacy_y, 98) - np.percentile(legacy_y, 2))
        robust_span = float(np.percentile(robust_y, 98) - np.percentile(robust_y, 2))
        legacy_spans.append(legacy_span)
        robust_spans.append(robust_span)
        robust_rmse.append(float(np.sqrt(np.mean(robust_y ** 2))))
        try:
            _center, _normal, local_residual = fit_frame_table_plane(
                points[frame], confidence[frame], masks[frame] == 1, gravity.n_grav
            )
            local_span = float(
                np.percentile(local_residual, 98) - np.percentile(local_residual, 2)
            )
            local_error = float(np.sqrt(np.mean(local_residual ** 2)))
            local_spans.append(local_span)
            local_rmse.append(local_error)
        except ValueError:
            local_span = float("nan")
            local_error = float("nan")
        per_frame.append({
            "frame": frame,
            "legacy_table_p2_p98_span_m": legacy_span * scale,
            "robust_table_p2_p98_span_m": robust_span * scale,
            "robust_table_rmse_m": robust_rmse[-1] * scale,
            "local_plane_p2_p98_span_m": local_span * scale,
            "local_plane_rmse_m": local_error * scale,
        })

    finite = np.isfinite(points).all(axis=3)
    halfmax = (
        (masks == 1) & finite
        & (confidence >= 0.5 * float(np.nanmax(confidence)))
    )
    support = points[halfmax]
    support_legacy = support @ trajectory_normal
    support_robust = support @ gravity.n_grav
    support_legacy -= np.median(support_legacy)
    support_robust -= np.median(support_robust)
    support_legacy_span = float(
        np.percentile(support_legacy, 98) - np.percentile(support_legacy, 2)
    )
    support_robust_span = float(
        np.percentile(support_robust, 98) - np.percentile(support_robust, 2)
    )

    diagnostics = {
        "legacy_source": "trajectory",
        "robust_source": gravity.source,
        "trajectory_vs_ground_deg": angle,
        "gravity_warning": gravity.warnings,
        "ground_debug": gravity.debug.get("ground_mask", {}),
        "selection_reason": gravity.debug.get("selection_reason"),
        "legacy_table_span_median_mm": float(np.median(legacy_spans) * scale * 1000),
        "robust_table_span_median_mm": float(np.median(robust_spans) * scale * 1000),
        "robust_table_rmse_median_mm": float(np.median(robust_rmse) * scale * 1000),
        "span_reduction_percent": float(
            (1.0 - np.median(robust_spans) / np.median(legacy_spans)) * 100.0
        ),
        "high_confidence_ground_support_points": int(len(support)),
        "legacy_high_confidence_ground_span_mm": support_legacy_span * scale * 1000,
        "robust_high_confidence_ground_span_mm": support_robust_span * scale * 1000,
        "robust_high_confidence_ground_rmse_mm": float(
            np.sqrt(np.mean(support_robust ** 2)) * scale * 1000
        ),
        "high_confidence_span_reduction_percent": float(
            (1.0 - support_robust_span / support_legacy_span) * 100.0
        ),
        "local_plane_span_median_mm": float(np.nanmedian(local_spans) * scale * 1000),
        "local_plane_rmse_median_mm": float(np.nanmedian(local_rmse) * scale * 1000),
        "per_frame": per_frame,
    }
    return diagnostics, gravity.n_grav


def registration_diagnostics(points: np.ndarray, masks: np.ndarray, scale: float) -> dict:
    centers = []
    lengths = []
    frames = []
    for frame in range(len(masks)):
        comp = select_component(masks[frame] == 2, 2, strict_ruler=False)
        if comp is None:
            continue
        cmask = component_mask(comp)
        measurement = ruler_endpoint_length(points[frame], cmask)
        if measurement is None:
            continue
        length, _p0, _p1 = measurement
        q = points[frame][cmask]
        q = q[np.isfinite(q).all(axis=1)]
        centers.append(np.median(q, axis=0))
        lengths.append(length)
        frames.append(frame)
    centers = np.asarray(centers)
    center_ref = np.median(centers, axis=0)
    drift = np.linalg.norm(centers - center_ref, axis=1) * scale
    lengths_m = np.asarray(lengths) * scale
    return {
        "n_ruler_frames": len(frames),
        "frames": frames,
        "ruler_center_drift_median_cm": float(np.median(drift) * 100),
        "ruler_center_drift_p95_cm": float(np.percentile(drift, 95) * 100),
        "ruler_length_all_visible_median_cm": float(np.median(lengths_m) * 100),
        "ruler_length_all_visible_iqr_cm": float(
            (np.percentile(lengths_m, 75) - np.percentile(lengths_m, 25)) * 100
        ),
    }


def make_figures(result: dict):
    viz = WORKSPACE / "visualizations"
    viz.mkdir(exist_ok=True)
    scale = result["scale_calibration"]["scale_m_per_vggt_unit"]

    anchors = result["scale_calibration"]["anchors"]
    frames = [a["frame"] for a in anchors]
    lengths = [a["length_after_global_scale_cm"] for a in anchors]
    scales = [a["scale_m_per_unit"] for a in anchors]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].plot(frames, lengths, "o-", color="#2f78d0", lw=2)
    axes[0].axhline(15, color="#d64541", ls="--", label="known 15 cm")
    axes[0].set(xlabel="frame", ylabel="reconstructed ruler length (cm)",
                title="Global-scale closure")
    axes[0].legend()
    axes[0].grid(alpha=.25)
    axes[1].plot(frames, scales, "o-", color="#8e44ad", lw=2)
    axes[1].axhline(scale, color="#222", ls="--", label="median scale")
    axes[1].set(xlabel="frame", ylabel="meters / VGGT unit",
                title="Per-frame scale anchors")
    axes[1].legend()
    axes[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(viz / "scale_calibration_stability.png", bbox_inches="tight")
    plt.close(fig)

    diag = result["alignment"]
    pf = diag["per_frame"]
    f = [x["frame"] for x in pf]
    legacy = [x["legacy_table_p2_p98_span_m"] * 1000 for x in pf]
    robust = [x["robust_table_p2_p98_span_m"] * 1000 for x in pf]
    local = [x["local_plane_p2_p98_span_m"] * 1000 for x in pf]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    axes[0].plot(f, legacy, "o-", label="legacy trajectory", color="#d64541")
    axes[0].plot(f, robust, "o-", label="semantic-ground override", color="#18a999")
    axes[0].plot(f, local, "o-", label="per-frame robust plane", color="#2f78d0")
    axes[0].set(xlabel="frame", ylabel="table elevation p2-p98 span (mm)",
                title=f"Per-frame table spread ({diag['trajectory_vs_ground_deg']:.1f}° conflict)")
    axes[0].grid(alpha=.25)
    axes[0].legend(fontsize=8)
    vals = [
        diag["legacy_high_confidence_ground_span_mm"],
        diag["robust_high_confidence_ground_span_mm"],
    ]
    axes[1].bar(["legacy", "semantic ground"], vals, color=["#d64541", "#18a999"])
    axes[1].set(ylabel="p2-p98 span (mm)",
                title=f"High-confidence RANSAC support\n{diag['high_confidence_ground_support_points']:,} points")
    axes[1].grid(axis="y", alpha=.25)
    for idx, value in enumerate(vals):
        axes[1].text(idx, value, f" {value:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(viz / "ground_alignment_comparison.png", bbox_inches="tight")
    plt.close(fig)

    objects = result["objects"]
    labels = [objects[str(i)]["name"] for i in OBJECT_IDS]
    med = [objects[str(i)]["volume_ml"]["median"] for i in OBJECT_IDS]
    q25 = [objects[str(i)]["volume_ml"]["q25"] for i in OBJECT_IDS]
    q75 = [objects[str(i)]["volume_ml"]["q75"] for i in OBJECT_IDS]
    err = np.vstack([np.asarray(med) - q25, np.asarray(q75) - med])
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    bars = ax.bar(labels, med, yerr=err, capsize=5,
                  color=[COLORS[i] for i in OBJECT_IDS], alpha=.88)
    ax.set(ylabel="elevation-integrated volume (mL)",
           title="Approximate object volumes (median and frame IQR)")
    ax.grid(axis="y", alpha=.25)
    for bar, sid in zip(bars, OBJECT_IDS):
        obj = objects[str(sid)]
        dims = [obj[k]["median"] * 100 for k in ("length_m", "width_m", "height_m")]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f" {dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f} cm\n n={obj['n_valid_frames']}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(viz / "object_volume_summary.png", bbox_inches="tight")
    plt.close(fig)

    # The semantic top-view is deliberately generated in
    # make_process_visualizations.py from the exact RGB reconstruction sample.
    # Keeping it out of this numerical-analysis plotter prevents class-balanced
    # sampling from silently changing the geometry of a visual comparison.


def main():
    profiler = ResourceProfiler(
        "scale_volume_analysis", WORKSPACE,
        metadata={"known_ruler_m": KNOWN_RULER_M, "volume_grid_res": GRID_RES},
    )
    with profiler.stage("load_reconstruction_artifacts"):
        loaded = np.load(PREDICTIONS)
        predictions = {key: np.array(loaded[key]) for key in loaded.files}
        points = predictions["world_points_from_depth"]
        confidence = predictions["depth_conf"]
    with profiler.stage("combine_yoloe_semantic_masks"):
        masks, semantic_meta = build_combined_masks(points.shape[1:3])

    with profiler.stage("calibrate_scale_from_15cm_ruler"):
        calibration = calibrate_scale(points, masks)
        scale = calibration["scale_m_per_vggt_unit"]
    with profiler.stage("gravity_alignment_and_ground_precision"):
        alignment, up = alignment_diagnostics(
            points, confidence, masks, predictions["extrinsic"], scale
        )
    with profiler.stage("object_height_grids_and_volume_integration"):
        objects = object_measurements(points, confidence, masks, up, scale)
    with profiler.stage("multiview_registration_diagnostics"):
        registration = registration_diagnostics(points, masks, scale)

    scale_min = calibration["scale_min"]
    scale_max = calibration["scale_max"]
    for semantic_id in OBJECT_IDS:
        obj = objects[str(semantic_id)]
        median_units3 = obj["volume_m3"]["median"] / scale ** 3
        obj["scale_anchor_only_volume_range_ml"] = [
            float(median_units3 * scale_min ** 3 * 1e6),
            float(median_units3 * scale_max ** 3 * 1e6),
        ]

    result = {
        "experiment": {
            "video": str(ROOT / "scale_test.mp4"),
            "workspace": str(WORKSPACE),
            "frames": int(points.shape[0]),
            "frame_interval_s": 0.5,
            "prediction_mode": "Depthmap and Camera Branch",
            "confidence_percentile": 50.0,
            "volume_method": (
                "semantic instance footprint + per-cell 90th percentile top elevation; "
                "nearest fill inside convex support; integrate height above per-frame tabletop"
            ),
            "resource_profile_path": str(profiler.path),
        },
        "semantics": semantic_meta,
        "scale_calibration": calibration,
        "alignment": alignment,
        "registration_precision": registration,
        "objects": objects,
        "limitations": [
            "The base run originally knew only the ruler size; user-supplied box dimensions are evaluated separately in box_ground_truth_evaluation.json.",
            "Volumes are surface/footprint approximations, not watertight mesh volumes.",
            "Reported frame spread includes VGGT depth/pose inconsistency and segmentation variation.",
        ],
    }
    out = WORKSPACE / "experiment_results.json"
    with profiler.stage("write_numeric_results"):
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    with profiler.stage("render_analysis_figures"):
        make_figures(result)
    profile_path = profiler.finish(metadata={
        "frames": int(points.shape[0]), "objects": len(OBJECT_IDS),
        "scale_m_per_vggt_unit": scale,
    })
    print(json.dumps({
        "results": str(out),
        "scale_m_per_unit": scale,
        "resource_profile_path": profile_path,
        "alignment": alignment,
        "registration_precision": registration,
        "objects": {
            obj["name"]: {
                "n": obj["n_valid_frames"],
                "dimensions_cm": [
                    obj["length_m"]["median"] * 100,
                    obj["width_m"]["median"] * 100,
                    obj["height_m"]["median"] * 100,
                ],
                "volume_ml": obj["volume_ml"]["median"],
                "volume_iqr_ml": [obj["volume_ml"]["q25"], obj["volume_ml"]["q75"]],
                "scale_anchor_only_range_ml": obj["scale_anchor_only_volume_range_ml"],
            }
            for obj in objects.values()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
