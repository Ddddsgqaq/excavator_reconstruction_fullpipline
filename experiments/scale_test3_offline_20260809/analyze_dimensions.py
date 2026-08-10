"""Metric outer-dimension analysis for scale_test3.mp4.

The script consumes only the artifacts from this repository's offline YOLOE
and VGGT services.  Metric scale comes from the 15 cm ruler.  Object sizes are
measured in a per-frame robust tabletop coordinate system and fused across
quality-controlled frames.
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
from experiments.scale_test_offline_20260804.analyze_scale_volume import (
    calibrate_scale,
    component_mask,
    fit_frame_table_plane,
    horizontal_basis,
    select_component,
)


WS = Path(__file__).resolve().parent
PREDICTIONS = WS / "predictions.npz"
KNOWN_RULER_M = 0.15
OBJECTS = {
    3: {"name": "red rectangular box", "height_percentile": 95.0, "color": "#d64541"},
    4: {"name": "red bottle", "height_percentile": 98.5, "color": "#7b3f98"},
}
CONFIDENCE_PERCENTILES = (50, 70, 78, 85)
PRIMARY_CONFIDENCE_PERCENTILE = 50


def _summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
        "cv": float(a.std(ddof=1) / a.mean()) if len(a) > 1 else 0.0,
    }


def robust_scale(points: np.ndarray, masks: np.ndarray) -> dict:
    """Reuse the original ruler endpoint method, then reject temporal outliers."""
    raw = calibrate_scale(points, masks)
    values = np.asarray([a["scale_m_per_unit"] for a in raw["anchors"]])
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = max(1.4826 * mad, med * 0.002)
    keep = np.abs(values - med) <= 3.5 * sigma
    if int(keep.sum()) < 5:
        keep[np.argsort(np.abs(values - med))[: min(5, len(values))]] = True
    accepted = []
    rejected = []
    for ok, anchor in zip(keep, raw["anchors"]):
        row = dict(anchor)
        row["accepted_by_temporal_mad"] = bool(ok)
        (accepted if ok else rejected).append(row)
    accepted_scales = values[keep]
    scale = float(np.median(accepted_scales))
    lengths_cm = np.asarray([a["length_vggt_units"] for a in accepted]) * scale * 100.0
    for anchor, length_cm in zip(accepted, lengths_cm):
        anchor["length_after_robust_scale_cm"] = float(length_cm)
        anchor["closure_error_cm"] = float(length_cm - 15.0)
    return {
        "known_length_m": KNOWN_RULER_M,
        "method": raw["method"] + "; temporal 3.5-MAD rejection",
        "n_candidates": len(values),
        "n_accepted": int(keep.sum()),
        "n_rejected": int((~keep).sum()),
        "scale_m_per_vggt_unit": scale,
        "accepted_scale_mean": float(accepted_scales.mean()),
        "accepted_scale_std": float(accepted_scales.std(ddof=1)),
        "accepted_scale_cv": float(accepted_scales.std(ddof=1) / accepted_scales.mean()),
        "accepted_scale_min": float(accepted_scales.min()),
        "accepted_scale_max": float(accepted_scales.max()),
        "ruler_closure_mae_cm": float(np.mean(np.abs(lengths_cm - 15.0))),
        "ruler_closure_rmse_cm": float(np.sqrt(np.mean((lengths_cm - 15.0) ** 2))),
        "anchors": accepted,
        "rejected_anchors": rejected,
        "raw_calibration": {
            "n_anchors": raw["n_anchors"],
            "scale_cv": raw["scale_cv"],
            "scale_min": raw["scale_min"],
            "scale_max": raw["scale_max"],
        },
    }


def robust_gravity(points: np.ndarray, confidence: np.ndarray, masks: np.ndarray,
                   extrinsic: np.ndarray, scale: float) -> tuple[np.ndarray, dict]:
    gravity = ga.estimate_gravity(
        extrinsic, points, masks == 1, confidence, conf_thres=0.5
    )
    local = []
    for frame in range(len(points)):
        try:
            _center, normal, residual = fit_frame_table_plane(
                points[frame], confidence[frame], masks[frame] == 1, gravity.n_grav
            )
        except ValueError:
            continue
        angle = np.degrees(np.arccos(np.clip(abs(float(normal @ gravity.n_grav)), 0, 1)))
        local.append({
            "frame": frame,
            "normal_vs_global_deg": float(angle),
            "table_p2_p98_span_mm": float(
                (np.percentile(residual, 98) - np.percentile(residual, 2)) * scale * 1000
            ),
            "table_rmse_mm": float(np.sqrt(np.mean(residual ** 2)) * scale * 1000),
        })
    return gravity.n_grav, {
        "source": gravity.source,
        "warnings": gravity.warnings,
        "selection_reason": gravity.debug.get("selection_reason"),
        "ground_debug": gravity.debug.get("ground_mask", {}),
        "local_plane_valid_frames": len(local),
        "local_plane_span_median_mm": float(np.median([x["table_p2_p98_span_mm"] for x in local])),
        "local_plane_rmse_median_mm": float(np.median([x["table_rmse_mm"] for x in local])),
        "local_normal_deviation_median_deg": float(np.median([x["normal_vs_global_deg"] for x in local])),
        "per_frame": local,
    }


def measure_one_frame(points: np.ndarray, confidence: np.ndarray, mask: np.ndarray,
                      semantic_id: int, up: np.ndarray, scale: float,
                      confidence_threshold: float, frame: int) -> dict | None:
    comp = select_component(mask == semantic_id, semantic_id)
    if comp is None or comp["area"] < 300:
        return None
    x, y, w, h = comp["bbox"]
    H, W = mask.shape
    if x < 2 or y < 2 or x + w > W - 2 or y + h > H - 2:
        return None
    cmask = component_mask(comp)
    keep = (
        cmask & np.isfinite(points).all(axis=2)
        & np.isfinite(confidence) & (confidence >= confidence_threshold)
    )
    q = points[keep]
    if len(q) < 150:
        return None
    try:
        table_center, local_up, table_residual = fit_frame_table_plane(
            points, confidence, mask == 1, up
        )
    except ValueError:
        return None
    heights = (q - table_center) @ local_up
    positive_fraction = float(np.mean(heights > 0))
    if positive_fraction < 0.60:
        return None
    floor_tol = max(float(np.percentile(np.abs(table_residual), 95)) * 1.5, 1e-4)
    ceiling = float(np.percentile(heights, 99.7))
    geom = (heights >= -floor_tol) & (heights <= ceiling)
    q = q[geom]
    heights = heights[geom]
    if len(q) < 120:
        return None
    axis_u, axis_v = horizontal_basis(local_up)
    uv = np.column_stack([q @ axis_u, q @ axis_v])
    center = np.median(uv, axis=0)
    _u, _s, axes = np.linalg.svd(uv - center, full_matrices=False)
    local = (uv - center) @ axes.T
    lo = np.percentile(local, 2, axis=0)
    hi = np.percentile(local, 98, axis=0)
    footprint = np.sort(hi - lo)[::-1]
    height_pct = OBJECTS[semantic_id]["height_percentile"]
    height_units = float(np.percentile(np.clip(heights, 0, None), height_pct))
    dims_cm = np.asarray([footprint[0], footprint[1], height_units]) * scale * 100.0
    return {
        "frame": frame,
        "time_s": (frame + 1) * 0.5,
        "semantic_pixels_model": comp["area"],
        "bbox_model": comp["bbox"],
        "point_count": int(len(q)),
        "positive_height_fraction": positive_fraction,
        "table_rmse_mm": float(np.sqrt(np.mean(table_residual ** 2)) * scale * 1000),
        "length_cm": float(dims_cm[0]),
        "width_cm": float(dims_cm[1]),
        "height_cm": float(dims_cm[2]),
    }


def robust_frame_selection(rows: list[dict], semantic_id: int) -> tuple[list[dict], list[dict]]:
    pre_rejected = []
    if semantic_id == 4:
        # A bottle should expose a compact footprint. Very thin footprints in
        # this sequence are frames where the transparent body has no usable
        # depth on one side, rather than plausible alternate bottle geometry.
        complete = []
        for row in rows:
            compactness = row["width_cm"] / max(row["length_cm"], 1e-6)
            row = dict(row)
            row["footprint_compactness"] = float(compactness)
            if compactness >= 0.72:
                complete.append(row)
            else:
                row["selected"] = False
                row["rejection_reason"] = "incomplete bottle footprint (minor/major < 0.72)"
                pre_rejected.append(row)
        rows = complete
    if len(rows) < 3:
        return rows, pre_rejected
    dims = np.asarray([[r["length_cm"], r["width_cm"], r["height_cm"]] for r in rows])
    logdims = np.log(np.maximum(dims, 1e-6))
    med = np.median(logdims, axis=0)
    mad = np.median(np.abs(logdims - med), axis=0)
    sigma = np.maximum(1.4826 * mad, 0.025)
    z = np.max(np.abs(logdims - med) / sigma, axis=1)
    keep = z <= 3.5
    target = min(5, len(rows))
    if int(keep.sum()) < target:
        keep[np.argsort(z)[:target]] = True
    selected, rejected = [], []
    for score, ok, row in zip(z, keep, rows):
        out = dict(row)
        out["robust_dimension_score"] = float(score)
        out["selected"] = bool(ok)
        (selected if ok else rejected).append(out)
    return selected, pre_rejected + rejected


def measure_objects(points: np.ndarray, confidence: np.ndarray, masks: np.ndarray,
                    up: np.ndarray, scale: float) -> tuple[dict, dict]:
    confidence_thresholds = {
        str(p): float(np.percentile(confidence[np.isfinite(confidence)], p))
        for p in CONFIDENCE_PERCENTILES
    }
    variants = {}
    for percentile in CONFIDENCE_PERCENTILES:
        threshold = confidence_thresholds[str(percentile)]
        objects = {}
        for semantic_id, cfg in OBJECTS.items():
            raw = []
            for frame in range(len(points)):
                row = measure_one_frame(
                    points[frame], confidence[frame], masks[frame], semantic_id,
                    up, scale, threshold, frame,
                )
                if row is not None:
                    raw.append(row)
            gated_id = semantic_id if percentile == PRIMARY_CONFIDENCE_PERCENTILE else -1
            selected, rejected = robust_frame_selection(raw, gated_id)
            if not selected:
                raise RuntimeError(f"No valid measurements for {cfg['name']} at P{percentile}")
            objects[str(semantic_id)] = {
                "name": cfg["name"],
                "height_percentile": cfg["height_percentile"],
                "n_raw_frames": len(raw),
                "n_selected_frames": len(selected),
                "dimensions_cm": {
                    key.replace("_cm", ""): _summary([r[key] for r in selected])
                    for key in ("length_cm", "width_cm", "height_cm")
                },
                "selected_frames": selected,
                "rejected_frames": rejected,
            }
        variants[str(percentile)] = objects
    return variants[str(PRIMARY_CONFIDENCE_PERCENTILE)], {
        "global_thresholds": confidence_thresholds,
        "variants": variants,
    }


def make_figures(result: dict) -> None:
    viz = WS / "visualizations"
    viz.mkdir(exist_ok=True)
    cal = result["scale_calibration"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=170)
    accepted = cal["anchors"]
    rejected = cal["rejected_anchors"]
    axes[0].plot([a["frame"] for a in accepted],
                 [a["length_after_robust_scale_cm"] for a in accepted],
                 "o-", label="accepted", color="#2f78d0")
    if rejected:
        axes[0].scatter([a["frame"] for a in rejected],
                        [a["length_vggt_units"] * cal["scale_m_per_vggt_unit"] * 100 for a in rejected],
                        marker="x", s=55, label="rejected", color="#d64541")
    axes[0].axhline(15, ls="--", color="black", label="known 15 cm")
    axes[0].set(xlabel="frame", ylabel="scaled ruler length (cm)", title="Ruler closure")
    axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot([a["frame"] for a in accepted],
                 [a["scale_m_per_unit"] for a in accepted], "o-", color="#7b3f98")
    axes[1].axhline(cal["scale_m_per_vggt_unit"], ls="--", color="black")
    axes[1].set(xlabel="frame", ylabel="metres / VGGT unit", title="Accepted scale anchors")
    axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(viz / "scale_calibration_stability.png", bbox_inches="tight"); plt.close(fig)

    primary = result["objects"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=170)
    for ax, (sid, cfg) in zip(axes, OBJECTS.items()):
        obj = primary[str(sid)]
        rows = obj["selected_frames"]
        for key, marker in (("length_cm", "o"), ("width_cm", "s"), ("height_cm", "^")):
            ax.plot([r["frame"] for r in rows], [r[key] for r in rows], marker + "-", label=key[:-3])
        d = obj["dimensions_cm"]
        ax.set(title=f"{cfg['name']}\n{d['length']['median']:.1f} × {d['width']['median']:.1f} × {d['height']['median']:.1f} cm",
               xlabel="frame", ylabel="outer dimension (cm)")
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(viz / "dimension_estimates_by_frame.png", bbox_inches="tight"); plt.close(fig)

    sensitivity = result["confidence_sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=170)
    for ax, (sid, cfg) in zip(axes, OBJECTS.items()):
        for dim, marker in (("length", "o"), ("width", "s"), ("height", "^")):
            vals = [sensitivity["variants"][str(p)][str(sid)]["dimensions_cm"][dim]["median"] for p in CONFIDENCE_PERCENTILES]
            ax.plot(CONFIDENCE_PERCENTILES, vals, marker + "-", label=dim)
        ax.set(title=cfg["name"], xlabel="global confidence percentile", ylabel="median dimension (cm)")
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle("Confidence sensitivity (P50 is the same primary gate as the prior experiment)")
    fig.tight_layout(); fig.savefig(viz / "confidence_sensitivity.png", bbox_inches="tight"); plt.close(fig)

    ground = result["ground_alignment"]
    rows = ground["per_frame"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=170)
    axes[0].plot([r["frame"] for r in rows], [r["table_p2_p98_span_mm"] for r in rows], "o-", color="#18a999")
    axes[0].set(xlabel="frame", ylabel="P2-P98 residual span (mm)", title="Robust local tabletop thickness")
    axes[0].grid(alpha=.25)
    axes[1].plot([r["frame"] for r in rows], [r["table_rmse_mm"] for r in rows], "o-", color="#2f78d0")
    axes[1].set(xlabel="frame", ylabel="plane residual RMSE (mm)", title="Local tabletop fit residual")
    axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(viz / "ground_plane_diagnostics.png", bbox_inches="tight"); plt.close(fig)


def write_csv(primary: dict) -> Path:
    path = WS / "dimension_measurements_per_frame.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "object_id", "object", "frame", "time_s", "length_cm", "width_cm",
            "height_cm", "point_count", "table_rmse_mm", "robust_dimension_score",
        ])
        writer.writeheader()
        for sid, obj in primary.items():
            for row in obj["selected_frames"]:
                writer.writerow({"object_id": sid, "object": obj["name"], **{
                    k: row[k] for k in writer.fieldnames if k in row
                }})
    return path


def main() -> None:
    profiler = ResourceProfiler(
        "scale_test3_dimension_analysis", WS,
        metadata={"known_ruler_m": KNOWN_RULER_M, "primary_confidence_percentile": 50},
    )
    with profiler.stage("load_offline_vggt_predictions"):
        loaded = np.load(PREDICTIONS)
        pred = {k: np.array(loaded[k]) for k in loaded.files}
        points = pred["world_points_from_depth"]
        confidence = pred["depth_conf"]
        masks = pred["semantic_masks"]
    with profiler.stage("robust_15cm_ruler_scale_calibration"):
        calibration = robust_scale(points, masks)
        scale = calibration["scale_m_per_vggt_unit"]
    with profiler.stage("semantic_ground_and_local_plane_fits"):
        up, ground = robust_gravity(points, confidence, masks, pred["extrinsic"], scale)
    with profiler.stage("multiframe_object_outer_dimensions"):
        primary, sensitivity = measure_objects(points, confidence, masks, up, scale)
    result = {
        "experiment": {
            "video": str(ROOT / "scale_test3.mp4"),
            "workspace": str(WS),
            "frames": int(len(points)),
            "frame_interval_s": 0.5,
            "pipeline": "repository offline YOLOE semantic masks + offline VGGT Depthmap and Camera Branch",
            "point_source": "world_points_from_depth",
            "primary_confidence_percentile": PRIMARY_CONFIDENCE_PERCENTILE,
            "dimension_method": "largest semantic component; per-frame robust tabletop; horizontal PCA P2-P98 footprint; object-specific top height percentile; robust temporal fusion",
        },
        "scale_calibration": calibration,
        "ground_alignment": ground,
        "objects": primary,
        "confidence_sensitivity": sensitivity,
        "limitations": [
            "No independent ground-truth dimensions were provided, so these are estimates rather than absolute-error measurements.",
            "Transparent ruler and transparent bottle regions can produce incomplete or biased monocular depth.",
            "The bottle is reported as major footprint × minor footprint × height; it is not assumed perfectly cylindrical.",
            "Frame IQR captures reconstruction and segmentation repeatability, not total systematic calibration uncertainty.",
        ],
    }
    with profiler.stage("write_results_and_csv"):
        out = WS / "dimension_results.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        csv_path = write_csv(primary)
    with profiler.stage("render_numeric_figures"):
        make_figures(result)
    profile = profiler.finish(metadata={
        "scale_m_per_unit": scale,
        "ruler_anchors": calibration["n_accepted"],
        "red_box_frames": primary["3"]["n_selected_frames"],
        "bottle_frames": primary["4"]["n_selected_frames"],
    })
    print(json.dumps({
        "results": str(out), "csv": str(csv_path), "resource_profile": profile,
        "scale": calibration,
        "ground": {k: ground[k] for k in ground if k != "per_frame"},
        "objects": {
            obj["name"]: {"n": obj["n_selected_frames"], "dimensions_cm": obj["dimensions_cm"]}
            for obj in primary.values()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
