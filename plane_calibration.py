"""Local-plane metric calibration from a known rectangular reference object.

The VGGT coordinate system has arbitrary scale and can have a different scale
along elevation.  This module deliberately estimates a horizontal scale and a
vertical scale separately.  It is shared by the API-facing elevation viewer and
is independent from the offline experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PlaneCalibrationRequest:
    object_semantic_id: int
    object_length_m: float
    object_width_m: float
    object_height_m: float
    ground_semantic_id: int = 1
    confidence_percentile: float = 50.0


def _robust_plane(points: np.ndarray, up_prior: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a local plane, trimming residual outliers without overturning gravity."""
    if len(points) < 100:
        raise ValueError("too few local-ground points")
    if len(points) > 20_000:
        points = points[np.random.default_rng(2718).choice(len(points), 20_000, replace=False)]

    fit = points
    for _ in range(4):
        center = fit.mean(axis=0)
        _, _, axes = np.linalg.svd(fit - center, full_matrices=False)
        normal = axes[-1]
        if float(np.dot(normal, up_prior)) < 0:
            normal = -normal
        # The semantic surface refines the gravity frame; it must not replace it.
        if abs(float(np.dot(normal, up_prior))) < np.cos(np.deg2rad(35.0)):
            normal = up_prior.copy()
        normal /= np.linalg.norm(normal)
        residual = (fit - center) @ normal
        cutoff = max(float(np.percentile(np.abs(residual), 70)), 1e-5)
        trimmed = fit[np.abs(residual) <= cutoff]
        if len(trimmed) < 100:
            break
        fit = trimmed

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


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return None
    areas = np.bincount(labels.ravel())[1:]
    label = int(np.argmax(areas)) + 1
    component = labels == label
    return component if int(component.sum()) >= 100 else None


def _measure_frame(points: np.ndarray, confidence: np.ndarray, semantic: np.ndarray,
                   up_prior: np.ndarray, request: PlaneCalibrationRequest) -> dict | None:
    finite = np.isfinite(points).all(axis=2) & np.isfinite(confidence)
    target = _largest_component(semantic == request.object_semantic_id)
    if target is None:
        return None
    # Reject clipped targets: the known reference must be completely observed.
    ys, xs = np.where(target)
    if xs.min() < 2 or ys.min() < 2 or xs.max() >= target.shape[1] - 2 or ys.max() >= target.shape[0] - 2:
        return None

    ground = (semantic == request.ground_semantic_id) & finite
    if int(ground.sum()) < 100:
        return None
    ground_threshold = float(np.percentile(confidence[ground], 70))
    ground_points = points[ground & (confidence >= ground_threshold)]
    try:
        center, normal, residual = _robust_plane(ground_points, up_prior)
    except ValueError:
        return None

    object_threshold = float(np.percentile(confidence[finite], request.confidence_percentile))
    q = points[target & finite & (confidence >= object_threshold)]
    if len(q) < 100:
        return None
    heights = (q - center) @ normal
    if float(np.mean(heights > 0)) < 0.5:
        return None
    lower = -max(float(np.percentile(np.abs(residual), 90)), 1e-4)
    upper = float(np.percentile(heights, 99.5))
    keep = (heights >= lower) & (heights <= upper)
    q, heights = q[keep], heights[keep]
    if len(q) < 100:
        return None

    # PCA removes arbitrary yaw for a non-square reference.  For a square, the
    # two in-plane singular values are equal and PCA has no stable orientation;
    # use a deterministic tangent basis in that case instead of a random 45 deg
    # rotation that would inflate the axis-aligned footprint.
    tangent = q - np.outer(heights, normal)
    tangent -= tangent.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(tangent, full_matrices=False)
    if singular_values[1] / max(singular_values[0], 1e-12) > 0.95:
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(normal, reference))) > 0.9:
            reference = np.array([1.0, 0.0, 0.0])
        axis_u = np.cross(normal, reference)
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        uv = np.column_stack([tangent @ axis_u, tangent @ axis_v])
    else:
        uv = tangent @ axes[:2].T
    lo, hi = np.percentile(uv, [2, 98], axis=0)
    side_a, side_b = np.sort(hi - lo)[::-1]
    if side_b <= 1e-8:
        return None
    return {
        "length_units": float(side_a),
        "width_units": float(side_b),
        "height_units": float(np.percentile(heights, 95)),
        "point_count": int(len(q)),
        "plane_rmse_units": float(np.sqrt(np.mean(residual ** 2))),
        "plane_normal": normal.tolist(),
    }


def calibrate_local_plane(points: np.ndarray, confidence: np.ndarray, semantic_masks: np.ndarray,
                          up_prior: np.ndarray, request: PlaneCalibrationRequest) -> dict:
    """Return horizontal/vertical metres-per-unit factors from all valid frames.

    The longer known side is paired with the longer measured footprint side so
    users may enter length and width in either camera orientation.
    """
    if min(request.object_length_m, request.object_width_m, request.object_height_m) <= 0:
        raise ValueError("all known reference dimensions must be positive")
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape (frames, height, width, 3)")
    if confidence.shape != points.shape[:3] or semantic_masks.shape != points.shape[:3]:
        raise ValueError("confidence and semantic masks must match the point grid")

    rows = []
    for frame in range(points.shape[0]):
        row = _measure_frame(points[frame], confidence[frame], semantic_masks[frame], up_prior, request)
        if row is not None:
            row["frame"] = frame
            rows.append(row)
    if not rows:
        raise ValueError("no valid calibration views; check semantic IDs, complete visibility, and ground mask")

    # Keep low-residual views, but fall back to all valid views when a short clip
    # does not leave enough samples after gating.
    rmses = np.asarray([r["plane_rmse_units"] for r in rows])
    cutoff = float(np.percentile(rmses, 75))
    selected = [r for r in rows if r["plane_rmse_units"] <= cutoff] or rows
    known_long, known_short = sorted((request.object_length_m, request.object_width_m), reverse=True)
    horizontal_candidates = []
    vertical_candidates = []
    for row in selected:
        horizontal_candidates.extend((known_long / row["length_units"], known_short / row["width_units"]))
        vertical_candidates.append(request.object_height_m / row["height_units"])
    horizontal_scale = float(np.median(horizontal_candidates))
    vertical_scale = float(np.median(vertical_candidates))

    def _rmad(values: list[float]) -> float:
        values_a = np.asarray(values, dtype=float)
        median = float(np.median(values_a))
        return float(np.median(np.abs(values_a - median)) / max(abs(median), 1e-12) * 100)

    return {
        "method": "known reference object + robust local ground plane + anisotropic scale",
        "horizontal_m_per_vggt_unit": horizontal_scale,
        "vertical_m_per_vggt_unit": vertical_scale,
        "vertical_to_horizontal_ratio": vertical_scale / horizontal_scale,
        "reference": {
            "semantic_id": request.object_semantic_id,
            "ground_semantic_id": request.ground_semantic_id,
            "length_m": request.object_length_m,
            "width_m": request.object_width_m,
            "height_m": request.object_height_m,
        },
        "selected_frames": [r["frame"] for r in selected],
        "total_valid_frames": len(rows),
        "horizontal_scale_rmad_percent": _rmad(horizontal_candidates),
        "vertical_scale_rmad_percent": _rmad(vertical_candidates),
        "frames": rows,
    }
