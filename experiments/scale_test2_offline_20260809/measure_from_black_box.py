"""Estimate red-bottle dimensions from scale_test2 using a known black box.

The metric conversion intentionally uses two factors: the black box's two
13 cm in-plane sides calibrate the local surface axes, and its 3 cm thickness
calibrates elevation.  This avoids hiding a vertical reconstruction bias in a
single isotropic scale factor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gravity_alignment as ga
from experiments.scale_test_offline_20260804 import analyze_scale_volume as az


WS = Path(__file__).resolve().parent
PRED = WS / "predictions.npz"
BLACK_BOX_ID = 2
RED_BOTTLE_ID = 3
KNOWN_BOX_CM = {"length": 13.0, "width": 13.0, "height": 3.0}


def rmad(values: list[float]) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)) / max(abs(median), 1e-12) * 100)


def raw_measurements(points: np.ndarray, confidence: np.ndarray, masks: np.ndarray,
                     up: np.ndarray, semantic_id: int) -> list[dict]:
    """The existing local-plane/elevation-grid estimator, before metric scaling."""
    threshold = float(np.percentile(confidence, 50))
    rows = []
    for frame in range(len(masks)):
        # ID 2 was a ruler in the earlier experiment and its shared selector
        # applies an elongated-shape gate. Here it is a square calibration box.
        components = az.connected_components(masks[frame] == semantic_id)
        comp = max(components, key=lambda row: row["area"]) if components else None
        if comp is None or comp["area"] < 500:
            continue
        x, y, w, h = comp["bbox"]
        H, W = masks[frame].shape
        if x < 2 or y < 2 or x + w > W - 2 or y + h > H - 2:
            continue
        obj_mask = az.component_mask(comp)
        keep = obj_mask & np.isfinite(points[frame]).all(axis=2) & (confidence[frame] >= threshold)
        q = points[frame][keep]
        if len(q) < 100:
            continue
        try:
            plane_center, local_up, residual = az.fit_frame_table_plane(
                points[frame], confidence[frame], masks[frame] == 1, up)
        except ValueError:
            continue
        heights = (q - plane_center) @ local_up
        if float(np.mean(heights > 0)) < 0.5:
            continue
        lower = -max(float(np.percentile(np.abs(residual), 90)), 1e-4)
        upper = float(np.percentile(heights, 99.5))
        geometry_keep = (heights >= lower) & (heights <= upper)
        q, heights = q[geometry_keep], heights[geometry_keep]
        if len(q) < 100:
            continue
        u, v = az.horizontal_basis(local_up)
        try:
            measure = az.elevation_grid_volume(np.column_stack([q @ u, q @ v]), heights)
        except ValueError:
            continue
        rows.append({
            "frame": int(frame),
            "length_units": measure["length_units"],
            "width_units": measure["width_units"],
            "height_units": measure["height_units"],
            "volume_units3": measure["volume_units3"],
            "observed_cell_fraction": measure["observed_cell_fraction"],
            "plane_rmse_units": float(np.sqrt(np.mean(residual ** 2))),
        })
    return rows


def quality_rows(rows: list[dict]) -> list[dict]:
    """Fixed robust gates remove incomplete views without selecting by size."""
    if not rows:
        return []
    fractions = np.asarray([r["observed_cell_fraction"] for r in rows])
    rmses = np.asarray([r["plane_rmse_units"] for r in rows])
    min_fraction = max(0.55, float(np.percentile(fractions, 25)))
    max_rmse = float(np.percentile(rmses, 75))
    selected = [r for r in rows if r["observed_cell_fraction"] >= min_fraction
                and r["plane_rmse_units"] <= max_rmse]
    return selected or rows


def summary(rows: list[dict], horizontal_scale: float, vertical_scale: float) -> dict:
    values = {}
    for key, scale in (("length_cm", horizontal_scale * 100),
                       ("width_cm", horizontal_scale * 100),
                       ("height_cm", vertical_scale * 100),
                       ("volume_ml", horizontal_scale ** 2 * vertical_scale * 1e6)):
        source = key.replace("_cm", "_units").replace("volume_ml", "volume_units3")
        array = np.asarray([row[source] * scale for row in rows], dtype=float)
        values[key] = {
            "median": float(np.median(array)),
            "q25": float(np.percentile(array, 25)),
            "q75": float(np.percentile(array, 75)),
            "rmad_percent": rmad(array.tolist()),
        }
    return values


def main() -> None:
    loaded = np.load(PRED)
    points = loaded["world_points"]
    confidence = loaded["world_points_conf"]
    masks = loaded["semantic_masks"]
    gravity = ga.estimate_gravity(loaded["extrinsic"], points, masks == 1, confidence, conf_thres=0.5)

    black_all = raw_measurements(points, confidence, masks, gravity.n_grav, BLACK_BOX_ID)
    bottle_all = raw_measurements(points, confidence, masks, gravity.n_grav, RED_BOTTLE_ID)
    black = quality_rows(black_all)
    bottle = quality_rows(bottle_all)
    if not black or not bottle:
        raise RuntimeError("No quality measurements survived for black box or red bottle")

    # The known box is square in the local plane, so average the two estimates.
    box_planar_units = np.asarray([(r["length_units"] + r["width_units"]) / 2 for r in black])
    box_vertical_units = np.asarray([r["height_units"] for r in black])
    horizontal_scale = 0.13 / float(np.median(box_planar_units))
    vertical_scale = 0.03 / float(np.median(box_vertical_units))
    bottle_summary = summary(bottle, horizontal_scale, vertical_scale)
    black_recovered = summary(black, horizontal_scale, vertical_scale)

    output = {
        "input": {"video": str(ROOT / "scale_test2.mp4"), "sampled_frames": int(len(masks)),
                  "known_black_box_cm": KNOWN_BOX_CM},
        "method": "existing local plane + elevation grid; black-box anisotropic scale calibration",
        "gravity_source": gravity.source,
        "calibration": {
            "horizontal_m_per_vggt_unit": horizontal_scale,
            "vertical_m_per_vggt_unit": vertical_scale,
            "vertical_to_horizontal_ratio": vertical_scale / horizontal_scale,
            "black_box_selected_frames": [r["frame"] for r in black],
            "black_box_total_valid_frames": len(black_all),
            "recovered_black_box": black_recovered,
        },
        "red_bottle": {
            "selected_frames": [r["frame"] for r in bottle],
            "total_valid_frames": len(bottle_all),
            "dimensions": bottle_summary,
        },
        "per_frame_raw": {"black_box": black_all, "red_bottle": bottle_all},
    }
    path = WS / "red_bottle_measurement.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
