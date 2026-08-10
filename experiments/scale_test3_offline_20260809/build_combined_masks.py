"""Combine YOLOE text semantics with a cleaned bottle visual prompt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from resource_profiler import ResourceProfiler


WS = Path(__file__).resolve().parent
TEXT_RESPONSE = WS / "yoloe_text_response.json"
BOTTLE_RESPONSE = WS / "yoloe_bottle_visual_response.json"
RULER_ID = 2
BOTTLE_ID = 4


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def main():
    profiler = ResourceProfiler("combine_yoloe_semantics", WS)
    with profiler.stage("load_text_and_visual_masks"):
        text_response = json.load(TEXT_RESPONSE.open(encoding="utf-8"))
        bottle_response = json.load(BOTTLE_RESPONSE.open(encoding="utf-8"))
        text_masks = np.load(text_response["semantic_masks_path"])["semantic_masks"]
        visual_masks = np.load(bottle_response["semantic_masks_path"])["semantic_masks"]
        if text_masks.shape != visual_masks.shape:
            raise ValueError(
                f"Mask shape mismatch: text={text_masks.shape}, visual={visual_masks.shape}")

    with profiler.stage("clean_and_merge_bottle_instances"):
        combined = text_masks.copy().astype(np.uint8)
        combined[combined == BOTTLE_ID] = 0
        bottle_areas = []
        for frame in range(len(combined)):
            bottle = largest_component(visual_masks[frame] > 0)
            # The ruler is the metric anchor and keeps priority over incidental
            # bottle-prompt leakage. Box/table labels are overwritten by bottle.
            bottle &= text_masks[frame] != RULER_ID
            combined[frame][bottle] = BOTTLE_ID
            bottle_areas.append(int(bottle.sum()))

    with profiler.stage("write_masks_metadata_and_preview"):
        output_path = WS / "semantic_masks_combined.npz"
        np.savez_compressed(output_path, semantic_masks=combined)
        semantic_id_map = {
            "wooden tabletop": 1,
            "transparent ruler": 2,
            "red rectangular plastic box": 3,
            "red plastic bottle": 4,
        }
        counts = {
            str(semantic_id): [
                int((combined[frame] == semantic_id).sum())
                for frame in range(len(combined))
            ]
            for semantic_id in range(5)
        }
        preview_ids = (0, 5, 10, 15, 20)
        palette = {
            1: np.array([130, 105, 210]),
            2: np.array([75, 220, 90]),
            3: np.array([235, 65, 60]),
            4: np.array([45, 200, 205]),
        }
        panels = []
        for frame in preview_ids:
            image = np.asarray(Image.open(WS / "images" / f"{frame:06d}.png").convert("RGB"))
            overlay = image.copy()
            for semantic_id, color in palette.items():
                region = combined[frame] == semantic_id
                overlay[region] = (
                    .42 * overlay[region] + .58 * color
                ).astype(np.uint8)
            panel = Image.fromarray(overlay)
            panel.thumbnail((640, 360))
            canvas = Image.new("RGB", (640, 390), "white")
            canvas.paste(panel, ((640 - panel.width) // 2, 30))
            ImageDraw.Draw(canvas).text(
                (8, 7), f"combined semantics · frame {frame}", fill="black")
            panels.append(canvas)
        sheet = Image.new("RGB", (640, 390 * len(panels)), (230, 230, 230))
        for index, panel in enumerate(panels):
            sheet.paste(panel, (0, index * 390))
        preview_path = WS / "combined_semantics_preview.jpg"
        sheet.save(preview_path, quality=88)

    profile_path = profiler.finish(metadata={"frames": len(combined)})
    metadata = {
        "semantic_masks_path": str(output_path.resolve()),
        "semantic_id_map": semantic_id_map,
        "merge_rule": (
            "text semantics; remove text bottle; overlay largest visual bottle "
            "component per frame except on ruler pixels"
        ),
        "text_response": str(TEXT_RESPONSE.resolve()),
        "bottle_visual_response": str(BOTTLE_RESPONSE.resolve()),
        "bottle_area_per_frame": bottle_areas,
        "semantic_pixel_counts_per_frame": counts,
        "preview": str(preview_path.resolve()),
        "resource_profile": profile_path,
    }
    with (WS / "semantic_masks_combined_meta.json").open(
            "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
