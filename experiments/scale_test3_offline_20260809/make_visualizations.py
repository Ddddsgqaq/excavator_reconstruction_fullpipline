"""Quick-look visualizations for the scale_test3 offline experiment.

The RGB and semantic reconstruction figures intentionally reuse the exact
same confidence-filtered point indices.  Only point color changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gravity_alignment as ga
from resource_profiler import ResourceProfiler


WS = Path(__file__).resolve().parent
VIZ = WS / "visualizations"
VIZ.mkdir(exist_ok=True)
FRAME_IDS = [0, 7, 14, 19]
CLASS_NAMES = {
    0: "unlabeled", 1: "tabletop", 2: "15 cm ruler",
    3: "red box", 4: "red bottle",
}
COLORS_RGB = {
    0: (105, 105, 105), 1: (155, 155, 155), 2: (45, 120, 210),
    3: (214, 69, 65), 4: (123, 63, 152),
}
COLORS = {k: np.asarray(v, dtype=float) / 255 for k, v in COLORS_RGB.items()}


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load():
    z = np.load(WS / "predictions.npz")
    pred = {k: np.asarray(z[k]) for k in z.files}
    results = json.loads((WS / "dimension_results.json").read_text(encoding="utf-8"))
    return pred, pred["semantic_masks"], results


def semantic_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).copy()
    resized = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
    for sid, color in COLORS_RGB.items():
        region = resized == sid
        if region.any():
            rgb[region] = (0.42 * rgb[region] + 0.58 * np.asarray(color)).astype(np.uint8)
    return Image.fromarray(rgb)


def panel(image: Image.Image, title: str, size=(280, 380)) -> Image.Image:
    canvas = Image.new("RGB", size, (248, 248, 248))
    image = image.convert("RGB")
    image.thumbnail((size[0] - 16, size[1] - 46))
    canvas.paste(image, ((size[0] - image.width) // 2, 38))
    ImageDraw.Draw(canvas).text((10, 9), title, font=font(16), fill=(25, 25, 25))
    return canvas


def pipeline_montage(masks: np.ndarray) -> Path:
    columns = ["RGB input", "YOLOE overlay", "VGGT depth", "VGGT point map", "VGGT + semantics"]
    rows = []
    for frame in FRAME_IDS:
        rgb = Image.open(WS / "images" / f"{frame:06d}.png")
        items = [
            rgb, semantic_overlay(rgb, masks[frame]),
            Image.open(WS / "depth_vis" / f"depth_{frame:04d}.png"),
            Image.open(WS / "pointmap_vis" / f"pointmap_{frame:04d}.png"),
            Image.open(WS / "semantic_pointmap_vis" / f"semantic_pointmap_{frame:04d}.png"),
        ]
        rows.append([panel(img, f"{title} · f{frame}") for img, title in zip(items, columns)])
    pw, ph = rows[0][0].size
    sheet = Image.new("RGB", (pw * len(columns), 46 + ph * len(rows)), (232, 232, 232))
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(columns):
        draw.text((col * pw + 12, 12), title, font=font(18), fill=(20, 20, 20))
    for row, images in enumerate(rows):
        for col, img in enumerate(images):
            sheet.paste(img, (col * pw, 46 + row * ph))
    out = VIZ / "pipeline_stage_montage.png"
    sheet.save(out)
    return out


def aligned_metric(pred: dict, masks: np.ndarray, scale: float):
    points = pred["world_points_from_depth"]
    confidence = pred["depth_conf"]
    gravity = ga.estimate_gravity(
        pred["extrinsic"], points, masks == 1, confidence, conf_thres=0.5
    )
    aligned = ga.apply_alignment_to_points(points, gravity.R_align)
    core = (
        (masks == 1) & np.isfinite(aligned).all(axis=3)
        & (confidence >= np.percentile(confidence, 70))
    )
    zero = float(np.median(aligned[..., 1][core]))
    metric = aligned * scale
    metric[..., 1] = (aligned[..., 1] - zero) * scale
    return metric


def shared_sample(pred: dict, masks: np.ndarray, metric: np.ndarray) -> dict:
    confidence = pred["depth_conf"]
    threshold = float(np.percentile(confidence, 78))
    keep = np.isfinite(metric).all(axis=3) & (confidence >= threshold)
    xyz = metric[keep]
    rgb = np.transpose(pred["images"], (0, 2, 3, 1))[keep]
    semantic = masks[keep]
    candidates = len(xyz)
    if len(xyz) > 120000:
        idx = np.random.default_rng(309).choice(len(xyz), 120000, replace=False)
        xyz, rgb, semantic = xyz[idx], rgb[idx], semantic[idx]
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    lo[1] = max(lo[1], -0.08)
    hi[1] = min(max(hi[1], 0.18), 0.35)
    return {
        "xyz": xyz, "rgb": np.clip(rgb, 0, 1), "semantic": semantic,
        "bounds": (lo, hi), "threshold": threshold, "candidates": candidates,
    }


def configure(ax, title: str, bounds, elev=29, azim=-58):
    lo, hi = bounds
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2]); ax.set_zlim(lo[1], hi[1])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("height (m)")
    ax.set_title(title); ax.view_init(elev=elev, azim=azim); ax.grid(alpha=.18)
    ax.set_box_aspect((hi[0]-lo[0], hi[2]-lo[2], max(hi[1]-lo[1], .18)))


def reconstruction_views(sample: dict) -> list[Path]:
    xyz, rgb, semantic = sample["xyz"], sample["rgb"], sample["semantic"]
    views = [(28, -58, "front oblique"), (34, 126, "reverse oblique")]
    outputs = []
    fig = plt.figure(figsize=(14, 6), dpi=180)
    for pos, (elev, azim, name) in enumerate(views, 1):
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], c=rgb, s=.18, alpha=.48,
                   linewidths=0, rasterized=True)
        configure(ax, f"RGB reconstruction · {name}", sample["bounds"], elev, azim)
    fig.suptitle(f"Offline VGGT · exact shared {len(xyz):,}-point RGB sample")
    fig.tight_layout()
    out = VIZ / "reconstruction_3d_rgb.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    outputs.append(out)

    colors = np.tile(COLORS[0], (len(semantic), 1))
    for sid, color in COLORS.items():
        colors[semantic == sid] = color
    fig = plt.figure(figsize=(14, 6), dpi=180)
    for pos, (elev, azim, name) in enumerate(views, 1):
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], c=colors, s=.18, alpha=.48,
                   linewidths=0, rasterized=True)
        configure(ax, f"YOLOE semantics · {name}", sample["bounds"], elev, azim)
        if pos == 1:
            ax.legend(handles=[Patch(facecolor=COLORS[s], label=CLASS_NAMES[s]) for s in CLASS_NAMES],
                      loc="upper left", fontsize=8)
    fig.suptitle("Semantics projected onto the exact RGB point indices · geometry unchanged")
    fig.tight_layout()
    out = VIZ / "reconstruction_3d_semantic.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    outputs.append(out)

    height = xyz[:, 1]
    fig = plt.figure(figsize=(14, 6), dpi=180)
    for pos, (elev, azim, name) in enumerate(views, 1):
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        cloud = ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], c=height, cmap="turbo",
                           vmin=-.02, vmax=np.percentile(height, 99), s=.18, alpha=.52,
                           linewidths=0, rasterized=True)
        configure(ax, f"elevation · {name}", sample["bounds"], elev, azim)
    fig.colorbar(cloud, ax=fig.axes, shrink=.65, pad=.04, label="height above tabletop (m)")
    fig.suptitle("VGGT metric elevation after 15 cm ruler scaling")
    out = VIZ / "reconstruction_3d_elevation.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    outputs.append(out)
    return outputs


def annotated_dimensions(masks: np.ndarray, results: dict) -> Path:
    selections = {}
    for sid in ("3", "4"):
        obj = results["objects"][sid]
        target = np.asarray([
            obj["dimensions_cm"][d]["median"] for d in ("length", "width", "height")
        ])
        rows = obj["selected_frames"]
        row = min(rows, key=lambda r: np.linalg.norm(
            np.log(np.asarray([r["length_cm"], r["width_cm"], r["height_cm"]]) / target)
        ))
        selections[int(sid)] = row
    panels = []
    for sid, row in selections.items():
        frame = int(row["frame"])
        image = Image.open(WS / "images" / f"{frame:06d}.png").convert("RGB")
        overlay = semantic_overlay(image, masks[frame])
        draw = ImageDraw.Draw(overlay)
        x, y, w, h = row["bbox_model"]
        sx, sy = image.width / masks.shape[2], image.height / masks.shape[1]
        box = (int(x*sx), int(y*sy), int((x+w)*sx), int((y+h)*sy))
        draw.rectangle(box, outline=COLORS_RGB[sid], width=8)
        label = (
            f"{CLASS_NAMES[sid]}  {row['length_cm']:.1f} x "
            f"{row['width_cm']:.1f} x {row['height_cm']:.1f} cm"
        )
        tx, ty = box[0] + 8, max(8, box[1] - 42)
        draw.rectangle((tx-5, ty-4, tx+660, ty+34), fill=(255, 255, 255))
        draw.text((tx, ty), label, font=font(25), fill=(20, 20, 20))
        panels.append(panel(overlay, f"Representative accepted frame {frame}", size=(720, 460)))
    sheet = Image.new("RGB", (720, 920), (235, 235, 235))
    for i, p in enumerate(panels):
        sheet.paste(p, (0, i*460))
    out = VIZ / "representative_dimension_overlays.png"; sheet.save(out)
    return out


def main():
    profiler = ResourceProfiler("scale_test3_visualizations", WS)
    with profiler.stage("load_predictions_and_measurements"):
        pred, masks, results = load()
    with profiler.stage("render_pipeline_montage"):
        montage = pipeline_montage(masks)
    with profiler.stage("align_and_select_shared_rgb_sample"):
        metric = aligned_metric(pred, masks, results["scale_calibration"]["scale_m_per_vggt_unit"])
        sample = shared_sample(pred, masks, metric)
    with profiler.stage("render_rgb_semantic_and_elevation_reconstructions"):
        recon = reconstruction_views(sample)
    with profiler.stage("render_representative_dimension_overlays"):
        annotated = annotated_dimensions(masks, results)
    manifest = {
        "shared_point_contract": "RGB, semantic, and elevation reconstructions use identical point indices",
        "shared_points": int(len(sample["xyz"])),
        "candidate_points": int(sample["candidates"]),
        "confidence_threshold": sample["threshold"],
        "figures": [str(montage), *map(str, recon), str(annotated)],
    }
    (VIZ / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    profile = profiler.finish(metadata={"shared_points": int(len(sample["xyz"]))})
    print(json.dumps({**manifest, "resource_profile": profile}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
