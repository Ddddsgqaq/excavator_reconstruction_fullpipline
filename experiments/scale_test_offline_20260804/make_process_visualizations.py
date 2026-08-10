"""Create intuitive quick-look figures for the scale-test offline experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gravity_alignment as ga
from resource_profiler import ResourceProfiler
from experiments.scale_test_offline_20260804 import analyze_scale_volume as analysis


WS = Path(__file__).resolve().parent
VIZ = WS / "visualizations"
VIZ.mkdir(exist_ok=True)

FRAME_IDS = [0, 7, 14, 19]
SEMANTIC_COLORS_RGB = {
    0: (105, 105, 105),
    1: (150, 150, 150),
    2: (45, 120, 210),
    3: (142, 68, 173),
    4: (214, 69, 65),
    5: (24, 169, 153),
}
SEMANTIC_COLORS_MPL = {
    key: np.asarray(value, dtype=np.float64) / 255.0
    for key, value in SEMANTIC_COLORS_RGB.items()
}
CLASS_NAMES = {
    0: "unlabeled",
    1: "tabletop",
    2: "15 cm ruler",
    3: "upright red box",
    4: "red flat box",
    5: "tissue box",
}


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _load_data():
    pred_file = np.load(WS / "predictions.npz")
    pred = {key: np.array(pred_file[key]) for key in pred_file.files}
    masks = np.load(WS / "semantic_masks_combined.npz")["semantic_masks"]
    with (WS / "experiment_results.json").open(encoding="utf-8") as f:
        results = json.load(f)
    with (WS / "native_terrain_analysis.json").open(encoding="utf-8") as f:
        terrain = json.load(f)
    return pred, masks, results, terrain


def semantic_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).copy()
    resized = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
    for semantic_id, color in SEMANTIC_COLORS_RGB.items():
        region = resized == semantic_id
        if region.any():
            rgb[region] = (
                0.42 * rgb[region] + 0.58 * np.asarray(color)
            ).astype(np.uint8)
    return Image.fromarray(rgb)


def labeled_panel(image: Image.Image, title: str, size=(260, 390)) -> Image.Image:
    canvas = Image.new("RGB", size, (248, 248, 248))
    image = image.convert("RGB")
    image.thumbnail((size[0] - 16, size[1] - 46))
    canvas.paste(image, ((size[0] - image.width) // 2, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), title, font=_font(17), fill=(25, 25, 25))
    return canvas


def make_pipeline_montage(masks: np.ndarray) -> Path:
    columns = ["RGB input", "semantic masks", "depth", "3D point map", "semantic 3D"]
    rows = []
    for frame in FRAME_IDS:
        rgb = Image.open(WS / "images" / f"{frame:06d}.png")
        images = [
            rgb,
            semantic_overlay(rgb, masks[frame]),
            Image.open(WS / "depth_vis" / f"depth_{frame:04d}.png"),
            Image.open(WS / "pointmap_vis" / f"pointmap_{frame:04d}.png"),
            Image.open(WS / "semantic_pointmap_vis" / f"semantic_pointmap_{frame:04d}.png"),
        ]
        rows.append([
            labeled_panel(image, f"{title}  |  frame {frame}")
            for image, title in zip(images, columns)
        ])

    panel_w, panel_h = rows[0][0].size
    header_h = 46
    sheet = Image.new(
        "RGB",
        (panel_w * len(columns), header_h + panel_h * len(rows)),
        (232, 232, 232),
    )
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(columns):
        draw.text(
            (col * panel_w + 12, 12), title,
            font=_font(19), fill=(20, 20, 20),
        )
    for row_idx, panels in enumerate(rows):
        for col_idx, panel in enumerate(panels):
            sheet.paste(panel, (col_idx * panel_w, header_h + row_idx * panel_h))
    out = VIZ / "pipeline_stage_montage.png"
    sheet.save(out)
    return out


def aligned_scene(pred: dict, masks: np.ndarray, results: dict):
    points = pred["world_points_from_depth"]
    confidence = pred["depth_conf"]
    gravity = ga.estimate_gravity(
        pred["extrinsic"], points, masks == 1, confidence, conf_thres=0.5
    )
    aligned = ga.apply_alignment_to_points(points, gravity.R_align)
    scale = results["scale_calibration"]["scale_m_per_vggt_unit"]
    table_core = (
        (masks == 1)
        & np.isfinite(aligned).all(axis=3)
        & (confidence >= 0.5 * float(confidence.max()))
    )
    ground_zero = float(np.median(aligned[..., 1][table_core]))
    metric = aligned * scale
    metric[..., 1] = (aligned[..., 1] - ground_zero) * scale
    return metric, gravity


def _configure_scene_axis(ax, title: str, elevation: float, azimuth: float,
                          bounds: tuple[np.ndarray, np.ndarray]):
    lo, hi = bounds
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[2], hi[2])
    ax.set_zlim(lo[1], hi[1])
    ax.set_xlabel("X (m)", labelpad=4)
    ax.set_ylabel("Z (m)", labelpad=4)
    ax.set_zlabel("height (m)", labelpad=4)
    ax.set_title(title, pad=8)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.grid(alpha=.18)
    ax.set_box_aspect((hi[0] - lo[0], hi[2] - lo[2], max(hi[1] - lo[1], .18)))


def select_shared_reconstruction_sample(
        pred: dict, masks: np.ndarray, metric: np.ndarray) -> dict:
    """Select one RGB-standard point sample reused by every reconstruction view."""
    confidence = pred["depth_conf"]
    confidence_threshold = float(np.percentile(confidence, 78))
    keep = (
        np.isfinite(metric).all(axis=3)
        & (confidence >= confidence_threshold)
    )
    xyz = metric[keep]
    images = np.transpose(pred["images"], (0, 2, 3, 1))
    rgb = np.clip(images[keep], 0, 1)
    semantic_ids = masks[keep]
    candidate_count = len(xyz)
    if len(xyz) > 120000:
        idx = np.random.default_rng(85).choice(len(xyz), 120000, replace=False)
        xyz, rgb, semantic_ids = xyz[idx], rgb[idx], semantic_ids[idx]
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    lo[1] = max(lo[1], -.08)
    hi[1] = min(max(hi[1], .12), .28)
    return {
        "xyz": xyz,
        "rgb": rgb,
        "semantic_ids": semantic_ids,
        "bounds": (lo, hi),
        "candidate_count": candidate_count,
        "confidence_threshold": confidence_threshold,
    }


RECONSTRUCTION_VIEWS = [
    (27, -57, "front oblique"),
    (32, 128, "reverse oblique"),
]


def make_rgb_reconstruction(sample: dict) -> Path:
    xyz = sample["xyz"]
    rgb = sample["rgb"]
    bounds = sample["bounds"]

    fig = plt.figure(figsize=(14, 6), dpi=180)
    for pos, (elev, azim, view_name) in enumerate(RECONSTRUCTION_VIEWS, start=1):
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        ax.scatter(
            xyz[:, 0], xyz[:, 2], xyz[:, 1],
            c=rgb, s=.18, alpha=.48, linewidths=0, rasterized=True,
        )
        _configure_scene_axis(
            ax, f"RGB reconstruction · {view_name}", elev, azim, bounds)
    fig.suptitle(
        f"Offline VGGT reconstruction · confidence top 22% · "
        f"shared {len(xyz):,}-point sample",
        fontsize=14, y=.98,
    )
    fig.tight_layout()
    out = VIZ / "reconstruction_3d_rgb.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _semantic_point_colors(semantic_ids: np.ndarray) -> np.ndarray:
    """Return total semantic coloring; unknown IDs remain visibly unlabeled."""
    colors = np.tile(
        SEMANTIC_COLORS_MPL[0], (len(semantic_ids), 1)
    ).astype(np.float64)
    for semantic_id, color in SEMANTIC_COLORS_MPL.items():
        colors[semantic_ids == semantic_id] = color
    return colors


def make_semantic_reconstruction(sample: dict) -> Path:
    xyz = sample["xyz"]
    semantic_ids = sample["semantic_ids"]
    bounds = sample["bounds"]
    semantic_rgb = _semantic_point_colors(semantic_ids)

    fig = plt.figure(figsize=(14, 6), dpi=180)
    for pos, (elev, azim, view_name) in enumerate(RECONSTRUCTION_VIEWS, start=1):
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        ax.scatter(
            xyz[:, 0], xyz[:, 2], xyz[:, 1],
            c=semantic_rgb, s=.18, alpha=.48, linewidths=0, rasterized=True,
        )
        _configure_scene_axis(
            ax, f"semantic colors · {view_name}", elev, azim, bounds)
        if pos == 1:
            handles = [
                Patch(facecolor=SEMANTIC_COLORS_MPL[sid], label=CLASS_NAMES[sid])
                for sid in range(6)
            ]
            ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.suptitle(
        "YOLOE colors on the exact RGB point sample · geometry and camera unchanged",
        fontsize=14, y=.98,
    )
    fig.tight_layout()
    out = VIZ / "reconstruction_3d_semantic.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_semantic_topview(sample: dict) -> Path:
    """Render a top projection of the exact sample used by both 3-D views."""
    xyz = sample["xyz"]
    semantic_ids = sample["semantic_ids"]
    lo, hi = sample["bounds"]
    semantic_rgb = _semantic_point_colors(semantic_ids)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    ax.scatter(
        xyz[:, 0], xyz[:, 2], c=semantic_rgb,
        s=.18, alpha=.48, linewidths=0, rasterized=True,
    )
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[2], hi[2])
    ax.set_aspect("equal")
    ax.set(
        xlabel="aligned X (m)", ylabel="aligned Z (m)",
        title=f"Semantic top projection · exact RGB-standard {len(xyz):,}-point sample",
    )
    handles = [
        Patch(facecolor=SEMANTIC_COLORS_MPL[sid], label=CLASS_NAMES[sid])
        for sid in range(6)
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)
    ax.grid(alpha=.15)
    fig.tight_layout()
    out = VIZ / "semantic_pointcloud_topview.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_elevation_surfaces(terrain: dict, results: dict) -> Path:
    grid = terrain["grid"]
    layers = terrain["layers"]
    res = int(grid["res"])
    scale = results["scale_calibration"]["scale_m_per_vggt_unit"]
    x = np.linspace(grid["x_min"], grid["x_max"], res) * scale
    z = np.linspace(grid["z_min"], grid["z_max"], res) * scale
    xx, zz = np.meshgrid(x, z)
    h_top = np.asarray(layers["H_top"], dtype=np.float64) * scale
    h_ground = np.asarray(layers["H_ground"], dtype=np.float64) * scale
    residual = np.asarray(layers["R"], dtype=np.float64) * scale
    semantic = np.asarray(layers["S_mode"], dtype=np.int32)
    zero = float(np.nanmedian(h_ground))
    h_top -= zero
    h_ground -= zero

    fig = plt.figure(figsize=(18, 6), dpi=170)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    surface = ax1.plot_surface(
        xx, zz, h_top, cmap="terrain", linewidth=0, antialiased=True,
        rcount=96, ccount=96,
    )
    ax1.set_title("H_top surface · metric elevation")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_zlabel("height (m)")
    ax1.view_init(31, -58)
    ax1.set_box_aspect((np.ptp(x), np.ptp(z), max(np.nanpercentile(h_top, 98) - np.nanpercentile(h_top, 2), .16)))
    fig.colorbar(surface, ax=ax1, shrink=.55, pad=.08, label="height (m)")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    residual_cm = residual * 100
    limit = float(np.nanpercentile(np.abs(residual_cm), 97))
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    facecolors = plt.get_cmap("coolwarm")(norm(np.nan_to_num(residual_cm)))
    ax2.plot_surface(
        xx, zz, residual_cm, facecolors=facecolors,
        linewidth=0, antialiased=True, rcount=96, ccount=96,
    )
    ax2.set_title("surface residual above ground")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_zlabel("residual (cm)")
    ax2.view_init(31, -58)
    ax2.set_box_aspect((np.ptp(x), np.ptp(z), max(2 * limit / 100, .14)))
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    fig.colorbar(scalar, ax=ax2, shrink=.55, pad=.08, label="residual (cm)")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(
        xx, zz, h_ground, color=(.75, .75, .75), alpha=.28,
        linewidth=0, rcount=80, ccount=80,
    )
    for semantic_id in (2, 3, 4, 5):
        region = (semantic == semantic_id) & np.isfinite(h_top)
        if region.any():
            ax3.scatter(
                xx[region], zz[region], h_top[region],
                s=6, alpha=.75, color=SEMANTIC_COLORS_MPL[semantic_id],
                label=CLASS_NAMES[semantic_id],
            )
    ax3.set_title("semantic objects over ground DEM")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_zlabel("height (m)")
    ax3.view_init(31, -58)
    ax3.set_box_aspect((np.ptp(x), np.ptp(z), .18))
    ax3.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Native offline elevation · 128×128 · gravity={terrain['gravity_source']}",
        fontsize=14, y=.98,
    )
    fig.tight_layout()
    out = VIZ / "elevation_3d_views.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_object_height_views(pred: dict, masks: np.ndarray, results: dict, gravity) -> Path:
    points = pred["world_points_from_depth"]
    confidence = pred["depth_conf"]
    scale = results["scale_calibration"]["scale_m_per_vggt_unit"]
    conf_threshold = float(np.percentile(confidence, 50))

    fig = plt.figure(figsize=(17, 5.7), dpi=180)
    for panel, semantic_id in enumerate((3, 4, 5), start=1):
        obj = results["objects"][str(semantic_id)]
        median_volume = obj["volume_ml"]["median"]
        row = min(obj["per_frame"], key=lambda item: abs(item["volume_ml"] - median_volume))
        frame = int(row["frame"])
        comp = analysis.select_component(masks[frame] == semantic_id, semantic_id)
        obj_mask = analysis.component_mask(comp)
        keep = (
            obj_mask
            & np.isfinite(points[frame]).all(axis=2)
            & (confidence[frame] >= conf_threshold)
        )
        q = points[frame][keep]
        center, local_up, _ = analysis.fit_frame_table_plane(
            points[frame], confidence[frame], masks[frame] == 1, gravity.n_grav
        )
        axis_u, axis_v = analysis.horizontal_basis(local_up)
        u = (q - center) @ axis_u * scale * 100
        v = (q - center) @ axis_v * scale * 100
        h = (q - center) @ local_up * scale * 100
        valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(h)
        u, v, h = u[valid], v[valid], h[valid]
        high = float(np.percentile(h, 99))
        low = -max(float(np.percentile(np.abs(h[h < 0]), 90)) if np.any(h < 0) else .1, .1)
        use = (h >= low) & (h <= high)
        u, v, h = u[use], v[use], h[use]
        if len(u) > 18000:
            idx = np.random.default_rng(700 + semantic_id).choice(len(u), 18000, replace=False)
            u, v, h = u[idx], v[idx], h[idx]

        ax = fig.add_subplot(1, 3, panel, projection="3d")
        scatter = ax.scatter(
            u, v, h, c=h, cmap="viridis", s=.8, alpha=.5,
            linewidths=0, rasterized=True,
        )
        u0, u1 = np.percentile(u, [1, 99])
        v0, v1 = np.percentile(v, [1, 99])
        plane_u, plane_v = np.meshgrid(
            np.linspace(u0, u1, 10), np.linspace(v0, v1, 10)
        )
        ax.plot_surface(
            plane_u, plane_v, np.zeros_like(plane_u),
            color=(.65, .65, .65), alpha=.28, linewidth=0,
        )
        dims = [
            obj[key]["median"] * 100
            for key in ("length_m", "width_m", "height_m")
        ]
        ax.set_title(
            f"{obj['name']} · frame {frame}\n"
            f"{dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f} cm · {median_volume:.1f} mL"
        )
        ax.set_xlabel("footprint U (cm)")
        ax.set_ylabel("footprint V (cm)")
        ax.set_zlabel("height (cm)")
        ax.set_zlim(min(0, np.percentile(h, 1)), max(np.percentile(h, 99), 1))
        ax.view_init(27, -55)
        ax.set_box_aspect((max(u1-u0, 1), max(v1-v0, 1), max(np.percentile(h, 99), 4)))
        fig.colorbar(scatter, ax=ax, shrink=.48, pad=.08, label="height (cm)")
    fig.suptitle(
        "Representative object point clouds above per-frame robust tabletop planes",
        fontsize=14, y=.99,
    )
    fig.tight_layout()
    out = VIZ / "object_height_3d_views.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def dashboard_tile(path: Path, title: str, size=(840, 520)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size[0] - 24, size[1] - 62))
    tile = Image.new("RGB", size, (247, 247, 247))
    tile.paste(image, ((size[0] - image.width) // 2, 50))
    ImageDraw.Draw(tile).text((16, 14), title, font=_font(23), fill=(20, 20, 20))
    return tile


def make_dashboard(paths: dict[str, Path]) -> Path:
    items = [
        ("Pipeline stages", paths["pipeline"]),
        ("RGB 3D reconstruction", paths["rgb"]),
        ("Semantic 3D reconstruction", paths["semantic"]),
        ("3D elevation views", paths["elevation"]),
        ("Object height clouds", paths["objects"]),
        ("Calibration and volume summary", VIZ / "experiment_overview.png"),
    ]
    tile_size = (840, 520)
    header_h = 70
    dashboard = Image.new(
        "RGB",
        (tile_size[0] * 2, header_h + tile_size[1] * 3),
        (226, 226, 226),
    )
    draw = ImageDraw.Draw(dashboard)
    draw.text(
        (20, 18), "scale_test.mp4 · offline reconstruction quick-look",
        font=_font(30), fill=(15, 15, 15),
    )
    for idx, (title, path) in enumerate(items):
        tile = dashboard_tile(path, title, tile_size)
        dashboard.paste(
            tile,
            ((idx % 2) * tile_size[0], header_h + (idx // 2) * tile_size[1]),
        )
    out = VIZ / "quicklook_dashboard.png"
    dashboard.save(out)
    return out


def main():
    profiler = ResourceProfiler("process_visualizations", WS)
    with profiler.stage("load_experiment_artifacts"):
        pred, masks, results, terrain = _load_data()
    with profiler.stage("metric_scale_and_gravity_alignment"):
        metric, gravity = aligned_scene(pred, masks, results)
    with profiler.stage("select_shared_reconstruction_points"):
        reconstruction_sample = select_shared_reconstruction_sample(pred, masks, metric)
    paths = {}
    with profiler.stage("render_pipeline_stage_montage"):
        paths["pipeline"] = make_pipeline_montage(masks)
    with profiler.stage("render_rgb_3d_reconstruction"):
        paths["rgb"] = make_rgb_reconstruction(reconstruction_sample)
    with profiler.stage("render_semantic_3d_reconstruction"):
        paths["semantic"] = make_semantic_reconstruction(reconstruction_sample)
    with profiler.stage("render_semantic_top_projection"):
        paths["semantic_topview"] = make_semantic_topview(reconstruction_sample)
    with profiler.stage("render_3d_elevation_views"):
        paths["elevation"] = make_elevation_surfaces(terrain, results)
    with profiler.stage("render_object_height_clouds"):
        paths["objects"] = make_object_height_views(pred, masks, results, gravity)
    with profiler.stage("compose_quicklook_dashboard"):
        paths["dashboard"] = make_dashboard(paths)
    manifest = {
        key: str(path.resolve())
        for key, path in paths.items()
    }
    semantic_ids = reconstruction_sample["semantic_ids"]
    manifest["reconstruction_sample"] = {
        "selection": "finite metric point and depth_conf >= percentile 78",
        "candidate_count": int(reconstruction_sample["candidate_count"]),
        "sample_count": int(len(reconstruction_sample["xyz"])),
        "confidence_threshold": float(
            reconstruction_sample["confidence_threshold"]),
        "semantic_histogram": {
            str(int(semantic_id)): int((semantic_ids == semantic_id).sum())
            for semantic_id in np.unique(semantic_ids)
        },
        "consumers": [
            "reconstruction_3d_rgb.png",
            "reconstruction_3d_semantic.png",
            "semantic_pointcloud_topview.png",
        ],
    }
    manifest["resource_profile"] = profiler.finish(metadata={
        "figures": len(paths),
        "shared_reconstruction_points": len(reconstruction_sample["xyz"]),
    })
    with (VIZ / "process_visualizations_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
