"""
VGGT reconstruction service — run in VGGT conda environment.

  cd /home/maomaoyu/WS/vggt
  python /home/maomaoyu/WS/vggt_yoloe/vggt_service.py --port 8002
"""

import os
import sys
import glob
import argparse
import gc
import json
import numpy as np
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import uvicorn
import cv2
from scipy.spatial.transform import Rotation

VGGT_DIR = "/home/maomaoyu/WS/vggt"
sys.path.insert(0, VGGT_DIR)
sys.path.append(os.path.join(VGGT_DIR, "vggt"))

from visual_util import (
    predictions_to_glb,
    generate_depth_visualizations,
    generate_pointmap_visualizations,
    generate_semantic_depth_visualizations,
    generate_semantic_pointmap_visualizations,
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

sys.path.insert(0, "/home/maomaoyu/WS/vggt_yoloe")
from elevation_plane import fit_elevation_to_glb, _select_ground_aligned_mask
from gravity_alignment import estimate_gravity, apply_alignment_to_points, apply_alignment_to_extrinsics

VGGT_YOLOE_DIR = "/home/maomaoyu/WS/vggt_yoloe"

app = FastAPI(title="VGGT Reconstruction Service")
app.mount("/static", StaticFiles(directory=VGGT_YOLOE_DIR), name="static")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading VGGT model...")
_model = VGGT()
_model.load_state_dict(torch.load(os.path.join(VGGT_DIR, "model.pt"), map_location=DEVICE))
_model.eval()
_model = _model.to(DEVICE)
print(f"VGGT model ready on {DEVICE}")


# ── Request models ────────────────────────────────────────────────────────────

class ReconstructRequest(BaseModel):
    working_dir: str
    semantic_masks_path: Optional[str] = None
    conf_thres: float = 50.0
    frame_filter: str = "All"
    mask_black_bg: bool = False
    mask_white_bg: bool = False
    show_cam: bool = True
    mask_sky: bool = False
    prediction_mode: str = "Depthmap and Camera Branch"
    enable_semantic: bool = False


class VisualizeRequest(BaseModel):
    working_dir: str
    semantic_masks_path: Optional[str] = None
    conf_thres: float = 50.0
    frame_filter: str = "All"
    mask_black_bg: bool = False
    mask_white_bg: bool = False
    show_cam: bool = True
    mask_sky: bool = False
    prediction_mode: str = "Depthmap and Camera Branch"
    enable_semantic: bool = False


class EditPointcloudRequest(BaseModel):
    working_dir: str
    semantic_masks_path: str
    selected_semantic_ids: List[int]
    operation: str = "delete"  # "delete" or "extract"
    conf_thres: float = 50.0
    frame_filter: str = "All"
    mask_black_bg: bool = False
    mask_white_bg: bool = False
    show_cam: bool = True
    mask_sky: bool = False
    prediction_mode: str = "Depthmap and Camera Branch"


class ElevationViewerRequest(BaseModel):
    working_dir: str
    conf_thres: float = 50.0
    prediction_mode: str = "Depthmap and Camera Branch"
    ground_percentile: float = 20.0
    use_ground_filter: bool = True
    # Optional semantic edit applied to the raw cloud before alignment/DEM.
    # Mirrors the Point Cloud Editing tab; empty list = use the original scene.
    semantic_filter_ids: List[int] = []
    semantic_filter_mode: str = "delete"  # "delete" or "extract"
    semantic_masks_path: str = ""


class FitElevationRequest(BaseModel):
    working_dir: str
    source_glb_path: str = ""          # path to the base GLB to merge with
    grid_resolution: int = 128         # DEM grid size (NxN)
    colormap: str = "terrain"          # matplotlib colormap name
    ground_percentile: float = 20.0    # lowest N% of points used as ground candidates (fallback)
    use_ransac: bool = True            # deprecated; superseded by gravity_alignment cascade
    conf_thres: float = 50.0
    prediction_mode: str = "Depthmap and Camera Branch"
    scale_factor: float = 1.0          # multiplier applied after gravity alignment for absolute units
    use_ground_filter: bool = True     # if false, interpolate DEM from all aligned points


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_inference(working_dir: str) -> dict:
    image_names = sorted(glob.glob(os.path.join(working_dir, "images", "*")))
    if not image_names:
        raise HTTPException(400, "No images found in working_dir/images/")

    images = load_and_preprocess_images(image_names).to(DEVICE)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = _model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions["pose_enc_list"] = None

    depth_map = predictions["depth"]
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )
    torch.cuda.empty_cache()
    return predictions


def _load_semantic_masks(masks_path: str, pred_h: int, pred_w: int) -> Optional[np.ndarray]:
    if not masks_path or not os.path.exists(masks_path):
        return None
    loaded = np.load(masks_path)
    masks = loaded["semantic_masks"]
    if masks.shape[1] != pred_h or masks.shape[2] != pred_w:
        resized = np.zeros((masks.shape[0], pred_h, pred_w), dtype=np.uint8)
        for i in range(masks.shape[0]):
            resized[i] = cv2.resize(masks[i], (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
        masks = resized
    return masks


def _glb_filename(working_dir, conf_thres, frame_filter, mask_black_bg,
                  mask_white_bg, show_cam, mask_sky, prediction_mode, enable_semantic) -> str:
    tag = (
        f"glbscene_{conf_thres}"
        f"_{frame_filter.replace('.','_').replace(':','').replace(' ','_')}"
        f"_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}"
        f"_sky{mask_sky}_pred{prediction_mode.replace(' ','_')}_sem{enable_semantic}.glb"
    )
    return os.path.join(working_dir, tag)


def _frame_filter_choices(working_dir: str) -> List[str]:
    images_dir = os.path.join(working_dir, "images")
    if not os.path.isdir(images_dir):
        return ["All"]
    files = sorted(os.listdir(images_dir))
    return ["All"] + [f"{i}: {f}" for i, f in enumerate(files)]


# ── Edited-scene registry ─────────────────────────────────────────────────────
# Records each point-cloud edit so the elevation viewer can re-derive the same
# filtered cloud from raw predictions. One JSON file per workspace.
SCENES_FILENAME = "edited_scenes.json"


def _scenes_path(working_dir: str) -> str:
    return os.path.join(working_dir, SCENES_FILENAME)


def _load_scenes(working_dir: str) -> List[dict]:
    path = _scenes_path(working_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _register_scene(working_dir: str, scene: dict) -> None:
    """Add or replace an edited-scene entry, keyed by its `id`."""
    scenes = _load_scenes(working_dir)
    scenes = [s for s in scenes if s.get("id") != scene["id"]]
    scenes.append(scene)
    with open(_scenes_path(working_dir), "w") as f:
        json.dump(scenes, f, indent=2)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/reconstruct")
def reconstruct(req: ReconstructRequest):
    if not os.path.isdir(req.working_dir):
        raise HTTPException(400, f"Working dir not found: {req.working_dir}")

    gc.collect()
    torch.cuda.empty_cache()

    predictions = _run_inference(req.working_dir)

    semantic_masks = None
    if req.enable_semantic and req.semantic_masks_path:
        pred_h, pred_w = predictions["depth"].shape[1], predictions["depth"].shape[2]
        semantic_masks = _load_semantic_masks(req.semantic_masks_path, pred_h, pred_w)
        if semantic_masks is not None:
            predictions["semantic_masks"] = semantic_masks

    # Save predictions
    save_dict = {k: v for k, v in predictions.items() if v is not None}
    np.savez(os.path.join(req.working_dir, "predictions.npz"), **save_dict)

    glbfile = _glb_filename(req.working_dir, req.conf_thres, req.frame_filter,
                             req.mask_black_bg, req.mask_white_bg, req.show_cam,
                             req.mask_sky, req.prediction_mode, req.enable_semantic)

    glbscene = predictions_to_glb(
        predictions,
        conf_thres=req.conf_thres,
        filter_by_frames=req.frame_filter,
        mask_black_bg=req.mask_black_bg,
        mask_white_bg=req.mask_white_bg,
        show_cam=req.show_cam,
        mask_sky=req.mask_sky,
        target_dir=req.working_dir,
        prediction_mode=req.prediction_mode,
        semantic_masks=semantic_masks,
        enable_sam=req.enable_semantic,
    )
    glbscene.export(file_obj=glbfile)

    depth_paths = generate_depth_visualizations(predictions, req.working_dir)
    pointmap_paths = generate_pointmap_visualizations(predictions, req.working_dir)

    semantic_depth_paths, semantic_pointmap_paths, sam_mask_paths = [], [], []
    if req.enable_semantic and semantic_masks is not None:
        semantic_depth_paths = generate_semantic_depth_visualizations(predictions, req.working_dir, semantic_masks)
        semantic_pointmap_paths = generate_semantic_pointmap_visualizations(predictions, req.working_dir, semantic_masks)
        mask_dir = os.path.join(req.working_dir, "sam_masks")
        if os.path.isdir(mask_dir):
            sam_mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "status": "ok",
        "glb_path": glbfile,
        "depth_paths": depth_paths,
        "pointmap_paths": pointmap_paths,
        "semantic_depth_paths": semantic_depth_paths,
        "semantic_pointmap_paths": semantic_pointmap_paths,
        "sam_mask_paths": sam_mask_paths,
        "frame_filter_choices": _frame_filter_choices(req.working_dir),
        "log": f"Reconstruction complete ({len(_frame_filter_choices(req.working_dir)) - 1} frames)",
    }


@app.post("/visualize")
def visualize(req: VisualizeRequest):
    predictions_path = os.path.join(req.working_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        raise HTTPException(400, "No saved predictions. Run /reconstruct first.")

    key_list = ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf",
                "images", "extrinsic", "intrinsic", "world_points_from_depth"]
    loaded = np.load(predictions_path)
    predictions = {k: np.array(loaded[k]) for k in key_list if k in loaded.files}

    semantic_masks = None
    if req.enable_semantic and req.semantic_masks_path:
        pred_h, pred_w = predictions["depth"].shape[1], predictions["depth"].shape[2]
        semantic_masks = _load_semantic_masks(req.semantic_masks_path, pred_h, pred_w)

    glbfile = _glb_filename(req.working_dir, req.conf_thres, req.frame_filter,
                             req.mask_black_bg, req.mask_white_bg, req.show_cam,
                             req.mask_sky, req.prediction_mode, req.enable_semantic)

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=req.conf_thres,
            filter_by_frames=req.frame_filter,
            mask_black_bg=req.mask_black_bg,
            mask_white_bg=req.mask_white_bg,
            show_cam=req.show_cam,
            mask_sky=req.mask_sky,
            target_dir=req.working_dir,
            prediction_mode=req.prediction_mode,
            semantic_masks=semantic_masks,
            enable_sam=req.enable_semantic,
        )
        glbscene.export(file_obj=glbfile)

    return {
        "status": "ok",
        "glb_path": glbfile,
        "log": "Visualization updated",
    }


@app.post("/edit_pointcloud")
def edit_pointcloud(req: EditPointcloudRequest):
    predictions_path = os.path.join(req.working_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        raise HTTPException(400, "No saved predictions. Run /reconstruct first.")
    if not req.selected_semantic_ids:
        raise HTTPException(400, "No semantic IDs selected.")

    key_list = ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf",
                "images", "extrinsic", "intrinsic", "world_points_from_depth"]
    loaded = np.load(predictions_path)
    predictions = {k: np.array(loaded[k]) for k in key_list if k in loaded.files}

    pred_h, pred_w = predictions["depth"].shape[1], predictions["depth"].shape[2]
    semantic_masks = _load_semantic_masks(req.semantic_masks_path, pred_h, pred_w)
    if semantic_masks is None:
        raise HTTPException(400, f"Could not load semantic masks from: {req.semantic_masks_path}")

    ids_tag = "_".join(str(i) for i in sorted(req.selected_semantic_ids))
    glbfile = os.path.join(
        req.working_dir,
        f"edited_{req.operation}_ids{ids_tag}_{req.conf_thres}"
        f"_{req.frame_filter.replace(' ','_').replace(':','')}.glb"
    )

    glbscene = predictions_to_glb(
        predictions,
        conf_thres=req.conf_thres,
        filter_by_frames=req.frame_filter,
        mask_black_bg=req.mask_black_bg,
        mask_white_bg=req.mask_white_bg,
        show_cam=req.show_cam,
        mask_sky=req.mask_sky,
        target_dir=req.working_dir,
        prediction_mode=req.prediction_mode,
        semantic_masks=semantic_masks,
        enable_sam=True,
        semantic_filter_ids=req.selected_semantic_ids,
        semantic_filter_mode=req.operation,
    )
    glbscene.export(file_obj=glbfile)

    class_ids_str = ", ".join(str(i) for i in req.selected_semantic_ids)
    op_label = "deleted" if req.operation == "delete" else "extracted"

    # Register this edit so the elevation viewer can re-derive the same cloud
    sorted_ids = sorted(req.selected_semantic_ids)
    scene_id = f"{req.operation}_ids{ids_tag}"
    _register_scene(req.working_dir, {
        "id": scene_id,
        "label": f"{op_label.capitalize()}: IDs [{class_ids_str}]",
        "semantic_filter_ids": sorted_ids,
        "semantic_filter_mode": req.operation,
        "semantic_masks_path": req.semantic_masks_path,
        "glb_path": glbfile,
    })

    return {
        "status": "ok",
        "glb_path": glbfile,
        "scene_id": scene_id,
        "log": f"Point cloud edit complete — semantic IDs [{class_ids_str}] {op_label}.",
    }


@app.post("/fit_elevation")
def fit_elevation(req: FitElevationRequest):
    predictions_path = os.path.join(req.working_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        raise HTTPException(400, "No saved predictions. Run /reconstruct first.")

    key_list = ["depth", "depth_conf", "world_points", "world_points_conf",
                "extrinsic", "intrinsic", "world_points_from_depth", "semantic_masks"]
    loaded = np.load(predictions_path)
    predictions = {k: np.array(loaded[k]) for k in key_list if k in loaded.files}

    try:
        result = fit_elevation_to_glb(
            predictions=predictions,
            working_dir=req.working_dir,
            source_glb_path=req.source_glb_path,
            grid_resolution=req.grid_resolution,
            colormap=req.colormap,
            ground_percentile=req.ground_percentile,
            use_ransac=req.use_ransac,
            conf_thres=req.conf_thres,
            prediction_mode=req.prediction_mode,
            scale_factor=req.scale_factor,
            use_ground_filter=req.use_ground_filter,
        )
    except Exception as e:
        raise HTTPException(500, f"Elevation fitting failed: {e}")

    return {
        "status": "ok",
        "elev_only_path": result["elev_only_path"],
        "merged_path": result["merged_path"],
        "gravity_source": result["gravity_source"],
        "n_grav": result["n_grav"],
        "R_align": result["R_align"],
        "scale_factor": result["scale_factor"],
        "use_ground_filter": result["use_ground_filter"],
        "dem_source": result["dem_source"],
        "warnings": result["warnings"],
        "log": result["log"],
    }


@app.get("/elevation_scenes")
def elevation_scenes(session: str = ""):
    """List point-cloud scenes the elevation viewer can load: the original
    reconstruction plus every edit registered by /edit_pointcloud."""
    scenes = [{
        "id": "original",
        "label": "Original (full reconstruction)",
        "semantic_filter_ids": [],
        "semantic_filter_mode": "delete",
        "semantic_masks_path": "",
    }]
    if session:
        for s in _load_scenes(session):
            scenes.append({
                "id": s.get("id"),
                "label": s.get("label", s.get("id")),
                "semantic_filter_ids": s.get("semantic_filter_ids", []),
                "semantic_filter_mode": s.get("semantic_filter_mode", "delete"),
                "semantic_masks_path": s.get("semantic_masks_path", ""),
            })
    return {"scenes": scenes}


@app.post("/elevation_viewer_data")
def elevation_viewer_data(req: ElevationViewerRequest):
    """
    Return all data needed by the Three.js elevation viewer:
      - point cloud vertices + colors (in GLB display space)
      - per-point aligned XYZ (gravity-aligned, Y=up)
      - R_align, T_display, scale_factor
      - camera extrinsics: raw + aligned, camera centers
      - DEM grid for hole-fill and volume computation
    """
    predictions_path = os.path.join(req.working_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        raise HTTPException(400, "No saved predictions. Run /reconstruct first.")

    key_list = ["depth", "depth_conf", "world_points", "world_points_conf",
                "extrinsic", "intrinsic", "world_points_from_depth", "images", "semantic_masks"]
    loaded = np.load(predictions_path)
    predictions = {k: np.array(loaded[k]) for k in key_list if k in loaded.files}

    extrinsic = predictions["extrinsic"]   # (S, 3, 4)
    S = extrinsic.shape[0]

    # ── Gravity alignment ────────────────────────────────────────────────────
    if req.prediction_mode == "Pointmap Branch":
        raw_pts = predictions.get("world_points")
        raw_conf = predictions.get("world_points_conf")
    else:
        raw_pts = predictions.get("world_points_from_depth")
        raw_conf = predictions.get("depth_conf")

    sem = predictions.get("semantic_masks")
    gmask_3d = (np.asarray(sem) == 1) if sem is not None else None
    ground_flat = None
    if sem is not None:
        sem_arr = np.asarray(sem)
        if sem_arr.shape == raw_pts.shape[:3]:
            ground_flat = (sem_arr == 1).reshape(-1)

    grav = estimate_gravity(
        extrinsic=extrinsic,
        world_points=raw_pts,
        ground_mask=gmask_3d,
        conf=raw_conf,
        conf_thres=req.conf_thres / 100.0,
    )
    R_align = grav.R_align  # (3,3)

    # ── T_display: the transform baked into the GLB by predictions_to_glb ───
    # T_display = inv(ext[0]) @ opengl @ rot180y
    opengl = np.eye(4); opengl[1, 1] = -1; opengl[2, 2] = -1
    rot180y = np.eye(4)
    rot180y[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    ext0_4x4 = np.eye(4); ext0_4x4[:3, :4] = extrinsic[0]
    T_display = np.linalg.inv(ext0_4x4) @ opengl @ rot180y  # (4,4)

    # ── Point cloud in display space (same as GLB) ───────────────────────────
    pts_world = raw_pts.reshape(-1, 3)          # (N, 3)
    conf_flat = raw_conf.reshape(-1)
    conf_thres_val = np.percentile(conf_flat, req.conf_thres) if req.conf_thres > 0 else 0.0
    keep = (conf_flat >= conf_thres_val) & np.isfinite(pts_world).all(axis=1)

    # ── Optional semantic edit (mirrors the Point Cloud Editing tab) ─────────
    # Apply the same np.isin filter on raw points so the chosen edited scene
    # drives gravity alignment and DEM fitting consistently with the GLB export.
    if req.semantic_filter_ids:
        n_pts = pts_world.shape[0]
        sem_for_filter = None
        # Prefer an explicit masks file; fall back to masks baked into predictions.
        if req.semantic_masks_path:
            pred_h, pred_w = predictions["depth"].shape[1], predictions["depth"].shape[2]
            sem_for_filter = _load_semantic_masks(req.semantic_masks_path, pred_h, pred_w)
        if sem_for_filter is None and sem is not None:
            sem_for_filter = np.asarray(sem)
        if sem_for_filter is not None and sem_for_filter.size == n_pts:
            sem_flat = sem_for_filter.reshape(-1)
            class_mask = np.isin(sem_flat, req.semantic_filter_ids)
            if req.semantic_filter_mode == "delete":
                keep = keep & ~class_mask
            else:  # extract
                keep = keep & class_mask

    pts_world_kept = pts_world[keep]
    conf_kept = conf_flat[keep]
    ground_kept = ground_flat[keep] if ground_flat is not None else None

    # Apply T_display to get display-space coords (matches GLB)
    pts_h = np.concatenate([pts_world_kept, np.ones((pts_world_kept.shape[0], 1))], axis=1)
    pts_display = (T_display @ pts_h.T).T[:, :3]

    # Apply R_align to world points → gravity-aligned coords (Y=up)
    pts_aligned = pts_world_kept @ R_align.T

    # Colors from images
    images = predictions.get("images")  # (S, 3, H, W) or (S, H, W, 3)
    if images is not None:
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        colors_flat = (images.reshape(-1, 3) * 255).astype(np.uint8)
        colors_kept = colors_flat[keep]
    else:
        colors_kept = np.full((pts_display.shape[0], 3), 180, dtype=np.uint8)

    # Subsample for browser performance (max 300k points)
    MAX_PTS = 300_000
    if pts_display.shape[0] > MAX_PTS:
        idx = np.random.default_rng(42).choice(pts_display.shape[0], MAX_PTS, replace=False)
        pts_display = pts_display[idx]
        pts_aligned = pts_aligned[idx]
        colors_kept = colors_kept[idx]
        if ground_kept is not None:
            ground_kept = ground_kept[idx]

    # ── Camera extrinsics ────────────────────────────────────────────────────
    # Raw camera centers in world space
    R_cams = extrinsic[:, :3, :3]
    t_cams = extrinsic[:, :3, 3]
    centers_world = -np.einsum("sij,sj->si", R_cams.transpose(0, 2, 1), t_cams)

    # Aligned camera centers
    centers_aligned = centers_world @ R_align.T

    # Camera centers in display space
    centers_h = np.concatenate([centers_world, np.ones((S, 1))], axis=1)
    centers_display = (T_display @ centers_h.T).T[:, :3]

    # Aligned extrinsics: R' = R @ R_align^T, t' = t
    ext_aligned = apply_alignment_to_extrinsics(extrinsic, R_align)

    cameras = []
    for i in range(S):
        cameras.append({
            "frame": i,
            "extrinsic_raw": extrinsic[i].tolist(),
            "extrinsic_aligned": ext_aligned[i].tolist(),
            "center_world": centers_world[i].tolist(),
            "center_aligned": centers_aligned[i].tolist(),
            "center_display": centers_display[i].tolist(),
        })

    # ── DEM grid (for hole-fill and volume) ──────────────────────────────────
    # Use gravity-aligned points; Y is elevation, (X,Z) is horizontal plane
    ground_filter_mask = _select_ground_aligned_mask(
        pts_aligned, ground_kept, req.ground_percentile
    )
    dem_source_mask = ground_filter_mask if req.use_ground_filter else np.ones(
        pts_aligned.shape[0], dtype=bool
    )
    ground_pts = pts_aligned[dem_source_mask]

    GRID_RES = 128
    x_min, x_max = float(pts_aligned[:, 0].min()), float(pts_aligned[:, 0].max())
    z_min, z_max = float(pts_aligned[:, 2].min()), float(pts_aligned[:, 2].max())
    x_pad = (x_max - x_min) * 0.02; z_pad = (z_max - z_min) * 0.02
    x_min -= x_pad; x_max += x_pad; z_min -= z_pad; z_max += z_pad

    xi = np.linspace(x_min, x_max, GRID_RES)
    zi = np.linspace(z_min, z_max, GRID_RES)
    xx, zz = np.meshgrid(xi, zi)

    from scipy.interpolate import griddata
    elev_linear = griddata(ground_pts[:, [0, 2]], ground_pts[:, 1],
                           (xx, zz), method="linear")
    elev_nearest = griddata(ground_pts[:, [0, 2]], ground_pts[:, 1],
                            (xx, zz), method="nearest")
    elev = np.where(np.isnan(elev_linear), elev_nearest, elev_linear)
    has_data = (~np.isnan(elev_linear)).astype(np.uint8)

    # ── Serialize ────────────────────────────────────────────────────────────
    return {
        "status": "ok",
        "working_dir": req.working_dir,
        "gravity_source": grav.source,
        "gravity_warnings": grav.warnings,
        "n_grav": grav.n_grav.tolist(),
        "R_align": R_align.tolist(),
        "T_display": T_display.tolist(),
        "scale_factor": 1.0,
        "use_ground_filter": bool(req.use_ground_filter),
        "dem_source": "ground-filtered" if req.use_ground_filter else "unfiltered",
        "n_points": int(pts_display.shape[0]),
        "points_display": pts_display.astype(np.float32).tolist(),
        "points_aligned": pts_aligned.astype(np.float32).tolist(),
        "ground_filter": {
            "function": "_select_ground_aligned",
            "enabled_for_dem": bool(req.use_ground_filter),
            "ground_percentile": float(req.ground_percentile),
            "used_semantic_ground_mask": bool(ground_kept is not None and ground_kept.any()),
            "kept_count": int(ground_filter_mask.sum()),
            "filtered_count": int(ground_filter_mask.shape[0] - ground_filter_mask.sum()),
            "dem_source_count": int(dem_source_mask.sum()),
            "keep_mask": ground_filter_mask.astype(np.uint8).tolist(),
        },
        "colors": colors_kept.tolist(),
        "cameras": cameras,
        "dem": {
            "grid_res": GRID_RES,
            "x_min": x_min, "x_max": x_max,
            "z_min": z_min, "z_max": z_max,
            "elev": elev.astype(np.float32).tolist(),
            "has_data": has_data.tolist(),
        },
    }


@app.get("/viewer")
def open_viewer(session: str = ""):
    url = f"/static/elevation_viewer.html?session={session}"
    return RedirectResponse(url=url)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
