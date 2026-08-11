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
import time
import threading
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

# This repo's own directory — derived from the file location so the project is
# portable (works regardless of where it is cloned).
VGGT_YOLOE_DIR = os.path.dirname(os.path.abspath(__file__))

# External VGGT repo. Defaults to a sibling ``vggt`` checkout next to this repo's
# parent; override with the VGGT_DIR environment variable if it lives elsewhere.
VGGT_DIR = os.environ.get(
    "VGGT_DIR",
    os.path.join(os.path.dirname(VGGT_YOLOE_DIR), "vggt"),
)
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

sys.path.insert(0, VGGT_YOLOE_DIR)
from elevation_plane import _select_ground_aligned_mask
import terrain_analysis
import elevation_export
from plane_calibration import PlaneCalibrationRequest, calibrate_local_plane
from gravity_alignment import estimate_gravity, apply_alignment_to_points, apply_alignment_to_extrinsics
from semantic_fusion import refine_masks_3d
from resource_profiler import ResourceProfiler, stage as profile_stage

app = FastAPI(title="VGGT Reconstruction Service")
app.mount("/static", StaticFiles(directory=VGGT_YOLOE_DIR), name="static")

# Real-time streaming link (M3): /stream/start|stop|status. Additive and isolated in the
# streaming/ package; wrapped so a streaming import error can never block service startup.
try:
    from streaming.endpoints import router as _stream_router
    app.include_router(_stream_router)
except Exception as _e:  # pragma: no cover - defensive: never break the offline service
    print(f"[stream] streaming endpoints not mounted: {_e}")

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
    # 3D label fusion: after reconstruction, reconcile per-frame masks in the
    # shared cloud so targets only segmented on some (key)frames are propagated
    # to every point, and isolated mis-detections are dropped. The fused mask
    # becomes the canonical mask baked into predictions.npz.
    fuse_3d: bool = False
    fuse_eps: float = 0.05            # clustering radius (world units)
    fuse_min_cluster: int = 30       # drop clusters smaller than this (noise)
    fuse_dilate_radius: float = 0.0  # relabel radius; 0 → fall back to fuse_eps
    fuse_seed_conf: float = 10.0     # seed conf percentile (decoupled from global conf_thres)


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
    # Object (excavator) overlay: render these semantic classes as a separate
    # RGB-colored mini-DEM on the same grid, and exclude them from the terrain
    # DEM. Empty list = auto-resolve excavator/vehicle ids from the run meta.
    object_semantic_ids: List[int] = []


class LocalPlaneCalibrationApiRequest(BaseModel):
    """Known-object calibration for the elevation viewer's anisotropic mode."""
    working_dir: str
    object_semantic_id: int
    object_length_m: float
    object_width_m: float
    object_height_m: float
    ground_semantic_id: int = 1
    confidence_percentile: float = 50.0
    semantic_masks_path: str = ""
    prediction_mode: str = "Depthmap and Camera Branch"


class TerrainAnalysisRequest(BaseModel):
    """Semantic×geometry terrain understanding on the gravity-aligned cloud.
    Shares the point-selection fields with ElevationViewerRequest so an edited
    scene drives the analysis consistently with the viewer."""
    working_dir: str
    conf_thres: float = 50.0
    prediction_mode: str = "Depthmap and Camera Branch"
    ground_percentile: float = 20.0
    semantic_filter_ids: List[int] = []
    semantic_filter_mode: str = "delete"
    semantic_masks_path: str = ""
    # Terrain-specific knobs
    grid_res: int = 128
    top_percentile: float = 90.0       # per-cell surface-top height percentile
    tau: Optional[float] = None        # absolute residual threshold; None = adaptive
    tau_frac: float = 0.1              # adaptive tau = tau_frac · (Y p2–p98 range)
    min_area_frac: float = 0.002       # drop connected components smaller than this
    sem_id_map: Optional[dict] = None  # class name → id; else read from run meta.json
    material_rules: Optional[dict] = None  # override DEFAULT_MATERIAL_RULES
    scale_factor: float = 1.0          # aligned 1 unit = N meters (for figure height labels)
    figure_mode: str = "worksite"      # "worksite" | "bev" | "diagnostic"


class ExportElevationRequest(BaseModel):
    """Export a DEM to the Unity terrain JSON format (TERRAIN_ELEVATION_FORMAT.md).
    Shares point-selection fields with the elevation viewer / terrain analysis so
    an edited scene exports consistently."""
    working_dir: str
    conf_thres: float = 50.0
    prediction_mode: str = "Depthmap and Camera Branch"
    ground_percentile: float = 20.0
    use_ground_filter: bool = True
    semantic_filter_ids: List[int] = []
    semantic_filter_mode: str = "delete"
    semantic_masks_path: str = ""
    # Export-specific knobs
    dem_source: str = "elev"           # "elev" (interpolated ground) | "htop" (surface top)
    grid_res: int = 128                # DEM grid size (NxN)
    scale_factor: float = 1.0          # backwards-compatible isotropic scale
    horizontal_scale: Optional[float] = None
    vertical_scale: Optional[float] = None
    height_resolution: float = 0.01    # int16 quantisation step (m); real height = data[i]*this
    tile_x: int = 0
    tile_y: int = 0
    tile_size_meters: Optional[float] = None  # default: X span * scale_factor
    # top_percentile only used when dem_source == "htop"
    top_percentile: float = 90.0
    out_path: str = ""                 # default: working_dir/elevation_tile_<x>_<y>.json
    # Attach a co-registered semantic (work-zone) layer. Only supported for
    # dem_source == "htop" (its grid matches terrain_analysis' zone_map).
    include_semantic: bool = False
    material_rules: Optional[dict] = None  # override DEFAULT_MATERIAL_RULES for zone classification


# ── Helpers ───────────────────────────────────────────────────────────────────

_INFERENCE_LOCK = threading.Lock()


def _run_inference(working_dir: str, profiler: Optional[ResourceProfiler] = None) -> dict:
    """Serialize access to the resident model across offline and streaming requests."""
    with _INFERENCE_LOCK:
        return _run_inference_unlocked(working_dir, profiler=profiler)


def _run_inference_unlocked(
        working_dir: str, profiler: Optional[ResourceProfiler] = None) -> dict:
    image_names = sorted(glob.glob(os.path.join(working_dir, "images", "*")))
    if not image_names:
        raise HTTPException(400, "No images found in working_dir/images/")

    with profile_stage(profiler, "load_and_preprocess_images", frames=len(image_names)):
        images = load_and_preprocess_images(image_names).to(DEVICE)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with profile_stage(
            profiler, "vggt_model_forward", frames=len(image_names), dtype=str(dtype)):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = _model(images)

    with profile_stage(profiler, "pose_decode_depth_unprojection"):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        predictions["pose_enc_list"] = None
        # `images` is already (S,3,H,W), unlike model outputs which carry a leading
        # batch dimension. Add it after the squeeze loop so multi-frame passes work.
        predictions["images"] = images.detach().cpu().numpy()

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


def _load_aligned_points_and_semantics(
        working_dir, conf_thres, prediction_mode, ground_percentile,
        semantic_filter_ids, semantic_filter_mode, semantic_masks_path):
    """Shared point-loading + gravity-alignment + keep-mask logic.

    Returns everything both the elevation viewer and terrain analysis need,
    BEFORE any browser subsampling. Points are returned in three frames:
      * pts_world_kept  — raw VGGT world frame (for GLB/display transforms)
      * pts_aligned     — gravity-aligned frame, Y=up (for DEM / terrain)
    plus per-point confidence, ground membership, and semantic-id labels, all
    masked by the same `keep` filter (conf gate + optional semantic edit).
    """
    predictions_path = os.path.join(working_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        raise HTTPException(400, "No saved predictions. Run /reconstruct first.")

    key_list = ["depth", "depth_conf", "world_points", "world_points_conf",
                "extrinsic", "intrinsic", "world_points_from_depth",
                "images", "semantic_masks"]
    loaded = np.load(predictions_path)
    predictions = {k: np.array(loaded[k]) for k in key_list if k in loaded.files}

    extrinsic = predictions["extrinsic"]

    if prediction_mode == "Pointmap Branch":
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
        extrinsic=extrinsic, world_points=raw_pts, ground_mask=gmask_3d,
        conf=raw_conf, conf_thres=conf_thres / 100.0,
    )
    R_align = grav.R_align

    pts_world = raw_pts.reshape(-1, 3)
    conf_flat = raw_conf.reshape(-1)
    conf_thres_val = np.percentile(conf_flat, conf_thres) if conf_thres > 0 else 0.0
    keep = (conf_flat >= conf_thres_val) & np.isfinite(pts_world).all(axis=1)

    # Per-point semantic labels (for terrain S_mode / material). Prefer an
    # explicit masks file, else masks baked into predictions.
    n_pts = pts_world.shape[0]
    sem_for_filter = None
    if semantic_masks_path:
        pred_h, pred_w = predictions["depth"].shape[1], predictions["depth"].shape[2]
        sem_for_filter = _load_semantic_masks(semantic_masks_path, pred_h, pred_w)
    if sem_for_filter is None and sem is not None:
        sem_for_filter = np.asarray(sem)
    sem_flat = sem_for_filter.reshape(-1) if (
        sem_for_filter is not None and sem_for_filter.size == n_pts) else None

    # Optional semantic edit (mirrors the Point Cloud Editing tab).
    if semantic_filter_ids and sem_flat is not None:
        class_mask = np.isin(sem_flat, semantic_filter_ids)
        if semantic_filter_mode == "delete":
            keep = keep & ~class_mask
        else:  # extract
            keep = keep & class_mask

    pts_world_kept = pts_world[keep]
    pts_aligned = pts_world_kept @ R_align.T
    conf_kept = conf_flat[keep]
    ground_kept = ground_flat[keep] if ground_flat is not None else None
    sem_kept = sem_flat[keep] if sem_flat is not None else None

    return {
        "predictions": predictions,
        "extrinsic": extrinsic,
        "grav": grav,
        "R_align": R_align,
        "keep": keep,
        "pts_world_kept": pts_world_kept,
        "pts_aligned": pts_aligned,
        "conf_kept": conf_kept,
        "ground_kept": ground_kept,
        "sem_kept": sem_kept,
    }


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

    t0 = time.perf_counter()
    profiler = ResourceProfiler(
        "vggt_reconstruct", req.working_dir, torch_module=torch,
        metadata={
            "prediction_mode": req.prediction_mode,
            "confidence_percentile": req.conf_thres,
            "semantic_enabled": req.enable_semantic,
            "fuse_3d": req.fuse_3d,
        },
    )
    with profiler.stage("pre_run_cleanup"):
        gc.collect()
        torch.cuda.empty_cache()

    predictions = _run_inference(req.working_dir, profiler=profiler)

    semantic_masks = None
    fused_masks_path = ""
    fuse_log = ""
    with profiler.stage("semantic_mask_load_and_fusion"):
        if req.enable_semantic and req.semantic_masks_path:
            pred_h, pred_w = predictions["depth"].shape[1], predictions["depth"].shape[2]
            semantic_masks = _load_semantic_masks(req.semantic_masks_path, pred_h, pred_w)
            if semantic_masks is not None:
                # With keyframe-only segmentation, propagate labels through the
                # shared 3D cloud so every target point can receive a label.
                if req.fuse_3d:
                    if req.prediction_mode == "Pointmap Branch" and "world_points" in predictions:
                        wp = predictions["world_points"]
                        wp_conf = predictions.get("world_points_conf")
                    else:
                        wp = predictions["world_points_from_depth"]
                        wp_conf = predictions.get("depth_conf")
                    if wp.shape[:3] == semantic_masks.shape:
                        n_before = int((semantic_masks != 0).sum())
                        semantic_masks = refine_masks_3d(
                            world_points=wp, semantic_masks=semantic_masks, conf=wp_conf,
                            conf_thres=req.conf_thres, seed_conf_thres=req.fuse_seed_conf,
                            selected_ids=None, eps=req.fuse_eps,
                            min_cluster_size=req.fuse_min_cluster,
                            dilate_radius=(req.fuse_dilate_radius or None),
                        )
                        n_after = int((semantic_masks != 0).sum())
                        fused_masks_path = os.path.join(
                            req.working_dir, "semantic_masks_fused.npz")
                        np.savez(fused_masks_path, semantic_masks=semantic_masks)
                        fuse_log = (
                            f" | 3D fusion: {n_before}→{n_after} labeled points "
                            f"(eps={req.fuse_eps})"
                        )
                predictions["semantic_masks"] = semantic_masks

    # Drop any stale fused mask from a previous run so the edit tab never picks
    # up an out-of-date file when this reconstruction did not fuse.
    with profiler.stage("save_predictions_npz"):
        if not fused_masks_path:
            stale = os.path.join(req.working_dir, "semantic_masks_fused.npz")
            if os.path.exists(stale):
                os.remove(stale)
        save_dict = {k: v for k, v in predictions.items() if v is not None}
        np.savez(os.path.join(req.working_dir, "predictions.npz"), **save_dict)

    glbfile = _glb_filename(req.working_dir, req.conf_thres, req.frame_filter,
                             req.mask_black_bg, req.mask_white_bg, req.show_cam,
                             req.mask_sky, req.prediction_mode, req.enable_semantic)

    with profiler.stage("build_and_export_glb"):
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

    with profiler.stage("render_depth_visualizations"):
        depth_paths = generate_depth_visualizations(predictions, req.working_dir)
    with profiler.stage("render_pointmap_visualizations"):
        pointmap_paths = generate_pointmap_visualizations(predictions, req.working_dir)

    semantic_depth_paths, semantic_pointmap_paths, sam_mask_paths = [], [], []
    with profiler.stage("render_semantic_visualizations"):
        if req.enable_semantic and semantic_masks is not None:
            semantic_depth_paths = generate_semantic_depth_visualizations(
                predictions, req.working_dir, semantic_masks)
            semantic_pointmap_paths = generate_semantic_pointmap_visualizations(
                predictions, req.working_dir, semantic_masks)
            mask_dir = os.path.join(req.working_dir, "sam_masks")
            if os.path.isdir(mask_dir):
                sam_mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    with profiler.stage("post_run_cleanup"):
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

    n_frames = len(_frame_filter_choices(req.working_dir)) - 1
    elapsed = time.perf_counter() - t0
    profile_path = profiler.finish(metadata={"frames": n_frames})
    print(f"[timing] reconstruct: {n_frames} frames in {elapsed:.2f}s", flush=True)

    return {
        "status": "ok",
        "glb_path": glbfile,
        "depth_paths": depth_paths,
        "pointmap_paths": pointmap_paths,
        "semantic_depth_paths": semantic_depth_paths,
        "semantic_pointmap_paths": semantic_pointmap_paths,
        "sam_mask_paths": sam_mask_paths,
        "fused_masks_path": fused_masks_path,
        "resource_profile_path": profile_path,
        "frame_filter_choices": _frame_filter_choices(req.working_dir),
        "log": f"Reconstruction complete ({n_frames} frames) in {elapsed:.1f}s{fuse_log}",
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
    # Prefer the 3D-fused mask produced at reconstruction time (consistent across
    # all frames, including those skipped by keyframe selection); fall back to the
    # raw per-frame mask from the YOLOe run.
    fused_path = os.path.join(req.working_dir, "semantic_masks_fused.npz")
    masks_src = fused_path if os.path.exists(fused_path) else req.semantic_masks_path
    semantic_masks = _load_semantic_masks(masks_src, pred_h, pred_w)
    if semantic_masks is None:
        raise HTTPException(400, f"Could not load semantic masks from: {masks_src}")

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
        "semantic_masks_path": masks_src,
        "glb_path": glbfile,
    })

    return {
        "status": "ok",
        "glb_path": glbfile,
        "scene_id": scene_id,
        "log": f"Point cloud edit complete — semantic IDs [{class_ids_str}] {op_label}.",
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


@app.post("/calibrate_local_plane")
def calibrate_local_plane_endpoint(req: LocalPlaneCalibrationApiRequest):
    """Estimate horizontal and vertical metric scales from a known reference object."""
    predictions_path = os.path.join(req.working_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        raise HTTPException(400, "No saved predictions. Run /reconstruct first.")
    loaded = np.load(predictions_path)
    if "semantic_masks" not in loaded.files:
        raise HTTPException(400, "Local-plane calibration requires semantic_masks in predictions.npz.")
    if req.prediction_mode == "Pointmap Branch":
        point_key, conf_key = "world_points", "world_points_conf"
    else:
        point_key, conf_key = "world_points_from_depth", "depth_conf"
    if point_key not in loaded.files or conf_key not in loaded.files:
        raise HTTPException(400, f"Missing {point_key} or {conf_key} in predictions.npz.")
    points = np.asarray(loaded[point_key])
    confidence = np.asarray(loaded[conf_key])
    semantic_masks = np.asarray(loaded["semantic_masks"])
    if req.semantic_masks_path:
        loaded_masks = _load_semantic_masks(req.semantic_masks_path, points.shape[1], points.shape[2])
        if loaded_masks is None:
            raise HTTPException(400, "Could not load calibration semantic_masks_path.")
        semantic_masks = np.asarray(loaded_masks)
    extrinsic = np.asarray(loaded["extrinsic"])
    if semantic_masks.shape != points.shape[:3]:
        raise HTTPException(400, "semantic_masks shape does not match reconstructed point grid.")
    ground_mask = semantic_masks == req.ground_semantic_id
    gravity = estimate_gravity(
        extrinsic=extrinsic, world_points=points, ground_mask=ground_mask, conf=confidence,
        conf_thres=req.confidence_percentile / 100.0,
    )
    try:
        result = calibrate_local_plane(
            points, confidence, semantic_masks, gravity.n_grav,
            PlaneCalibrationRequest(
                object_semantic_id=req.object_semantic_id,
                object_length_m=req.object_length_m, object_width_m=req.object_width_m,
                object_height_m=req.object_height_m, ground_semantic_id=req.ground_semantic_id,
                confidence_percentile=req.confidence_percentile,
            ),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result["status"] = "ok"
    result["gravity_source"] = gravity.source
    result["gravity_warnings"] = gravity.warnings
    return result


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
    t0 = time.perf_counter()
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

    # ── Object (excavator) semantic ids + per-point labels ───────────────────
    # Build a full-cloud semantic id array (masks file preferred, else baked)
    # so the excavator can be rasterized separately from the terrain DEM.
    n_pts = pts_world.shape[0]
    sem_all = None
    if req.semantic_masks_path:
        pred_h, pred_w = predictions["depth"].shape[1], predictions["depth"].shape[2]
        sem_all = _load_semantic_masks(req.semantic_masks_path, pred_h, pred_w)
    if sem_all is None and sem is not None:
        sem_all = np.asarray(sem)
    sem_all_flat = sem_all.reshape(-1) if (sem_all is not None and sem_all.size == n_pts) else None

    object_ids = list(req.object_semantic_ids)
    object_names = []
    if not object_ids:
        sem_id_map = _resolve_sem_id_map(req.working_dir, None)
        for name, cid in sem_id_map.items():
            low = str(name).lower()
            if any(k in low for k in ("excavator", "digger", "truck", "vehicle", "machine")):
                object_ids.append(int(cid))
                object_names.append(name)
    else:
        inv = {int(v): k for k, v in _resolve_sem_id_map(req.working_dir, None).items()}
        object_names = [inv.get(i, str(i)) for i in object_ids]

    pts_world_kept = pts_world[keep]
    conf_kept = conf_flat[keep]
    ground_kept = ground_flat[keep] if ground_flat is not None else None
    sem_kept = sem_all_flat[keep] if sem_all_flat is not None else None

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
        if sem_kept is not None:
            sem_kept = sem_kept[idx]

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
    # Never let object (excavator) points leak into the terrain surface.
    object_mask = (np.isin(sem_kept, object_ids)
                   if (sem_kept is not None and object_ids)
                   else np.zeros(pts_aligned.shape[0], dtype=bool))
    dem_source_mask = dem_source_mask & ~object_mask
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

    # ── Object (excavator) mini-DEM on the SAME grid ─────────────────────────
    # Rasterize object points into cells: per-cell top-percentile height + mean
    # RGB. Only cells that actually contain object points get has_data=1.
    object_dem = {"present": False, "semantic_ids": [int(i) for i in object_ids],
                  "names": object_names}
    if object_mask.any():
        obj_pts = pts_aligned[object_mask]
        obj_col = colors_kept[object_mask].astype(np.float32)
        # Cell index per point (clamped to grid bounds).
        cj = np.clip(((obj_pts[:, 0] - x_min) / (x_max - x_min) * (GRID_RES - 1)
                      ).round().astype(int), 0, GRID_RES - 1)
        ci = np.clip(((obj_pts[:, 2] - z_min) / (z_max - z_min) * (GRID_RES - 1)
                      ).round().astype(int), 0, GRID_RES - 1)
        obj_elev = np.full((GRID_RES, GRID_RES), np.nan, dtype=np.float32)
        obj_hd = np.zeros((GRID_RES, GRID_RES), dtype=np.uint8)
        obj_rgb = np.zeros((GRID_RES, GRID_RES, 3), dtype=np.uint8)
        lin = ci * GRID_RES + cj
        for cell in np.unique(lin):
            m = lin == cell
            ii, jj = divmod(int(cell), GRID_RES)
            obj_elev[ii, jj] = float(np.percentile(obj_pts[m, 1], 90))
            obj_rgb[ii, jj] = obj_col[m].mean(axis=0).round().astype(np.uint8)
            obj_hd[ii, jj] = 1
        # Fill empty cells so the mesh interpolation isn't NaN (unused where hd=0).
        obj_elev = np.where(np.isnan(obj_elev), float(np.nanmin(obj_elev)), obj_elev)
        object_dem.update({
            "present": True,
            "elev": obj_elev.astype(np.float32).tolist(),
            "has_data": obj_hd.tolist(),
            "rgb": obj_rgb.tolist(),
            "point_count": int(object_mask.sum()),
        })

    # ── Serialize ────────────────────────────────────────────────────────────
    print(f"[timing] elevation_viewer_data: {S} frames in "
          f"{time.perf_counter() - t0:.2f}s", flush=True)
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
        "object_dem": object_dem,
    }


def _resolve_sem_id_map(working_dir: str, override: Optional[dict]) -> dict:
    """class name → id map for material rules. Prefer an explicit override,
    else the most-recent YOLOe run meta.json under working_dir/yoloe_runs/."""
    if override:
        return override
    runs_dir = os.path.join(working_dir, "yoloe_runs")
    if not os.path.isdir(runs_dir):
        return {}
    metas = sorted(glob.glob(os.path.join(runs_dir, "*", "meta.json")))
    for meta_path in reversed(metas):  # newest first
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("semantic_id_map"):
                return meta["semantic_id_map"]
        except (json.JSONDecodeError, OSError):
            continue
    return {}


@app.post("/terrain_analysis")
def terrain_analysis_endpoint(req: TerrainAnalysisRequest):
    """Semantic×geometry terrain understanding: rasterize the gravity-aligned
    cloud to a BEV grid, extract mounds/pits/slope/roughness/volume, and confirm
    each region's material via keyword rules. Returns BEV layers + region list."""
    t0 = time.perf_counter()
    profiler = ResourceProfiler(
        "terrain_analysis", req.working_dir, torch_module=torch,
        metadata={"grid_res": req.grid_res, "prediction_mode": req.prediction_mode},
    )
    with profiler.stage("load_filter_and_gravity_align_points"):
        data = _load_aligned_points_and_semantics(
            working_dir=req.working_dir,
            conf_thres=req.conf_thres,
            prediction_mode=req.prediction_mode,
            ground_percentile=req.ground_percentile,
            semantic_filter_ids=req.semantic_filter_ids,
            semantic_filter_mode=req.semantic_filter_mode,
            semantic_masks_path=req.semantic_masks_path,
        )
    if data["pts_aligned"].shape[0] < 100:
        raise HTTPException(400, f"Too few valid points ({data['pts_aligned'].shape[0]}).")

    # class name → id (from override or run meta); invert to id → name for lookup.
    with profiler.stage("resolve_semantic_map"):
        sem_id_map = _resolve_sem_id_map(req.working_dir, req.sem_id_map)
        id_to_name = {int(v): k for k, v in sem_id_map.items()}

    try:
        with profiler.stage("rasterize_and_analyze_terrain"):
            result = terrain_analysis.analyze_terrain(
                pts_aligned=data["pts_aligned"],
                sem_labels=data["sem_kept"],
                ground_mask=data["ground_kept"],
                id_to_name=id_to_name,
                grid_res=req.grid_res,
                top_percentile=req.top_percentile,
                tau=req.tau,
                tau_frac=req.tau_frac,
                min_area_frac=req.min_area_frac,
                material_rules=req.material_rules,
            )
    except Exception as e:
        raise HTTPException(500, f"Terrain analysis failed: {e}")

    grav = data["grav"]
    result["gravity_source"] = grav.source
    result["gravity_warnings"] = grav.warnings
    result["sem_id_map"] = sem_id_map
    result["n_points"] = int(data["pts_aligned"].shape[0])
    result["resource_profile_path"] = profiler.finish(metadata={
        "n_points": result["n_points"], "regions": len(result["regions"]),
    })
    print(f"[timing] terrain_analysis: {req.grid_res}² grid, "
          f"{result['n_points']} pts, {len(result['regions'])} regions in "
          f"{time.perf_counter() - t0:.2f}s", flush=True)
    return result


@app.post("/terrain_analysis_figure")
def terrain_analysis_figure(req: TerrainAnalysisRequest):
    """Same analysis as /terrain_analysis, but render a top-down PNG figure
    (H_top | residual+regions | semantic | slope) and return it as an image.
    Browser-free way to inspect results (also saved under working_dir)."""
    from fastapi.responses import FileResponse
    data = _load_aligned_points_and_semantics(
        working_dir=req.working_dir, conf_thres=req.conf_thres,
        prediction_mode=req.prediction_mode, ground_percentile=req.ground_percentile,
        semantic_filter_ids=req.semantic_filter_ids,
        semantic_filter_mode=req.semantic_filter_mode,
        semantic_masks_path=req.semantic_masks_path,
    )
    if data["pts_aligned"].shape[0] < 100:
        raise HTTPException(400, f"Too few valid points ({data['pts_aligned'].shape[0]}).")

    sem_id_map = _resolve_sem_id_map(req.working_dir, req.sem_id_map)
    id_to_name = {int(v): k for k, v in sem_id_map.items()}
    try:
        result = terrain_analysis.analyze_terrain(
            pts_aligned=data["pts_aligned"], sem_labels=data["sem_kept"],
            ground_mask=data["ground_kept"], id_to_name=id_to_name,
            grid_res=req.grid_res, top_percentile=req.top_percentile,
            tau=req.tau, tau_frac=req.tau_frac, min_area_frac=req.min_area_frac,
            material_rules=req.material_rules,
        )
        result["sem_id_map"] = sem_id_map
        sess = os.path.basename(req.working_dir.rstrip('/'))
        if req.figure_mode == "diagnostic":
            out_path = os.path.join(req.working_dir, "terrain_analysis.png")
            terrain_analysis.render_analysis_figure(
                result, out_path, title=f"Terrain Analysis · {sess}",
                scale_factor=req.scale_factor)
            fname = "terrain_analysis.png"
        elif req.figure_mode == "bev":
            out_path = os.path.join(req.working_dir, "worksite_bev.png")
            terrain_analysis.render_worksite_bev(
                result, out_path, title=f"Worksite BEV · {sess}",
                scale_factor=req.scale_factor)
            fname = "worksite_bev.png"
        else:  # worksite fused map (default)
            out_path = os.path.join(req.working_dir, "worksite_map.png")
            terrain_analysis.render_worksite_map(
                result, out_path, title=f"Worksite Map · {sess}",
                scale_factor=req.scale_factor)
            fname = "worksite_map.png"
    except Exception as e:
        raise HTTPException(500, f"Terrain figure failed: {e}")
    return FileResponse(out_path, media_type="image/png", filename=fname)


@app.post("/export_elevation_json")
def export_elevation_json(req: ExportElevationRequest):
    """Export the DEM to the Unity terrain JSON format (TERRAIN_ELEVATION_FORMAT.md).

    Selects the elevation grid via `dem_source`:
      * "elev" — interpolated ground surface (same grid the elevation viewer uses)
      * "htop" — per-cell surface-top height from terrain_analysis.rasterize_bev
    Quantises to int16 (real height m = data[i] * height_resolution), writes one
    tile file, and returns its path plus a summary.
    """
    t0 = time.perf_counter()
    profiler = ResourceProfiler(
        "export_elevation_json", req.working_dir, torch_module=torch,
        metadata={"grid_res": req.grid_res, "dem_source": req.dem_source},
    )
    with profiler.stage("load_filter_and_gravity_align_points"):
        data = _load_aligned_points_and_semantics(
            working_dir=req.working_dir, conf_thres=req.conf_thres,
            prediction_mode=req.prediction_mode, ground_percentile=req.ground_percentile,
            semantic_filter_ids=req.semantic_filter_ids,
            semantic_filter_mode=req.semantic_filter_mode,
            semantic_masks_path=req.semantic_masks_path,
        )
    pts_aligned = data["pts_aligned"]
    if pts_aligned.shape[0] < 100:
        raise HTTPException(400, f"Too few valid points ({pts_aligned.shape[0]}).")

    zone_map = None          # optional semantic layer, filled for htop below
    legend = None

    with profiler.stage("build_elevation_grid"):
        if req.dem_source == "htop":
            rast = terrain_analysis.rasterize_bev(
                pts_aligned, sem_labels=data["sem_kept"], ground_mask=data["ground_kept"],
                grid_res=req.grid_res, top_percentile=req.top_percentile,
            )
            elev = rast["H_top"]
            has_data = np.isfinite(elev)
            x_min, x_max, z_min, z_max = rast["bounds"]

            if req.include_semantic:
                sem_id_map = _resolve_sem_id_map(req.working_dir, None)
                id_to_name = {int(v): k for k, v in sem_id_map.items()}
                rules = req.material_rules or terrain_analysis.DEFAULT_MATERIAL_RULES
                geom = terrain_analysis.extract_geometry(rast)
                regions = terrain_analysis.confirm_semantics(
                    geom, rast["S_mode"], id_to_name, rules)
                worksite = terrain_analysis.build_worksite_map(
                    rast, geom, regions, id_to_name, rules)
                zone_map = worksite["zone_map"]
                legend = terrain_analysis.zone_legend()
        elif req.dem_source == "elev":
            ground_mask = _select_ground_aligned_mask(
                pts_aligned, data["ground_kept"], req.ground_percentile)
            src_mask = ground_mask if req.use_ground_filter else np.ones(
                pts_aligned.shape[0], dtype=bool)
            ground_pts = pts_aligned[src_mask]
            x_min, x_max = float(pts_aligned[:, 0].min()), float(pts_aligned[:, 0].max())
            z_min, z_max = float(pts_aligned[:, 2].min()), float(pts_aligned[:, 2].max())
            x_pad = (x_max - x_min) * 0.02; z_pad = (z_max - z_min) * 0.02
            x_min -= x_pad; x_max += x_pad; z_min -= z_pad; z_max += z_pad
            xi = np.linspace(x_min, x_max, req.grid_res)
            zi = np.linspace(z_min, z_max, req.grid_res)
            xx, zz = np.meshgrid(xi, zi)
            from scipy.interpolate import griddata
            elev_linear = griddata(
                ground_pts[:, [0, 2]], ground_pts[:, 1], (xx, zz), method="linear")
            elev_nearest = griddata(
                ground_pts[:, [0, 2]], ground_pts[:, 1], (xx, zz), method="nearest")
            elev = np.where(np.isnan(elev_linear), elev_nearest, elev_linear)
            has_data = ~np.isnan(elev_linear)
            if req.include_semantic:
                print("[export_elevation_json] include_semantic ignored: only "
                      "supported for dem_source='htop' (elev grid has no zone_map).",
                      flush=True)
        else:
            raise HTTPException(
                400, f"Unknown dem_source '{req.dem_source}' (use 'elev' or 'htop').")

    try:
        with profiler.stage("quantize_and_write_elevation_json"):
            msg = elevation_export.dem_to_elevation_msg(
                elev, x_bounds=(x_min, x_max), z_bounds=(z_min, z_max),
                has_data=has_data, scale_factor=req.scale_factor,
                horizontal_scale=req.horizontal_scale, vertical_scale=req.vertical_scale,
                height_resolution=req.height_resolution,
                zone_map=zone_map, legend=legend,
                tile_x=req.tile_x, tile_y=req.tile_y,
                tile_size_meters=req.tile_size_meters,
            )
            out_path = req.out_path or os.path.join(
                req.working_dir, f"elevation_tile_{req.tile_x}_{req.tile_y}.json")
            elevation_export.write_elevation_json(out_path, msg)
    except ValueError as e:
        raise HTTPException(400, f"Elevation export failed: {e}")

    meta = msg["metadata"]
    profile_path = profiler.finish(metadata={"width": meta["width"], "height": meta["height"]})
    print(f"[timing] export_elevation_json: {req.dem_source} "
          f"{meta['width']}x{meta['height']} in "
          f"{time.perf_counter() - t0:.2f}s -> {out_path}", flush=True)
    return {
        "status": "ok",
        "out_path": out_path,
        "dem_source": req.dem_source,
        "has_semantic": "semantic" in msg,
        "metadata": meta,
        "resource_profile_path": profile_path,
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
