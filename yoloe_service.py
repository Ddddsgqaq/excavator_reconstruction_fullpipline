"""
YOLOe segmentation service — run in YOLOe conda environment.

  cd /home/maomaoyu/WS/yoloe
  python /home/maomaoyu/WS/vggt_yoloe/yoloe_service.py --port 8001
"""

import os
import sys
import glob
import gc
import json
import shutil
import time
import argparse
import numpy as np
import cv2
from typing import List, Optional, Dict, Sequence

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# External YOLOe repo. Defaults to a sibling ``yoloe`` checkout next to this
# repo's parent; override with the YOLOE_DIR environment variable.
_YOLOE_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOE_DIR = os.environ.get(
    "YOLOE_DIR",
    os.path.join(os.path.dirname(_YOLOE_SELF_DIR), "yoloe"),
)
sys.path.insert(0, YOLOE_DIR)
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage
from resource_profiler import ResourceProfiler

app = FastAPI(title="YOLOe Segmentation Service")

_models: Dict[str, YOLOE] = {}
_last_nc: Dict[str, int] = {}  # last class count set per model key
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Colour palette for semantic IDs 1-9
_PALETTE = [
    (255, 60, 60), (60, 255, 60), (60, 60, 255), (255, 255, 60),
    (255, 60, 255), (60, 255, 255), (200, 100, 0), (0, 200, 100), (100, 0, 200),
]


def get_model(model_id: str, is_pf: bool = False) -> YOLOE:
    key = f"{model_id}_{'pf' if is_pf else 'seg'}"
    if key not in _models:
        filename = f"{model_id}-seg.pt" if not is_pf else f"{model_id}-seg-pf.pt"
        path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
        m = YOLOE(path)
        m.eval()
        m.to(DEVICE)
        _models[key] = m
    return _models[key]


def reload_model(model_id: str, is_pf: bool = False) -> YOLOE:
    """Force-reload from the locally HF-cached .pt, dropping all in-memory state."""
    key = f"{model_id}_{'pf' if is_pf else 'seg'}"
    if key in _models:
        del _models[key]
    if key in _last_nc:
        del _last_nc[key]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return get_model(model_id, is_pf=is_pf)


def get_model_for_classes(model_id: str, n_classes: int, is_pf: bool = False) -> YOLOE:
    """Return a model ready to accept `n_classes`. Only reloads when the class
    count differs from the last invocation — YOLOe's set_classes is stable for
    same-arity changes but corrupts head/conv shapes when arity changes.
    """
    key = f"{model_id}_{'pf' if is_pf else 'seg'}"
    last = _last_nc.get(key)
    if last is not None and last != n_classes:
        m = reload_model(model_id, is_pf=is_pf)
    else:
        m = get_model(model_id, is_pf=is_pf)
    _last_nc[key] = n_classes
    return m


def get_image_list(working_dir: str) -> List[str]:
    images_dir = os.path.join(working_dir, "images")
    if not os.path.isdir(images_dir):
        raise HTTPException(400, f"Images directory not found: {images_dir}")
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(paths)


# ── Keyframe similarity: ORB feature matching (measures VIEWPOINT change) ──────
# Keyframes are picked by how much the camera VIEWPOINT moved, not by colour.
# An HSV histogram only compares colour distributions, so it reports
# near-identical scores across large viewpoint swings (same scene from a
# different angle has the same palette). ORB keypoints + a RANSAC homography
# measure geometric overlap instead, which is what "did the view change?" means.
_ORB_FEATURES = 1500     # ORB keypoint budget per frame
_ORB_RATIO    = 0.75     # Lowe ratio-test threshold
_ORB_RANSAC   = 4.0      # homography RANSAC reprojection threshold (px)
_ORB_WIDTH    = 320      # downscale frames to this width before ORB (speed)


def _frame_signature(img_path: str):
    """ORB signature for viewpoint comparison: (keypoint xy, descriptors, diag).

    The frame is downscaled to `_ORB_WIDTH` before detection: ORB at full
    1080p costs ~56 ms/frame, vs ~8 ms at 320px. The similarity score is
    scale-invariant — `inlier_ratio` is a ratio, and `_frame_similarity`
    normalizes parallax by the (same-scale) image diagonal — so the downscale
    does not change the keep/skip decision.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    if _ORB_WIDTH and w > _ORB_WIDTH:
        scale = _ORB_WIDTH / w
        img = cv2.resize(img, (_ORB_WIDTH, int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    kp, des = orb.detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.empty((0, 2), np.float32)
    diag = float(np.hypot(*img.shape[:2]))
    return pts, des, diag


def _frame_similarity(sig_a, sig_b) -> float:
    """Viewpoint similarity in [0,1] between two ORB signatures (1 = same view)."""
    ptsA, desA, diag = sig_a
    ptsB, desB, _ = sig_b
    if desA is None or desB is None or len(ptsA) < 8 or len(ptsB) < 8:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(desA, desB, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < _ORB_RATIO * n.distance]
    if len(good) < 8:
        return 0.0
    pA = np.float32([ptsA[m.queryIdx] for m in good])
    pB = np.float32([ptsB[m.trainIdx] for m in good])
    H, mask = cv2.findHomography(pA, pB, cv2.RANSAC, _ORB_RANSAC)
    if H is None or mask is None:
        return 0.0
    mask = mask.ravel().astype(bool)
    n_in = int(mask.sum())
    if n_in == 0:
        return 0.0
    # Backbone = inlier_ratio (graded geometric overlap), discounted by parallax
    # (median pixel motion of inlier matches, relative to the image diagonal —
    # a proxy for camera-motion angle).
    inlier_ratio = n_in / len(good)
    parallax = float(np.median(np.linalg.norm(pA[mask] - pB[mask], axis=1)))
    par_term = 1.0 / (1.0 + (parallax / diag / 0.06) ** 2)
    return float(inlier_ratio * par_term)


def select_keyframe_indices(image_paths: List[str], mode: str = "all",
                            stride: int = 3, sim_thresh: float = 0.92) -> List[int]:
    """Pick which frames to actually segment.

    A scene is many overlapping views; segmenting near-duplicate frames wastes
    time. Returns the indices to run YOLOe on; skipped frames are left unlabeled
    and recovered later by 3D label fusion at reconstruction time.

    Modes:
      "all"        — segment every frame (no saving; default, back-compatible).
      "stride"     — keep every `stride`-th frame (plus first and last).
      "similarity" — keep a frame only when its VIEWPOINT differs enough (ORB
                     viewpoint similarity < `sim_thresh`) from the last kept
                     keyframe. `sim_thresh` is in [0,1]; higher keeps more
                     frames. ~0.45 is a good default.
    """
    n = len(image_paths)
    if mode == "all" or n <= 2:
        return list(range(n))
    if mode == "stride":
        idx = list(range(0, n, max(1, stride)))
        if (n - 1) not in idx:
            idx.append(n - 1)
        return idx
    # similarity — keep a frame when its viewpoint differs enough from the last
    # kept keyframe (ORB viewpoint similarity < sim_thresh).
    keep = [0]
    last_sig = _frame_signature(image_paths[0])
    for i in range(1, n):
        sig = _frame_signature(image_paths[i])
        sim = _frame_similarity(last_sig, sig)
        if sim < sim_thresh:
            keep.append(i)
            last_sig = sig
    if (n - 1) not in keep:
        keep.append(n - 1)
    return sorted(keep)


def make_run_dir(working_dir: str, mode: str) -> str:
    """Create a fresh per-invocation directory; never overwrites previous runs."""
    runs_root = os.path.join(working_dir, "yoloe_runs")
    os.makedirs(runs_root, exist_ok=True)
    run_id = f"run_{int(time.time() * 1000)}_{mode}"
    run_dir = os.path.join(runs_root, run_id)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def save_masks_npz(masks: np.ndarray, run_dir: str) -> str:
    path = os.path.join(run_dir, "semantic_masks.npz")
    np.savez(path, semantic_masks=masks)
    return path


def save_run_meta(run_dir: str, meta: dict) -> None:
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def make_previews(image_paths: List[str], masks: np.ndarray, run_dir: str,
                  frame_indices: Optional[Sequence[int]] = None,
                  max_width: int = 640, fmt: str = "jpg") -> List[str]:
    """Render mask-overlay previews.

    Only the frames in `frame_indices` are rendered (others were skipped during
    keyframe-only segmentation and carry no mask, so their preview would just be
    the raw image). Previews are downscaled to `max_width` and saved as JPEG by
    default, since full-resolution PNG encoding dominates the endpoint wall-clock.
    """
    preview_dir = os.path.join(run_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    idx_iter = range(len(image_paths)) if frame_indices is None else frame_indices
    ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"
    out = []
    max_id = int(masks.max())
    for i in idx_iter:
        img = cv2.imread(image_paths[i])
        if img is None:
            continue
        mask = masks[i]
        overlay = img.copy()
        for sem_id in range(1, max_id + 1):
            region = mask == sem_id
            if region.any():
                color = np.array(_PALETTE[(sem_id - 1) % len(_PALETTE)], dtype=np.float32)
                overlay[region] = (0.4 * overlay[region] + 0.6 * color).astype(np.uint8)
        h, w = overlay.shape[:2]
        if max_width and w > max_width:
            scale = max_width / w
            overlay = cv2.resize(overlay, (max_width, int(round(h * scale))),
                                 interpolation=cv2.INTER_AREA)
        p = os.path.join(preview_dir, f"preview_{i:04d}.{ext}")
        cv2.imwrite(p, overlay)
        out.append(p)
    return out


def masks_from_results(results, H: int, W: int, sem_id_map: Dict[str, int]) -> np.ndarray:
    mask_out = np.zeros((H, W), dtype=np.uint8)
    result = results[0]
    if result.masks is None or len(result.masks) == 0:
        return mask_out
    # masks.data is in letterboxed space; use orig_shape to crop padding before resize
    masks_data = result.masks.data.cpu().numpy()  # (N, mH, mW)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    orig_h, orig_w = result.orig_shape  # original image dimensions
    mH, mW = masks_data.shape[1], masks_data.shape[2]
    # Compute letterbox scale and padding
    scale = min(mH / orig_h, mW / orig_w)
    pad_h = (mH - orig_h * scale) / 2
    pad_w = (mW - orig_w * scale) / 2
    y1, y2 = int(round(pad_h)), int(round(pad_h + orig_h * scale))
    x1, x2 = int(round(pad_w)), int(round(pad_w + orig_w * scale))
    for mask_data, cls_id in zip(masks_data, classes):
        class_name = result.names[cls_id]
        sem_id = sem_id_map.get(class_name, 0)
        if sem_id == 0:
            continue
        cropped = mask_data[y1:y2, x1:x2]
        m = cv2.resize(cropped.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
        mask_out[m > 0] = sem_id
    return mask_out


# ── Request models ────────────────────────────────────────────────────────────

class TextSegRequest(BaseModel):
    working_dir: str
    text_prompts: List[str]
    semantic_id_map: Optional[Dict[str, int]] = None
    model_id: str = "yoloe-v8l"
    image_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.70
    # Keyframe selection: "all" | "stride" | "similarity" (see select_keyframe_indices)
    keyframe_mode: str = "all"
    keyframe_stride: int = 3
    keyframe_sim_thresh: float = 0.45


class VisualSegRequest(BaseModel):
    working_dir: str
    reference_image_path: str
    bboxes: List[List[float]]          # [[x1,y1,x2,y2], ...]
    model_id: str = "yoloe-v8l"
    image_size: int = 640
    conf_threshold: float = 0.20
    iou_threshold: float = 0.70
    keyframe_mode: str = "all"
    keyframe_stride: int = 3
    keyframe_sim_thresh: float = 0.45


class PromptFreeRequest(BaseModel):
    working_dir: str
    model_id: str = "yoloe-v8l"
    image_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.70
    max_classes: int = 9
    keyframe_mode: str = "all"
    keyframe_stride: int = 3
    keyframe_sim_thresh: float = 0.45


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/segment")
@smart_inference_mode()
def segment_text(req: TextSegRequest):
    t0 = time.perf_counter()
    profiler = ResourceProfiler(
        "yoloe_segment_text", req.working_dir, torch_module=torch,
        metadata={
            "model_id": req.model_id, "image_size": req.image_size,
            "prompt_count": len(req.text_prompts), "keyframe_mode": req.keyframe_mode,
        },
    )
    with profiler.stage("discover_input_frames"):
        image_paths = get_image_list(req.working_dir)
    if not image_paths:
        raise HTTPException(400, "No images found")

    sem_id_map = req.semantic_id_map or {cls: i + 1 for i, cls in enumerate(req.text_prompts)}

    # Only reload when prompt arity changes; otherwise reuse cached model.
    with profiler.stage("load_model_and_encode_text_prompts"):
        model = get_model_for_classes(req.model_id, len(req.text_prompts))
        model.set_classes(req.text_prompts, model.get_text_pe(req.text_prompts))
    print(f"[YOLOe text] prompts={req.text_prompts} -> sem_id_map={sem_id_map} "
          f"model.names={model.names if hasattr(model, 'names') else '?'}", flush=True)

    first = cv2.imread(image_paths[0])
    H, W = first.shape[:2]
    semantic_masks = np.zeros((len(image_paths), H, W), dtype=np.uint8)
    detected_classes: Dict[str, List[str]] = {}
    total_detections = 0

    with profiler.stage("select_keyframes"):
        keyframe_idx = select_keyframe_indices(
            image_paths, req.keyframe_mode, req.keyframe_stride, req.keyframe_sim_thresh)
    print(f"[YOLOe text] keyframe_mode={req.keyframe_mode}: segmenting "
          f"{len(keyframe_idx)}/{len(image_paths)} frames", flush=True)

    with profiler.stage("segment_keyframes_and_build_masks", metadata={
            "keyframes": len(keyframe_idx), "total_frames": len(image_paths)}):
        for i in keyframe_idx:
            img_path = image_paths[i]
            pil_img = PILImage.open(img_path).convert("RGB")
            results = model.predict(source=pil_img, imgsz=req.image_size,
                                    conf=req.conf_threshold, iou=req.iou_threshold, verbose=False)
            semantic_masks[i] = masks_from_results(results, H, W, sem_id_map)
            result = results[0]
            if result.boxes is not None:
                n_box = len(result.boxes.cls)
                total_detections += n_box
                cls_arr = result.boxes.cls.cpu().numpy().astype(int)
                names = result.names
                if isinstance(names, list):
                    names = {i: n for i, n in enumerate(names)}
                print(f"[YOLOe text] frame {i}: result.names={result.names} "
                      f"raw_cls_ids={cls_arr.tolist()} "
                      f"mapped={[names.get(int(c), '?') for c in cls_arr]}", flush=True)
                detected_classes[str(i)] = list(
                    {names[int(c)] for c in cls_arr if int(c) in names})
            else:
                detected_classes[str(i)] = []

    print(f"[YOLOe text] {total_detections} detections across {len(image_paths)} frames "
          f"(prompts={req.text_prompts}, conf={req.conf_threshold})", flush=True)

    with profiler.stage("write_masks_previews_and_metadata"):
        run_dir = make_run_dir(req.working_dir, "text")
        preview_paths = make_previews(image_paths, semantic_masks, run_dir,
                                      frame_indices=keyframe_idx)
        masks_path = save_masks_npz(semantic_masks, run_dir)
        save_run_meta(run_dir, {
            "mode": "text", "prompts": req.text_prompts, "model_id": req.model_id,
            "image_size": req.image_size, "conf_threshold": req.conf_threshold,
            "iou_threshold": req.iou_threshold, "semantic_id_map": sem_id_map,
            "total_detections": total_detections, "num_frames": len(image_paths),
            "keyframe_mode": req.keyframe_mode, "num_keyframes": len(keyframe_idx),
            "keyframe_indices": keyframe_idx, "timestamp": time.time(),
            "resource_profile_path": str(profiler.path),
        })

    profile_path = profiler.finish(metadata={
        "frames": len(image_paths), "keyframes": len(keyframe_idx),
        "detections": total_detections,
    })

    print(f"[timing] segment (text): {len(keyframe_idx)}/{len(image_paths)} "
          f"frames in {time.perf_counter() - t0:.2f}s", flush=True)
    return {
        "status": "ok",
        "run_dir": run_dir,
        "semantic_masks_path": masks_path,
        "preview_paths": preview_paths,
        "detected_classes": detected_classes,
        "semantic_id_map": sem_id_map,
        "num_frames": len(image_paths),
        "num_keyframes": len(keyframe_idx),
        "total_detections": total_detections,
        "resource_profile_path": profile_path,
    }


@app.post("/segment_visual")
@smart_inference_mode()
def segment_visual(req: VisualSegRequest):
    t0 = time.perf_counter()
    profiler = ResourceProfiler(
        "yoloe_segment_visual", req.working_dir, torch_module=torch,
        metadata={
            "model_id": req.model_id, "image_size": req.image_size,
            "bbox_count": len(req.bboxes), "keyframe_mode": req.keyframe_mode,
        },
    )
    with profiler.stage("discover_input_frames"):
        image_paths = get_image_list(req.working_dir)
    if not image_paths:
        raise HTTPException(400, "No images found")
    if not os.path.exists(req.reference_image_path):
        raise HTTPException(400, f"Reference image not found: {req.reference_image_path}")

    with profiler.stage("load_model_and_encode_visual_prompt"):
        ref_img = PILImage.open(req.reference_image_path).convert("RGB")
        model = get_model_for_classes(req.model_id, 1)
        bboxes = np.array(req.bboxes)
        prompts = {"bboxes": bboxes, "cls": np.zeros(len(bboxes), dtype=int)}
        model.predict(source=ref_img, imgsz=req.image_size, conf=req.conf_threshold,
                      iou=req.iou_threshold, return_vpe=True,
                      prompts=prompts, predictor=YOLOEVPSegPredictor, verbose=False)
        model.set_classes(["object"], model.predictor.vpe)
        model.predictor = None

    first = cv2.imread(image_paths[0])
    H, W = first.shape[:2]
    semantic_masks = np.zeros((len(image_paths), H, W), dtype=np.uint8)
    total_detections = 0

    with profiler.stage("select_keyframes"):
        keyframe_idx = select_keyframe_indices(
            image_paths, req.keyframe_mode, req.keyframe_stride, req.keyframe_sim_thresh)
    print(f"[YOLOe visual] keyframe_mode={req.keyframe_mode}: segmenting "
          f"{len(keyframe_idx)}/{len(image_paths)} frames", flush=True)

    with profiler.stage("segment_keyframes_and_build_masks", metadata={
            "keyframes": len(keyframe_idx), "total_frames": len(image_paths)}):
        for i in keyframe_idx:
            img_path = image_paths[i]
            pil_img = PILImage.open(img_path).convert("RGB")
            results = model.predict(source=pil_img, imgsz=req.image_size,
                                    conf=req.conf_threshold, iou=req.iou_threshold, verbose=False)
            semantic_masks[i] = masks_from_results(results, H, W, {"object": 1})
            if results[0].boxes is not None:
                total_detections += len(results[0].boxes.cls)

    print(f"[YOLOe visual] {total_detections} detections across {len(image_paths)} frames "
          f"(conf={req.conf_threshold})", flush=True)

    with profiler.stage("write_masks_previews_and_metadata"):
        run_dir = make_run_dir(req.working_dir, "visual")
        preview_paths = make_previews(image_paths, semantic_masks, run_dir,
                                      frame_indices=keyframe_idx)
        masks_path = save_masks_npz(semantic_masks, run_dir)
        save_run_meta(run_dir, {
            "mode": "visual", "reference_image_path": req.reference_image_path,
            "bboxes": req.bboxes, "model_id": req.model_id,
            "image_size": req.image_size, "conf_threshold": req.conf_threshold,
            "iou_threshold": req.iou_threshold, "semantic_id_map": {"object": 1},
            "total_detections": total_detections, "num_frames": len(image_paths),
            "keyframe_mode": req.keyframe_mode, "num_keyframes": len(keyframe_idx),
            "keyframe_indices": keyframe_idx, "timestamp": time.time(),
            "resource_profile_path": str(profiler.path),
        })

    profile_path = profiler.finish(metadata={
        "frames": len(image_paths), "keyframes": len(keyframe_idx),
        "detections": total_detections,
    })

    print(f"[timing] segment (visual): {len(keyframe_idx)}/{len(image_paths)} "
          f"frames in {time.perf_counter() - t0:.2f}s", flush=True)
    return {
        "status": "ok",
        "run_dir": run_dir,
        "semantic_masks_path": masks_path,
        "preview_paths": preview_paths,
        "detected_classes": {},
        "semantic_id_map": {"object": 1},
        "num_frames": len(image_paths),
        "num_keyframes": len(keyframe_idx),
        "total_detections": total_detections,
        "resource_profile_path": profile_path,
    }


@app.post("/segment_promptfree")
@smart_inference_mode()
def segment_promptfree(req: PromptFreeRequest):
    t0 = time.perf_counter()
    profiler = ResourceProfiler(
        "yoloe_segment_promptfree", req.working_dir, torch_module=torch,
        metadata={
            "model_id": req.model_id, "image_size": req.image_size,
            "max_classes": req.max_classes, "keyframe_mode": req.keyframe_mode,
        },
    )
    with profiler.stage("discover_input_frames"):
        image_paths = get_image_list(req.working_dir)
    if not image_paths:
        raise HTTPException(400, "No images found")

    with profiler.stage("load_models_and_build_promptfree_vocabulary"):
        ram_tag_path = os.path.join(YOLOE_DIR, "tools", "ram_tag_list.txt")
        with open(ram_tag_path) as f:
            texts = [x.strip() for x in f.readlines()]
        model = get_model(req.model_id)
        vocab = model.get_vocab(texts)
        pf_model = get_model(req.model_id, is_pf=True)
        pf_model.set_vocab(vocab, names=texts)
        pf_model.model.model[-1].is_fused = True
        pf_model.model.model[-1].conf = 0.001
        pf_model.model.model[-1].max_det = 1000

    first = cv2.imread(image_paths[0])
    H, W = first.shape[:2]

    with profiler.stage("select_keyframes"):
        keyframe_idx = select_keyframe_indices(
            image_paths, req.keyframe_mode, req.keyframe_stride, req.keyframe_sim_thresh)
    print(f"[YOLOe prompt-free] keyframe_mode={req.keyframe_mode}: segmenting "
          f"{len(keyframe_idx)}/{len(image_paths)} frames", flush=True)

    # Two-pass: collect classes first, then build ID map
    results_by_idx: Dict[int, object] = {}
    class_freq: Dict[str, int] = {}
    with profiler.stage("promptfree_keyframe_inference", metadata={
            "keyframes": len(keyframe_idx), "total_frames": len(image_paths)}):
        for i in keyframe_idx:
            pil_img = PILImage.open(image_paths[i]).convert("RGB")
            res = pf_model.predict(source=pil_img, imgsz=req.image_size,
                                   conf=req.conf_threshold, iou=req.iou_threshold, verbose=False)
            results_by_idx[i] = res[0]
            if res[0].boxes is not None:
                for cls_id in res[0].boxes.cls.cpu().numpy().astype(int):
                    name = res[0].names[cls_id]
                    class_freq[name] = class_freq.get(name, 0) + 1

    top_classes = sorted(class_freq, key=lambda x: -class_freq[x])[:req.max_classes]
    sem_id_map = {cls: i + 1 for i, cls in enumerate(top_classes)}

    semantic_masks = np.zeros((len(image_paths), H, W), dtype=np.uint8)
    detected_classes: Dict[str, List[str]] = {}

    with profiler.stage("build_promptfree_semantic_masks"):
        for i, result in results_by_idx.items():
            if result.masks is None or len(result.masks) == 0:
                detected_classes[str(i)] = []
                continue
            masks_data = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            orig_h, orig_w = result.orig_shape
            mH, mW = masks_data.shape[1], masks_data.shape[2]
            scale = min(mH / orig_h, mW / orig_w)
            pad_h = (mH - orig_h * scale) / 2
            pad_w = (mW - orig_w * scale) / 2
            ry1, ry2 = int(round(pad_h)), int(round(pad_h + orig_h * scale))
            rx1, rx2 = int(round(pad_w)), int(round(pad_w + orig_w * scale))
            frame_cls = []
            for mask_data, cls_id in zip(masks_data, classes):
                name = result.names[cls_id]
                sem_id = sem_id_map.get(name, 0)
                if sem_id == 0:
                    continue
                cropped = mask_data[ry1:ry2, rx1:rx2]
                m = cv2.resize(
                    cropped.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
                semantic_masks[i][m > 0] = sem_id
                frame_cls.append(name)
            detected_classes[str(i)] = list(set(frame_cls))

    total_detections = sum(class_freq.values())
    print(f"[YOLOe prompt-free] {total_detections} detections across {len(image_paths)} frames "
          f"(top classes: {list(sem_id_map.keys())}, conf={req.conf_threshold})", flush=True)

    with profiler.stage("write_masks_previews_and_metadata"):
        run_dir = make_run_dir(req.working_dir, "promptfree")
        preview_paths = make_previews(image_paths, semantic_masks, run_dir,
                                      frame_indices=keyframe_idx)
        masks_path = save_masks_npz(semantic_masks, run_dir)
        save_run_meta(run_dir, {
            "mode": "promptfree", "model_id": req.model_id,
            "image_size": req.image_size, "conf_threshold": req.conf_threshold,
            "iou_threshold": req.iou_threshold, "max_classes": req.max_classes,
            "semantic_id_map": sem_id_map, "total_detections": total_detections,
            "num_frames": len(image_paths), "keyframe_mode": req.keyframe_mode,
            "num_keyframes": len(keyframe_idx), "keyframe_indices": keyframe_idx,
            "timestamp": time.time(), "resource_profile_path": str(profiler.path),
        })

    profile_path = profiler.finish(metadata={
        "frames": len(image_paths), "keyframes": len(keyframe_idx),
        "detections": total_detections,
    })

    print(f"[timing] segment (prompt-free): {len(keyframe_idx)}/{len(image_paths)} "
          f"frames in {time.perf_counter() - t0:.2f}s", flush=True)
    return {
        "status": "ok",
        "run_dir": run_dir,
        "semantic_masks_path": masks_path,
        "preview_paths": preview_paths,
        "detected_classes": detected_classes,
        "semantic_id_map": sem_id_map,
        "num_frames": len(image_paths),
        "total_detections": total_detections,
        "resource_profile_path": profile_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
