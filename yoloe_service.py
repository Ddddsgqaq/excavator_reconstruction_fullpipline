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
from typing import List, Optional, Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, "/home/maomaoyu/WS/yoloe")
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage

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


def make_previews(image_paths: List[str], masks: np.ndarray, run_dir: str) -> List[str]:
    preview_dir = os.path.join(run_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    out = []
    for i, (img_path, mask) in enumerate(zip(image_paths, masks)):
        img = cv2.imread(img_path)
        if img is None:
            continue
        overlay = img.copy()
        for sem_id in range(1, int(masks.max()) + 1):
            region = mask == sem_id
            if region.any():
                color = np.array(_PALETTE[(sem_id - 1) % len(_PALETTE)], dtype=np.float32)
                overlay[region] = (0.4 * overlay[region] + 0.6 * color).astype(np.uint8)
        p = os.path.join(preview_dir, f"preview_{i:04d}.png")
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


class VisualSegRequest(BaseModel):
    working_dir: str
    reference_image_path: str
    bboxes: List[List[float]]          # [[x1,y1,x2,y2], ...]
    model_id: str = "yoloe-v8l"
    image_size: int = 640
    conf_threshold: float = 0.20
    iou_threshold: float = 0.70


class PromptFreeRequest(BaseModel):
    working_dir: str
    model_id: str = "yoloe-v8l"
    image_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.70
    max_classes: int = 9


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/segment")
@smart_inference_mode()
def segment_text(req: TextSegRequest):
    image_paths = get_image_list(req.working_dir)
    if not image_paths:
        raise HTTPException(400, "No images found")

    sem_id_map = req.semantic_id_map or {cls: i + 1 for i, cls in enumerate(req.text_prompts)}

    # Only reload when prompt arity changes; otherwise reuse cached model.
    model = get_model_for_classes(req.model_id, len(req.text_prompts))
    model.set_classes(req.text_prompts, model.get_text_pe(req.text_prompts))
    print(f"[YOLOe text] prompts={req.text_prompts} -> sem_id_map={sem_id_map} "
          f"model.names={model.names if hasattr(model, 'names') else '?'}", flush=True)

    first = cv2.imread(image_paths[0])
    H, W = first.shape[:2]
    semantic_masks = np.zeros((len(image_paths), H, W), dtype=np.uint8)
    detected_classes: Dict[str, List[str]] = {}
    total_detections = 0

    for i, img_path in enumerate(image_paths):
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
            detected_classes[str(i)] = list({names[int(c)] for c in cls_arr if int(c) in names})
        else:
            detected_classes[str(i)] = []

    print(f"[YOLOe text] {total_detections} detections across {len(image_paths)} frames "
          f"(prompts={req.text_prompts}, conf={req.conf_threshold})", flush=True)

    run_dir = make_run_dir(req.working_dir, "text")
    preview_paths = make_previews(image_paths, semantic_masks, run_dir)
    masks_path = save_masks_npz(semantic_masks, run_dir)
    save_run_meta(run_dir, {
        "mode": "text",
        "prompts": req.text_prompts,
        "model_id": req.model_id,
        "image_size": req.image_size,
        "conf_threshold": req.conf_threshold,
        "iou_threshold": req.iou_threshold,
        "semantic_id_map": sem_id_map,
        "total_detections": total_detections,
        "num_frames": len(image_paths),
        "timestamp": time.time(),
    })

    return {
        "status": "ok",
        "run_dir": run_dir,
        "semantic_masks_path": masks_path,
        "preview_paths": preview_paths,
        "detected_classes": detected_classes,
        "semantic_id_map": sem_id_map,
        "num_frames": len(image_paths),
        "total_detections": total_detections,
    }


@app.post("/segment_visual")
@smart_inference_mode()
def segment_visual(req: VisualSegRequest):
    image_paths = get_image_list(req.working_dir)
    if not image_paths:
        raise HTTPException(400, "No images found")
    if not os.path.exists(req.reference_image_path):
        raise HTTPException(400, f"Reference image not found: {req.reference_image_path}")

    ref_img = PILImage.open(req.reference_image_path).convert("RGB")
    # Visual mode ends up calling set_classes(["object"], ...) — arity 1.
    # get_model_for_classes will reload only if arity actually changed.
    model = get_model_for_classes(req.model_id, 1)

    bboxes = np.array(req.bboxes)
    prompts = {"bboxes": bboxes, "cls": np.zeros(len(bboxes), dtype=int)}

    # Extract visual prompt embedding
    model.predict(source=ref_img, imgsz=req.image_size, conf=req.conf_threshold,
                  iou=req.iou_threshold, return_vpe=True,
                  prompts=prompts, predictor=YOLOEVPSegPredictor, verbose=False)
    model.set_classes(["object"], model.predictor.vpe)
    model.predictor = None

    first = cv2.imread(image_paths[0])
    H, W = first.shape[:2]
    semantic_masks = np.zeros((len(image_paths), H, W), dtype=np.uint8)
    total_detections = 0

    for i, img_path in enumerate(image_paths):
        pil_img = PILImage.open(img_path).convert("RGB")
        results = model.predict(source=pil_img, imgsz=req.image_size,
                                conf=req.conf_threshold, iou=req.iou_threshold, verbose=False)
        semantic_masks[i] = masks_from_results(results, H, W, {"object": 1})
        if results[0].boxes is not None:
            total_detections += len(results[0].boxes.cls)

    print(f"[YOLOe visual] {total_detections} detections across {len(image_paths)} frames "
          f"(conf={req.conf_threshold})", flush=True)

    run_dir = make_run_dir(req.working_dir, "visual")
    preview_paths = make_previews(image_paths, semantic_masks, run_dir)
    masks_path = save_masks_npz(semantic_masks, run_dir)
    save_run_meta(run_dir, {
        "mode": "visual",
        "reference_image_path": req.reference_image_path,
        "bboxes": req.bboxes,
        "model_id": req.model_id,
        "image_size": req.image_size,
        "conf_threshold": req.conf_threshold,
        "iou_threshold": req.iou_threshold,
        "semantic_id_map": {"object": 1},
        "total_detections": total_detections,
        "num_frames": len(image_paths),
        "timestamp": time.time(),
    })

    return {
        "status": "ok",
        "run_dir": run_dir,
        "semantic_masks_path": masks_path,
        "preview_paths": preview_paths,
        "detected_classes": {},
        "semantic_id_map": {"object": 1},
        "num_frames": len(image_paths),
        "total_detections": total_detections,
    }


@app.post("/segment_promptfree")
@smart_inference_mode()
def segment_promptfree(req: PromptFreeRequest):
    image_paths = get_image_list(req.working_dir)
    if not image_paths:
        raise HTTPException(400, "No images found")

    ram_tag_path = "/home/maomaoyu/WS/yoloe/tools/ram_tag_list.txt"
    with open(ram_tag_path) as f:
        texts = [x.strip() for x in f.readlines()]

    # Prompt-free uses a fixed vocabulary (RAM tag list); class count never
    # changes between calls, so a plain cached load is safe.
    model = get_model(req.model_id)
    vocab = model.get_vocab(texts)
    pf_model = get_model(req.model_id, is_pf=True)
    pf_model.set_vocab(vocab, names=texts)
    pf_model.model.model[-1].is_fused = True
    pf_model.model.model[-1].conf = 0.001
    pf_model.model.model[-1].max_det = 1000

    first = cv2.imread(image_paths[0])
    H, W = first.shape[:2]

    # Two-pass: collect classes first, then build ID map
    all_results = []
    class_freq: Dict[str, int] = {}
    for img_path in image_paths:
        pil_img = PILImage.open(img_path).convert("RGB")
        res = pf_model.predict(source=pil_img, imgsz=req.image_size,
                               conf=req.conf_threshold, iou=req.iou_threshold, verbose=False)
        all_results.append(res[0])
        if res[0].boxes is not None:
            for cls_id in res[0].boxes.cls.cpu().numpy().astype(int):
                name = res[0].names[cls_id]
                class_freq[name] = class_freq.get(name, 0) + 1

    top_classes = sorted(class_freq, key=lambda x: -class_freq[x])[:req.max_classes]
    sem_id_map = {cls: i + 1 for i, cls in enumerate(top_classes)}

    semantic_masks = np.zeros((len(image_paths), H, W), dtype=np.uint8)
    detected_classes: Dict[str, List[str]] = {}

    for i, result in enumerate(all_results):
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
            m = cv2.resize(cropped.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
            semantic_masks[i][m > 0] = sem_id
            frame_cls.append(name)
        detected_classes[str(i)] = list(set(frame_cls))

    run_dir = make_run_dir(req.working_dir, "promptfree")
    preview_paths = make_previews(image_paths, semantic_masks, run_dir)
    masks_path = save_masks_npz(semantic_masks, run_dir)

    total_detections = sum(class_freq.values())
    print(f"[YOLOe prompt-free] {total_detections} detections across {len(image_paths)} frames "
          f"(top classes: {list(sem_id_map.keys())}, conf={req.conf_threshold})", flush=True)

    save_run_meta(run_dir, {
        "mode": "promptfree",
        "model_id": req.model_id,
        "image_size": req.image_size,
        "conf_threshold": req.conf_threshold,
        "iou_threshold": req.iou_threshold,
        "max_classes": req.max_classes,
        "semantic_id_map": sem_id_map,
        "total_detections": total_detections,
        "num_frames": len(image_paths),
        "timestamp": time.time(),
    })

    return {
        "status": "ok",
        "run_dir": run_dir,
        "semantic_masks_path": masks_path,
        "preview_paths": preview_paths,
        "detected_classes": detected_classes,
        "semantic_id_map": sem_id_map,
        "num_frames": len(image_paths),
        "total_detections": total_detections,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
