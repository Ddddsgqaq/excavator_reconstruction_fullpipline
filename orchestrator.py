"""
Orchestrator — Gradio UI that coordinates YOLOe service and VGGT service.

  python /home/maomaoyu/WS/vggt_yoloe/orchestrator.py \
      --yoloe-url http://localhost:8001 \
      --vggt-url  http://localhost:8002

Requires only: gradio, requests, numpy, opencv-python, Pillow
"""

import os
import sys
import shutil
import glob
import json
import time
import gc
import argparse
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import requests
import gradio as gr
from PIL import Image, ImageDraw
from streaming import gradio_controls as live_controls
from streaming import session_gradio_controls as session_controls
from resource_profiler import ResourceProfiler

# ── CLI args (parsed before Gradio builds the UI) ────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--yoloe-url", default="http://localhost:8001")
_parser.add_argument("--vggt-url",  default="http://localhost:8002")
_parser.add_argument("--port",      type=int, default=7860)
_args, _ = _parser.parse_known_args()

YOLOE_URL = _args.yoloe_url
VGGT_URL  = _args.vggt_url

ORCH_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.join(ORCH_DIR, "workspaces")
os.makedirs(WORKSPACE_ROOT, exist_ok=True)

SAM_MASK_CHOICES = ["Best Mask", "Mask 1", "Mask 2", "Mask 3"]


# ── Service helpers ───────────────────────────────────────────────────────────

def _post(url: str, payload: dict, timeout: int = 300) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to {url}. Is the service running?")
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        raise RuntimeError(f"Service error: {detail or str(e)}")


def check_services() -> str:
    msgs = []
    for name, base in [("YOLOe", YOLOE_URL), ("VGGT", VGGT_URL)]:
        try:
            r = requests.get(f"{base}/health", timeout=5)
            d = r.json()
            msgs.append(f"✅ {name} ({d.get('device','?')})")
        except Exception:
            msgs.append(f"❌ {name} — not reachable at {base}")
    return "  |  ".join(msgs)


# ── Upload / frame extraction ─────────────────────────────────────────────────


def _uploaded_path(value, preferred_key=None):
    """Normalize Gradio 4/5 upload values, FileData objects, and legacy paths."""
    if value is None or value == "":
        return None
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, dict):
        keys = ([preferred_key] if preferred_key else []) + ["path", "name", "video"]
        for key in keys:
            if key and value.get(key):
                return _uploaded_path(value[key])
    for attr in ("path", "name"):
        candidate = getattr(value, attr, None)
        if candidate:
            return _uploaded_path(candidate)
    raise ValueError(f"Unsupported Gradio upload value: {type(value).__name__}")

def handle_uploads(input_video, input_images, frame_interval_sec=1.0, max_frames=0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(WORKSPACE_ROOT, f"session_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images)
    profiler = ResourceProfiler(
        "offline_input_preparation", target_dir,
        metadata={
            "frame_interval_s": float(frame_interval_sec),
            "max_frames": int(max_frames),
            "has_video": bool(input_video),
            "uploaded_image_count": len(input_images or []),
        },
    )

    image_paths = []

    with profiler.stage("copy_uploaded_images"):
        if input_images:
            for file_data in input_images:
                fp = _uploaded_path(file_data)
                dst = os.path.join(target_dir_images, os.path.basename(fp))
                shutil.copy(fp, dst)
                image_paths.append(dst)

    with profiler.stage("decode_video_and_write_frames"):
        if input_video:
            vp = _uploaded_path(input_video, preferred_key="video")
            vs = cv2.VideoCapture(vp)
            fps = vs.get(cv2.CAP_PROP_FPS) or 25.0
            interval = max(1, int(fps * frame_interval_sec))
            max_f = int(max_frames)
            count = vf = 0
            while True:
                ok, frame = vs.read()
                if not ok:
                    break
                count += 1
                if count % interval == 0:
                    p = os.path.join(target_dir_images, f"{vf:06d}.png")
                    cv2.imwrite(p, frame)
                    image_paths.append(p)
                    vf += 1
                    if max_f > 0 and vf >= max_f:
                        break
            vs.release()

    image_paths = sorted(image_paths)
    profiler.finish(metadata={"output_frames": len(image_paths)})
    return target_dir, image_paths


def update_gallery_on_upload(input_video, input_images, frame_interval_sec, max_frames):
    empty_class_selector = gr.CheckboxGroup(choices=[], value=[])
    if not input_video and not input_images:
        return (None, None, None,
                "Upload images or a video to begin.",
                gr.Dropdown(choices=["0"], value="0"),
                None, "{}", "{}", "**Status:** No annotations", None, [],
                "", "{}", [], [], [], [], [],
                "**Status:** Ready", empty_class_selector)

    target_dir, image_paths = handle_uploads(input_video, input_images, frame_interval_sec, max_frames)
    frame_dd = _frame_dropdown(target_dir)
    ann_img, summary = _refresh_annotation_display(target_dir, "{}", "0")
    return (None, target_dir, image_paths,
            "Upload complete. Annotate objects, then click Reconstruct.",
            frame_dd, ann_img, "{}", "{}", summary, None, [],
            "", "{}", [], [], [], [], [],
            "**Status:** Ready", empty_class_selector)


# ── Annotation helpers (same logic as demo_gradio_senmatic.py) ────────────────

def _frame_dropdown(target_dir):
    if not target_dir or not os.path.isdir(os.path.join(target_dir, "images")):
        return gr.Dropdown(choices=["0"], value="0")
    imgs = sorted(os.listdir(os.path.join(target_dir, "images")))
    choices = [str(i) for i in range(len(imgs))]
    return gr.Dropdown(choices=choices or ["0"], value="0")


def _load_image_for_annotation(target_dir, frame_idx):
    if not target_dir:
        return None, "No images loaded"
    imgs_dir = os.path.join(target_dir, "images")
    if not os.path.isdir(imgs_dir):
        return None, "Images directory not found"
    imgs = sorted(os.listdir(imgs_dir))
    idx = int(frame_idx) if str(frame_idx).isdigit() else 0
    if idx >= len(imgs):
        return None, f"Frame {idx} not found"
    return os.path.join(imgs_dir, imgs[idx]), f"Loaded frame {idx}"


def _draw_annotations(image_path, annotations_json, current_frame):
    if not image_path or not os.path.exists(image_path):
        return None
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    try:
        annotations = json.loads(annotations_json) if annotations_json else {}
    except Exception:
        annotations = {}
    fk = str(current_frame)
    if fk in annotations:
        for i, (pt, lbl) in enumerate(zip(annotations[fk].get("points", []),
                                          annotations[fk].get("labels", []))):
            x, y = pt
            color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
            r = 8
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=(255,255,255), width=3)
            draw.text((x+r+5, y-r), str(i+1), fill=(255,255,255))
    draw.text((10, h-30), f"Image size: {w} x {h}", fill=(255,255,0))
    return image


def _refresh_annotation_display(target_dir, annotations_json, frame_idx):
    img_path, _ = _load_image_for_annotation(target_dir, frame_idx)
    ann_img = _draw_annotations(img_path, annotations_json, frame_idx) if img_path else None
    try:
        annotations = json.loads(annotations_json) if annotations_json else {}
        fk = str(frame_idx)
        if fk in annotations:
            pts = annotations[fk].get("points", [])
            lbls = annotations[fk].get("labels", [])
            fg = sum(1 for l in lbls if l == 1)
            summary = f"Frame {frame_idx}: {len(pts)} points ({fg} fg, {len(pts)-fg} bg)"
        else:
            summary = f"Frame {frame_idx}: No annotations"
    except Exception:
        summary = "No annotations"
    return ann_img, summary


def on_image_click(evt: gr.SelectData):
    if evt is None or evt.index is None:
        return gr.Number(value=0), gr.Number(value=0)
    return gr.Number(value=int(evt.index[0])), gr.Number(value=int(evt.index[1]))


def add_point_manual(annotations_json, frame_idx, x, y, is_foreground, semantic_id):
    try:
        annotations = json.loads(annotations_json) if annotations_json else {}
    except Exception:
        annotations = {}
    fk = str(frame_idx)
    if fk not in annotations:
        annotations[fk] = {"points": [], "labels": [], "semantic_id": int(semantic_id)}
    annotations[fk]["semantic_id"] = int(semantic_id)
    annotations[fk]["points"].append([int(x), int(y)])
    annotations[fk]["labels"].append(1 if is_foreground else 0)
    updated = json.dumps(annotations, indent=2)
    pts = annotations[fk]["points"]
    fg = sum(1 for l in annotations[fk]["labels"] if l == 1)
    summary = f"Frame {frame_idx}: {len(pts)} points ({fg} fg, {len(pts)-fg} bg)"
    return updated, updated, summary


def add_point_all_frames(target_dir, annotations_json, x, y, is_foreground, semantic_id):
    try:
        annotations = json.loads(annotations_json) if annotations_json else {}
    except Exception:
        annotations = {}
    imgs_dir = os.path.join(target_dir, "images") if target_dir else None
    if not imgs_dir or not os.path.isdir(imgs_dir):
        return annotations_json, annotations_json, "**Status:** No images loaded"
    imgs = sorted(os.listdir(imgs_dir))
    pt = [int(x), int(y)]
    lbl = 1 if is_foreground else 0
    sem_id = int(semantic_id)
    for fi in range(len(imgs)):
        fk = str(fi)
        if fk not in annotations:
            annotations[fk] = {"points": [], "labels": [], "semantic_id": sem_id}
        annotations[fk]["semantic_id"] = sem_id
        annotations[fk]["points"].append(pt.copy())
        annotations[fk]["labels"].append(lbl)
    updated = json.dumps(annotations, indent=2)
    ptype = "foreground" if lbl == 1 else "background"
    return updated, updated, f"**Status:** Added ({pt[0]},{pt[1]}) as {ptype} to {len(imgs)} frames"


def remove_last_point(annotations_json, frame_idx):
    try:
        annotations = json.loads(annotations_json) if annotations_json else {}
    except Exception:
        return annotations_json, annotations_json, "No annotations"
    fk = str(frame_idx)
    if fk in annotations and annotations[fk]["points"]:
        annotations[fk]["points"].pop()
        annotations[fk]["labels"].pop()
        if not annotations[fk]["points"]:
            del annotations[fk]
    updated = json.dumps(annotations, indent=2)
    if fk in annotations:
        pts = annotations[fk]["points"]
        fg = sum(1 for l in annotations[fk]["labels"] if l == 1)
        summary = f"Frame {frame_idx}: {len(pts)} points ({fg} fg, {len(pts)-fg} bg)"
    else:
        summary = f"Frame {frame_idx}: No annotations"
    return updated, updated, summary


def clear_frame_annotations(annotations_json, frame_idx):
    try:
        annotations = json.loads(annotations_json) if annotations_json else {}
    except Exception:
        return "{}", "{}", "No annotations"
    fk = str(frame_idx)
    if fk in annotations:
        del annotations[fk]
    updated = json.dumps(annotations, indent=2)
    return updated, updated, f"Frame {frame_idx}: Cleared"


def clear_all_annotations():
    return "{}", "{}", "All annotations cleared"


# ── YOLOe segmentation calls ──────────────────────────────────────────────────

def run_yoloe_text(target_dir, text_prompts_str, model_id, image_size,
                   conf_threshold, iou_threshold,
                   keyframe_mode="all", keyframe_stride=3, keyframe_sim_thresh=0.45):
    """Call YOLOe service with text prompts; return (masks_path, preview_paths, sem_id_map, status)."""
    if not target_dir or not os.path.isdir(target_dir):
        return None, [], {}, "**Status:** No images loaded"
    prompts = [p.strip() for p in text_prompts_str.split(",") if p.strip()]
    if not prompts:
        return None, [], {}, "**Status:** No text prompts provided"
    try:
        result = _post(f"{YOLOE_URL}/segment", {
            "working_dir": target_dir,
            "text_prompts": prompts,
            "model_id": model_id,
            "image_size": int(image_size),
            "conf_threshold": float(conf_threshold),
            "iou_threshold": float(iou_threshold),
            "keyframe_mode": keyframe_mode,
            "keyframe_stride": int(keyframe_stride),
            "keyframe_sim_thresh": float(keyframe_sim_thresh),
        })
        sem_map = result.get("semantic_id_map", {})
        n_det = result.get("total_detections", 0)
        run_dir = result.get("run_dir", "")
        run_id = os.path.basename(run_dir) if run_dir else ""
        map_str = ", ".join(f"{k}={v}" for k, v in sem_map.items())
        if n_det == 0:
            status = (f"**Status:** [{run_id}] YOLOe ran but found 0 objects for [{map_str}]. "
                      f"Try lowering Confidence (e.g. 0.10), increasing Image Size, "
                      f"or different prompt words.")
        else:
            status = f"**Status:** [{run_id}] Found {n_det} objects. Classes: {map_str}"
        return result["semantic_masks_path"], result["preview_paths"], sem_map, status
    except Exception as e:
        return None, [], {}, f"**Status:** YOLOe error — {e}"


def run_yoloe_visual(target_dir, reference_image_path, bboxes_json,
                     model_id, image_size, conf_threshold, iou_threshold,
                     keyframe_mode="all", keyframe_stride=3, keyframe_sim_thresh=0.45):
    if not target_dir or not os.path.isdir(target_dir):
        return None, [], {}, "**Status:** No images loaded"
    if not reference_image_path or not os.path.exists(reference_image_path):
        return None, [], {}, "**Status:** No reference image"
    try:
        bboxes = json.loads(bboxes_json) if bboxes_json else []
    except Exception:
        return None, [], {}, "**Status:** Invalid bboxes JSON"
    if not bboxes:
        return None, [], {}, "**Status:** No bounding boxes provided"
    try:
        result = _post(f"{YOLOE_URL}/segment_visual", {
            "working_dir": target_dir,
            "reference_image_path": reference_image_path,
            "bboxes": bboxes,
            "model_id": model_id,
            "image_size": int(image_size),
            "conf_threshold": float(conf_threshold),
            "iou_threshold": float(iou_threshold),
            "keyframe_mode": keyframe_mode,
            "keyframe_stride": int(keyframe_stride),
            "keyframe_sim_thresh": float(keyframe_sim_thresh),
        })
        n_det = result.get("total_detections", 0)
        run_dir = result.get("run_dir", "")
        run_id = os.path.basename(run_dir) if run_dir else ""
        if n_det == 0:
            status = (f"**Status:** [{run_id}] Visual segmentation ran but found 0 objects. "
                      "Try lowering Confidence or refining the reference bbox.")
        else:
            status = f"**Status:** [{run_id}] Visual segmentation found {n_det} objects."
        return result["semantic_masks_path"], result["preview_paths"], {"object": 1}, status
    except Exception as e:
        return None, [], {}, f"**Status:** YOLOe error — {e}"


def run_yoloe_promptfree(target_dir, model_id, image_size, conf_threshold, iou_threshold, max_classes,
                         keyframe_mode="all", keyframe_stride=3, keyframe_sim_thresh=0.45):
    if not target_dir or not os.path.isdir(target_dir):
        return None, [], {}, "**Status:** No images loaded"
    try:
        result = _post(f"{YOLOE_URL}/segment_promptfree", {
            "working_dir": target_dir,
            "model_id": model_id,
            "image_size": int(image_size),
            "conf_threshold": float(conf_threshold),
            "iou_threshold": float(iou_threshold),
            "max_classes": int(max_classes),
            "keyframe_mode": keyframe_mode,
            "keyframe_stride": int(keyframe_stride),
            "keyframe_sim_thresh": float(keyframe_sim_thresh),
        })
        sem_map = result.get("semantic_id_map", {})
        n_det = result.get("total_detections", 0)
        run_dir = result.get("run_dir", "")
        run_id = os.path.basename(run_dir) if run_dir else ""
        map_str = ", ".join(f"{k}={v}" for k, v in sem_map.items())
        if n_det == 0:
            status = f"**Status:** [{run_id}] Prompt-free ran but found 0 objects. Try lowering Confidence."
        else:
            status = f"**Status:** [{run_id}] Prompt-free found {n_det} objects. Classes: {map_str}"
        return result["semantic_masks_path"], result["preview_paths"], sem_map, status
    except Exception as e:
        return None, [], {}, f"**Status:** YOLOe error — {e}"


# ── VGGT reconstruction / visualization calls ─────────────────────────────────

def gradio_reconstruct(target_dir, conf_thres, frame_filter, mask_black_bg,
                        mask_white_bg, show_cam, mask_sky, prediction_mode,
                        semantic_masks_path, enable_semantic,
                        fuse_3d=False, fuse_eps=0.05, fuse_dilate_radius=0.0,
                        fuse_min_cluster=30):
    if not target_dir or not os.path.isdir(target_dir):
        return None, "No valid target directory. Please upload first.", None, [], [], [], [], []
    try:
        result = _post(f"{VGGT_URL}/reconstruct", {
            "working_dir": target_dir,
            "semantic_masks_path": semantic_masks_path,
            "conf_thres": float(conf_thres),
            "frame_filter": frame_filter or "All",
            "mask_black_bg": bool(mask_black_bg),
            "mask_white_bg": bool(mask_white_bg),
            "show_cam": bool(show_cam),
            "mask_sky": bool(mask_sky),
            "prediction_mode": prediction_mode,
            "enable_semantic": bool(enable_semantic),
            "fuse_3d": bool(fuse_3d),
            "fuse_eps": float(fuse_eps),
            "fuse_dilate_radius": float(fuse_dilate_radius),
            "fuse_min_cluster": int(fuse_min_cluster),
        }, timeout=600)
    except Exception as e:
        return None, f"VGGT error: {e}", None, [], [], [], [], []

    choices = result.get("frame_filter_choices", ["All"])
    dd = gr.Dropdown(choices=choices, value=frame_filter or "All", interactive=True)
    profile_note = result.get("resource_profile_path", "")
    reconstruction_log = result["log"]
    if profile_note:
        reconstruction_log += f" | resource profile: {profile_note}"
    return (
        result["glb_path"],
        reconstruction_log,
        dd,
        result["depth_paths"],
        result["pointmap_paths"],
        result["semantic_depth_paths"],
        result["semantic_pointmap_paths"],
        result["sam_mask_paths"],
    )


def gradio_visualize(target_dir, conf_thres, frame_filter, mask_black_bg,
                      mask_white_bg, show_cam, mask_sky, prediction_mode,
                      is_example, semantic_masks_path, enable_semantic):
    if is_example == "True":
        return None, "No reconstruction available. Click Reconstruct first."
    if not target_dir or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Click Reconstruct first."
    try:
        result = _post(f"{VGGT_URL}/visualize", {
            "working_dir": target_dir,
            "semantic_masks_path": semantic_masks_path,
            "conf_thres": float(conf_thres),
            "frame_filter": frame_filter or "All",
            "mask_black_bg": bool(mask_black_bg),
            "mask_white_bg": bool(mask_white_bg),
            "show_cam": bool(show_cam),
            "mask_sky": bool(mask_sky),
            "prediction_mode": prediction_mode,
            "enable_semantic": bool(enable_semantic),
        })
        return result["glb_path"], result["log"]
    except Exception as e:
        return None, f"Visualization error: {e}"


def _parse_selected_ids(selected_classes):
    """Parse semantic IDs from CheckboxGroup choices like 'person (1)', 'car (2)'."""
    ids = []
    for cls_str in selected_classes:
        try:
            ids.append(int(cls_str.rsplit("(", 1)[-1].rstrip(")")))
        except (ValueError, IndexError):
            pass
    return ids


def run_edit_pointcloud(target_dir, semantic_masks_path, selected_classes, operation,
                         conf_thres, frame_filter, mask_black_bg, mask_white_bg,
                         show_cam, mask_sky, prediction_mode):
    if not target_dir or not os.path.isdir(target_dir):
        return None, "**Status:** No reconstruction available. Upload and Reconstruct first.", None
    if not semantic_masks_path or not os.path.exists(semantic_masks_path):
        return None, "**Status:** No semantic masks found. Run YOLOe segmentation first.", None
    if not selected_classes:
        return None, "**Status:** No classes selected. Check at least one class.", None

    selected_ids = _parse_selected_ids(selected_classes)
    if not selected_ids:
        return None, "**Status:** Could not parse class IDs from selection.", None

    op = "delete" if "Delete" in operation else "extract"
    try:
        result = _post(f"{VGGT_URL}/edit_pointcloud", {
            "working_dir": target_dir,
            "semantic_masks_path": semantic_masks_path,
            "selected_semantic_ids": selected_ids,
            "operation": op,
            "conf_thres": float(conf_thres),
            "frame_filter": frame_filter or "All",
            "mask_black_bg": bool(mask_black_bg),
            "mask_white_bg": bool(mask_white_bg),
            "show_cam": bool(show_cam),
            "mask_sky": bool(mask_sky),
            "prediction_mode": prediction_mode,
        })
        glb_path = result["glb_path"]
        return glb_path, f"**Status:** {result['log']}", glb_path
    except Exception as e:
        return None, f"**Status:** Edit error — {e}", None


# ── Example pipeline ──────────────────────────────────────────────────────────

def example_pipeline(input_video, num_images_str, input_images, conf_thres,
                     mask_black_bg, mask_white_bg, show_cam, mask_sky,
                     prediction_mode, is_example_str):
    target_dir, image_paths = handle_uploads(input_video, input_images)
    glb, log, dd, dp, pp, sdp, spp, smp = gradio_reconstruct(
        target_dir, conf_thres, "All", mask_black_bg, mask_white_bg,
        show_cam, mask_sky, prediction_mode, None, False
    )
    return glb, log, target_dir, dd, image_paths, dp, pp, sdp, spp, None, smp


# ── Gradio UI ─────────────────────────────────────────────────────────────────

theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

CSS = """
.custom-log * {
    font-style: italic; font-size: 22px !important; font-weight: bold !important;
    background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
    -webkit-background-clip: text; background-clip: text;
    color: transparent !important; text-align: center !important;
}
.example-log * {
    font-style: italic; font-size: 16px !important;
    background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent !important;
}
#my_radio .wrap { display:flex; flex-wrap:nowrap; justify-content:center; align-items:center; }
#my_radio .wrap label { display:flex; width:50%; justify-content:center; align-items:center;
    margin:0; padding:10px 0; box-sizing:border-box; }
"""

with gr.Blocks(theme=theme, css=CSS) as demo:
    is_example        = gr.Textbox(visible=False, value="None")
    num_images        = gr.Textbox(visible=False, value="None")
    target_dir_output = gr.Textbox(visible=False, value="None")
    # Stores path to the current semantic_masks.npz produced by YOLOe
    semantic_masks_path_state = gr.Textbox(visible=False, value="")
    # Stores the class→semantic_id mapping from the last YOLOe run
    semantic_id_map_state = gr.Textbox(visible=False, value="{}")

    gr.HTML("""
    <h1>🏛️ VGGT + YOLOe: Semantic 3D Reconstruction</h1>
    <p>
      <a href="https://github.com/facebookresearch/vggt">🐙 VGGT GitHub</a> |
      <a href="https://github.com/THU-MIG/yoloe">🐙 YOLOe GitHub</a>
    </p>
    <p>Upload images or a video, annotate objects with YOLOe (text / visual / prompt-free),
    then click <strong>Reconstruct</strong> to get a semantically-labelled 3D point cloud.</p>
    """)

    service_status = gr.Markdown(check_services())
    refresh_status_btn = gr.Button("🔄 Refresh Service Status", size="sm")
    refresh_status_btn.click(fn=check_services, outputs=[service_status], queue=False)

    # ── Live camera monitor (isolated from the existing offline workflow) ─────
    with gr.Accordion("📷 实时相机与在线建图", open=False):
        gr.Markdown(
            "默认使用下方上传的文件做静态重建。只有显式勾选后才会开放实时取流按钮；"
            "进入实时模式后，仍需先做 **仅取流预览**，再手动启动 VGGT 滑窗重建。"
        )
        enable_live_mode = gr.Checkbox(
            label="我要接入实时视频流",
            value=False,
            info="默认关闭。取流必须经过此开关和下方的‘连接并预览’两步。",
        )
        with gr.Row():
            live_source_type = gr.Dropdown(
                choices=["usb", "http", "rtsp", "video"], value="usb", label="输入类型"
            )
            live_source_uri = gr.Textbox(
                value="0", label="设备编号 / HTTP根地址或MJPEG URL / RTSP URL / 视频路径",
                info=("USB 可填 0 或 /dev/video0；IP Webcam 根地址示例："
                      "http://192.168.1.20:8080/；URL 凭据不会显示在状态信息中"),
            )
            live_backend = gr.Dropdown(
                choices=["auto", "v4l2", "ffmpeg", "gstreamer"],
                value="auto", label="OpenCV 后端",
            )
            live_target_fps = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="ORB 模式候选帧 FPS")
        with gr.Row():
            live_use_orb = gr.Checkbox(
                value=True, label="启用 ORB 视角关键帧筛选",
                info="关闭后完全跳过 ORB，按右侧时间间隔采样帧送入 VGGT。",
            )
            live_frame_sample_interval = gr.Slider(
                0.1, 10.0, value=1.0, step=0.1, label="关闭 ORB 时的采样间隔（秒）",
                interactive=False,
            )
        with gr.Row():
            live_interval = gr.Slider(3.0, 20.0, value=6.0, step=1.0, label="重建间隔（秒）")
            live_min_frames = gr.Slider(2, 12, value=4, step=1, label="最少关键帧")
            live_capacity = gr.Slider(4, 24, value=12, step=1, label="滑窗容量")
            live_file_out = gr.Textbox(
                value="/mnt/d/tuanjie/exea1/excavator-app-unity-main/live_elevation",
                label="DEM 输出目录（Unity 正在轮询；留空则自动创建 workspace）",
            )
            live_save_glb = gr.Checkbox(
                value=False, label="保存 VGGT 重建点云 GLB（存至 pointclouds/，便于监督过程）"
            )
            live_fusion = gr.Checkbox(
                value=True, label="启用跨轮 DEM 融合",
                info="开启：右侧 Global DEM 持久融合并供 Unity 发布；关闭：每轮 DEM 覆盖旧结果。",
            )
            live_scale_factor = gr.Number(
                value=28.0, label="物理尺度（1 unit = ? m）", precision=2,
                info="与原版 Elevation Viewer 默认值一致；Unity 50m tile 下决定有效栅格覆盖率。",
            )
        with gr.Row():
            live_connect_btn = gr.Button(
                "① 连接并预览（不运行 VGGT）", variant="secondary", interactive=False
            )
            live_reconstruct_btn = gr.Button(
                "② 启动相机 → VGGT 滑窗重建", variant="primary", interactive=False
            )
            live_stop_btn = gr.Button("停止实时任务", variant="stop", interactive=False)
            live_refresh_btn = gr.Button("刷新预览/状态", interactive=False)
        live_action_status = gr.Markdown("**状态：** 未连接")
        live_monitor_status = gr.Markdown("**监控：** 未运行")
        live_preview = gr.Image(label="后端最新帧", type="pil", interactive=False, height=420)
        gr.HTML(f"""
        <div style="margin:12px 0 8px;display:flex;justify-content:space-between;align-items:center">
          <strong>实时高程双视图（本轮 DEM + 融合 Global DEM）</strong>
          <a href="{VGGT_URL}/stream/viewer?v=dem_fusion_v3" target="_blank">在新窗口打开双视图</a>
        </div>
        <iframe src="{VGGT_URL}/stream/viewer?v=dem_fusion_v3" title="实时高程查看器"
          style="width:100%;height:1180px;border:1px solid #334155;border-radius:10px;background:#071018"></iframe>
        """)
        with gr.Accordion("🗺️ 正式两阶段 MapSession", open=False):
            gr.Markdown(
                "初始化会在采满关键帧后停止并等待审核；只有 READY 会话才能开始可信增量更新。"
            )
            with gr.Row():
                live_session_dir = gr.Textbox(value="", label="MapSession 目录（留空自动创建）")
                live_init_frames = gr.Slider(6, 48, value=12, step=1, label="初始化关键帧数")
            with gr.Row():
                live_init_btn = gr.Button("③ 初始化基准地图", variant="primary", interactive=False)
                live_approve_btn = gr.Button("④ 审核通过 → READY", interactive=False)
                live_reject_btn = gr.Button(
                    "审核拒绝 → 重新初始化", variant="stop", interactive=False)
                live_update_btn = gr.Button(
                    "⑤ 启动可信增量更新", variant="primary", interactive=False)

        # Idle by default: only poll the backend once a stream has been started, so an
        # untouched page never hammers VGGT (and never blocks on /stream/status).
        live_timer = gr.Timer(value=2.0, active=False)

        def _connect(enabled, st, uri, backend, fps, use_orb, sample_interval):
            if not enabled:
                return "**状态：** ❌ 请先勾选‘我要接入实时视频流’。", gr.Timer(active=False)
            msg = live_controls.connect_camera(
                VGGT_URL, st, uri, backend, fps, use_orb, sample_interval
            )
            return msg, gr.Timer(active="❌" not in msg)

        def _start_recon(enabled, st, uri, backend, fps, use_orb, sample_interval,
                         interval, min_frames, capacity, out_dir, save_glb, fusion, scale_factor):
            if not enabled:
                return ("**状态：** ❌ 请先勾选‘我要接入实时视频流’。",
                        str(out_dir or ""), gr.Timer(active=False))
            msg, out = live_controls.start_live_reconstruction(
                VGGT_URL, WORKSPACE_ROOT, st, uri, backend, fps,
                interval, min_frames, capacity, out_dir, save_glb,
                use_orb, sample_interval, fusion, scale_factor,
            )
            return msg, out, gr.Timer(active="❌" not in msg)

        def _init_session(enabled, st, uri, backend, fps, use_orb, sample_interval,
                          interval, capacity, path, init_frames):
            if not enabled:
                return ("**状态：** ❌ 请先勾选‘我要接入实时视频流’。",
                        str(path or ""), gr.Timer(active=False))
            msg, path = session_controls.initialize_session(
                VGGT_URL, WORKSPACE_ROOT, st, uri, backend, fps,
                use_orb, sample_interval, interval, capacity, path, init_frames,
            )
            return msg, path, gr.Timer(active="❌" not in msg)

        def _review_session(enabled, path, approved):
            if not enabled:
                return "**状态：** ❌ 请先勾选‘我要接入实时视频流’。"
            return session_controls.finalize_session(VGGT_URL, path, approved)

        def _start_session_update(enabled, st, uri, backend, fps, use_orb,
                                  sample_interval, interval, min_frames, capacity,
                                  path, out_dir):
            if not enabled:
                return ("**状态：** ❌ 请先勾选‘我要接入实时视频流’。",
                        str(out_dir or ""), gr.Timer(active=False))
            msg, out = session_controls.start_session_update(
                VGGT_URL, st, uri, backend, fps, use_orb, sample_interval,
                interval, min_frames, capacity, path, out_dir,
            )
            return msg, out, gr.Timer(active="❌" not in msg)


        def _toggle_live_mode(enabled):
            button_update = gr.update(interactive=bool(enabled))
            if enabled:
                return (button_update, button_update, button_update, button_update,
                        button_update, button_update, button_update, button_update,
                        "**状态：** 实时模式已开放，请先连接并预览。",
                        None, "**监控：** 未运行", gr.Timer(active=False))
            live_controls.stop_camera(VGGT_URL)
            return (button_update, button_update, button_update, button_update,
                    button_update, button_update, button_update, button_update,
                    "**状态：** 已返回静态重建模式。",
                    None, "**监控：** 未运行", gr.Timer(active=False))

        enable_live_mode.change(
            fn=_toggle_live_mode,
            inputs=[enable_live_mode],
            outputs=[live_connect_btn, live_reconstruct_btn, live_stop_btn, live_refresh_btn,
                     live_init_btn, live_approve_btn, live_reject_btn, live_update_btn,
                     live_action_status, live_preview, live_monitor_status, live_timer],
            queue=False,
        )

        live_use_orb.change(
            fn=lambda enabled: gr.update(interactive=not bool(enabled)),
            inputs=[live_use_orb], outputs=[live_frame_sample_interval], queue=False,
        )

        live_connect_btn.click(
            fn=_connect,
            inputs=[enable_live_mode, live_source_type, live_source_uri, live_backend, live_target_fps,
                    live_use_orb, live_frame_sample_interval],
            outputs=[live_action_status, live_timer],
            queue=False,
        )
        live_reconstruct_btn.click(
            fn=_start_recon,
            inputs=[enable_live_mode, live_source_type, live_source_uri, live_backend, live_target_fps,
                    live_use_orb, live_frame_sample_interval, live_interval, live_min_frames,
                    live_capacity, live_file_out, live_save_glb, live_fusion, live_scale_factor],
            outputs=[live_action_status, live_file_out, live_timer],
            queue=False,
        )
        live_init_btn.click(
            fn=_init_session,
            inputs=[enable_live_mode, live_source_type, live_source_uri, live_backend, live_target_fps,
                    live_use_orb, live_frame_sample_interval, live_interval, live_capacity,
                    live_session_dir, live_init_frames],
            outputs=[live_action_status, live_session_dir, live_timer],
            queue=False,
        )
        live_approve_btn.click(
            fn=lambda enabled, path: _review_session(enabled, path, True),
            inputs=[enable_live_mode, live_session_dir],
            outputs=[live_action_status],
            queue=False,
        )
        live_reject_btn.click(
            fn=lambda enabled, path: _review_session(enabled, path, False),
            inputs=[enable_live_mode, live_session_dir],
            outputs=[live_action_status],
            queue=False,
        )
        live_update_btn.click(
            fn=_start_session_update,
            inputs=[enable_live_mode, live_source_type, live_source_uri, live_backend, live_target_fps,
                    live_use_orb, live_frame_sample_interval, live_interval, live_min_frames,
                    live_capacity, live_session_dir, live_file_out],
            outputs=[live_action_status, live_file_out, live_timer],
            queue=False,
        )

        live_stop_btn.click(
            fn=lambda: (live_controls.stop_camera(VGGT_URL), gr.Timer(active=False)),
            outputs=[live_action_status, live_timer],
            queue=False,
        )
        live_refresh_btn.click(
            fn=lambda: live_controls.poll_camera(VGGT_URL),
            outputs=[live_preview, live_monitor_status],
            queue=False,
        )
        live_timer.tick(
            fn=lambda: live_controls.poll_camera(VGGT_URL),
            outputs=[live_preview, live_monitor_status],
            show_progress="hidden",
            queue=False,
        )

    # ── Row: upload + 3D viewer ───────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            with gr.Row():
                frame_interval_sec = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Frame Interval (s)")
                max_frames = gr.Number(value=0, precision=0, label="Max Frames (0=no limit)")
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
            image_gallery = gr.Gallery(label="Preview", columns=4, height="300px",
                                       show_download_button=True, object_fit="contain", preview=True)

        with gr.Column(scale=4):
            gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
            log_output = gr.Markdown("Upload images or a video, then click Reconstruct.",
                                     elem_classes=["custom-log"])
            reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn  = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output,
                     target_dir_output, image_gallery], scale=1)

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    value="Depthmap and Camera Branch", label="Prediction Mode",
                    scale=1, elem_id="my_radio")

            with gr.Row():
                conf_thres    = gr.Slider(0, 100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter  = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam      = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky      = gr.Checkbox(label="Filter Sky",  value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

            # ── Tabs ──────────────────────────────────────────────────────────
            with gr.Tabs():
                # Depth Map
                with gr.TabItem("Depth Map"):
                    depth_gallery = gr.Gallery(label="Depth Map", columns=4, height="300px",
                                               show_download_button=True, object_fit="contain", preview=True)
                    depth_dl_btn = gr.DownloadButton("Download All Depth Maps (.zip)", variant="secondary")

                # Point Map
                with gr.TabItem("Point Map"):
                    pointmap_gallery = gr.Gallery(label="Point Map", columns=4, height="300px",
                                                  show_download_button=True, object_fit="contain", preview=True)
                    pointmap_dl_btn = gr.DownloadButton("Download All Point Maps (.zip)", variant="secondary")

                # Semantic Depth Map
                with gr.TabItem("Semantic Depth Map"):
                    semantic_depth_gallery = gr.Gallery(label="Semantic Depth Map", columns=4, height="300px",
                                                        show_download_button=True, object_fit="contain", preview=True)
                    sem_depth_dl_btn = gr.DownloadButton("Download (.zip)", variant="secondary")

                # Semantic Point Map
                with gr.TabItem("Semantic Point Map"):
                    semantic_pointmap_gallery = gr.Gallery(label="Semantic Point Map", columns=4, height="300px",
                                                           show_download_button=True, object_fit="contain", preview=True)
                    sem_ptmap_dl_btn = gr.DownloadButton("Download (.zip)", variant="secondary")

                # YOLOe Segmentation tab
                with gr.TabItem("YOLOe Segmentation"):
                    gr.Markdown("## YOLOe Semantic Segmentation")
                    gr.Markdown("Segment objects across all frames before reconstruction. "
                                "Choose a prompt mode, configure, then click **Run YOLOe**.")

                    enable_semantic_cb = gr.Checkbox(label="Enable Semantic Segmentation in Reconstruction", value=False)

                    with gr.Accordion("3D Label Fusion (recover keyframe-skipped frames)", open=False):
                        gr.Markdown(
                            "After reconstruction, propagate semantic labels through the "
                            "shared 3D cloud: fills targets only segmented on some "
                            "(key)frames and drops isolated mis-detections. Applied during "
                            "**Reconstruct** when Semantic Segmentation is enabled.")
                        recon_fuse_3d = gr.Checkbox(value=False, label="Enable 3D Fusion")
                        with gr.Row():
                            recon_fuse_eps = gr.Slider(
                                0.005, 0.5, value=0.05, step=0.005,
                                label="Cluster radius (eps)", info="World units")
                            recon_fuse_radius = gr.Slider(
                                0.0, 0.5, value=0.0, step=0.005,
                                label="Dilate radius", info="0 = same as eps")
                        recon_fuse_min = gr.Slider(
                            1, 200, value=30, step=1, label="Min cluster size",
                            info="Drop clusters smaller than this as noise")

                    with gr.Row():
                        # Left: annotation image
                        with gr.Column(scale=2):
                            gr.Markdown("### Frame Preview (click to set X/Y)")
                            yoloe_annotator_image = gr.Image(
                                label="Annotated Frame", type="pil",
                                interactive=True, height=480)

                        # Right: controls
                        with gr.Column(scale=1):
                            gr.Markdown("### Prompt Mode")
                            yoloe_prompt_mode = gr.Radio(
                                ["Text", "Visual", "Prompt-Free"],
                                value="Text", label="YOLOe Prompt Mode")

                            # ── Text prompt controls ──────────────────────────
                            with gr.Group(visible=True) as text_group:
                                gr.Markdown("**Text Prompts** (comma-separated class names)")
                                yoloe_text_prompts = gr.Textbox(
                                    value="person,car", label="Classes",
                                    placeholder="person,car,building")

                            # ── Visual prompt controls ────────────────────────
                            with gr.Group(visible=False) as visual_group:
                                gr.Markdown("**Visual Prompt** — upload a reference image and draw bounding boxes")
                                yoloe_ref_image = gr.Image(
                                    label="Reference Image", type="filepath",
                                    interactive=True, height=200)
                                yoloe_bboxes_json = gr.Textbox(
                                    label="Bounding Boxes JSON [[x1,y1,x2,y2],...]",
                                    placeholder='[[100,200,300,400]]',
                                    lines=2)

                            # ── Prompt-free controls ──────────────────────────
                            with gr.Group(visible=False) as pf_group:
                                gr.Markdown("**Prompt-Free** — auto-detect all objects")
                                yoloe_max_classes = gr.Slider(1, 9, value=5, step=1,
                                                              label="Max Semantic Classes")

                            gr.Markdown("### Model Settings")
                            yoloe_model_id = gr.Dropdown(
                                choices=["yoloe-v8s","yoloe-v8m","yoloe-v8l",
                                         "yoloe-11s","yoloe-11m","yoloe-11l"],
                                value="yoloe-v8l", label="YOLOe Model")
                            yoloe_image_size = gr.Slider(320, 1280, value=640, step=32, label="Image Size")
                            yoloe_conf = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Confidence")
                            yoloe_iou  = gr.Slider(0.0, 1.0, value=0.70, step=0.05, label="IoU")

                            gr.Markdown("### Keyframe Selection")
                            gr.Markdown(
                                "Segment only representative frames to save time. "
                                "Skipped frames are recovered by **3D Fusion** at "
                                "reconstruction (enable it on the Reconstruct controls).")
                            yoloe_keyframe_mode = gr.Radio(
                                ["all", "stride", "similarity"], value="all",
                                label="Keyframe Mode",
                                info="all: every frame · stride: every Nth · "
                                     "similarity: skip near-duplicate frames")
                            yoloe_keyframe_stride = gr.Slider(
                                1, 20, value=3, step=1, label="Stride (N)",
                                info="Used in 'stride' mode")
                            yoloe_keyframe_sim = gr.Slider(
                                0.1, 0.9, value=0.45, step=0.05,
                                label="Viewpoint similarity threshold",
                                info="Used in 'similarity' mode (ORB viewpoint match); "
                                     "higher = keep more frames")

                            run_yoloe_btn = gr.Button("▶ Run YOLOe", variant="primary", size="lg")
                            yoloe_status  = gr.Markdown("**Status:** Ready")

                            gr.Markdown("---")
                            gr.Markdown("### Manual Point Annotation (optional)")
                            gr.Markdown("Click image → coordinates fill below → Add Point")

                            yoloe_frame_selector = gr.Dropdown(
                                choices=["0"], value="0",
                                label="Select Frame", interactive=True)
                            yoloe_semantic_id = gr.Number(value=1, label="Semantic ID (1-9)",
                                                          minimum=1, maximum=9, precision=0)
                            with gr.Row():
                                point_x = gr.Number(label="X", value=0, precision=0)
                                point_y = gr.Number(label="Y", value=0, precision=0)
                            point_type = gr.Radio(
                                ["Foreground (Green)", "Background (Red)"],
                                value="Foreground (Green)", label="Point Type")
                            add_point_btn          = gr.Button("➕ Add Point", variant="primary")
                            add_point_all_btn      = gr.Button("Add to All Frames", variant="secondary")
                            with gr.Row():
                                undo_btn       = gr.Button("↶ Undo")
                                clear_frame_btn = gr.Button("🗑️ Clear Frame")
                            clear_all_btn = gr.Button("🗑️ Clear All", variant="stop")

                            ann_status = gr.Markdown("**Status:** No annotations")
                            ann_display = gr.Textbox(label="Annotations JSON", lines=4,
                                                     interactive=False)
                            ann_json_hidden = gr.Textbox(visible=False, value="{}")

                    # YOLOe preview gallery
                    yoloe_preview_gallery = gr.Gallery(
                        label="YOLOe Segmentation Preview", columns=4, height="300px",
                        show_download_button=True, object_fit="contain", preview=True)

                # SAM Masks (kept for compatibility)
                with gr.TabItem("Seg Masks"):
                    sam_preview_image = gr.Image(label="Mask Preview", type="pil",
                                                 interactive=False, height=300)
                    sam_mask_gallery  = gr.Gallery(label="Segmentation Masks", columns=4,
                                                   height="300px", show_download_button=True,
                                                   object_fit="contain", preview=True)

                # Point Cloud Editing
                with gr.TabItem("Point Cloud Editing"):
                    gr.Markdown("## Point Cloud Editing")
                    gr.Markdown(
                        "Edit 3D scene points based on YOLOE segmentation results. "
                        "**Requires:** Run YOLOe segmentation first, then Reconstruct with "
                        "**Enable Semantic Segmentation** checked."
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            edit_result_viewer = gr.Model3D(
                                height=480, zoom_speed=0.5, pan_speed=0.5,
                                label="Edited Scene")
                            edit_log = gr.Markdown("**Status:** Run YOLOe and Reconstruct first.")
                            edit_dl_btn = gr.DownloadButton(
                                "Download Edited Scene (.glb)", variant="secondary")

                        with gr.Column(scale=1):
                            gr.Markdown("### Select Semantic Classes")
                            edit_class_selector = gr.CheckboxGroup(
                                choices=[], label="Classes to Edit",
                                info="Run YOLOe first to populate this list")

                            edit_operation = gr.Radio(
                                ["Delete from scene", "Extract separately"],
                                value="Delete from scene",
                                label="Operation",
                                info="Delete: remove selected points from scene. "
                                     "Extract: keep only selected points.")

                            apply_edit_btn = gr.Button(
                                "▶ Apply Edit", variant="primary", size="lg")

                # Independent two-scene overlay tool (does not share elevation viewer state).
                with gr.TabItem("GLB Scene Alignment"):
                    gr.Markdown("## 双 GLB 场景叠加预览")
                    gr.Markdown(
                        "用于验证挖掘前后两个导出场景的初始重叠效果。"
                        "该工具在独立页面运行，不会影响高程查看器；当前仅按导出坐标叠加，"
                        "后续可在此基础上加入自动对齐。"
                    )
                    open_fusion_viewer_btn = gr.Button(
                        "🌐 打开双 GLB 叠加工具", variant="primary", size="lg"
                    )

                # Elevation Plane
                with gr.TabItem("Elevation Plane"):
                    gr.Markdown("## 3D Elevation Viewer")
                    gr.Markdown(
                        "Open the interactive 3D elevation viewer to inspect the "
                        "gravity-aligned point cloud, DEM, camera trajectory, and "
                        "volume selection. **Requires:** Run Reconstruct first."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            elev_use_ground_filter = gr.Checkbox(
                                label="Use ground filter for DEM",
                                value=True,
                                info="On: build the DEM from filtered ground candidates. Off: from all aligned points.")
                            open_viewer_btn = gr.Button(
                                "🌐 Open 3D Elevation Viewer", variant="primary", size="lg")

    # ── Examples ──────────────────────────────────────────────────────────────
    _vggt_dir = os.environ.get("VGGT_DIR", os.path.join(os.path.dirname(ORCH_DIR), "vggt"))
    EXAMPLES_DIR = os.path.join(_vggt_dir, "examples", "videos")
    # (ORCH_DIR is this repo's dir; sibling ``vggt`` checkout is the default.)
    _ex_videos = {
        "colosseum": f"{EXAMPLES_DIR}/Colosseum.mp4",
        "pyramid":   f"{EXAMPLES_DIR}/pyramid.mp4",
        "room":      f"{EXAMPLES_DIR}/room.mp4",
        "kitchen":   f"{EXAMPLES_DIR}/kitchen.mp4",
    }
    examples_list = [
        [v, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"]
        for v in _ex_videos.values() if os.path.exists(v)
    ]

    if examples_list:
        gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])
        gr.Examples(
            examples=examples_list,
            inputs=[input_video, num_images, input_images, conf_thres,
                    mask_black_bg, mask_white_bg, show_cam, mask_sky,
                    prediction_mode, is_example],
            outputs=[reconstruction_output, log_output, target_dir_output, frame_filter,
                     image_gallery, depth_gallery, pointmap_gallery,
                     semantic_depth_gallery, semantic_pointmap_gallery,
                     sam_preview_image, sam_mask_gallery],
            fn=example_pipeline,
            cache_examples=False,
            examples_per_page=50,
        )

    # ── Event wiring ──────────────────────────────────────────────────────────

    # Upload triggers
    _upload_outputs = [reconstruction_output, target_dir_output, image_gallery, log_output,
                       yoloe_frame_selector, yoloe_annotator_image,
                       ann_json_hidden, ann_display, ann_status,
                       sam_preview_image, sam_mask_gallery,
                       semantic_masks_path_state, semantic_id_map_state,
                       yoloe_preview_gallery, depth_gallery, pointmap_gallery,
                       semantic_depth_gallery, semantic_pointmap_gallery,
                       yoloe_status, edit_class_selector]
    _upload_inputs  = [input_video, input_images, frame_interval_sec, max_frames]
    for _comp in [input_video, input_images, frame_interval_sec, max_frames]:
        _comp.change(fn=update_gallery_on_upload, inputs=_upload_inputs, outputs=_upload_outputs)

    # Clear button extras
    clear_btn.click(fn=lambda: (None, None, None, None, None, None),
                    outputs=[depth_gallery, pointmap_gallery, semantic_depth_gallery,
                              semantic_pointmap_gallery, sam_preview_image, sam_mask_gallery])

    # YOLOe prompt mode visibility
    def _toggle_prompt_groups(mode):
        return (gr.update(visible=mode == "Text"),
                gr.update(visible=mode == "Visual"),
                gr.update(visible=mode == "Prompt-Free"))

    yoloe_prompt_mode.change(
        fn=_toggle_prompt_groups,
        inputs=[yoloe_prompt_mode],
        outputs=[text_group, visual_group, pf_group])

    # Run YOLOe button
    def _run_yoloe_dispatch(target_dir, mode, text_prompts, ref_img, bboxes_json,
                             model_id, img_size, conf, iou, max_cls,
                             kf_mode, kf_stride, kf_sim):
        if mode == "Text":
            masks_path, previews, sem_map, status = run_yoloe_text(
                target_dir, text_prompts, model_id, img_size, conf, iou,
                kf_mode, kf_stride, kf_sim)
        elif mode == "Visual":
            masks_path, previews, sem_map, status = run_yoloe_visual(
                target_dir, ref_img, bboxes_json, model_id, img_size, conf, iou,
                kf_mode, kf_stride, kf_sim)
        else:
            masks_path, previews, sem_map, status = run_yoloe_promptfree(
                target_dir, model_id, img_size, conf, iou, max_cls,
                kf_mode, kf_stride, kf_sim)
        return masks_path or "", previews, status, json.dumps(sem_map)

    run_yoloe_btn.click(
        fn=_run_yoloe_dispatch,
        inputs=[target_dir_output, yoloe_prompt_mode, yoloe_text_prompts,
                yoloe_ref_image, yoloe_bboxes_json,
                yoloe_model_id, yoloe_image_size, yoloe_conf, yoloe_iou,
                yoloe_max_classes,
                yoloe_keyframe_mode, yoloe_keyframe_stride, yoloe_keyframe_sim],
        outputs=[semantic_masks_path_state, yoloe_preview_gallery, yoloe_status,
                 semantic_id_map_state])

    # Frame selector change → refresh annotation display
    yoloe_frame_selector.change(
        fn=_refresh_annotation_display,
        inputs=[target_dir_output, ann_json_hidden, yoloe_frame_selector],
        outputs=[yoloe_annotator_image, ann_status])

    # Click on image → fill X/Y
    yoloe_annotator_image.select(fn=on_image_click, outputs=[point_x, point_y])

    # Add point
    add_point_btn.click(
        fn=lambda j, f, x, y, pt, sid: add_point_manual(j, f, x, y, pt == "Foreground (Green)", sid),
        inputs=[ann_json_hidden, yoloe_frame_selector, point_x, point_y, point_type, yoloe_semantic_id],
        outputs=[ann_json_hidden, ann_display, ann_status]
    ).then(fn=_refresh_annotation_display,
           inputs=[target_dir_output, ann_json_hidden, yoloe_frame_selector],
           outputs=[yoloe_annotator_image, ann_status])

    add_point_all_btn.click(
        fn=lambda td, j, x, y, pt, sid: add_point_all_frames(td, j, x, y, pt == "Foreground (Green)", sid),
        inputs=[target_dir_output, ann_json_hidden, point_x, point_y, point_type, yoloe_semantic_id],
        outputs=[ann_json_hidden, ann_display, ann_status]
    ).then(fn=_refresh_annotation_display,
           inputs=[target_dir_output, ann_json_hidden, yoloe_frame_selector],
           outputs=[yoloe_annotator_image, ann_status])

    undo_btn.click(
        fn=remove_last_point,
        inputs=[ann_json_hidden, yoloe_frame_selector],
        outputs=[ann_json_hidden, ann_display, ann_status]
    ).then(fn=_refresh_annotation_display,
           inputs=[target_dir_output, ann_json_hidden, yoloe_frame_selector],
           outputs=[yoloe_annotator_image, ann_status])

    clear_frame_btn.click(
        fn=clear_frame_annotations,
        inputs=[ann_json_hidden, yoloe_frame_selector],
        outputs=[ann_json_hidden, ann_display, ann_status]
    ).then(fn=_refresh_annotation_display,
           inputs=[target_dir_output, ann_json_hidden, yoloe_frame_selector],
           outputs=[yoloe_annotator_image, ann_status])

    clear_all_btn.click(
        fn=clear_all_annotations,
        outputs=[ann_json_hidden, ann_display, ann_status]
    ).then(fn=_refresh_annotation_display,
           inputs=[target_dir_output, ann_json_hidden, yoloe_frame_selector],
           outputs=[yoloe_annotator_image, ann_status])

    # Reconstruct button
    submit_btn.click(
        fn=lambda: None, outputs=[reconstruction_output]
    ).then(
        fn=lambda: "Loading and Reconstructing...", outputs=[log_output]
    ).then(
        fn=gradio_reconstruct,
        inputs=[target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg,
                show_cam, mask_sky, prediction_mode,
                semantic_masks_path_state, enable_semantic_cb,
                recon_fuse_3d, recon_fuse_eps, recon_fuse_radius, recon_fuse_min],
        outputs=[reconstruction_output, log_output, frame_filter,
                 depth_gallery, pointmap_gallery,
                 semantic_depth_gallery, semantic_pointmap_gallery, sam_mask_gallery]
    ).then(fn=lambda: "False", outputs=[is_example])

    # Real-time visualization updates
    _viz_inputs = [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg,
                   show_cam, mask_sky, prediction_mode, is_example,
                   semantic_masks_path_state, enable_semantic_cb]
    _viz_outputs = [reconstruction_output, log_output]
    for _comp in [conf_thres, frame_filter, mask_black_bg, mask_white_bg,
                  show_cam, mask_sky, prediction_mode, enable_semantic_cb]:
        _comp.change(fn=gradio_visualize, inputs=_viz_inputs, outputs=_viz_outputs)

    # Download buttons
    def _zip(subdir, name, target_dir):
        if target_dir and target_dir != "None":
            p = os.path.join(target_dir, f"{name}.zip")
            if os.path.exists(p):
                return p
        return None

    depth_dl_btn.click(fn=lambda td: _zip("", "depth_maps", td),
                       inputs=[target_dir_output], outputs=[depth_dl_btn])
    pointmap_dl_btn.click(fn=lambda td: _zip("", "pointmap_maps", td),
                          inputs=[target_dir_output], outputs=[pointmap_dl_btn])
    sem_depth_dl_btn.click(fn=lambda td: _zip("", "semantic_depth_maps", td),
                           inputs=[target_dir_output], outputs=[sem_depth_dl_btn])
    sem_ptmap_dl_btn.click(fn=lambda td: _zip("", "semantic_pointmap_maps", td),
                           inputs=[target_dir_output], outputs=[sem_ptmap_dl_btn])

    # Point Cloud Editing — populate class selector when YOLOe runs
    def _update_class_selector(sem_id_map_json):
        try:
            sem_id_map = json.loads(sem_id_map_json) if sem_id_map_json else {}
        except Exception:
            sem_id_map = {}
        choices = [f"{name} ({sid})" for name, sid in sem_id_map.items()]
        return gr.CheckboxGroup(choices=choices, value=[])

    semantic_id_map_state.change(
        fn=_update_class_selector,
        inputs=[semantic_id_map_state],
        outputs=[edit_class_selector])

    # Apply Edit button
    apply_edit_btn.click(
        fn=run_edit_pointcloud,
        inputs=[target_dir_output, semantic_masks_path_state,
                edit_class_selector, edit_operation,
                conf_thres, frame_filter, mask_black_bg, mask_white_bg,
                show_cam, mask_sky, prediction_mode],
        outputs=[edit_result_viewer, edit_log, edit_dl_btn])

    # Open 3D Elevation Viewer — pure client-side, opens new tab
    open_viewer_btn.click(
        fn=None,
        inputs=[target_dir_output, elev_use_ground_filter],
        outputs=[],
        js=f"""(td, useGroundFilter) => {{
            if (!td || td === 'None') {{ alert('Upload and Reconstruct first.'); return; }}
            const url = '{VGGT_URL}/viewer?session=' + encodeURIComponent(td)
              + '&use_ground_filter=' + encodeURIComponent(Boolean(useGroundFilter));
            window.open(url, '_blank');
        }}"""
    )

    # Standalone GLB overlay page, served by the VGGT static-file endpoint.
    open_fusion_viewer_btn.click(
        fn=None,
        inputs=[],
        outputs=[],
        js=f"""() => window.open('{VGGT_URL}/static/fusion_viewer.html', '_blank')"""
    )


demo.allowed_paths = [WORKSPACE_ROOT]

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=_args.port,
        show_error=True,
        share=False,
        allowed_paths=[WORKSPACE_ROOT],
    )
