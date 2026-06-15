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

# ── CLI args (parsed before Gradio builds the UI) ────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--yoloe-url", default="http://localhost:8001")
_parser.add_argument("--vggt-url",  default="http://localhost:8002")
_parser.add_argument("--port",      type=int, default=7860)
_args, _ = _parser.parse_known_args()

YOLOE_URL = _args.yoloe_url
VGGT_URL  = _args.vggt_url

WORKSPACE_ROOT = "/home/maomaoyu/WS/vggt_yoloe/workspaces"
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

def handle_uploads(input_video, input_images, frame_interval_sec=1.0, max_frames=0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(WORKSPACE_ROOT, f"session_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images)

    image_paths = []

    if input_images:
        for file_data in input_images:
            fp = file_data["name"] if isinstance(file_data, dict) else file_data
            dst = os.path.join(target_dir_images, os.path.basename(fp))
            shutil.copy(fp, dst)
            image_paths.append(dst)

    if input_video:
        vp = input_video["name"] if isinstance(input_video, dict) else input_video
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
                   conf_threshold, iou_threshold):
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
                     model_id, image_size, conf_threshold, iou_threshold):
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


def run_yoloe_promptfree(target_dir, model_id, image_size, conf_threshold, iou_threshold, max_classes):
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
                        semantic_masks_path, enable_semantic):
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
        }, timeout=600)
    except Exception as e:
        return None, f"VGGT error: {e}", None, [], [], [], [], []

    choices = result.get("frame_filter_choices", ["All"])
    dd = gr.Dropdown(choices=choices, value=frame_filter or "All", interactive=True)
    return (
        result["glb_path"],
        result["log"],
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


def run_fit_elevation(target_dir, source_glb_path, grid_resolution,
                      colormap, ground_percentile, use_ground_filter, use_ransac,
                      conf_thres, prediction_mode):
    if not target_dir or not os.path.isdir(target_dir):
        return None, None, "**Status:** No reconstruction available. Upload and Reconstruct first."
    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, None, "**Status:** No saved predictions. Run Reconstruct first."
    try:
        result = _post(f"{VGGT_URL}/fit_elevation", {
            "working_dir": target_dir,
            "source_glb_path": source_glb_path or "",
            "grid_resolution": int(grid_resolution),
            "colormap": colormap,
            "ground_percentile": float(ground_percentile),
            "use_ground_filter": bool(use_ground_filter),
            "use_ransac": bool(use_ransac),
            "conf_thres": float(conf_thres),
            "prediction_mode": prediction_mode,
        }, timeout=120)
        return result["elev_only_path"], result["merged_path"], f"**Status:** {result['log']}"
    except Exception as e:
        return None, None, f"**Status:** Elevation fitting error — {e}"


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
    refresh_status_btn.click(fn=check_services, outputs=[service_status])

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

                # Elevation Plane
                with gr.TabItem("Elevation Plane"):
                    gr.Markdown("## Elevation Plane Fitting")
                    gr.Markdown(
                        "Fit a DEM (Digital Elevation Model) to the reconstructed point cloud. "
                        "You can interpolate from filtered ground candidates or from all aligned "
                        "points before filtering, then export a colored elevation grid. "
                        "**Requires:** Run Reconstruct first."
                    )
                    with gr.Row():
                        # Left: elevation-only viewer
                        with gr.Column(scale=3):
                            elev_only_viewer = gr.Model3D(
                                height=420, zoom_speed=0.5, pan_speed=0.5,
                                label="Elevation Mesh Only")
                            elev_only_dl_btn = gr.DownloadButton(
                                "Download Elevation Mesh (.glb)", variant="secondary")

                        # Right: merged viewer
                        with gr.Column(scale=3):
                            elev_merged_viewer = gr.Model3D(
                                height=420, zoom_speed=0.5, pan_speed=0.5,
                                label="Point Cloud + Elevation Mesh")
                            elev_merged_dl_btn = gr.DownloadButton(
                                "Download Merged Scene (.glb)", variant="secondary")

                    elev_log = gr.Markdown("**Status:** Run Reconstruct first, then click Fit Elevation.")

                    with gr.Row():
                        with gr.Column(scale=2):
                            elev_source_glb = gr.Textbox(
                                label="Source GLB path (leave blank to use current reconstruction)",
                                placeholder="Optional: paste a .glb path to merge with",
                                value="")
                            elev_grid_res = gr.Slider(
                                minimum=32, maximum=512, value=128, step=32,
                                label="Grid Resolution (N×N)",
                                info="Higher = finer DEM, slower computation")
                            elev_colormap = gr.Dropdown(
                                choices=["terrain", "viridis", "plasma", "inferno",
                                         "magma", "cividis", "RdYlGn", "coolwarm"],
                                value="terrain",
                                label="Colormap")

                        with gr.Column(scale=1):
                            elev_ground_pct = gr.Slider(
                                minimum=5, maximum=50, value=20, step=5,
                                label="Ground Percentile (%)",
                                info="Use lowest N% of points as ground candidates")
                            elev_use_ground_filter = gr.Checkbox(
                                label="Use ground filter before DEM fitting",
                                value=True,
                                info="On: interpolate from filtered ground candidates. Off: interpolate from all aligned points.")
                            elev_use_ransac = gr.Checkbox(
                                label="Refine with RANSAC", value=True,
                                info="Fit a plane to ground candidates for cleaner results")
                            fit_elevation_btn = gr.Button(
                                "▶ Fit Elevation", variant="primary", size="lg")
                            open_viewer_btn = gr.Button(
                                "🌐 Open 3D Elevation Viewer", variant="secondary", size="lg")

    # ── Examples ──────────────────────────────────────────────────────────────
    EXAMPLES_DIR = "/home/maomaoyu/WS/vggt/examples/videos"
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
                             model_id, img_size, conf, iou, max_cls):
        if mode == "Text":
            masks_path, previews, sem_map, status = run_yoloe_text(
                target_dir, text_prompts, model_id, img_size, conf, iou)
        elif mode == "Visual":
            masks_path, previews, sem_map, status = run_yoloe_visual(
                target_dir, ref_img, bboxes_json, model_id, img_size, conf, iou)
        else:
            masks_path, previews, sem_map, status = run_yoloe_promptfree(
                target_dir, model_id, img_size, conf, iou, max_cls)
        return masks_path or "", previews, status, json.dumps(sem_map)

    run_yoloe_btn.click(
        fn=_run_yoloe_dispatch,
        inputs=[target_dir_output, yoloe_prompt_mode, yoloe_text_prompts,
                yoloe_ref_image, yoloe_bboxes_json,
                yoloe_model_id, yoloe_image_size, yoloe_conf, yoloe_iou,
                yoloe_max_classes],
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
                semantic_masks_path_state, enable_semantic_cb],
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

    # Fit Elevation button
    fit_elevation_btn.click(
        fn=lambda: "**Status:** Fitting elevation plane...",
        outputs=[elev_log]
    ).then(
        fn=run_fit_elevation,
        inputs=[target_dir_output, elev_source_glb, elev_grid_res,
                elev_colormap, elev_ground_pct, elev_use_ground_filter, elev_use_ransac,
                conf_thres, prediction_mode],
        outputs=[elev_only_viewer, elev_merged_viewer, elev_log]
    ).then(
        fn=lambda p: p,
        inputs=[elev_only_viewer],
        outputs=[elev_only_dl_btn]
    ).then(
        fn=lambda p: p,
        inputs=[elev_merged_viewer],
        outputs=[elev_merged_dl_btn]
    )

    # Open 3D Elevation Viewer — pure client-side, opens new tab
    open_viewer_btn.click(
        fn=None,
        inputs=[target_dir_output, elev_use_ground_filter],
        outputs=[],
        js="""(td, useGroundFilter) => {
            if (!td || td === 'None') { alert('Upload and Reconstruct first.'); return; }
            const url = 'http://localhost:8002/viewer?session=' + encodeURIComponent(td)
              + '&use_ground_filter=' + encodeURIComponent(Boolean(useGroundFilter));
            window.open(url, '_blank');
        }"""
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
