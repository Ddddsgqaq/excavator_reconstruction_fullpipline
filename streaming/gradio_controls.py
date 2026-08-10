"""Thin Gradio-facing HTTP client for the streaming service."""

from __future__ import annotations

import io
import os
from datetime import datetime

import requests
from PIL import Image


def _post(base_url: str, path: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
    response = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(detail or f"HTTP {response.status_code}")
    return response.json()


def _source_payload(source_type, source_uri, backend, target_fps,
                    use_orb=True, frame_sample_interval=None) -> dict:
    payload = {
        "source_type": str(source_type).lower(),
        "source_uri": str(source_uri or "").strip(),
        "camera_backend": str(backend).lower(),
        "target_fps": float(target_fps),
        "use_orb": bool(use_orb),
    }
    if frame_sample_interval is not None:
        payload["frame_sample_interval"] = float(frame_sample_interval)
    return payload


def connect_camera(base_url, source_type, source_uri, backend, target_fps,
                   use_orb=True, frame_sample_interval=None) -> str:
    """Replace any prior stream with a capture-only diagnostic session."""
    try:
        _post(base_url, "/stream/stop", timeout=12.0)
        payload = _source_payload(
            source_type, source_uri, backend, target_fps, use_orb, frame_sample_interval
        )
        payload["run_reconstruction"] = False
        result = _post(base_url, "/stream/start", payload)
        return f"**状态：** 已启动取流预览（{result.get('mode', 'capture_only')}），等待首帧…"
    except Exception as exc:
        return f"**状态：** ❌ 连接失败：{exc}"


def start_live_reconstruction(
    base_url, workspace_root, source_type, source_uri, backend, target_fps,
    interval, min_frames, capacity, file_out, save_glb=False,
    use_orb=True, frame_sample_interval=None, fusion=True, scale_factor=28.0,
) -> tuple[str, str]:
    """Start the existing rolling reconstruction as a camera-to-VGGT check."""
    try:
        _post(base_url, "/stream/stop", timeout=12.0)
        output = str(file_out or "").strip()
        if not output:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output = os.path.join(workspace_root, f"live_{stamp}", "tiles")
        payload = _source_payload(
            source_type, source_uri, backend, target_fps, use_orb, frame_sample_interval
        )
        payload.update({
            "run_reconstruction": True,
            "interval": float(interval),
            "min_frames": int(min_frames),
            "capacity": int(capacity),
            "file_out": output,
            "save_glb": bool(save_glb),
            "freeze_anchor": True,
            "registration": True,
            "fusion": bool(fusion),
            "scale_factor": float(scale_factor),
        })
        _post(base_url, "/stream/start", payload)
        mode = "跨轮融合" if fusion else "单轮覆盖"
        msg = f"**状态：** ✅ 相机到 VGGT 滑窗重建已启动（{mode}）。"
        if save_glb:
            glb_dir = os.path.join(os.path.dirname(output.rstrip("/")), "pointclouds")
            msg += f"（点云 GLB 保存至 `{glb_dir}`）"
        return msg, output
    except Exception as exc:
        return f"**状态：** ❌ 启动重建失败：{exc}", str(file_out or "")


def stop_camera(base_url: str) -> str:
    try:
        result = _post(base_url, "/stream/stop", timeout=12.0)
        return f"**状态：** {result.get('status', 'stopped')}"
    except Exception as exc:
        return f"**状态：** ❌ 停止失败：{exc}"


def poll_camera(base_url: str):
    """Fetch status and at most one JPEG; never opens a camera in Gradio."""
    try:
        response = requests.get(f"{base_url}/stream/status", timeout=3.0)
        response.raise_for_status()
        status = response.json()
    except Exception as exc:
        return None, f"**监控：** VGGT 服务不可达：{exc}"

    session = status.get("session") or {}
    if not status.get("running"):
        if session:
            report = session.get("last_change_report") or session.get("quality_report") or {}
            reasons = report.get("reasons") or []
            detail = f" · gate={'; '.join(reasons[:2])}" if reasons else ""
            return None, (
                f"**监控：** session={session.get('state')} · "
                f"map_version={session.get('map_version', 0)}{detail}"
            )
        return None, "**监控：** 未运行"
    src = status.get("source_status") or {}
    summary = (
        f"**监控：** mode={status.get('mode')} · connected={src.get('connected', False)} · "
        f"capture={src.get('capture_fps', 0)} FPS · age={src.get('last_frame_age_ms', '-')} ms · "
        f"frames={src.get('decoded', status.get('offered', 0))} · keyframes={status.get('kept', 0)} · "
        f"采样={status.get('keyframe_mode', '-')} / {status.get('frame_sample_interval', 0):.2f}s · "
        f"ORB最近/平均={status.get('orb_last_ms', 0):.1f}/{status.get('orb_avg_ms', 0):.1f}ms · "
        f"passes={status.get('passes', 0)} · reconnects={src.get('reconnects', 0)}"
    )
    if session:
        summary += (
            f" · session={session.get('state')} · map_version={session.get('map_version')}"
        )
    try:
        frame_response = requests.get(f"{base_url}/stream/frame.jpg", timeout=3.0)
        frame_response.raise_for_status()
        image = Image.open(io.BytesIO(frame_response.content)).convert("RGB")
        return image, summary
    except Exception:
        return None, summary + " · 等待首帧"
