"""Gradio-facing client helpers for the formal MapSession workflow."""

from __future__ import annotations

import os
from datetime import datetime

from .gradio_controls import _post, _source_payload


def initialize_session(base_url, workspace_root, source_type, source_uri, backend,
                       target_fps, use_orb, frame_sample_interval, interval, capacity,
                       session_dir, initialization_frames):
    try:
        _post(base_url, "/stream/stop", timeout=12.0)
        path = str(session_dir or "").strip()
        if not path:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(workspace_root, f"map_{stamp}")
        payload = _source_payload(
            source_type, source_uri, backend, target_fps, use_orb, frame_sample_interval
        )
        payload.update({
            "session_dir": path,
            "interval": float(interval),
            "capacity": max(int(capacity), int(initialization_frames)),
            "initialization_frames": int(initialization_frames),
        })
        _post(base_url, "/stream/session/initialize", payload, timeout=20.0)
        return "**状态：** ✅ 正在采集初始化关键帧；完成后将进入 INIT_REVIEW。", path
    except Exception as exc:
        return f"**状态：** ❌ 初始化启动失败：{exc}", str(session_dir or "")


def finalize_session(base_url, session_dir, approved):
    path = str(session_dir or "").strip()
    if not path:
        return "**状态：** ❌ 请先填写或创建 MapSession 目录。"
    try:
        result = _post(base_url, "/stream/session/finalize", {
            "session_dir": path, "approved": bool(approved),
            "note": "approved in Gradio" if approved else "rejected in Gradio",
        })
        return f"**状态：** 会话已进入 {result.get('state')}，map_version={result.get('map_version')}。"
    except Exception as exc:
        return f"**状态：** ❌ 审核失败：{exc}"


def start_session_update(base_url, source_type, source_uri, backend, target_fps,
                         use_orb, frame_sample_interval, interval, min_frames,
                         capacity, session_dir, file_out):
    path = str(session_dir or "").strip()
    if not path:
        return "**状态：** ❌ 请先填写 READY MapSession 目录。", str(file_out or "")
    try:
        _post(base_url, "/stream/stop", timeout=12.0)
        output = str(file_out or "").strip() or os.path.join(path, "tiles")
        payload = _source_payload(
            source_type, source_uri, backend, target_fps, use_orb, frame_sample_interval
        )
        payload.update({
            "session_dir": path, "file_out": output,
            "interval": float(interval), "min_frames": int(min_frames),
            "capacity": int(capacity), "run_reconstruction": True,
        })
        _post(base_url, "/stream/session/update", payload, timeout=20.0)
        return "**状态：** ✅ 可信增量更新已启动；未通过门控的结果不会发布。", output
    except Exception as exc:
        return f"**状态：** ❌ 增量更新启动失败：{exc}", str(file_out or "")
