"""FrameSource construction kept separate from API and reconstruction policy."""

from __future__ import annotations

from .camera_source import OpenCvCameraSource, parse_camera_source
from .frame_source import FrameSource, VideoFileSource


def build_frame_source(
    *,
    source_type: str,
    source_uri: str | None,
    video_path: str | None,
    target_fps: float,
    loop_video: bool = True,
    backend: str = "auto",
    reconnect: bool = True,
    open_timeout_ms: int = 5000,
    read_timeout_ms: int = 3000,
    width: int | None = None,
    height: int | None = None,
) -> FrameSource:
    """Create a source while preserving the legacy ``video_path`` contract."""
    source_type = (source_type or "video").strip().lower()
    if source_type == "video":
        path = video_path or source_uri
        if not path:
            raise ValueError("video_path (or source_uri) is required for a video source")
        return VideoFileSource(path, target_fps=target_fps, loop=loop_video)
    if source_type in {"usb", "rtsp", "http", "mjpeg"}:
        source = parse_camera_source(source_type, source_uri)
        resolved_backend = backend
        if backend == "auto":
            resolved_backend = "v4l2" if source_type == "usb" else "ffmpeg"
        return OpenCvCameraSource(
            source,
            target_fps=target_fps,
            backend=resolved_backend,
            reconnect=reconnect,
            open_timeout_ms=open_timeout_ms,
            read_timeout_ms=read_timeout_ms,
            width=width,
            height=height,
        )
    raise ValueError("source_type must be one of: video, usb, rtsp, http")


def source_diagnostics(source: FrameSource) -> dict:
    status_fn = getattr(source, "status", None)
    if callable(status_fn):
        return status_fn()
    return {
        "connected": True,
        "source": str(getattr(source, "path", type(source).__name__)),
    }

