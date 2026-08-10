"""Reliable live-camera sources for the streaming reconstruction package.

This module is intentionally independent from FastAPI, Gradio, torch, and VGGT.  It
continuously drains a USB/V4L2 or RTSP stream, keeps only the newest preview frame, and
emits frames at a modest rate for the existing keyframe buffer.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from .frame_source import FrameSource


@dataclass
class CameraStatus:
    connected: bool = False
    source: str = ""
    width: int = 0
    height: int = 0
    native_fps: float = 0.0
    capture_fps: float = 0.0
    decoded: int = 0
    emitted: int = 0
    reconnects: int = 0
    read_errors: int = 0
    last_frame_age_ms: float | None = None
    last_error: str = ""


def safe_source_label(source: str | int) -> str:
    """Return a status-safe source label with URL credentials removed."""
    value = str(source)
    if "://" not in value:
        return value
    parts = urlsplit(value)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, ""))


class OpenCvCameraSource(FrameSource):
    """USB/V4L2 or RTSP source with latest-frame semantics and reconnects.

    The camera is decoded continuously, preventing a slow reconstruction pass from
    building an old-frame backlog.  Every decoded image replaces the preview frame, while
    only ``target_fps`` images per second are yielded to the ORB keyframe selector.
    """

    _BACKENDS = {
        "auto": None,
        "ffmpeg": getattr(cv2, "CAP_FFMPEG", None),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", None),
        "v4l2": getattr(cv2, "CAP_V4L2", None),
    }

    def __init__(
        self,
        source: str | int,
        *,
        target_fps: float = 3.0,
        backend: str = "auto",
        reconnect: bool = True,
        reconnect_initial: float = 0.5,
        reconnect_max: float = 8.0,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 3000,
        width: int | None = None,
        height: int | None = None,
    ):
        if backend not in self._BACKENDS:
            raise ValueError(f"unsupported camera backend: {backend}")
        if backend != "auto" and self._BACKENDS[backend] is None:
            raise ValueError(f"OpenCV backend is unavailable: {backend}")
        self.source = source
        self.path = safe_source_label(source)
        self.target_fps = max(0.1, float(target_fps))
        self.backend = backend
        self.reconnect = bool(reconnect)
        self.reconnect_initial = max(0.1, float(reconnect_initial))
        self.reconnect_max = max(self.reconnect_initial, float(reconnect_max))
        self.open_timeout_ms = max(0, int(open_timeout_ms))
        self.read_timeout_ms = max(0, int(read_timeout_ms))
        self.request_width = int(width) if width else None
        self.request_height = int(height) if height else None

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._cap = None
        self._latest: np.ndarray | None = None
        self._last_frame_monotonic: float | None = None
        self._status = CameraStatus(source=self.path)
        self._fps_started = time.monotonic()
        self._fps_decoded = 0

    def _open(self):
        backend_id = self._BACKENDS[self.backend]
        params = []
        if self.backend in {"ffmpeg", "gstreamer"}:
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms])
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms])
        if backend_id is None:
            cap = cv2.VideoCapture(self.source)
        elif params:
            # Open/read timeouts are open-only properties for FFmpeg and GStreamer.
            cap = cv2.VideoCapture(self.source, backend_id, params)
        else:
            cap = cv2.VideoCapture(self.source, backend_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.request_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.request_width)
        if self.request_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.request_height)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open camera source {self.path}")
        with self._lock:
            self._cap = cap
            self._status.connected = True
            self._status.native_fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2)
            self._status.last_error = ""
        return cap

    def _mark_disconnected(self, error: str = "") -> None:
        with self._lock:
            self._status.connected = False
            if error:
                self._status.last_error = error
            cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    def frames(self) -> Iterator[np.ndarray]:
        self._stop.clear()
        delay = self.reconnect_initial
        next_emit = time.monotonic()
        while not self._stop.is_set():
            try:
                cap = self._open()
                delay = self.reconnect_initial
                while not self._stop.is_set():
                    ok, frame_bgr = cap.read()
                    if not ok:
                        with self._lock:
                            self._status.read_errors += 1
                        raise RuntimeError("camera read failed")

                    now = time.monotonic()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    self._fps_decoded += 1
                    elapsed = now - self._fps_started
                    with self._lock:
                        self._latest = frame_rgb
                        self._last_frame_monotonic = now
                        self._status.decoded += 1
                        self._status.height, self._status.width = frame_rgb.shape[:2]
                        if elapsed >= 1.0:
                            self._status.capture_fps = round(self._fps_decoded / elapsed, 2)
                            self._fps_started = now
                            self._fps_decoded = 0

                    if now >= next_emit:
                        next_emit = now + 1.0 / self.target_fps
                        with self._lock:
                            self._status.emitted += 1
                        yield frame_rgb.copy()
            except GeneratorExit:
                self._mark_disconnected()
                raise
            except Exception as exc:
                self._mark_disconnected(str(exc))
                if self._stop.is_set() or not self.reconnect:
                    break
                with self._lock:
                    self._status.reconnects += 1
                if self._stop.wait(delay):
                    break
                delay = min(self.reconnect_max, delay * 2.0)
        self._mark_disconnected()

    def close(self) -> None:
        """Request stop; the reader thread owns and releases VideoCapture."""
        self._stop.set()

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def status(self) -> dict:
        with self._lock:
            data = asdict(self._status)
            if self._last_frame_monotonic is not None:
                data["last_frame_age_ms"] = round(
                    (time.monotonic() - self._last_frame_monotonic) * 1000.0, 1
                )
            return data


def normalize_mjpeg_url(source_uri: str | None) -> str:
    """Accept an IP Webcam root URL or an explicit HTTP MJPEG endpoint."""
    value = (source_uri or "").strip()
    if not value:
        raise ValueError("source_uri is required for an HTTP MJPEG camera")
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError("HTTP camera source_uri must be a full http:// or https:// URL")
    path = parts.path or "/"
    if path == "/":
        path = "/video"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def parse_camera_source(source_type: str, source_uri: str | None) -> str | int:
    """Normalize a public API source value without guessing RTSP credentials."""
    if source_type == "usb":
        value = (source_uri or "0").strip()
        return int(value) if value.lstrip("-").isdigit() else value
    if source_type in {"http", "mjpeg"}:
        return normalize_mjpeg_url(source_uri)
    if source_type == "rtsp":
        value = (source_uri or "").strip()
        if not value:
            raise ValueError("source_uri is required for an RTSP camera")
        if not value.lower().startswith("rtsp://"):
            raise ValueError("RTSP source_uri must start with rtsp://")
        return value
    raise ValueError(f"unsupported live source_type: {source_type}")

