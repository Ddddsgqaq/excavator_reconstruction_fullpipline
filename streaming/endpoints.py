"""FastAPI routes for live capture and rolling streaming reconstruction.

The routes remain additive to ``vggt_service``.  Legacy callers may continue posting only
``video_path``; new callers can select USB/RTSP and may run capture-only diagnostics before
allowing any VGGT work.
"""

from __future__ import annotations

import os
import threading

import cv2
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from .elevation_publisher import ElevationPublisher, NullPublisher
from .reconstruct_loop import LoopConfig, ReconstructLoop
from .source_factory import build_frame_source, source_diagnostics

router = APIRouter(prefix="/stream", tags=["streaming"])

_loop: ReconstructLoop | None = None
_loop_lock = threading.Lock()
_last_terrain_snapshot: dict = {
    "sequence": 0,
    "published_at": None,
    "tile_count": 0,
    "tiles": [],
}


class StreamStartRequest(BaseModel):
    # Backward-compatible source contract. Existing video_path-only clients still work.
    source_type: str = "video"               # video | usb | rtsp | http (MJPEG)
    source_uri: str | None = None             # USB index/path or full RTSP URL
    video_path: str | None = None
    camera_backend: str = "auto"              # auto | ffmpeg | gstreamer | v4l2
    reconnect: bool = True
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 3000
    camera_width: int | None = None
    camera_height: int | None = None
    run_reconstruction: bool = True            # False = stable-capture/preview diagnostic

    file_out: str | None = None
    save_glb: bool = False                     # also save each pass's raw point cloud as .glb
    glb_dir: str | None = None                 # defaults to a "pointclouds" sibling of file_out
    mqtt: bool = False
    broker: str = "127.0.0.1"
    port: int = 1883
    loop_video: bool = True

    interval: float = 6.0
    min_frames: int = 4
    capacity: int = 12
    sim_thresh: float = 0.92
    target_fps: float = 3.0
    use_orb: bool = True
    frame_sample_interval: float | None = None
    grid_resolution: int = 128
    scale_factor: float = 28.0
    height_resolution: float = 0.01
    tile_x: int = 0
    tile_y: int = 0
    tile_size_meters: float | None = None
    freeze_anchor: bool = True
    registration: bool = True
    fusion: bool = False
    world_size_m: float = 150.0
    tile_size_m: float = 50.0
    fusion_decay: float = 0.5
    change_thresh: float = 0.05
    top_percentile: float = 70.0
    auto_fusion_extent: bool = False
    fusion_extent_margin: float = 1.25
    max_registration_rmse_m: float = 1.0
    max_changed_fraction: float = 0.35
    min_change_neighbors: int = 3


def _build_loop(req: StreamStartRequest, *, publisher_override=None,
                loop_class=ReconstructLoop, loop_kwargs=None) -> ReconstructLoop:
    if (publisher_override is None and req.run_reconstruction
            and not req.file_out and not req.mqtt):
        raise ValueError("reconstruction needs at least one channel: file_out and/or mqtt")
    if req.target_fps <= 0 or req.interval <= 0:
        raise ValueError("target_fps and interval must be positive")
    if req.frame_sample_interval is not None and req.frame_sample_interval <= 0:
        raise ValueError("frame_sample_interval must be positive")
    if req.min_frames < 1 or req.capacity < req.min_frames:
        raise ValueError("capacity must be >= min_frames >= 1")

    # Diagnostic point-cloud dumps default to a "pointclouds" sibling of the tiles dir so
    # each live workspace keeps its GLBs next to the DEM tiles it produced.
    glb_dir = req.glb_dir
    if req.save_glb and not glb_dir:
        if req.file_out:
            glb_dir = os.path.join(os.path.dirname(req.file_out.rstrip("/")), "pointclouds")
        else:
            raise ValueError("save_glb needs file_out or an explicit glb_dir")

    effective_fps = (
        1.0 / req.frame_sample_interval
        if not req.use_orb and req.frame_sample_interval is not None
        else req.target_fps
    )
    cfg = LoopConfig(
        interval=req.interval,
        min_frames=req.min_frames,
        capacity=req.capacity,
        sim_thresh=req.sim_thresh,
        target_fps=effective_fps,
        use_orb=req.use_orb,
        frame_sample_interval=1.0 / effective_fps,
        capture_only=not req.run_reconstruction,
        grid_resolution=req.grid_resolution,
        scale_factor=req.scale_factor,
        height_resolution=req.height_resolution,
        tile_x=req.tile_x,
        tile_y=req.tile_y,
        tile_size_meters=req.tile_size_meters,
        freeze_anchor=req.freeze_anchor,
        register=req.registration,
        fusion=req.fusion,
        world_size_m=req.world_size_m,
        tile_size_m=req.tile_size_m,
        fusion_decay=req.fusion_decay,
        change_thresh=req.change_thresh,
        top_percentile=req.top_percentile,
        auto_fusion_extent=req.auto_fusion_extent,
        fusion_extent_margin=req.fusion_extent_margin,
        max_registration_rmse_m=req.max_registration_rmse_m,
        max_changed_fraction=req.max_changed_fraction,
        min_change_neighbors=req.min_change_neighbors,
        save_glb=req.save_glb,
        glb_dir=glb_dir,
    )
    source = build_frame_source(
        source_type=req.source_type,
        source_uri=req.source_uri,
        video_path=req.video_path,
        target_fps=effective_fps,
        loop_video=req.loop_video,
        backend=req.camera_backend,
        reconnect=req.reconnect,
        open_timeout_ms=req.open_timeout_ms,
        read_timeout_ms=req.read_timeout_ms,
        width=req.camera_width,
        height=req.camera_height,
    )
    if publisher_override is not None:
        publisher = publisher_override
    elif req.run_reconstruction:
        publisher = ElevationPublisher(
            file_out=req.file_out,
            mqtt=req.mqtt,
            broker=req.broker,
            port=req.port,
            archive_existing_tiles=bool(req.file_out),
        )
    else:
        publisher = NullPublisher()
    return loop_class(source, publisher, cfg, **(loop_kwargs or {}))


@router.post("/start")
def stream_start(req: StreamStartRequest):
    global _loop, _last_terrain_snapshot
    with _loop_lock:
        if _loop is not None and _loop.status().running:
            raise HTTPException(status_code=409, detail="a stream is already running; /stream/stop first")
        try:
            _loop = _build_loop(req)
            _loop.start()
            _last_terrain_snapshot = {
                "sequence": 0,
                "published_at": None,
                "tile_count": 0,
                "tiles": [],
            }
        except Exception as exc:
            _loop = None
            raise HTTPException(status_code=400, detail=f"failed to start stream: {exc}") from exc
        return {
            "status": "started",
            "mode": "reconstruction" if req.run_reconstruction else "capture_only",
            "channels": _loop.publisher.channels,
        }


@router.post("/stop")
def stream_stop():
    global _loop, _last_terrain_snapshot
    # Detach the loop under the lock, then release it BEFORE the (potentially
    # multi-second) thread join. Holding _loop_lock across stop() would block every
    # concurrent /stream/status poll (the Gradio 2 s Timer) for the whole join window,
    # which reads as a frozen live panel.
    with _loop_lock:
        loop, _loop = _loop, None
        if loop is not None:
            snapshot_fn = getattr(loop.publisher, "snapshot", None)
            if callable(snapshot_fn):
                _last_terrain_snapshot = snapshot_fn()
    if loop is None:
        return {"status": "not_running"}
    loop.stop()
    return {"status": "stopped"}


@router.get("/status")
def stream_status():
    with _loop_lock:
        if _loop is None:
            return {"running": False, "mode": "idle", "source_status": {"connected": False}}
        s = _loop.status()
        session_fn = getattr(_loop, "session_summary", None)
        session = session_fn() if callable(session_fn) else None
        mode = getattr(_loop, "session_mode", None)
        return {
            "running": s.running,
            "mode": mode or ("capture_only" if _loop.cfg.capture_only else "reconstruction"),
            "source": s.source,
            "source_status": source_diagnostics(_loop.source),
            "channels": s.channels,
            "file_out": getattr(_loop.publisher, "file_out", None),
            "window": s.window,
            "offered": s.offered,
            "kept": s.kept,
            "passes": s.passes,
            "published": s.published,
            "last_pass_seconds": s.last_pass_seconds,
            "last_gravity_source": s.last_gravity_source,
            "last_elev_range_m": s.last_elev_range_m,
            "anchor_frozen": s.anchor_frozen,
            "last_registered": s.last_registered,
            "last_reg_rmse": s.last_reg_rmse,
            "last_reg_yaw_deg": s.last_reg_yaw_deg,
            "fusion_enabled": s.fusion_enabled,
            "observed_cells": s.observed_cells,
            "session": session,
            "tiles_published_total": s.tiles_published_total,
            "last_changed_tiles": s.last_changed_tiles,
            "last_change_report": s.last_change_report,
            "keyframe_mode": s.keyframe_mode,
            "frame_sample_interval": s.frame_sample_interval,
            "orb_calls": s.orb_calls,
            "orb_total_seconds": s.orb_total_seconds,
            "orb_last_ms": s.orb_last_ms,
            "orb_avg_ms": s.orb_avg_ms,
            "last_error": s.last_error,
        }


@router.get("/terrain")
def stream_terrain():
    """Latest exact ElevationMsg payload(s) published to Unity, for diagnostics only."""
    with _loop_lock:
        loop = _loop
        if loop is None:
            snapshot = _last_terrain_snapshot
            running = False
        else:
            snapshot_fn = getattr(loop.publisher, "snapshot", None)
            snapshot = snapshot_fn() if callable(snapshot_fn) else _last_terrain_snapshot
            running = loop.status().running
    return {
        "available": bool(snapshot.get("tiles")),
        "running": running,
        **snapshot,
    }


@router.get("/viewer", include_in_schema=False)
def stream_viewer():
    """Serve the live viewer without browser caching so UI changes appear immediately."""
    viewer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "stream_terrain_viewer.html")
    return FileResponse(
        viewer_path,
        media_type="text/html",
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
    )


@router.get("/frame.jpg")
def stream_frame():
    """Return the newest source frame for monitoring; this never touches VGGT."""
    with _loop_lock:
        if _loop is None:
            raise HTTPException(status_code=404, detail="no active stream")
        latest_fn = getattr(_loop.source, "latest_frame", None)
        frame_rgb = latest_fn() if callable(latest_fn) else None
    if frame_rgb is None:
        raise HTTPException(status_code=404, detail="no frame received yet")
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode preview frame")
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.get("/pass")
def stream_pass():
    """Diagnostics for the latest pass: keyframes, full RGB cloud (downsampled),
    ground-point count and DEM footprint. Read-only; never triggers reconstruction."""
    with _loop_lock:
        loop = _loop
        running = loop is not None and loop.status().running
        diag_fn = getattr(loop, "pass_diagnostics", None) if loop is not None else None
        diag = diag_fn() if callable(diag_fn) else None
    if diag is None:
        return {"available": False, "running": running}
    return {"available": True, "running": running, **diag}


@router.get("/pass/frame/{idx}.jpg")
def stream_pass_frame(idx: int):
    """Return the idx-th keyframe that fed the most recent reconstruction pass."""
    with _loop_lock:
        loop = _loop
        frame_fn = getattr(loop, "pass_frame", None) if loop is not None else None
        frame_rgb = frame_fn(idx) if callable(frame_fn) else None
    if frame_rgb is None:
        raise HTTPException(status_code=404, detail="no such keyframe for the last pass")
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode keyframe")
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )

from .session_api import router as _session_router
router.include_router(_session_router)
