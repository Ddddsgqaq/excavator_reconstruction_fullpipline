"""FastAPI routes for the opt-in persistent two-stage mapping workflow."""

from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import endpoints as runtime
from .change_detection import ChangePolicy
from .elevation_publisher import NullPublisher
from .initialization import InitializationPolicy, finalize_initialization
from .map_session import MapSession, SessionState
from .two_stage_loop import TwoStageReconstructLoop


router = APIRouter(prefix="/session", tags=["map-session"])


class SessionInitializeRequest(runtime.StreamStartRequest):
    session_dir: str
    initialization_frames: int = 12
    min_initial_points: int = 1000
    min_initial_coverage: float = 0.15
    max_initial_elevation_span_m: float = 100.0


class SessionUpdateRequest(runtime.StreamStartRequest):
    session_dir: str
    max_registration_rmse_m: float = 1.0
    min_update_coverage: float = 0.05
    max_changed_fraction: float = 0.35
    max_abs_height_change_m: float = 5.0
    degraded_after_rejections: int = 2
    reinit_after_rejections: int = 5


class SessionFinalizeRequest(BaseModel):
    session_dir: str
    approved: bool
    note: str = ""


def _ensure_idle():
    if runtime._loop is not None and runtime._loop.status().running:
        raise HTTPException(status_code=409, detail="a stream is already running; /stream/stop first")


@router.post("/initialize")
def initialize_session(req: SessionInitializeRequest):
    """Collect one controlled batch and stop in INIT_REVIEW for human approval."""
    with runtime._loop_lock:
        _ensure_idle()
        session = None
        try:
            if req.initialization_frames < 3:
                raise ValueError("initialization_frames must be >= 3")
            req.run_reconstruction = True
            req.min_frames = req.initialization_frames
            req.capacity = max(req.capacity, req.initialization_frames)
            config = {
                "grid_resolution": req.grid_resolution,
                "world_size_m": req.world_size_m,
                "tile_size_m": req.tile_size_m,
            }
            manifest_path = os.path.join(req.session_dir, MapSession.MANIFEST)
            if os.path.isfile(manifest_path):
                session = MapSession.load(req.session_dir)
                if session.state not in {SessionState.REINIT_REQUIRED, SessionState.READY}:
                    raise ValueError(f"reinitialization requires REINIT_REQUIRED or READY, got {session.state.value}")
                session.manifest.config = config
            else:
                session = MapSession.create(req.session_dir, config=config)
            session.transition(SessionState.INITIALIZING)
            policy = InitializationPolicy(
                min_frames=req.initialization_frames,
                min_points=req.min_initial_points,
                min_coverage=req.min_initial_coverage,
                max_elevation_span_m=req.max_initial_elevation_span_m,
            )
            runtime._loop = runtime._build_loop(
                req, publisher_override=NullPublisher(), loop_class=TwoStageReconstructLoop,
                loop_kwargs={
                    "session": session, "mode": "initialization",
                    "initialization_policy": policy,
                },
            )
            runtime._loop.start()
            return {"status": "started", "mode": "initialization", "session": session.summary()}
        except HTTPException:
            raise
        except Exception as exc:
            runtime._loop = None
            if session is not None and session.state == SessionState.INITIALIZING:
                session.transition(SessionState.REINIT_REQUIRED)
            raise HTTPException(status_code=400, detail=f"failed to initialize session: {exc}") from exc


@router.post("/finalize")
def finalize_session(req: SessionFinalizeRequest):
    try:
        return finalize_initialization(
            MapSession.load(req.session_dir), approved=req.approved, note=req.note
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to finalize session: {exc}") from exc


@router.post("/update")
def update_session(req: SessionUpdateRequest):
    """Load a reviewed READY map and publish only observations passing all trust gates."""
    with runtime._loop_lock:
        _ensure_idle()
        session = None
        try:
            session = MapSession.load(req.session_dir)
            if session.state not in {SessionState.READY, SessionState.DEGRADED}:
                raise ValueError(f"incremental update requires READY or DEGRADED, got {session.state.value}")
            if req.reinit_after_rejections <= req.degraded_after_rejections:
                raise ValueError("reinit_after_rejections must be greater than degraded_after_rejections")
            req.run_reconstruction = True
            req.freeze_anchor = True
            req.registration = True
            req.fusion = True
            session.transition(SessionState.UPDATING)
            policy = ChangePolicy(
                max_registration_rmse_m=req.max_registration_rmse_m,
                min_coverage=req.min_update_coverage,
                max_changed_fraction=req.max_changed_fraction,
                min_change_m=req.change_thresh,
                max_abs_height_change_m=req.max_abs_height_change_m,
                degraded_after_rejections=req.degraded_after_rejections,
                reinit_after_rejections=req.reinit_after_rejections,
            )
            runtime._loop = runtime._build_loop(
                req, loop_class=TwoStageReconstructLoop,
                loop_kwargs={"session": session, "mode": "update", "change_policy": policy},
            )
            runtime._loop.start()
            return {
                "status": "started", "mode": "update", "channels": runtime._loop.publisher.channels,
                "session": session.summary(),
            }
        except HTTPException:
            raise
        except Exception as exc:
            runtime._loop = None
            if session is not None and session.state == SessionState.UPDATING:
                session.transition(SessionState.READY)
            raise HTTPException(status_code=400, detail=f"failed to update session: {exc}") from exc


@router.get("/status")
def session_status(session_dir: str):
    try:
        return MapSession.load(session_dir).summary()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"failed to load session: {exc}") from exc
