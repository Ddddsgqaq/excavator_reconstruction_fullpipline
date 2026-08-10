"""Opt-in reconstruction loop for persistent initialization and trusted updates."""

from __future__ import annotations

import time
import traceback

from . import pipeline
from .change_detection import ChangePolicy, assess_change, merge_accepted_dem
from .global_dem import FusionConfig
from .initialization import InitializationPolicy, stage_initial_map
from .map_session import MapSession, SessionState
from .reconstruct_loop import ReconstructLoop


class TwoStageReconstructLoop(ReconstructLoop):
    """A session-aware loop; the legacy ``ReconstructLoop`` remains unchanged."""

    def __init__(self, source, publisher, cfg, *, session, mode,
                 initialization_policy=None, change_policy=None):
        if mode not in {"initialization", "update"}:
            raise ValueError(f"unknown two-stage mode: {mode}")
        super().__init__(source, publisher, cfg)
        self.map_session: MapSession = session
        self.session_mode = mode
        self.initialization_policy = initialization_policy or InitializationPolicy()
        self.change_policy = change_policy or ChangePolicy()
        self._reference_elev = self._reference_valid = None
        if mode == "update":
            if session.state != SessionState.UPDATING:
                raise ValueError(f"update loop requires UPDATING, got {session.state.value}")
            self._anchor = session.load_anchor()
            self._gdem = session.load_global_dem()
            self._reference_elev, self._reference_valid = session.load_dem()
            self.cfg.freeze_anchor = True
            self.cfg.register = True
            self.cfg.fusion = True
        elif session.state != SessionState.INITIALIZING:
            raise ValueError(f"initialization loop requires INITIALIZING, got {session.state.value}")

    def session_summary(self):
        # Finalize runs through a separate request/MapSession instance. Reloading here
        # prevents the monitor from showing stale INIT_REVIEW after an external review.
        try:
            return MapSession.load(self.map_session.root).summary()
        except Exception:
            return self.map_session.summary()

    def stop(self, join_timeout: float = 10.0) -> None:
        super().stop(join_timeout)
        if self.session_mode == "update" and self.map_session.state == SessionState.UPDATING:
            self.map_session.transition(SessionState.READY)

    def _finish_initialization_capture(self):
        self._stop.set()
        close_source = getattr(self.source, "close", None)
        if callable(close_source):
            close_source()
        with self._status_lock:
            self._status.running = False

    def _record_common_status(self, res, kept, elapsed, elev_range=None):
        with self._status_lock:
            self._status.passes += 1
            self._status.last_pass_seconds = round(elapsed, 2)
            self._status.last_gravity_source = res.gravity_source
            if elev_range is not None:
                self._status.last_elev_range_m = elev_range
            self._status.anchor_frozen = self._anchor is not None
            self._status.last_registered = res.registered
            self._status.last_reg_rmse = (round(res.registration_rmse, 4)
                                          if res.registration_rmse is not None else None)
            self._status.last_reg_yaw_deg = (round(res.registration_yaw_deg, 2)
                                             if res.registration_yaw_deg is not None else None)
            self._status.last_error = ""
            self._last_reconstructed_kept = kept

    def _fusion_config(self):
        return FusionConfig(
            world_size_m=self.cfg.world_size_m,
            tile_size_m=self.cfg.tile_size_m,
            tile_res=self.cfg.grid_resolution,
            decay=self.cfg.fusion_decay,
            t_ref=self.cfg.interval,
            change_thresh=self.cfg.change_thresh,
            top_percentile=self.cfg.top_percentile,
            height_resolution=self.cfg.height_resolution,
        )

    def _handle_update(self, res):
        report = assess_change(
            res, self._reference_elev, self._reference_valid, self.change_policy
        )
        self.map_session.manifest.last_change_report = report
        if not report["accepted"]:
            count = self.map_session.manifest.consecutive_rejections + 1
            self.map_session.manifest.consecutive_rejections = count
            if count >= self.change_policy.reinit_after_rejections:
                self.map_session.transition(SessionState.REINIT_REQUIRED)
                self._stop.set()
            elif count >= self.change_policy.degraded_after_rejections:
                self.map_session.transition(SessionState.DEGRADED)
                self._stop.set()
            else:
                self.map_session.save_manifest()
            return None

        self.map_session.manifest.consecutive_rejections = 0
        if not report["has_change"]:
            self.map_session.save_manifest()
            return None
        elev_range = self._publish_fusion(res)
        self._reference_elev, self._reference_valid = merge_accepted_dem(
            self._reference_elev, self._reference_valid, res
        )
        self.map_session.save_dem(self._reference_elev, self._reference_valid)
        self.map_session.save_global_dem(self._gdem)
        self.map_session.commit_map_version()
        return elev_range

    def _recon_loop(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(self.cfg.interval):
                break
            kept = self.buffer.stats.kept
            frames = self.buffer.snapshot()
            if len(frames) < self.cfg.min_frames or kept <= self._last_reconstructed_kept:
                continue
            try:
                started = time.monotonic()
                res = pipeline.reconstruct_frames_to_dem(
                    frames, prev_anchor=self._anchor, register=self.cfg.register,
                    grid_resolution=self.cfg.grid_resolution, scale_factor=self.cfg.scale_factor,
                    glb_out=self._glb_out_path(),
                )
                if self.session_mode == "initialization":
                    report = stage_initial_map(
                        self.map_session, res, frames,
                        policy=self.initialization_policy, fusion_config=self._fusion_config(),
                    )
                    self._anchor = pipeline.Anchor.from_result(res, keep_reference=True)
                    self._record_common_status(res, kept, time.monotonic() - started)
                    if not report["passed"]:
                        with self._status_lock:
                            self._status.last_error = "initialization quality review failed"
                    self._finish_initialization_capture()
                    break

                elev_range = self._handle_update(res)
                self._record_common_status(res, kept, time.monotonic() - started, elev_range)
                if self._stop.is_set():
                    close_source = getattr(self.source, "close", None)
                    if callable(close_source):
                        close_source()
                    with self._status_lock:
                        self._status.running = False
            except Exception:
                with self._status_lock:
                    self._status.last_error = "session recon: " + traceback.format_exc(limit=3)
