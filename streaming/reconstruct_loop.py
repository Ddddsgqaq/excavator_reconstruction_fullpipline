"""reconstruct_loop.py — the streaming orchestration loop (M3, A-scheme).

Ties M1 + M2 together into a self-driving loop:

    pump thread:   FrameSource.frames() ──offer──▶ KeyframeBuffer
    recon thread:  every T seconds ──snapshot──▶ reconstruct_frames_to_dem ──▶ publish

Two daemon threads so neither blocks the other; a recon pass that overruns T just means the
next pass starts late (we skip, never queue — dropping is better than backlog for live view).

Runs inside the vggt_service process to reuse the resident VGGT `_model` (heavy imports in
reconstruct_frames_to_dem are deferred, so importing THIS module stays cheap). Non-invasive:
nothing here touches the offline endpoints.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass, field

from .frame_source import FrameSource, VideoFileSource
from .keyframe_buffer import KeyframeBuffer
from .elevation_publisher import ElevationPublisher
from . import pipeline

# Cap on ground points surfaced per pass to the diagnostics viewer. The DEM is interpolated
# from all ground points; the viewer only needs a representative cloud, so we downsample with
# a deterministic stride (no RNG) to keep the /stream/pass payload light and stable.
_PASS_POINT_CAP = 100000


@dataclass
class LoopConfig:
    interval: float = 6.0            # T: seconds between reconstruction passes (M1: warm ~5s)
    min_frames: int = 4              # don't reconstruct until the window has this many keyframes
    capacity: int = 12               # keyframe window size
    sim_thresh: float = 0.92         # keyframe keep threshold (offline-aligned)
    target_fps: float = 3.0          # frame pump emit rate
    use_orb: bool = True             # False: keep each interval-sampled source frame
    frame_sample_interval: float = 1.0 / 3.0
    capture_only: bool = False       # pump frames for diagnostics; never invoke VGGT
    grid_resolution: int = 128
    scale_factor: float = 28.0
    height_resolution: float = 0.01
    tile_x: int = 0
    tile_y: int = 0
    tile_size_meters: float | None = None
    # M4 coordinate stability
    freeze_anchor: bool = True       # M4.1: freeze gravity+scale+footprint from pass 1
    register: bool = True            # M4.2: horizontally register each pass onto the anchor
    # M5 persistent fusion + multi-tile (opt-in; fusion=False → exact M4 single-tile behavior)
    fusion: bool = False             # M5: fuse passes into a persistent global DEM, publish changed tiles
    world_size_m: float = 150.0      # global grid edge (anchor ±75 m)
    tile_size_m: float = 50.0        # one Unity tile edge (must match Unity tileSizeMeters)
    fusion_decay: float = 0.5        # old-weight decay per pass (fast-follow digging)
    change_thresh: float = 0.05      # max|Δh| (m) for a tile to be republished
    top_percentile: float = 70.0     # legacy raw-point fusion percentile
    auto_fusion_extent: bool = False  # size 3x3 global tiles from first DEM footprint
    fusion_extent_margin: float = 1.25
    max_registration_rmse_m: float = 1.0
    max_changed_fraction: float = 0.35
    min_change_neighbors: int = 3
    # Diagnostics: optionally save each pass's raw VGGT point cloud as a .glb for
    # offline supervision. Purely additive — never affects the DEM/tiles output.
    save_glb: bool = False
    glb_dir: str | None = None       # where per-pass .glb files are written


@dataclass
class LoopStatus:
    running: bool = False
    source: str = ""
    channels: list = field(default_factory=list)
    window: int = 0                  # current keyframes buffered
    offered: int = 0                 # frames offered by pump
    kept: int = 0                    # keyframes accepted
    passes: int = 0                  # completed reconstruction passes
    published: int = 0               # ElevationMsgs published
    last_pass_seconds: float = 0.0
    last_gravity_source: str = ""
    last_elev_range_m: list = field(default_factory=list)
    last_error: str = ""
    # M4 coordinate stability
    anchor_frozen: bool = False      # True once pass 1 has set the anchor
    last_registered: bool = False    # did the last pass apply an M4.2 transform
    last_reg_rmse: float | None = None
    last_reg_yaw_deg: float | None = None
    # M5 fusion
    fusion_enabled: bool = False
    observed_cells: int = 0          # global-grid cells with fused data
    tiles_published_total: int = 0   # cumulative tile messages published (fusion mode)
    last_changed_tiles: list = field(default_factory=list)  # (tile_x,tile_y) changed last pass
    last_change_report: dict = field(default_factory=dict)
    keyframe_mode: str = "orb"
    frame_sample_interval: float = 0.0
    orb_calls: int = 0
    orb_total_seconds: float = 0.0
    orb_last_ms: float = 0.0
    orb_avg_ms: float = 0.0


class ReconstructLoop:
    """Owns the pump + reconstruction threads for one streaming session."""

    def __init__(self, source: FrameSource, publisher: ElevationPublisher,
                 cfg: LoopConfig | None = None):
        self.source = source
        self.publisher = publisher
        self.cfg = cfg or LoopConfig()
        self.buffer = KeyframeBuffer(
            capacity=self.cfg.capacity,
            sim_thresh=self.cfg.sim_thresh,
            use_orb=self.cfg.use_orb,
        )
        self._stop = threading.Event()
        self._pump_thread: threading.Thread | None = None
        self._recon_thread: threading.Thread | None = None
        self._status = LoopStatus(source=str(getattr(source, "path", type(source).__name__)),
                                  channels=publisher.channels)
        self._status_lock = threading.Lock()
        self._anchor: pipeline.Anchor | None = None   # M4: frozen after the first pass
        self._gdem = None                             # M5: GlobalDem, created after anchor freeze
        self._vertical_datum_m: float | None = None  # first-pass main plane, shared by all tiles
        self._tiles_published = 0                     # M5: cumulative published tile count
        self._last_reconstructed_kept = 0             # do not rerun an unchanged window
        # Diagnostics for the live viewer: the exact keyframes + ground cloud of the most
        # recent successful pass. Guarded by its OWN lock so JPEG encoding of a stored frame
        # never blocks the /stream/status poll (which holds _status_lock).
        self._diag_lock = threading.Lock()
        self._last_pass_diag: dict | None = None

    def _glb_out_path(self) -> str | None:
        """Per-pass .glb destination when save_glb is on; keyed by pass index so each
        reconstruction is preserved for offline supervision."""
        if not self.cfg.save_glb or not self.cfg.glb_dir:
            return None
        return os.path.join(self.cfg.glb_dir, f"pass_{self._status.passes:04d}.glb")

    # ── lifecycle ───────────────────────────────────────────────
    def start(self) -> None:
        if self._pump_thread is not None:
            raise RuntimeError("loop already started")
        self._stop.clear()
        with self._status_lock:
            self._status.running = True
        self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True, name="stream-pump")
        self._pump_thread.start()
        if not self.cfg.capture_only:
            self._recon_thread = threading.Thread(
                target=self._recon_loop, daemon=True, name="stream-recon"
            )
            self._recon_thread.start()

    def stop(self, join_timeout: float = 10.0) -> None:
        self._stop.set()
        close_source = getattr(self.source, "close", None)
        if callable(close_source):
            close_source()
        for th in (self._recon_thread, self._pump_thread):
            if th is not None:
                th.join(timeout=join_timeout)
        self.publisher.close()
        with self._status_lock:
            self._status.running = False

    def status(self) -> LoopStatus:
        with self._status_lock:
            s = self._status
            bs = self.buffer.stats
            # merge live buffer counters
            return LoopStatus(
                running=s.running, source=s.source, channels=list(s.channels),
                window=bs.window, offered=bs.offered, kept=bs.kept,
                passes=s.passes, published=s.published,
                last_pass_seconds=s.last_pass_seconds,
                last_gravity_source=s.last_gravity_source,
                last_elev_range_m=list(s.last_elev_range_m),
                last_error=s.last_error,
                anchor_frozen=s.anchor_frozen,
                last_registered=s.last_registered,
                last_reg_rmse=s.last_reg_rmse,
                last_reg_yaw_deg=s.last_reg_yaw_deg,
                fusion_enabled=self.cfg.fusion,
                observed_cells=s.observed_cells,
                tiles_published_total=s.tiles_published_total,
                last_changed_tiles=list(s.last_changed_tiles),
                last_change_report=dict(s.last_change_report),
                keyframe_mode=bs.selection_mode,
                frame_sample_interval=self.cfg.frame_sample_interval,
                orb_calls=bs.orb_calls,
                orb_total_seconds=round(bs.orb_total_seconds, 6),
                orb_last_ms=round(bs.orb_last_ms, 3),
                orb_avg_ms=round(
                    bs.orb_total_seconds * 1000.0 / bs.orb_calls, 3
                ) if bs.orb_calls else 0.0,
            )

    # ── threads ─────────────────────────────────────────────────
    def _pump_loop(self) -> None:
        try:
            for frame in self.source.frames():
                if self._stop.is_set():
                    break
                self.buffer.offer(frame)
        except Exception:
            with self._status_lock:
                self._status.last_error = "pump: " + traceback.format_exc(limit=2)
        # Source exhausted (non-looping file): pump ends, recon keeps serving last window
        # until stop() is called.

    def _recon_loop(self) -> None:
        while not self._stop.is_set():
            # pace: wait interval, but wake early on stop
            if self._stop.wait(self.cfg.interval):
                break
            kept = self.buffer.stats.kept
            frames = self.buffer.snapshot()
            if len(frames) < self.cfg.min_frames or kept <= self._last_reconstructed_kept:
                continue
            try:
                t0 = time.monotonic()
                res = pipeline.reconstruct_frames_to_dem(
                    frames,
                    prev_anchor=self._anchor,                  # M4: None on pass 1, frozen after
                    register=self.cfg.register,
                    grid_resolution=self.cfg.grid_resolution,
                    scale_factor=self.cfg.scale_factor,
                    fixed_footprint=not self.cfg.fusion,
                    glb_out=self._glb_out_path(),
                )
                # M4.1: freeze the anchor from the first successful pass. Every later pass
                # then reuses its gravity + scale + footprint (and registers onto its ground).
                if self.cfg.freeze_anchor and self._anchor is None:
                    self._anchor = pipeline.Anchor.from_result(
                        res, keep_reference=self.cfg.register
                    )

                # Publish: M5 fusion (multi-tile, changed only) or M4 single-tile.
                if self.cfg.fusion:
                    elev_range = self._publish_fusion(res)
                else:
                    elev_range = self._publish_single_tile(res)

                dt = time.monotonic() - t0
                with self._status_lock:
                    self._status.passes += 1
                    self._status.last_pass_seconds = round(dt, 2)
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
                    pass_index = self._status.passes
                # Diagnostics snapshot (own lock; never nested under _status_lock).
                self._store_pass_diag(pass_index, frames, res, elev_range)
            except Exception:
                with self._status_lock:
                    self._status.last_error = "recon: " + traceback.format_exc(limit=3)

    # ── publish paths ───────────────────────────────────────────
    def _ensure_vertical_datum(self, elev, valid) -> float:
        """Freeze one shared terrain datum so the main plane maps to Unity Y=0."""
        import numpy as np
        values = np.asarray(elev, dtype=np.float64)[np.asarray(valid, dtype=bool)]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("cannot establish vertical datum from an empty DEM")
        if self._vertical_datum_m is None:
            self._vertical_datum_m = float(np.median(values))
        return self._vertical_datum_m

    def _publish_single_tile(self, res) -> list | None:
        """M0–M4 behaviour: publish the whole pass as one fixed tile."""
        datum = self._ensure_vertical_datum(res.elev, res.has_data)
        msg = pipeline.dem_result_to_msg(
            res,
            height_resolution=self.cfg.height_resolution,
            tile_x=self.cfg.tile_x, tile_y=self.cfg.tile_y,
            tile_size_meters=self.cfg.tile_size_meters,
            timestamp=time.time(),
        )
        msg["metadata"]["vertical_datum_m"] = datum
        self.publisher.publish(msg)
        m = msg["metadata"]
        with self._status_lock:
            self._status.published += 1
        return [round(m["min_elevation"], 3), round(m["max_elevation"], 3)]

    def _publish_fusion(self, res) -> list | None:
        """Fuse the pass's Elevation Viewer surface into the persistent Global DEM.

        The fused/Unity path deliberately consumes the same interpolated + hole-filled DEM
        shown in the per-pass viewer. Ground-only points are reserved for registration and
        must not become a second, flatter terrain definition.
        """
        from elevation_export import dem_to_elevation_msg
        from elevation_plane import fill_elevation_view_holes
        from . import global_dem as gdm

        import numpy as np

        pass_elev, pass_valid = fill_elevation_view_holes(res.elev, res.has_data)
        if not np.asarray(pass_valid, dtype=bool).any():
            with self._status_lock:
                self._status.last_error = "fusion: pass had no Elevation Viewer surface; skipped"
            return None

        datum = self._ensure_vertical_datum(pass_elev, pass_valid)

        # Lazily build the global DEM once the anchor origin is known.
        initial_global_pass = self._gdem is None
        if self._gdem is None:
            if self._anchor is not None and self._anchor.x_bounds is not None:
                ox = 0.5 * (self._anchor.x_bounds[0] + self._anchor.x_bounds[1])
                oz = 0.5 * (self._anchor.z_bounds[0] + self._anchor.z_bounds[1])
            else:
                # no anchor (freeze_anchor off): centre on this pass's own footprint
                ox = 0.5 * (res.x_bounds[0] + res.x_bounds[1])
                oz = 0.5 * (res.z_bounds[0] + res.z_bounds[1])
            if self.cfg.auto_fusion_extent:
                footprint = max(float(res.x_bounds[1] - res.x_bounds[0]),
                                float(res.z_bounds[1] - res.z_bounds[0]))
                tile_size_m = max(footprint * float(self.cfg.fusion_extent_margin), 1e-3)
                world_size_m = tile_size_m * 3.0
            else:
                tile_size_m = self.cfg.tile_size_m
                world_size_m = self.cfg.world_size_m
            fcfg = gdm.FusionConfig(
                world_size_m=world_size_m,
                tile_size_m=tile_size_m,
                tile_res=self.cfg.grid_resolution,
                decay=self.cfg.fusion_decay,
                t_ref=self.cfg.interval,
                change_thresh=self.cfg.change_thresh,
                top_percentile=self.cfg.top_percentile,
                height_resolution=self.cfg.height_resolution,
            )
            self._gdem = gdm.GlobalDem(origin_xz=(ox, oz), cfg=fcfg)

        # Sample the exact per-pass Elevation Viewer mesh in its registered world frame.
        # GlobalDem then performs only temporal/cross-pass fusion; it does not re-derive a
        # different terrain from ground-only points.
        rows, cols = pass_elev.shape
        xi = np.linspace(float(res.x_bounds[0]), float(res.x_bounds[1]), cols)
        zi = np.linspace(float(res.z_bounds[0]), float(res.z_bounds[1]), rows)
        xx, zz = np.meshgrid(xi, zi)
        valid = np.asarray(pass_valid, dtype=bool) & np.isfinite(pass_elev)
        surface_xyz = np.column_stack((xx[valid], pass_elev[valid], zz[valid]))
        if (not initial_global_pass and self.cfg.register
                and (not bool(res.registered)
                     or res.registration_rmse is None
                     or not np.isfinite(res.registration_rmse)
                     or res.registration_rmse > self.cfg.max_registration_rmse_m)):
            change_report = {
                "decision": "rejected", "reason": "registration_quality_gate",
                "registered": bool(res.registered),
                "registration_rmse_m": res.registration_rmse,
            }
        else:
            change_report = self._gdem.integrate(
                surface_xyz, time.time(), aggregation="mean",
                min_change_m=self.cfg.change_thresh,
                max_changed_fraction=self.cfg.max_changed_fraction,
                min_change_neighbors=self.cfg.min_change_neighbors,
            )
        updates = self._gdem.changed_tiles()

        elev_lo = elev_hi = None
        for u in updates:
            filled_elev, filled_valid = fill_elevation_view_holes(u.elev, u.has_data)
            msg = dem_to_elevation_msg(
                filled_elev, u.x_bounds, u.z_bounds, has_data=filled_valid,
                # GlobalDem coordinates and heights are already metric-scaled.
                scale_factor=1.0,
                height_resolution=self.cfg.height_resolution,
                tile_x=u.tile_x, tile_y=u.tile_y,
                tile_size_meters=self._gdem.cfg.tile_size_m,
                timestamp=time.time(),
            )
            msg["metadata"]["dem_preprocessing"] = "elevation_viewer_fill_20"
            msg["metadata"]["vertical_datum_m"] = datum
            msg["metadata"]["source_nodata_count"] = int((~u.has_data.astype(bool)).sum())
            msg["source_valid"] = u.has_data.astype("uint8").reshape(-1).tolist()
            self.publisher.publish(msg)
            self._tiles_published += 1
            m = msg["metadata"]
            elev_lo = m["min_elevation"] if elev_lo is None else min(elev_lo, m["min_elevation"])
            elev_hi = m["max_elevation"] if elev_hi is None else max(elev_hi, m["max_elevation"])

        gstat = self._gdem.status()
        with self._status_lock:
            self._status.published += len(updates)
            self._status.fusion_enabled = True
            self._status.observed_cells = gstat["observed_cells"]
            self._status.tiles_published_total = self._tiles_published
            self._status.last_changed_tiles = [[u.tile_x, u.tile_y] for u in updates]
            self._status.last_change_report = dict(change_report)
        if elev_lo is None:
            return None
        return [round(elev_lo, 3), round(elev_hi, 3)]

    # ── per-pass diagnostics (live viewer) ──────────────────────
    def _store_pass_diag(self, pass_index: int, frames, res, elev_range) -> None:
        """Cache the just-finished pass's keyframes + RGB cloud for /stream/pass.

        Only the latest pass is retained. Frame arrays are stored by reference (the buffer
        never mutates a kept frame in place); the full cloud is downsampled to a cap with
        a deterministic stride so repeat fetches return an identical, light payload."""
        import numpy as np

        pts = getattr(res, "points_aligned", None)
        if pts is None:
            pts = getattr(res, "ground_xyz", None)
        colors = getattr(res, "point_colors", None)
        if pts is not None:
            pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
                pts = None
                colors = None
            else:
                if colors is not None:
                    colors = np.asarray(colors, dtype=np.uint8)
                    if colors.shape[0] != pts.shape[0]:
                        colors = None
                if pts.shape[0] > _PASS_POINT_CAP:
                    idx = np.linspace(0, pts.shape[0] - 1, _PASS_POINT_CAP).astype(np.int64)
                    pts = pts[idx]
                    if colors is not None:
                        colors = colors[idx]
        dem_elev = np.asarray(res.elev, dtype=np.float32)
        dem_valid = np.asarray(res.has_data, dtype=np.uint8)
        dem_finite = dem_elev[np.isfinite(dem_elev)]
        # res.elev is already metric-scaled in pipeline step 4.
        dem_min = float(dem_finite.min()) if dem_finite.size else None
        diag = {
            "pass_index": int(pass_index),
            "n_frames": int(getattr(res, "n_frames", len(frames))),
            "n_points": int(getattr(res, "n_points", 0)),
            "x_bounds": [float(res.x_bounds[0]), float(res.x_bounds[1])],
            "z_bounds": [float(res.z_bounds[0]), float(res.z_bounds[1])],
            "elev_range": list(elev_range) if elev_range is not None else None,
            "gravity_source": res.gravity_source,
            "registered": bool(res.registered),
            "frames": list(frames),
            "points": None if pts is None else pts[:, :3],
            "colors": colors,
            "ground_count": (0 if getattr(res, "ground_xyz", None) is None
                             else int(res.ground_xyz.shape[0])),
            "dem_min_elevation": dem_min,
            "dem_elev": dem_elev,
            "dem_valid": dem_valid,
        }
        with self._diag_lock:
            self._last_pass_diag = diag

    def pass_diagnostics(self) -> dict | None:
        """JSON-ready meta + downsampled RGB cloud for the last pass (no raw frames)."""
        fusion_dem = self._gdem.viewer_dem() if self._gdem is not None else None
        if fusion_dem is not None and self._vertical_datum_m is not None:
            fusion_dem["vertical_datum_m"] = float(self._vertical_datum_m)
            fusion_dem["unity_base_y_m"] = 0.0
        with self._diag_lock:
            d = self._last_pass_diag
            if d is None:
                return None
            pts = d["points"]
            return {
                "pass_index": d["pass_index"],
                "n_frames": d["n_frames"],
                "n_points": d["n_points"],
                "n_points_shown": 0 if pts is None else int(pts.shape[0]),
                "frame_count": len(d["frames"]),
                "x_bounds": d["x_bounds"],
                "z_bounds": d["z_bounds"],
                "elev_range": d["elev_range"],
                "gravity_source": d["gravity_source"],
                "registered": d["registered"],
                "ground_count": d["ground_count"],
                "dem_min_elevation": d["dem_min_elevation"],
                "min_elev": None if pts is None else float(pts[:, 1].min()),
                "points": [] if pts is None else pts.tolist(),
                "colors": [] if d["colors"] is None else d["colors"].tolist(),
                "fusion_enabled": bool(self.cfg.fusion),
                "fusion_source": "elevation_view_dem" if self.cfg.fusion else None,
                "fusion_dem": fusion_dem,
                "dem": {
                    "grid_res": int(d["dem_elev"].shape[0]),
                    "x_min": d["x_bounds"][0], "x_max": d["x_bounds"][1],
                    "z_min": d["z_bounds"][0], "z_max": d["z_bounds"][1],
                    "elev": d["dem_elev"].tolist(),
                    "has_data": d["dem_valid"].tolist(),
                },
            }

    def pass_frame(self, idx: int):
        """Return the idx-th keyframe (RGB ndarray) used in the last pass, or None."""
        with self._diag_lock:
            d = self._last_pass_diag
            if d is None:
                return None
            frames = d["frames"]
            if idx < 0 or idx >= len(frames):
                return None
            return frames[idx]


def build_loop_from_video(
    video_path: str,
    *,
    file_out: str | None = None,
    mqtt: bool = False,
    broker: str = "127.0.0.1",
    port: int = 1883,
    cfg: LoopConfig | None = None,
    loop_video: bool = True,
) -> ReconstructLoop:
    """Convenience builder: mp4 file source + file/mqtt publisher → ready-to-start loop."""
    cfg = cfg or LoopConfig()
    source = VideoFileSource(video_path, target_fps=cfg.target_fps, loop=loop_video)
    publisher = ElevationPublisher(file_out=file_out, mqtt=mqtt, broker=broker, port=port)
    return ReconstructLoop(source, publisher, cfg)
