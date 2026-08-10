"""Durable state for the formal two-stage live mapping workflow.

The legacy rolling stream keeps all state in memory. ``MapSession`` is an opt-in,
versioned directory which owns the frozen anchor, reference frames, current DEM and
the persistent fusion grid. Every file is replaced atomically so a service crash
cannot leave a half-written manifest or numpy archive.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import Anchor


SCHEMA_VERSION = 1


class SessionState(str, Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    INIT_REVIEW = "INIT_REVIEW"
    READY = "READY"
    UPDATING = "UPDATING"
    DEGRADED = "DEGRADED"
    REINIT_REQUIRED = "REINIT_REQUIRED"


@dataclass
class SessionManifest:
    schema_version: int
    session_id: str
    state: str
    map_version: int
    created_at: float
    updated_at: float
    config: dict[str, Any] = field(default_factory=dict)
    quality_report: dict[str, Any] = field(default_factory=dict)
    last_change_report: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    consecutive_rejections: int = 0
    artifacts: dict[str, str] = field(default_factory=lambda: {
        "anchor": "anchor.npz",
        "dem": "current_dem.npz",
        "global_dem": "global_dem.npz",
        "reference_frames": "reference_frames",
    })


_ALLOWED_TRANSITIONS = {
    SessionState.IDLE: {SessionState.INITIALIZING},
    SessionState.INITIALIZING: {SessionState.INIT_REVIEW, SessionState.REINIT_REQUIRED},
    SessionState.INIT_REVIEW: {SessionState.READY, SessionState.REINIT_REQUIRED},
    SessionState.READY: {SessionState.UPDATING, SessionState.INITIALIZING},
    SessionState.UPDATING: {SessionState.READY, SessionState.DEGRADED, SessionState.REINIT_REQUIRED},
    SessionState.DEGRADED: {SessionState.UPDATING, SessionState.READY, SessionState.REINIT_REQUIRED},
    SessionState.REINIT_REQUIRED: {SessionState.INITIALIZING},
}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class MapSession:
    """A validated handle to one persistent map directory."""

    MANIFEST = "manifest.json"

    def __init__(self, root: str | os.PathLike, manifest: SessionManifest):
        self.root = Path(root).expanduser().resolve()
        self.manifest = manifest
        self._lock = threading.RLock()

    @classmethod
    def create(cls, root, *, session_id=None, config=None, overwrite=False) -> "MapSession":
        path = Path(root).expanduser().resolve()
        manifest_path = path / cls.MANIFEST
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"MapSession already exists: {path}")
        path.mkdir(parents=True, exist_ok=True)
        now = time.time()
        manifest = SessionManifest(
            schema_version=SCHEMA_VERSION,
            session_id=session_id or path.name or uuid.uuid4().hex,
            state=SessionState.IDLE.value,
            map_version=0,
            created_at=now,
            updated_at=now,
            config=dict(config or {}),
        )
        session = cls(path, manifest)
        session.save_manifest()
        return session

    @classmethod
    def load(cls, root) -> "MapSession":
        path = Path(root).expanduser().resolve()
        with (path / cls.MANIFEST).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported MapSession schema {raw.get('schema_version')}; expected {SCHEMA_VERSION}"
            )
        names = SessionManifest.__dataclass_fields__.keys()
        manifest = SessionManifest(**{key: raw[key] for key in names if key in raw})
        SessionState(manifest.state)
        return cls(path, manifest)

    @property
    def state(self) -> SessionState:
        return SessionState(self.manifest.state)

    def save_manifest(self) -> None:
        with self._lock:
            self.manifest.updated_at = time.time()
            _atomic_json(self.root / self.MANIFEST, asdict(self.manifest))

    def transition(self, state: SessionState, *, force=False) -> None:
        with self._lock:
            current = self.state
            if state == current:
                return
            if not force and state not in _ALLOWED_TRANSITIONS[current]:
                raise ValueError(f"invalid MapSession transition: {current.value} -> {state.value}")
            self.manifest.state = state.value
            self.save_manifest()

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.manifest.session_id,
            "session_dir": str(self.root),
            "state": self.manifest.state,
            "map_version": self.manifest.map_version,
            "created_at": self.manifest.created_at,
            "updated_at": self.manifest.updated_at,
            "quality_report": self.manifest.quality_report,
            "last_change_report": self.manifest.last_change_report,
            "review": self.manifest.review,
            "consecutive_rejections": self.manifest.consecutive_rejections,
        }

    def save_anchor(self, anchor: Anchor) -> None:
        ref = anchor.ref_ground_xyz
        _atomic_npz(
            self.root / self.manifest.artifacts["anchor"],
            R_align=np.asarray(anchor.R_align, dtype=np.float64),
            scale_factor=np.asarray([anchor.scale_factor], dtype=np.float64),
            x_bounds=np.asarray(anchor.x_bounds if anchor.x_bounds is not None else [], dtype=np.float64),
            z_bounds=np.asarray(anchor.z_bounds if anchor.z_bounds is not None else [], dtype=np.float64),
            ref_ground_xyz=np.asarray(ref if ref is not None else np.empty((0, 3)), dtype=np.float64),
        )

    def load_anchor(self) -> Anchor:
        path = self.root / self.manifest.artifacts["anchor"]
        with np.load(path, allow_pickle=False) as data:
            xb, zb = data["x_bounds"], data["z_bounds"]
            ref = data["ref_ground_xyz"]
            return Anchor(
                R_align=data["R_align"].copy(),
                scale_factor=float(data["scale_factor"][0]),
                x_bounds=tuple(xb.tolist()) if xb.size else None,
                z_bounds=tuple(zb.tolist()) if zb.size else None,
                ref_ground_xyz=ref.copy() if ref.size else None,
            )

    def save_dem(self, elev, has_data) -> None:
        _atomic_npz(
            self.root / self.manifest.artifacts["dem"],
            elev=np.asarray(elev, dtype=np.float64),
            has_data=np.asarray(has_data, dtype=np.bool_),
        )

    def load_dem(self):
        with np.load(self.root / self.manifest.artifacts["dem"], allow_pickle=False) as data:
            return data["elev"].copy(), data["has_data"].astype(bool, copy=True)

    def save_global_dem(self, global_dem) -> None:
        _atomic_npz(self.root / self.manifest.artifacts["global_dem"], **global_dem.to_snapshot())

    def load_global_dem(self):
        from .global_dem import GlobalDem
        with np.load(self.root / self.manifest.artifacts["global_dem"], allow_pickle=False) as data:
            snapshot = {key: data[key].copy() for key in data.files}
        return GlobalDem.from_snapshot(snapshot)

    def save_reference_frames(self, frames) -> int:
        from PIL import Image
        target = self.root / self.manifest.artifacts["reference_frames"]
        staging = self.root / f".reference_frames.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for index, frame in enumerate(frames):
                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = (arr * 255.0 if arr.size and arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr[..., :3]).save(staging / f"{index:06d}.jpg", quality=90)
            backup = self.root / f".reference_frames.{uuid.uuid4().hex}.old"
            if target.exists():
                os.replace(target, backup)
            os.replace(staging, target)
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return len(frames)

    def commit_map_version(self) -> int:
        with self._lock:
            self.manifest.map_version += 1
            self.save_manifest()
            return self.manifest.map_version
