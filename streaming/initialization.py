"""Quality review and persistence for stage-one map initialization."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
import numpy as np

from .global_dem import FusionConfig, GlobalDem
from .map_session import MapSession, SessionState
from .pipeline import Anchor, DemResult


@dataclass
class InitializationPolicy:
    min_frames: int = 12
    min_points: int = 1000
    min_coverage: float = 0.15
    max_elevation_span_m: float = 100.0


def assess_initialization(res: DemResult, policy: InitializationPolicy) -> dict:
    finite = np.isfinite(res.elev) & np.asarray(res.has_data, dtype=bool)
    values = np.asarray(res.elev)[finite]
    coverage = float(finite.mean()) if finite.size else 0.0
    span = float(np.ptp(values)) if values.size else float("inf")
    reasons = []
    if res.n_frames < policy.min_frames:
        reasons.append(f"frames {res.n_frames} < {policy.min_frames}")
    if res.n_points < policy.min_points:
        reasons.append(f"points {res.n_points} < {policy.min_points}")
    if coverage < policy.min_coverage:
        reasons.append(f"coverage {coverage:.3f} < {policy.min_coverage:.3f}")
    if not np.isfinite(span) or span > policy.max_elevation_span_m:
        reasons.append(f"elevation_span_m {span:.3f} > {policy.max_elevation_span_m:.3f}")
    return {
        "passed": not reasons, "reasons": reasons,
        "metrics": {
            "frames": int(res.n_frames), "points": int(res.n_points),
            "coverage": round(coverage, 6),
            "elevation_span_m": round(span, 6) if np.isfinite(span) else None,
            "gravity_source": res.gravity_source, "warnings": list(res.warnings),
        },
        "policy": asdict(policy), "assessed_at": time.time(),
    }


def stage_initial_map(session, res, frames, *, policy, fusion_config) -> dict:
    if session.state != SessionState.INITIALIZING:
        raise ValueError(f"initialization requires INITIALIZING, got {session.state.value}")
    report = assess_initialization(res, policy)
    anchor = Anchor.from_result(res, keep_reference=True)
    ox = 0.5 * (res.x_bounds[0] + res.x_bounds[1])
    oz = 0.5 * (res.z_bounds[0] + res.z_bounds[1])
    global_dem = GlobalDem((ox, oz), fusion_config)
    if res.ground_xyz is not None:
        global_dem.integrate(res.ground_xyz, time.time())
        global_dem.changed_tiles()
    session.save_anchor(anchor)
    session.save_dem(res.elev, res.has_data)
    session.save_global_dem(global_dem)
    session.save_reference_frames(frames)
    session.manifest.quality_report = report
    session.manifest.review = {}
    session.save_manifest()
    session.transition(SessionState.INIT_REVIEW)
    return report


def finalize_initialization(session: MapSession, *, approved: bool, note: str = "") -> dict:
    if session.state != SessionState.INIT_REVIEW:
        raise ValueError(f"finalize requires INIT_REVIEW, got {session.state.value}")
    if approved and not bool(session.manifest.quality_report.get("passed")):
        raise ValueError("cannot approve: initialization quality gates did not pass")
    session.manifest.review = {
        "approved": bool(approved), "note": str(note), "reviewed_at": time.time(),
    }
    if approved:
        session.commit_map_version()
        session.transition(SessionState.READY)
    else:
        session.transition(SessionState.REINIT_REQUIRED)
    return session.summary()
