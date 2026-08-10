"""Trust gates for stage-two observations before they may mutate/publish the map."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class ChangePolicy:
    max_registration_rmse_m: float = 1.0
    min_coverage: float = 0.05
    max_changed_fraction: float = 0.35
    min_change_m: float = 0.05
    max_abs_height_change_m: float = 5.0
    require_registration: bool = True
    degraded_after_rejections: int = 2
    reinit_after_rejections: int = 5


def assess_change(res, reference_elev, reference_valid, policy: ChangePolicy) -> dict:
    current = np.asarray(res.elev, dtype=np.float64)
    current_valid = np.asarray(res.has_data, dtype=bool) & np.isfinite(current)
    reference = np.asarray(reference_elev, dtype=np.float64)
    ref_valid = np.asarray(reference_valid, dtype=bool) & np.isfinite(reference)
    if current.shape != reference.shape:
        raise ValueError(f"DEM shape changed: {current.shape} != {reference.shape}")
    overlap = current_valid & ref_valid
    coverage = float(overlap.mean()) if overlap.size else 0.0
    delta = np.abs(current - reference)
    changed = overlap & (delta >= policy.min_change_m)
    changed_fraction = float(changed.sum() / max(1, overlap.sum()))
    max_delta = float(delta[overlap].max()) if overlap.any() else float("inf")
    reasons = []
    if policy.require_registration and not bool(res.registered):
        reasons.append("registration was not confirmed")
    if res.registration_rmse is None or not np.isfinite(res.registration_rmse):
        if policy.require_registration:
            reasons.append("registration RMSE is unavailable")
    elif res.registration_rmse > policy.max_registration_rmse_m:
        reasons.append(f"registration_rmse_m {res.registration_rmse:.3f} > {policy.max_registration_rmse_m:.3f}")
    if coverage < policy.min_coverage:
        reasons.append(f"coverage {coverage:.3f} < {policy.min_coverage:.3f}")
    if changed_fraction > policy.max_changed_fraction:
        reasons.append(f"changed_fraction {changed_fraction:.3f} > {policy.max_changed_fraction:.3f}")
    if not np.isfinite(max_delta) or max_delta > policy.max_abs_height_change_m:
        reasons.append(f"max_height_change_m {max_delta:.3f} > {policy.max_abs_height_change_m:.3f}")
    return {
        "accepted": not reasons, "has_change": bool(changed.any()), "reasons": reasons,
        "metrics": {
            "registered": bool(res.registered), "registration_rmse_m": res.registration_rmse,
            "registration_yaw_deg": res.registration_yaw_deg, "coverage": round(coverage, 6),
            "changed_fraction": round(changed_fraction, 6), "changed_cells": int(changed.sum()),
            "max_height_change_m": round(max_delta, 6) if np.isfinite(max_delta) else None,
        },
        "policy": asdict(policy), "assessed_at": time.time(),
    }


def merge_accepted_dem(reference_elev, reference_valid, res):
    elev = np.asarray(reference_elev, dtype=np.float64).copy()
    valid = np.asarray(reference_valid, dtype=bool).copy()
    observed = np.asarray(res.has_data, dtype=bool) & np.isfinite(res.elev)
    elev[observed] = np.asarray(res.elev)[observed]
    valid[observed] = True
    return elev, valid
