"""streaming — real-time VGGT reconstruction → Unity elevation link (Milestone 1+).

This package is ADDITIVE and MUST NOT change the offline reconstruction path:
  * The offline endpoints/scripts never import this package.
  * Everything here reuses the offline building blocks (vggt_service._run_inference,
    gravity_alignment, elevation_plane, elevation_export) rather than reimplementing
    them, so the streaming DEM matches the offline DEM.

See REALTIME_LINK_PLAN.md and the [[realtime-link-noninvasive]] memory.
"""
