"""verify_m4.py — real-GPU smoke test for M4 (coordinate stability).

Runs the streaming ReconstructLoop against a real video IN-PROCESS (reusing the resident
vggt_service `_model`) and reports whether M4 is doing its job:

  * the anchor freezes after pass 1,
  * later passes report `registered=True` with a sane yaw/RMSE,
  * and — the decisive check — the DEM of *static* terrain stops moving between passes.

The static-terrain check runs the SAME video twice: once with M4 registration OFF
(freeze footprint only) and once with it ON, then compares the mean cross-pass DEM RMS
over cells present in consecutive passes. With M4 working, ON should be markedly lower
than OFF (the terrain stays put; only real digging changes cells).

Usage (in the vggt_service conda env, GPU available):
    python tools/verify_m4.py dynamic_execave_video.mp4
    python tools/verify_m4.py <video> --seconds 45 --interval 6
    python tools/verify_m4.py <video> --on-only     # skip the OFF comparison run

Nothing here touches the offline path; it only drives the streaming package.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Allow running as `python tools/verify_m4.py` from anywhere: put the repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _collect_dems(video_path, *, register, seconds, interval, min_frames, capacity, target_fps):
    """Run one streaming session, capturing elevation-view DEMs (float grid + validity)."""
    # Import here so `python tools/verify_m4.py -h` works without loading the model.
    import vggt_service  # noqa: F401 — module-level load puts VGGT on the GPU (resident _model)
    import streaming.reconstruct_loop as RL
    from streaming.frame_source import VideoFileSource
    from streaming.elevation_publisher import ElevationPublisher
    from streaming import pipeline

    captured = []  # list of dicts: {elev, valid, registered, rmse, yaw, grav}

    # Wrap dem_result_to_msg so we snapshot each pass's DEM without disturbing publishing.
    orig_to_msg = pipeline.dem_result_to_msg

    def spy_to_msg(res, **kw):
        captured.append({
            "elev": np.asarray(res.elev, dtype=np.float64).copy(),
            "valid": np.asarray(res.has_data, dtype=bool).copy(),
            "registered": res.registered,
            "rmse": res.registration_rmse,
            "yaw": res.registration_yaw_deg,
            "grav": res.gravity_source,
        })
        return orig_to_msg(res, **kw)

    pipeline.dem_result_to_msg = spy_to_msg

    import tempfile
    tmp = tempfile.mkdtemp(prefix="verify_m4_")
    # We capture DEMs via the spy above; the publisher just needs a valid (throwaway) channel.
    source = VideoFileSource(video_path, target_fps=target_fps, loop=True)
    publisher = ElevationPublisher(file_out=tmp, mqtt=False)
    cfg = RL.LoopConfig(
        interval=interval, min_frames=min_frames, capacity=capacity,
        target_fps=target_fps, freeze_anchor=True, register=register,
    )
    loop = RL.ReconstructLoop(source, publisher, cfg)
    tag = "ON " if register else "OFF"
    print(f"\n[{tag}] streaming {seconds}s @ interval={interval}s ...")
    loop.start()
    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            time.sleep(interval)
            s = loop.status()
            print(f"  [{tag}] pass={s.passes} window={s.window} "
                  f"grav={s.last_gravity_source} anchor_frozen={s.anchor_frozen} "
                  f"registered={s.last_registered} yaw={s.last_reg_yaw_deg} rmse={s.last_reg_rmse} "
                  f"{s.last_error[:60]}")
    finally:
        loop.stop()
        pipeline.dem_result_to_msg = orig_to_msg
    return captured


def _cross_pass_rms(dems):
    """Mean RMS elevation difference between consecutive passes, over co-valid cells."""
    diffs = []
    for a, b in zip(dems[:-1], dems[1:]):
        m = a["valid"] & b["valid"] & np.isfinite(a["elev"]) & np.isfinite(b["elev"])
        if m.sum() < 50:
            continue
        diffs.append(float(np.sqrt(np.mean((a["elev"][m] - b["elev"][m]) ** 2))))
    return diffs


def _render_dem_grid(dems, path, title):
    """Render each pass's elevation-view DEM in a row so drift is visible (a static scene should
    look identical pass-to-pass; drift shows as the pattern shifting/rotating)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(dems)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    # shared colour range across passes
    allv = np.concatenate([d["elev"][np.isfinite(d["elev"])].ravel() for d in dems if np.isfinite(d["elev"]).any()])
    lo, hi = (np.percentile(allv, 2), np.percentile(allv, 98)) if allv.size else (0, 1)
    for i, (ax, d) in enumerate(zip(axes, dems)):
        ax.imshow(np.ma.masked_invalid(d["elev"]), origin="lower", cmap="terrain", vmin=lo, vmax=hi)
        reg = "reg" if d.get("registered") else "—"
        ax.set_title(f"pass {i+1} ({reg})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Real-GPU smoke test for M4 coordinate stability.")
    ap.add_argument("video", help="path to an mp4 to replay as a live stream")
    ap.add_argument("--seconds", type=float, default=40.0, help="run length per session")
    ap.add_argument("--interval", type=float, default=6.0, help="T: seconds between passes")
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--capacity", type=int, default=12)
    ap.add_argument("--target-fps", type=float, default=3.0)
    ap.add_argument("--on-only", action="store_true", help="skip the registration-OFF comparison run")
    ap.add_argument("--viz", action="store_true", help="save DEM images (ON vs OFF, consecutive passes)")
    ap.add_argument("--viz-dir", default=None, help="output dir for viz (default: ./verify_m4_viz)")
    args = ap.parse_args()

    kw = dict(seconds=args.seconds, interval=args.interval, min_frames=args.min_frames,
              capacity=args.capacity, target_fps=args.target_fps)

    print("=" * 72)
    print("M4 verification — this loads VGGT on the GPU (cold start ~20s the first time).")
    print("=" * 72)

    on = _collect_dems(args.video, register=True, **kw)
    on_rms = _cross_pass_rms(on)

    # --- Checks on the ON run ---
    print("\n--- M4 ON summary ---")
    print(f"  passes captured: {len(on)}")
    if on:
        print(f"  pass 1 gravity_source: {on[0]['grav']}  (fresh estimate, expected 'trajectory'/'cloud_ransac')")
    later = on[1:]
    reg_ok = [p for p in later if p["registered"]]
    print(f"  later passes registered: {len(reg_ok)}/{len(later)}")
    for i, p in enumerate(later, start=2):
        print(f"    pass {i}: grav={p['grav']} registered={p['registered']} "
              f"yaw={None if p['yaw'] is None else round(p['yaw'],2)} "
              f"rmse={None if p['rmse'] is None else round(p['rmse'],4)}")
    if on_rms:
        print(f"  cross-pass DEM RMS (ON):  mean={np.mean(on_rms):.4f} m  per-pair={[round(x,3) for x in on_rms]}")

    verdict = []
    if len(on) >= 2:
        if all(p["grav"] == "anchor" for p in later):
            verdict.append("PASS: later passes reuse the frozen anchor gravity")
        else:
            verdict.append("WARN: some later passes did NOT reuse the anchor (grav != 'anchor')")
        if reg_ok:
            verdict.append("PASS: cross-pass registration ran and converged")
        else:
            verdict.append("WARN: no later pass reported registered=True (check overlap / relief)")
    else:
        verdict.append("INCONCLUSIVE: fewer than 2 passes — increase --seconds or lower --interval")

    if not args.on_only:
        off = _collect_dems(args.video, register=False, **kw)
        off_rms = _cross_pass_rms(off)
        print("\n--- Decisive comparison (static-terrain drift) ---")
        print(f"  cross-pass DEM RMS  OFF (footprint only): {np.mean(off_rms):.4f} m" if off_rms else "  OFF: n/a")
        print(f"  cross-pass DEM RMS  ON  (+registration):  {np.mean(on_rms):.4f} m" if on_rms else "  ON:  n/a")
        if on_rms and off_rms:
            if np.mean(on_rms) < np.mean(off_rms):
                verdict.append(f"PASS: registration reduces cross-pass drift "
                               f"({np.mean(off_rms):.3f} → {np.mean(on_rms):.3f} m)")
            else:
                verdict.append(f"WARN: registration did NOT reduce drift "
                               f"({np.mean(off_rms):.3f} → {np.mean(on_rms):.3f} m) — "
                               f"terrain may be moving a lot, or overlap too low")

    if args.viz:
        import os
        vdir = args.viz_dir or os.path.join(os.getcwd(), "verify_m4_viz")
        os.makedirs(vdir, exist_ok=True)
        _render_dem_grid(on, os.path.join(vdir, "m4_ON_registration.png"),
                         "M4 ON — anchor + cross-pass registration (static scene → passes should match)")
        if not args.on_only:
            _render_dem_grid(off, os.path.join(vdir, "m4_OFF_footprint_only.png"),
                             "M4 OFF — frozen footprint only (no registration → passes drift)")
        print(f"\n  viz written to: {vdir}")

    print("\n=== VERDICT ===")
    for v in verdict:
        print("  " + v)
    print()
    # Non-zero exit if any WARN/INCONCLUSIVE, so this is scriptable.
    sys.exit(0 if all(v.startswith("PASS") for v in verdict) else 1)


if __name__ == "__main__":
    main()
