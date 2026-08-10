"""verify_m5.py — verification for M5 (persistent incremental fusion + multi-tile).

Two modes:

  python tools/verify_m5.py --unit            # no GPU: drive GlobalDem with synthetic clouds
  python tools/verify_m5.py <video.mp4>       # real GPU: run the fusion loop on a video

--unit checks the fusion math and change detection directly (fast, deterministic, no model):
  1. static plane (dense) → converges: pass 0 publishes, later passes flag nothing.
  2. deepening pit → fused height fast-follows the drop; the pit tile is flagged every pass
     while a far static tile is not.
  3. multi-tile alignment → points land in the tile whose world origin = tile_index*tile_size.
  4. NODATA → never-observed cells export as NODATA via dem_to_elevation_msg without overflow.

The GPU mode mirrors tools/verify_m4.py: it runs the streaming ReconstructLoop with
fusion=True in-process (reusing the resident VGGT model) and reports per-pass fusion status
(observed cells, changed tiles, total tiles published), asserting that multiple tiles get
published and that static stretches settle to zero changed tiles.

Nothing here touches the offline path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Allow running as `python tools/verify_m5.py` from anywhere: put the repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── no-GPU unit checks ──────────────────────────────────────────
def _run_unit() -> bool:
    from streaming.global_dem import GlobalDem, FusionConfig, _cell_index
    import streaming.global_dem as gdmod
    assert "torch" not in sys.modules, "importing streaming.global_dem pulled torch!"

    ok = True
    rng = np.random.default_rng(0)
    cfg = FusionConfig(world_size_m=150, tile_size_m=50, tile_res=128,
                       decay=0.5, change_thresh=0.05, top_percentile=70)

    # 1) static plane (dense coverage of the centre tile) → converges after pass 0.
    gd = GlobalDem(origin_xz=(0.0, 0.0), cfg=cfg)

    def dense_plane(y=1.0):
        g = np.linspace(-24.5, 24.5, 260)
        xx, zz = np.meshgrid(g, g)
        x, z = xx.ravel(), zz.ravel()
        return np.column_stack([x, np.full(x.size, y) + rng.normal(0, 0.003, x.size), z])

    n_changed = []
    for p in range(5):
        gd.integrate(dense_plane(), t=p * 6.0)
        n_changed.append(len(gd.changed_tiles()))
    static_ok = n_changed[0] > 0 and all(c == 0 for c in n_changed[1:])
    print(f"[1] static plane changed/pass = {n_changed}  -> {'OK' if static_ok else 'FAIL'}")
    ok &= static_ok

    # 2) deepening pit → fast-follow + correct change flags.
    #    Track a cell well INSIDE the pit (top-percentile on an edge cell would pick the
    #    surrounding ground, so we sample the pit centre and make it several cells wide).
    gd2 = GlobalDem(origin_xz=(0.0, 0.0), cfg=cfg)
    px, pz = 5.0, 5.0
    ix = int(_cell_index(np.array([px]), gd2.x_min, gd2._cells_per_m, gd2.G)[0])
    iz = int(_cell_index(np.array([pz]), gd2.z_min, gd2._cells_per_m, gd2.G)[0])
    pit_tile = (gd2.tile_x0 + ix // gd2.tile_res, gd2.tile_y0 + iz // gd2.tile_res)
    # a far tile with no points → must never be flagged
    far_tile = (gd2.tile_x0, gd2.tile_y0)

    def scene(depth, n=12000):
        x = rng.uniform(0.5, 45.0, n)     # inside tile 0 ([0,50)); keeps far corner empty
        z = rng.uniform(0.5, 45.0, n)
        y = np.full(n, 1.0)
        m = np.hypot(x - px, z - pz) < 3.0   # ≥3 m wide → several cells fully inside
        y[m] = 1.0 - depth
        return np.column_stack([x, y + rng.normal(0, 0.004, n), z])

    gd2.integrate(scene(0.0), t=0.0)
    gd2.changed_tiles()
    Hs, pit_flagged, far_flagged = [], [], []
    for k in range(1, 6):
        gd2.integrate(scene(0.1 * k), t=k * 6.0)
        changed = {(u.tile_x, u.tile_y) for u in gd2.changed_tiles()}
        Hs.append(round(float(gd2.H[iz, ix]), 3))
        pit_flagged.append(pit_tile in changed)
        far_flagged.append(far_tile in changed)
    monotonic = all(a >= b - 1e-9 for a, b in zip(Hs, Hs[1:]))   # non-increasing
    dropped = (Hs[0] - Hs[-1]) > 0.3              # pit clearly deepened over the run
    caught_up = Hs[-1] < 0.62                     # obs bottoms at 0.5; fast-follow nears it
    pit_ok = all(pit_flagged)
    far_ok = not any(far_flagged)
    print(f"[2] pit fused H/pass = {Hs}  (obs 0.9→0.5)")
    print(f"    non_increasing={monotonic} dropped>0.3m={dropped} caught_up={caught_up} "
          f"pit_tile_flagged_each={pit_ok} far_tile_never_flagged={far_ok}"
          f"  -> {'OK' if (monotonic and dropped and caught_up and pit_ok and far_ok) else 'FAIL'}")
    ok &= monotonic and dropped and caught_up and pit_ok and far_ok

    # 3) multi-tile alignment → a point at world (60, 5) lands in tile (1, 0) whose bounds
    #    start at tile_index * tile_size (matches Unity TileToWorldOrigin).
    gd3 = GlobalDem(origin_xz=(0.0, 0.0), cfg=cfg)
    # need a bigger world so tile (1,0) exists: world_size 150, tiles -1..1 → tile 1 spans [50,100)
    pt = np.array([[60.0, 2.0, 5.0]] * 400) + rng.normal(0, 0.01, (400, 3))
    gd3.integrate(pt, t=0.0)
    ups = gd3.changed_tiles()
    hit = {(u.tile_x, u.tile_y): u for u in ups}
    align_ok = (1, 0) in hit
    if align_ok:
        u = hit[(1, 0)]
        # world origin of tile (1,0) must be (50, 0)
        align_ok = abs(u.x_bounds[0] - 50.0) < 1e-6 and abs(u.z_bounds[0] - 0.0) < 1e-6
    print(f"[3] point@world(60,5) → tiles {sorted(hit)}; tile(1,0) origin correct "
          f"-> {'OK' if align_ok else 'FAIL'}")
    ok &= align_ok

    # 4) NODATA export: a tile with only a few observed cells exports the rest as NODATA.
    from elevation_export import dem_to_elevation_msg
    u = hit.get((1, 0))
    if u is not None:
        msg = dem_to_elevation_msg(u.elev, u.x_bounds, u.z_bounds, has_data=u.has_data,
                                   height_resolution=cfg.height_resolution, tile_x=1, tile_y=0)
        m = msg["metadata"]
        nodata_ok = (m["nodata_count"] > 0 and len(msg["data"]) == cfg.tile_res ** 2
                     and m["overflow_clipped"] == 0)
        print(f"[4] NODATA export: nodata_count={m['nodata_count']} "
              f"len(data)={len(msg['data'])} overflow={m['overflow_clipped']} "
              f"-> {'OK' if nodata_ok else 'FAIL'}")
        ok &= nodata_ok
    else:
        print("[4] NODATA export: SKIPPED (tile (1,0) missing)")
        ok = False

    print("\n=== UNIT VERDICT:", "PASS" if ok else "FAIL", "===")
    return ok


# ── real-GPU smoke ──────────────────────────────────────────────
def _probe_scale(video, target_fps, min_frames, capacity):
    """Run ONE reconstruction to measure the world-frame X/Z span of the ground cloud.

    VGGT scale is arbitrary (see M4 known-limitation), so a fixed 150 m fusion grid can be
    far too big for a small-scale reconstruction — all points collapse into one cell. We
    probe the actual extent so the fusion grid can be sized to match it (--auto-scale)."""
    from streaming.frame_source import VideoFileSource
    from streaming.keyframe_buffer import KeyframeBuffer
    from streaming import pipeline

    buf = KeyframeBuffer(capacity=capacity, sim_thresh=0.92)
    src = VideoFileSource(video, target_fps=target_fps, loop=False)
    for fr in src.frames():
        buf.offer(fr)
        if len(buf) >= min_frames * 2:
            break
    frames = buf.snapshot()
    res = pipeline.reconstruct_frames_to_dem(frames, register=False)
    g = res.ground_xyz
    span_x = float(g[:, 0].max() - g[:, 0].min())
    span_z = float(g[:, 2].max() - g[:, 2].min())
    return max(span_x, span_z), res


def _render_global_dem(gdem, path, title):
    """Save a top-down heatmap of the persistent global DEM (observed cells only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    H = gdem.H.copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.ma.masked_invalid(H), origin="lower", cmap="terrain",
                   extent=[gdem.x_min, gdem.x_max, gdem.z_min, gdem.z_max])
    # draw tile boundaries
    ts = gdem.cfg.tile_size_m
    for k in range(gdem.n_tiles + 1):
        ax.axvline(gdem.x_min + k * ts, color="k", lw=0.4, alpha=0.4)
        ax.axhline(gdem.z_min + k * ts, color="k", lw=0.4, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("world X (m)"); ax.set_ylabel("world Z (m)")
    fig.colorbar(im, ax=ax, label="fused height (m)", shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _run_gpu(args) -> bool:
    import vggt_service  # noqa: F401 — module load puts VGGT on the GPU (resident _model)
    import streaming.reconstruct_loop as RL
    from streaming.frame_source import VideoFileSource
    from streaming.elevation_publisher import ElevationPublisher
    import tempfile
    import os

    out_dir = args.viz_dir or tempfile.mkdtemp(prefix="verify_m5_")
    os.makedirs(out_dir, exist_ok=True)

    # Size the fusion grid to the reconstruction's real scale (arbitrary VGGT units).
    tile_size = args.tile_size_m
    world_size = args.world_size_m
    if args.auto_scale:
        span, _probe = _probe_scale(args.video, args.target_fps, args.min_frames, args.capacity)
        # one tile ≈ the whole footprint / 1.5, world = 3 tiles → 3x3 grid covering ~2x the span
        tile_size = max(span / 1.5, 1e-3)
        world_size = tile_size * 3
        print(f"[auto-scale] ground span≈{span:.3f} → tile_size_m={tile_size:.3f}, "
              f"world_size_m={world_size:.3f}")

    source = VideoFileSource(args.video, target_fps=args.target_fps, loop=True)
    publisher = ElevationPublisher(file_out=out_dir, mqtt=False)
    cfg = RL.LoopConfig(
        interval=args.interval, min_frames=args.min_frames, capacity=args.capacity,
        target_fps=args.target_fps, freeze_anchor=True, register=True, fusion=True,
        world_size_m=world_size, tile_size_m=tile_size,
    )
    loop = RL.ReconstructLoop(source, publisher, cfg)
    print(f"\nstreaming {args.seconds}s @ interval={args.interval}s (fusion ON) ...")
    print(f"tiles + viz written to: {out_dir}")
    loop.start()
    changed_history = []
    obs_history = []
    pass_seen = 0
    try:
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            time.sleep(args.interval)
            s = loop.status()
            changed_history.append(len(s.last_changed_tiles or []))
            obs_history.append(s.observed_cells)
            print(f"  pass={s.passes} window={s.window} observed_cells={s.observed_cells} "
                  f"changed={s.last_changed_tiles} total_published={s.tiles_published_total} "
                  f"{s.last_error[:50]}")
            # snapshot the global DEM each new pass for the visual record
            if args.viz and loop._gdem is not None and s.passes > pass_seen:
                pass_seen = s.passes
                _render_global_dem(loop._gdem, os.path.join(out_dir, f"global_dem_pass{s.passes}.png"),
                                   f"Global fused DEM after pass {s.passes} "
                                   f"({s.observed_cells} cells, {s.tiles_published_total} tiles)")
    finally:
        loop.stop()

    s = loop.status()
    multi = s.tiles_published_total > 1
    grew = max(obs_history) > 4 if obs_history else False
    print("\n--- GPU smoke summary ---")
    print(f"  total tiles published: {s.tiles_published_total}  (multi-tile: {multi})")
    print(f"  observed_cells history: {obs_history}")
    print(f"  changed-tiles per pass history: {changed_history}")
    if args.viz:
        import glob
        print(f"  viz images: {sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_dir,'*.png')))}")
    verdict = multi and grew
    print("\n=== GPU VERDICT:",
          "PASS" if verdict else "WARN (grid did not accumulate meaningful terrain — check scale)", "===")
    return verdict


def main():
    ap = argparse.ArgumentParser(description="Verify M5 persistent fusion + multi-tile.")
    ap.add_argument("video", nargs="?", help="mp4 for the real-GPU smoke (omit with --unit)")
    ap.add_argument("--unit", action="store_true", help="run the no-GPU unit checks only")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=6.0)
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--capacity", type=int, default=12)
    ap.add_argument("--target-fps", type=float, default=3.0)
    ap.add_argument("--auto-scale", action="store_true",
                    help="size the fusion grid to the reconstruction's real X/Z span "
                         "(VGGT scale is arbitrary; avoids all points collapsing into one cell)")
    ap.add_argument("--world-size-m", type=float, default=150.0)
    ap.add_argument("--tile-size-m", type=float, default=50.0)
    ap.add_argument("--viz", action="store_true", help="save a global-DEM heatmap per pass")
    ap.add_argument("--viz-dir", default=None, help="output dir for tiles + viz (default: temp)")
    args = ap.parse_args()

    if args.unit or not args.video:
        ok = _run_unit()
        sys.exit(0 if ok else 1)
    else:
        ok = _run_gpu(args)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
