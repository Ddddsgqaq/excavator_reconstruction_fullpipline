"""global_dem.py — persistent incremental DEM fusion for streaming (M5.1 + M5.2).

M0–M4 reconstruct one DEM per pass and (M4) freeze a world-frame anchor so passes share
gravity, scale and horizontal pose. M5 goes further: instead of throwing away each pass and
re-rasterising a single tile, we keep a PERSISTENT global height grid in that fixed world
frame and fuse every pass's ground points into it. The world map therefore *grows* as the
excavator moves and *evolves* over time (a pit that gets dug keeps getting deeper).

Because M4 already transforms each pass's `ground_xyz` into the anchor's fixed frame (see
pipeline.py step 5b), every pass's points land in ONE consistent coordinate system — so
fusion is just "splat this pass's points into the shared grid".

Two responsibilities:
  * M5.1 — `integrate(ground_xyz, t)`: rasterise the pass to a per-cell observation
    (top-percentile height + point count), time-decay existing weights, then weighted-average
    the observation in. Fast-follow tuning (aggressive decay) so digging — a height DROP — is
    tracked within a few passes instead of being held up by the old higher average.
  * M5.2 — `changed_tiles()`: slice the global grid into Unity-aligned tiles, diff each
    touched tile against its last published snapshot, and return only the tiles that changed
    beyond a threshold. Static tiles are not re-published (Unity only refreshes what moves).

Pure numpy/scipy; NO torch (deferred/no heavy imports) so importing this stays cheap. Only
the recon thread touches a GlobalDem instance (integrate + changed_tiles run there), so no
locking is needed; status() returns a plain value snapshot.

Coordinate/layout conventions (match terrain_analysis.py:_cell_index and the DEM layout):
  aligned frame X-right, Z-forward, Y-up; grid rows = Z, cols = X, row-major (data[z*W + x]).
Tile placement matches Unity TerrainTileManager.TileToWorldOrigin = tile_index * tile_size_m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np


@dataclass
class FusionConfig:
    """Tuning for the global fusion grid. Defaults target 'fast-follow' digging."""
    world_size_m: float = 150.0       # global grid edge length (anchor ±75 m)
    tile_size_m: float = 50.0         # one Unity tile edge (must match Unity tileSizeMeters)
    tile_res: int = 128               # cells per tile edge (matches the 128 contract)
    top_percentile: float = 70.0      # per-cell observation height percentile (noise-robust,
                                      #   lower than 90 so a dug pit pulls the cell down faster)
    decay: float = 0.5                # old-weight multiplier per t_ref (fast-follow core)
    t_ref: float = 6.0                # decay reference interval in seconds (≈ one pass T)
    n_ref: float = 8.0               # points-per-cell that map to observation weight 1.0
    w_cap: float = 4.0                # cap on a single pass's observation weight
    min_pts_per_cell: int = 1         # a cell needs this many points to count as observed
    change_thresh: float = 0.05       # max|Δh| (m) for a tile to count as "changed"
    height_resolution: float = 0.01   # int16 quantisation step passed to dem_to_elevation_msg


@dataclass
class TileUpdate:
    """One tile whose fused heights changed enough this pass to be republished."""
    tile_x: int
    tile_y: int
    elev: np.ndarray                  # (tile_res, tile_res) float64, NaN where no data
    has_data: np.ndarray              # (tile_res, tile_res) bool
    x_bounds: tuple[float, float]     # world X extent of this tile
    z_bounds: tuple[float, float]     # world Z extent of this tile
    max_delta: float                  # largest |Δh| vs last publish (diagnostic)


def _cell_index(coord, lo, cells_per_m, res):
    """Continuous coord → integer cell in [0, res) (clipped). Same mapping as
    terrain_analysis.py:52 but expressed via cells-per-metre so X and Z share spacing:
    cell = floor((coord - lo) / cell_m) = floor((coord - lo) * cells_per_m)."""
    idx = ((coord - lo) * cells_per_m).astype(np.int64)
    return np.clip(idx, 0, res - 1)


class GlobalDem:
    """Persistent world-frame height grid that fuses per-pass ground points over time."""

    def __init__(self, origin_xz: tuple[float, float], cfg: FusionConfig | None = None):
        self.cfg = cfg or FusionConfig()
        cfg = self.cfg

        # Grid geometry: a square world grid centred on the anchor origin. We derive the
        # global cell count from an INTEGER number of tiles so tile slicing is exact (no
        # fractional-tile drift): n_tiles tiles per edge, each tile_res cells.
        self.n_tiles = max(1, int(round(cfg.world_size_m / cfg.tile_size_m)))
        # keep it odd so the anchor tile sits in the centre (tile indices symmetric ±k)
        if self.n_tiles % 2 == 0:
            self.n_tiles += 1
        self.tile_res = int(cfg.tile_res)
        self.G = self.n_tiles * self.tile_res              # global grid edge in cells
        self.cell_m = cfg.tile_size_m / self.tile_res      # metres per cell
        self._cells_per_m = 1.0 / self.cell_m              # for _cell_index over the grid

        # World extent, TILE-ALIGNED. The grid's lower corner must fall exactly on a Unity
        # tile boundary (a multiple of tile_size_m), else every tile is half-offset from
        # Unity's TileToWorldOrigin = tile_index * tile_size_m. So we index tiles by the
        # Unity convention: the tile CONTAINING the anchor origin is floor(origin/tile_size),
        # and the grid spans `half` tiles either side of it.
        half = self.n_tiles // 2                            # tiles either side of centre
        cx, cz = float(origin_xz[0]), float(origin_xz[1])
        center_tx = int(np.floor(cx / cfg.tile_size_m))
        center_tz = int(np.floor(cz / cfg.tile_size_m))
        self.tile_x0 = center_tx - half                     # lowest tile index in the grid
        self.tile_y0 = center_tz - half
        self.x_min = self.tile_x0 * cfg.tile_size_m         # exactly tile-aligned
        self.z_min = self.tile_y0 * cfg.tile_size_m
        self.x_max = self.x_min + self.G * self.cell_m
        self.z_max = self.z_min + self.G * self.cell_m

        # Per-cell fused state.
        self.H = np.full((self.G, self.G), np.nan, dtype=np.float64)   # fused height
        self.W = np.zeros((self.G, self.G), dtype=np.float64)          # accumulated weight
        self.T = np.zeros((self.G, self.G), dtype=np.float64)          # last-update time

        # M5.2 bookkeeping: last-published tile snapshots + which tiles got new points.
        self._last_pub: dict[tuple[int, int], np.ndarray] = {}
        self._touched: set[tuple[int, int]] = set()

        # diagnostics
        self._n_passes = 0
        self._last_changed: list[tuple[int, int]] = []

    # ── M5.1 fusion ─────────────────────────────────────────────
    def integrate(self, surface_xyz: np.ndarray, t: float, *, aggregation: str = "percentile",
                  min_change_m: float | None = None, max_changed_fraction: float = 0.35,
                  min_change_neighbors: int = 3) -> dict:
        """Fuse aligned surface samples into the global grid and return a decision report.

        ``aggregation="mean"`` is used for samples taken from an already interpolated
        Elevation Viewer DEM; percentile mode remains available for legacy raw-point callers.
        """
        pts = np.asarray(surface_xyz, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            return {"decision": "rejected", "reason": "empty_observation"}
        self._n_passes += 1

        # Drop points outside the world grid (they'd clip onto the border otherwise).
        inb = ((pts[:, 0] >= self.x_min) & (pts[:, 0] < self.x_max) &
               (pts[:, 2] >= self.z_min) & (pts[:, 2] < self.z_max))
        pts = pts[inb]
        if pts.shape[0] == 0:
            return {"decision": "rejected", "reason": "outside_global_grid"}

        ix = _cell_index(pts[:, 0], self.x_min, self._cells_per_m, self.G)
        iz = _cell_index(pts[:, 2], self.z_min, self._cells_per_m, self.G)
        flat = iz * self.G + ix
        y = pts[:, 1]

        # Per-cell observation: point count + top-percentile height.
        # Vectorised group-percentile (mirrors terrain_analysis.rasterize_bev lines 104-119).
        order = np.argsort(flat, kind="stable")
        flat_s = flat[order]
        y_s = y[order]
        boundaries = np.flatnonzero(np.diff(flat_s)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(flat_s)]))

        cells = flat_s[starts]                         # unique occupied cells (flat idx)
        counts = ends - starts
        if aggregation not in {"percentile", "mean"}:
            raise ValueError(f"unsupported aggregation: {aggregation}")
        h_obs = np.empty(cells.shape[0], dtype=np.float64)
        for k, (s, e) in enumerate(zip(starts, ends)):
            h_obs[k] = (float(np.mean(y_s[s:e])) if aggregation == "mean"
                        else float(np.percentile(y_s[s:e], self.cfg.top_percentile)))

        keep = counts >= self.cfg.min_pts_per_cell
        cells, counts, h_obs = cells[keep], counts[keep], h_obs[keep]
        if cells.shape[0] == 0:
            return {"decision": "rejected", "reason": "insufficient_samples"}

        cz = cells // self.G
        cx = cells % self.G
        h_old_all = self.H[cz, cx]
        fresh_all = ~np.isfinite(h_old_all)
        overlap_all = ~fresh_all
        vertical_bias = 0.0
        changed_all = np.ones(cells.shape[0], dtype=bool)
        changed_fraction = 0.0

        if min_change_m is not None and overlap_all.any():
            # Most overlap is assumed unchanged. Remove the robust whole-pass vertical
            # offset before classifying local terrain change.
            vertical_bias = float(np.median(h_obs[overlap_all] - h_old_all[overlap_all]))
            h_obs = h_obs - vertical_bias
            changed_all = fresh_all | (np.abs(h_obs - h_old_all) >= float(min_change_m))
            changed_overlap = changed_all & overlap_all

            # Reject isolated changed cells: reconstruction speckle must not mutate the map.
            if min_change_neighbors > 1 and changed_overlap.any():
                changed_cells = set(zip(cz[changed_overlap].tolist(), cx[changed_overlap].tolist()))
                supported = np.zeros_like(changed_all)
                for q in np.flatnonzero(changed_overlap):
                    r, c = int(cz[q]), int(cx[q])
                    neighbours = sum((r + dr, c + dc) in changed_cells
                                     for dr in (-1, 0, 1) for dc in (-1, 0, 1))
                    supported[q] = neighbours >= int(min_change_neighbors)
                changed_all = fresh_all | supported
                changed_overlap = changed_all & overlap_all

            changed_fraction = float(changed_overlap.sum() / max(1, overlap_all.sum()))
            if changed_fraction > float(max_changed_fraction):
                return {
                    "decision": "rejected", "reason": "changed_fraction_too_large",
                    "observed_cells": int(cells.size), "overlap_cells": int(overlap_all.sum()),
                    "changed_cells": int(changed_overlap.sum()),
                    "changed_fraction": changed_fraction, "vertical_bias_m": vertical_bias,
                }
            if not changed_all.any():
                return {
                    "decision": "unchanged", "reason": "below_change_threshold",
                    "observed_cells": int(cells.size), "overlap_cells": int(overlap_all.sum()),
                    "changed_cells": 0, "changed_fraction": 0.0,
                    "vertical_bias_m": vertical_bias,
                }

        cells, counts, h_obs = cells[changed_all], counts[changed_all], h_obs[changed_all]
        cz, cx = cz[changed_all], cx[changed_all]

        # 1) time-decay existing weight only for cells accepted as real changes.
        dt = np.maximum(0.0, t - self.T[cz, cx])
        # first-ever-observed cells have T=0 → treat as no prior weight regardless of dt
        w_old = self.W[cz, cx] * (self.cfg.decay ** (dt / self.cfg.t_ref))
        h_old = self.H[cz, cx]
        fresh = ~np.isfinite(h_old)                    # never observed before
        w_old = np.where(fresh, 0.0, w_old)

        # 2) observation weight (more points = more trust, capped).
        w_obs = np.minimum(counts / self.cfg.n_ref, self.cfg.w_cap)

        # 3) weighted average (fresh cells take the observation directly).
        denom = w_old + w_obs
        h_new = np.where(denom > 0,
                         (np.where(fresh, 0.0, h_old) * w_old + h_obs * w_obs) / np.maximum(denom, 1e-12),
                         h_obs)
        self.H[cz, cx] = h_new
        self.W[cz, cx] = denom
        self.T[cz, cx] = t

        # record touched tiles for change detection
        for txi, tyi in self._cells_to_tiles(cx, cz):
            self._touched.add((txi, tyi))
        return {
            "decision": "updated", "reason": "new_or_confirmed_change",
            "observed_cells": int(overlap_all.size), "overlap_cells": int(overlap_all.sum()),
            "changed_cells": int((changed_all & overlap_all).sum()),
            "new_cells": int(fresh_all.sum()), "changed_fraction": changed_fraction,
            "vertical_bias_m": vertical_bias,
        }

    # ── M5.2 tile slicing + change detection ────────────────────
    def _cells_to_tiles(self, cx: np.ndarray, cz: np.ndarray):
        """Map global cell (col,row) arrays → the set of (tile_x, tile_y) indices they hit."""
        tx = self.tile_x0 + (cx // self.tile_res)
        ty = self.tile_y0 + (cz // self.tile_res)
        return set(zip(tx.tolist(), ty.tolist()))

    def _tile_slice(self, tile_x: int, tile_y: int):
        """Row/col slices into the global grid for a given tile index (or None if outside)."""
        col0 = (tile_x - self.tile_x0) * self.tile_res
        row0 = (tile_y - self.tile_y0) * self.tile_res
        if col0 < 0 or row0 < 0 or col0 >= self.G or row0 >= self.G:
            return None
        return slice(row0, row0 + self.tile_res), slice(col0, col0 + self.tile_res)

    def _tile_bounds(self, tile_x: int, tile_y: int):
        """World (x_bounds, z_bounds) of a tile — origin = tile_index * tile_size_m (Unity)."""
        x0 = tile_x * self.cfg.tile_size_m
        z0 = tile_y * self.cfg.tile_size_m
        return (x0, x0 + self.cfg.tile_size_m), (z0, z0 + self.cfg.tile_size_m)

    def changed_tiles(self) -> list[TileUpdate]:
        """Return tiles touched this pass whose fused heights changed beyond change_thresh.

        A tile is emitted if it is newly non-empty, or its max |Δh| vs the last published
        snapshot exceeds change_thresh. Emitting updates the stored snapshot. Clears the
        touched set (call once per pass, after integrate)."""
        out: list[TileUpdate] = []
        for (tx, ty) in sorted(self._touched):
            sl = self._tile_slice(tx, ty)
            if sl is None:
                continue
            rs, cs = sl
            elev = self.H[rs, cs].copy()
            has = np.isfinite(elev)
            if not has.any():
                continue

            prev = self._last_pub.get((tx, ty))
            if prev is None:
                max_delta = float("inf")               # first appearance → always publish
            else:
                both = has & np.isfinite(prev)
                max_delta = float(np.max(np.abs(elev[both] - prev[both]))) if both.any() else float("inf")
                # also treat newly-appeared valid cells as a change
                newly = has & ~np.isfinite(prev)
                if newly.any():
                    max_delta = float("inf")

            if max_delta > self.cfg.change_thresh:
                self._last_pub[(tx, ty)] = elev.copy()
                xb, zb = self._tile_bounds(tx, ty)
                out.append(TileUpdate(tile_x=tx, tile_y=ty, elev=elev, has_data=has,
                                      x_bounds=xb, z_bounds=zb, max_delta=max_delta))

        self._last_changed = [(u.tile_x, u.tile_y) for u in out]
        self._touched.clear()
        return out

    # ── diagnostics ─────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "grid_cells": int(self.G),
            "n_tiles_edge": int(self.n_tiles),
            "cell_m": round(self.cell_m, 4),
            "observed_cells": int(np.isfinite(self.H).sum()),
            "tiles_published": len(self._last_pub),
            "last_changed_tiles": list(self._last_changed),
            "passes": self._n_passes,
            "world_bounds": [round(self.x_min, 2), round(self.x_max, 2),
                             round(self.z_min, 2), round(self.z_max, 2)],
        }

    def viewer_dem(self) -> dict | None:
        """Return the observed portion of the fused grid as a JSON-ready float DEM."""
        valid = np.isfinite(self.H)
        if not valid.any():
            return None
        rows, cols = np.nonzero(valid)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        elev = self.H[r0:r1, c0:c1]
        has = np.isfinite(elev)
        return {
            "rows": int(elev.shape[0]), "cols": int(elev.shape[1]),
            "x_min": float(self.x_min + (c0 + 0.5) * self.cell_m),
            "x_max": float(self.x_min + (c1 - 0.5) * self.cell_m),
            "z_min": float(self.z_min + (r0 + 0.5) * self.cell_m),
            "z_max": float(self.z_min + (r1 - 0.5) * self.cell_m),
            "elev": np.where(has, elev, 0.0).tolist(),
            "has_data": has.astype(np.uint8).tolist(),
        }

    def to_snapshot(self) -> dict[str, np.ndarray]:
        """Return a pickle-free numpy snapshot suitable for atomic persistence."""
        keys = np.asarray(sorted(self._last_pub), dtype=np.int64).reshape(-1, 2)
        pubs = (np.stack([self._last_pub[tuple(key)] for key in keys], axis=0)
                if len(keys) else np.empty((0, self.tile_res, self.tile_res), dtype=np.float64))
        return {
            "format_version": np.asarray([1], dtype=np.int64),
            "config_json": np.asarray([json.dumps(self.cfg.__dict__, sort_keys=True)]),
            "origin_xz": np.asarray([
                self.x_min + 0.5 * self.G * self.cell_m,
                self.z_min + 0.5 * self.G * self.cell_m,
            ], dtype=np.float64),
            "H": self.H.copy(), "W": self.W.copy(), "T": self.T.copy(),
            "last_pub_keys": keys, "last_pub_values": pubs,
            "n_passes": np.asarray([self._n_passes], dtype=np.int64),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, np.ndarray]) -> "GlobalDem":
        version = int(np.asarray(snapshot["format_version"]).reshape(-1)[0])
        if version != 1:
            raise ValueError(f"unsupported GlobalDem snapshot version: {version}")
        raw_cfg = str(np.asarray(snapshot["config_json"]).reshape(-1)[0])
        obj = cls(tuple(np.asarray(snapshot["origin_xz"], dtype=float).tolist()),
                  FusionConfig(**json.loads(raw_cfg)))
        for name in ("H", "W", "T"):
            value = np.asarray(snapshot[name], dtype=np.float64)
            if value.shape != (obj.G, obj.G):
                raise ValueError(f"invalid {name} shape in GlobalDem snapshot: {value.shape}")
            setattr(obj, name, value.copy())
        keys = np.asarray(snapshot["last_pub_keys"], dtype=np.int64).reshape(-1, 2)
        values = np.asarray(snapshot["last_pub_values"], dtype=np.float64)
        obj._last_pub = {tuple(key.tolist()): values[i].copy() for i, key in enumerate(keys)}
        obj._n_passes = int(np.asarray(snapshot["n_passes"]).reshape(-1)[0])
        return obj
