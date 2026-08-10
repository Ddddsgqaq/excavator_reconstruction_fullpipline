"""
3D semantic-label fusion for multi-view segmentation.

Problem this solves
--------------------
A scene is reconstructed from S images. VGGT unprojects every pixel of every
frame into a single shared 3D point cloud (`world_points` (S, H, W, 3)). YOLOe
segments each frame *independently*, producing `semantic_masks` (S, H, W).

Because segmentation is per-frame, two failure modes appear in the merged cloud:
  1. A frame that *missed* the target leaves that object's points unlabeled
     → holes when extracting, leftover points when deleting.
  2. A frame with a *spurious* detection drops stray mislabeled points into the
     cloud → speckle artifacts.

Since `semantic_masks.reshape(-1)` is point-aligned with
`world_points.reshape(-1, 3)`, both issues can be fixed *in 3D, once*, instead of
trusting every 2D frame: cluster the labeled points in 3D (dropping tiny noise
clusters), then relabel co-located points the skipped frames missed. Coverage
gaps from missed frames get filled; isolated mislabels vanish.

Avoiding "bleed" onto the ground
--------------------------------
The naive relabel step — "label every point within a radius of the object
cluster" — over-grows: an excavator sits on / touches the ground, so any ball
around its surface points also swallows the adjacent ground points (on dense
real clouds this mislabels far more ground than object). The fix is a
foreground/background *contest* (`background_compete`, default on): a candidate
keeps the object label only if it is nearer to an object anchor than to ANY
background anchor, where background anchors are the points that segmented
(keyframe) frames labeled as not-this-object. Ground next to the excavator is
always closer to its own ground anchors, so it loses the contest and stays
background. The relabel radius then only caps reach; the contest sets the
boundary, which makes the result robust to the radius value.

This module is pure NumPy/SciPy (no sklearn): density clustering is done with a
KD-tree radius graph + connected components (single-linkage at `eps`), which is
equivalent to DBSCAN's reachability with `min_samples` mapped to a minimum
cluster size.
"""

from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _cluster_points(pts: np.ndarray, eps: float, min_cluster_size: int) -> np.ndarray:
    """Single-linkage density clustering via KD-tree radius graph.

    Returns a label array (len == len(pts)); -1 marks noise (clusters smaller
    than `min_cluster_size`).
    """
    n = pts.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    if n == 1:
        # A single point can only survive if a cluster of size 1 is allowed.
        return np.array([0 if min_cluster_size <= 1 else -1], dtype=np.int64)

    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")  # (M, 2)
    if pairs.shape[0] == 0:
        # No edges: every point is its own component.
        comp_labels = np.arange(n)
        n_comp = n
    else:
        data = np.ones(pairs.shape[0], dtype=np.uint8)
        graph = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        n_comp, comp_labels = connected_components(
            graph, directed=False, connection="weak"
        )

    # Keep only components meeting the minimum size; renumber survivors 0..k-1.
    # Vectorized via a lookup table (LUT) instead of a per-point Python loop.
    counts = np.bincount(comp_labels, minlength=n_comp)
    keep = np.where(counts >= min_cluster_size)[0]
    lut = np.full(n_comp, -1, dtype=np.int64)
    lut[keep] = np.arange(keep.shape[0])
    labels = lut[comp_labels]
    return labels


def refine_masks_3d(
    world_points: np.ndarray,
    semantic_masks: np.ndarray,
    conf: Optional[np.ndarray] = None,
    conf_thres: float = 0.0,
    selected_ids: Optional[Sequence[int]] = None,
    eps: float = 0.05,
    min_cluster_size: int = 30,
    dilate_radius: Optional[float] = None,
    max_seed_points: int = 20_000,
    background_compete: bool = True,
    segmented_frames: Optional[Sequence[int]] = None,
    max_bg_points: int = 300_000,
    seed_conf_thres: Optional[float] = None,
) -> np.ndarray:
    """Make per-frame semantic labels consistent in 3D.

    Parameters
    ----------
    world_points : (S, H, W, 3) float
        Per-pixel 3D coordinates in a shared frame.
    semantic_masks : (S, H, W) int
        Per-pixel semantic id (0 = background/unlabeled).
    conf : (S, H, W) float, optional
        Per-point confidence; points below the percentile `seed_conf_thres`
        (which defaults to `conf_thres`) are not used as cluster seeds (but can
        still *receive* a label via relabeling).
    conf_thres : float
        Percentile in [0, 100] applied to `conf` to gate seed points. NOTE: in
        the pipeline this carries the *global* cloud-confidence threshold (often
        ~50), which is too aggressive for seeds because target objects (e.g. an
        excavator) are systematically low-confidence — at 50 it discards ~95% of
        object seeds. Prefer setting `seed_conf_thres` to decouple the two.
    selected_ids : sequence of int, optional
        Which semantic ids to refine. Defaults to every non-zero id present.
    eps : float
        Neighbour radius for clustering, in world units.
    min_cluster_size : int
        Clusters smaller than this are treated as noise and dropped.
    dilate_radius : float, optional
        Max relabeling radius around surviving clusters. Defaults to `eps` when
        None. With `background_compete` it only caps how far a label may reach;
        the foreground/background contest (not this radius) sets the boundary.
    max_seed_points : int
        Cap on seed points per id (random-free uniform subsample) to bound the
        KD-tree / radius-graph cost on dense clouds. Default 20_000: on a dense
        scene this is the dominant fusion cost (`_cluster_points` builds a radius
        graph over the seeds), so capping seeds keeps clustering ~1s instead of
        ~6s with negligible IoU change. When the seed count is below the cap no
        subsampling happens, so sparse inputs are unaffected.
    background_compete : bool
        If True (default), a candidate point is relabeled to `sid` only when it
        is closer to a foreground (object) anchor than to ANY background anchor.
        Background anchors are the points that segmented frames labeled as *not*
        this id (i.e. trustworthy background). This fixes the over-dilation /
        "bleed" failure of one-sided radius growth: because an object sits on /
        touches the ground, a pure radius around object seeds also swallows the
        adjacent ground; the contest instead lets the ground's own (much closer)
        background anchors win those points back. If False, fall back to the
        original one-sided dilation (every point within `dilate_radius` of a
        cluster gets the label).
    segmented_frames : sequence of int, optional
        Indices of frames that were actually segmented (keyframes). Their
        non-`sid` pixels become background anchors. Defaults to inferring "any
        frame that carries a non-zero label" — correct for keyframe-only
        segmentation, where skipped frames are all-zero and must be recovered.
    max_bg_points : int
        Cap on background anchors (uniform subsample) to bound KD-tree cost.
    seed_conf_thres : float, optional
        Percentile in [0, 100] used to gate *seed* points, decoupled from the
        global `conf_thres`. When None, falls back to `conf_thres`. Set this to a
        gentle value (0–10) so low-confidence object points still seed the
        clusters; the global `conf_thres` can stay high for cloud display without
        starving fusion of seeds.

    Returns
    -------
    refined : (S, H, W) int
        New semantic mask. Selected ids are recomputed from the 3D clusters;
        unselected ids are carried over unchanged.
    """
    S, H, W = semantic_masks.shape
    flat_pts = world_points.reshape(-1, 3)
    flat_sem = semantic_masks.reshape(-1).astype(np.int64)
    finite = np.isfinite(flat_pts).all(axis=1)

    if dilate_radius is None:
        dilate_radius = eps

    # Confidence gate for seeds. Decoupled from the global cloud `conf_thres`:
    # target objects are systematically low-confidence, so reusing a high global
    # threshold here would discard most object seeds. `seed_conf_thres` overrides.
    seed_thr = conf_thres if seed_conf_thres is None else seed_conf_thres
    if conf is not None and seed_thr > 0:
        conf_flat = conf.reshape(-1)
        thr = np.percentile(conf_flat[np.isfinite(conf_flat)], seed_thr)
        seed_ok = finite & (conf_flat >= thr)
    else:
        seed_ok = finite

    if selected_ids is None:
        selected_ids = [int(i) for i in np.unique(flat_sem) if i != 0]

    # Which pixels belong to a segmented (keyframe) frame? Their non-object
    # labels are trustworthy background; all-zero (skipped) frames are unknowns
    # to be recovered. Used only by the background-competition path.
    if background_compete:
        if segmented_frames is not None:
            seg_frame = np.zeros(S, dtype=bool)
            seg_frame[np.asarray(list(segmented_frames), dtype=np.int64)] = True
        else:
            seg_frame = (flat_sem.reshape(S, -1) != 0).any(axis=1)
        seg_px = np.repeat(seg_frame, H * W)

    refined = flat_sem.copy()
    # Clear the selected ids so we can rewrite them from 3D evidence; other ids
    # (and background) are left as-is.
    sel_arr = np.asarray(list(selected_ids), dtype=np.int64)
    refined[np.isin(refined, sel_arr)] = 0

    # Build the candidate set once: all finite points are eligible to receive a
    # label (seeds included, so surviving clusters get re-stamped).
    cand_idx = np.where(finite)[0]
    cand_pts = flat_pts[cand_idx]

    for sid in selected_ids:
        seed_mask = (flat_sem == sid) & seed_ok
        seed_idx = np.where(seed_mask)[0]
        if seed_idx.size == 0:
            continue

        # Bound seed count for tractability via uniform stride subsampling.
        if seed_idx.size > max_seed_points:
            stride = int(np.ceil(seed_idx.size / max_seed_points))
            seed_idx = seed_idx[::stride]

        seed_pts = flat_pts[seed_idx]
        cl = _cluster_points(seed_pts, eps=eps, min_cluster_size=min_cluster_size)
        valid = cl >= 0
        if not valid.any():
            # No cluster survived the noise filter → keep this id empty.
            continue
        valid_seed_pts = seed_pts[valid]

        # Distance from every candidate to the nearest surviving object cluster,
        # capped at dilate_radius (inf where no cluster point is in range).
        cluster_tree = cKDTree(valid_seed_pts)
        d_obj, _ = cluster_tree.query(
            cand_pts, k=1, distance_upper_bound=dilate_radius, workers=-1
        )
        in_range = np.isfinite(d_obj)

        # Background anchors: points that segmented frames labeled as NOT this id
        # (trustworthy background). A candidate is kept for `sid` only if it is
        # nearer to the object than to any such background point — so ground next
        # to the object is reclaimed by its own (closer) ground anchors.
        bg_idx = np.where(seg_px & finite & (flat_sem != sid)) [0] \
            if background_compete else np.empty(0, dtype=np.int64)
        if background_compete and bg_idx.size > 0:
            if bg_idx.size > max_bg_points:
                bg_idx = bg_idx[::int(np.ceil(bg_idx.size / max_bg_points))]
            bg_tree = cKDTree(flat_pts[bg_idx])
            d_bg, _ = bg_tree.query(cand_pts, k=1, workers=-1)
            hit_mask = in_range & (d_obj < d_bg)
        else:
            # No background to contest (or feature disabled) → one-sided dilation.
            hit_mask = in_range

        refined[cand_idx[hit_mask]] = sid

    return refined.reshape(S, H, W).astype(semantic_masks.dtype)
