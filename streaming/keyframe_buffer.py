"""keyframe_buffer.py — online rolling keyframe buffer for streaming reconstruction (M2).

Feeds the M1 `reconstruct_frames_to_dem`: as frames arrive from a FrameSource, decide
online which ones are "new enough" viewpoints to keep, hold the most recent N in a ring,
and hand a thread-safe snapshot to the reconstruction loop (M3).

The keep/skip criterion is the SAME ORB viewpoint metric the offline path uses
(`yoloe_service.select_keyframe_indices` mode="similarity"): downscale to 320px, ORB
match, keep a frame when its viewpoint similarity to the last kept keyframe drops below a
threshold. We reimplement it here against in-memory np.ndarray frames (the offline version
takes file paths and runs as a batch) — the offline code is left untouched (0b constraint).

Thread model: the frame pump thread calls `offer(frame)`; the reconstruction loop thread
calls `snapshot()`. Both take a lock; snapshots are shallow copies of the frame list.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# Mirror the offline ORB constants (yoloe_service.py) so keep/skip matches.
_ORB_FEATURES = 1500
_ORB_RATIO = 0.75
_ORB_RANSAC = 4.0
_ORB_WIDTH = 320


def _frame_signature(img: np.ndarray):
    """ORB signature (keypoint xy, descriptors, diag) for an in-memory RGB/gray frame.

    Downscales to _ORB_WIDTH first (same as offline): the similarity score is
    scale-invariant, so this only saves time, it doesn't change decisions.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    h, w = gray.shape[:2]
    if _ORB_WIDTH and w > _ORB_WIDTH:
        scale = _ORB_WIDTH / w
        gray = cv2.resize(gray, (_ORB_WIDTH, int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    kp, des = orb.detectAndCompute(gray, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.empty((0, 2), np.float32)
    diag = float(np.hypot(*gray.shape[:2]))
    return pts, des, diag


def _frame_similarity(sig_a, sig_b) -> float:
    """Viewpoint similarity in [0,1] (1 = same view). Identical math to the offline path."""
    ptsA, desA, diag = sig_a
    ptsB, desB, _ = sig_b
    if desA is None or desB is None or len(ptsA) < 8 or len(ptsB) < 8:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(desA, desB, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < _ORB_RATIO * n.distance]
    if len(good) < 8:
        return 0.0
    pA = np.float32([ptsA[m.queryIdx] for m in good])
    pB = np.float32([ptsB[m.trainIdx] for m in good])
    H, mask = cv2.findHomography(pA, pB, cv2.RANSAC, _ORB_RANSAC)
    if H is None or mask is None:
        return 0.0
    mask = mask.ravel().astype(bool)
    n_in = int(mask.sum())
    if n_in == 0:
        return 0.0
    inlier_ratio = n_in / len(good)
    parallax = float(np.median(np.linalg.norm(pA[mask] - pB[mask], axis=1)))
    par_term = 1.0 / (1.0 + (parallax / diag / 0.06) ** 2)
    return float(inlier_ratio * par_term)


@dataclass
class BufferStats:
    offered: int = 0        # frames offered by the pump
    kept: int = 0           # frames accepted as keyframes
    skipped: int = 0        # frames rejected as too-similar
    evicted: int = 0        # keyframes dropped from the ring (age-out)
    window: int = 0         # current keyframes held
    selection_mode: str = "orb"
    orb_calls: int = 0
    orb_total_seconds: float = 0.0
    orb_last_ms: float = 0.0


class KeyframeBuffer:
    """Thread-safe rolling window of the most recent N keyframes.

    A frame is kept when its ORB viewpoint similarity to the last kept keyframe falls
    below `sim_thresh` (i.e. the view moved enough). The window holds at most `capacity`
    keyframes; the oldest is evicted when full — this is the A-scheme sliding window.

    `sim_thresh` default 0.92 matches the offline `select_keyframe_indices(mode="similarity")`
    default: higher keeps MORE frames (stricter "must differ"), lower keeps fewer. A
    near-static camera legitimately yields few keyframes; that's correct, not a bug.
    """

    def __init__(self, capacity: int = 12, sim_thresh: float = 0.92,
                 use_orb: bool = True):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.sim_thresh = sim_thresh
        self.use_orb = bool(use_orb)
        self._frames: deque[np.ndarray] = deque(maxlen=capacity)
        self._last_sig = None
        self._lock = threading.Lock()
        self._stats = BufferStats(selection_mode="orb" if self.use_orb else "interval")

    def offer(self, frame: np.ndarray) -> bool:
        """Offer a frame from the pump. Returns True if kept as a keyframe.

        The first frame is always kept. Subsequent frames are kept only if their
        viewpoint differs enough from the last kept keyframe.
        """
        frame = np.asarray(frame)
        sig = None
        if self.use_orb:
            started = time.perf_counter()
            sig = _frame_signature(frame)
        with self._lock:
            self._stats.offered += 1
            if not self.use_orb:
                keep = True
            elif self._last_sig is None:
                keep = True
            else:
                sim = _frame_similarity(self._last_sig, sig)
                keep = sim < self.sim_thresh
            if self.use_orb:
                orb_seconds = time.perf_counter() - started
                self._stats.orb_calls += 1
                self._stats.orb_total_seconds += orb_seconds
                self._stats.orb_last_ms = orb_seconds * 1000.0
            if keep:
                if len(self._frames) == self.capacity:
                    self._stats.evicted += 1  # deque maxlen auto-drops oldest
                self._frames.append(frame)
                if self.use_orb:
                    self._last_sig = sig
                self._stats.kept += 1
            else:
                self._stats.skipped += 1
            self._stats.window = len(self._frames)
            return keep

    def snapshot(self) -> list[np.ndarray]:
        """Return a shallow copy of the current keyframe window (oldest→newest).

        Safe to call from another thread; the returned list won't change under the
        reconstruction loop even as the pump keeps offering frames.
        """
        with self._lock:
            return list(self._frames)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def stats(self) -> BufferStats:
        with self._lock:
            # return a copy so callers can't mutate internal state
            s = self._stats
            return BufferStats(
                offered=s.offered, kept=s.kept, skipped=s.skipped,
                evicted=s.evicted, window=s.window,
                selection_mode=s.selection_mode, orb_calls=s.orb_calls,
                orb_total_seconds=s.orb_total_seconds, orb_last_ms=s.orb_last_ms,
            )
