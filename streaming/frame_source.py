"""frame_source.py — frame pumps that feed the keyframe buffer (M2).

A FrameSource yields frames over time. The streaming loop (M3) runs a pump thread that
pulls frames from a source and offers them to a KeyframeBuffer.

Now: `VideoFileSource` replays a local mp4 at wall-clock pace to emulate a live stream —
so the whole streaming pipeline can be developed before a real camera is wired in.
Later: a `MediaMtxSource` (WHEP/RTSP pull from the excavator camera) implements the same
`frames()` iterator, so only this file changes when switching to a real source.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Iterator

import cv2
import numpy as np


class FrameSource(ABC):
    """A source of frames over time. Subclasses implement `frames()`."""

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        """Yield RGB uint8 frames (H, W, 3). Blocks/sleeps to pace real-time sources."""
        raise NotImplementedError


class VideoFileSource(FrameSource):
    """Replay a video file at a target FPS to emulate a live stream.

    Args:
        path: video file path.
        target_fps: frames yielded per second (wall-clock paced). This is the *emulated
            capture rate*, independent of the file's native fps — we sample the file to
            approximate it. Keep modest (e.g. 2-5): the keyframe buffer + reconstruction
            downstream can't consume 30 fps anyway (a warm pass is ~5 s, see M1 timing).
        loop: restart from the beginning when the file ends (useful for long demos).
        max_frames: stop after yielding this many frames (None = until EOF/loop forever).
    """

    def __init__(self, path: str, target_fps: float = 3.0,
                 loop: bool = False, max_frames: int | None = None):
        self.path = path
        self.target_fps = max(0.1, float(target_fps))
        self.loop = loop
        self.max_frames = max_frames
        self._stop = threading.Event()
        self._latest = None

    def frames(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"VideoFileSource: cannot open {self.path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Sample every `step`-th decoded frame so the *emitted* rate ≈ target_fps.
        step = max(1, int(round(native_fps / self.target_fps)))
        period = 1.0 / self.target_fps  # wall-clock seconds between yielded frames

        emitted = 0
        decoded = 0
        # NOTE: no time.monotonic() at import; we call it live inside the loop.
        next_deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        decoded = 0
                        continue
                    break
                if decoded % step == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # Pace to wall clock: sleep until this frame's deadline.
                    now = time.monotonic()
                    if next_deadline > now:
                        time.sleep(next_deadline - now)
                    next_deadline += period
                    self._latest = frame_rgb
                    yield frame_rgb
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break
                decoded += 1
        finally:
            cap.release()

    def close(self) -> None:
        self._stop.set()

    def latest_frame(self):
        return None if self._latest is None else self._latest.copy()

    def status(self) -> dict:
        return {"connected": self._latest is not None and not self._stop.is_set(), "source": self.path}


class FrameListSource(FrameSource):
    """Yield a fixed list of in-memory frames at a target FPS (for tests, no file I/O)."""

    def __init__(self, frames: list[np.ndarray], target_fps: float = 3.0):
        self._frames = frames
        self.period = 1.0 / max(0.1, float(target_fps))

    def frames(self) -> Iterator[np.ndarray]:
        next_deadline = time.monotonic()
        for fr in self._frames:
            now = time.monotonic()
            if next_deadline > now:
                time.sleep(next_deadline - now)
            next_deadline += self.period
            yield np.asarray(fr)


# Real USB/RTSP sources live in camera_source.py; keeping them separate prevents
# camera backend details from affecting deterministic file replay and tests.
