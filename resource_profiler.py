"""Lightweight stage timing and resource profiling for the offline pipeline.

Profiles are written incrementally so a partial JSON file survives if a service
request fails.  CUDA values come from PyTorch when the calling service passes
its already-imported ``torch`` module; CPU-only callers do not import PyTorch.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


MIB = 1024.0 * 1024.0


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _read_proc_status() -> Dict[str, Optional[float]]:
    """Return current and lifetime-peak RSS in MiB without extra dependencies."""
    values: Dict[str, Optional[float]] = {"rss_mb": None, "rss_peak_mb": None}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    values["rss_mb"] = float(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    values["rss_peak_mb"] = float(line.split()[1]) / 1024.0
    except OSError:
        # Fallback for non-Linux development environments. ru_maxrss is KiB on
        # Linux and bytes on macOS; this is a lifetime peak, not current RSS.
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        values["rss_peak_mb"] = peak / (MIB if sys.platform == "darwin" else 1024.0)
    return values


class ResourceProfiler:
    """Incremental per-stage profiler for one pipeline operation."""

    schema_version = 1

    def __init__(
        self,
        operation: str,
        working_dir: os.PathLike[str] | str,
        *,
        torch_module: Any = None,
        enabled: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        env_enabled = os.environ.get("PIPELINE_RESOURCE_PROFILE", "1").strip().lower()
        self.enabled = (env_enabled not in {"0", "false", "off", "no"}) if enabled is None else enabled
        self.operation = operation
        self.working_dir = Path(working_dir).resolve()
        self.torch = torch_module
        self._lock = threading.Lock()
        self._started_wall = time.perf_counter()
        self._started_cpu = time.process_time()
        self._started_at = datetime.now(timezone.utc)
        self._finished = False

        stamp = self._started_at.strftime("%Y%m%dT%H%M%S_%fZ")
        profile_dir = self.working_dir / "resource_profiles"
        self.path = profile_dir / f"{operation}_{stamp}_pid{os.getpid()}.json"
        self.data: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "operation": operation,
            "status": "running",
            "started_at_utc": self._started_at.isoformat(),
            "completed_at_utc": None,
            "working_dir": str(self.working_dir),
            "profile_path": str(self.path),
            "pid": os.getpid(),
            "metadata": metadata or {},
            "measurement_notes": {
                "wall_time": "CUDA is synchronized at stage boundaries when available.",
                "rss_mb": "Current Linux process resident memory at the boundary.",
                "rss_peak_mb": "Lifetime process RSS high-water mark, not a per-stage-only peak.",
                "cuda_allocated_mb": "Live tensors owned by the current PyTorch process.",
                "cuda_reserved_mb": "Memory held by the current PyTorch caching allocator.",
                "cuda_peak_extra_allocated_mb": "Stage peak allocated minus stage-start allocated.",
                "cuda_device_used_mb": "Whole-device usage (all processes), derived from mem_get_info.",
            },
            "baseline": self._snapshot(),
            "stages": [],
            "total": None,
        }
        if self.enabled:
            profile_dir.mkdir(parents=True, exist_ok=True)
            self._write()

    def _cuda_available(self) -> bool:
        try:
            return bool(self.torch is not None and self.torch.cuda.is_available())
        except Exception:
            return False

    def _sync_cuda(self) -> None:
        if self._cuda_available():
            self.torch.cuda.synchronize()

    def _reset_cuda_peaks(self) -> None:
        if not self._cuda_available():
            return
        self.torch.cuda.reset_peak_memory_stats()

    def _snapshot(self, *, include_peak: bool = False) -> Dict[str, Any]:
        cpu = _read_proc_status()
        snap: Dict[str, Any] = {
            "rss_mb": _round(cpu["rss_mb"]),
            "rss_peak_mb": _round(cpu["rss_peak_mb"]),
            "cuda_available": self._cuda_available(),
        }
        if not snap["cuda_available"]:
            return snap
        try:
            device = self.torch.cuda.current_device()
            props = self.torch.cuda.get_device_properties(device)
            free_bytes, total_bytes = self.torch.cuda.mem_get_info(device)
            snap.update({
                "cuda_device_index": int(device),
                "cuda_device_name": props.name,
                "cuda_device_total_mb": _round(total_bytes / MIB),
                "cuda_device_used_mb": _round((total_bytes - free_bytes) / MIB),
                "cuda_allocated_mb": _round(self.torch.cuda.memory_allocated(device) / MIB),
                "cuda_reserved_mb": _round(self.torch.cuda.memory_reserved(device) / MIB),
            })
            if include_peak:
                snap["cuda_peak_allocated_mb"] = _round(
                    self.torch.cuda.max_memory_allocated(device) / MIB
                )
                snap["cuda_peak_reserved_mb"] = _round(
                    self.torch.cuda.max_memory_reserved(device) / MIB
                )
        except Exception as exc:  # Profiling must never break reconstruction.
            snap["cuda_measurement_error"] = f"{type(exc).__name__}: {exc}"
        return snap

    def _write(self) -> None:
        if not self.enabled:
            return
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp_path, self.path)

    @contextmanager
    def stage(
        self, name: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> Iterator[None]:
        """Measure one non-overlapping pipeline stage and persist immediately."""
        if not self.enabled:
            yield
            return

        self._sync_cuda()
        self._reset_cuda_peaks()
        started = self._snapshot()
        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        error = None
        try:
            yield
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._sync_cuda()
            ended = self._snapshot(include_peak=True)
            start_alloc = started.get("cuda_allocated_mb")
            peak_alloc = ended.get("cuda_peak_allocated_mb")
            peak_extra = None
            if start_alloc is not None and peak_alloc is not None:
                peak_extra = max(0.0, float(peak_alloc) - float(start_alloc))
            entry = {
                "name": name,
                "status": "error" if error else "ok",
                "wall_time_s": _round(time.perf_counter() - wall0, 6),
                "cpu_time_s": _round(time.process_time() - cpu0, 6),
                "metadata": metadata or {},
                "start": started,
                "end": ended,
                "rss_delta_mb": _round(
                    (ended["rss_mb"] - started["rss_mb"])
                    if ended.get("rss_mb") is not None and started.get("rss_mb") is not None
                    else None
                ),
                "cuda_allocated_delta_mb": _round(
                    (ended["cuda_allocated_mb"] - started["cuda_allocated_mb"])
                    if ended.get("cuda_allocated_mb") is not None
                    and started.get("cuda_allocated_mb") is not None
                    else None
                ),
                "cuda_peak_extra_allocated_mb": _round(peak_extra),
                "error": error,
            }
            self.data["stages"].append(entry)
            if error:
                self.data["status"] = "error"
            self._write()
            gpu_peak = entry["end"].get("cuda_peak_allocated_mb")
            gpu_text = f", cuda_peak={gpu_peak:.1f} MiB" if gpu_peak is not None else ""
            rss_text = entry["end"].get("rss_mb")
            rss_label = f", rss={rss_text:.1f} MiB" if rss_text is not None else ""
            print(
                f"[profile] {self.operation}/{name}: {entry['wall_time_s']:.3f}s"
                f"{rss_label}{gpu_text}",
                flush=True,
            )

    def finish(
        self, *, status: str = "ok", metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if not self.enabled:
            return ""
        if self._finished:
            return str(self.path)
        self._sync_cuda()
        if metadata:
            self.data["metadata"].update(metadata)
        self.data["status"] = "error" if self.data["status"] == "error" else status
        self.data["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.data["total"] = {
            "wall_time_s": _round(time.perf_counter() - self._started_wall, 6),
            "cpu_time_s": _round(time.process_time() - self._started_cpu, 6),
            "end": self._snapshot(include_peak=False),
            "stage_count": len(self.data["stages"]),
        }
        self._finished = True
        self._write()
        return str(self.path)


def stage(profiler: Optional[ResourceProfiler], name: str, **metadata: Any):
    """Return a real stage context or a no-op context for optional profiling."""
    if profiler is None:
        return _null_stage()
    return profiler.stage(name, metadata=metadata or None)


@contextmanager
def _null_stage() -> Iterator[None]:
    yield

