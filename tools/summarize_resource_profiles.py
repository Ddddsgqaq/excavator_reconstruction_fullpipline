#!/usr/bin/env python3
"""Flatten offline pipeline resource profiles into CSV and a compact JSON summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CSV_FIELDS = [
    "profile_file", "operation", "profile_status", "started_at_utc",
    "stage_index", "stage", "stage_status", "wall_time_s", "cpu_time_s",
    "rss_end_mb", "rss_lifetime_peak_mb", "rss_delta_mb",
    "cuda_device_name", "cuda_allocated_end_mb", "cuda_reserved_end_mb",
    "cuda_peak_allocated_mb", "cuda_peak_extra_allocated_mb",
    "cuda_device_used_end_mb",
]


def _load_profiles(profile_dir: Path):
    profiles = []
    for path in sorted(profile_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("schema_version") == 1 and data.get("operation"):
            profiles.append((path, data))
    return profiles


def _flatten(profile_path: Path, profile: dict):
    rows = []
    for index, item in enumerate(profile.get("stages", [])):
        end = item.get("end") or {}
        rows.append({
            "profile_file": str(profile_path),
            "operation": profile.get("operation"),
            "profile_status": profile.get("status"),
            "started_at_utc": profile.get("started_at_utc"),
            "stage_index": index,
            "stage": item.get("name"),
            "stage_status": item.get("status"),
            "wall_time_s": item.get("wall_time_s"),
            "cpu_time_s": item.get("cpu_time_s"),
            "rss_end_mb": end.get("rss_mb"),
            "rss_lifetime_peak_mb": end.get("rss_peak_mb"),
            "rss_delta_mb": item.get("rss_delta_mb"),
            "cuda_device_name": end.get("cuda_device_name"),
            "cuda_allocated_end_mb": end.get("cuda_allocated_mb"),
            "cuda_reserved_end_mb": end.get("cuda_reserved_mb"),
            "cuda_peak_allocated_mb": end.get("cuda_peak_allocated_mb"),
            "cuda_peak_extra_allocated_mb": item.get("cuda_peak_extra_allocated_mb"),
            "cuda_device_used_end_mb": end.get("cuda_device_used_mb"),
        })
    return rows


def summarize(workspace: Path, output_dir: Path | None = None):
    profile_dir = workspace / "resource_profiles"
    output_dir = output_dir or profile_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = _load_profiles(profile_dir)
    rows = [row for path, profile in profiles for row in _flatten(path, profile)]

    csv_path = output_dir / "resource_profile_stages.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    operations = []
    for path, profile in profiles:
        stages = profile.get("stages", [])
        slowest = max(stages, key=lambda x: x.get("wall_time_s") or 0, default=None)
        gpu_stages = [
            s for s in stages if (s.get("end") or {}).get("cuda_peak_allocated_mb") is not None
        ]
        gpu_peak = max(
            ((s.get("end") or {}).get("cuda_peak_allocated_mb") for s in gpu_stages),
            default=None,
        )
        gpu_peak_extra = max(
            (s.get("cuda_peak_extra_allocated_mb") or 0 for s in gpu_stages),
            default=None,
        )
        rss_end_max = max(
            ((s.get("end") or {}).get("rss_mb") or 0 for s in stages),
            default=None,
        )
        rss_delta_max = max(
            (s.get("rss_delta_mb") or 0 for s in stages),
            default=None,
        )
        rss_peak = max(
            ((s.get("end") or {}).get("rss_peak_mb") or 0 for s in stages),
            default=None,
        )
        operations.append({
            "operation": profile.get("operation"),
            "status": profile.get("status"),
            "profile_path": str(path),
            "total_wall_time_s": (profile.get("total") or {}).get("wall_time_s"),
            "stage_count": len(stages),
            "slowest_stage": slowest.get("name") if slowest else None,
            "slowest_stage_wall_time_s": slowest.get("wall_time_s") if slowest else None,
            "rss_baseline_mb": (profile.get("baseline") or {}).get("rss_mb"),
            "rss_stage_end_max_mb": rss_end_max,
            "rss_stage_delta_max_mb": rss_delta_max,
            "process_lifetime_rss_peak_mb": rss_peak,
            "cuda_baseline_allocated_mb": (
                profile.get("baseline") or {}).get("cuda_allocated_mb"),
            "cuda_peak_allocated_mb": gpu_peak,
            "cuda_peak_extra_allocated_mb": gpu_peak_extra,
        })

    summary = {
        "workspace": str(workspace.resolve()),
        "profile_count": len(profiles),
        "stage_count": len(rows),
        "operations": operations,
        "stage_csv": str(csv_path.resolve()),
    }
    json_path = output_dir / "resource_profile_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, json_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize <workspace>/resource_profiles/*.json into JSON and CSV.")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary, json_path, csv_path = summarize(args.workspace, args.output_dir)
    print(json.dumps({
        "profiles": summary["profile_count"],
        "stages": summary["stage_count"],
        "summary_json": str(json_path.resolve()),
        "stages_csv": str(csv_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
