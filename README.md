# Excavator Reconstruction Full Pipeline

This project builds a reconstruction pipeline that combines VGGT 3D reconstruction with YOLOe semantic segmentation. It supports image/video input, depth and point-cloud visualization, semantic point-cloud editing, and an interactive Three.js elevation viewer with gravity-aligned DEM.

## Main Features

- Video or image sequence upload through a Gradio orchestrator.
- YOLOe open-vocabulary semantic segmentation service.
- VGGT-based 3D reconstruction service.
- Depth map, point map, semantic depth map, and semantic point map export.
- Semantic point-cloud delete/extract operations.
- Gravity-aligned elevation DEM generation (on demand, for the viewer).
- Interactive elevation viewer with reference plane, DEM display, gravity diagnostics, selection tools, and rough volume estimation.

## Core Files

| File | Purpose |
| --- | --- |
| `orchestrator.py` | Gradio UI and workflow orchestration. |
| `yoloe_service.py` | FastAPI YOLOe segmentation service, usually on port `8001`. |
| `vggt_service.py` | FastAPI VGGT reconstruction/elevation-viewer service, usually on port `8002`. |
| `gravity_alignment.py` | Gravity direction estimation and coordinate alignment. |
| `elevation_plane.py` | DEM helpers (point extraction, ground selection, grid interpolation) for the viewer and streaming paths. |
| `elevation_viewer.html` | Standalone Three.js elevation viewer. |
| `start_all.sh` | Starts YOLOe, VGGT, and Gradio services. |
| `stop_all.sh` | Helper script for stopping services. |
| `terrain_analysis.py` | Fuses YOLOe semantics with the VGGT DEM on a BEV grid (mounds/pits + material). |
| `semantic_fusion.py` | Semantic point-cloud fusion helpers. |
| `test_terrain_vlm.py` / `make_vlm_report.py` | Feed the BEV worksite map to a VLM for structured dig decisions and build an HTML report. |
| `streaming/` | Real-time VGGT→Unity link: keyframe buffer, sliding-window reconstruction loop, elevation publisher, and session API. |
| `experiments/` | Offline studies (scale/volume calibration, arm-motion state, DINOv2 segmentation, interaction updates) with per-experiment reports. |
| `tests/` | Pytest suite covering gravity alignment, elevation export, streaming, plane calibration, and ground-truth evaluation. |
| `tools/` | Verification and profiling utilities. |

## Pipeline Overview

1. Input video/images are converted into a workspace under `workspaces/`.
2. YOLOe optionally generates semantic masks for selected classes.
3. VGGT reconstructs depth, camera poses, and world points.
4. The system estimates gravity direction using a cascade:
   - camera trajectory PCA,
   - YOLOe ground-mask RANSAC,
   - whole-cloud RANSAC fallback.
5. Points are rotated into a gravity-aligned frame where `Y` is elevation.
6. Ground candidates are selected by semantic ground mask or low-height percentile.
7. A regular DEM grid is interpolated over the aligned `(X, Z)` plane.
8. The DEM is served to the Three.js viewer for inspection and rough volume estimation.

## Run

The project assumes the related VGGT and YOLOe repositories/environments are available at the paths used in `start_all.sh`:

- `/home/maomaoyu/WS/vggt`
- `/home/maomaoyu/WS/yoloe`
- `/home/maomaoyu/WS/vggt_yoloe`

Start all services:

```bash
bash /home/maomaoyu/WS/vggt_yoloe/start_all.sh
```

Default URLs:

- Gradio UI: `http://localhost:7860`
- YOLOe API docs: `http://localhost:8001/docs`
- VGGT API docs: `http://localhost:8002/docs`

## Typical Usage

1. Open `http://localhost:7860`.
2. Upload a video or image sequence.
3. Run reconstruction.
4. Optionally run YOLOe segmentation for classes such as `ground`, `excavator`, or `building`.
5. Use semantic editing to delete or extract selected classes if needed.
6. Open the 3D elevation viewer from the `Elevation Plane` tab for DEM inspection and volume selection.

## Outputs

Typical generated outputs include:

- `predictions.npz`: VGGT predictions.
- `semantic_masks.npz`: YOLOe semantic masks.
- `glbscene_*.glb`: reconstructed scene.
- `edited_<op>_ids<...>.glb`: semantically edited scene from the editing tab.
- `edited_scenes.json`: registry of edits, used by the elevation viewer's scene selector.

Generated data is stored under `workspaces/` and is intentionally ignored by Git.

## Notes

- The semantic ground class is currently expected to use ID `1`.
- The elevation viewer has a **scene selector** (sidebar): each edit applied in the Point Cloud Editing tab is recorded to `edited_scenes.json` in the workspace, and the viewer can re-derive that filtered cloud from raw predictions. Switching scenes re-runs gravity alignment and DEM fitting on the selected (filtered) point cloud.
- Gravity estimation is handled by the cascade in `gravity_alignment.py` (trajectory PCA → ground-mask RANSAC → whole-cloud RANSAC).
- The elevation viewer uses a fixed `128 x 128` DEM grid built on demand by `/elevation_viewer_data`; no DEM files are exported.
- Large media, workspaces, GLB/NPZ/ZIP outputs, and model weights are excluded by `.gitignore`.

## Real-time link (streaming)

The `streaming/` package provides a non-invasive VGGT→Unity streaming path that runs alongside the offline reconstruction without regressing it. It maintains a keyframe buffer, runs a sliding-window reconstruction loop, and publishes gravity-aligned elevation tiles through a session API. See `REALTIME_LINK_PLAN.md` and `SLIDING_WINDOW_RECONSTRUCTION_LOOP.md` for design and milestones.

## Documentation

Additional project notes:

- `architecture.html`
- `technical_report.html`
- `elevation_viewer_internals.html`
- `elevation_plane_route_summary.html`

Planning and research notes (Markdown):

- `RESEARCH_PROGRESS.md`, `RESOURCE_PROFILING.md`
- `REALTIME_LINK_PLAN.md`, `SLIDING_WINDOW_RECONSTRUCTION_LOOP.md`, `LIVE_CAMERA_TWO_STAGE_PLAN.md`
- `SCENE_GRAPH_PLAN.md`, `EXCAVATOR_POSE_PLAN.md`, `EXP_DYN0_PLAN.md`
- `TERRAIN_ELEVATION_FORMAT.md`, `TERRAIN_LAYERS_DATA_SOURCES.md`, `WORKZONE_CRITERIA.md`, `WORKLOG_terrain_vlm.md`
- `VGGT_ALIGNMENT_ARCHITECTURE.md`

## Repository layout note

The working tree's Git metadata lives in `.gitrepo/` (not the usual `.git/`). Use it explicitly, e.g.:

```bash
git --git-dir=.gitrepo --work-tree=. status
```
