# Excavator Reconstruction Full Pipeline

This project builds a reconstruction pipeline that combines VGGT 3D reconstruction with YOLOe semantic segmentation. It supports image/video input, depth and point-cloud visualization, semantic point-cloud editing, elevation DEM fitting, and an interactive Three.js elevation viewer.

## Main Features

- Video or image sequence upload through a Gradio orchestrator.
- YOLOe open-vocabulary semantic segmentation service.
- VGGT-based 3D reconstruction service.
- Depth map, point map, semantic depth map, and semantic point map export.
- Semantic point-cloud delete/extract operations.
- Gravity-aligned elevation DEM generation.
- DEM GLB export and point-cloud + DEM merged GLB export.
- Interactive elevation viewer with reference plane, DEM display, gravity diagnostics, selection tools, and rough volume estimation.

## Core Files

| File | Purpose |
| --- | --- |
| `orchestrator.py` | Gradio UI and workflow orchestration. |
| `yoloe_service.py` | FastAPI YOLOe segmentation service, usually on port `8001`. |
| `vggt_service.py` | FastAPI VGGT reconstruction/elevation service, usually on port `8002`. |
| `gravity_alignment.py` | Gravity direction estimation and coordinate alignment. |
| `elevation_plane.py` | DEM fitting, elevation mesh generation, and GLB export. |
| `elevation_viewer.html` | Standalone Three.js elevation viewer. |
| `start_all.sh` | Starts YOLOe, VGGT, and Gradio services. |
| `stop_all.sh` | Helper script for stopping services. |

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
8. The DEM is exported as GLB and can be viewed or used for rough volume estimation.

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
6. Run elevation fitting from the `Elevation Plane` tab.
7. Open the 3D elevation viewer for DEM inspection and volume selection.

## Outputs

Typical generated outputs include:

- `predictions.npz`: VGGT predictions.
- `semantic_masks.npz`: YOLOe semantic masks.
- `glbscene_*.glb`: reconstructed scene.
- `elev_r<N>_<cmap>_aligned_only.glb`: elevation-only DEM mesh.
- `elev_r<N>_<cmap>_aligned_merged.glb`: point cloud plus DEM mesh.
- `elev_r<N>_<cmap>_aligned_meta.json`: gravity alignment and fitting metadata.

Generated data is stored under `workspaces/` and is intentionally ignored by Git.

## Notes

- The semantic ground class is currently expected to use ID `1`.
- `use_ransac` is kept for API compatibility; gravity estimation is handled by the cascade in `gravity_alignment.py`.
- The elevation viewer uses a fixed `128 x 128` DEM grid from `/elevation_viewer_data`, while exported DEM GLBs use the UI-selected grid resolution.
- Large media, workspaces, GLB/NPZ/ZIP outputs, and model weights are excluded by `.gitignore`.

## Documentation

Additional project notes:

- `architecture.html`
- `technical_report.html`
- `elevation_viewer_internals.html`
- `elevation_plane_route_summary.html`
