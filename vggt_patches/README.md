# VGGT patches

This project imports helper functions from the external VGGT repo's `visual_util.py`
that **do not exist in the upstream VGGT release**. Specifically, `vggt_service.py`
imports these from `visual_util`:

- `predictions_to_glb`
- `generate_depth_visualizations`
- `generate_pointmap_visualizations`
- `generate_semantic_depth_visualizations`
- `generate_semantic_pointmap_visualizations`

If you run against a fresh/official VGGT checkout, `vggt_service.py` will fail at
import time with `ImportError`. This folder ships the modified `visual_util.py`
so a small migration only needs to copy one file.

## What to do (placement)

1. Clone the official VGGT repo as a sibling of this repo (or set `VGGT_DIR`):

   ```
   <parent>/
   ├── vggt_yoloe/   ← this repo
   ├── vggt/         ← official VGGT checkout (+ model.pt weights)
   └── yoloe/
   ```

2. Copy the patched file over the upstream one, replacing it:

   ```bash
   cp vggt_yoloe/vggt_patches/visual_util.py  vggt/visual_util.py
   ```

   (If `VGGT_DIR` points elsewhere, copy to `$VGGT_DIR/visual_util.py` instead.)

3. Start services as usual (`bash start_all.sh`).

## Notes

- Only `visual_util.py` is required. The other locally-modified files in the VGGT
  repo (`demo_colmap.py`, `demo_gradio.py`, `demo_viser.py`, `requirements_demo.txt`,
  `vggt/dependency/vggsfm_utils.py`) are upstream demo/dependency files that this
  project does **not** import, so they are intentionally not shipped here.
- `visual_util.py` only depends on third-party packages (trimesh, gradio, numpy,
  matplotlib, scipy, cv2, requests) — no cross-dependency on the other VGGT changes.
- YOLOe needs no code patch: it runs on the official repo and downloads weights
  from HuggingFace (`jameslahm/yoloe`) on first run.
