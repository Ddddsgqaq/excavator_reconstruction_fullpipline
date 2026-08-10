# Figure Prompt — Elevation DEM Construction (gravity alignment → ground selection → DEM grid)

> **Purpose.** Prompt + layout spec for an image-generation model to draw a
> CVPR-style algorithm diagram of the elevation-DEM module that the Three.js
> elevation viewer consumes. Code reference:
> `vggt_service.py: elevation_viewer_data`,
> `gravity_alignment.py: estimate_gravity, estimate_from_trajectory`,
> `elevation_plane.py: _select_ground_aligned_mask` + DEM grid via `scipy.griddata`.

---

## 1. What the algorithm does (ground truth for the figure)

Turn an unoriented VGGT point cloud into a **gravity-aligned Digital Elevation
Model (DEM)** — a regular height grid `Y = f(X,Z)` usable for hole-filling and
cut/fill volume computation.

1. **Inputs:** VGGT predictions — point cloud `world_points`, per-point
   confidence, camera extrinsics `(S,3,4)`, optional YOLOe ground mask
   (semantic id = 1).
2. **Gravity estimation (cascade, trajectory-first):**
   - **Primary — camera-trajectory plane:** PCA/SVD on the camera centers; the
     plane normal (smallest-variance direction) approximates "up". Rejected if
     the trajectory is degenerate (nearly a line; second/first eigenvalue ratio
     too small).
   - **Check / fallback — ground-mask plane:** RANSAC plane fit to the masked
     ground points; sanity-checked against the trajectory normal (warn if they
     disagree by > a threshold angle).
   - **Last-resort fallback — whole-cloud RANSAC plane.**
   - Convert the chosen up-normal `n` to a rotation `R_align` (shortest-arc
     rotation taking `n → +Y`).
3. **Align:** rotate points `P' = P · R_alignᵀ` so **Y is elevation** and
   **(X,Z) is the horizontal plane**; extrinsics updated as `R' = R·R_alignᵀ`.
   (A separate `T_display` puts points in the GLB/viewer display space.)
4. **Confidence filter:** drop points below the `conf_thres` percentile and
   non-finite points; optional semantic extract/delete edit.
5. **Ground-point selection** (`_select_ground_aligned_mask`):
   - If a ground mask exists: keep masked points whose Y lies within
     ±band·(Y-range) of the masked-points median Y (a tight horizontal slab).
   - Else: take the lowest `ground_percentile`% of points by Y.
6. **DEM grid:** lay a regular **128×128** lattice over the (X,Z) bounding box
   (+2% padding); interpolate ground-point heights onto it with
   `griddata` **linear**, fill NaN gaps with **nearest** neighbour; keep a
   `has_data` mask marking truly observed vs hole-filled cells.
7. **Output:** the DEM (`elev[128,128]`, `has_data`), `R_align`, aligned points
   & camera centers — streamed to the viewer for height-coloured terrain,
   hole-fill, and volume.

## 2. Diagram type & overall layout

- **Type:** vertical (top→bottom) staged pipeline with three "geometry"
  illustrations and a final 3D DEM render. Inputs enter at the top.
- **Canvas:** portrait-ish or 4:3; clearly numbered stages.
- **Stage 1 — Inputs (top):** three small icons feeding in: a tilted point
  cloud blob, a camera-trajectory polyline of camera frustums, and a small
  "ground mask" frame (terrain with a highlighted ground region).
- **Stage 2 — Gravity estimation (cascade box):** a vertical decision stack:
  - "Camera-trajectory PCA plane" (draw camera centers as dots with a fitted
    plane and its normal arrow `n`); a branch "degenerate? → fallback".
  - "Ground-mask RANSAC plane" (a plane through highlighted ground points, with
    a small "agree within θ?" check against the trajectory normal).
  - "Whole-cloud RANSAC plane (last resort)".
  - All converge to a node **`R_align`: rotate up-normal n → +Y**.
- **Stage 3 — Alignment (before/after pair):** left = tilted cloud with a
  slanted gravity arrow; right = same cloud rotated so the gravity arrow points
  straight up (+Y), ground now horizontal, axes labeled **Y = elevation**,
  **(X,Z) = horizontal**.
- **Stage 4 — Ground selection:** side view of the aligned cloud; highlight a
  thin horizontal **slab** (±band around median ground Y) in green = "ground
  points"; the excavator/objects above it greyed out. Caption the two modes
  ("semantic ground slab" vs "lowest percentile").
- **Stage 5 — DEM grid (bottom, the payoff):** a 128×128 lattice over the (X,Z)
  footprint; show interpolation filling cell heights (linear) and a few
  gap cells filled by nearest-neighbour (mark them with a distinct hatch /
  "hole-filled"); render the result as a 3D height-colored surface (low→high
  colormap) i.e. the DEM. Optionally hint volume as the signed gap between DEM
  and a reference datum plane.

## 3. Required labeled elements

- Stage numbers/titles: **Inputs · Gravity estimation (trajectory-first
  cascade) · Gravity alignment · Ground selection · DEM grid**.
- Term labels: **camera centers**, **PCA/SVD plane normal n**, **degenerate
  check (eig ratio)**, **ground-mask RANSAC**, **agreement angle θ**,
  **R_align (shortest-arc n→+Y)**, **Y = elevation**, **±band ground slab**,
  **ground_percentile**, **128×128 lattice**, **griddata linear + nearest
  fill**, **has_data mask**, **DEM (height grid)**.
- Math snippets (serif): `P' = P R_alignᵀ`, `R_align : n ↦ +Y`,
  `ground = { y : |y − median(y_g)| ≤ band·range }`.
- Colormap legend for elevation (e.g. viridis/plasma low→high).
- Caption strip: *"Estimate gravity (camera trajectory first, ground-mask and
  whole-cloud RANSAC as fallbacks), align so Y is up, select ground points, and
  interpolate a 128×128 DEM for hole-fill and volume."*

## 4. Visual style (CVPR / academic)

- Flat technical-illustration style: point clouds as dot scatters, planes as
  semi-transparent quads with a normal arrow, cameras as small wireframe
  frustums, the DEM as a clean 3D mesh/surface with a perceptual colormap.
- Palette: near-black text on white; one structural accent (blue `#2C5FAA`) for
  geometry/planes/normals, emerald green `#0E9D6E` for selected ground, neutral
  grey for non-ground, and a separate elevation colormap **only** on the final
  DEM surface. Avoid decorative gradients elsewhere.
- Sans-serif labels (Helvetica/Inter); serif math for equations.
- Clear directional arrows for the cascade and the rotation; dashed arrows for
  fallback branches.
- Grayscale-safe: rely on labels, hatching, and line style; reserve the
  colormap for the DEM where height encoding is the point.
- Academic, uncluttered, printable; no photorealism, no skeuomorphic 3D bevels.

## 5. One-paragraph prompt (paste-ready)

> A CVPR-style multi-stage algorithm figure, flat technical-illustration vector
> style on white, thin strokes, numbered top-to-bottom stages. Stage 1 "Inputs":
> small icons of a tilted 3D point cloud, a polyline of camera frustums, and a
> ground-mask terrain frame. Stage 2 "Gravity estimation (trajectory-first
> cascade)": a vertical decision stack — "camera-trajectory PCA plane" with
> camera-center dots, a fitted semi-transparent plane and a blue normal arrow n,
> a dashed fallback branch "degenerate?" to "ground-mask RANSAC plane" (plane
> through highlighted green ground points, with an "agree within θ?" check) and
> a final dashed "whole-cloud RANSAC (last resort)", all converging to a node
> "R_align: rotate n → +Y". Stage 3 "Alignment": a before/after pair — a tilted
> cloud with a slanted gravity arrow on the left, the same cloud rotated upright
> on the right with the arrow pointing to +Y, axes labeled "Y = elevation" and
> "(X,Z) horizontal", serif equation P' = P·R_alignᵀ. Stage 4 "Ground selection":
> a side view of the aligned cloud with a thin horizontal green slab "±band ground
> slab" highlighted and objects above it greyed, serif set notation for the band.
> Stage 5 "DEM grid": a 128×128 lattice over the X–Z footprint, interpolation
> filling cell heights (linear) with a few nearest-neighbour gap-filled cells
> hatched and labeled "hole-filled", rendered as a clean 3D height-colored DEM
> surface with a low→high colormap legend. Blue accent for geometry/planes/normals,
> emerald green for selected ground, grey for non-ground, elevation colormap only
> on the final DEM. Helvetica-like sans labels, serif math, legend, academic paper
> aesthetic, high-contrast, grayscale-safe, no gradients elsewhere, no photoreal
> rendering.
