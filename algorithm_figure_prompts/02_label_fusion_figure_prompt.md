# Figure Prompt — 3D Semantic-Label Fusion (with foreground/background competition)

> **Purpose.** Prompt + layout spec for an image-generation model to draw a
> CVPR-style algorithm diagram of the 3D label-fusion module, including the
> over-dilation ("bleed") failure and the fix. Code reference:
> `semantic_fusion.py: refine_masks_3d, _cluster_points`.

---

## 1. What the algorithm does (ground truth for the figure)

A scene is reconstructed by VGGT into one shared 3D point cloud
`world_points (S,H,W,3)`. YOLOe segments **only keyframes**, so most frames
contribute no labels → the merged cloud has **holes** (object points the
skipped frames never labeled) and **speckle** (isolated false positives).
Because `semantic_masks.reshape(-1)` is point-aligned with
`world_points.reshape(-1,3)`, labels are reconciled **once, in 3D**:

1. **Seed gather.** Collect 3D points labeled with the target id from the
   keyframes (optionally gated by confidence percentile).
2. **Density clustering** (`_cluster_points`): KD-tree radius graph at `eps`
   (single-linkage = connected components); drop clusters smaller than
   `min_cluster_size` → **removes speckle / false positives**.
3. **Relabel by foreground/background competition** (the key step, default on):
   - **Object (foreground) anchors** = surviving cluster points.
   - **Background anchors** = points that keyframes labeled as *not* this id
     (trustworthy background, e.g. ground).
   - For every candidate point in the cloud, compute distance to nearest object
     anchor (`d_obj`, capped at `dilate_radius`) and to nearest background
     anchor (`d_bg`). Assign the object label **iff `d_obj < d_bg`** and within
     range. → **fills holes without bleeding onto the ground.**

**Why the competition matters (the failure it fixes):** the naive version
("label every point within `dilate_radius` of the object cluster") over-grows.
An excavator sits on / touches the ground, so any ball around its surface also
swallows adjacent ground points. On a real cloud this produced **91% bleed**
(1.34M labeled, only ~124k truly object). The competition lets ground points be
reclaimed by their own (much closer) background anchors → bleed **91% → 9%**,
recall ~0.94, and the result is **insensitive to the radius value**.

## 2. Diagram type & overall layout

- **Type:** a 4-stage horizontal pipeline of small 3D point-cloud "vignettes",
  PLUS a dedicated **before/after comparison pair** for the bleed fix.
- **Canvas:** landscape, ~16:9.
- **Top row — the pipeline (4 vignettes, left→right), each a top-down point cloud:**
  - **(a) INPUT:** sparse green object seeds over a grey ground cloud; annotate
    "holes (missed frames)" inside the object and dashed red circles around a
    few stray clusters labeled "speckle (false positives)".
  - **(b) CLUSTER:** the same seeds; the main object cluster kept (green),
    small clusters crossed out in red and labeled "dropped < min_cluster_size".
    Overlay a faint dotted circle of radius `eps` on a couple of points to
    convey the radius-graph neighbourhood.
  - **(c) COMPETITION:** show a zoomed boundary between object and ground. Draw
    a candidate point with two arrows: a short green arrow to nearest **object
    anchor** (`d_obj`) and a short red arrow to nearest **background anchor**
    (`d_bg`); a small inequality `d_obj < d_bg ?` decides its colour. Include
    one object-side point that wins (turns green, "hole filled") and one
    ground-side point that loses (stays grey, "bleed prevented").
  - **(d) OUTPUT:** object solid & complete in green; speckle gone; ground clean.
- **Bottom row — bleed comparison (two top-down clouds side by side):**
  - **OLD: one-sided dilation** — green object plus a large red halo of
    mislabeled ground; caption "bleed 91%".
  - **NEW: fg/bg competition** — compact green object, negligible red; caption
    "bleed 9%". Put a bold arrow "≈100× less bleed" between them.

## 3. Required labeled elements

- Stage labels **(a) INPUT · (b) CLUSTER · (c) COMPETITION · (d) OUTPUT**.
- Box/term labels: **shared 3D cloud (VGGT)**, **keyframe seeds (YOLOe)**,
  **KD-tree radius graph (eps)**, **connected components / single-linkage**,
  **min_cluster_size (noise drop)**, **object anchors**, **background anchors**,
  **nearest-anchor test d_obj < d_bg**, **dilate_radius (reach cap)**.
- The decision inequality rendered in serif math: `assign id ⇔ d_obj < d_bg`.
- Colour legend: green = object (foreground), grey = unlabeled/background,
  red = speckle / bleed / rejected.
- Caption strip: *"Reconcile per-keyframe labels once in 3D: cluster to drop
  false positives, then relabel by nearest-anchor competition to fill holes
  without bleeding onto the ground."*

## 4. Visual style (CVPR / academic)

- Flat scientific look. Point clouds rendered as small dot scatters (top-down),
  not photoreal meshes; thin axes or none, light grey grid optional.
- Palette: emerald green `#0E9D6E` (object), neutral grey `#9AA0A6`
  (background/unlabeled), muted red `#D1495B` (speckle/bleed/reject). White
  background, near-black text. No gradients, no 3D bevels.
- Distance arrows: green for `d_obj`, red for `d_bg`, with small italic labels.
- Sans-serif labels (Helvetica/Inter), serif for the inequality.
- Grayscale-safe: distinguish classes by dot density + outline + label, not
  colour alone. Consistent arrowheads, subtle rounded-rectangle node boxes.

## 5. One-paragraph prompt (paste-ready)

> A CVPR-style algorithm figure, flat scientific vector style on white, thin
> strokes, point clouds drawn as small top-down dot scatters. Top row: a
> four-stage left-to-right pipeline labeled (a) INPUT, (b) CLUSTER, (c)
> COMPETITION, (d) OUTPUT. (a) sparse green object seeds over a grey ground
> cloud, annotated "holes (missed frames)" and dashed red circles "speckle
> (false positives)". (b) the main green cluster kept while small clusters are
> crossed out in red and labeled "dropped < min_cluster_size", with a faint
> dotted neighbourhood circle of radius eps. (c) a zoomed object/ground boundary:
> one candidate point with a short green arrow to its nearest object anchor
> (d_obj) and a red arrow to its nearest background anchor (d_bg), a serif math
> test "assign id ⇔ d_obj < d_bg", one point turning green labeled "hole filled"
> and one staying grey labeled "bleed prevented". (d) a solid complete green
> object on clean grey ground. Bottom row: a before/after comparison of two
> top-down clouds — left "OLD: one-sided dilation" with a large red halo of
> mislabeled ground (caption "bleed 91%"), right "NEW: fg/bg competition" with a
> compact green object and almost no red (caption "bleed 9%"), and a bold arrow
> between them "≈100× less bleed". Emerald green = object, grey = background, muted
> red = speckle/bleed. Helvetica-like sans labels, serif math, legend, academic
> paper aesthetic, high-contrast, grayscale-safe, no gradients, no photoreal 3D.
