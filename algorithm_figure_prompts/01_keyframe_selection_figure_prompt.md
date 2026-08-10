# Figure Prompt — Viewpoint-Similarity Keyframe Selection

> **Purpose of this file.** A detailed prompt + layout spec to hand to an
> image-generation model so it draws a *CVPR-style algorithm/architecture
> diagram* for the keyframe-selection module. It describes the algorithm logic,
> the visual elements, the flow, and the styling. Code reference:
> `yoloe_service.py: _frame_signature, _frame_similarity, select_keyframe_indices`.

---

## 1. What the algorithm does (ground truth for the figure)

A video of a scene yields many near-duplicate frames. Segmenting every frame
with YOLOe is wasteful, so we keep only frames whose **camera viewpoint** has
changed enough from the last kept keyframe. Crucially, similarity is measured
**geometrically (ORB feature matching), not by colour** — an HSV colour
histogram cannot tell a static camera from a large viewpoint swing, because the
palette barely changes.

Per consecutive comparison (last kept keyframe vs. candidate frame *i*):

1. **ORB keypoints + descriptors** for each frame (≤1500 features).
2. **Brute-force Hamming matching** with **Lowe ratio test** (0.75) → "good" matches.
3. **RANSAC homography** (reproj. thresh 4 px) → geometric **inliers**.
4. **Viewpoint similarity score** `s ∈ [0,1]`:
   `s = inlier_ratio × 1/(1 + (median_parallax / diag / 0.06)²)`
   where `inlier_ratio = #inliers / #good`, `median_parallax` = median pixel
   displacement of inlier matches, `diag` = image diagonal.
   `s≈1` ⇒ same viewpoint; `s↓` ⇒ bigger viewpoint change.
5. **Keep rule:** if `s < τ` (default τ=0.45), mark frame *i* as a new keyframe
   and reset the reference to it; else skip it. First and last frames always kept.

Output: a sparse set of keyframe indices; skipped frames are left unlabeled and
recovered later by 3D label fusion (a separate module).

## 2. Diagram type & overall layout

- **Type:** left-to-right pipeline / data-flow diagram with one highlighted
  "comparison unit" expanded as an inset, plus a bottom timeline strip.
- **Canvas:** landscape, ~16:9, generous white margins.
- **Three horizontal bands:**
  - **Top band — frame strip:** a row of ~10 small video-frame thumbnails
    (abstract: rounded rectangles with a tiny stylized excavator-on-terrain
    glyph at slightly rotating angles to imply an orbiting camera). A few are
    outlined in a bold accent colour and tagged **"KEYFRAME"**; the rest are
    greyed/dashed and tagged **"skipped"**.
  - **Middle band — the comparison unit (the heart of the figure):** an
    expanded block showing two frames (reference keyframe vs candidate) feeding
    a 4-step horizontal chain:
    `ORB keypoints → ratio-test matches → RANSAC homography inliers → similarity score s`.
    Draw the two frames side by side with green lines connecting matched
    keypoints between them (classic feature-match visualization); a subset of
    lines in solid green (inliers) and some faint/red dashed (outliers rejected
    by RANSAC). End the chain in a rounded "score" node showing `s` and a
    threshold gate `s < τ ?` with two branches: **YES → keep (new keyframe)**,
    **NO → skip**.
  - **Bottom band — decision timeline:** a horizontal axis "frame index", a
    line plot of `s` vs frame, a dashed horizontal threshold line `τ = 0.45`,
    and green vertical markers where `s` dips below τ (= selected keyframes).
    Add a faint second flat line labeled "HSV colour similarity (baseline,
    ~constant)" to contrast that colour stays high while ORB drops.

## 3. Required labeled elements

- Equation for `s` rendered in clean serif math (LaTeX-style), placed near the
  score node: `s = r_in · (1 + (p̄/(d·0.06))²)⁻¹`.
- Boxes labeled: **ORB (≤1500 feats)**, **Lowe ratio 0.75**, **RANSAC homography
  (4 px)**, **inlier ratio**, **median parallax**, **viewpoint similarity s**,
  **threshold gate τ=0.45**.
- Legend: solid green = inlier match; red dashed = outlier; bold outline =
  keyframe; grey dashed = skipped.
- A short caption strip: *"Keep a frame only when its viewpoint similarity to
  the last keyframe drops below τ — geometric (ORB), not colour-based."*

## 4. Visual style (CVPR / academic)

- Flat vector look, thin (1–1.5 pt) strokes, rounded-rectangle nodes, subtle
  drop shadows or none.
- Restrained palette: near-black text on white; **one** accent (emerald green
  `#0E9D6E`) for "keep"/inliers, a muted red (`#D1495B`) for rejects/outliers,
  neutral grey for skipped/baseline. Avoid gradients and 3D bevels.
- Typeface: clean grotesque sans (Helvetica/Inter) for labels; serif math for
  equations.
- Arrows: solid for the kept path, dashed for the skip path. Consistent arrowheads.
- Panel tags **(a) (b) (c)** in bold if panels are separated.
- No photorealism; frames are stylized glyphs. Keep it printable in grayscale
  (rely on line style + labels, not only colour).

## 5. One-paragraph prompt (paste-ready)

> A clean CVPR-style algorithm pipeline figure, flat vector style on white
> background, thin strokes and rounded-rectangle nodes, emerald-green and muted-red
> accents only. Title region implied by a horizontal row of ~10 stylized video
> frame thumbnails (small rounded rectangles each showing an abstract excavator on
> terrain at slightly different camera angles); a few frames bold-outlined and
> labeled "KEYFRAME", the others greyed and dashed labeled "skipped". Below, an
> expanded "comparison unit": two frames side by side connected by green
> feature-match lines (solid green = inliers, faint red dashed = outliers),
> feeding a left-to-right chain of labeled boxes: "ORB keypoints (≤1500)" →
> "Lowe ratio-test matches (0.75)" → "RANSAC homography inliers (4px)" →
> a score node "viewpoint similarity s∈[0,1]" with a serif LaTeX equation
> s = inlier_ratio · (1+(parallax/diag/0.06)^2)^-1, then a diamond gate "s < τ
> (0.45)?" branching to "keep → new keyframe" (green solid arrow) and "skip"
> (grey dashed arrow). At the bottom, a small line plot: x-axis "frame index",
> curve of s with a dashed threshold line at 0.45, green dots where s dips below
> it, and a faint flat line labeled "HSV colour baseline (≈constant)" for
> contrast. Helvetica-like sans labels, serif math, legend box, academic paper
> aesthetic, high-contrast, grayscale-safe, no photoreal rendering, no gradients.
