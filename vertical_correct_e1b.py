"""
vertical_correct_e1b.py  (experiment E1b)

Upgrade the single-scalar vertical correction (E1a) toward a low-DOF model, and
— more importantly — characterize WHICH model the data supports and what we still
cannot decide with only 2 control objects. We compare three candidate maps from
true height H to reconstructed apparent height h:

    M0  single scalar (calibrated on excavator):   h = H / k_exc
    M1  proportional, least-squares on all objs:   h = s * H            (through origin)
    M2  affine with offset (slope + intercept):    h = s * H + b        (fixed height loss)

M2 distinguishes "everything scaled by k" from "a fixed height-loss + a scale"
(the latter hits small objects harder — a natural explanation for k_person>k_exc).
With 2 controls M2 fits EXACTLY (2 params, 2 points) so it is a HYPOTHESIS, not a
validated fit; the panel that matters is how differently the models invert a
shallow apparent feature (e.g. a pit) — that gap is exactly what calibration
poles in the big-pit experiment will resolve.

All matplotlib text English. Outputs: vertical_correct_e1b.png, *_result.json.
"""
import sys, os, json
sys.path.insert(0, "/home/maomaoyu/WS/vggt_yoloe")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scale_calibration as sc, gravity_alignment as ga

W = sys.argv[1] if len(sys.argv) > 1 else "workspaces/session_20260611_162643_869764"
# control objects: (label, real height m).  Extend this list when poles are available.
CONTROLS = [("excavator", 3.0), ("person", 1.7)]

preds = sc.load_predictions(W)
conf = preds["world_points_conf"]
scale = json.load(open(f"{W}/scale_result.json"))["metric_scene"]["scale_m_per_unit"]
g = ga.estimate_gravity(preds["extrinsic"], preds["world_points"], conf=conf)
al = ga.apply_alignment_to_points(preds["world_points"], g.R_align)
masks = np.load(f"{W}/masks_f0.npz")
P0 = al[0]; fin0 = np.isfinite(P0).all(2)


def apparent_h(cls):
    m = masks[cls] & fin0
    P = P0[m]; C = conf[0][m]
    p = P[C >= np.percentile(C, 75)]
    return (np.percentile(p[:, 1], 98) - np.percentile(p[:, 1], 2)) * scale


H = np.array([h for _, h in CONTROLS])              # true heights
h = np.array([apparent_h(c) for c, _ in CONTROLS])  # reconstructed apparent heights
labels = [c for c, _ in CONTROLS]

# ── Fit the three models (h as a function of H) ──────────────────────────────
k_exc = H[0] / h[0]                                  # M0: scalar from excavator
M0 = dict(name="M0 single-scalar (calib=excavator)", s=1.0/k_exc, b=0.0)
s1 = float(np.sum(H * h) / np.sum(H * H))            # M1: proportional LS
M1 = dict(name="M1 proportional (LS, through origin)", s=s1, b=0.0)
A = np.c_[H, np.ones_like(H)]; s2, b2 = np.linalg.lstsq(A, h, rcond=None)[0]  # M2 affine
M2 = dict(name="M2 affine with offset", s=float(s2), b=float(b2))
MODELS = [M0, M1, M2]


def invert(model, h_app):       # apparent -> true height (the correction)
    return (np.asarray(h_app) - model["b"]) / model["s"]


# residual of each model: predicted apparent vs measured apparent (per control)
for M in MODELS:
    pred = M["s"] * H + M["b"]
    M["pred_apparent"] = pred.tolist()
    M["max_abs_resid_pct"] = float(np.max(np.abs((pred - h) / h)) * 100)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
xs = np.linspace(0, 3.3, 100)

# A: H (true) vs h (reconstructed) with fitted lines
ax[0].scatter(H, h, s=90, c="#e76f51", zorder=5, label="control objects")
for c, xx, yy in zip(labels, H, h):
    ax[0].annotate(c, (xx, yy), textcoords="offset points", xytext=(6, 6), fontsize=9)
for M, col, ls in zip(MODELS, ["#999", "#3a86b8", "#2a9d8f"], ["--", "-.", "-"]):
    ax[0].plot(xs, M["s"]*xs + M["b"], col, ls=ls, label=M["name"])
ax[0].plot(xs, xs, "k:", lw=1, label="ideal (no compression)")
ax[0].set_xlabel("true height H (m)"); ax[0].set_ylabel("reconstructed apparent height h (m)")
ax[0].set_title("fit: how VGGT shrinks true height"); ax[0].legend(fontsize=8); ax[0].set_xlim(0, 3.3); ax[0].set_ylim(0, 3.4)

# B: implied per-object compression factor k = H/h, and whether it depends on size
kvals = H / h
ax[1].bar(labels, kvals, color=["#2a9d8f", "#e9c46a"])
for i, v in enumerate(kvals):
    ax[1].text(i, v+0.2, f"k={v:.1f}", ha="center", fontsize=11, weight="bold")
ax[1].set_ylabel("compression factor k = H / h")
ax[1].set_title("smaller object -> larger k ?\n(offset model M2 would explain this)")

# C: THE decisive panel — how differently each model corrects a shallow apparent feature
app = np.linspace(0.05, 1.0, 100)   # e.g. a pit reconstructed this many metres deep
for M, col, ls in zip(MODELS, ["#999", "#3a86b8", "#2a9d8f"], ["--", "-.", "-"]):
    ax[2].plot(app, invert(M, app), col, ls=ls, lw=2, label=M["name"])
# mark a worked example at apparent 0.3 m
for M, col in zip(MODELS, ["#999", "#3a86b8", "#2a9d8f"]):
    ax[2].scatter([0.3], [invert(M, 0.3)], color=col, zorder=5)
ax[2].axvline(0.3, color="gray", ls=":", lw=1)
ax[2].set_xlabel("apparent (reconstructed) depth of a feature (m)")
ax[2].set_ylabel("corrected TRUE depth (m)")
ax[2].set_title("models diverge for shallow features\n=> need >=3 poles to pick the model")
ax[2].legend(fontsize=8)

fig.suptitle("E1b vertical-correction models — 2 controls fit exactly, cannot yet validate; "
             "divergence (panel C) is why the big-pit needs calibration poles", fontsize=12)
fig.tight_layout()
out = f"{W}/vertical_correct_e1b.png"; fig.savefig(out, dpi=115); print("wrote", out)

# worked example numbers
ex_app = 0.30
example = {M["name"]: round(float(invert(M, ex_app)), 2) for M in MODELS}

res = dict(
    workspace=W, scale_m_per_unit=scale,
    controls=[dict(label=l, true_h=float(t), apparent_h=round(float(a), 3), k=round(float(t/a), 2))
              for l, t, a in zip(labels, H, h)],
    models=[{k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in M.items()} for M in MODELS],
    worked_example_apparent_0p3m_to_true=example,
    notes="M2 (offset) fits 2 points exactly -> hypothesis only. M2 slope/intercept imply a "
          "fixed additive height-loss that hits small objects harder, a candidate explanation for "
          "k_person>k_exc. Distinguishing M0/M1/M2 requires >=3 known-height controls (calibration "
          "poles at varied positions) -> concrete requirement for the big-pit capture.")
json.dump(res, open(f"{W}/vertical_correct_e1b_result.json", "w"), indent=1)
print(json.dumps(res, indent=1, ensure_ascii=False))
