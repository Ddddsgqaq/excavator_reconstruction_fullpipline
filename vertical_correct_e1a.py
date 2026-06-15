"""
vertical_correct_e1a.py  (experiment E1a)

Single-scalar vertical de-compression. E0 showed the compression is affine
(depth-independent), so the simplest correction is a global vertical stretch
about the reference ground plane:

    Y_corrected = ground_Y + k * (Y - ground_Y)

We calibrate k on one control object and validate on the other (leave-one-out),
quantifying the residual a *single scalar* leaves — which motivates the low-DOF
affine of E1b. All matplotlib text is English (Chinese only in the HTML caption).

Outputs: vertical_correct_e1a.png, vertical_correct_e1a_result.json. Run in `vggt`.
"""
import sys, os, json
sys.path.insert(0, "/home/maomaoyu/WS/vggt_yoloe")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scale_calibration as sc, gravity_alignment as ga

W = sys.argv[1] if len(sys.argv) > 1 else "workspaces/session_20260611_162643_869764"
CONTROLS = {"excavator": 3.0, "person": 1.7}

preds = sc.load_predictions(W)
conf = preds["world_points_conf"]
scale = json.load(open(f"{W}/scale_result.json"))["metric_scene"]["scale_m_per_unit"]
g = ga.estimate_gravity(preds["extrinsic"], preds["world_points"], conf=conf)
al = ga.apply_alignment_to_points(preds["world_points"], g.R_align)
masks = np.load(f"{W}/masks_f0.npz")

# reference ground plane Y = a*X + b*Z + c (robust, lower band)
P0 = al[0]; fin0 = np.isfinite(P0).all(2)
flat = al.reshape(-1, 3); cflat = conf.reshape(-1)
fk = np.isfinite(flat).all(1) & (cflat > 0.2 * cflat.max())
fp = flat[fk]; gp = fp[fp[:, 1] <= np.percentile(fp[:, 1], 60)]
coef, *_ = np.linalg.lstsq(np.c_[gp[:, 0], gp[:, 2], np.ones(len(gp))], gp[:, 1], rcond=None)
ground_Y = lambda X, Z: coef[0]*X + coef[1]*Z + coef[2]


def obj_points(cls):
    m = masks[cls] & fin0
    P = P0[m]; C = conf[0][m]
    return P[C >= np.percentile(C, 75)]


def apparent_h(pts):
    return (np.percentile(pts[:, 1], 98) - np.percentile(pts[:, 1], 2)) * scale


pts = {c: obj_points(c) for c in CONTROLS}
raw_h = {c: apparent_h(pts[c]) for c in CONTROLS}
k = {c: CONTROLS[c] / raw_h[c] for c in CONTROLS}          # per-object calibration
k_exc, k_per = k["excavator"], k["person"]
k_mean = float(np.mean([k_exc, k_per]))

# leave-one-out: calibrate on A, predict B's height
def predict(cls, kk):
    return kk * raw_h[cls]                                  # stretch multiplies extent by k

loo = {
    "calib_excavator_k": round(k_exc, 2),
    "predict_person_m": round(predict("person", k_exc), 3),
    "person_residual_pct": round(100*(predict("person", k_exc)-1.7)/1.7, 1),
    "calib_person_k": round(k_per, 2),
    "predict_excavator_m": round(predict("excavator", k_per), 3),
    "excavator_residual_pct": round(100*(predict("excavator", k_per)-3.0)/3.0, 1),
    "k_mean": round(k_mean, 2),
    "person_resid_kmean_pct": round(100*(predict("person", k_mean)-1.7)/1.7, 1),
    "excavator_resid_kmean_pct": round(100*(predict("excavator", k_mean)-3.0)/3.0, 1),
}


def stretch(pts, kk):
    base = ground_Y(pts[:, 0], pts[:, 2])
    out = pts.copy(); out[:, 1] = base + kk * (pts[:, 1] - base)
    return out


fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
for i, (cls, Hreal) in enumerate(CONTROLS.items()):
    p = pts[cls]; base = ground_Y(p[:, 0].mean(), p[:, 2].mean())
    pc = stretch(p, k_exc)                                   # corrected with excavator-calibrated k
    z = (p[:, 2] - p[:, 2].mean()) * scale
    ax[i].scatter(z, (p[:, 1]-base)*scale, s=7, c="#bbbbbb", label=f"raw  ({raw_h[cls]:.2f} m)")
    ax[i].scatter(z, (pc[:, 1]-base)*scale, s=7, c="#2a9d8f", alpha=.7,
                  label=f"corrected x{k_exc:.1f} ({predict(cls,k_exc):.2f} m)")
    ax[i].plot([z.min()-0.2]*2, [0, Hreal], "r-", lw=4, label=f"real {Hreal} m")
    ax[i].set_aspect("equal"); ax[i].set_xlabel("Z aligned (m)"); ax[i].set_ylabel("height above ground (m)")
    tag = "calibration obj" if cls == "excavator" else "validation obj"
    ax[i].set_title(f"{cls} ({tag})"); ax[i].legend(fontsize=8, loc="upper right")

# panel C: real vs raw vs corrected(k_exc)
labels = list(CONTROLS); xpos = np.arange(len(labels)); w = 0.26
real = [CONTROLS[c] for c in labels]; rawv = [raw_h[c] for c in labels]
corr = [predict(c, k_exc) for c in labels]
ax[2].bar(xpos-w, real, w, label="real", color="#e76f51")
ax[2].bar(xpos,   rawv, w, label="raw VGGT", color="#bbbbbb")
ax[2].bar(xpos+w, corr, w, label=f"corrected x{k_exc:.1f}", color="#2a9d8f")
for j, c in enumerate(labels):
    resid = 100*(corr[j]-real[j])/real[j]
    ax[2].text(xpos[j]+w, corr[j]+0.08, f"{resid:+.0f}%", ha="center", fontsize=9, weight="bold")
ax[2].set_xticks(xpos); ax[2].set_xticklabels(labels); ax[2].set_ylabel("height (m)")
ax[2].set_title("single scalar (calib=excavator): person residual "
                f"{loo['person_residual_pct']:+.0f}%"); ax[2].legend(fontsize=8)

fig.suptitle(f"E1a single-scalar vertical de-compression  |  k_exc={k_exc:.1f}  k_person={k_per:.1f}  "
             f"k_mean={k_mean:.1f}  (residual ~30% => need low-DOF affine, E1b)", fontsize=12)
fig.tight_layout()
out = f"{W}/vertical_correct_e1a.png"; fig.savefig(out, dpi=115); print("wrote", out)

res = dict(workspace=W, scale_m_per_unit=scale,
           raw_apparent_height_m={c: round(raw_h[c], 3) for c in CONTROLS},
           per_object_k={c: round(k[c], 2) for c in CONTROLS},
           leave_one_out=loo,
           verdict="single scalar fixes order-of-magnitude but leaves ~30% cross-object "
                   "residual; affine (E0) is depth-independent so a low-DOF affine fit "
                   "with >=3 control objects (E1b / big-pit poles) should tighten it")
json.dump(res, open(f"{W}/vertical_correct_e1a_result.json", "w"), indent=1)
print(json.dumps(res, indent=1, ensure_ascii=False))
