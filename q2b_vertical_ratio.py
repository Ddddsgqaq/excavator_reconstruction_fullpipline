"""
q2b_vertical_ratio.py  (experiment Q2-absolute, identifiable part)

Absolute vertical compression k_v is NOT cleanly identifiable per scene: the only
known sizes (person 1.7 m, excavator 3.0 m) are VERTICAL, the compressed axis, so
calibrating metric scale from them absorbs the compression (scale & k_v entangle).
An external horizontal ruler / telemetry is required -> exactly what the big-pit
calibration poles provide.

What IS identifiable and scale-free is the cross-object vertical-extent ratio
    R = excavator_Yext / person_Yext      (ideal, uniform compression: 3.0/1.7 = 1.76)
which is nearly halo-free (halo inflates the horizontal footprint, not vertical Y).
We test its stability across scenes. Cleaner than the aspect deficit (Q2).

Reuses vertical_fidelity_results.json. English labels. Outputs png + json.
"""
import os, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

VY = "/home/maomaoyu/WS/vggt_yoloe"
IDEAL = 3.0 / 1.7
PX_OK = 350   # person mask px above which the small-object probe is reliable

d = json.load(open(f"{VY}/vertical_fidelity_results.json"))
by = collections.defaultdict(dict)
for e in d:
    by[e["scene"]][e["class"]] = e

rows = []
for sc, cd in by.items():
    if "excavator" in cd and "person" in cd:
        R = cd["excavator"]["Y_ext"] / cd["person"]["Y_ext"]
        rows.append(dict(scene=sc.replace("session_", ""), R=R,
                         person_px=cd["person"]["mask_px"]))
rows.sort(key=lambda r: r["person_px"])
R = np.array([r["R"] for r in rows]); px = np.array([r["person_px"] for r in rows])
ok = px >= PX_OK
mean_ok, std_ok = float(R[ok].mean()), float(R[ok].std())
cv_ok = std_ok / mean_ok * 100

fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))

# A: R per scene, split by probe reliability
xi = np.arange(len(rows))
ax[0].scatter(xi[ok], R[ok], s=80, color="#2a9d8f", zorder=5,
              label=f"person well-resolved (>={PX_OK}px), n={ok.sum()}")
ax[0].scatter(xi[~ok], R[~ok], s=80, color="#bbbbbb", zorder=5,
              label=f"person too small (<{PX_OK}px) — artifact")
ax[0].axhline(mean_ok, color="#2a9d8f", ls="--", lw=1.5,
              label=f"well-resolved mean = {mean_ok:.2f} (CV {cv_ok:.1f}%)")
ax[0].axhline(IDEAL, color="#e76f51", ls=":", lw=1.5, label=f"ideal uniform = {IDEAL:.2f}")
ax[0].set_xlabel("scene (sorted by person mask size)")
ax[0].set_ylabel("R = excavator_Yext / person_Yext")
ax[0].set_title("cross-object vertical ratio is stable when both probes are clean")
ax[0].legend(fontsize=8.5)

# B: R vs person px
ax[1].scatter(px, R, s=80, c=np.where(ok, "#2a9d8f", "#bbbbbb"), zorder=5)
ax[1].axhline(mean_ok, color="#2a9d8f", ls="--", lw=1.2)
ax[1].axhline(IDEAL, color="#e76f51", ls=":", lw=1.2)
ax[1].axvline(PX_OK, color="gray", ls=":", lw=1)
ax[1].set_xlabel("person mask size (px)"); ax[1].set_ylabel("R")
ax[1].set_title("the low-R scenes are exactly the small-person (noisy) ones")

rel = mean_ok / IDEAL
fig.suptitle(f"Q2b cross-object vertical-ratio stability — R={mean_ok:.2f} (CV {cv_ok:.1f}%) across "
             f"{ok.sum()} clean scenes;  R/ideal={rel:.2f} => person compressed {rel:.2f}x more than excavator",
             fontsize=12)
fig.tight_layout()
out = f"{VY}/q2b_vertical_ratio.png"; fig.savefig(out, dpi=115); print("wrote", out)

res = dict(
    metric="R = excavator_Yext / person_Yext (scale-free, halo-light), ideal uniform = 1.76",
    ideal=round(IDEAL, 3), px_threshold=PX_OK,
    well_resolved=dict(n=int(ok.sum()), mean=round(mean_ok, 3), std=round(std_ok, 3), cv_pct=round(cv_ok, 1)),
    rows=[{**r, "R": round(r["R"], 3)} for r in rows],
    relative_compression_person_over_excavator=round(rel, 2),
    identifiability_note="absolute k_v needs an external horizontal ruler / telemetry (the big-pit "
                         "poles); only the scale-free RATIO is identifiable here.",
    verdict=f"cross-object vertical ratio is highly stable (CV {cv_ok:.1f}%) when both probes are "
            "well-resolved -> reinforces Q2: vertical distortion is systematic & reproducible at the "
            f"object level. Person is compressed {rel:.2f}x more than excavator (small-object effect).",
)
json.dump(res, open(f"{VY}/q2b_vertical_ratio_result.json", "w"), indent=1)
print(json.dumps(res, indent=1, ensure_ascii=False))
