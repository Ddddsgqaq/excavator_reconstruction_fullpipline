#!/usr/bin/env python
"""
test_coarse_dem.py — 独立测试：对 DEM 做粗网格重采样，平滑细小起伏、突出大地形。

不接入服务/界面，直接读某个 session 的 predictions.npz：
  1. 重力对齐 → rasterize_bev 得到细 DEM (H_top)。
  2. 把细网格按 factor×factor 的块聚合成粗网格（中位数/均值，抗噪）。
  3. 再上采样回细网格分辨率以便逐像素比较（最近邻，保持块状），
     或直接以粗网格显示。
  4. 出「粗采样前 / 后 / 残差」三联对比图。

用法：
  python test_coarse_dem.py <predictions.npz> [--grid 128] [--factor 8]
        [--agg median|mean] [--scale 28.0] [--out coarse_dem.png]
"""
import argparse
import numpy as np

from gravity_alignment import estimate_gravity
import terrain_analysis as ta


def coarse_resample(H, factor, agg="median"):
    """把 (res,res) 的高程图按 factor 块聚合成粗网格，返回：
       coarse (rc,rc) 粗网格；up (res,res) 上采样回原分辨率（块状）。
    NaN 空格在块内被忽略；整块全 NaN → 粗格为 NaN。"""
    res = H.shape[0]
    rc = res // factor
    usable = rc * factor                      # 丢掉不能整除的边缘
    Hc = H[:usable, :usable]
    blocks = Hc.reshape(rc, factor, rc, factor)

    if agg == "mean":
        coarse = np.nanmean(_all_nan_to_nan(blocks), axis=(1, 3))
    else:  # median，抗噪更强
        coarse = _nanmedian_blocks(blocks)

    # 上采样回原分辨率（最近邻/块复制），便于与原图逐像素比较与作残差
    up = np.kron(coarse, np.ones((factor, factor)))
    # 补回被整除丢掉的边缘（用最近粗格值填充）
    if usable < res:
        pad = res - usable
        up = np.pad(up, ((0, pad), (0, pad)), mode="edge")
    return coarse, up


def _all_nan_to_nan(blocks):
    return blocks


def _nanmedian_blocks(blocks):
    rc, f1, _, f2 = blocks.shape
    flat = blocks.transpose(0, 2, 1, 3).reshape(rc, rc, f1 * f2)
    with np.errstate(all="ignore"):
        return np.nanmedian(flat, axis=2)


def load_htop(npz_path, grid_res, conf=50.0):
    d = np.load(npz_path)
    pred = {k: np.array(d[k]) for k in d.files}
    pts = pred["world_points_from_depth"]
    cf = pred["depth_conf"]
    sem = pred.get("semantic_masks")
    gmask = (sem == 1) if sem is not None else None

    grav = estimate_gravity(extrinsic=pred["extrinsic"], world_points=pts,
                            ground_mask=gmask, conf=cf, conf_thres=conf / 100.0)
    pf = pts.reshape(-1, 3)
    cff = cf.reshape(-1).astype(np.float32)
    keep = np.isfinite(pf).all(1) & (cff >= (conf / 100.0) * cff.max())
    pa = pf[keep] @ grav.R_align.T
    semf = sem.reshape(-1)[keep] if sem is not None else None
    gf = (semf == 1) if semf is not None else None

    rast = ta.rasterize_bev(pa, semf, gf, grid_res=grid_res)
    return rast, grav


def render_compare(H_fine, H_up, coarse, bounds, out_path, sf, factor, agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, x1, z0, z1 = bounds
    extent = [x0, x1, z0, z1]

    Hf = H_fine * sf
    Hc = H_up * sf
    resid = (H_fine - H_up) * sf            # 被粗采样“平滑掉”的细节

    unit = "m" if sf != 1.0 else "u"
    lo = np.nanpercentile(Hf, 2)
    hi = np.nanpercentile(Hf, 98)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    fig.suptitle(f"Coarse DEM resample  ·  factor={factor} ({agg})  ·  1u={sf:g}m",
                 fontsize=14)

    im0 = axes[0].imshow(Hf, extent=extent, origin="lower", cmap="terrain",
                         vmin=lo, vmax=hi)
    axes[0].set_title(f"Before  ({H_fine.shape[0]}² fine DEM)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(Hc, extent=extent, origin="lower", cmap="terrain",
                         vmin=lo, vmax=hi, interpolation="nearest")
    axes[1].set_title(f"After  ({coarse.shape[0]}² coarse, upsampled)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    rmax = np.nanpercentile(np.abs(resid), 98) or 1.0
    im2 = axes[2].imshow(resid, extent=extent, origin="lower", cmap="RdBu_r",
                         vmin=-rmax, vmax=rmax)
    axes[2].set_title(f"Removed detail (before − after, {unit})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("X"); ax.set_ylabel("Z")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="path to predictions.npz")
    ap.add_argument("--grid", type=int, default=128, help="fine DEM resolution")
    ap.add_argument("--factor", type=int, default=8, help="coarse block size")
    ap.add_argument("--agg", choices=["median", "mean"], default="median")
    ap.add_argument("--scale", type=float, default=1.0, help="1 unit = N meters")
    ap.add_argument("--conf", type=float, default=50.0)
    ap.add_argument("--out", default="coarse_dem.png")
    args = ap.parse_args()

    rast, grav = load_htop(args.npz, args.grid, args.conf)
    H_fine = rast["H_top"]
    coarse, H_up = coarse_resample(H_fine, args.factor, args.agg)

    # 统计：粗采样抹掉了多少细节
    resid = H_fine - H_up
    rstd = float(np.nanstd(resid)) * args.scale
    rmax = float(np.nanmax(np.abs(resid))) * args.scale
    print(f"gravity={grav.source}  fine={H_fine.shape[0]}²  "
          f"coarse={coarse.shape[0]}²  factor={args.factor}({args.agg})")
    print(f"removed-detail: std={rstd:.3f}  max|.|={rmax:.3f}  (scaled by {args.scale})")

    render_compare(H_fine, H_up, coarse, rast["bounds"],
                   args.out, args.scale, args.factor, args.agg)
