"""
terrain_analysis.py — 语义×高程融合的地形理解（纯计算，无服务依赖）。

输入是已经处于**重力对齐帧**的点（Y=up）与逐点语义标签，输出 BEV 栅格图层
（高程/残差/语义/坡度/粗糙度/连通域）以及结构化的区域列表（土堆/坑，附材质确认）。

数据流：
    rasterize_bev      —— 步骤2：把点聚合到俯视 X-Z 栅格（高度中位/顶百分位、语义众数）
    extract_geometry   —— 步骤4：残差 R = H_top - H_ground，连通域=土堆/坑，坡度/粗糙度/体积
    confirm_semantics  —— 步骤5：每个连通域取语义众数 → 关键字规则 → 可挖料堆 / 排除物体
    analyze_terrain    —— 顶层编排，返回可 JSON 化的 dict

坐标约定：点为 (N,3) 的对齐帧坐标，列 [X, Y, Z]，其中 Y 为高程（up），(X,Z) 为水平面。
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata


# ── 步骤5 材质关键字规则表（可被入参覆盖） ────────────────────────────────────
# 按类名（YOLOe text prompt / sem_id_map 的键）做子串匹配，判定材质类别。
DEFAULT_MATERIAL_RULES = {
    "excavatable": [
        "soil", "dirt", "earth", "mud", "clay",
        "gravel", "sand", "aggregate", "rubble",
        "mound", "pile", "heap", "spoil", "stockpile",
    ],
    "obstacle": [
        "vehicle", "truck", "car", "excavator", "digger", "machine",
        "rock", "boulder", "stone",
        "person", "human", "worker", "people",
        "building", "wall", "pole", "tree",
    ],
}


def classify_material(class_name: str, rules: dict) -> str:
    """类名 → 类别（excavatable / obstacle / unknown），子串不区分大小写匹配。"""
    if not class_name:
        return "unknown"
    name = class_name.lower()
    for category, keywords in rules.items():
        for kw in keywords:
            if kw in name:
                return category
    return "unknown"


# ── 步骤2 栅格化到 BEV ────────────────────────────────────────────────────────

def _cell_index(coord, lo, hi, res):
    """把连续坐标映射到 [0, res) 的整数格索引（越界裁剪）。"""
    idx = ((coord - lo) / (hi - lo + 1e-12) * res).astype(np.int64)
    return np.clip(idx, 0, res - 1)


def rasterize_bev(pts_aligned: np.ndarray,
                  sem_labels: np.ndarray | None,
                  ground_mask: np.ndarray | None,
                  grid_res: int = 128,
                  top_percentile: float = 90.0,
                  bounds: tuple | None = None):
    """
    把对齐帧的点聚合到俯视 (X,Z) 栅格。

    参数
    ----
    pts_aligned : (N,3)  对齐帧点，列 [X, Y(up), Z]
    sem_labels  : (N,)   逐点语义 id（与 pts 同序），可为 None
    ground_mask : (N,)   逐点地面布尔掩码（用于参考地面插值），可为 None
    grid_res    : 栅格边长（格数）
    top_percentile : 每格取该高度百分位作为“表面顶” H_top（抗噪）
    bounds      : (x_min,x_max,z_min,z_max)，None 时按点范围+2% padding 自动取

    返回 dict
    ----
    H_top    (res,res) 每格表面顶高度（空格为 NaN）
    H_ground (res,res) 参考地面高度（对 ground 点插值，全网格有值）
    S_mode   (res,res) 每格语义众数（int，空格为 0）
    count    (res,res) 每格点数
    xx, zz   (res,res) 网格中心坐标
    bounds   实际使用的 (x_min,x_max,z_min,z_max)
    cell_area 单格面积（世界单位²）
    """
    x = pts_aligned[:, 0]
    y = pts_aligned[:, 1]
    z = pts_aligned[:, 2]

    if bounds is None:
        x_min, x_max = float(x.min()), float(x.max())
        z_min, z_max = float(z.min()), float(z.max())
        xp = (x_max - x_min) * 0.02
        zp = (z_max - z_min) * 0.02
        x_min -= xp; x_max += xp; z_min -= zp; z_max += zp
    else:
        x_min, x_max, z_min, z_max = bounds

    ix = _cell_index(x, x_min, x_max, grid_res)
    iz = _cell_index(z, z_min, z_max, grid_res)
    flat = iz * grid_res + ix          # 行=z, 列=x
    ncell = grid_res * grid_res

    # 每格点数
    count = np.bincount(flat, minlength=ncell).reshape(grid_res, grid_res)

    # H_top：每格高度的 top_percentile。用分组排序一次算完，避免逐格循环。
    H_top = np.full(ncell, np.nan, dtype=np.float64)
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    y_sorted = y[order]
    # 每个非空格的 [start,end) 区间
    boundaries = np.flatnonzero(np.diff(flat_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(flat_sorted)]))
    for s, e in zip(starts, ends):
        cell = flat_sorted[s]
        H_top[cell] = np.percentile(y_sorted[s:e], top_percentile)
    H_top = H_top.reshape(grid_res, grid_res)

    # S_mode：每格语义众数（0 视为背景/无标签）
    S_mode = np.zeros((grid_res, grid_res), dtype=np.int32)
    if sem_labels is not None:
        lab_sorted = np.asarray(sem_labels)[order]
        for s, e in zip(starts, ends):
            vals = lab_sorted[s:e]
            vals = vals[vals > 0]
            if vals.size:
                u, c = np.unique(vals, return_counts=True)
                S_mode.flat[flat_sorted[s]] = int(u[np.argmax(c)])

    # 网格中心坐标
    xi = np.linspace(x_min, x_max, grid_res)
    zi = np.linspace(z_min, z_max, grid_res)
    xx, zz = np.meshgrid(xi, zi)

    # H_ground：对 ground 点插值出参考地面（linear + nearest 回退，全网格有值）
    H_ground = _interp_ground(pts_aligned, ground_mask, xx, zz)

    cell_area = ((x_max - x_min) / grid_res) * ((z_max - z_min) / grid_res)

    return {
        "H_top": H_top,
        "H_ground": H_ground,
        "S_mode": S_mode,
        "count": count,
        "xx": xx,
        "zz": zz,
        "bounds": (x_min, x_max, z_min, z_max),
        "cell_area": float(cell_area),
    }


def _interp_ground(pts_aligned, ground_mask, xx, zz):
    """对 ground 点（或低百分位回退点）插值出参考地面高度网格。"""
    if ground_mask is not None and np.asarray(ground_mask, dtype=bool).sum() >= 50:
        gpts = pts_aligned[np.asarray(ground_mask, dtype=bool)]
    else:
        # 回退：取全局低 20% 高度的点作为地面近似
        y = pts_aligned[:, 1]
        thresh = np.percentile(y, 20.0)
        gpts = pts_aligned[y <= thresh]

    src = gpts[:, [0, 2]]
    val = gpts[:, 1]
    lin = griddata(src, val, (xx, zz), method="linear")
    nea = griddata(src, val, (xx, zz), method="nearest")
    return np.where(np.isnan(lin), nea, lin)


# ── 步骤4 几何结构提取（全部在残差 R 上，无需训练） ──────────────────────────

def _label_components(binary: np.ndarray, min_cells: int):
    """4-连通标记，丢弃小于 min_cells 的碎片。返回 (label_map, [region_slices...])。"""
    lab, n = ndimage.label(binary)
    if n == 0:
        return np.zeros_like(lab), []
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep_ids = [i + 1 for i, s in enumerate(sizes) if s >= min_cells]
    # 重新编号为连续 1..K
    remap = np.zeros(n + 1, dtype=np.int32)
    for new_id, old_id in enumerate(keep_ids, start=1):
        remap[old_id] = new_id
    return remap[lab], keep_ids


def extract_geometry(rast: dict,
                     tau: float | None = None,
                     tau_frac: float = 0.1,
                     min_area_frac: float = 0.002):
    """
    从栅格图层提取几何结构。

    参数
    ----
    rast : rasterize_bev 的返回
    tau  : 残差阈值（世界单位）；|R|>tau 视为显著起伏。None 时按 tau_frac 自适应：
           tau = tau_frac · (H_top 的 p2–p98 值域)。VGGT 存在垂直压缩，绝对阈值
           跨场景不可移植，故默认相对。
    tau_frac : 自适应比例（默认 0.1）
    min_area_frac : 连通域最小面积占比（相对整幅栅格），过滤碎片

    返回 dict
    ----
    R         (res,res) 残差 H_top - H_ground（空 H_top 处为 NaN）
    slope     (res,res) 坡度幅值 |∇H_top|
    roughness (res,res) 局部高度方差
    mound_id  (res,res) 土堆连通域标签（0=背景）
    pit_id    (res,res) 坑/沟连通域标签（0=背景）
    mounds/pits : [dict...] 每连通域的几何量（id/area/rel_volume/peak/centroid/cells）
    """
    H_top = rast["H_top"]
    H_ground = rast["H_ground"]
    xx, zz = rast["xx"], rast["zz"]
    cell_area = rast["cell_area"]
    res = H_top.shape[0]

    R = H_top - H_ground                       # NaN 处保持 NaN
    valid = np.isfinite(R)

    if tau is None:
        ht = H_top[np.isfinite(H_top)]
        y_range = float(np.percentile(ht, 98) - np.percentile(ht, 2)) if ht.size else 1.0
        tau = max(tau_frac * y_range, 1e-6)

    # 坡度：对 H_top 填洞后求梯度幅值
    H_fill = np.where(valid, H_top, H_ground)
    gz, gx = np.gradient(H_fill)
    slope = np.hypot(gx, gz)

    # 粗糙度：局部均方 - 局部均值²（3x3 窗）
    win = 3
    mean = ndimage.uniform_filter(H_fill, size=win)
    sq = ndimage.uniform_filter(H_fill ** 2, size=win)
    roughness = np.clip(sq - mean ** 2, 0.0, None)

    min_cells = max(4, int(min_area_frac * res * res))
    mound_bin = valid & (R > tau)
    pit_bin = valid & (R < -tau)
    mound_id, _ = _label_components(mound_bin, min_cells)
    pit_id, _ = _label_components(pit_bin, min_cells)

    mounds = _summarize_components(mound_id, R, xx, zz, cell_area, sign=+1)
    pits = _summarize_components(pit_id, R, xx, zz, cell_area, sign=-1)

    return {
        "R": R,
        "slope": slope,
        "roughness": roughness,
        "mound_id": mound_id,
        "pit_id": pit_id,
        "mounds": mounds,
        "pits": pits,
        "tau": float(tau),
    }


def _summarize_components(label_map, R, xx, zz, cell_area, sign):
    """对每个连通域计算 面积/相对体积/峰值/质心/格数。"""
    out = []
    n = int(label_map.max())
    for cid in range(1, n + 1):
        m = label_map == cid
        rvals = R[m]
        rvals = rvals[np.isfinite(rvals)]
        if rvals.size == 0:
            continue
        area = float(m.sum() * cell_area)
        rel_volume = float(np.abs(rvals).sum() * cell_area)   # Σ|R|·格面积
        peak = float(rvals.max() if sign > 0 else rvals.min())
        centroid = [float(xx[m].mean()), float(zz[m].mean())]
        out.append({
            "id": cid,
            "cells": int(m.sum()),
            "area": area,
            "rel_volume": rel_volume,
            "peak": peak,
            "centroid": centroid,
        })
    return out


# ── 步骤5 语义×几何联合确认 ──────────────────────────────────────────────────

def confirm_semantics(geom: dict,
                      S_mode: np.ndarray,
                      id_to_name: dict,
                      rules: dict):
    """
    对每个连通域取语义众数 → 反查类名 → 规则表分类。就地为每个 region 追加
    material/category/keep，并返回统一的 regions 列表（土堆+坑）。

    - 土堆：keep = (category == 'excavatable')
    - 坑/沟：不做材质门控，keep=True（挖掘目标坑），material 仅作参考
    """
    regions = []
    for kind, key in (("mound", "mounds"), ("pit", "pits")):
        for comp in geom[key]:
            m = geom[f"{kind}_id"] == comp["id"]
            labels = S_mode[m]
            labels = labels[labels > 0]
            if labels.size:
                u, c = np.unique(labels, return_counts=True)
                sem_id = int(u[np.argmax(c)])
            else:
                sem_id = 0
            name = id_to_name.get(sem_id, "") if sem_id else ""
            category = classify_material(name, rules)
            keep = (category == "excavatable") if kind == "mound" else True
            regions.append({
                "kind": kind,
                **comp,
                "sem_id": sem_id,
                "material": name or "unlabeled",
                "category": category,
                "keep": bool(keep),
            })
    return regions


# ── 步骤6' 多层融合 → 作业地图（work-site map） ──────────────────────────────
# 不靠语义细分土的种类（2D 分割难做到），而是用 几何(R)+坡度+语义排除 推导每格的
# “作业语义”：对挖掘任务意味着什么。类型少而明确。

# 作业类型编码（zone_map 的取值）与配色（RGB 0-1，供 PNG/viewer 共用）
ZONE_FLAT     = 0   # 平地/可通行
ZONE_DIG      = 1   # 挖掘区（坑/沟，可下铲）
ZONE_DUMP     = 2   # 放料区（大片平坦、远离障碍）
ZONE_PILE     = 3   # 料堆（可挖凸起）
ZONE_HAZARD   = 4   # 危险（陡坎/紧邻障碍的高凸起）
ZONE_OBSTACLE = 5   # 障碍（挖机/人/车…）
ZONE_EMPTY    = -1  # 无数据

ZONE_NAMES = {
    ZONE_FLAT: "flat", ZONE_DIG: "dig", ZONE_DUMP: "dump",
    ZONE_PILE: "pile", ZONE_HAZARD: "hazard", ZONE_OBSTACLE: "obstacle",
}
ZONE_COLORS = {
    ZONE_FLAT:     [0.82, 0.82, 0.80],
    ZONE_DIG:      [0.95, 0.55, 0.15],   # 橙
    ZONE_DUMP:     [0.20, 0.55, 0.90],   # 蓝
    ZONE_PILE:     [0.30, 0.75, 0.35],   # 绿
    ZONE_HAZARD:   [0.90, 0.15, 0.15],   # 红
    ZONE_OBSTACLE: [0.45, 0.45, 0.45],   # 灰
}

# 可挖性：dig(坑/沟) 与 pile(可挖料堆) 可下铲；平地/放料区/危险/障碍不可挖。
# 供外部程序（如 Unity 联动）按作业区判定「可挖 vs 不可挖」，见 zone_legend()。
ZONE_DIGGABLE = {
    ZONE_FLAT:     False,
    ZONE_DIG:      True,
    ZONE_DUMP:     False,
    ZONE_PILE:     True,
    ZONE_HAZARD:   False,
    ZONE_OBSTACLE: False,
}


def zone_legend() -> list:
    """作业区图例：[{code, name, diggable, color}...]，供导出到地形 JSON 的 semantic 块。

    与 zone_map 的取值一一对应（不含 ZONE_EMPTY=-1，空格由消费方按无数据处理，
    默认 diggable=false）。颜色为 RGB 0-1，复用 ZONE_COLORS。
    """
    return [
        {
            "code": int(code),
            "name": ZONE_NAMES[code],
            "diggable": bool(ZONE_DIGGABLE[code]),
            "color": list(ZONE_COLORS[code]),
        }
        for code in ZONE_NAMES
    ]


def build_worksite_map(rast: dict, geom: dict, regions: list,
                       id_to_name: dict, rules: dict,
                       slope_hazard: float = 0.6,
                       dump_min_area_frac: float = 0.01):
    """把多层信息融合成一张作业地图。

    返回 dict:
      zone_map   (res,res) int  每格作业类型（见 ZONE_* 常量），空格=ZONE_EMPTY
      zones      [dict]         每类聚合（type/name/cells/area）
      next_scoop dict|None      下一铲建议：{xz, zone, score, heading_deg}
    """
    H_top = rast["H_top"]
    S_mode = rast["S_mode"]
    cell_area = rast["cell_area"]
    R = geom["R"]
    slope = geom["slope"]
    tau = geom["tau"]
    res = R.shape[0]
    valid = np.isfinite(R)

    zone = np.full((res, res), ZONE_EMPTY, dtype=np.int32)

    # 1) 障碍：语义命中 obstacle 规则的格子（最高优先级，先铺）
    obstacle_ids = {sid for sid, name in id_to_name.items()
                    if classify_material(name, rules) == "obstacle"}
    obstacle_mask = np.isin(S_mode, list(obstacle_ids)) if obstacle_ids else \
        np.zeros_like(S_mode, dtype=bool)

    # 2) 基础按 R 分：平地 / 凸起 / 凹陷
    flat_mask = valid & (np.abs(R) <= tau)
    up_mask   = valid & (R > tau)
    down_mask = valid & (R < -tau)

    zone[flat_mask] = ZONE_FLAT
    zone[down_mask] = ZONE_DIG          # 坑/沟 → 挖掘区
    zone[up_mask]   = ZONE_PILE         # 凸起 → 料堆（下面再按语义/坡度细化）

    # 3) 危险：坡度过陡（陡坎），或凸起且语义=障碍旁 → HAZARD
    hazard_mask = valid & (slope > slope_hazard)
    zone[hazard_mask] = ZONE_HAZARD

    # 4) 障碍覆盖（压过前面所有）
    zone[obstacle_mask] = ZONE_OBSTACLE

    # 5) 料堆材质门控：凸起若语义=障碍类，已在 4) 变 OBSTACLE；若语义=excavatable
    #    或未知，则保持 PILE。（不额外处理，规则已隐含）

    # 6) 放料区：在平地里选“最大的连通平坦块”作为 DUMP（远离障碍）
    from scipy import ndimage as _ndi
    flat_only = (zone == ZONE_FLAT)
    # 远离障碍：对障碍做距离变换，剔除太靠近障碍的平地
    if obstacle_mask.any():
        dist = _ndi.distance_transform_edt(~obstacle_mask)
        flat_only = flat_only & (dist > res * 0.05)
    lab, n = _ndi.label(flat_only)
    if n > 0:
        sizes = _ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        biggest = int(np.argmax(sizes)) + 1
        if sizes[biggest - 1] >= dump_min_area_frac * res * res:
            zone[lab == biggest] = ZONE_DUMP

    # ── 作业类型聚合 ──
    zones = []
    for z, name in ZONE_NAMES.items():
        cells = int((zone == z).sum())
        if cells:
            zones.append({"type": int(z), "name": name,
                          "cells": cells, "area": float(cells * cell_area)})

    # ── 机械/障碍物：逐连通域给出类名 + 质心（让图上能明确标“挖掘机在这”）──
    machines = []
    if obstacle_mask.any():
        xx, zz = rast["xx"], rast["zz"]
        olab, on = _ndi.label(obstacle_mask)
        for cid in range(1, on + 1):
            m = olab == cid
            if int(m.sum()) < 3:              # 丢弃碎片
                continue
            # 该连通域的语义众数 → 类名
            labs = S_mode[m]; labs = labs[labs > 0]
            sid = int(np.bincount(labs).argmax()) if labs.size else 0
            name = id_to_name.get(sid, "obstacle") if sid else "obstacle"
            machines.append({
                "label": name,
                "sem_id": sid,
                "cells": int(m.sum()),
                "area": float(m.sum() * cell_area),
                "centroid": [float(xx[m].mean()), float(zz[m].mean())],
            })

    # ── 下一铲下铲点：在 DIG(优先) 或最大可挖 PILE 内选综合评分最高格 ──
    next_scoop = _pick_next_scoop(zone, R, slope, obstacle_mask, rast, tau)

    return {"zone_map": zone, "zones": zones,
            "machines": machines, "next_scoop": next_scoop}


def _pick_next_scoop(zone, R, slope, obstacle_mask, rast, tau):
    """下铲点 = 最深坑的边缘。

    逻辑：
      1. 在挖掘区(ZONE_DIG)的连通域里挑**最深的坑**（域内 R 最小 = 最负）。
         若无 dig，退而用最高的料堆(ZONE_PILE)。
      2. 取该坑的**边缘格**（坑掩码腐蚀一圈后的差集 = 内边界）。
      3. 边缘格里选**最深**(|R| 最大)的一格作为下铲点。
    在坑边下铲符合挖掘作业习惯（沿坡逐层向下挖）。
    """
    from scipy import ndimage as _ndi

    use_dig = (zone == ZONE_DIG).any()
    target_zone = ZONE_DIG if use_dig else ZONE_PILE
    mask = (zone == target_zone)
    if not mask.any():
        return None

    # 1) 找最深(或最高)的连通域
    lab, n = _ndi.label(mask)
    best_cid, best_depth = 0, None
    for cid in range(1, n + 1):
        m = lab == cid
        rv = R[m]
        rv = rv[np.isfinite(rv)]
        if rv.size == 0:
            continue
        # dig: 最负的 R 越小越深；pile: 最大的 R 越大越高
        depth = -rv.min() if use_dig else rv.max()
        if best_depth is None or depth > best_depth:
            best_depth, best_cid = depth, cid
    if best_cid == 0:
        return None
    comp = (lab == best_cid)

    # 2) 边缘格：腐蚀一圈后从原掩码里去掉 → 保留最外一圈（内边界）
    eroded = _ndi.binary_erosion(comp, iterations=1, border_value=0)
    edge = comp & ~eroded
    if not edge.any():          # 连通域太小(全是边)时退回整域
        edge = comp

    # 3) 边缘里选最深(|R|最大)的一格
    depth_field = np.where(edge, np.abs(R), -np.inf)
    idx = int(np.argmax(depth_field))
    i, j = np.unravel_index(idx, depth_field.shape)
    if not np.isfinite(depth_field[i, j]):
        return None

    xx, zz = rast["xx"], rast["zz"]
    px, pz = float(xx[i, j]), float(zz[i, j])

    # 朝向：从坑质心指向下铲点(=坑外侧法向)，代表铲斗朝坑内挖的反向站位参考
    ci, cj = np.where(comp)
    ccx, ccz = float(xx[ci, cj].mean()), float(zz[ci, cj].mean())
    heading = float(np.degrees(np.arctan2(pz - ccz, px - ccx)))

    return {
        "xz": [px, pz],
        "cell": [int(i), int(j)],
        "zone": ZONE_NAMES[int(zone[i, j])],
        "depth": float(R[i, j]),               # 该点残差（负=坑深）
        "target_peak": float(-best_depth if use_dig else best_depth),
        "heading_deg": heading,
    }


# ── 顶层编排 ─────────────────────────────────────────────────────────────────

def analyze_terrain(pts_aligned: np.ndarray,
                    sem_labels: np.ndarray | None,
                    ground_mask: np.ndarray | None,
                    id_to_name: dict | None = None,
                    grid_res: int = 128,
                    top_percentile: float = 90.0,
                    tau: float | None = None,
                    tau_frac: float = 0.1,
                    min_area_frac: float = 0.002,
                    material_rules: dict | None = None) -> dict:
    """步骤2→4→5 的完整流水线，返回可 JSON 化的结果 dict。"""
    rules = material_rules or DEFAULT_MATERIAL_RULES
    id_to_name = id_to_name or {}

    rast = rasterize_bev(pts_aligned, sem_labels, ground_mask,
                         grid_res=grid_res, top_percentile=top_percentile)
    geom = extract_geometry(rast, tau=tau, tau_frac=tau_frac,
                            min_area_frac=min_area_frac)
    regions = confirm_semantics(geom, rast["S_mode"], id_to_name, rules)
    worksite = build_worksite_map(rast, geom, regions, id_to_name, rules)

    x_min, x_max, z_min, z_max = rast["bounds"]

    def _grid(a):
        # Emit JSON null (not NaN) for empty/invalid cells — strict JSON encoders
        # (FastAPI/starlette) reject NaN/Inf. Frontend treats null as “no data”.
        a = np.asarray(a, dtype=np.float64)
        return [[(float(v) if np.isfinite(v) else None) for v in row] for row in a]

    return {
        "status": "ok",
        "grid": {
            "res": grid_res,
            "x_min": x_min, "x_max": x_max,
            "z_min": z_min, "z_max": z_max,
            "cell_area": rast["cell_area"],
        },
        "params": {
            "top_percentile": top_percentile,
            "tau": geom["tau"],
            "tau_frac": tau_frac,
            "min_area_frac": min_area_frac,
        },
        "layers": {
            "H_top": _grid(rast["H_top"]),
            "H_ground": _grid(rast["H_ground"]),
            "R": _grid(geom["R"]),
            "S_mode": rast["S_mode"].astype(np.int32).tolist(),
            "slope": _grid(geom["slope"]),
            "roughness": _grid(geom["roughness"]),
            "mound_id": geom["mound_id"].astype(np.int32).tolist(),
            "pit_id": geom["pit_id"].astype(np.int32).tolist(),
            "zone_map": worksite["zone_map"].astype(np.int32).tolist(),
        },
        "regions": regions,
        "worksite": {
            "zones": worksite["zones"],
            "machines": worksite["machines"],
            "next_scoop": worksite["next_scoop"],
            "zone_names": ZONE_NAMES,
            "zone_colors": ZONE_COLORS,
        },
        "material_rules": rules,
    }


# ── 可视化导出（matplotlib，服务端出图，不依赖浏览器） ──────────────────────

def render_analysis_figure(result: dict, out_path: str, title: str = "",
                           scale_factor: float = 1.0):
    """把 analyze_terrain 的结果渲染成一张多面板 PNG 俯视图并保存到 out_path。

    面板：H_top（表面高程）| R（残差，标出土堆/坑区域）| 语义 S_mode | 坡度。

    scale_factor : 对齐帧 1 单位 = 多少米。所有高度量（H_top、R、坡度、峰值、
                   相对体积）按它换算为米/米³后再显示，与查看器一致。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap, BoundaryNorm

    sf = float(scale_factor) if scale_factor else 1.0
    g = result["grid"]
    L = result["layers"]
    extent = [g["x_min"], g["x_max"], g["z_min"], g["z_max"]]

    def arr(key):
        a = np.array([[np.nan if v is None else v for v in row] for row in L[key]],
                     dtype=np.float64)
        return a

    H_top = arr("H_top") * sf                 # → meters
    R = arr("R") * sf                          # → meters
    slope = arr("slope")                       # 无量纲梯度（比值），不乘 sf
    S_mode = np.array(L["S_mode"], dtype=np.int32)
    tau_m = result["params"]["tau"] * sf
    unit = "m" if sf != 1.0 else "units"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    stitle = title or "Semantic Terrain Analysis"
    fig.suptitle(f"{stitle}   [scale: 1 unit = {sf:g} m]", fontsize=14)

    im0 = axes[0].imshow(H_top, extent=extent, origin="lower", cmap="terrain")
    axes[0].set_title(f"H_top surface elevation ({unit})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    rmax = np.nanmax(np.abs(R)) or 1.0
    im1 = axes[1].imshow(R, extent=extent, origin="lower", cmap="RdBu_r",
                         vmin=-rmax, vmax=rmax)
    axes[1].set_title(f"Residual R ({unit})  (τ={tau_m:.3f} {unit})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    # 标注区域：料堆=绿, 排除=红, 坑=蓝；峰值/体积按 sf 换算
    for r in result["regions"]:
        cx, cz = r["centroid"]
        col = "#3b82f6" if r["kind"] == "pit" else ("#22c55e" if r["keep"] else "#ef4444")
        axes[1].plot(cx, cz, "o", color=col, ms=7, mec="k", mew=0.6)
        tag = f"pit{r['id']}" if r["kind"] == "pit" else (
            f"pile{r['id']}" if r["keep"] else f"excl{r['id']}")
        peak_m = r["peak"] * sf
        axes[1].annotate(f"{tag}\n{r['material']}\n{peak_m:+.2f}{unit}", (cx, cz),
                         fontsize=6.5, color=col, ha="center", va="bottom")

    # ── 语义面板：离散配色 + 灰底，让小面积类别也醒目 ────────────────────────
    id_name = {int(v): k for k, v in (result.get("sem_id_map") or {}).items()}
    present = sorted(int(v) for v in np.unique(S_mode) if v > 0)
    axes[2].imshow(np.zeros_like(S_mode), extent=extent, origin="lower",
                   cmap="Greys", vmin=0, vmax=1)   # 灰色背景铺满
    if present:
        base = plt.cm.tab10.colors
        colors = [base[(cid - 1) % 10] for cid in present]
        cmap = ListedColormap(colors)
        bounds = np.arange(len(present) + 1) - 0.5
        norm = BoundaryNorm(bounds, cmap.N)
        # 把实际 id 映射到 0..K-1 以配合离散 cmap
        remap = np.full(S_mode.shape, np.nan)
        for k, cid in enumerate(present):
            remap[S_mode == cid] = k
        masked = np.ma.masked_invalid(remap)
        axes[2].imshow(masked, extent=extent, origin="lower", cmap=cmap, norm=norm)
        handles = [Patch(color=colors[k],
                         label=f"{cid}:{id_name.get(cid, '?')} "
                               f"({int((S_mode == cid).sum())} cells)")
                   for k, cid in enumerate(present)]
        axes[2].legend(handles=handles, fontsize=7, loc="upper right",
                       framealpha=0.9)
    axes[2].set_title(f"Semantic S_mode ({len(present)} class)")

    im3 = axes[3].imshow(slope, extent=extent, origin="lower", cmap="magma")
    axes[3].set_title("Slope |∇H| (dimensionless)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("X"); ax.set_ylabel("Z")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def render_worksite_map(result: dict, out_path: str, title: str = "",
                        scale_factor: float = 1.0):
    """融合多层信息 → 一张作业地图（挖掘区/放料区/料堆/危险/障碍 + 下一铲点）。

    这是给挖掘任务看的“最终表示”：一眼看清场内有几种作业地形、下铲点在哪。
    左：作业分区着色 + 下一铲标记；右：图例 + 作业信息摘要。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap, BoundaryNorm

    sf = float(scale_factor) if scale_factor else 1.0
    unit = "m" if sf != 1.0 else "u"
    g = result["grid"]
    extent = [g["x_min"], g["x_max"], g["z_min"], g["z_max"]]
    zone = np.array(result["layers"]["zone_map"], dtype=np.int32)
    ws = result["worksite"]

    # 离散配色：把 ZONE_* 值映射到连续 index
    order = [ZONE_FLAT, ZONE_DIG, ZONE_DUMP, ZONE_PILE, ZONE_HAZARD, ZONE_OBSTACLE]
    present = [z for z in order if (zone == z).any()]
    colors = [ZONE_COLORS[z] for z in present]
    cmap = ListedColormap(colors)
    bounds = np.arange(len(present) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    remap = np.full(zone.shape, np.nan)
    for k, z in enumerate(present):
        remap[zone == z] = k

    fig, (axm, axl) = plt.subplots(1, 2, figsize=(13, 6.2),
                                   gridspec_kw={"width_ratios": [3, 1]})
    stitle = title or "Worksite Map"
    fig.suptitle(f"{stitle}   [scale: 1 unit = {sf:g} m]", fontsize=14)

    axm.imshow(np.ma.masked_invalid(remap), extent=extent, origin="lower",
               cmap=cmap, norm=norm, interpolation="nearest")
    axm.set_title("Fused work-zone map")
    axm.set_xlabel("X"); axm.set_ylabel("Z")

    # 下一铲标记 + 朝向箭头
    ns = ws.get("next_scoop")
    if ns:
        px, pz = ns["xz"]
        axm.plot(px, pz, marker="*", color="#111", ms=22, mec="w", mew=1.5, zorder=5)
        axm.annotate("NEXT SCOOP", (px, pz), color="#111", fontsize=10, weight="bold",
                     ha="center", va="top", xytext=(0, -12), textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="w", ec="#111", alpha=0.85))
        if ns.get("heading_deg") is not None:
            import math
            L = (g["x_max"] - g["x_min"]) * 0.12
            hx = px + L * math.cos(math.radians(ns["heading_deg"]))
            hz = pz + L * math.sin(math.radians(ns["heading_deg"]))
            axm.annotate("", xy=(hx, hz), xytext=(px, pz),
                         arrowprops=dict(arrowstyle="-|>", color="#111", lw=2))

    # 机械/障碍物标注：明确标出“挖掘机在这”（类名 + 符号）
    for mc in ws.get("machines", []):
        mx, mz = mc["centroid"]
        axm.plot(mx, mz, marker="s", color="#111", ms=13, mec="#ffd400", mew=2, zorder=6)
        axm.annotate(f"{mc['label']}", (mx, mz), color="#111", fontsize=10,
                     weight="bold", ha="center", va="bottom",
                     xytext=(0, 10), textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="#ffd400", ec="#111", alpha=0.9))

    # 右侧图例 + 摘要
    axl.axis("off")
    handles = [Patch(color=ZONE_COLORS[z], label=ZONE_NAMES[z]) for z in present]
    axl.legend(handles=handles, loc="upper left", fontsize=11, title="Zone types",
               framealpha=1.0)
    lines = ["", "Zone areas:"]
    for zinfo in sorted(ws["zones"], key=lambda d: -d["area"]):
        lines.append(f"  {zinfo['name']:9s} {zinfo['area']*sf*sf:8.2f} {unit}²")
    if ws.get("machines"):
        lines += ["", "Machines / obstacles:"]
        for mc in ws["machines"]:
            lines.append(f"  {mc['label']}  ({mc['area']*sf*sf:.1f} {unit}²)")
    if ns:
        lines += ["", "Next scoop (deepest-pit edge):",
                  f"  pos   = ({ns['xz'][0]:.2f}, {ns['xz'][1]:.2f})",
                  f"  in    = {ns['zone']}",
                  f"  depth = {ns['depth']*sf:.2f} {unit}",
                  f"  pit peak = {ns['target_peak']*sf:.2f} {unit}"]
        if ns.get("heading_deg") is not None:
            lines.append(f"  heading = {ns['heading_deg']:.0f}°")
    else:
        lines += ["", "Next scoop: none", "  (no diggable zone)"]
    axl.text(0.02, 0.55, "\n".join(lines), fontsize=10, family="monospace",
             va="top", transform=axl.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def render_worksite_bev(result: dict, out_path: str, title: str = "",
                        scale_factor: float = 1.0):
    """自动驾驶 BEV 风格的作业地图：深色背景 + 明显网格线 + 作业分区 +
    机械框 + 下一铲点/朝向。方形画布、等比坐标，metric 刻度。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle, FancyArrow
    from matplotlib.colors import ListedColormap, BoundaryNorm

    sf = float(scale_factor) if scale_factor else 1.0
    unit = "m" if sf != 1.0 else "u"
    g = result["grid"]
    res = g["res"]
    # metric extent（把栅格坐标换算成米，BEV 以场景中心为原点）
    cx = (g["x_min"] + g["x_max"]) / 2.0
    cz = (g["z_min"] + g["z_max"]) / 2.0
    ex0 = (g["x_min"] - cx) * sf; ex1 = (g["x_max"] - cx) * sf
    ez0 = (g["z_min"] - cz) * sf; ez1 = (g["z_max"] - cz) * sf
    extent = [ex0, ex1, ez0, ez1]

    def to_m(px, pz):
        return (px - cx) * sf, (pz - cz) * sf

    zone = np.array(result["layers"]["zone_map"], dtype=np.int32)
    ws = result["worksite"]

    order = [ZONE_FLAT, ZONE_DIG, ZONE_DUMP, ZONE_PILE, ZONE_HAZARD, ZONE_OBSTACLE]
    present = [z for z in order if (zone == z).any()]
    colors = [ZONE_COLORS[z] for z in present]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(present) + 1) - 0.5, cmap.N)
    remap = np.full(zone.shape, np.nan)
    for k, z in enumerate(present):
        remap[zone == z] = k

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    stitle = title or "Worksite BEV"
    fig.suptitle(f"{stitle}   [1 unit = {sf:g} m]", fontsize=14, color="#e6edf3")

    ax.imshow(np.ma.masked_invalid(remap), extent=extent, origin="lower",
              cmap=cmap, norm=norm, interpolation="nearest", alpha=0.92, zorder=1)

    # BEV 网格线：与作业分区**同一颗粒度**——每条线落在真实栅格 cell 边界上。
    # res 个格子 → res+1 条边界线；用 imshow 的 extent 等分即可精确对齐。
    xb = np.linspace(ex0, ex1, res + 1)   # 列(X)方向 cell 边界
    zb = np.linspace(ez0, ez1, res + 1)   # 行(Z)方向 cell 边界
    # 网格过密(128格)会糊成一片，故按 cell 数自适应抽稀：目标约 32 条主线
    stride = max(1, int(round(res / 32)))
    step = (ex1 - ex0) / res * stride     # 一条主网格线间距（米），供标注尺寸参考
    for gx in xb[::stride]:
        ax.axvline(gx, color="#30363d", lw=0.6, zorder=2)
    for gz in zb[::stride]:
        ax.axhline(gz, color="#30363d", lw=0.6, zorder=2)
    # 中心十字（场景原点参考）
    ax.axhline(0, color="#58a6ff", lw=1.0, alpha=0.5, zorder=2)
    ax.axvline(0, color="#58a6ff", lw=1.0, alpha=0.5, zorder=2)

    # 机械：黄框 + 类名
    for mc in ws.get("machines", []):
        mx, mz = to_m(*mc["centroid"])
        side = max(np.sqrt(mc["area"]) * sf, step * 0.3)
        ax.add_patch(Rectangle((mx - side/2, mz - side/2), side, side,
                     fill=False, ec="#ffd400", lw=2.2, zorder=6))
        ax.text(mx, mz + side/2 + step*0.15, mc["label"], color="#ffd400",
                fontsize=11, weight="bold", ha="center", va="bottom", zorder=7)

    # 下一铲点 + 朝向
    ns = ws.get("next_scoop")
    if ns:
        px, pz = to_m(*ns["xz"])
        ax.plot(px, pz, marker="*", color="#f0f6fc", ms=26, mec="#111", mew=1.5, zorder=8)
        ax.text(px, pz - step*0.2, "NEXT SCOOP", color="#f0f6fc", fontsize=11,
                weight="bold", ha="center", va="top", zorder=8,
                bbox=dict(boxstyle="round", fc="#161b22", ec="#f0f6fc", alpha=0.9))
        if ns.get("heading_deg") is not None:
            import math
            L = step * 1.2
            ax.add_patch(FancyArrow(px, pz,
                         L*math.cos(math.radians(ns["heading_deg"])),
                         L*math.sin(math.radians(ns["heading_deg"])),
                         width=step*0.04, head_width=step*0.18,
                         color="#f0f6fc", zorder=8, length_includes_head=True))

    ax.set_xlim(ex0, ex1); ax.set_ylim(ez0, ez1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"X ({unit})", color="#e6edf3")
    ax.set_ylabel(f"Z ({unit})", color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    for s in ax.spines.values():
        s.set_color("#30363d")

    handles = [Patch(color=ZONE_COLORS[z], label=ZONE_NAMES[z]) for z in present]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=10,
                    title="Zones", framealpha=0.9, facecolor="#161b22",
                    edgecolor="#30363d", labelcolor="#e6edf3")
    leg.get_title().set_color("#e6edf3")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _nice_step(x):
    """把任意步长归到 1/2/5×10^n 的“好看”刻度。"""
    if x <= 0:
        return 1.0
    import math
    e = math.floor(math.log10(x))
    base = x / (10 ** e)
    nice = 1 if base < 1.5 else (2 if base < 3.5 else (5 if base < 7.5 else 10))
    return nice * (10 ** e)


# ── 自测：读某个 session 的 predictions.npz，对齐后跑通并打印摘要 ─────────────
if __name__ == "__main__":
    import argparse
    from gravity_alignment import estimate_gravity, apply_alignment_to_points

    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="path to predictions.npz")
    ap.add_argument("--conf", type=float, default=50.0)
    ap.add_argument("--tau", type=float, default=None,
                    help="absolute residual threshold; omit for adaptive")
    ap.add_argument("--grid", type=int, default=128)
    args = ap.parse_args()

    d = np.load(args.npz)
    pred = {k: np.array(d[k]) for k in d.files}
    pts = pred.get("world_points_from_depth")
    conf = pred.get("depth_conf")
    sem = pred.get("semantic_masks")
    S, H, W, _ = pts.shape

    gmask_3d = (sem == 1) if sem is not None else None
    grav = estimate_gravity(extrinsic=pred["extrinsic"], world_points=pts,
                            ground_mask=gmask_3d, conf=conf,
                            conf_thres=args.conf / 100.0)

    pts_flat = pts.reshape(-1, 3)
    conf_flat = conf.reshape(-1).astype(np.float32)
    keep = np.isfinite(pts_flat).all(axis=1) & (
        conf_flat >= (args.conf / 100.0) * conf_flat.max())
    pts_aligned = apply_alignment_to_points(pts_flat[keep], grav.R_align)

    sem_flat = sem.reshape(-1)[keep] if sem is not None else None
    ground_flat = (sem.reshape(-1)[keep] == 1) if sem is not None else None

    res = analyze_terrain(pts_aligned, sem_flat, ground_flat,
                          id_to_name={}, grid_res=args.grid, tau=args.tau)
    print(f"gravity={grav.source}  points={pts_aligned.shape[0]}")
    print(f"effective tau={res['params']['tau']:.4f}  "
          f"regions={len(res['regions'])}")
    for r in res["regions"]:
        print(f"  {r['kind']:5s} id={r['id']} area={r['area']:.3f} "
              f"vol={r['rel_volume']:.3f} peak={r['peak']:+.3f} "
              f"mat={r['material']} cat={r['category']} keep={r['keep']}")
