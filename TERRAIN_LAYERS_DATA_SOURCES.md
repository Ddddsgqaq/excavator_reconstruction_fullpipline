# 语义地形理解 — 图层数据来源

本文档记录 `/terrain_analysis` 输出的各 BEV 栅格图层与区域字段的**数据来源与计算方式**，
便于后续维护与写论文时引用。代码在 `terrain_analysis.py`，服务端点在 `vggt_service.py`。

## 总体数据流

```
VGGT predictions.npz                YOLOe semantic_masks(.npz 或 predictions 内)
   │  world_points_from_depth (S,H,W,3)      │  逐像素语义 id (S,H,W)
   │  depth_conf              (S,H,W)        │
   └──────────────┬──────────────────────────┘
                  ▼  vggt_service._load_aligned_points_and_semantics()
     ① 置信度过滤 keep = depth_conf ≥ P(conf_thres)  且  point 有限
     ② 可选语义编辑 (semantic_filter_ids / mode，与点云编辑标签页一致)
     ③ 重力对齐：estimate_gravity → R_align；pts_aligned = pts_world_kept @ R_align.T
        对齐后 Y=up（高程），(X,Z)=水平面
                  ▼  terrain_analysis.analyze_terrain()
     ④ rasterize_bev  → H_top / H_ground / S_mode / count
     ⑤ extract_geometry → R / slope / roughness / mound_id / pit_id / 区域几何量
     ⑥ confirm_semantics → 每个区域的 material / category / keep
```

对齐帧的点 `pts_aligned` 与逐点语义 `sem_kept`、地面掩码 `ground_kept` 都经过**同一个
keep 掩码**过滤，因此三者严格同序对齐。

## 栅格化基础

- **网格**：在对齐帧的 (X,Z) 平面上按点云范围（+2% padding）划 `grid_res×grid_res`（默认 128）。
  行 index i 对应 Z，列 index j 对应 X。
- **落格**：每个点按 (X,Z) 归到一个格子（`_cell_index`，越界裁剪）。
- **cell_area** = (x跨度/res)·(z跨度/res)，用于体积积分。

## 各图层来源

| 图层 | 来源 | 计算方式 |
|------|------|----------|
| **H_top** | VGGT 点高度 Y（对齐帧） | 每格落入点的 **top_percentile 分位高度**（默认 90%），抗噪的“表面顶”。空格=null |
| **H_ground** | VGGT ground 点插值 | 取 YOLOe 地面掩码点（`ground_mask`，即 semantic id==1）；不足 50 个时**回退**为全局低 20% 高度点。对其 (X,Z)→Y 做 `griddata` linear 插值 + nearest 回退，全网格有值 |
| **R（残差）** | H_top − H_ground | 每格表面高相对参考地面的高差。>+τ=土堆，<−τ=坑。空 H_top 处=null |
| **S_mode（语义）** | YOLOe 逐点语义 id | 每格落入点的**语义众数**（忽略 id=0 背景）。0=无标签 |
| **count** | 落格点数 | `np.bincount`，每格点数（用于诊断，未在图中展示） |
| **slope（坡度）** | H_top | 先用 H_ground 填洞得 H_fill，再 `np.gradient` 求 (∂/∂x, ∂/∂z)，取幅值 `hypot` |
| **roughness（粗糙度）** | H_top | H_fill 上 3×3 窗的局部方差 = 局部均方 − 局部均值²（`ndimage.uniform_filter`） |
| **mound_id / pit_id** | R | R>+τ / R<−τ 的布尔图做 4-连通标记（`ndimage.label`），丢弃 < `min_area_frac`·总格数 的碎片，重新编号 1..K |

**τ（起伏阈值）**：默认自适应 `τ = tau_frac · (H_top 的 p2–p98 值域)`，tau_frac 默认 0.1。
因 VGGT 低估垂直起伏、绝对阈值跨场景不可移植，故用相对值。也可传绝对 `tau` 覆盖。

## 区域（regions）字段来源

每个连通域（土堆或坑）汇总为一条记录：

| 字段 | 来源 | 说明 |
|------|------|------|
| kind | mound_id/pit_id | "mound" 或 "pit" |
| id | 连通域编号 | 同图层标签值 |
| cells / area | 连通域格数 | area = cells · cell_area |
| rel_volume | R + cell_area | **Σ\|R\|·cell_area**，相对体积（无量纲可比，非绝对方量） |
| peak | R | 域内 R 的极值（土堆取 max，坑取 min） |
| centroid | 网格坐标 xx/zz | 域内格中心的 (X,Z) 均值 |
| sem_id | S_mode | 域内语义众数 |
| material | sem_id → 类名 | 由 YOLOe run `meta.json` 的 `semantic_id_map` 反查（`_resolve_sem_id_map`） |
| category | material 关键字规则 | excavatable / obstacle / unknown（`DEFAULT_MATERIAL_RULES`，可覆盖） |
| keep | category + kind | 土堆：keep = (category==excavatable)；坑：keep=True |

## 导出的 PNG 面板

`/terrain_analysis_figure` → `render_analysis_figure`（matplotlib，服务端出图）。
四面板俯视：**H_top（terrain 色）| R（RdBu，叠区域质心+标注）| S_mode（tab10+图例）| slope（magma）**。
同时保存到 `<session>/terrain_analysis.png`。

## 已知边界 / 注意

- **NaN→null**：图层空格返回 JSON `null`（FastAPI 拒绝 NaN）；前端与出图都按“无数据”处理。
- **绝对方量未标定**：rel_volume 无量纲，需尺度标定才能得真实 m³。
- **材质依赖 YOLOe 覆盖**：某类若未被 YOLOe 分割，其上的土堆会是 material=unlabeled/category=unknown。
- **step6 任务约束（源/目标区域筛选）未实现**：keep/category 字段是接入规则引擎的预留钩子。
