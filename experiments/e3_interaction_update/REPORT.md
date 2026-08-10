# E3 — Interaction-Triggered Incremental Update（交互触发的增量更新）

对应 ICRA 概念文档 **E3 / C3** 与 `SCENE_GRAPH_PLAN.md` **机制 5（动作条件化的增量维护）**。

## 结论

在**真实 VGGT 重建的地形**上，用挖掘机-地形的**交互线索**（dig 事件）定位受影响 ROI、只重估该
区域，能在**保持精度**的同时把需要重新栅格化的**真实 VGGT 点数**降到全量重建的约 **28%**：

| 策略 | 处理真实点数 | 处理格数 | 最终 DEM RMSE | 最终 DEM MAE | cut 体积误差 | change recall | map consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| A · 周期性全量重建 | 5.72 M | 46,080 | 13.5 mm | 5.6 mm | 45.8% | 0.150 | 0.962 |
| B · 纯几何 change-only | 5.72 M | 46,080 | 14.4 mm | 6.4 mm | 40.4% | 0.247 | 0.886 |
| **C · 交互触发 ROI（本方案）** | **1.62 M** | **11,229** | **14.2 mm** | 6.4 mm | 40.6% | 0.174 | **0.976** |

要点：

- **代价用真实 VGGT 点数衡量**：重估一个 DEM 格子 = 重新栅格化落在该格的真实 VGGT 点
  （`terrain_analysis.rasterize_bev` 的 `count`）。全量/change-only 每次都要重跑全部 **114 万**真实点、
  全程累计 5.72 M；ROI 只重跑交互区内的点，累计 1.62 M（**28%**）。
- **精度基本持平**：C 的最终 RMSE（14.2 mm）与全量重建 A（13.5 mm）相当。
- **B 省不下计算**：change-only 为了发现变化**仍要重建整幅 DEM**（点数与 A 相同），且在全网格上受
  垂直压缩噪声干扰，伪更新更多（consistency 0.886，最低）。ROI 的 consistency 最高（0.976）。

![代价-精度权衡](visualizations/cost_accuracy_tradeoff.png)

## 什么是真实的、什么是注入的（诚实边界）

这一版把 E3 落在**真实 VGGT 输出**上，而不是合成栅格：

**真实：**
- **地形基底**：demo session `session_20260629_100116_092814` 的真实 `world_points_from_depth`
  地面点，重力对齐后经主程序 `terrain_analysis.rasterize_bev`（H_top P70）栅格化，得到 96×96、
  约 1.84 cm/格的真实 VGGT DEM（6587 个有效格、114 万真实点）。策略融合的表面、代价的点数权重，
  **都来自这次真实重建**。
- **交互线索**：铲斗离地高度/状态来自 `experiments/arm_motion_state/motion_state.json`；铲斗每帧
  三维锚点由该 session 的 `world_points` 恢复并映射到 DEM 格。dig 事件从这条真实信号检出。
- **观测噪声**：从 `vertical_fidelity_results.json` 标定的垂直压缩比中位数 **2.32**（VGGT 只重建出
  约 0.43× 真实起伏）注入观测，直面 VGGT 招牌缺陷。

**注入（唯一的诚实缺口）：**
- **挖掘动作本身**。我们没有「地形在 VGGT 下真实改变」的开挖视频（`SCENE_GRAPH_PLAN.md` §7 数据
  缺口）。因此在真实基底上、真实铲斗锚点处注入运动学楔形挖方并体积守恒地堆到弃土区，使每一步真值
  DEM 完全已知、cut/fill 误差可测。

> 换句话说：**几何、交互线索、噪声、代价单位都真**，只有「被挖」这件事是注入的。

![真实交互线索时间线](visualizations/interaction_timeline.png)

## 三种策略（同一事件序列回放）

共享同一次真实全场栅格化作 bootstrap，然后在同一 dig 事件序列上回放，区别只在**每个事件重新栅格化
哪些真实点**：

- **A · full**：每个事件重跑全部真实点、重建整幅 DEM。
- **B · change**：重跑全部真实点，只提交与当前地图差异超过 τ=1.5 cm 的格子
  （`streaming/change_detection` 门控思路）。
- **C · roi（本方案）**：dig 事件在铲斗真实锚点圈 ROI（挖点 + 卸土点，均由**同一条真实铲斗轨迹**
  定位），只重跑 ROI 内的真实点。

融合复用 `streaming/global_dem` 的加权平均 + fast-follow 时间衰减（`FusionConfig.decay`），使一次
挖掘造成的高度下降能在一两个 pass 内被跟上。

![机制5 因果定位](visualizations/roi_localization.png)

上图：每个 dig 事件真实高度变化 Δh（蓝=挖低、红=堆高）恰落在铲斗锚点（实线 ROI = cut）与卸土点
（虚线 ROI = fill）——这正是 ROI 只需重估这些区域的依据，无需靠图像相似度去猜变化位置。

![真实基底 + 注入挖掘的演化与各策略最终融合](visualizations/dem_evolution.png)

## 指标定义（对标 `SCENE_GRAPH_PLAN.md` §10 / E3 表 metrics）

- **processed_points / processed_cells**：累计重新栅格化的真实 VGGT 点数（与格数），正比于计算量
  （Update Latency / Processed Area 的真实代理）。
- **dem_mae_m / dem_rmse_m**：融合 DEM 相对真值 DEM 的高度误差（Δh Error）。
- **cut_volume_error_frac**：|估计净挖方 − 真值净挖方| / 真值（ΔV Error）。
- **change_recall**：真值变化格中被更新到的比例（Change Recall）。
- **map_consistency**：1 − 伪更新格 / 总更新格；高 = 很少在没变化处乱改地图（Map Consistency）。

## 复现

```
~/miniconda3/envs/vggt/bin/python experiments/e3_interaction_update/build_real_dem.py       # 真实 VGGT 基底
~/miniconda3/envs/vggt/bin/python experiments/e3_interaction_update/run_e3.py                # 回放 + 指标
~/miniconda3/envs/vggt/bin/python experiments/e3_interaction_update/make_e3_visualizations.py
```

产物：

- `build_real_dem.py` → `e3_real_base.npz`：真实 VGGT 基底 DEM、逐格真实点数、真实铲斗锚点格。
- `run_e3.py` → `e3_results.json`、`e3_dems.npz`：事件检出 + 真实基底上注入挖掘 + 三策略回放 + 指标。
- `make_e3_visualizations.py` → `visualizations/`：四张图 + `viz_manifest.json`。

复用的主程序模块：`terrain_analysis.rasterize_bev`（真实基底 DEM + 逐格点数）、
`streaming/global_dem.py`（融合衰减/权重与变化检测约定）、`streaming/change_detection.py`（变化门控）。

## 局限

1. **挖掘动作是注入的，不是真实开挖 GT。** 地形基底、交互线索、噪声、代价单位都真，但「被挖」这件事
   由运动学楔形 + 体积守恒模拟。真实端到端验证仍需符合 `SCENE_GRAPH_PLAN.md` §7 采集规格的开挖视频
   （挖机连续作业、地形肉眼可见改变）——那是唯一能把「注入」也去掉的路径。
2. **demo session 是 14 帧短片、地形近乎平坦**（真实高度 p2–p98 仅约 ±7 mm，铲斗水平位移也很小）。
   基底真实但起伏小，注入挖掘的绝对尺度偏小；cut 体积误差偏高（~40%）主要来自垂直压缩噪声
   与小起伏之比，此处只用于**跨策略相对比较**，非绝对精度。
3. **change recall 绝对值偏低**（<0.25）同样是压缩噪声 + 单次融合权重所致，只作相对比较；提高需多视角
   融合与置信度传播（接机制 6）。
4. dig/dump ROI 半径与 τ 为固定值；`SCENE_GRAPH_PLAN.md` §8 风险 7 已列出对触发阈值做敏感度分析的
   后续项。
