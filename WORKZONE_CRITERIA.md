# Work-Zone 作业地形判定标准

本文档记录 `build_worksite_map()`（`terrain_analysis.py`）如何把多层地形信息融合成
一张作业地图（work-zone map）。核心思想：**2D 语义分割难区分土的种类，所以不靠语义
细分土质，而是用「几何残差 R + 坡度 + 语义排除」推导每格的"作业语义"**——即这块地对
挖掘任务意味着什么。

## 输入（多层栅格，均在重力对齐帧、同一 BEV 网格 res×res）

| 变量 | 含义 | 来源 |
|------|------|------|
| `R` | 残差 = H_top − H_ground（每格表面高 − 参考地面高） | `extract_geometry` |
| `slope` | 坡度幅值 \|∇H_top\| | `extract_geometry` |
| `S_mode` | 每格语义众数（YOLOe 类 id） | `rasterize_bev` |
| `tau (τ)` | 起伏阈值，默认自适应 = `tau_frac · (H_top 的 p2–p98 值域)`，`tau_frac=0.1` | `extract_geometry` |

> τ 为什么用相对值：VGGT 低估垂直起伏，绝对高度阈值跨场景不可移植（见 memory
> `vertical-compression-rootcause-refs`）。

## 六种作业类型（zone_map 取值）

| 编码 | 名称 | 颜色 | 含义 |
|------|------|------|------|
| 0 | flat | 浅灰 | 可通行平地 |
| 1 | dig | 橙 | 挖掘区（坑/沟，可下铲） |
| 2 | dump | 蓝 | 放料区（大片平坦、远离障碍） |
| 3 | pile | 绿 | 料堆（可挖凸起） |
| 4 | hazard | 红 | 危险（陡坎） |
| 5 | obstacle | 灰 | 障碍（挖机/人/车…） |
| -1 | (empty) | — | 无数据（空格） |

## 判定规则（按顺序执行，后面的覆盖前面的）

**关键：规则有优先级，后写的图层覆盖先写的。最终每格类型 = 最后一个命中它的规则。**

1. **障碍 obstacle（语义）**
   先算：`S_mode` 命中 obstacle 材质规则的类 id（如 Excavator/person/vehicle，见
   `DEFAULT_MATERIAL_RULES`）→ `obstacle_mask`。（此步只是算掩码，第 4 步才铺色。）

2. **按残差 R 三分（基础层）**
   - `|R| ≤ τ` → **flat**（平地）
   - `R > +τ` → **pile**（凸起 = 料堆）
   - `R < −τ` → **dig**（凹陷 = 坑/沟 = 挖掘区）

3. **危险 hazard（坡度）**
   `slope > slope_hazard`（默认 0.6）→ **hazard**，覆盖第 2 步。代表陡坎，不宜作业。

4. **障碍覆盖**
   `obstacle_mask` 的格子 → **obstacle**，覆盖前面所有。
   （所以凸起若语义=挖机，会从 pile 变成 obstacle，被排除在可挖料之外。）

5. **放料区 dump（在平地里选）**
   在 flat 格子中：
   - 若场内有障碍，先对障碍做距离变换，剔除离障碍太近的平地（距离 ≤ `res·0.05`）；
   - 对剩余平地做连通域标记，选**面积最大的连通块**；
   - 若其面积 ≥ `dump_min_area_frac`（默认 0.01，即 1% 网格）→ 标为 **dump**。
   只选一块（最大的），代表推荐的卸料/堆放场地。

> 注：pile 的材质门控隐含在第 4 步——凸起若语义=障碍类已被覆盖为 obstacle；
> 语义=可挖或未知的凸起保持 pile。当前**未**对 pile 做进一步材质细分。

## 可调参数

| 参数 | 默认 | 作用 |
|------|------|------|
| `tau_frac` | 0.1 | 起伏阈值占值域比例。调大→dig/pile 收缩到更明显的凹凸 |
| `slope_hazard` | 0.6 | 坡度超此值判危险 |
| `dump_min_area_frac` | 0.01 | 放料区最小面积占比 |

## 下一铲点（next_scoop）判定

**逻辑：在最深的坑的边缘下铲。**（`_pick_next_scoop`）
1. 在 dig 连通域里挑**最深的坑**（域内 R 最负）；无 dig 才退回最高 pile。
2. 取该坑的**内边缘**（掩码腐蚀一圈后的边界环）。
3. 边缘格里选 **\|R\| 最大（最深）**的一格作为下铲点。
4. 朝向 heading = 从坑质心指向该点（坑外侧法向，代表铲斗朝坑内挖的站位参考）。

返回：`{xz, cell, zone, depth, target_peak, heading_deg}`。

## 已知局限（当前样例）

- **dig 区偏大**：VGGT 垂直压缩使 τ 很小，很多轻微低洼都算 dig。想聚焦可调大 `tau_frac`。
- **hazard 依赖坡度阈值**：坡度是无量纲梯度，`slope_hazard` 需按场景标定；当前样例几乎无 hazard。
- **dump 只选一块**：仅取最大连通平地，未考虑运距/朝向等作业约束（属未实现的 step6 任务约束）。
- **材质细分未做**：pile 不区分土/砂/砾，仅靠语义排除障碍。

## 相关文件

- 判定实现：`terrain_analysis.py` → `build_worksite_map` / `_pick_next_scoop`
- 材质规则：`terrain_analysis.py` → `DEFAULT_MATERIAL_RULES` / `classify_material`
- 图层数据来源：`TERRAIN_LAYERS_DATA_SOURCES.md`
- 服务端点：`/terrain_analysis`（JSON）、`/terrain_analysis_figure`（PNG: worksite/bev/diagnostic）
