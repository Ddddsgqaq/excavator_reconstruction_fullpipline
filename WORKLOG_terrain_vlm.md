# 语义地形理解 & VLM 决策 —— 工作记录

> 本文件记录一次完整开发的所有产出：从「语义×高程融合」到「VLM 读地形出决策」。
> 日期跨度：2026-07-13 ~ 2026-07-14。相关代码均在 `WS/vggt_yoloe/`。

## 总目标

把项目里**分离的语义（YOLOe）与高程（VGGT DEM）** 融合成一个「能理解地形」的系统，
并逐步升级为「面向挖掘任务的作业地图 + 下铲决策」，最终验证「把地形表示喂给 VLM
直接输出挖掘决策」的可行性。

分阶段（由浅入深）：
1. 语义×几何融合的地形理解（栅格化 + 几何提取 + 材质确认）
2. 服务端点 + 查看器叠加
3. 多层信息融合成一张「作业地图」（挖掘/放料/料堆/危险/障碍 + 下铲点）
4. PNG 导出（普通 / BEV 自动驾驶风格）
5. 粗网格 DEM 重采样（平滑细节、突出大地形）
6. **VLM 读地形出结构化决策**（本次高潮）+ HTML 可视化报告

---

## 系统数据流总览

```
VGGT predictions.npz            YOLOe semantic_masks
  · world_points_from_depth       · 逐像素语义 id (S,H,W)
  · depth_conf                     · yoloe_runs/*/meta.json (类名→id)
        └──────────┬───────────────────┘
                   ▼  _load_aligned_points_and_semantics (vggt_service.py)
   ① 置信度过滤  ② 可选语义编辑  ③ 重力对齐 → pts_aligned (Y=up)
                   ▼  terrain_analysis.analyze_terrain
   ④ rasterize_bev   → H_top / H_ground / S_mode / count
   ⑤ extract_geometry→ R=H_top-H_ground / slope / roughness / 连通域 / 体积
   ⑥ confirm_semantics→ 每区 material/category/keep（关键字规则）
   ⑦ build_worksite_map→ zone_map(6类) + machines + next_scoop
                   ▼
   PNG(worksite/bev/diagnostic) · JSON · three.js 叠加 · VLM 决策
```

---

## 新增 / 修改的文件

### 核心模块
| 文件 | 内容 |
|------|------|
| `terrain_analysis.py`（新） | 纯计算核心。`rasterize_bev`（栅格化）、`extract_geometry`（残差/坡度/粗糙度/连通域/体积）、`confirm_semantics`（材质规则）、`build_worksite_map`（六类作业分区 + 下铲点）、`analyze_terrain`（编排）。渲染函数：`render_analysis_figure`（四面板诊断图）、`render_worksite_map`（作业地图）、`render_worksite_bev`（BEV 网格风格）。材质规则表 `DEFAULT_MATERIAL_RULES` |
| `vggt_service.py`（改） | 新增 `TerrainAnalysisRequest`、共用 helper `_load_aligned_points_and_semantics`、`_resolve_sem_id_map`（读 yoloe meta 类名）、`POST /terrain_analysis`（JSON）、`POST /terrain_analysis_figure`（PNG，`figure_mode` = worksite/bev/diagnostic）|
| `elevation_viewer.html`（改） | 「语义地形理解」面板：运行分析、图层切换（作业分区/残差/语义/坡度/粗糙度）、三个导出按钮（作业地图/BEV/诊断）、three.js 把 zone_map 投射到地形 + 挖机框 + 下铲星标/朝向 |

### 独立测试脚本
| 文件 | 内容 |
|------|------|
| `test_coarse_dem.py`（新） | DEM 粗网格重采样测试：块聚合（中位/均值）平滑细节、突出大地形，出「前/后/残差」三联对比图。参数 `--factor/--agg/--scale` |
| `test_terrain_vlm.py`（新） | **VLM 读地形出决策**：analyze → BEV 图 → 图+结构化上下文喂 VLM → 结构化决策 JSON + 决策叠加图。支持多 session 批量，产出 `manifest.json`。走 OpenAI 兼容接口（通义千问 qwen3-vl-plus）|
| `make_vlm_report.py`（新） | 读 manifest 生成自包含 HTML 可视化报告（图片内嵌 base64）|

### 文档
| 文件 | 内容 |
|------|------|
| `TERRAIN_LAYERS_DATA_SOURCES.md` | 各 BEV 图层/区域字段的数据来源与算法 |
| `WORKZONE_CRITERIA.md` | work-zone 六类判定标准 + 下铲点逻辑 + 可调参数 |
| `WORKLOG_terrain_vlm.md` | 本文件（总记录）|

### 产物目录
- `coarse_dem_tests/` — 粗采样对比图（factor 2/4/8/16/24）
- `vlm_report/` — VLM 决策报告：`report.html` + 各 session 的 bev/overlay/decision.json

---

## 关键设计决策

1. **τ（起伏阈值）用相对值，不用绝对值**
   默认 `τ = tau_frac · (H_top 的 p2–p98 值域)`，`tau_frac=0.1`。因为 VGGT 低估垂直
   起伏，测试场景整体垂直只 ~0.12，绝对阈值跨场景不可移植。

2. **不靠语义细分土质，靠「几何+坡度+语义排除」推导作业语义**
   2D 语义分割难区分土的种类，所以 work-zone 判定主要用残差 R + 坡度，语义只用来
   **排除障碍**（挖机/人/车）。六类：flat/dig/dump/pile/hazard/obstacle，规则有优先级
   （后覆盖前）：残差三分 → 坡度判危险 → 语义覆盖障碍 → 平地选最大块为放料区。

3. **下铲点 = 最深坑的边缘**（用户明确要求，替换了早期"越远越好"的评分）
   在最深的 dig 连通域取内边缘、选 |R| 最深的一格。符合沿坡逐层下挖的作业习惯。

4. **BEV 网格线对齐真实 cell 边界**（用户要求）
   不再随意加网格；网格线落在 128 格的真实边界上，按 stride 抽稀（默认每 4 格一线）。

5. **材质关键字规则表可覆盖**
   `DEFAULT_MATERIAL_RULES`：excavatable（soil/dirt/gravel/sand/mound/pile…）、
   obstacle（excavator/person/vehicle/rock…）。子串匹配，可入参覆盖。

6. **粗采样类比 patch token**（用户洞察）
   factor=4 → 32×32 粗 DEM，等价于一张「地形 token 网格」，天然适合喂 transformer/VLM。

7. **VLM 决策：几何管数值，VLM 管语义/策略**
   把 BEV 图 + 精简结构化上下文（不含大数组，省 token）喂 VLM，返回结构化 JSON
   （terrain_summary / zones_readout / risks / work_order / next_action / confidence）。

---

## 验证结果

- **地形分析端到端**：`/terrain_analysis` HTTP 200，1.8s，输出 zone_map + regions + machines
  + next_scoop；Excavator 正确识别为 obstacle 并从可挖料中排除。
- **粗采样**：factor=4（32²）在测试场景是平滑/保真的较好平衡点；factor≥8 会把挖机等
  小结构一起抹平（残差图凸显），印证粗 DEM 只宜看宏观趋势。
- **VLM 决策**（qwen3-vl-plus，4 个 session）：
  - 能读懂 BEV 图，准确说出分区/挖机/下铲点，并给出作业顺序与风险。
  - 会因场景不同给不同动作（dig / move），说明是真读图判断而非套模板。
  - **定量数值不可靠**：曾把下铲坐标算到场景边界外（如 -20.3,14.9 越界）。
    → 结论：**精确坐标用几何模块，VLM 用于高层语义/排序/风险**。

---

## 已知局限 & 后续方向

**局限**
- dig 区偏大（VGGT 垂直压缩 → τ 小，轻微低洼都算 dig），可调大 `tau_frac` 收紧。
- pile 未做材质细分（土/砂/砾），仅靠语义排除障碍。
- dump 只选最大连通平地，未考虑运距/朝向（属未实现的 step6 任务约束）。
- VLM 定量坐标偏差（可在 prompt 里强约束坐标必须落在 bounds 内改善）。
- three.js 三维投射未做真人视觉实测（数据契约与 PNG 已验证）。

**后续方向**
- step6 任务约束规则引擎（源/目标区域、可作业性、运距）。
- 自建轻量 transformer 地形编码器（需自监督或标注）。
- 端到端动作 policy（决策 transformer / RL / 行为克隆，需 Unity 仿真轨迹数据）——
  衔接 scene-graph ICRA 方向。

---

## 服务与运行备忘

- VGGT 服务端口 8002（`start_all.sh` 启动全部；单独起：
  `cd WS/vggt && conda run -n vggt python ../vggt_yoloe/vggt_service.py --port 8002`）。
- **改 `.py` 后服务需重启才生效**（长驻进程）。
- VLM 测试环境无通用多模态 key；本次用通义千问中转 `https://api.silra.cn/v1/chat/completions`
  的 `qwen3-vl-plus`（OpenAI 兼容，脚本 provider 无关，可换 key/base_url/model）。
- 命令行出图/跑决策示例：
  ```bash
  # 粗采样对比
  python test_coarse_dem.py <session>/predictions.npz --factor 4 --scale 28 --out x.png
  # VLM 决策（批量）
  python test_terrain_vlm.py <s1>/predictions.npz <s2>/predictions.npz \
      --model qwen3-vl-plus --scale 28 --out-dir vlm_report --api-key <KEY>
  python make_vlm_report.py vlm_report/manifest.json
  ```
