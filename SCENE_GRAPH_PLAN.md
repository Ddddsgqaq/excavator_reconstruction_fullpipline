# 施工场景任务条件化 Scene Graph 开发方案（ICRA 方向）

> 状态：方案草案 **v0.3.2** · 2026-06-29
> 目标会议：ICRA（机器人 / 自动化方向）
> 范围：在现有 VGGT + YOLOe 单目土方系统之上，新增一个 **任务驱动、地形感知为核心** 的动态 scene graph 层，并以此提供面向自主挖掘的 **闭环决策支持（next-scoop）**。
>
> v0.2 变更：将系统重定义为「任务驱动 / 地形为核心 / 输出 scene graph 辅助施工机器决策」；数据模型改为 **L0–L4 分层 + 任务条件化**；引入机器人领域成熟 scene graph 范式作为学理支撑（见 §11 参考）。
>
> v0.3 变更（吸收 Interaction-Driven Updates, ICRA 2025）：新增**贡献 4 — 动作条件化的 scene graph 维护**（§5 机制 5）：把图更新嵌入挖掘动作本身，用 dig 事件**因果定位**被改变的区域，只重估受影响子图，而非逐帧全量重建；新增**视图条件化残差置信**（§5 机制 6 / §8 观测点选择的单目类比），把垂直压缩问题接入图机制；§10 增加 **图结构级评测指标**（区域/边识别 + 虚假区域率）；§11 补入该 ICRA 2025 工作并改写差异化论述。
>
> v0.3.1 变更（2026-06-29，强化决策闭环为论文脊柱）：重写 **§6** 为「以 scene graph 为接口的 next-scoop 信息闭环」——残差升级为**信念场** `belief={mu,sigma}`（§3.1/§6.1）；决策框架锁定**不确定加权贪婪**（σ 降权 + 驱动维护，不背主动感知主张）；决策输出回写为**带 provenance 的 recommendation 图对象**（§6.3）；给出**终止条件**与**两条验证路线**（§6.5：仿真/oracle 三级消融 + 静态视频例证，均不依赖开挖视频）；主动感知/信息增益降为 §8 风险9 扩展。
>
> v0.3.2 变更（2026-06-29，接入工作装置姿态）：把**挖机臂姿态估计**（大臂/小臂/铲斗 4 关键点）正式接入 **L2 施动者层**——作为 `state[t]` 的物理基础、§5 机制 5 dig 事件的触发器与因果定位锚（铲斗 3D 位置 = 被改变地形点的定位）。定位为**使能模块**，服务于地形核心叙事，不另立贡献点。方案与几何法 baseline 见 `EXCAVATOR_POSE_PLAN.md`；GT 来源 = Unity 合成数据，模型选型 HRNet-W18 热图回归 + MMPose。

---

## 0. 一句话定位

> 一个 **任务驱动（task-driven）**、以 **地形感知为核心（terrain-centric）** 的系统：从单目视频估计地形世界状态，由施工任务条件化地实例化一张动态 scene graph，并以此为施工机械提供闭环决策支持（下一铲挖哪）。

闭环结构（三个关键词其实是一条以"地形状态"为世界模型的回路）：

```
任务规格(Task: BIM/人给定目标地形)            ← 顶层"为什么"，条件化整张图
        ↓ 定义"什么地形特征重要、要感知什么、什么算成功"
地形感知(DEM/区域/残差场 = 世界状态)           ← 核心 state（系统强项）
        ↓ 组织成
Scene Graph(地形为脊柱 + 机械为施动者 + 任务条件下的可供性边)
        ↓ 残差驱动 + 可供性约束
决策(在可达/可供范围内选下一动作 next-scoop)
        ↓ 机械执行 → 地形改变 → 回到地形感知重估
```

把系统从"单目体积测量工具"抬升为"任务驱动的工地世界模型 + 机器人决策表征"。

---

## 1. 贡献点（论文主张）

1. **任务驱动、地形为核心的 scene graph**：作为连接单目感知与施工机械决策的中间表征；不同于既有室内 DSG 以离散物体关系为主，本表征以**连续地形残差场**为核心可决策状态。
2. **目标条件化的地形推理**：scene graph 编码"到设计面的残差场"+ 任务可供性边，直接支撑下一动作选择。
3. **以地形状态为世界模型的单目闭环**：感知→图→决策→动作→地形改变→重感知；用 VGGT 共享世界系替代度量 SLAM + 回环，并量化单目深度误差（垂直压缩）向残差场与决策的传播鲁棒性。
4. **动作条件化的 scene graph 维护（borrow: Interaction-Driven Updates, ICRA 2025）**：把图更新嵌入挖掘动作本身——dig 事件既是更新触发器、又是更新的**因果定位器**（已知哪台机械挖了哪个区域），因此只重估受影响的 region 节点及其可供性边，而非逐帧全量重建地形。相比室内"图像相似度变化检测 + VLM 重识别"的被动维护，土方场景的更新由**施动者自身动作因果驱动**，定位更准、语义更明确——是 interaction-driven maintenance 的一个更强实例。

> 与既有工作的差异（可辩护性，详见 §11）：把 DSG/Hydra 的分层动态图从**室内物体**迁到**室外土方、地形为核心**；用 Taskography/SayPlan/Clio 的**任务相关子图**思想做"任务条件化"，但条件化对象是**地形可供性 + 残差场**而非室内物体子集；用 Interaction-Driven Updates 的**任务执行中维护**思想做动态更新，但触发与定位由**几何动作（铲取体积变化）**而非 VLM 图像比对承担，且更新对象是**连续残差场**而非离散物体增删；全程**单目**。

---

## 2. 系统总览：现有能力 → 新增能力

| 层 | 现有（可复用） | 本方案新增 |
|---|---|---|
| 几何 | VGGT 点云/深度/位姿 `predictions.npz`；重力对齐 `R_align`（`gravity_alignment.py`） | 跨帧实例 3D 关联（共享世界系，无需新模型） |
| 语义 | YOLOe 实例 mask `semantic_masks.npz`；3D 标签融合 `semantic_fusion.py` | 实例持久 ID + 轨迹；状态机（辅助信号） |
| 地形 | DEM 拟合 + 坑自动定位 + 体积多边形（`elevation_plane.py`） | 区域分割（坑/料堆/平台/坡）+ **残差场** + 区域级体积时间序列 |
| 表征 | 散点输出 | **任务条件化分层 scene graph（L0–L4）** |
| 任务 | — | **任务规格层**（goal terrain + 约束 + 子目标 + 成功判据） |
| 维护 | — | **动作条件化增量维护**：dig 事件触发、单区域重估、边刷新（替代逐帧全量重建） |
| 工作装置姿态 | — | **挖机臂姿态估计**（大臂/小臂/铲斗 4 关键点 → 关节角 + 铲斗 3D 位置）：L2 `state[t]` 的物理基础、dig 事件触发器、机制 5 因果定位锚（使能模块，非主线贡献；详见 `EXCAVATOR_POSE_PLAN.md`） |
| 决策 | — | **残差驱动 + 可供性约束**的 next-scoop 推荐器 |
| 可视化 | Three.js elevation viewer (`elevation_viewer.html`) | 分层图叠加 + 残差场热力 + 推荐铲位高亮 |

---

## 3. 数据模型：任务条件化的分层 Scene Graph（L0–L4）

借 Hydra/3D-DSG 的分层 + 持久/瞬态思想，但把脊柱从"房间"换成"地形"：

```
L4  任务层(Task)        目标地形状态 + 约束 + 子目标；定义"相关性"与成功判据      ← 顶层驱动
L3  可供性/交互层        任务条件化的边：diggable / dumpable / reachable /          ← 任务条件化发生处
                        traversable / load_into；符号谓词在此 grounding
L2  施动者层(Agent)      机械/人：位姿、轨迹、状态机（瞬态）
L1  地形层(Terrain·核心)  区域节点(pit/pile/flat/slope/bench/haul-road) +           ← 持久世界状态
                        体积 + 残差场(current DEM − goal DEM)
L0  度量-语义层          VGGT 点云/深度/位姿 + YOLOe 语义（重力对齐系，原始几何）
```

- **持久 vs 瞬态**（borrow: 3D-DSG/Hydra）：L1 地形是会话内持久世界模型；L2 施动者瞬态。决策主要读 L1+L3，**不依赖 L2 完美**（动作识别降级为事件归因/确认）。
- **可查询/可规划**（borrow: Taskography）：L3 暴露为 planner 可用的符号谓词集。

### 3.1 节点
- **机械节点** `machine`（L2）：挖机/卡车/人。属性 `class, instance_id, centroid_xyz[t], bbox3d, state[t]`。其中 `state[t]`（digging/moving/idle…）与铲斗 3D 位置由**工作装置姿态估计**提供（大臂/小臂/铲斗 4 关键点，见 `EXCAVATOR_POSE_PLAN.md`）：单目深度/点云 → HRNet-W18 热图回归 4 关键点 → 配 `world_points` 抬升为 3D 关节，给出关节角、连杆姿态、铲斗尖位置。是 §5 机制 5 dig 事件触发与因果定位的物理信号来源（使能模块，不另立贡献点）。
- **地形区域节点** `region`（L1·核心）：`pit/pile/flat/slope/bench/haul-road`。属性 `type, boundary_polygon, volume[t], surface_height, centroid_xyz`，以及**信念场** `belief[t] = {mu[cell], sigma[cell]}`（残差均值+不确定，σ 来自机制6；见 §6.1）——残差不再是标量而是信念场。
- **派生节点** `workspace`（L2）：挖机可达工作区（回转半径 × 臂长近似圆环，参数可配）。
- **任务/目标** `task`（L4）：`task_type, goal_surface(DEM), constraints, subgoals, success_metric`。

### 3.2 边
- **空间边**（L1↔L2，单帧可算）：`near, adjacent_to, bearing`，`person↔machine distance`。
- **时序边**：`tracks(trajectory)`、`volume_trend / residual_trend(region)`。
- **可供性边**（L3，任务条件化）：`diggable, dumpable, reachable, traversable, load_into`（定义见 §5 机制 3）。
- **动作/状态边**（L2，辅助）：`digging, loading, contains`。

### 3.3 事件
`dig_cycle`（含起止帧、目标区域、估计铲取体积）、`cut_event/fill_event`、`material_flow(region→truck)`、`safety_violation`（可选）。

### 3.4 机械状态机（辅助信号）
`怠机/挖掘/回转/卸料/移动`，由挖机质心运动 + 铲斗-地形邻近（铲斗定位见 §8 待验证点）+ 目标区域 DEM 变化联合判定。**用于事件归因/确认，非决策承重墙。**

### 3.5 序列化（JSON schema 草案）
```json
{
  "session_id": "...",
  "world_frame": {"R_align": [[...]], "scale_factor": 1.0, "up": [x,y,z]},
  "task": {"task_type": "cut_to_grade", "goal_surface_ref": "bim/design_dem.npz",
            "constraints": {"max_slope": 34}, "success_metric": "residual_volume<=0.5"},
  "frames": [{"t": 0, "timestamp": 0.0}],
  "nodes": [
    {"id": "exc_1", "layer": "agent", "type": "machine", "class": "excavator",
     "state": [{"t":0,"value":"idle","centroid":[x,y,z]}]},
    {"id": "pit_1", "layer": "terrain", "type": "region", "region_type": "pit",
     "boundary": [[x,z]], "volume": [{"t":0,"v":12.4}],
     "residual": [{"t":0,"to_cut":8.0,"to_fill":0.0}]}
  ],
  "edges": [
    {"src":"exc_1","dst":"pit_1","layer":"affordance","type":"diggable",
     "active_under_task":["cut_to_grade"],"intervals":[[0,30]]}
  ],
  "events": [
    {"type":"dig_cycle","machine":"exc_1","region":"pit_1",
     "t_start":5,"t_end":18,"scooped_volume_est":0.8}
  ],
  "predicates": ["reachable(exc_1,pit_1)","diggable(exc_1,pit_1)","cut_remaining(pit_1)=8.0"]
}
```
落盘 `workspaces/<session>/scene_graph.json`，与 `edited_scenes.json` / `*_meta.json` 平级。

---

## 4. 第一版锁定的任务类型（two tasks）

覆盖两种条件化模式，既全面又不摊太大：

| 任务 | 类型 | 核心激活 | 决策目标 |
|---|---|---|---|
| **T1 挖到设计标高** `cut_to_grade` | 纯地形 | region 残差场 + `diggable` 边；卡车折叠 | 在残差最大且可达单元给 next-scoop |
| **T2 清运料堆装车** `load_to_truck` | 含物料流 | `pile`(to-remove) + `load_into` + 物料流边；远处坑折叠 | 取高价值可达单元 → 卸入卡车，跟踪车斗料位 |

> 目标 `goal_surface` 由人/BIM 给定（外部输入），不自动推断。

---

## 5. 任务条件化机制（设计核心）

任务不是被动 target，而是**主动塑形整张图**。四个机制，每个都有机器人领域对标：

**机制 1 · 相关性门控（borrow: SayPlan 折叠-展开；Taskography SCRUB 任务条件化稀疏化；Clio 信息瓶颈任务相关聚类）**
任务类型决定**哪些节点/边被实例化或激活**。同一场景：T1 激活残差场 + `diggable`，卡车折叠；T2 激活卡车 + `load_into` + 物料流，远处坑折叠。图始终小而任务相关。

**机制 2 · 目标 grounding → 残差场（核心 state）**
goal_surface 投到地形层，生成 **残差场 = current DEM − goal DEM**，自动给区域打 `to-cut / to-fill / on-grade` 属性。这是把连续地形变成可决策对象的关键一步，也是与室内离散关系图的核心形式差异。

**机制 3 · 任务条件化可供性边（borrow: affordance；功能性 scene graph）**
边存在 ⇔ `(任务, 机械能力, 当前地形状态)` 下动作可行且有益，例如：
```
diggable(machine, region) ⇔ region.residual.to_cut > 0
                            ∧ within_reach(machine, region)
                            ∧ slope_stable(region)
```
边随地形变化动态失活/迁移（挖掉一铲后自动更新）。

**机制 4 · 符号谓词层 → 衔接决策（borrow: Taskography PDDL 谓词；scene graph as planner state）**
L3 暴露 PDDL 风格谓词（`reachable, diggable, cut_remaining(region)=v` …），任务表达为目标条件。next-scoop = 在可供性约束下选最大化残差缩减的动作——一个 grounded 在 scene graph 上的 task-level 决策。

**机制 5 · 动作条件化的增量维护（borrow: Interaction-Driven Updates, ICRA 2025 的 maintenance-during-execution + change-detection）**
不逐帧全量重建。维护循环嵌入挖掘动作：
```
dig_cycle(machine, region) 检出  →  仅对该 region 重算 DEM/残差/体积
                                  →  仅刷新与该 region 关联的可供性边（diggable/load_into…）
                                  →  邻接区域按需懒更新（料堆增长、边坡迁移）
```
- **触发**：用区域 DEM 的体积/残差变化（几何 change-detection）替代室内方法的图像 embedding 余弦相似度阈值 θ——动作完成即触发，且**天然知道改了哪个 region**（动作的目标区域 = 更新定位），无需视觉比对去猜变化位置。
- **因果定位的物理锚**：dig 事件由**铲斗关键点轨迹**检出（铲斗下沉→贴地→抬升的姿态序列 = 一次挖掘），其 3D 位置（姿态估计的铲斗尖关键点 + `world_points` 抬升）直接给出"这一铲改的是地形哪个 (x,y)"——比图像相似度比对的位置精确得多，是本方案"几何动作因果驱动定位"主张的具体兑现（见 `EXCAVATOR_POSE_PLAN.md`）。姿态信号缺失时回退到下面的几何体积阈值兜底。
- **代价**：把"每帧 O(全场)"降为"每事件 O(单区域 + 邻接)"，对长序列与流式增量友好（呼应 §8 风险 6）。
- **降级鲁棒**：dig 事件漏检时退化为周期性区域重估；区域更新失败时保留上一持久状态（持久 vs 瞬态，§3）。

**机制 6 · 视图条件化的残差置信（observation-point selection 的单目类比；直面垂直压缩）**
室内方法主动选最佳观测点；单目回放不可控相机，但可**选最优观测帧/视角子集**来估某区域残差。
```
view_value(region, frame) ∝ 视差基线(frame 对该 region) · 掠射角适宜度 · 可见性(无遮挡)
```
- 为每个 region 选 value 最高的关键帧集合做 DEM/残差融合，规避低基线、近正俯视等会放大**垂直压缩**误差的视角。
- 把"区域残差置信"作为节点属性写入图（`residual_confidence`），供决策层在低置信单元降权（接 §6 score）。
- 与机制 5 协同：每次事件更新时按视图价值重选融合帧，而非盲用全部帧。

### 5.1 同场景两任务 → 两张被条件化的图（worked example）
场景：1 挖机、1 料堆 pile_1、1 待挖区 pit_1、1 卡车 truck_1。
- **T1**：激活 `pit_1` 残差(to-cut=8m³) + `diggable(exc,pit_1)`；truck_1 折叠 → 决策取 pit_1 残差最大可达单元。
- **T2**：激活 `pile_1`(to-remove) + `load_into(exc,truck_1)` + 物料流；pit_1 折叠 → 取 pile_1 高价值可达单元卸入卡车，跟踪料位。

**同一份感知、同一张底图，任务一变，实例化子图与决策完全不同**——这是"任务条件化 scene graph"的可演示卖点。

---

## 6. 决策层设计：以 scene graph 为接口的 next-scoop 信息闭环

> 论文脊柱。本节回答"如何根据当前重建结果给挖机下一步信息"——把 scene graph 定位为**将不确定单目重建转译为可执行决策变量**的接口。决策框架锁定为**不确定加权贪婪**（v0.3.1 决策）：σ 只用于给残差降权 + 驱动动作条件化维护，不背"主动感知/信息增益"主张（后者降为 §8 扩展）。

**目标输入**：人/BIM 给定（设计开挖面 / 待清料堆 / 目标标高）。

### 6.1 残差升级为信念场（核心形式变化）
单目重建（尤其垂直压缩 7.9–10.8×）给出的 `current_DEM` 有偏且不确定；若把残差当确定标量，"挖残差最大格子"可能在挖**压缩伪影撑出的假残差**。故 L1 region 节点携带的不是标量残差，而是**信念场**：
```
belief(cell) = { mu: 估计待挖体积, sigma: 不确定度 }   # sigma 来自机制6视图条件化置信
```
σ 高 = 低视差 / 近正俯视 / 被压缩污染的格子。**scene graph 由此从"几何快照"升为"地形信念状态"。** 这把项目招牌缺陷（垂直压缩）作为 σ 的来源接入决策，而非藏起来。

### 6.2 候选铲位打分（不确定加权贪婪）
对残差场内可达单元/采样点：
```
U(cell) = E[residual_reduction(cell)]         # 该处一铲期望可缩减的残差体积（按 belief.mu）
        · residual_confidence(cell)           # = f(1/sigma)；低置信单元降权（机制6）
        · (1 - recently_dug(cell))            # 避免重复挖同一处
        - w · slope_risk(cell)                # 边坡稳定性惩罚
        s.t. reachable(cell)                  # 不可达直接排除（硬门控，非软项）
```
输出推荐铲位坐标 + 朝向（相对挖机方位），叠加 DEM。

### 6.3 决策输出回写进图（"设计到 scene graph 中"的兑现）
推荐不是外挂结果，而是一个**带 provenance 的图对象**，可追溯"为什么挖这里"：
```json
{"type":"recommendation","target_region":"pit_1","cell":[x,z],
 "approach_bearing":θ,"expected_volume":μ,"confidence":1-σ_norm,
 "justified_by":["diggable(exc,pit_1)","cut_remaining(cell)=0.8±0.1","reachable(exc,cell)"]}
```
`justified_by` 边指回让该推荐成立的 L3 谓词——决策全程 grounding 在表征上。

### 6.4 闭环（动作条件化维护，机制 5）
每完成 dig_cycle → **仅重估被挖 region 的 belief**（非全场）→ 该区 **σ 下降**（已知铲取体积作度量约束 + 事后新视角重观测）→ 刷新该区可供性边 → 重算受影响单元 U → 下一铲。
**终止条件**（信念空间可写）：`∀cell ∈ task_region: μ ≤ tol ∧ σ ≤ τ`——挖到设计标高**且**对此有信心。

### 6.5 闭环验证路线（短期无开挖视频，两条并行）
- **(A) 仿真/oracle 闭环（定量主结果）**：合成/oracle `goal_DEM` + 起始 `current_DEM`；模拟一铲 = 按运动学挖掉选中格子的楔形体积；**模拟观测噪声 = 套用本项目实测垂直压缩噪声模型**（7.9–10.8× 那批数据）给 σ 赋值。跑三级 baseline 对比 **收敛铲数 / 终末残差误差 / 浪费在伪影上的铲数**：
  - ① 贪心 max-residual（无置信）
  - ② + 不确定加权（σ 降权，本方案）
  - ③ +（扩展）主动 info_gain —— 见 §8，仅作上界参照
- **(B) 静态视频例证（定性图）**：在现有静态重建结果上跑一次 §6.2 打分，产出"真实场景上的 next-scoop 推荐 + justified_by 溯源"叠加图，作为方法可视化。
- 不依赖真实开挖视频即可支撑决策章，与 E-DYN-0（仅验 L2 几何可行性）解耦。

---

## 7. 视频数据要求（采集规格）

**内容**：≥1 台挖机连续作业，完成 3–5 个完整挖掘周期，地形肉眼可见改变（坑变深/料堆变大）；最好含 1 辆自卸车被装料（T2 物料流）+ 1 个人（安全边）；视野含待处理目标区域。
**相机**：倾斜俯视角（非贴地）；先短暂环绕/平移扫场（几秒，给 VGGT 视差+尺度），再固定稳定视角拍作业；避免纯原地旋转；作业区全程在画面内、保持重叠；尽量稳。
**加分项**：含已知尺寸参照物（标定杆/锥桶/挖机履带），用于公制尺度 + 对付垂直压缩，串入 ablation。
**技术参数**：1080p+、24–30 fps、光照均匀、单段 1–3 分钟。

---

## 8. 风险与待验证点（后续尝试）

1. **垂直压缩 7.9–10.8×**：污染区域体积/残差场/"斗离地高度"→污染状态机与决策。
   - 行动：接入现有标定（`scale_calibration.py` / `vertical_*`），把"决策对残差误差的敏感度"做成 ablation。
   - 行动（v0.3 新增）：用机制 6 的**视图条件化残差置信**把高压缩误差视角降权/排除，作为不依赖标定的几何缓解手段，与标定法对比串入 ablation。
2. **铲斗定位可行性**（关键前置，本轮暂忽略）：YOLOe 文本 prompt `excavator bucket` 能否稳定分出；不行则用挖机实例 mask 几何近似（最低/最前端）。**TODO/future test。**
3. **next-scoop 真实闭环/仿真验证**（本轮暂作测试要点）：是否引入仿真（挖机运动学）或 oracle 对照定量评估推荐质量。**TODO/future test。**
4. **单目 VGGT 漂移**：长序列跟踪稳定性 → MVP 限定单会话单段视频。
5. **决策目标自动推断**（暂不做）：当前目标由人/BIM 给定；从 scene graph 自动推断目标留作扩展。
6. **多 VGGT 窗口/流式增量**：长视频分窗重建世界系拼接 → 后续工程化再处理（机制 5 的事件级增量更新为此减负）。
7. **动作条件化维护的触发依赖**（机制 5 前置）：dig_cycle 漏检/误检会导致区域不更新或错更新；MVP 用 DEM 体积变化阈值兜底（几何触发，不依赖完美动作识别），并对触发阈值做敏感度分析。
8. **视图价值估计可行性**（机制 6 前置）：单目下视差基线/掠射角的逐区域估计是否稳定；不稳则退化为"用全部可见帧 + 残差方差当置信"的简化版。**TODO/future test。**
9. **主动感知 / 信息增益决策（扩展，非本版核心）**：把 next-scoop 从 §6.2 的不确定加权贪婪升级为**信念空间决策** `U = E[residual_reduction] − λ·risk + β·info_gain`，让"挖最不确定处"主动降低重建不确定（已知铲取体积作度量约束 + 事后重观测）。novelty 更强但需辩护"一铲确能降 σ"的因果 claim，验证负担更重；v0.3.1 决策**降为扩展**，仅在 §6.5 仿真中作 baseline ③ 上界参照。

---

## 9. 分阶段开发计划

### Phase 0 — 任务本体 + Schema + 评测协议（地基）
- 定稿 §3 分层 schema + §4 两任务的任务本体 + 残差场定义。
- 确定 §10 指标；测试视频上标注：挖掘周期、机械状态、区域体积/残差 GT（衔接大坑 GT 实验）。
- **交付**：`scene_graph_schema.json`、任务规格模板、标注规范、评测脚本骨架。

### Phase 1 — 单帧分层关系图 MVP（纯复用）
- DEM 区域分割（坑/堆/平台/坡）+ **残差场计算**（基于 `elevation_plane.py` 残差/曲率）。
- YOLOe 实例 → 3D 质心 machine 节点。
- 空间边 + 初版可供性边；查看器叠加分层图 + 残差热力。
- **交付**：单帧 + 任务条件化的 `scene_graph.json` + 查看器叠加。

### Phase 2 — 时序跟踪 + 状态机 + 事件
- 3D 质心跨帧关联（贪心/匈牙利）→ 持久 ID + 轨迹。
- 机械状态机（辅助）+ 区域残差/体积时间序列 + 挖掘事件检测。
- **交付**：时序 `scene_graph.json`（state[t]、residual[t]、events）+ 事件日志。

### Phase 3 — 任务条件化 + 动作条件化维护 + 决策闭环（ICRA 核心）
- 实现机制 1–4（门控/残差 grounding/可供性边/谓词导出）。
- 实现机制 5（**动作条件化增量维护**：事件触发 → 单区域重估 → 边刷新）+ 机制 6（视图条件化残差置信）。
- 物料流边（T2）+ **残差驱动 next-scoop 推荐器**；查看器高亮推荐铲位 + 残差置信热力。
- 回放式闭环 demo（随场景演化**增量**更新图与推荐，并对比逐帧全量基线的代价）。
- **交付**：任务条件化引擎 + 维护引擎 + 决策模块 + 闭环 demo。

### Phase 4 — 评测与成文
- 跑 §10 指标；核心 ablation：有无垂直压缩校正对残差→决策的影响。
- 产图、消融、鲁棒性讨论 + 方法/实验章节。

---

## 10. 评测协议与指标

| 维度 | 指标 | GT 来源 |
|---|---|---|
| 任务条件化 | 任务相关子图是否正确激活/折叠（精确率/召回） | 标注 |
| 图结构(节点) | **区域识别准确率 RRA**（区域数/类型，NRA 的土方类比）；**虚假区域率 SRR**（不存在的区域被实例化，NHP 类比） | 标注 |
| 图结构(边) | **边识别准确率 ERA**（可供性/空间边与真实条件相符比例） | 标注 |
| 图结构(整体) | **加权 Jaccard WJS**（估计局部图 vs 真实局部图，区域体积/残差为权） | 标注 |
| 维护(机制5) | 事件触发的更新**正确性**（被改区域是否正确定位重估）+ **更新代价**（每事件耗时 vs 逐帧全量基线） | 标注 + 计时 |
| 状态识别 | 机械状态/挖掘周期检测准确率、F1 | 人工标注 |
| 地形/残差 | 区域 cut/fill 体积误差、残差场误差 | 坑 GT / 参照物标定 |
| 物料流(T2) | 挖取-装载体积一致性 | 标注 + 车斗估计 |
| 决策 | next-scoop 与专家/oracle 一致性；任务进度（残差缩减）速度 | 专家/oracle |
| 鲁棒性 | 有无垂直压缩校正 / 有无视图条件化置信下，残差与决策的变化 | ablation |

> 图结构指标（RRA/ERA/WJS/SRR）直接对标 Interaction-Driven Updates (ICRA 2025) 的 NRA/ERA/WJS/NHP，便于审稿人横向理解；但权重 w(x) 改为**区域体积/残差**（连续量），而非室内的物体计数。

---

## 11. 参考工作（已联网核实出处）

机器人 scene graph 范式，作为本设计的学理支撑与对标：

1. **3D Scene Graph**（分层抽象的起点）— Armeni, He, Gwak, Zamir, Fischer, Malik, Savarese. *3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera.* ICCV 2019, pp.5664–5673. DOI:10.1109/ICCV.2019.00576. → 借：building→room→object→camera 的分层抽象（我们换成地形脊柱）。
2. **3D Dynamic Scene Graphs / Kimera**（动态 + agent 层）— Rosinol, Gupta, Abate, Shi, Carlone. *3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans.* RSS 2020. arXiv:2002.06289。期刊版 Kimera, IJRR 2021, 40(12–14):1510–1546. → 借：agent 层（机械/人）+ 持久/瞬态 + actionable 表征。
3. **Hydra**（实时增量构建 + 状态优化）— Hughes, Chang, Carlone. *Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization.* RSS 2022. arXiv:2201.13360. → 借：在线增量构建 + 回环优化（VGGT 共享世界系作替身）。
4. **ConceptGraphs**（开放词表）— Gu, Kuwajerwala, Morin, Jatavallabhula 等. *ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning.* ICRA 2024, pp.5021–5028. DOI:10.1109/ICRA57147.2024.10610243. → 借：基础模型开放词表（YOLOe 文本 prompt 已契合）。
5. **Taskography**（任务条件化稀疏化 + PDDL 规划）— Agia, Jatavallabhula, Khodeir, Miksik, Vineet, Mukadam, Paull, Shkurti. *Taskography: Evaluating Robot Task Planning over Large 3D Scene Graphs.* CoRL 2021, PMLR. arXiv:2207.05006。其 **SCRUB = task-conditioned 3DSG sparsification**。 → 借：机制 1（任务条件化剪枝）+ 机制 4（PDDL 谓词规划）。
6. **SayPlan**（层级折叠-展开 + 任务相关子图）— Rana, Haviland, Garg, Abou-Chakra, Reid, Sünderhauf. *SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning.* CoRL 2023, PMLR 229:23–72. arXiv:2307.06135. → 借：机制 1（折叠-展开 + 任务相关子图语义搜索 + 模拟器迭代重规划）。
7. **Clio**（任务驱动、信息瓶颈）— Maggio, Chang, Hughes, Trang, Griffith, Dougherty, Cristofalo, Schmid, Carlone. *Clio: Real-Time Task-Driven Open-Set 3D Scene Graphs.* IEEE RA-L 2024, 9(10). arXiv:2404.13696. → 借：任务驱动地决定"保留什么粒度/子图"（信息瓶颈视角佐证机制 1）。
8. **Interaction-Driven Updates**（任务执行中维护 + 观测点选择，最贴近本方案的最新对标）— Li, Zhang, Chen, Zhao, Niu. *Interaction-Driven Updates: 3D Scene Graph Maintenance During Robot Task Execution.* ICRA 2025, pp.11933–11939. DOI:10.1109/ICRA55743.2025.11128194. → 借：**机制 5**（把更新嵌入交互/动作、change-detection 门控、只更新受影响子图，避免全量重扫）+ **机制 6**（observation-point selection → 单目最优观测帧选择）+ **§10 图结构指标**（NRA/ERA/WJS/NHP 的土方类比）。**关键差异**：其触发与定位靠 VLM 图像相似度比对、更新对象是离散物体增删、室内仿真为主；本方案触发与定位靠**几何动作（铲取体积变化）因果驱动**、更新对象是**连续残差场**、室外单目土方，且无需 LLM/VLM 在环。

> 我们的定位：以上多为**室内导航/操作、离散物体关系**；本方案迁移到**室外土方、地形为核心、连续残差场 + 任务条件化可供性 + 动作条件化几何维护 + 单目**，即为 novelty。其中相对最新的 ICRA 2025 维护工作（参考 8），本方案把"interaction-driven"从被动视觉变化检测推进为**施动者动作因果定位的几何增量维护**，是更强且更适合土方闭环的实例。

---

## 12. 里程碑（粗排，待视频到位后细化）
- M0 任务本体 + Schema + 评测（Phase 0）
- M1 单帧分层图 + 残差场 + 查看器叠加（Phase 1）
- M2 时序图 + 状态机 + 事件（Phase 2）
- M3 任务条件化 + 动作条件化维护 + 决策闭环 demo（Phase 3）
- M4 评测 + 论文章节（Phase 4）

> 下一步：①你提供符合 §7 的视频；②确认 §3 分层 schema 与 §4 两任务定义；③Phase 0 启动任务本体与评测脚本。
