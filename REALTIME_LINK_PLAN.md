# 实时联动计划：VGGT-YOLOe 流式重建 → Unity 高程更新

> 目标：把当前"离线一次性重建 + 静态高程图"改造成**按频率更新的流式重建**，
> 并把每次更新的 DEM 实时推送到 `excavator-app-unity-main` 的高程导入模块，
> 实现挖机应用里地形的准实时刷新。
>
> 关联 memory：[[research-direction-volume]]、[[vertical-compression-rootcause-refs]]、
> [[semantic-terrain-fusion]]、[[scene-graph-icra-direction]]、[[edited-scenes-registry]]。

---

## 📌 当前状态总览（更新于 2026-07-20）

**已完成：M0 → M5（M5.1+M5.2）。链路全通:实时联动(M0-M3)+坐标稳定(M4)+持久融合多tile(M5)，全部脱 GPU 验证过。**

| 里程碑 | 状态 | 一句话 |
|---|---|---|
| M0 地基/契约 | ✅ 完成 | 修 Unity 路由 bug、加自动轮询、冒烟工具，格式契约打通 |
| M1 管线函数化 | ✅ 完成 | `reconstruct_frames_to_dem` 复用离线代码；稳态 ~4.6s/次 |
| M2 流式取帧 | ✅ 完成 | 帧泵 + 在线关键帧滚动缓冲，脱 GPU 验证过 |
| M3 循环+发布+端点 | ✅ 完成 | 自动循环每 T 秒重建并发布；`/stream/*` 端点；文件通道 PASS |
| M4 坐标稳定 | ✅ 完成 | 冻结锚点（重力+尺度+footprint）+ 轮间 yaw/平移配准；静态地形 DEM 跨轮 RMS 0.49m→0.00m |
| M5 增量融合 | ✅ 完成(M5.1+M5.2) | 持久全局 DEM(加权+时间衰减,快跟随挖掘)+ 多 tile 变化检测,只发变化 tile。M5.3(小窗口提频)留 future |

**新增文件（全部隔离在 `streaming/` + `tools/`，离线路径零改动）：**
- `streaming/pipeline.py` — 帧→DEM→ElevationMsg（M1）+ 冻结 footprint 网格、轮间配准接线（M4）
- `streaming/frame_source.py` — 帧泵（VideoFileSource；MediaMtxSource 留占位）（M2）
- `streaming/keyframe_buffer.py` — 线程安全滚动关键帧缓冲（M2）
- `streaming/elevation_publisher.py` — file+mqtt 双通道发布（M3，多 tile 天然支持）
- `streaming/reconstruct_loop.py` — pump+recon 双线程循环（M3）+ 锚点生命周期（M4）+ 融合发布分支（M5）
- `streaming/endpoints.py` — `/stream/start|stop|status`（M3）+ M4/M5 开关/状态
- `streaming/registration.py` — 轮间 2D 刚体（yaw+平移）配准（M4.2）
- `streaming/global_dem.py` — 持久全局 DEM 累加器 + 多 tile 切分/变化检测（M5，新增）
- `tools/send_test_tile.py` — 冒烟工具（M0）
- `tools/verify_m4.py` — M4 真 GPU 冒烟 + 配准开关对照
- `tools/verify_m5.py` — M5 脱 GPU 单元 (`--unit`) + 真 GPU 冒烟

**改动的既有文件（最小、非破坏）：**
- `vggt_service.py` — 只加一段 try 包裹的 `app.include_router(_stream_router)`
- `Assets/map/MqttManager.cs` — 修高程路由（→ TerrainTileManager.OnElevationTile）
- `Assets/Terrain/ElevationFileLoader.cs` — 加 Auto Poll 自动轮询 + 心跳日志

**⏳ 下一步 / 待办（按优先级）：**
1. **接 Unity 实跑**（用户侧）：`python vggt_service.py` → `POST /stream/start`（M5 加 `"fusion":true`）→
   Unity Inspector 设 `singleFixedTile=false`、`activeRadius=1` → 看地形多 tile 持续生长、只刷新变化处。
   M3/M4/M5 都还没在真 Unity 里目视确认。
2. **M4/M5 真 GPU 冒烟**：脱 GPU 单元验证全过；还需 vggt_service 进程内跑真视频:
   `python tools/verify_m4.py <mp4>`（配准开关对照）、`python tools/verify_m5.py <mp4>`（多 tile 发布 + 挖掘跟随）。
3. **MQTT 端到端验证**：需在 Windows 装 mosquitto（`127.0.0.1:1883`）。代码已就绪，只差 broker。
4. **M5.3（future）小窗口 VGGT 提频**：新关键帧只对小窗口跑 VGGT，靠融合累积全局形状 → 提高更新频率。
   本次未做（需真 GPU 才能验证提频收益）。对接 [[scene-graph-icra-direction]] 的 4D 场景图。
5. （可选）M0 遗留：目录三元组 `(heightmapResolution=129, grid=128, tile_size)` 已确认匹配，无需改。

**已知局限（如实记录，非 bug）：**
- 更新频率是"每几秒"（稳态 ~5s/次），非帧率；提频要等 M5。
- 模型冷加载 ~20s：务必让 vggt_service 常驻，避免吃掉首轮重建。
- 垂直压缩：VGGT 固有问题，深层解决属研究范畴。M4 冻结了尺度（消除跨轮缩放跳动），
  但绝对尺度仍是首轮估计值、非真值标定；见 [[vertical-compression-rootcause-refs]]。
- M4 锚点绑定"首轮成功重建"：若首轮重力/尺度估计差，会污染整段流。M4.3（RTK 地理配准）留作后续。
- M4 配准依赖地形有起伏（relief 才能定 yaw/平移）；纯平地某方向的平移根本不可观测（无害，平地平移后本就一样）。
- M5 融合默认 `fusion=false`（opt-in）：关时行为与 M4 完全一致（单 tile）。开时全局栅格默认 150m×150m（锚点±75m）、
  3×3 tile；挖机跑出该范围的点会被丢弃（未来 M5 可扩为动态生长/更大 world_size_m）。
- M5 快跟随 `decay=0.5`：对挖掘响应快（~2-3 轮跟上），但对单轮噪声也更敏感；稳/快是可配折中，真机再调。
- M5 更新频率仍是"每几秒整窗重建"，全局融合不降低单轮成本；真正提频要 M5.3 小窗口 VGGT（future）。

---

## 0. 已锁定的设计决策（2026-07-16）

| 维度 | 选择 | 理由 |
|---|---|---|
| **视频源** | 先用本地 mp4 按时间轴回放模拟"实时到帧" | 先把流式管线跑通，再换 MediaMTX/WebRTC 真源 |
| **更新方式** | **先 A（滑窗周期重跑）后 B（持久增量融合）** | A 复用现有全部管线、最快端到端；B 是研究级升级 |
| **坐标锚定** | 先 `singleFixedTile` 最简：每次 DEM 覆盖同一块 tile，只做重力对齐 + 尺度冻结，暂不做严格配准 | 快速验证，跳动问题留到 Milestone 4 |
| **传输通道** | **MQTT 主 + 文件落盘备**：paho-mqtt 发 `01/map/elevation`；同时写 JSON 供 `ElevationFileLoader` 回放/调试 | Unity 侧两条通道都已存在 |

## 0b. 全局硬约束（贯穿所有里程碑）

> **任何改动都不得影响已有的离线重建链路。** 具体守则：
> - 所有流式代码放在**新增** `streaming/` 包与 `tools/` 下；离线路径不 import 它们。
> - `vggt_service.py` 里只**新增** `/stream/*` 端点，不改 `/reconstruct`、`/fit_elevation` 等既有端点的行为。
> - 不强行改动共享 conda 环境（`vggt50`/`yoloe`）里已固定的依赖版本；新依赖必须是纯增量
>   （如 `paho-mqtt`），且离线流程不依赖它。
> - Unity 侧改动最小化，且保持与既有 `ElevationFileLoader` 相同的路由语义，不破坏原有消息处理。

## 1. 关键事实（两侧现状，作为改造起点）

**Python 侧（当前项目）——纯离线批处理**
- 链路：`视频→抽帧→VGGT 一次前向→点云→重力对齐(Y=up)→DEM 网格→elevation_export.py`。
- `vggt_service.py`（FastAPI :8002）已有 `/reconstruct`、`/fit_elevation`、`/terrain_analysis`、`/export_elevation_json`。
- VGGT 模型在 `vggt_service.py` 进程内常驻（`_model`）；`_run_inference` 是一次性、吃全部帧。
- **`elevation_export.py:dem_to_elevation_msg()` 产出的 JSON 就是 Unity 的 `ElevationMsg` 格式**（int16、行主序、`height_resolution` 量化）——传输格式已天然对齐。
- ❌ 无任何流式/增量/滑窗/WebSocket 机制。

**Unity 侧（excavator-app-unity）——导入通道已就绪**
- MQTT 客户端（M2Mqtt）已连 broker（默认 `127.0.0.1:1883`），订阅 `01/map/elevation`。
- `MqttManager.Update()` 主线程出队 → 应调 `TerrainTileManager.OnElevationTile(msg)` → `SetHeights`。
- **1Hz 反复调用安全**：TerrainData 是 clone、非累积、约 5ms/次（前提：关掉着色）。
- 备用通道：`ElevationFileLoader`（读磁盘 JSON，目前手动按钮触发）、WebRTC/MediaMTX（WHEP 入站视频）。
- ⚠️ **已知 bug**：`MqttManager.HandleElevation` 现在错误路由到 `HandleElevationMap`（其 terrain=null），需改成 `TerrainTileManager.OnElevationTile`。

## 2. 核心判断

1. **传输是现成的**——格式已对齐、Unity 消费端已能 1Hz 更新。Python 只需加一个发布者。
2. **工程量在 Python 侧"流式化"**——VGGT 是批模型，输出在"第 0 帧相机系、任意尺度"里。
3. **命门是跨更新的世界系一致性**——不锚定则 Unity 地形每次更新会跳动/缩放（Milestone 4 处理）。

---

## Milestone 0 — 地基与契约（无 ML，先把两端能"说上话"）

**目标**：把两侧的配置对齐、把 Unity 的接收端修通，做到"我手动扔一个 JSON，Unity 地形就变"。

**M0.1 固定 tile / 网格契约（先量，别猜）**
- 在 Unity 编辑器里查 `Assets/Terrain/GroundTerrainData.asset` 的 `heightmapResolution`（如 513）。
- 约束：导出网格 `width == height == heightmapResolution - 1`（如 512）。当前离线默认 128，需要
  统一——要么把 DEM 网格分辨率提到 512，要么在导出时重采样到 `res-1`。
- 定死单 tile 的物理边长 `tile_size_meters`（Unity 默认 50m），并与 Python 侧 `scale_factor` 打通。
- 产出：在本文件附录记录三元组 `(heightmapResolution, grid_res, tile_size_meters)`。

**M0.2 修 Unity 接收端（本次唯一的 Unity 改动，很小）**
- `MqttManager.HandleElevation`：路由从 `HandleElevationMap` 改为
  `FindFirstObjectByType<TerrainTileManager>().OnElevationTile(elevation)`。
- 确认场景 `singleFixedTile = true`，且 `TerrainTileManager` + `ElevationTileStore` 挂载正常。
- （备通道）给 `ElevationFileLoader` 加一个可选的"定时轮询目录"开关（现在只有手动按钮），
  用于文件落盘回放。

**M0.3 联调冒烟测试（不含 VGGT）**
- 写一个 `tools/send_test_tile.py`：读现有某个 `workspaces/<id>/elevation_tile_0_0.json`，
  经 MQTT publish 到 `01/map/elevation`，同时拷一份到 Unity 的落盘目录。
- 起 mosquitto broker，Unity 运行 → 看到地形按该 tile 变形即 M0 通过。

**验证**：Unity 地形随手动发送的 JSON 变化；文件通道与 MQTT 通道都验证一遍。

---

## Milestone 1 — 把离线管线抽成可复用函数（Python 内重构，行为不变）

**目标**：把"一组帧 → ElevationMsg + 世界变换"从 Gradio/一次性流程里剥离成一个纯函数，
供流式循环反复调用；**复用常驻的 `_model`，不重复加载**。

**M1.1 提取核心函数**（放 `streaming/pipeline.py`，或 `vggt_service.py` 内部函数）
```
def reconstruct_frames_to_dem(frames, *, prev_anchor=None, cfg) -> DemResult
    # frames: List[np.ndarray]  (已抽好的帧)
    # 复用: _run_inference -> 点云；gravity_alignment；elevation_plane/terrain_analysis 的 DEM
    # 返回: elev 网格(float,H×W)、has_data 掩码、x/z bounds、R_align、scale_factor、gravity_source
```
- 内部直接调现有 `_run_inference`、`gravity_alignment`、DEM 构建，不走 HTTP、不写一堆磁盘产物。
- `prev_anchor` 预留给 Milestone 4（先传 None）。

**M1.2 DEM → ElevationMsg 适配**
- 直接调现有 `elevation_export.dem_to_elevation_msg(...)`，补 `timestamp`、`tile_x/y=0`。
- 若 M0.1 决定重采样到 `res-1`，在这里做（scipy 或 cv2.resize，注意 NODATA 掩码）。

**M1.3 单元验证**
- 用一段现有 mp4 抽 N 帧，调 `reconstruct_frames_to_dem` → 导出 JSON → 走 M0.3 通道进 Unity，
  结果应与旧离线流程一致（回归对比 DEM 网格）。

**验证**：新函数产出的 DEM 与旧 `/fit_elevation` 数值一致（复用同一套代码，应当逐格相等）。

---

## Milestone 2 — 流式取帧（视频文件模拟"实时到帧"）

**目标**：一个帧泵，按墙钟节奏从 mp4 吐帧，喂给一个滚动关键帧缓冲，模拟真实视频流。
换成 MediaMTX/WebRTC 时只替换这一个模块。

**M2.1 帧泵**（`streaming/frame_pump.py`）
- `VideoFileSource(path, target_fps)`：`cv2.VideoCapture` + 按 `1/target_fps` sleep，模拟实时。
- 抽象出 `FrameSource` 接口，后续 `MediaMtxSource`（RTSP/WHEP 拉流）实现同接口即可无缝替换。

**M2.2 滚动关键帧缓冲**（`streaming/keyframe_buffer.py`）
- 复用现有 ORB 相似度关键帧选择逻辑（`yoloe_service.py:select_keyframe_indices` 的"similarity"模式）
  做在线版：新帧与缓冲里最近帧比视差/内点比，够新才入缓冲。
- 环形缓冲，容量 N（A 方案的滑窗大小，先设 8~16，可配）。
- 线程安全：帧泵线程写、重建循环线程读，加锁快照。

**M2.3 验证**
- 干跑帧泵 + 缓冲，打印"每秒入缓冲关键帧数、当前窗口大小"，确认关键帧筛选合理、无阻塞。

---

## Milestone 3 — 周期重跑循环 + 双通道发布（A 方案，端到端打通）

**目标**：**这一步交付可演示的实时联动**。每隔 T 秒对当前窗口重建一次并推送。

**M3.1 重建循环**（`streaming/reconstruct_loop.py`，跑在 vggt_service 进程内以复用 `_model`）
```
每隔 T 秒（或每新增 K 个关键帧）:
    frames = keyframe_buffer.snapshot()
    if len(frames) < min_frames: continue
    dem = reconstruct_frames_to_dem(frames, prev_anchor=anchor, cfg)   # M1
    msg = dem_to_elevation_msg(dem...)                                  # M1.2
    publisher.publish(msg)                                             # M3.3
```
- T 现实值：单次 VGGT 推理耗时决定，先设 **每 3~10 秒一次**（准实时，非 30fps）。先测实际耗时定 T。
- 循环与帧泵各自线程，互不阻塞；重建耗时 > T 时跳过本轮（丢帧优于堆积）。

**M3.2 服务控制端点**（加到 `vggt_service.py`）
- `POST /stream/start`（入参：视频路径/源、T、窗口大小、tile 配置）→ 起帧泵+循环。
- `POST /stream/stop`、`GET /stream/status`（当前窗口大小、上次重建耗时、已发布计数）。

**M3.3 双通道发布者**（`streaming/elevation_publisher.py`）
- MQTT：paho-mqtt 连 broker，publish JSON 到 `01/map/elevation`（QoS 0，保留最新）。
- 文件：写 `<out_dir>/elevation_tile_0_0.json`（原子写：临时文件 + rename），供 `ElevationFileLoader` 轮询。
- 载荷体积：128×128≈100KB、512×512≈1MB。若走 512 且 broker 吃力，则 MQTT 发 128、文件发 512，或加压缩。

**M3.4 端到端验证**
- 起 broker → `/stream/start` 喂一段 mp4 → Unity 里地形每 T 秒刷新一次。
- 观察是否有"跳动/缩放"（大概率有——这正是 Milestone 4 要解决的）。

**验证**：Unity 地形随视频推进周期性更新；记录端到端延迟、每轮耗时、跳动幅度作为 M4 的基线。

---

## Milestone 4 — 跨更新坐标稳定（消除跳动/缩放）

**目标**：让相邻两轮 DEM 落在同一世界系里，Unity 地形平滑演化而非整体跳动。
命门问题，[[vertical-compression-rootcause-refs]] 里的尺度/垂直歧义在此集中体现。

**M4.1 冻结锚（先做，最简）**
- 第一轮成功重建后，冻结其 `R_align`（重力对齐旋转）+ `scale_factor` + 水平原点，存为 `anchor`。
- 后续轮次不再各自估计尺度，直接套用 `anchor` 的尺度与重力方向 → 至少消除"缩放跳动"。

**M4.2 轮间配准（消除平移/旋转漂移）**
- 每轮新点云先按 `anchor` 对齐，再与上一轮点云做水平配准（ICP 或基于重叠区域的 2D 平移+偏航估计）。
- 把估计到的 `T_cur→anchor` 应用到 DEM 的 x/z bounds，使 tile 内容对齐。
- 单 tile 模式下，配准后仍覆盖同一 tile；这一步保证覆盖前内容已对齐。

**M4.3（可选）RTK 地理配准**
- 若后续接真源，用 Unity 侧 `RtkGpsMsg` 把 anchor 绑到真实地理坐标，跨会话稳定。先留接口。

**验证**：连续多轮更新下，Unity 地形静止区域不再抖动；只有真正变化处（挖掘）在变。

---

## Milestone 5 — 升级到 B：持久增量融合（研究级）

**目标**：从"每轮重算整块"升级为"固定世界系里持续生长的全局 DEM，只发变化 tile"。
对接 [[scene-graph-icra-direction]] 的 4D 动态场景图。

**M5.1 持久全局 DEM 累加器**
- 固定世界系的规则栅格，每格维护 `(running_height, weight, last_update_t)`。
- 每轮重建的局部 DEM 经 M4 配准后融合进全局栅格（加权滑动平均；挖掘=高度下降，用时间衰减跟随）。

**M5.2 多 tile 化 + 变化检测**
- 全局栅格切成 Unity 的 tile 网格（`tile_x/tile_y`），只对本轮"变化超阈值"的 tile 发 `ElevationMsg`。
- 切到 Unity 的多 tile 模式（`singleFixedTile = false`，`TerrainTileManager` 已支持 3×3 滚动）。

**M5.3 小窗口 VGGT**
- 新关键帧只对小窗口跑 VGGT，靠融合累积全局形状，单轮成本有界 → 提高更新频率。

**验证**：地形随挖机作业连续演化；挖掘坑随时间正确加深；单轮耗时不随总时长增长。

---

## 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| VGGT 单轮推理耗时 | 更新频率上限（非 30fps） | T 设 3~10s；M5 用小窗口降本 |
| 跨轮尺度漂移 | 地形缩放跳动 | M4.1 冻结尺度 |
| **垂直压缩**（已知研究问题） | 挖掘深度被低估 | 沿用现有垂直校正实验；作为已知局限记录 |
| tile 分辨率不匹配 | 地形错位/拉伸 | M0.1 先量 `heightmapResolution` 并对齐 |
| MQTT 载荷过大（512×512） | broker 卡/丢 | 文件走高分辨率、MQTT 走低分辨率，或压缩 |
| MQTT 回调线程 vs Unity 主线程 | 崩溃/无效 | 已有 `MqttManager.Update` 出队机制，勿绕过 |

## 新增/改动文件清单

**Python（本项目，新增）**
- `streaming/frame_pump.py` — 帧泵（先 VideoFile，后 MediaMTX）
- `streaming/keyframe_buffer.py` — 在线关键帧滚动缓冲
- `streaming/pipeline.py` — `reconstruct_frames_to_dem`（M1 抽取）
- `streaming/reconstruct_loop.py` — 周期重跑循环（M3），M5 演化为融合循环
- `streaming/elevation_publisher.py` — MQTT + 文件双通道发布
- `tools/send_test_tile.py` — M0 冒烟工具
- `vggt_service.py` — 加 `/stream/start|stop|status` 端点
- 依赖：`paho-mqtt`

**Unity（改动，很小）**
- `Assets/map/MqttManager.cs::HandleElevation` — 路由改到 `TerrainTileManager.OnElevationTile`（M0.2）
- `Assets/Terrain/ElevationFileLoader.cs` — 加可选定时轮询目录（M0.2，备通道）

## 里程碑依赖顺序

```
M0（地基/修通）→ M1（管线函数化）→ M2（取帧）→ M3（周期循环+发布，端到端可演示）
                                                      → M4（坐标稳定，消跳动）
                                                              → M5（增量融合，研究级）
```
M0~M3 交付"可演示的实时联动"；M4 让它"稳"；M5 是研究级升级。

---

## 执行记录

### M0（进行中，2026-07-16）
- ✅ **M0.1 契约确认**：Unity 侧现有可用 tile JSON 为 **128×128**（→ `GroundTerrainData.heightmapResolution=129`）。
  说明 Python 侧默认 128 网格**本就匹配**，无需提到 512。`tile_size_meters` 单固定 tile 模式下由
  Unity Inspector 的 `tileSizeMeters`（默认 50m）决定（`ApplyToTerrain(keepSize:true)`），文件里的
  `tile_size_meters` 仅作参考。
- ✅ **M0.2 Unity 路由修复**：`Assets/map/MqttManager.cs::HandleElevation` 改为优先
  `TerrainTileManager.OnElevationTile`，回退 `HandleElevationMap`，与 `ElevationFileLoader.Apply` 一致。
  （唯一的 Unity 改动，最小侵入。）
- ✅ **M0.3 冒烟工具**：新增 `tools/send_test_tile.py`。走真实导出路径
  `elevation_export.dem_to_elevation_msg`，支持合成 DEM / 回放已有 JSON，双通道（file / mqtt），
  `--animate` 移动+加深坑洞以肉眼验证实时更新。
  - 文件通道：已验证，产出合法 `ElevationMsg`（128×128、int16、data 长度 16384）。
  - MQTT 通道：已用临时 broker 做发布→订阅回环，订阅端确认收到（格式正确）。
- ⏳ **待办（需人工）**：本机无 broker（未装 mosquitto，无 sudo）。**MQTT 端到端真验证需在 Unity 运行 +
  一个 broker**（Windows 侧跑 mosquitto，或任意可达 broker）。先用**文件通道**即可零依赖验证 Unity 收图。
- ✅ **文件通道自动刷新**：`Assets/Terrain/ElevationFileLoader.cs` 加了 `autoPoll` / `pollDirectory` /
  `pollInterval`，按间隔扫描目录、**仅对写入时间变化的 `*.json` 重应用**（跳过未变文件，避免重复
  `SetHeights`）。已验证 Python 工具的原子写每帧更新 mtime、不留半写文件 → 轮询变化检测成立。

### 依赖/环境影响说明（遵守 0b 约束）
- 新增 `paho-mqtt`（vggt50，纯增量，离线路径不依赖）。
- 测试期间误升 `websockets` 15.0.1 曾与 gradio 冲突，**已回滚为 14.2**；gradio 5.17.1 导入正常。
- `markupsafe`/`pillow` 的告警为**环境既有**，非本次引入，未改动。
- **未触碰任何离线端点/脚本行为。**

### 建议的 broker 方案（M3 前定）
- 开发/演示：Windows 侧装 mosquitto（`127.0.0.1:1883`，Unity `MqttManager` 默认即连它）。
- 或在**独立**环境跑纯 Python broker（勿装进 vggt50，避免污染离线依赖）。

### M1（完成，2026-07-16）
- ✅ 新增 `streaming/` 包 + `streaming/pipeline.py`，核心函数 `reconstruct_frames_to_dem(frames, ...)`
  与适配器 `dem_result_to_msg(...)`。
- **忠实复用策略**：`load_and_preprocess_images` 只吃路径 → 函数把内存帧写进临时 `images/` 目录，
  再调**现有** `_run_inference` / `estimate_gravity` / `apply_alignment_to_points` /
  `_select_ground_aligned` / `_build_elevation_grid` / `dem_to_elevation_msg`。流式与离线走同一套代码。
- **延迟导入契约**：`import streaming.pipeline` 不触发 torch / vggt_service 加载（已测），
  重型依赖只在真正调用时载入 → 意在 vggt_service 进程内复用常驻 `_model`。
- **默认值对齐离线**：conf_thres=50、prediction_mode="Depthmap and Camera Branch"、
  ground_percentile=20、grid=128、scale=1.0、height_resolution=0.01。
- ✅ **回归验证**（8 帧真实 workspace）：端到端 PASS。DEM 128×128 全格有限、`R_align` det=1.0、
  合法 `ElevationMsg`、无 int16 溢出。**离线路径零改动。**
- 📊 **关键耗时**（RTX 5070 Ti，8 帧）：模型冷启动 23s（一次性）；**常驻后稳态 ≈4.6–4.9s/次**。
  → **M3 的 T 取 5–8 秒现实**；确认"准实时（每几秒）而非帧率"。
- 🔎 **两个待 M4 处理的现象**：
  - `gravity_source=cloud_ransac`（未走轨迹平面，8 帧相机运动不足）→ 跨更新重力方向可能抖 → 需冻结锚。
  - 尺度任意（tile≈0.73m、高程幅度~0.5m）→ 叠加已知垂直压缩 → 需 M4 冻结/标定 `scale_factor`。
- `Anchor` dataclass 与 `prev_anchor` 参数已预留（M1 传 None，行为同离线），M4 直接接。

### M2（完成，2026-07-16）
- ✅ **`streaming/frame_source.py`**：`FrameSource` 抽象接口 + `VideoFileSource`（按墙钟节奏回放
  mp4 模拟实时流）+ `FrameListSource`（测试用）。`MediaMtxSource`（WHEP/RTSP 真源）留占位，
  换真源只改这一个文件。
- ✅ **`streaming/keyframe_buffer.py`**：`KeyframeBuffer` 线程安全滚动窗口。keep/skip 判据与离线
  `select_keyframe_indices(mode="similarity")` **同一套 ORB 度量**（320px 降采样、内点比×视差折扣），
  但改吃内存帧、在线增量（离线代码未动）。pump 线程 `offer()`、重建线程 `snapshot()`，各自加锁。
- ✅ **M2.3 验证**（无 GPU）：
  - 帧泵节奏正确：12 帧 @4fps → 2.81s（≈预期 2.75s），非瞬间读完。
  - 关键帧筛选正确：运动素材 8 帧 → 保留 6 跳过 2（相似度>阈值的被跳）；静止相机素材只留 1 帧（正确）。
  - 滚动淘汰正确：capacity=4、保留 6 → 窗口封顶 4、淘汰 2，计数一致。
  - 线程安全：pump 线程写 + 主线程并发 snapshot 无错。
- 🔧 **验证中修正**：`sim_thresh` 默认从 0.45 改为 **0.92**，与离线默认对齐（越高保留越多；0.45 对
  高重叠视频过苛，几乎留不下帧）。这正是 M2.3 该暴露的问题。
- **确认约束**：M2 import 不触发 torch/vggt_service（延迟契约保持）；未改任何离线文件。

### M3（完成，2026-07-16；MQTT 端到端待 broker）
- ✅ **`streaming/elevation_publisher.py`**：双通道发布者。file（原子写 `tmp`+`os.replace`，Unity 轮询
  不会读到半写文件）+ mqtt（paho，可选，无 broker 时报错而非崩）。
- ✅ **`streaming/reconstruct_loop.py`**：`ReconstructLoop` = pump 线程（帧泵→缓冲）+ recon 线程
  （每 T 秒 `snapshot()`→`reconstruct_frames_to_dem`→`publish`）。超时则跳过本轮不堆积；
  `LoopConfig`/`LoopStatus` + `start/stop/status`。
- ✅ **`streaming/endpoints.py`**：`/stream/start|stop|status`（APIRouter，单会话，重复 start 返回 409）。
- ✅ **挂载**：`vggt_service.py` 只加一段 `try: app.include_router(...)`（try 包裹，streaming 出错也
  绝不阻塞服务启动）。
- ✅ **M3.4 端到端（文件通道）PASS**：预热模型后跑 26s，自动完成 **2 个重建周期**（t≈10s、22s，间隔≈T=6s），
  每次写新 tile、高程范围随窗口变化（证明是新重建非重复）；稳态 3.76–4.8s/次（与 M1 一致）。
  循环健壮：冷加载 20s 内不崩不堆积，`min_frames` 门槛生效。
- ✅ **非侵入验证**：整体 `import vggt_service` 后离线端点（/reconstruct、/fit_elevation、
  /export_elevation_json、/terrain_analysis…）全部健在，/stream/* 正常挂载；endpoints 导入不拉重型依赖。
- ⏳ **待办**：MQTT 端到端需 broker（file 通道已完整验证）；接 Unity 观察实际刷新（见运行指引）。

### 运行指引（文件通道端到端联动）
1. 起服务：`python vggt_service.py`（模型常驻，避免冷加载吃掉首轮）。
2. 启流：`POST /stream/start {"video_path":"<mp4>", "file_out":"/mnt/d/.../live_elevation", "interval":6}`。
   （M4 默认开：`freeze_anchor=true, registration=true`；如需对照关掉可传 `false`。）
3. Unity：`ElevationFileLoader` 开 `Auto Poll`，`pollDirectory` 指向同一目录 → 地形每 ~6s 自动刷新。
4. 查看：`GET /stream/status`（含 `anchor_frozen` / `last_registered` / `last_reg_rmse` / `last_reg_yaw_deg`）；
   停止：`POST /stream/stop`。

### M4（完成，2026-07-17；真 GPU 冒烟待跑）
**目标达成**：相邻轮 DEM 落在同一世界系，静态地形不再跳/转/缩放，只有真正变化处（挖掘）在动。

- ✅ **M4.1 冻结锚（`streaming/pipeline.py`）**：`Anchor` 从占位升级为完整冻结引用——
  `R_align`（重力）+ `scale_factor`（尺度）+ `x/z_bounds`（**footprint**）+ `ref_ground_xyz`（配准参考）。
  `Anchor.from_result()` 在首轮成功后一次性冻结。后续轮次：
  - 复用锚点重力/尺度（`gravity_source="anchor"`，跳过每轮重估 → 消除重力抖 + 缩放跳动）；
  - 新增 `_build_grid_fixed_bounds()`：在**冻结的** footprint 上栅格化（离线 `_build_elevation_grid`
    未动，这是流式专用孪生），→ tile 不再平移/缩放。
- ✅ **M4.2 轮间配准（`streaming/registration.py`，新增）**：`register_horizontal()` 估计
  yaw + 水平平移的 2D 刚体变换，把当前轮地面点对齐到锚点参考地面点。
  - **关键设计**：对应关系在 **3D**（X,Y,Z）里找——地形的可辨识结构在**高程 Y** 上，(X,Z) 分布
    只是 footprint、几乎无特征，纯 2D ICP 的 yaw 严重欠约束；但求解只解 yaw+XZ 平移（Y 已被锚点固定）。
  - **两段式**：先密集"粗 yaw 扫描"（每 6°，单次 NN 查询）定位 ICP 收敛盆地（ICP 自身盆地仅 ~10° 宽，
    45° 间隔的种子会漏），再从最优几个种子跑截断 ICP（trimmed，对挖掘/场景变化造成的部分重叠鲁棒）。
- ✅ **锚点生命周期（`streaming/reconstruct_loop.py`）**：首轮成功后 `Anchor.from_result` 冻结，
  之后每轮 `prev_anchor=self._anchor`；`LoopConfig` 加 `freeze_anchor`/`register`，`LoopStatus`/`/stream/status`
  加 `anchor_frozen`/`last_registered`/`last_reg_rmse`/`last_reg_yaw_deg`。
- ✅ **脱 GPU 单元验证**（`vggt50` 环境，无需模型）：
  - **配准恢复**：7 个合成用例（yaw 全圆、平移、drop 50% 部分重叠）→ yaw 误差 <0.02°、平移精确、
    ~0.23s/次（相对 ~5s 一轮可忽略）。粗扫+ICP 两段式修好了 45°-种子漏盆地的问题。
  - **冻结 footprint 网格**：固定 bounds 栅格化正确、128×128 全格有限。
  - **锚点生命周期**：驱动真 `ReconstructLoop`（stub 掉 GPU 重建）→ 首轮 `prev_anchor=None`、
    之后全部非 None、`anchor_frozen=True`、`last_registered=True`、状态字段齐全。
  - **决定性对照**：同一静态地形从"新 frame-0 位姿"（yaw 85°+平移）观察，
    **仅冻结 footprint 不配准 → DEM 跨轮 RMS 0.49m**（可见跳/转）；**加 M4.2 配准 → 0.0000m**（地形钉住）。
- ✅ **非侵入**：改动全在 `streaming/`（`pipeline`/`reconstruct_loop`/`endpoints` + 新 `registration`）；
  离线文件零改动；`import streaming.*` 仍不触发 torch（延迟契约保持）；pydantic 字段避开 `register`
  与 `BaseModel.register` 撞名（请求字段用 `registration`，内部参数仍 `register`）。
- ⏳ **待办**：真 GPU 跑一段视频冒烟（确认 `last_registered=true`、`rmse` 合理、Unity 里地形稳定）；
  M4.3 RTK 地理配准留接口未做。

### M5（完成 M5.1+M5.2，2026-07-20；真 GPU 冒烟待跑）
**目标达成**：从"每轮重算整块单 tile"升级为"固定世界系里持续生长的全局 DEM,只发变化 tile"。
**opt-in**:`fusion=false`(默认)时行为与 M4 完全一致;`fusion=true` 时走融合多 tile 路径。

- ✅ **M5.1 持久融合累加器（`streaming/global_dem.py`,新增）**:`GlobalDem` = 固定世界系规则栅格,
  每格 `(H 高度, W 权重, T 时间戳)`。`integrate(ground_xyz, t)`:
  - 复用 M4 成果——配准后 `res.ground_xyz` 已在锚点固定世界系,直接 splat 进全局栅格,坐标天然对齐;
  - 每格取落点**高百分位**(默认 70%,抗噪且对挖掘敏感)作本轮观测;
  - **时间衰减加权融合**:旧权重每轮 ×`decay`(默认 0.5),`H ← (H·W_decayed + h_obs·w_obs)/(W_decayed+w_obs)`。
    这让挖掘(高度下降)在 ~2-3 轮内被新观测主导,而非被历史高度卡住(纯 EMA 会把坑填平)。
  - 栅格化用 `_cell_index` + argsort 分组,镜像 `terrain_analysis.rasterize_bev`(不 import 它以免拉依赖)。
- ✅ **M5.2 多 tile 切分 + 变化检测**:`changed_tiles()` 把全局栅格按 `tile_size_m` 切成 Unity tile,
  与每 tile **上次发布快照**做差,`max|Δh| > change_thresh`(默认 0.05m)或新增有效格才发。
  - **tile 边界严格对齐 Unity**:全局栅格原点 snap 到 `tile_size_m` 整数倍,tile 世界原点 = `tile_index*tile_size_m`
    (= Unity `TileToWorldOrigin`)。首版几何 bug(原点 -75 不在 tile 边界→tile 全体半格错位)已在单元测试中抓出并修正。
  - 静态 tile 不发 → 省带宽、Unity 只刷新动的地方。tile_x/tile_y 可负(锚点在全局中心),Unity/文件名均支持。
- ✅ **接线(`reconstruct_loop.py` + `endpoints.py`)**:`LoopConfig.fusion` + 融合调参;
  `_recon_loop` 成功一轮后按 `fusion` 分流 `_publish_single_tile`(M4 原路)/`_publish_fusion`(M5);
  首轮冻结 anchor 后用其 footprint 中心 new `GlobalDem`;`LoopStatus`/`/stream/status` 加
  `fusion_enabled`/`observed_cells`/`tiles_published_total`/`last_changed_tiles`。
  某轮 `ground_xyz` 为空则跳过融合本轮(记 warning),不崩。**`elevation_publisher.py` 零改动**(已支持多 tile 命名)。
- ✅ **Unity 侧零 C# 改动**:多 tile 已被 `TerrainTileManager.OnElevationTile`/`ElevationTileStore`/
  `ElevationFileLoader.PollOnce`(扫全部 *.json)完整支持;**唯一操作**是 Inspector 设 `singleFixedTile=false`。
- ✅ **脱 GPU 验证**(`tools/verify_m5.py --unit`,4 项全 PASS):
  1. **静态平面收敛**:密集覆盖下 pass0 发布、之后无 tile 再判变化(变化检测不误报)。
  2. **加深坑快跟随**:坑 cell 融合 H = `0.997→0.997→0.719→0.627→0.591`(观测底 0.5),非增、跌>0.3m、追上;
     坑所在 tile 每轮判变化,远处静态 tile 从不判变化。
  3. **多 tile 对齐**:world(60,5) 落进 tile(1,0),其世界原点 =(50,0) 精确对齐 Unity。
  4. **NODATA**:未观测格导出为 NODATA(16383/16384),不溢出 int16。
- ✅ **集成验证**(驱动真 `ReconstructLoop`,stub GPU):`fusion=true` 5 轮 → 发布 2 个不同 tile
  `(0,0)`+`(1,0)`、`observed_cells=19848`、`tiles_published_total=10`;`fusion=false` → 单 `elevation_tile_0_0.json`、
  `fusion_enabled=false`(**M4 路径完全不受影响**)。
- ✅ **非侵入**:`import streaming.global_dem`/`streaming.endpoints` 不触发 torch;`/stream/*` 三路由健在;
  离线端点零改动;无 pydantic 撞名警告。
- ⏳ **待办**:真 GPU 冒烟 `python tools/verify_m5.py <mp4>`(确认多 tile 发布、挖掘 tile 反复变化、静态段收敛);
  接 Unity 目视多 tile 生长;M5.3 小窗口 VGGT 提频留 future。

### M5 运行指引（文件通道，多 tile 融合）
1. 起服务:`python vggt_service.py`(模型常驻)。
2. 启流(**加 `"fusion":true`**):
   `POST /stream/start {"video_path":"<mp4>","file_out":"/mnt/d/.../live_elevation","interval":6,"fusion":true}`。
   可选调参:`world_size_m`(默认150)、`tile_size_m`(默认50,须匹配 Unity)、`fusion_decay`(默认0.5)、`change_thresh`(默认0.05)。
3. Unity Inspector:`TerrainTileManager.singleFixedTile=false`、`activeRadius=1`(3×3);
   `ElevationFileLoader` 开 Auto Poll 指向 `file_out`(它会扫描并各自 apply 所有 `elevation_tile_*_*.json`)。
4. 观察:地形随挖机移动**多 tile 持续生长**、只有变化 tile 刷新、挖掘坑随时间加深。
5. `GET /stream/status` 看 `tiles_published_total`/`last_changed_tiles`/`observed_cells`;`POST /stream/stop`。

