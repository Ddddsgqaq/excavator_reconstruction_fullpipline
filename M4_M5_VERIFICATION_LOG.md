# M4/M5 真 GPU 冒烟验证 — 资源与结果记录

> 单次真机验证的登记表,供之后生成完整可视化报告使用。
> 关联:`REALTIME_LINK_PLAN.md`(里程碑全文)、memory `realtime-link-progress`。

## 运行环境
- **日期**:2026-07-20
- **GPU**:NVIDIA RTX 5070 Ti(16GB)
- **conda 环境**:`vggt50`(`~/miniconda3/envs/vggt50/bin/python`)
- **工作目录**:`/home/maomaoyu/WS/vggt_yoloe`
- **模型**:VGGT 常驻加载(进程内),冷启动 ~20s;稳态 ~5s/轮

## 输入视频
- **文件**:`dji_fly_20260511_161858_0_1778489356002_video.mp4`(航拍,720×1280,1040帧@30fps,~35s,23MB)
- **为何选它**:相机真实运动 → 有视差(VGGT 多视重建必需)。手持/近静止相机
  (如 `dynamic_execave_video.mp4`)ORB 相似度恒 ~1.0,关键帧缓冲只留 1 帧,重建失败。
- **取帧率**:`--target-fps 1.0`(拉开帧间视差;3.0 时航拍前段仍太相似)。

---

## 测试 1 — M4 坐标稳定(`tools/verify_m4.py`)
- **命令**:
  ```
  python tools/verify_m4.py dji_fly_20260511_161858_0_1778489356002_video.mp4 \
      --seconds 40 --interval 6 --target-fps 1.0 --viz
  ```
- **做法**:同视频跑两遍——配准 ON / OFF——对比静态地形跨轮 DEM 漂移。
- **结果(PASS)**:
  - 首轮 `gravity_source=cloud_ransac`,后续轮全 `grav=anchor`(复用冻结锚点)✓
  - 配准收敛轮:yaw≈60.6°、rmse=0.0149m(收敛 1/3 轮;航拍相机持续移动,
    其余轮窗口与锚点重叠不足→配准未触发,降级纯锚点,防御正确)
  - **决定性对照**:跨轮 DEM RMS **OFF 0.0077m → ON 0.0056m**(配准降低漂移)
- **产出图**(`verify_m4_viz/`):
  - `m4_ON_registration.png` — 配准开,逐轮 DEM(pass1–4,共享色标)
  - `m4_OFF_footprint_only.png` — 只冻 footprint 不配准,逐轮 DEM
  - **注**:移动相机下每轮 footprint 本就不同,M4 的"静止不动"保证更适合用**数值**(RMS)
    而非这两张图佐证;图仅作过程留存。

---

## 测试 2 — M5 持久融合 + 多 tile(`tools/verify_m5.py`)
- **命令**:
  ```
  python tools/verify_m5.py dji_fly_20260511_161858_0_1778489356002_video.mp4 \
      --seconds 50 --interval 6 --target-fps 1.0 --auto-scale --viz --viz-dir ./verify_m5_viz
  ```
- **`--auto-scale`**:VGGT 尺度任意,先探一次重建测实际地面 X/Z 跨度(≈1.2),据此定
  `tile_size_m=0.80`、`world_size_m=2.40`(3×3 tile)。**不加则默认 150m 栅格太大 → 所有点挤进 1 格**
  (这是 M4 已记录的"尺度任意"局限的直接体现,真源接入需按真实米制标定)。
- **结果(PASS)**:
  - **多 tile**:发布 **6 个不同 tile**,累计 21 次 tile 更新
  - **地形持续累积**:`observed_cells` 逐轮 **11770 → 16633 → 19279 → 21007 → 25666**(生长)
  - **变化检测收敛**:changed-tiles/轮 = `[6,6,3,4,4,4,4,4]`(稳定到每轮 4 个)
- **产出图**(`verify_m5_viz/`):
  - `global_dem_pass{1..5}.png` — 每轮全局融合 DEM 俯视热力图(terrain 色标 + tile 边界网格),
    **pass1 稀疏(11770格)→ pass5 填满(25666格)** 肉眼可见地形生长。
  - `elevation_tile_{-1,0}_{-1,-2,-3}.json` — 6 个真实发布的 Unity ElevationMsg tile
    (`width=height=128`、`tile_size_meters≈0.80`、int16、含 NODATA)。多 tile 拼接的直接产物。

---

## tile JSON 字段样例(`elevation_tile_-1_-1.json`)
```
width=128 height=128 tile_x=-1 tile_y=-1
tile_size_meters≈0.7997  min_elevation≈-0.351  max_elevation≈-0.329  nodata_count=13309
```
- 尺度小(米制未标定)、大量 NODATA(单 tile 只被部分观测)均为**预期**,非缺陷。

## 证明了什么
- ✅ **M4**:冻结锚点 + 轮间配准在真机端到端跑通,数值上减小跨轮漂移。
- ✅ **M5**:全局融合累加器 + 多 tile 切分 + 变化检测在真重建数据上真正工作——
  地形随视频**持续生长并填充**、发布多个 tile、变化检测收敛。
- ⚠️ **尺度**:绝对尺度仍任意(VGGT 固有),`--auto-scale` 仅为演示适配;真源需米制标定。
- ⏳ **未覆盖**:接真 Unity 目视多 tile 拼接;MQTT 通道(缺 broker)。

## 复现要点
- 必用**有相机运动的航拍视频** + `--target-fps 1.0`。
- M5 必加 `--auto-scale`(否则点挤进 1 格)。
- 两脚本已支持从任意目录运行(自动把 repo root 加进 `sys.path`)。
- `--viz` 依赖 matplotlib(`vggt50` 内已装 3.10.8)。
