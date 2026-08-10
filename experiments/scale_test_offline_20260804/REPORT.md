# scale_test.mp4 离线重建、尺度校准与体积实验报告

## 1. 结论摘要

本实验严格使用仓库原有离线链路：

> 视频抽帧 → YOLOE 语义分割 → VGGT 离线重建 → 语义点云分割 → 地面/重力对齐 → 高程栅格 → 15 cm 尺度标定 → 分物体体积估算

没有使用 streaming 在线视频流程。

| 项目 | 结果 |
|---|---:|
| 全局尺度 | **0.45616 m / VGGT unit** |
| 3 个完整尺子锚点的尺度 CV | **15.3%** |
| 尺度锚点相对极差 | **31.1%** |
| 对体积的一阶放大影响 | 约 **93.4%** |
| 全局尺度下尺子闭合 MAE / RMSE | **1.44 / 1.80 cm** |
| 静止尺子中心跨帧漂移，中位 / P95 | **4.32 / 9.12 cm** |
| 轨迹法与桌面语义法的法向夹角 | **19.9°** |
| 高置信地面核心 p2–p98 高程跨度，旧 / 新 | **47.0 / 21.5 mm** |
| 高置信地面核心 RMSE（新） | **5.54 mm** |
| 逐帧局部桌面平面 p2–p98 / RMSE 中位数 | **23.80 / 6.51 mm** |

三个盒子的高程积分体积：

| 物体 | 有效帧 | 中位尺寸 L×W×H | 体积中位数 | 帧间 IQR | 仅尺度锚点导致的体积范围 |
|---|---:|---:|---:|---:|---:|
| 立放红盒 | 7 | 7.2×5.1×8.3 cm | **70.9 mL** | 30.1–152.3 mL | 50.9–124.6 mL |
| 红色圆角扁盒 | 7 | 11.2×6.4×3.6 cm | **154.8 mL** | 122.1–228.9 mL | 111.0–272.0 mL |
| 蓝白纸巾盒 | 6 | 15.3×9.3×7.8 cm | **565.6 mL** | 420.4–599.3 mL | 405.6–993.6 mL |

这些体积是“语义实例足迹 + 顶部高程积分”的近似值，不是闭合网格真值。2026-08-08 已补充三个盒子的实测外形尺寸并完成独立偏差评估，详见 7.1；表中的帧间 IQR 和尺度范围仍应与中位数一起阅读。

## 2. 输入与抽帧

- 视频：scale_test.mp4
- 分辨率：540×960
- 帧率：30 fps
- 总帧数：302
- 时长：约 10.07 s
- 抽帧方式：复用 orchestrator.py 中 handle_uploads 的等价规则，每 0.5 s 一帧
- 离线重建帧数：20
- VGGT 分支：Depthmap and Camera Branch
- 点云置信度门限：全局第 50 百分位

![抽帧接触表](visualizations/contact_sheet.jpg)

画面中可见四个目标：透明 15 cm 直尺、立放红盒、红色圆角扁盒、蓝白纸巾盒。

## 3. 语义识别与实例分割

YOLOE-v8l 文本提示及 ID：

1. white tabletop → ID 1
2. transparent ruler → ID 2
3. upright red box → ID 3
4. red plastic storage box → ID 4
5. blue tissue box → ID 5

20 帧共得到 95 个文本检测。ID 1 必须留给地面/桌面，因为现有高程代码固定把语义 ID 1 当作 ground；实验过程中已显式检查并纠正这一约定。

![文本语义叠加](visualizations/text_semantic_contact_sheet.jpg)

文本提示对立放红盒只有 4 个高质量视角。使用原离线 /segment_visual 接口，以首帧实例框补做视觉提示分割，获得 13 个检测。最终测量掩码以文本掩码为底，并用视觉实例掩码覆盖 ID 3。

![立放红盒视觉提示](visualizations/upright_visual_contact_sheet.jpg)

透明尺类别会少量误检圆形灯光反射。尺度标定前使用连通分量面积、长宽比、长轴像素长度和画面边界检查，只保留完整且细长的尺子实例。

## 4. VGGT 离线重建与精度诊断

离线 /reconstruct 在 20 帧上生成 predictions.npz、语义 GLB、20 张深度图、20 张点图、20 张语义深度图和 20 张语义点图。重建输出包含 5,366,480 个逐像素三维点。

为便于直接比较，本节依次展示“网络逐帧输出 → 纯 RGB 几何 → 语义三维点云 → 高程表面”。后三类三维图在展示阶段统一应用后文得到的 15 cm 尺度和语义桌面重力对齐，因此坐标轴可按米制比较；点云几何本身仍来自同一个离线 `predictions.npz`，没有使用其他重建流程。

### 4.1 从图像到三维点图

![VGGT 离线重建阶段对照](visualizations/pipeline_stage_montage.png)

每一行对应同一个代表帧，从左到右分别是 RGB 输入、YOLOE 语义掩码、VGGT 深度、VGGT 三维点图和语义映射后的三维点图。该图主要用于检查深度边缘是否与物体边界一致，以及语义错误是否会直接传递到三维点云。

### 4.2 仅 VGGT 几何重建（不使用语义着色）

![纯 RGB 三维重建](visualizations/reconstruction_3d_rgb.png)

两个相反斜视角显示同一份 RGB 点云。桌面和三个盒子的整体空间关系已经形成，但物体边缘存在双层轮廓，桌面也不是完全平面；这些现象来自不同视角间的深度/位姿不一致，而不是语义配色造成的视觉错觉。

### 4.3 VGGT 重建叠加 YOLOE 语义

![语义三维重建](visualizations/reconstruction_3d_semantic.png)

该图严格复用上一张 RGB 图的高置信度筛选、12 万个采样点、坐标范围、两个观察角度、点大小和透明度，仅把同一批点的 RGB 颜色替换为 YOLOE 类别颜色。因此两图几何轮廓应完全一致，可以直接判断颜色变化来自语义投影，而不是点云筛选差异。深灰色为未标注点，浅灰色为桌面，蓝色为 15 cm 尺，紫色为立放红盒，红色为扁红盒，青色为纸巾盒。

![语义点云俯视图](visualizations/semantic_pointcloud_topview.png)

该俯视图继续严格复用前两张图的高置信度筛选和同一批 12 万个点（包含未标注点），只把观察方式改为 X-Z 顶视投影；没有按类别重新抽样，也没有排除背景点。相同颜色出现多个邻近簇，说明同一静止物体在不同视角的三维位置没有完全重合。

### 4.4 从点云到高程重建

![三维高程重建对照](visualizations/elevation_3d_views.png)

三幅图严格使用同一 128×128 网格、相同 X-Z 范围和同一重力对齐结果；左图为原生 `H_top` 表面，中图为同网格顶面相对地面的高程残差，右图将该网格中的三个语义物体叠加到地面 DEM。这里比较的是同一高程数据的派生层，而不是三份不同点云。该对比揭示：整体地面方向已被校正，但局部翘曲、孔洞和离群高度仍会影响后续物体高度与体积积分。

### 4.4b 网格化 DEM 高程图

上面 `elevation_3d_views.png` 是平滑着色的 `H_top` 曲面，不显示栅格单元。为得到真正带网格线的
数字高程模型（DEM），下图**严格复用主程序 `/elevation_viewer_data` 的 DEM 生成路径**：同一点源
`world_points_from_depth`、同一次 `gravity_alignment.estimate_gravity` 重力对齐、P50 置信度门限、
语义地面过滤（并剔除物体点），再交给 `elevation_plane.build_elevation_view_grid`（128×128 栅格、
2% padding、linear+nearest griddata）。高度用 15 cm 尺度换算成米并以桌面为零面。栅格约
**0.49 cm/格**，地面 DEM 有效栅格 8878 个。

- 左：地面 DEM。这条正是查看器所用路径，会剔除物体点，因此曲面就是桌面本身，反映重建平整度；
  可见本视频的桌面翘曲比 scale_test3 明显（与 6.3 中全量点 p2–p98 约 110 mm 的诊断一致）。
- 中 / 右：表面顶 DEM，来自主程序的另一 DEM 生产者 `terrain_analysis.rasterize_bev` 的 `H_top`
  （每格 P90，`export_elevation_json` 的 `dem_source='htop'`）。它保留物体点，所以三个盒子会从
  网格中凸起，这才是“有网格的高程图”而不只是点高度散点。右图为对应俯视 DEM 热力图加等高线，
  可直接读出三个盒子的位置与相对高度。

![网格化 DEM：地面 DEM 与表面顶 DEM](visualizations/reconstruction_3d_dem_grid.png)

由 `make_dem_figure.py` 生成，栅格契约记录在 `visualizations/dem_grid_manifest.json`。它与
`elevation_3d_views.png` 使用同一 128×128 网格与同一重力对齐，只是显式画出栅格线并给出俯视热力图。

### 4.5 跨帧重建精度诊断

静止尺子的跨帧诊断：

- 12 个可用尺子视角
- 中心相对中位中心的漂移中位数：4.32 cm
- P95 漂移：9.12 cm
- 所有可见尺子帧的长度中位数：11.49 cm
- 长度 IQR：2.11 cm

这里的漂移是多视图重建重复性指标，不是对外部测量系统的绝对位置误差。它说明当前 VGGT 输出存在明显帧间深度/位姿不一致，后续全局融合和体积估计会受影响。

## 5. 15 cm 尺度标定

尺度锚点仅采用完整、未触边、长宽比高的尺子连通分量。对每个分量沿二维主轴取两端各 2% 像素，分别读取三维点中位数，两端距离作为该帧 VGGT 长度。

| 帧 | 时间 | VGGT 长度 | 单帧尺度 | 用全局尺度还原的尺长 |
|---:|---:|---:|---:|---:|
| 13 | 7.0 s | 0.27253 unit | 0.55040 m/unit | 12.43 cm |
| 14 | 7.5 s | 0.32883 unit | 0.45616 m/unit | 15.00 cm |
| 15 | 8.0 s | 0.36737 unit | 0.40831 m/unit | 16.76 cm |

取单帧尺度中位数：**1 VGGT unit = 0.45616 m**。

![尺度稳定性](visualizations/scale_calibration_stability.png)

尺度锚点 CV 为 15.3%，极差相对中位数为 31.1%。由于体积按尺度三次方变化，仅这一项就对应约 93.4% 的一阶体积相对极差。因此当前结果可以用于量级估计和相对比较，不应表述为高精度计量结果。

## 6. 地面判定改进与高程精度

### 6.1 发现的问题

原 gravity_alignment.py 优先使用相机轨迹 PCA 法向；语义地面仅作一致性检查，即使两者相差超过 10°，仍固定使用轨迹法。

本视频是绕桌拍摄，轨迹平面与桌面平面相差 19.9°。桌面语义 RANSAC 则有约 19k 个高置信点、98.9%–99.0% 内点率。继续无条件使用轨迹法不合理。

### 6.2 已实施的改进

现在当以下条件同时满足时，语义地面覆盖轨迹法：

- 两种法向夹角超过 10°；
- 语义地面 RANSAC 内点数不少于 100；
- 内点率不少于 70%。

同时新增 ground_inlier_ratio、selection_reason、明确选择告警和合成倾斜轨迹回归测试。真实离线 terrain_analysis 已返回 gravity_source = ground_mask。

### 6.3 校准效果

![地面对齐对比](visualizations/ground_alignment_comparison.png)

旧轨迹法与新语义桌面法在完全相同的 19,366 个高置信桌面点上计算，未分别筛点。在这批重力 RANSAC 支持点上：

- 旧轨迹法 p2–p98 高程跨度：47.0 mm
- 语义桌面法：21.5 mm
- 降低：54.2%
- 语义桌面法 RMSE：5.54 mm

但若纳入全部第 50 百分位以上桌面点，单帧桌面仍有明显翘曲，语义重力修正后的 p2–p98 跨度中位数仍约 110.8 mm，RMSE 中位数约 32.7 mm。这说明全局方向误差已改善，但 VGGT 深度/位姿本身的低置信变形仍然存在。

需要特别说明：本次 `depth_conf` 有 74.62% 的值精确等于下限 1.0，因此 P50 阈值也等于 1.0，上述“第 50 百分位以上”实际保留了全部有限点，并没有删掉一半点。新增的置信度扫档实验从首次脱离该平台的 P76 开始分析。

为降低翘曲对物体高度的影响，体积计算又对每帧高置信桌面点做鲁棒局部平面拟合：

- 局部平面 p2–p98 残差中位数：23.80 mm
- 局部平面 RMSE 中位数：6.51 mm

严谨结论：高置信地面核心与局部平面上可达到约 5–7 mm RMSE；全部保留点的表面高程噪声仍可达厘米级到十厘米级，不能只给出统一的“毫米精度”而忽略置信度和局部翘曲。

### 6.4 不同置信度下的误差水平

该补充实验只读取同一份离线 `predictions.npz` 的 RGB 标准分支 `world_points_from_depth`，不重新重建，也不调用在线视频流程。所有档位使用同一个语义地面法向和同一 15 cm 尺语义掩码。

误差口径如下：

- 桌面高程：各帧桌面点到固定语义地面法向的残差，报告跨帧 RMSE 和 p2–p98 跨度中位数；这是平面度/高程误差代理，不是测量仪器给出的绝对高程误差。
- 尺度：完整尺子掩码固定二维 2%–98% 端点，高置信三维点拟合鲁棒尺轴模型并外推到固定端点；每个测试帧只用另外两帧标定尺度，避免自标定闭环。
- 配准：尺中心相对跨帧中位中心的漂移，仅表示重复性。随着阈值升高，可用帧数下降，因此不能把少量帧的低漂移直接解释为精度提升。

![不同置信度下的误差曲线](visualizations/confidence_error_sweep.png)

| 档位 | 实际场景保留 | 尺点保留 | 桌面 RMSE | 桌面 p2–p98 | 尺交叉验证 MAE | 完整尺锚点 | 尺中心漂移中位数 / 帧数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 100.00% | 100.00% | 32.72 mm | 110.80 mm | 2.53 cm | 3 | 4.17 cm / 12 |
| P76 | 25.38% | 22.94% | 25.66 mm | 88.76 mm | 2.58 cm | 3 | 1.21 cm / 6 |
| P78 | 22.18% | 15.62% | 25.19 mm | 86.10 mm | 2.63 cm | 3 | 1.03 cm / 5 |
| P80 | 20.05% | 10.76% | 23.94 mm | 85.60 mm | — | 2 | 0.61 cm / 4 |
| P84 | 16.00% | 4.93% | 21.87 mm | 77.73 mm | — | 0 | — / 1（无统计意义） |
| P88 | 12.00% | 1.13% | 18.81 mm | 68.31 mm | — | 0 | — / 1（无统计意义） |
| P90 | 10.00% | 0.22% | 15.29 mm | 55.63 mm | — | 0 | — / 0 |

![不同置信度的空间覆盖](visualizations/confidence_spatial_coverage.png)

结论：置信度对不透明桌面具有明显筛噪价值。P78 相比全量点将桌面 RMSE 降低约 23.0%，P90 降低约 53.3%，但分别只保留 13.67% 和 4.85% 的桌面点。尺度误差没有同步改善：尺交叉验证 MAE 从全量的 2.53 cm 变为 P78 的 2.63 cm；P80 起完整尺锚点不足 3 个，已不能交叉验证。透明尺被高阈值不成比例地删除，因此不应使用一个全局置信度阈值同时决定桌面高程与尺度标定。建议桌面/高程使用约 P78–P90 的高置信核心，而透明尺尺度继续使用独立的语义完整性与几何覆盖门控，不能直接套用桌面阈值。

## 7. 体积计算方法与结果

每个物体逐帧执行：

1. 从组合语义掩码取目标连通分量；
2. 去除触边或严重截断帧；
3. 使用该帧高置信桌面拟合局部地面平面；
4. 将物体点投影到局部水平面；
5. 用二维主轴确定物体足迹；
6. 构建 64×64 目标高程网格；
7. 每格取 90 百分位顶面高度；
8. 在语义足迹凸包内部做最近邻补洞；
9. 对顶面高度减局部桌面高度积分；
10. 乘以全局尺度三次方并换算为 mL。

![体积汇总](visualizations/object_volume_summary.png)

最终近似结果：

- 立放红盒：**70.9 mL**，帧间 IQR 30.1–152.3 mL
- 红色圆角扁盒：**154.8 mL**，帧间 IQR 122.1–228.9 mL
- 蓝白纸巾盒：**565.6 mL**，帧间 IQR 420.4–599.3 mL

IQR 反映分割、遮挡、视角、局部高程和 VGGT 重建变化；“仅尺度锚点范围”单独反映三个尺子锚点对尺度三次方的影响，两者不能简单相加。

### 7.1 实测盒子尺寸偏差

用户提供的外形尺寸按长×宽×高解释：立放红盒 3.9×2.7×8.5 cm、红色圆角扁盒 11×7.5×4.4 cm、蓝色盒子 15×9.5×6 cm。参考体积采用外包络 `L×W×H`；圆角、壁厚、凹槽和内部可用容积未扣除，因此这里评估的是外包络几何与重建结果的偏差。

![盒子真实尺寸偏差](visualizations/box_ground_truth_error.png)

| 物体 | 实测 L×W×H | 重建中位 L×W×H | 长偏差 | 宽偏差 | 高偏差 | 外包络真值 | 高程积分估算 | 体积偏差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 立放红盒 | 3.9×2.7×8.5 cm | 7.24×5.12×8.25 cm | +85.5% | +89.7% | −2.9% | 89.5 mL | 70.9 mL | **−20.8%** |
| 红色圆角扁盒 | 11×7.5×4.4 cm | 11.20×6.42×3.64 cm | +1.8% | −14.4% | −17.2% | 363.0 mL | 154.8 mL | **−57.3%** |
| 蓝色盒子 | 15×9.5×6 cm | 15.32×9.35×7.77 cm | +2.1% | −1.6% | +29.5% | 855.0 mL | 565.6 mL | **−33.8%** |

九个尺寸分量的总体 MAPE 为 27.18%，三个高程积分体积的 MAPE 为 37.32%；三物体外包络体积合计 1307.5 mL，重建积分合计 791.4 mL，总体偏差 −39.47%。只有立放红盒的外包络真值落在逐帧体积 IQR 内，另外两个盒子的真值均高于 IQR 上界。

误差结构并不一致：立放红盒的高度准确但水平足迹严重膨胀；扁盒长度准确但宽度与高度均偏小；蓝盒长宽准确但高度偏大。与此同时，三者的高程积分体积全部偏小。这说明当前主要问题不仅是统一尺度，而是语义足迹、局部高程、遮挡补洞和跨帧重影共同作用；尤其“外包络尺寸乘积”与“稀疏高程积分体积”不能互相替代。作为辅助诊断，重建中位 L×W×H 的乘积相对真值分别为 +241.7%、−27.9%、+30.1%，与高程积分体积的偏差方向也不总一致。

## 8. 原生高程输出

修复后的原离线高程接口已生成原生 terrain analysis、四层诊断图（H_top、残差、语义、坡度）和尺度标定 Htop DEM JSON。

![原生高程诊断](visualizations/native_terrain_analysis.png)

| 字段 | 值 |
|---|---:|
| 网格 | 128×128 |
| 高程量化 | 0.001 m |
| 横向标称分辨率 | 0.004873 m |
| X 跨度 | 0.62379 m |
| Z 跨度 | 0.75112 m |
| NODATA 单元 | 5,665 |
| 高程范围 | -0.69071 至 -0.12050 m |

绝对高程零点来自 VGGT 局部坐标系，没有外部海拔意义；相对高差和物体高度才是本实验使用的量。

## 9. 稳健性结论与后续建议

当前流程可以完成语义识别、重建、尺度转换、高程与分物体体积估算，但尚不满足精密计量要求。主要误差顺序为：

1. 多视图点云重影和位姿深度不一致；
2. 单帧尺度随视角变化；
3. 低置信桌面翘曲；
4. 透明尺与反光的语义混淆；
5. 遮挡导致物体顶面高程栅格不完整。

建议下一轮：

1. 在 VGGT 后增加基于静态桌面和尺子的 Sim(3) 或 bundle adjustment；
2. 用多个已知长度或 ArUco/棋盘格同时约束尺度与平面；
3. 高程计算保留并传播点置信度，地面 DEM 使用 RANSAC 内点和局部平面；
4. 对每个物体做跨帧实例关联，再做置信度加权表面融合；
5. 已有三个盒子的外形尺寸；下一步应补测圆角/凹槽后的真实排水体积或内部容积，以区分外包络误差与真实体积误差。

## 10. 过程图与快速目视检查

为便于不打开数值文件就完成第一轮质量判断，本实验额外固化了多张过程图。推荐先看总览，再按异常位置打开原分辨率单图。

![实验过程快速总览](visualizations/quicklook_dashboard.png)

- [流程阶段拼图](visualizations/pipeline_stage_montage.png)：同一帧并列查看 RGB 输入、语义掩码、VGGT 深度、三维点图和语义三维映射，可快速发现漏分割、错分割与深度边缘异常。
- [RGB 三维重建](visualizations/reconstruction_3d_rgb.png)：两个视角观察尺度标定并重力对齐后的整体点云，重点检查多视图重影、桌面翘曲和物体轮廓完整性。
- [语义三维重建](visualizations/reconstruction_3d_semantic.png)：以类别着色显示桌面、尺子和三个盒子，便于核对点云分割是否粘连或遗漏。
- [三维高程视图](visualizations/elevation_3d_views.png)：并列显示原生 H_top、高于地面的残差和物体覆盖在地面 DEM 上的结果；色条均保留物理单位。
- [物体局部高度云](visualizations/object_height_3d_views.png)：三个物体分别投影到逐帧鲁棒局部桌面，显示体积积分直接使用的高度分布、代表帧和估算尺寸/体积。
- [校准与体积摘要](visualizations/object_volume_summary.png)：集中查看尺子尺度稳定性、地面残差和各物体体积区间。
- [置信度—误差曲线](visualizations/confidence_error_sweep.png)：对比桌面高程误差、尺交叉验证误差、配准重复性与实际保留率。
- [置信度空间覆盖](visualizations/confidence_spatial_coverage.png)：在完全相同的基准抽样索引上观察 P78/P84/P88 对不同物体的非均匀删点。
- [盒子真实尺寸偏差](visualizations/box_ground_truth_error.png)：逐项比较实测长宽高、重建外包络和高程积分体积。
- [网格化 DEM](visualizations/reconstruction_3d_dem_grid.png)：复用主程序 DEM 路径的带网格高程图，地面 DEM（剔除物体）与表面顶 H_top DEM（保留三个盒子）并列，附俯视热力图。

完整图像索引与观察要点见 [visualizations/README.md](visualizations/README.md)。上述图可由 `make_process_visualizations.py` 从本实验固化数据重复生成。

## 11. 复现与产物

核心复现实验命令：

    /home/maomaoyu/miniconda3/envs/vggt50/bin/python experiments/scale_test_offline_20260804/analyze_scale_volume.py
    /home/maomaoyu/miniconda3/envs/vggt50/bin/python experiments/scale_test_offline_20260804/run_confidence_error_sweep.py
    /home/maomaoyu/miniconda3/envs/vggt50/bin/python experiments/scale_test_offline_20260804/evaluate_box_ground_truth.py
    /home/maomaoyu/miniconda3/envs/vggt50/bin/python experiments/scale_test_offline_20260804/make_dem_figure.py

关键产物：

- experiment_results.json：全部逐帧和汇总数值
- confidence_error_sweep.json / .csv：各置信度档逐指标结果与扁平表格
- confidence_error_sweep_manifest.json：置信度实验图、数据和 profile 清单
- box_ground_truth_evaluation.json / .csv：三个盒子的逐维度与体积偏差
- box_ground_truth_evaluation_manifest.json：真实尺寸评估图、数据和 profile 清单
- semantic_masks_combined.npz：文本 + 视觉提示组合掩码
- native_terrain_analysis.json：原生离线地形分析输出
- elevation_tile_scale_calibrated.json：尺度标定 Htop DEM
- make_dem_figure.py、visualizations/reconstruction_3d_dem_grid.png、visualizations/dem_grid_manifest.json：复用主程序 DEM 路径生成的网格化高程图（地面 DEM + 表面顶 H_top DEM）
- visualizations：全部图像
- visualizations/process_visualizations_manifest.json：过程图清单
- resource_profiles：逐阶段时间、CPU 内存与 CUDA 显存记录
- glbscene 文件：语义重建场景

性能记录字段、测量口径和汇总命令见项目根目录 `RESOURCE_PROFILING.md`。完整离线流程后可运行：

    /home/maomaoyu/miniconda3/envs/vggt50/bin/python tools/summarize_resource_profiles.py experiments/scale_test_offline_20260804

## 12. 本次性能复跑结果

2026-08-05 使用同一批 20 帧、相同 YOLOE 文本提示和 VGGT `Depthmap and Camera Branch` 完成真实性能复跑；2026-08-06 在固化重建数据上重跑了修正后的过程图生成；2026-08-07 补充置信度误差扫档；2026-08-08 补充盒子实测尺寸偏差。GPU 为 NVIDIA GeForce RTX 5070 Ti 16 GB。以下 CUDA “新增峰值”均相对阶段起点，能排除大部分常驻模型显存。

| 操作 | 总时间 | 最慢阶段 | 阶段时间 | CUDA 绝对峰值 | CUDA 新增峰值 |
|---|---:|---|---:|---:|---:|
| YOLOE 文本分割 | 6.83 s | 模型加载与文本提示编码 | 5.13 s | 1,355.6 MiB | 1,355.6 MiB |
| YOLOE 常驻后 20 帧推理/掩码 | — | 分割与掩码生成 | 1.14 s | 349.4 MiB | 109.3 MiB |
| VGGT 离线重建 | 23.35 s | VGGT 模型前向 | 17.03 s | 14,196.1 MiB | 9,337.7 MiB |
| 地形分析 | 34.32 s | 536 万点栅格化与分析 | 33.54 s | 常驻 4,805.5 MiB | 0 MiB |
| H_top 高程导出 | 38.28 s | 128×128 高程网格 | 37.61 s | 常驻 4,805.5 MiB | 0 MiB |
| 尺度/精度/体积后处理 | 5.18 s | 物体高度网格与体积积分 | 1.97 s | CPU | CPU |
| 过程图生成（最新） | 8.78 s | 语义三维重建图 | 2.06 s | CPU | CPU |
| 置信度误差扫档 | 4.47 s | 9 档误差计算 | 2.07 s | CPU（RSS 峰值 854.1 MiB） | CPU |
| 盒子真实尺寸偏差 | 0.30 s | 对比图渲染 | 0.29 s | CPU（RSS 峰值 102.0 MiB） | CPU |

关键判断：VGGT 前向是显存瓶颈，峰值约 13.86 GiB；地形和 H_top 高程步骤是 CPU 时间瓶颈，耗时几乎全部集中在数百万点的重复栅格统计，JSON 量化与写入仅约 0.006 s。地形/高程行显示的约 4.69 GiB 是同一 VGGT 服务进程保留的常驻模型显存，并非这两个 CPU 阶段新增的显存。

本次结构化记录位于 `resource_profiles/`：共 13 份 profile、84 个阶段，覆盖 8 类操作。过程图生成保留三次迭代记录；置信度扫档保留四次记录，其中一次为 JSON 序列化失败的诊断记录，其余记录保留尺子测量与无效样本门控的改进过程，表中采用最终成功结果。汇总见 `resource_profile_summary.json`，逐阶段明细见 `resource_profile_stages.csv`。

回归验证：

- 新增地面冲突选择测试：通过
- RGB/语义共享点集与未知语义 ID 安全着色测试：2 项通过
- 置信度扫档尺轴外推与留一交叉标定测试：3 项通过
- 真实尺寸有符号误差与零偏差立方体测试：2 项通过
- 现有 elevation-view DEM padding/holes 测试：通过
- 现有 M4 fixed-footprint 契约测试：通过
- gravity_alignment.py 与新增测试语法编译：通过

项目环境未安装 pytest，因此本次使用 `unittest` 直接执行相关纯函数测试；所有断言均通过。
