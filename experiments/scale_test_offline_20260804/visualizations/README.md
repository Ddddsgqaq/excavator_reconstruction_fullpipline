# `scale_test.mp4` 离线实验图像索引

## 建议第一眼顺序

1. [quicklook_dashboard.png](quicklook_dashboard.png)：六宫格总览，先判断流程是否存在明显分割、重影、高程或体积异常。
2. [pipeline_stage_montage.png](pipeline_stage_montage.png)：逐行对应帧 0、7、14、19；横向对比 RGB、语义、深度、三维点图、语义三维映射。
3. [reconstruction_3d_rgb.png](reconstruction_3d_rgb.png)：看整体几何是否成形，以及桌面/物体边缘是否有双层重影。
4. [elevation_3d_views.png](elevation_3d_views.png)：看 H_top 连续性、地面残差和语义物体相对 DEM 的位置。
5. [object_height_3d_views.png](object_height_3d_views.png)：看三个体积结果实际依赖的局部高度云。
6. [confidence_error_sweep.png](confidence_error_sweep.png)：看置信度升高时误差降低与有效样本流失之间的权衡。
7. [box_ground_truth_error.png](box_ground_truth_error.png)：直接查看三个盒子的实测尺寸和体积偏差。

## 图像说明

| 文件 | 内容 | 主要观察点 |
|---|---|---|
| [quicklook_dashboard.png](quicklook_dashboard.png) | 实验快速总览 | 是否需要进一步放大检查某一阶段 |
| [pipeline_stage_montage.png](pipeline_stage_montage.png) | 四个代表帧的五阶段对照 | 语义边界、深度突变、三维坐标连续性 |
| [reconstruction_3d_rgb.png](reconstruction_3d_rgb.png) | 尺度标定和重力对齐后的 RGB 点云 | 重影、桌面翘曲、物体完整性 |
| [reconstruction_3d_semantic.png](reconstruction_3d_semantic.png) | 与 RGB 图完全相同点集和视角的语义着色点云 | 逐点对比语义投影，检查类别粘连和漏标 |
| [elevation_3d_views.png](elevation_3d_views.png) | H_top、地面残差、语义物体叠加 DEM | 高程噪声、孔洞、物体是否落在合理地面上 |
| [object_height_3d_views.png](object_height_3d_views.png) | 代表帧局部平面上的物体高度云 | 高度离群点、足迹范围、顶面缺失 |
| [ground_alignment_comparison.png](ground_alignment_comparison.png) | 轨迹法与语义地面法对比 | 地面跨度和 RMSE 改善幅度 |
| [native_terrain_analysis.png](native_terrain_analysis.png) | 原生高程四层诊断 | H_top、残差、语义和坡度的一致性 |
| [scale_calibration_stability.png](scale_calibration_stability.png) | 15 cm 尺尺度锚点 | 不同帧尺度估计的波动 |
| [object_volume_summary.png](object_volume_summary.png) | 校准、高程与体积汇总 | 体积中位数、帧间 IQR 和尺度不确定性 |
| [semantic_pointcloud_topview.png](semantic_pointcloud_topview.png) | 与 RGB/语义三维图完全相同 12 万点的俯视投影 | 在不改变点集的条件下检查平面布局与跨帧重影 |
| [confidence_error_sweep.png](confidence_error_sweep.png) | 9 个置信度档的高程、尺度、配准与覆盖率 | 哪类误差随置信度改善，何时因样本不足而失效 |
| [confidence_spatial_coverage.png](confidence_spatial_coverage.png) | 同一基准索引在全量、P78、P84、P88 下的俯视覆盖 | 高置信筛选是否对桌面、透明尺和盒子产生类别偏置 |
| [box_ground_truth_error.png](box_ground_truth_error.png) | 实测尺寸、重建尺寸、外包络体积和高程积分体积 | 哪个尺寸方向失真，以及体积低估是否由同一方向误差解释 |

## 读图约定

- 三维重建和高程图已使用 15 cm 尺换算到米制，并应用语义桌面重力对齐。
- RGB 三维图、语义三维图和语义俯视图严格复用同一个高置信度 12 万点样本；三者只改变颜色或投影方向。
- 地面对齐前后严格使用同一批 19,366 个高置信桌面点；三幅高程图严格使用同一 128×128 X-Z 网格。
- 高程残差图以厘米显示；其余三维几何坐标按图中坐标轴单位读取。
- 灰色点表示桌面/背景，彩色点表示语义物体；类别颜色在同一组新过程图中保持一致。
- 这些图用于快速诊断，不替代 `experiment_results.json` 中的完整逐帧统计和 `REPORT.md` 中的不确定性说明。
- `process_visualizations_manifest.json` 记录共享样本的筛选规则、候选数、实际点数、置信度阈值、类别直方图和三幅消费者图，可用于机器复核。
- `confidence_error_sweep.json` 和 `.csv` 保存每档的实际保留率、有效帧数和误差指标；缺少 3 个完整尺锚点时尺度误差明确留空，不用零值替代。
- `box_ground_truth_evaluation.json` 和 `.csv` 保存用户实测尺寸、逐维度有符号误差、外包络体积及高程积分体积误差。
