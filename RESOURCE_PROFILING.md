# 离线流程时间与资源记录

离线流程默认启用阶段级性能记录。每次请求或实验脚本运行都会在当前工作目录下创建：

```text
<working_dir>/resource_profiles/<operation>_<UTC时间>_pid<进程号>.json
```

文件采用增量写入：每完成一个阶段就原子更新一次，因此服务异常退出时仍可保留已完成阶段和错误信息。相关 API 响应也会返回 `resource_profile_path`。

## 已覆盖流程

| 操作 | 主要记录阶段 |
|---|---|
| `offline_input_preparation` | 上传图片复制、视频解码与抽帧 |
| `yoloe_segment_text` | 输入发现、模型/文本提示、关键帧、分割与掩码、预览写入 |
| `yoloe_segment_visual` | 输入发现、视觉提示编码、关键帧、分割与掩码、预览写入 |
| `yoloe_segment_promptfree` | 词表构建、关键帧推理、语义掩码、预览写入 |
| `vggt_reconstruct` | 清理、图像预处理、VGGT 前向、位姿/深度反投影、语义融合、NPZ、GLB、诊断图 |
| `terrain_analysis` | 点云加载/过滤/重力对齐、语义映射、地形栅格化与分析 |
| `export_elevation_json` | 点云加载/对齐、高程网格、量化与 JSON 写入 |
| `scale_volume_analysis` | 语义合并、尺子定标、地面对齐、体积积分、配准诊断、绘图 |
| `process_visualizations` | 数据加载、尺度/重力转换和每张过程图生成 |

## 记录字段

每个 `stages[]` 条目包括：

- `wall_time_s`：墙钟时间；CUDA 可用时在阶段边界同步 GPU。
- `cpu_time_s`：当前进程消耗的 CPU 时间。
- `start/end.rss_mb`：阶段边界的进程常驻内存。
- `end.rss_peak_mb`：进程自启动以来的 RSS 高水位，并非该阶段独占峰值。
- `start/end.cuda_allocated_mb`：当前 PyTorch 进程存活张量占用。
- `start/end.cuda_reserved_mb`：PyTorch 缓存分配器保留显存。
- `end.cuda_peak_allocated_mb`：该阶段内 PyTorch 已分配显存峰值。
- `cuda_peak_extra_allocated_mb`：相对阶段起点新增的峰值显存。
- `end.cuda_device_used_mb`：整张 GPU 的已用显存，可能包含其他进程。

同时保留设备名称、总显存、阶段状态、错误文本、输入帧数/网格尺寸等元数据。

## 汇总与比较

```bash
python tools/summarize_resource_profiles.py <working_dir>
```

输出：

- `resource_profiles/resource_profile_summary.json`：每次操作的总时间、最慢阶段、RSS 基线/阶段增量，以及 CUDA 绝对峰值/新增峰值。
- `resource_profiles/resource_profile_stages.csv`：逐阶段平铺表，可直接用 Excel、Python 或绘图工具比较。

## 测量口径与开销

- VGGT/YOLOE 模型常驻服务显存包含在阶段起点值中；判断某阶段新增需求时优先看 `cuda_peak_extra_allocated_mb`。
- 同理，持久服务的 `process_lifetime_rss_peak_mb` 可能来自更早的重建请求；判断当前操作内存增长时看 `rss_baseline_mb`、`rss_stage_end_max_mb` 和 `rss_stage_delta_max_mb`。
- `cuda_device_used_mb` 是设备级指标，不应当作本程序独占显存；程序自身显存以 PyTorch allocated/reserved 字段为准。
- 为得到可信的 GPU 阶段耗时，记录器会在阶段边界执行 CUDA synchronize，因此开启剖析时会带来少量同步和 JSON 写入开销。
- 默认开启。若需要无剖析基准，可在启动服务前设置 `PIPELINE_RESOURCE_PROFILE=0`。
