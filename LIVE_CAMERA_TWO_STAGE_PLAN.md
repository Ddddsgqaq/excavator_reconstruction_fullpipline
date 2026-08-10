# 真实相机与两阶段建图实施说明

## 目标架构

系统按一个持久化 `MapSession` 分为两个阶段：

1. **初始化重建**：受控采集较大关键帧集，生成基准点云、DEM、世界坐标锚点、参考帧和质量报告。
2. **持续增量更新**：加载已确认的 `MapSession`，用小窗口局部重建定位到基准地图，只融合并发布可信变化 Tile。

相机取流、关键帧筛选和 VGGT 推理是两个阶段共享的基础设施，但初始化和增量更新拥有独立的策略、状态与验收条件。

## 本轮已落地

- `streaming/camera_source.py`
  - USB/V4L2、RTSP 和 HTTP/MJPEG 输入；IP Webcam 根地址自动解析到 /video。
  - 持续解码、最新帧覆盖，避免重建慢导致视频积压。
  - 目标 FPS 采样、断线重连、超时、状态统计和凭据脱敏。
- `streaming/source_factory.py`
  - 隔离输入源构造。
  - 保留旧版 `video_path` 调用协议。
- `streaming/endpoints.py`
  - `/stream/start` 支持 `video | usb | rtsp | http`。
  - `run_reconstruction=false` 可只验证取流，不调用 VGGT。
  - `/stream/frame.jpg` 提供后端最新帧预览。
  - `/stream/status` 提供连接、FPS、帧龄、重连和重建状态。
- `streaming/reconstruct_loop.py`
  - 取流诊断模式不创建重建线程。
  - 停止任务时主动释放相机。
  - 没有新增关键帧时不重复运行相同 VGGT 窗口。
- `vggt_service.py`
  - 在线和离线请求共享 VGGT 模型互斥锁，避免并发 GPU 调用影响原功能。
- `orchestrator.py`
  - 新增独立的“实时相机与在线建图”折叠面板。
  - 可连接预览、查看健康指标、停止任务和验证相机到 VGGT 滑窗闭环。

本轮没有改变 `/reconstruct`、`/fit_elevation`、YOLOE 或原 Gradio 上传工作流的请求和返回格式。


## MapSession 后续开发（已落地）

正式两阶段工作流已作为可选路径接入，原滑窗验证路径保持不变：

- `streaming/map_session.py`
  - v1 manifest、地图版本、锚点、当前 DEM、全局 DEM 和参考帧的原子保存/加载。
  - 显式校验状态转换，服务重启后可加载继续。
- `streaming/initialization.py`
  - 初始化帧数、点数、覆盖率和高程跨度质量门控；产物进入 `INIT_REVIEW`。
- `streaming/change_detection.py`
  - 配准 RMSE、覆盖率、变化面积比例和物理高度范围门控，并记录拒绝原因。
- `streaming/two_stage_loop.py`
  - 初始化只执行一次受控批量重建；增量阶段只融合、持久化和发布通过门控的变化。
- `streaming/session_api.py`
  - 新增 initialize、finalize、update、status 接口；原 `/stream/start` 请求协议不变。
- Gradio 实时面板
  - 保留“滑窗闭环验证”，另加初始化、人工通过/拒绝和可信增量更新控件。

状态机：

```text
IDLE → INITIALIZING → INIT_REVIEW → READY → UPDATING
                                         ↘ DEGRADED → REINIT_REQUIRED
```

正式增量更新从 `READY` 启动；`DEGRADED` 可进行受控恢复尝试。连续拒绝达到阈值后进入 `REINIT_REQUIRED`，同一目录可以重新初始化。每轮只有通过全部门控且确有变化时才推进地图版本并发布 Tile。

## 当前边界

- `/reconstruct`、`/fit_elevation`、YOLOE 和原 Gradio 上传协议未改变；新能力仅由 `/stream/session/*` 显式启用。
- 已在 `vggt50` 环境通过 14 项 CPU/兼容性测试和 API 路由契约检查。
- 尚未在本机任务中连接真实相机执行长时间 GPU 实测；默认门控阈值仍需结合现场尺度、相机运动和地形噪声标定。

## API 示例

正式两阶段初始化：

```json
POST /stream/session/initialize
{
  "source_type": "http",
  "source_uri": "http://192.168.31.132:8080/",
  "session_dir": "/path/to/workspaces/site_a",
  "initialization_frames": 12,
  "capacity": 12
}
```

质量报告通过后人工确认：

```json
POST /stream/session/finalize
{"session_dir": "/path/to/workspaces/site_a", "approved": true}
```

从已确认地图启动增量更新：

```json
POST /stream/session/update
{
  "source_type": "http",
  "source_uri": "http://192.168.31.132:8080/",
  "session_dir": "/path/to/workspaces/site_a",
  "file_out": "/path/to/workspaces/site_a/tiles",
  "min_frames": 4,
  "capacity": 12
}
```


只验证 HTTP/MJPEG（IP Webcam）取流：

```json
POST /stream/start
{
  "source_type": "http",
  "source_uri": "http://192.168.31.132:8080/",
  "target_fps": 3,
  "run_reconstruction": false
}
```

后端会自动使用 `http://192.168.31.132:8080/video`。也可以直接填写完整 MJPEG 地址。

只验证 RTSP 取流：

```json
POST /stream/start
{
  "source_type": "rtsp",
  "source_uri": "rtsp://user:password@camera:554/stream",
  "camera_backend": "auto",
  "target_fps": 3,
  "run_reconstruction": false
}
```

验证 USB 相机到 VGGT：

```json
POST /stream/start
{
  "source_type": "usb",
  "source_uri": "0",
  "target_fps": 3,
  "run_reconstruction": true,
  "file_out": "/home/maomaoyu/WS/vggt_yoloe/workspaces/live_test/tiles",
  "interval": 6,
  "min_frames": 4,
  "capacity": 12
}
```

旧版视频请求仍然有效：

```json
POST /stream/start
{
  "video_path": "/path/to/video.mp4",
  "file_out": "/path/to/output"
}
```
