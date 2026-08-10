# `elevation_plane.py` 函数解析

本文档逐个说明 `elevation_plane.py` 中函数的输入、输出、用途和调用关系。该模块提供在重力对齐坐标系（Y 轴为高程）中构建 DEM 所需的 helper，供高程查看器接口 `/elevation_viewer_data` 和流式管线复用。模块本身不导出 GLB。

## 模块整体流程

调用方（`vggt_service.py:/elevation_viewer_data` 与 `streaming/pipeline.py`）按以下顺序使用这些 helper：

```text
_extract_points_with_conf
  -> gravity_alignment.estimate_gravity
  -> gravity_alignment.apply_alignment_to_points
  -> _select_ground_aligned  (或 _select_ground_aligned_mask)
  -> build_elevation_view_grid
```

其中，`gravity_alignment.estimate_gravity()` 和 `apply_alignment_to_points()` 来自 `gravity_alignment.py`，不在本文件内定义。

---

## 1. `_extract_points_with_conf()`

```python
def _extract_points_with_conf(
    predictions: dict,
    conf_thres: float,
    prediction_mode: str
)
```

### 作用

从 VGGT 的 `predictions` 字典中取出点云、置信度，并可选地取出语义地面 mask。输出的点仍处于原始 VGGT world 坐标系，还没有做重力对齐。

### 输入

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `predictions` | `dict` | VGGT 推理结果，通常由 `predictions.npz` 读取而来。 |
| `conf_thres` | `float` | 置信度阈值，按百分比输入，例如 `50.0` 表示保留置信度大于最大置信度 50% 的点。 |
| `prediction_mode` | `str` | 点云来源模式。若为 `"Pointmap Branch"`，使用 `world_points`；否则使用 `world_points_from_depth`。 |

### 依赖的 `predictions` 字段

当 `prediction_mode == "Pointmap Branch"`：

```text
world_points
world_points_conf
```

其他情况：

```text
world_points_from_depth
depth_conf
```

可选字段：

```text
semantic_masks
```

如果 `semantic_masks` 的形状是 `(S, H, W)`，函数会把 `semantic_masks == 1` 的像素标记为地面点。

### 输出

返回三元组：

```python
(pts_world, conf_world, ground_world)
```

| 返回值 | 形状 | 说明 |
| --- | --- | --- |
| `pts_world` | `(N, 3)` | 过滤后的点云坐标，仍在原始 VGGT world frame。 |
| `conf_world` | `(N,)` | 与点一一对应的置信度。 |
| `ground_world` | `(N,)` 或 `None` | 与过滤后点一一对应的布尔地面 mask；无语义 mask 时为 `None`。 |

### 关键逻辑

1. 按 `prediction_mode` 选择点云来源。
2. 检查点云必须是 `(S, H, W, 3)`。
3. 将点云展平成 `(N, 3)`。
4. 将置信度展平成 `(N,)`；如果没有置信度，则全部设为 `1`。
5. 若存在 `semantic_masks`，把 ID 为 `1` 的像素视作地面。
6. 过滤无穷、NaN 点。
7. 按置信度阈值过滤。

### 可能抛出的异常

- 没有找到点云字段时：`ValueError("No world points found in predictions.")`
- 点云不是四维 `(S, H, W, 3)` 时：`ValueError("Expected (S, H, W, 3) point map, got shape ...")`

### 调用位置

`vggt_service.py:/elevation_viewer_data` 和 `streaming/pipeline.py`。

---

## 2. `_select_ground_aligned_mask()`

```python
def _select_ground_aligned_mask(
    points_aligned: np.ndarray,
    ground_mask: np.ndarray | None,
    ground_percentile: float,
    band: float = 0.05
) -> np.ndarray
```

### 作用

在重力对齐坐标系（Y 为高程）中，返回一个布尔 mask，标记哪些源点被选作地面候选。把筛选逻辑集中在一处，便于查看器在过滤前后可视化对比。

### 关键逻辑

- 有语义地面 mask 且点数足够（≥ 50）：取这些点 Y 中位数附近 `±band·(P98−P2)` 窄带内的点；窄带内点数不足时退回整组语义地面点。
- 否则：取 Y 最低的 `ground_percentile` 百分位点。

### 调用位置

`_select_ground_aligned()`（下）以及 `vggt_service.py:/elevation_viewer_data` 中的地面点可视化与终点筛选。

---

## 3. `_select_ground_aligned()`

```python
def _select_ground_aligned(
    points_aligned: np.ndarray,
    ground_mask: np.ndarray | None,
    ground_percentile: float,
    band: float = 0.05
) -> np.ndarray
```

### 作用

`_select_ground_aligned_mask()` 的点集版本：用 mask 直接返回筛选后的地面候选点 `(M, 3)`，用于 DEM 插值。

### 调用位置

`streaming/pipeline.py`（离线路径已改为直接在查看器接口内使用 mask 版本）。

---

## 4. `build_elevation_view_grid()`

```python
def build_elevation_view_grid(
    ground_pts: np.ndarray,
    all_pts: np.ndarray,
    grid_resolution: int
)
```

### 作用

构建 `/elevation_viewer_data` 使用的 DEM。在对齐坐标的 `(X, Z)` 平面上，用 `scipy.interpolate.griddata` 对地面点的 `Y` 做规则网格插值。

### 关键逻辑

1. 用全部对齐点的 `X`、`Z` 范围确定覆盖区域，两端各加 **2%** padding（区别于历史 GLB 导出路径的 5%）。
2. 生成 `grid_resolution × grid_resolution` 网格（查看器固定 `128 × 128`）。
3. 线性插值为主，NaN 处用最近邻补齐。
4. 返回 `has_data` 掩码，标记哪些格元来自线性插值的有效区域；最近邻补齐的格元被查看器视为 NODATA。

### 输出

```python
(xx, zz, elev, has_data, (x_min, x_max), (z_min, z_max))
```

### 调用位置

`vggt_service.py:/elevation_viewer_data` 和 `streaming/pipeline.py`（后者用固定 footprint 复现同一插值以保证跨帧一致）。

---

## 函数关系速查表

| 函数 | 可见性 | 作用 | 调用方 |
| --- | --- | --- | --- |
| `_extract_points_with_conf` | 内部 | 提取点云 + 置信度 + 地面 mask | 查看器接口、流式管线 |
| `_select_ground_aligned_mask` | 内部 | 地面候选布尔 mask | `_select_ground_aligned`、查看器接口 |
| `_select_ground_aligned` | 内部 | 地面候选点集 | 流式管线 |
| `build_elevation_view_grid` | public | 构建查看器 DEM（2% padding、`has_data`） | 查看器接口、流式管线 |

## 当前实现中的几个细节

1. **地面语义 ID 硬编码为 1。** `elevation_plane.py` 与 `vggt_service.py` 都用 `semantic_masks == 1` 作为 ground mask。
2. **DEM 只在查看器请求时即时构建。** 固定 `128 × 128`、2% padding，随每次 `/elevation_viewer_data` 生成，不落地为文件。
3. **RANSAC 只用于重力估计。** DEM 本身不做平面 RANSAC 拟合；RANSAC 仅在 `gravity_alignment` 的重力方向级联估计中作为退化 fallback。
