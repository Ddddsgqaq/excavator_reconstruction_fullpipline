# `elevation_plane.py` 函数解析

本文档逐个说明 `elevation_plane.py` 中函数的输入、输出、用途和调用关系。该模块的核心目标是：从 VGGT 重建结果中提取点云，估计重力方向，将点云旋转到“Y 轴为高程”的坐标系，然后生成 DEM 高程网格并导出 GLB。

## 模块整体流程

`fit_elevation_to_glb()` 是唯一公开入口，内部按以下顺序调用 helper：

```text
fit_elevation_to_glb
  -> _extract_points_with_conf
  -> gravity_alignment.estimate_gravity
  -> gravity_alignment.apply_alignment_to_points
  -> _select_ground_aligned
  -> _build_elevation_grid
  -> _elevation_to_mesh
  -> _scene_with / _merge_with_existing
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

- 没有找到点云字段时：

```text
ValueError("No world points found in predictions.")
```

- 点云不是四维 `(S, H, W, 3)` 时：

```text
ValueError("Expected (S, H, W, 3) point map, got shape ...")
```

### 调用位置

只在 `fit_elevation_to_glb()` 中调用。

---

## 2. `_select_ground_aligned()`

```python
def _select_ground_aligned(
    points_aligned: np.ndarray,
    ground_mask: np.ndarray | None,
    ground_percentile: float,
    band: float = 0.05
) -> np.ndarray
```

### 作用

在重力对齐后的点云中筛选地面候选点。此时坐标系已经满足：`Y` 是高程轴，`(X, Z)` 是水平平面。

### 输入

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `points_aligned` | `np.ndarray`，形状 `(N, 3)` | 已经经过 `R_align` 旋转的点云。 |
| `ground_mask` | `np.ndarray` 或 `None` | 与 `points_aligned` 对应的布尔地面 mask。来自 `_extract_points_with_conf()`。 |
| `ground_percentile` | `float` | 无语义地面 mask 时，取最低多少百分位的点作为地面候选，默认外部传入 `20.0`。 |
| `band` | `float` | 有语义 mask 时，在地面点 Y 中位数附近保留的窄带比例，默认 `0.05`。 |

### 输出

```python
ground_pts
```

类型为 `np.ndarray`，形状 `(M, 3)`，表示筛选出的地面候选点。

### 关键逻辑

函数有两条路线：

#### 路线 A：有语义地面 mask

当 `ground_mask` 不为空，且其中至少有一个 True：

1. 取出语义标记为地面的点：

```python
gpts = points_aligned[ground_mask]
```

2. 如果地面点数量不少于 50，计算全点云 Y 的稳健范围：

```python
y_range = percentile(Y, 98) - percentile(Y, 2)
```

3. 计算语义地面点的 Y 中位数：

```python
y_med = median(gpts[:, 1])
```

4. 只保留中位数附近的窄带地面点：

```python
abs(Y - y_med) <= band * y_range
```

5. 如果窄带内仍有至少 50 个点，返回窄带点；否则返回全部语义地面点。

#### 路线 B：无语义 mask 或语义点不足

按 Y 的低百分位选择点：

```python
threshold = percentile(Y, ground_percentile)
ground_pts = points_aligned[Y <= threshold]
```

### 注意

这里的“低 Y 值”被当前代码当作地面候选 fallback。这与重力对齐后的 Y 方向解释有关：系统把 `R_align` 映射到 +Y，但实际场景中最低百分位点被当作地面候选使用。

### 调用位置

只在 `fit_elevation_to_glb()` 中调用。

---

## 3. `_build_elevation_grid()`

```python
def _build_elevation_grid(
    ground_pts: np.ndarray,
    all_pts: np.ndarray,
    grid_resolution: int
)
```

### 作用

基于地面候选点，在重力对齐坐标系的 `(X, Z)` 平面上构建规则 DEM 网格，并插值得到每个网格点的高程 `Y`。

### 输入

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `ground_pts` | `np.ndarray`，形状 `(M, 3)` | 地面候选点，通常来自 `_select_ground_aligned()`。 |
| `all_pts` | `np.ndarray`，形状 `(N, 3)` | 全部对齐后的点云，用于确定 DEM 覆盖范围。 |
| `grid_resolution` | `int` | 网格分辨率。`128` 表示生成 `128 x 128` 的 DEM。 |

### 输出

返回六元组：

```python
(xx, zz, elev, valid, (x_min, x_max), (z_min, z_max))
```

| 返回值 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `xx` | `(R, R)` | 每个 DEM 网格点的 X 坐标。 |
| `zz` | `(R, R)` | 每个 DEM 网格点的 Z 坐标。 |
| `elev` | `(R, R)` | 每个 DEM 网格点插值得到的 Y 高程。 |
| `valid` | `(R, R)`，bool | 线性插值有效区域。`True` 表示来自 `linear` 插值，`False` 表示后续用最近邻补上。 |
| `(x_min, x_max)` | tuple | DEM X 覆盖范围。 |
| `(z_min, z_max)` | tuple | DEM Z 覆盖范围。 |

### 关键逻辑

1. 使用全部点云 `all_pts` 的 X/Z 最小最大值确定 DEM 范围。
2. 在 X 和 Z 两个方向各增加 5% padding：

```python
x_pad = (x_max - x_min) * 0.05
z_pad = (z_max - z_min) * 0.05
```

3. 用 `np.linspace` 生成规则网格坐标。
4. 使用 `scipy.interpolate.griddata` 做两次插值：

```python
elev_linear = griddata(src_xz, src_y, (xx, zz), method="linear")
elev_nearest = griddata(src_xz, src_y, (xx, zz), method="nearest")
```

5. 优先使用线性插值，线性插值产生 NaN 的区域用最近邻补齐：

```python
elev = np.where(np.isnan(elev_linear), elev_nearest, elev_linear)
```

### 用法示例

```python
xx, zz, elev, valid, x_range, z_range = _build_elevation_grid(
    ground_pts=ground_pts,
    all_pts=pts_aligned,
    grid_resolution=128,
)
```

### 调用位置

只在 `fit_elevation_to_glb()` 中调用。

---

## 4. `_elevation_to_mesh()`

```python
def _elevation_to_mesh(
    xx,
    zz,
    elev,
    colormap_name: str = "terrain",
    elev_min=None,
    elev_max=None
) -> trimesh.Trimesh
```

### 作用

把 DEM 网格转换为 `trimesh.Trimesh` 三角网格，并根据高程值给每个顶点设置颜色。

### 输入

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `xx` | `np.ndarray`，形状 `(R, C)` | DEM 网格 X 坐标。 |
| `zz` | `np.ndarray`，形状 `(R, C)` | DEM 网格 Z 坐标。 |
| `elev` | `np.ndarray`，形状 `(R, C)` | DEM 网格 Y 高程。 |
| `colormap_name` | `str` | matplotlib colormap 名称，默认 `"terrain"`。 |
| `elev_min` | optional | 颜色归一化下限；不传时使用高程 2 分位数。 |
| `elev_max` | optional | 颜色归一化上限；不传时使用高程 98 分位数。 |

### 输出

```python
mesh
```

类型为 `trimesh.Trimesh`。

### 网格顶点

每个 DEM 网格点转成一个 3D 顶点：

```python
vertex = [x, elev, z]
```

代码中对应：

```python
verts = np.column_stack([xx.ravel(), elev.ravel(), zz.ravel()])
```

### 三角面生成方式

每个网格四边形被拆成两个三角形：

```text
i00 ---- i01
 |      / |
 |    /   |
i10 ---- i11
```

对应 faces：

```python
faces[0::2] = [i00, i10, i01]
faces[1::2] = [i10, i11, i01]
```

### 顶点颜色

1. 取高程值 `elev.ravel()`。
2. 用 `elev_min/elev_max` 或 2/98 分位数做归一化。
3. 调用 matplotlib colormap。
4. 转成 0 到 255 的 RGBA。

```python
norm = clip((Y - lo) / (hi - lo + 1e-8), 0, 1)
rgba = colormap(norm) * 255
```

### 用法示例

```python
mesh = _elevation_to_mesh(xx, zz, elev, colormap_name="terrain")
```

### 调用位置

只在 `fit_elevation_to_glb()` 中调用。

---

## 5. `_scene_with()`

```python
def _scene_with(mesh: trimesh.Trimesh) -> trimesh.Scene
```

### 作用

把单个 `trimesh.Trimesh` 包装成 `trimesh.Scene`，方便导出为 GLB。

### 输入

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `mesh` | `trimesh.Trimesh` | 高程 DEM 网格。 |

### 输出

```python
scene
```

类型为 `trimesh.Scene`。

### 关键逻辑

```python
s = trimesh.Scene()
s.add_geometry(mesh, node_name="elevation_plane")
return s
```

### 用法示例

```python
_scene_with(elev_mesh).export(file_obj=elev_only_path)
```

### 调用位置

在 `fit_elevation_to_glb()` 中用于导出：

- 高程单独 GLB：`*_only.glb`
- 当没有传入有效 `source_glb_path` 时，也用于导出 `*_merged.glb`

---

## 6. `_merge_with_existing()`

```python
def _merge_with_existing(
    source_glb_path: str,
    mesh: trimesh.Trimesh,
    R_align: np.ndarray,
    scale: float
) -> trimesh.Scene
```

### 作用

读取已有 GLB 场景，将高程 DEM mesh 叠加进去，返回合并后的 `trimesh.Scene`。

### 输入

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `source_glb_path` | `str` | 原始或已有点云 GLB 文件路径。 |
| `mesh` | `trimesh.Trimesh` | DEM 高程 mesh。 |
| `R_align` | `np.ndarray`，形状 `(3, 3)` | 重力对齐旋转矩阵。 |
| `scale` | `float` | 尺度系数，通常来自 `scale_factor`。 |

### 输出

```python
merged
```

类型为 `trimesh.Scene`，包含原始 GLB 场景和新增的 DEM mesh。

### 关键逻辑

1. 加载原始 GLB：

```python
base = trimesh.load(source_glb_path, force="scene")
```

2. 复制一份源场景：

```python
merged = base.copy()
```

3. 复制 DEM mesh，构造 4x4 变换矩阵：

```python
T = np.eye(4)
T[:3, :3] = R_align * scale
```

4. 把变换应用到 DEM mesh：

```python
overlay.apply_transform(T)
```

5. 将 DEM mesh 加入原始场景：

```python
merged.add_geometry(overlay, node_name="elevation_plane")
```

### 注意

源码注释说明：源 GLB 中已经包含 VGGT 的 display alignment，因此这里额外应用 `R_align * scale`，使 DEM mesh 叠加到点云上。

### 调用位置

只在 `fit_elevation_to_glb()` 中调用，条件是：

```python
if source_glb_path and os.path.exists(source_glb_path):
```

---

## 7. `fit_elevation_to_glb()`

```python
def fit_elevation_to_glb(
    predictions: dict,
    working_dir: str,
    source_glb_path: str = "",
    grid_resolution: int = 128,
    colormap: str = "terrain",
    ground_percentile: float = 20.0,
    use_ransac: bool = True,
    conf_thres: float = 50.0,
    prediction_mode: str = "Depthmap and Camera Branch",
    scale_factor: float = 1.0,
) -> dict
```

### 作用

这是本模块的公开 API。它完成从 VGGT predictions 到高程 GLB 文件的完整流程：

1. 提取有效点云。
2. 估计重力方向。
3. 将点云旋转到重力对齐坐标系。
4. 筛选地面候选点。
5. 构建 DEM 高程网格。
6. 转换为三角 mesh。
7. 导出高程 GLB 和合并 GLB。
8. 写入元数据 JSON。
9. 返回结果字典。

### 输入

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `predictions` | `dict` | 必填 | VGGT 输出数据。至少需要点云、置信度和 `extrinsic`。 |
| `working_dir` | `str` | 必填 | 输出文件所在目录。 |
| `source_glb_path` | `str` | `""` | 可选源 GLB 路径。存在时会生成点云 + DEM 合并场景。 |
| `grid_resolution` | `int` | `128` | DEM 网格分辨率，生成 `N x N` 网格。 |
| `colormap` | `str` | `"terrain"` | DEM 顶点颜色使用的 matplotlib colormap。 |
| `ground_percentile` | `float` | `20.0` | 无语义地面 mask 时，取最低多少百分位点作为地面候选。 |
| `use_ransac` | `bool` | `True` | 兼容旧 API，当前函数内直接 `del use_ransac`，不再使用。 |
| `conf_thres` | `float` | `50.0` | 置信度过滤阈值，百分比。 |
| `prediction_mode` | `str` | `"Depthmap and Camera Branch"` | 决定使用 `world_points_from_depth` 还是 `world_points`。 |
| `scale_factor` | `float` | `1.0` | 尺度校准系数，应用在重力对齐后的点云上。 |

### 依赖的 `predictions` 字段

主要字段：

```text
extrinsic
world_points_from_depth
depth_conf
```

当 `prediction_mode == "Pointmap Branch"` 时，点提取阶段还会用：

```text
world_points
world_points_conf
```

可选字段：

```text
semantic_masks
```

### 输出

返回字典：

```python
{
    "elev_only_path": str,
    "merged_path": str,
    "gravity_source": str,
    "n_grav": list,
    "R_align": list,
    "scale_factor": float,
    "warnings": list,
    "log": str,
}
```

| 字段 | 说明 |
| --- | --- |
| `elev_only_path` | 高程 mesh 单独 GLB 文件路径。 |
| `merged_path` | 合并场景 GLB 文件路径。 |
| `gravity_source` | 重力估计来源，如 `trajectory`、`ground_mask`、`cloud_ransac`。 |
| `n_grav` | 原始 VGGT world 坐标系中的重力向上方向。 |
| `R_align` | 将 `n_grav` 映射到 +Y 的旋转矩阵。 |
| `scale_factor` | 实际使用的尺度系数。 |
| `warnings` | 重力估计过程中的警告。 |
| `log` | 简短日志字符串，供 UI 显示。 |

### 输出文件

文件名前缀：

```python
tag = f"elev_r{grid_resolution}_{colormap}_aligned"
```

导出文件：

| 文件 | 说明 |
| --- | --- |
| `{tag}_only.glb` | 只包含 DEM mesh 的 GLB。 |
| `{tag}_merged.glb` | 包含原始 GLB + DEM 的合并 GLB；如果没有源 GLB，则只包含 DEM。 |
| `{tag}_meta.json` | 重力估计和对齐元数据。 |

元数据 JSON 包含：

```json
{
  "gravity_source": "...",
  "gravity_inliers": 0,
  "n_grav": [0, 1, 0],
  "R_align": [[...], [...], [...]],
  "scale_factor": 1.0,
  "warnings": [],
  "debug": {}
}
```

### 内部执行步骤

#### Step 1：提取点云

```python
pts_world, conf_world, ground_world = _extract_points_with_conf(
    predictions, conf_thres, prediction_mode
)
```

如果有效点少于 100 个，直接报错。

#### Step 2：估计重力方向

```python
grav = estimate_gravity(
    extrinsic=extrinsic,
    world_points=raw_pts,
    ground_mask=gmask_3d,
    conf=raw_conf,
    conf_thres=conf_thres / 100.0,
)
```

`estimate_gravity()` 内部会使用轨迹 PCA、地面 mask RANSAC、全点云 RANSAC 级联策略。

#### Step 3：重力对齐和尺度缩放

```python
pts_aligned = apply_alignment_to_points(pts_world, grav.R_align) * scale_factor
```

得到以 Y 为高程轴的新坐标。

#### Step 4：筛选地面候选点

```python
ground_pts = _select_ground_aligned(
    pts_aligned, ground_world, ground_percentile
)
```

如果地面候选点少于 50 个，直接报错。

#### Step 5：构建 DEM

```python
xx, zz, elev, _valid, _, _ = _build_elevation_grid(
    ground_pts, pts_aligned, grid_resolution
)
```

#### Step 6：构建 mesh

```python
elev_mesh = _elevation_to_mesh(xx, zz, elev, colormap)
```

#### Step 7：导出 GLB

```python
_scene_with(elev_mesh).export(file_obj=elev_only_path)
```

如果 `source_glb_path` 有效，则导出合并场景：

```python
_merge_with_existing(...).export(file_obj=merged_path)
```

否则 `merged_path` 也导出单独 DEM scene。

#### Step 8：写入 metadata

```python
json.dump(meta, f, indent=2)
```

### 可能抛出的异常

- 有效点云太少：

```text
ValueError("Too few valid points (...)")
```

- 重力估计失败时，会由 `estimate_gravity()` 抛出异常。

- 地面候选点太少：

```text
ValueError("Too few ground candidates after alignment (...)")
```

### 调用位置

在 `vggt_service.py` 的 `/fit_elevation` 接口中调用：

```python
result = fit_elevation_to_glb(
    predictions=predictions,
    working_dir=req.working_dir,
    source_glb_path=req.source_glb_path,
    grid_resolution=req.grid_resolution,
    colormap=req.colormap,
    ground_percentile=req.ground_percentile,
    use_ransac=req.use_ransac,
    conf_thres=req.conf_thres,
    prediction_mode=req.prediction_mode,
    scale_factor=req.scale_factor,
)
```

前端 Gradio 的 `Elevation Plane` tab 最终会通过 `orchestrator.py` 调用 VGGT 服务的 `/fit_elevation`。

---

## 函数关系速查表

| 函数 | 类型 | 输入重点 | 输出重点 | 用途 |
| --- | --- | --- | --- | --- |
| `_extract_points_with_conf` | private helper | `predictions`, `conf_thres`, `prediction_mode` | 点云、置信度、地面 mask | 从 VGGT 结果中提取有效点 |
| `_select_ground_aligned` | private helper | 对齐后点云、地面 mask、百分位 | 地面候选点 | 给 DEM 插值提供地面采样 |
| `_build_elevation_grid` | private helper | 地面点、全点云、网格分辨率 | `xx`, `zz`, `elev`, `valid` | 构建 DEM 高程图 |
| `_elevation_to_mesh` | private helper | DEM 网格、高程、colormap | `trimesh.Trimesh` | 把 DEM 变成彩色三角网格 |
| `_scene_with` | private helper | mesh | `trimesh.Scene` | 包装 scene 并导出 GLB |
| `_merge_with_existing` | private helper | 源 GLB、mesh、旋转、尺度 | 合并后的 scene | 把 DEM 叠加到已有点云 GLB |
| `fit_elevation_to_glb` | public API | VGGT predictions、输出目录、参数 | 输出路径、重力信息、日志 | 完整高程拟合和导出流程 |

---

## 当前实现中的几个细节

1. `semantic_masks == 1` 被硬编码为地面类别。
2. `use_ransac` 当前不再控制本文件内逻辑，只为兼容旧接口保留。
3. DEM 插值先用 `linear`，空洞用 `nearest` 补齐。
4. mesh 颜色默认按高程 2% 到 98% 分位数归一化，减少极端值影响。
5. `fit_elevation_to_glb()` 导出的 metadata 不包含完整 DEM 网格，只包含重力对齐和调试信息。
