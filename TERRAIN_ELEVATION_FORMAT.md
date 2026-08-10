# 地形高程数据格式规范（供外部程序生成 → 本项目读取）

> 目的：如果有另一个程序（离线处理/仿真/测绘等）需要生成一份地形文件，
> 交给本项目加载并应用到 Unity `Terrain`，双方需要约定的数据格式写在本文档。
>
> 现状说明：高程数据有两条输入路径，共用同一套 JSON 结构（`ElevationMsg`）：
> 1. 运行时通过 MQTT topic `01/map/elevation` 接收（见 `Assets/Networking/MQTT/ElevationMsg.cs`、
>    `Assets/Terrain/HandleElevationMap.cs`、`Assets/Terrain/TerrainTileManager.cs`）。
> 2. 从磁盘读取本文档定义格式的 JSON 文件（`Assets/Terrain/ElevationFileLoader.cs`，**已实现**，
>    见第 5 节），反序列化后直接调用与 MQTT 路径相同的 `TerrainTileManager.OnElevationTile(msg)`
>    （多 tile 场景）或 `HandleElevationMap.OnElevationDataReceived(msg)`（单 Terrain 场景），
>    不需要再写一套新的解析/应用逻辑。

## 1. 顶层结构（对应 `ElevationMsg`）

```json
{
  "timestamp": 1751414400.0,
  "metadata": {
    "width": 128,
    "height": 128,
    "resolution": 0.5,
    "height_resolution": 0.01,
    "origin": { "x": 0.0, "y": 0.0, "z": 0.0 },
    "coordinate_system": "local_enu",
    "min_elevation": 0.0,
    "max_elevation": 5.0,
    "tile_x": 0,
    "tile_y": 0,
    "tile_size_meters": 50.0
  },
  "data_type": "int16",
  "data": [0, 12, 34, ...],
  "data_order": "row_major"
}
```

每个文件对应**一个 tile**（一份 `TerrainData` 的高度图）。如果外部程序要一次性导出多个 tile，
建议给每个 tile 单独存一个文件，文件名用 `tile_x`/`tile_y` 编号区分，例如：

```
elevation_tile_0_0.json
elevation_tile_0_1.json
elevation_tile_-1_0.json
```

（暂不建议把多个 tile 塞进一个数组字段里，因为现有 `ElevationMsg` 解析器 `JsonUtility.FromJson<ElevationMsg>`
按单对象解析，一次只认一个 tile；要支持多 tile 数组需要额外扩展解析代码，目前未实现。）

## 2. 字段说明

### 2.1 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `timestamp` | double | 否 | 生成时间戳（Unix 秒），仅记录用途，当前代码不消费 |
| `metadata` | object | **是** | 见下表 |
| `data_type` | string | **是** | 目前只认 `"int16"`，其他值会被直接丢弃（`HandleElevationMap.cs:70` `if (msg.data_type != "int16") return;`） |
| `data` | int 数组 | **是** | 长度必须等于 `width * height`，行主序（row-major）展开的**原始整数高程值** |
| `data_order` | string | 否 | 仅作说明用途，当前代码里未读取此字段，实际按 `data_order: "row_major"` 硬编码处理（`raw = data[y*w + x]`）。如果导出顺序不是行主序，需要先自己转换好，否则地形会错位 |

### 2.2 `metadata` 字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `width` | int | **是** | 高程网格宽度（像素数），必须与 `height` 搭配、和目标 Terrain 的分辨率匹配（见第 3 节） |
| `height` | int | **是** | 高程网格高度（像素数） |
| `resolution` | float | 否 | 水平分辨率（米/像素），当前应用逻辑未使用这个值来计算尺寸，只用来辅助换算 `tile_size_meters`（如果你自己算 `tile_size_meters = resolution * width` 更省事） |
| `height_resolution` | float | **是** | 高度量化步长（米）。实际米数 = `data[i] * height_resolution`。这是唯一控制"整数 → 实际高度（米）"换算比例的字段，务必填对 |
| `origin` | object `{x,y,z}` | 否 | 当前应用逻辑未使用，仅记录用途 |
| `coordinate_system` | string | 否 | 仅记录用途，未被消费 |
| `min_elevation` / `max_elevation` | float | 否 | **仅供参考，不会被信任**：运行时会重新扫描 `data` 数组算出实际 min/max 并归一化（`HandleElevationMap.cs:77-94`），所以这两个字段填不填、填得准不准都不影响最终地形高度，只是给人看的元信息 |
| `tile_x` / `tile_y` | int | 多 tile 场景下**必填** | tile 编号，决定这份数据对应场景里哪一块 Terrain（`TerrainTileManager` 按 `(tile_x, tile_y)` 做 Dictionary key）。单 Terrain 场景（只用 `HandleElevationMap`，不走 `TerrainTileManager`）可以不填 |
| `tile_size_meters` | float | 多 tile 场景下**必填** | 单个 tile 的边长（米），必须和场景里 `TerrainTileManager.tileSizeMeters` 的网格间距一致（默认 50），否则 tile 之间会重叠或留缝。填了这个字段后代码会自动把对应 Terrain 的 `TerrainData.size` 设置为 `(tile_size_meters, 原高度, tile_size_meters)` |

## 3. 关键约束（写文件前必须满足，否则地形会错乱或报错）

1. **`data.Length == width * height`**，且是行主序：`data[y * width + x]`。
2. **`width` / `height` 必须比目标 Terrain 的 `heightmapResolution` 小 1**。
   Unity `TerrainData.SetHeights` 要求高度图是 `(heightmapResolution) × (heightmapResolution)` 网格，
   本项目内部会自动把 `width × height` 的数据在边缘补一行/一列（复制最后一行/列，见
   `HandleElevationMap.cs:100-118`），所以只要 `width == height == terrain.heightmapResolution - 1`
   就能刚好填满，不会拉伸变形。项目里 `TerrainTileManager.RandomizeActiveTiles`（测试用随机地形按钮）
   就是按这个规则算的：`res = terrain.terrainData.heightmapResolution - 1`。
   Unity 默认新建 Terrain 的 `heightmapResolution` 是 513（即 `width=height=512`），
   但本项目实际用的 `Assets/Terrain/GroundTerrainData.asset` 具体配置了多大分辨率，请在编辑器里
   选中该资产、Inspector 里查看 `Terrain Height` 设置区的分辨率数值后再对齐，不要凭经验假设。
3. **NODATA 用 `-32768`（`int16` 最小值）表示无效像素**，代码会把它替换成当前 tile 内的实际最小值
   （`HandleElevationMap.cs:75,83,105`），不会插值或报错，但大面积 NODATA 会导致那片区域被"拉平"到 min 值。
4. **`height_resolution` 决定精度和量程**：整数存储范围是 `int16`（约 -32768~32767），
   实际米数 = `原始值 × height_resolution`。例如要表示 0~50 米、精度到厘米，
   `height_resolution = 0.01`，原始整数范围只需 0~5000，远小于 int16 上限，足够。
5. **多 tile 时 `tile_size_meters` 必须和场景 `TerrainTileManager.tileSizeMeters` 一致**
   （当前场景配置为 50 米，见 `.agents/PROJECT_REFERENCE.md`），否则 tile 摆放位置会和高度数据的
   实际物理尺寸不匹配。

## 4. 示例：单 tile、64×64、0~3米高度

```json
{
  "timestamp": 1751414400.0,
  "metadata": {
    "width": 64,
    "height": 64,
    "resolution": 0.78,
    "height_resolution": 0.01,
    "origin": { "x": 0, "y": 0, "z": 0 },
    "coordinate_system": "local_enu",
    "min_elevation": 0.0,
    "max_elevation": 3.0,
    "tile_x": 0,
    "tile_y": 0,
    "tile_size_meters": 50.0
  },
  "data_type": "int16",
  "data": [0, 0, 5, 10, 20, 300, ...],
  "data_order": "row_major"
}
```

（`data` 数组长度需为 64*64=4096，此处省略。）

## 5. 文件加载脚本（已实现）

- `Assets/Terrain/ElevationFileLoader.cs`：读取本地 JSON 文件（或整个目录），反序列化为
  `ElevationMsg`，做基本校验（`data_type == "int16"`、`data.Length == width*height`），
  然后调用场景里现成的 `TerrainTileManager.OnElevationTile`（多 tile 场景优先）或
  `HandleElevationMap.OnElevationDataReceived`（单 Terrain 场景，无 `TerrainTileManager` 时使用）。
  - `LoadFile(string filePath)`：加载单个文件（对应一个 tile）。
  - `LoadDirectory(string dirPath)`：加载目录下所有 `*.json`，每个文件对应一个 tile。
- `Assets/Terrain/ElevationFileLoadButton.cs`：运行时 `OnGUI` 按钮"读取地形文件"，点击后调用
  `ElevationFileLoader.LoadFile`（若填了 `filePath`）或 `LoadDirectory`（若填了 `directoryPath`，
  `filePath` 优先）。需要挂载在场景对象上，并在 Inspector 里填好 `filePath`/`directoryPath`。

使用方式：把 `ElevationFileLoader` 和 `ElevationFileLoadButton` 一起挂在场景任意 GameObject 上
（例如 `TerrainTileSystem`），`ElevationFileLoadButton.loader` 留空会自动查找。运行时点击按钮即可
从磁盘读取按本文档格式生成的 JSON 文件并应用到地形。

## 6. 语义层 `semantic`（可选：区分可挖 / 不可挖区域）

在顶层 `ElevationMsg` 上可附加一个**可选**的 `semantic` 块，用来把每个格子标注成一种
「作业区」类型，并据此区分哪里可挖、哪里不可挖。它与高程 `data` **共享同一 BEV 栅格**
（同 `width`/`height`、同行主序 `data[y*width + x]`），因此逐格对齐、无需额外配准。
没有这个块时（旧格式）一切照旧，完全向后兼容。

```json
{
  "timestamp": 1751414400.0,
  "metadata": { "...": "同前，width/height 与 semantic 一致" },
  "data_type": "int16",
  "data": [ "...高程原始整数..." ],
  "data_order": "row_major",
  "semantic": {
    "layer_type": "zone",
    "width": 128,
    "height": 128,
    "data": [ -1, 0, 1, 3, 5, ... ],
    "legend": [
      {"code": 0, "name": "flat",     "diggable": false, "color": [0.82, 0.82, 0.80]},
      {"code": 1, "name": "dig",      "diggable": true,  "color": [0.95, 0.55, 0.15]},
      {"code": 2, "name": "dump",     "diggable": false, "color": [0.20, 0.55, 0.90]},
      {"code": 3, "name": "pile",     "diggable": true,  "color": [0.30, 0.75, 0.35]},
      {"code": 4, "name": "hazard",   "diggable": false, "color": [0.90, 0.15, 0.15]},
      {"code": 5, "name": "obstacle", "diggable": false, "color": [0.45, 0.45, 0.45]}
    ]
  }
}
```

### 6.1 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `layer_type` | string | 否 | 目前固定 `"zone"`（作业区语义）。留给以后扩展别的语义层用 |
| `width` / `height` | int | **是** | 语义栅格尺寸，**必须等于 `metadata.width` / `metadata.height`**，否则无法与高程对齐 |
| `data` | int 数组 | **是** | 长度 = `width*height`，行主序 `data[y*width + x]`，每格取值为一个作业区 `code`（见 `legend`）。`-1` 表示无数据格（empty） |
| `legend` | 数组 | **是** | 作业区代码字典。每项 `{code, name, diggable, color}` |

`legend` 每项：

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | int | 作业区代码，与 `semantic.data` 里的取值对应 |
| `name` | string | 人类可读名（flat/dig/dump/pile/hazard/obstacle） |
| `diggable` | bool | **本类是否可挖**。当前约定：`dig`、`pile` = `true`；`flat`/`dump`/`hazard`/`obstacle` = `false` |
| `color` | float[3] | 显示配色（RGB 0-1），用于 Unity 语义分区着色 |

### 6.2 作业区代码（与 `terrain_analysis.py` 一致）

| code | name | 含义 | diggable |
|---|---|---|---|
| 0 | flat | 平地 / 可通行 | 否 |
| 1 | dig | 挖掘区（坑 / 沟） | **是** |
| 2 | dump | 放料区（大片平坦） | 否 |
| 3 | pile | 料堆（可挖凸起） | **是** |
| 4 | hazard | 危险（陡坎 / 障碍旁） | 否 |
| 5 | obstacle | 障碍（挖机 / 车 / 人） | 否 |
| -1 | (empty) | 无数据 | 否 |

### 6.3 生成方（VGGT 侧）

由 `/export_elevation_json` 端点在 `dem_source="htop"` 且 `include_semantic=true` 时产出：
高程网格取 `terrain_analysis.rasterize_bev` 的 `H_top`，语义 `zone_map` 由
`extract_geometry → confirm_semantics → build_worksite_map` 得到（与 `H_top` 同栅格）。
`legend` 来自 `terrain_analysis.zone_legend()`。`elev` 源不带语义（网格不同源）。

### 6.4 消费方（Unity 侧）

- `Assets/Networking/MQTT/ElevationMsg.cs`：`ElevationMsg.semantic`（`SemanticLayer` / `SemanticLegend`）。
- `Assets/Terrain/TerrainElevationPainter.cs`：单层贴图着色器。三种上色模式
  （`Elevation` 高程渐变 / `Semantic` 作业区配色 / `Off`）+ 网格开关（热键 `C` 切模式、`G` 开关网格，
  或屏幕按钮）。并解析 `legend.diggable` 生成可挖掩码，提供 `IsDiggable(worldX, worldZ)` 供挖掘逻辑联动。
- `HandleElevationMap` 在 `SetHeights` 后调用 `painter.Repaint(normalizedMap, msg.semantic)`；
  旧格式（无 `semantic`）时语义模式自动回退为高程着色。
