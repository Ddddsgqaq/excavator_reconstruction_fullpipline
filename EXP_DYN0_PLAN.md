# E-DYN-0 — 动态铰接挖机的 VGGT+YOLOe 可行性探针

> 状态：实验设计 v1（**只规划，未执行**） · 2026-06-29
> 输入视频：`dynamic_execave_video.mp4`（890×278, 29.9 fps, 428 帧, 14.3 s）
> 关联方案：`SCENE_GRAPH_PLAN.md` v0.3 — 本实验只触及 **L2 施动者层**，de-risk 后续所有动态机制的前置假设。

---

## 0. 这段视频是什么 / 不是什么

帧采样（f0 / f214 / f426）确认：

- ✅ **相机绕挖机环绕**（方位角连续变化）→ 给 VGGT 提供视差基线。
- ✅ **挖机铰接运动**：底盘基本原地，**上车回转 + 大臂/斗杆/铲斗摆动下挖**。这是"准静态底盘 + 动态臂"的良性动态case。
- ❌ **地形是棋盘格波浪测试地板，全程不变**：没有坑、没有料堆、无任何地形形变。
- ⚠️ 画面含**鼠标光标**；分辨率 890×278 竖直信息极少（截图态）。

**因此本视频能验证 L2（机械感知/定位/动态鲁棒性），喂不了 L1 地形脊柱（残差场 / next-scoop）。** 主线需要另一段"真实开挖、地形可见变化"的视频（§7 规格）；当前暂不可得，故先把 L2 这条最危险的前置假设钉死。

---

## 1. 被检验的核心假设（为什么这是第一个实验）

> **VGGT 假设场景静态多视图。** 一台一边（可能）平移、一边铰接运动的挖机，在绕拍相机下，是否会被重建成拖影/鬼影，甚至污染周围静态地形几何？

方案 §8 只列了"VGGT 长序列漂移"（风险4），没正面回答这个**动态物体一致性**问题。不验证它，后面 Phase 1 的「YOLOe 实例→3D 质心 machine 节点」、Phase 2 跨帧关联与状态机、Phase 3 dig 事件因果定位全都建在未证实的地基上。

---

## 2. 四个探针（P1–P4）

均为**纯复用**：先用现有 orchestrator → YOLOe(text) + VGGT 产出 `predictions.npz` 与 `semantic_masks.npz`，再用 `exp_dyn0_probe.py` 离线量化。不训练、不改模型。

| 探针 | 问题 | 度量 | 退/进判据 |
|---|---|---|---|
| **P1 静态地形是否被动态机械污染** | 棋盘格地板重建是否平整 | 非挖机点对 RANSAC 最佳平面的 RMS 残差；分"含挖机视锥区域 / 远离区域"两段对比 | RMS(近机) ≲ RMS(远机)×1.5 → 地形未被显著污染，可同时拿世界系+agent；否则需单帧窗口/掩膜重建 |
| **P2 挖机几何形态** | 重建成连贯块 / 沿轨迹拖影 / 多位置鬼影 | 挖机点云在世界系的空间方差、主轴长度；逐帧质心两两距离分布 | 质心簇紧（散布 ≲ 机身尺度）→"质心"有物理意义；散布≫机身 → 鬼影，需逐帧/分窗 |
| **P3 YOLOe 跨视角分割稳定性** | `excavator`/`excavator arm`/`bucket` 文本 prompt 在环绕视角下是否稳定 | 每帧各类命中率（有 mask 的帧占比）、mask 像素数变异系数 | excavator 命中率 ≥0.9 → body 节点可用；bucket 命中率即方案 §8 风险2 的直接答案 |
| **P4 逐帧 3D 质心轨迹** | 能否得到合理的 machine 节点 state[t] | 每帧 挖机mask∩点云 质心轨迹；速度/加速度是否物理合理（无瞬移） | 轨迹连续、无 >机身尺度跳变 → Phase 1 machine 节点最小验证通过 |

### 退化路径（每个探针失败时怎么办，写进报告）
- P1 失败 → 改用"挖机 mask 排除后重建地形"或单帧窗口；记入 §8 风险6 多窗口工程化。
- P2 鬼影 → agent 节点改为"逐帧局部点云质心"，放弃全局单块假设。
- P3 bucket 不稳 → 退化为挖机实例 mask 几何近似（最低/最前端点），对齐方案 §8 风险2 既定退路。
- P4 跳变 → 加时序平滑 / 置信加权质心；标注为 Phase 2 跟踪需求。

---

## 3. 执行步骤（上游用现有 pipeline，下游用本仓库新脚本）

```bash
# 1) 起服务（GPU）
conda activate yoloe && python yoloe_service.py            # :8001
conda activate vggt  && python vggt_service.py             # :8002
python orchestrator.py                                     # :7860 Gradio

# 2) 在 Gradio 里：
#    - 上传 dynamic_execave_video.mp4，frame_interval_sec≈0.3（~48 帧），max_frames=60
#    - YOLOe text prompts: "excavator, excavator arm, bucket, ground"
#    - VGGT reconstruct（enable_semantic=True）→ 得到 workspaces/<session>/{predictions.npz, semantic_masks.npz}

# 3) 离线量化（不碰 GPU）
python exp_dyn0_probe.py --session workspaces/<session> \
       --machine-classes excavator,"excavator arm",bucket --ground-class ground \
       --out output/exp_dyn0
```

产出：`output/exp_dyn0/report.json`（P1–P4 全部度量）+ `centroid_traj.png` + `terrain_residual_hist.png` + 一段 Markdown 小结。

---

## 4. 通过 / 不通过 的总判据

- **GO（继续 Phase 1 agent 层）**：P1 未显著污染 ∧ P3 excavator 命中≥0.9 ∧ P4 轨迹连续。即使 P2 略散、bucket 不稳，也走既定退化路径，不阻塞。
- **STOP-AND-RETHINK**：P1 地形被严重带歪（RMS 近机 ≫ 远机）→ 说明动态物体污染世界系，需要在方案里把"动态掩膜重建 / 分窗"提前为主线工序，而非后续工程化。

---

## 5. 已知局限（报告里必须明说，避免高估）

1. 棋盘格非真实地形、无开挖 → **本实验不证明残差场/decision 任何东西**，只证明 L2 可行性。
2. 底盘准静态 → 这是动态case里**最容易**的一档（真正平移作业的挖机更难），结论对"原地作业"成立，对"行走中作业"需另测。
3. 890×278 低竖直分辨率 + 光标 → 可能压低 VGGT 质量；下一段视频应去光标、提分辨率。
4. 无公制尺度参照物 → P2/P4 的"机身尺度"用重建自洽尺度（相对量），不做公制结论。

---

## 6. 这条结果如何接回主线

- GO → 进入 Phase 1：把 P4 的逐帧质心固化为 `machine` 节点写入 `scene_graph.json`（schema 见方案 §3.5），先只填 L2，L1 留空待开挖视频。
- 期间可并行推进 **Phase 0 地基**（schema 定稿 + 两任务本体 + 评测脚本骨架），不依赖视频。
- 待"开挖视频"到位，再接 L1 残差场 / dig 事件 / next-scoop。
