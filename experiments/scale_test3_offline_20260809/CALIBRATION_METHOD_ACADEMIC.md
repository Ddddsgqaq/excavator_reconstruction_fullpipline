# 基于已知长度参照物与语义地面的 VGGT 尺度—高程联合校准算法

## 摘要

单目多视图重建通常只能恢复相似变换意义下的三维结构，其世界坐标仍具有未知全局尺度，且
VGGT 世界坐标系随首帧相机姿态发生倾斜。本文给出一种面向离线 VGGT 重建的尺度—高程联合
校准算法：首先利用 YOLOE 得到已知长度尺子的多帧语义掩膜，通过二维主轴约束和三维稳健端点
估计构造逐帧尺度观测；随后以中位数与中位绝对偏差进行时间维异常检测，得到全局米制尺度；
最后结合相机轨迹平面、语义桌面 RANSAC 与逐帧局部平面细化，建立以桌面为零高程的米制坐标系。
该方法不使用待测物体真实尺寸参与校准，因此可用独立物体真值评价下游尺寸精度。

本实验中，13 个尺子候选锚点保留 10 个，最终尺度为
**0.408574 m/VGGT unit**，接受锚点的尺度变异系数为 **0.739%**；尺子闭环 MAE 为
**0.090 cm**。使用用户提供的两个物体真值进行独立评价后，六个尺寸轴的总体 MAPE 为
**4.03%**。

---

## 1. 问题定义

设离线序列包含 $S$ 个 RGB 帧。VGGT 对第 $i$ 帧输出：

$$
\mathcal{P}_i=\{\mathbf X_{i,u}\in\mathbb R^3\}_{u\in\Omega},
\qquad
\mathcal{C}_i=\{c_{i,u}\}_{u\in\Omega},
\qquad
\mathbf E_i=[\mathbf R_i\mid\mathbf t_i],
$$

其中，$\Omega$ 为模型图像网格，$\mathbf X_{i,u}$ 是 VGGT 世界坐标中的深度反投影点，
$c_{i,u}$ 是深度置信度，$\mathbf E_i$ 是世界到相机的外参。YOLOE 提供与该网格对齐的
语义标签图 $\mathbf M_i$。尺子的已知真实长度记为

$$
L_0=0.15\ {\rm m}.
$$

算法要求估计：

$$
\Theta=\left(s,\mathbf R_{\rm align},
\{(\mathbf o_i,\mathbf n_i)\}_{i=1}^{S}\right),
$$

其中 $s$ 是米/重建单位的全局尺度，$\mathbf R_{\rm align}$ 将全局“向上”方向旋转到
$+Y$ 轴，$(\mathbf o_i,\mathbf n_i)$ 分别是第 $i$ 帧局部桌面平面的锚点和单位法向。

对任意重建点 $\mathbf X$，其米制对齐坐标与相对桌面高程定义为

$$
\mathbf X^{m}=s\,\mathbf R_{\rm align}\mathbf X ,
$$

$$
h_i(\mathbf X)
=s\,(\mathbf X-\mathbf o_i)^\mathsf T\mathbf n_i.
$$

由于尺度是各向同性标量，先旋转后缩放与先缩放后旋转等价。

### 1.1 基本假设

1. 已知长度尺子与待测物体在拍摄过程中保持静止。
2. 至少 3 帧、实际实现优选至少 5 帧包含未严重裁切的完整尺子。
3. YOLOE 能提供尺子与桌面的语义支持区域，但允许少量漏分割和误分割。
4. VGGT 多帧重建已处于统一世界坐标系；算法只恢复相似变换中的尺度与竖直方向。
5. 桌面在待测区域内可近似为局部平面。

---

## 2. 多帧尺子尺度校准

### 2.1 尺子连通域质量筛选

令 $\mathcal R_i=\{u\in\Omega:M_i(u)=2\}$ 为第 $i$ 帧尺子像素集合。对其中每个
8 邻域连通分量 $\mathcal R_{i,k}$，计算：

- 面积 $A_{i,k}$；
- 像素坐标去中心化矩阵的奇异值 $\sigma_1\geq\sigma_2$；
- 细长度 $\rho_{i,k}=\sigma_1/\max(\sigma_2,\epsilon)$；
- 最小外接旋转矩形长边 $\ell_{i,k}$；
- 与模型图像边界的最小距离 $b_{i,k}$。

当前 294×518 模型网格上的严格门控为

$$
A_{i,k}\geq3000,\quad
\rho_{i,k}\geq3.5,\quad
\ell_{i,k}\geq150,\quad
b_{i,k}\geq3.
$$

若有多个分量满足条件，选择使 $A_{i,k}\rho_{i,k}$ 最大的分量。面积和像素长度阈值随
模型分辨率变化；迁移到其他分辨率时应按图像面积和对角线归一化。

### 2.2 二维主轴约束下的三维端点

设通过门控的尺子像素坐标为
$\mathbf p_j=[x_j,y_j]^\mathsf T$。对去中心化坐标矩阵作 SVD，取第一右奇异向量
$\mathbf a_i$ 作为尺子二维主轴，并计算轴向投影：

$$
t_j=(\mathbf p_j-\bar{\mathbf p})^\mathsf T\mathbf a_i .
$$

记 $Q_\alpha(t)$ 为 $t$ 的 $\alpha$ 分位数。尺子的两个端部像素集合为

$$
\mathcal T_i^-=\{j:t_j\leq Q_{0.02}(t)\},
\qquad
\mathcal T_i^+=\{j:t_j\geq Q_{0.98}(t)\}.
$$

把端部像素索引映射到同一帧的三维点，并以逐坐标中位数估计三维端点：

$$
\mathbf q_i^-=
\operatorname{median}_{j\in\mathcal T_i^-}\mathbf X_{i,\mathbf p_j},
\qquad
\mathbf q_i^+=
\operatorname{median}_{j\in\mathcal T_i^+}\mathbf X_{i,\mathbf p_j}.
$$

每个端部至少需要 5 个有限三维点。采用端部区域中位数而非极端单点，可以抑制掩膜毛刺、
透明尺边缘深度异常及孤立点。

第 $i$ 帧的重建长度和候选尺度为

$$
d_i=\|\mathbf q_i^+-\mathbf q_i^-\|_2,
\qquad
s_i=\frac{L_0}{d_i}.
$$

### 2.3 时间维 MAD 异常剔除

首先计算候选尺度中位数与 MAD：

$$
\tilde{s}=\operatorname{median}_i(s_i),
\qquad
\operatorname{MAD}_s=
\operatorname{median}_i|s_i-\tilde{s}|.
$$

为避免几乎零 MAD 导致阈值退化，定义稳健尺度：

$$
\hat{\sigma}_s=
\max\left(1.4826\,\operatorname{MAD}_s,\;0.002\,\tilde{s}\right).
$$

接受集合为

$$
\mathcal I=
\left\{i:\ |s_i-\tilde{s}|\leq3.5\hat{\sigma}_s\right\}.
$$

若 $|\mathcal I|<5$，则保留与 $\tilde{s}$ 距离最小的至多 5 个候选锚点作为保护性回退。
最终尺度取接受集合的中位数：

$$
s^\star=\operatorname{median}_{i\in\mathcal I}s_i.
$$

该时序门控能够剔除“二维仍然细长、但只分割出尺子局部”的帧。本实验被拒绝的帧 3、5、10
正属于此类情况。

---

## 3. 语义地面与高程方向校准

### 3.1 相机轨迹先验

由世界到相机外参得到相机中心：

$$
\mathbf C_i=-\mathbf R_i^\mathsf T\mathbf t_i.
$$

对去中心化相机中心矩阵作 SVD，最小方差方向作为轨迹平面法向
$\mathbf n_{\rm traj}$。若轨迹 PCA 的第二与第一特征值之比小于 0.1，则轨迹近似退化为
直线，不能稳定定义平面。

法向符号根据场景点相对轨迹平面的多数侧确定，使法向指向“向上”方向。

### 3.2 语义桌面 RANSAC

由桌面语义构造支持集

$$
\mathcal G=
\left\{\mathbf X_{i,u}:
M_i(u)=1,\ 
c_{i,u}\geq0.5\,c_{\max},\
\mathbf X_{i,u}\ {\rm finite}\right\}.
$$

在 $\mathcal G$ 上运行 400 次三点平面 RANSAC。重建单位下的内点距离阈值为 0.03；若有
轨迹法向先验，仅接受满足

$$
|\mathbf n^\mathsf T\mathbf n_{\rm traj}|>0.5
$$

的候选平面。最优平面按内点数选择，再用全部内点 SVD 重估法向
$\mathbf n_{\rm ground}$。

若轨迹法向与语义桌面法向夹角

$$
\theta=
\arccos\left(
|\mathbf n_{\rm traj}^\mathsf T\mathbf n_{\rm ground}|
\right)
$$

超过 $10^\circ$，且语义地面同时满足内点数不少于 100、内点率不少于 70%，则使用语义
地面覆盖轨迹结果；否则保留轨迹法向。两者均不可用时，才在全点云上执行平面 RANSAC 回退。

### 3.3 最短弧旋转

令最终全局向上单位向量为 $\mathbf n_g$，构造最短弧旋转

$$
\mathbf R_{\rm align}\mathbf n_g=[0,1,0]^\mathsf T.
$$

实现使用 Rodrigues 形式，并显式处理同向与 $180^\circ$ 反向退化情况。经过该旋转后，
$Y$ 为高程轴，$X$-$Z$ 为水平面。

### 3.4 逐帧局部桌面细化

全局法向用于确定方向，但尺寸测量采用逐帧局部桌面平面降低小范围翘曲和配准残差：

1. 在该帧桌面语义内取置信度前 30% 的点，即不低于桌面置信度 P70；
2. 点数超过 20,000 时固定随机种子下采样；
3. 以全局法向为先验，执行 4 轮 SVD 平面拟合；
4. 每轮保留绝对残差不高于 P70 的点；
5. 若候选局部法向偏离全局法向超过 $35^\circ$，拒绝候选并回退全局法向。

最终局部平面为

$$
\Pi_i:\quad
(\mathbf X-\mathbf o_i)^\mathsf T\mathbf n_i=0,
$$

从而任意物体点的米制局部高程为

$$
h_i(\mathbf X)=
s^\star(\mathbf X-\mathbf o_i)^\mathsf T\mathbf n_i.
$$

---

## 4. 算法三线表

下表按学术论文中的 booktabs 三线表结构组织：仅保留顶线、表头分隔线与底线，不使用竖线。

<table style="border-collapse:collapse; width:100%; border-top:2px solid #222; border-bottom:2px solid #222;">
  <thead>
    <tr style="border-bottom:1.5px solid #222;">
      <th style="text-align:left; padding:6px; width:8%;">行</th>
      <th style="text-align:left; padding:6px; width:18%;">阶段</th>
      <th style="text-align:left; padding:6px;">操作</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:5px;">1</td><td style="padding:5px;">输入</td><td style="padding:5px;">读取多帧世界点 Pᵢ、置信度 Cᵢ、外参 Eᵢ、语义图 Mᵢ 与已知尺长 L₀。</td></tr>
    <tr><td style="padding:5px;">2</td><td style="padding:5px;">尺子筛选</td><td style="padding:5px;">提取尺子连通域，按面积、细长度、旋转矩形长边和边界距离执行严格门控。</td></tr>
    <tr><td style="padding:5px;">3</td><td style="padding:5px;">主轴</td><td style="padding:5px;">对尺子二维像素作 SVD，取最大方差方向 aᵢ。</td></tr>
    <tr><td style="padding:5px;">4</td><td style="padding:5px;">端部集合</td><td style="padding:5px;">沿主轴投影，分别取前 2% 与后 2% 像素作为两个端部。</td></tr>
    <tr><td style="padding:5px;">5</td><td style="padding:5px;">三维端点</td><td style="padding:5px;">端部对应三维点逐坐标取中位数，得到 qᵢ⁻ 与 qᵢ⁺。</td></tr>
    <tr><td style="padding:5px;">6</td><td style="padding:5px;">候选尺度</td><td style="padding:5px;">计算 dᵢ = ‖qᵢ⁺ − qᵢ⁻‖₂ 与 sᵢ = L₀/dᵢ。</td></tr>
    <tr><td style="padding:5px;">7</td><td style="padding:5px;">时序稳健化</td><td style="padding:5px;">以 3.5-MAD 门控剔除不完整尺子帧，若不足 5 帧则采用最近邻锚点回退。</td></tr>
    <tr><td style="padding:5px;">8</td><td style="padding:5px;">全局尺度</td><td style="padding:5px;">取接受尺度中位数 s★，将全部重建点转换为米制。</td></tr>
    <tr><td style="padding:5px;">9</td><td style="padding:5px;">轨迹法向</td><td style="padding:5px;">由相机中心 PCA 得到 n_traj，检测轨迹退化。</td></tr>
    <tr><td style="padding:5px;">10</td><td style="padding:5px;">语义地面</td><td style="padding:5px;">在高置信桌面点上执行 400 次 RANSAC，并以内点 SVD 重拟合 n_ground。</td></tr>
    <tr><td style="padding:5px;">11</td><td style="padding:5px;">法向选择</td><td style="padding:5px;">若两法向差异超过 10° 且语义平面支持强，则由语义地面覆盖轨迹；否则保留轨迹。</td></tr>
    <tr><td style="padding:5px;">12</td><td style="padding:5px;">坐标对齐</td><td style="padding:5px;">求最短弧旋转 R_align，使最终向上方向映射到 +Y。</td></tr>
    <tr><td style="padding:5px;">13</td><td style="padding:5px;">局部细化</td><td style="padding:5px;">逐帧迭代拟合桌面平面 (oᵢ, nᵢ)，并以 35° 先验约束防止翻转。</td></tr>
    <tr><td style="padding:5px;">14</td><td style="padding:5px;">输出</td><td style="padding:5px;">输出 s★、R_align、逐帧局部平面及米制高程函数 hᵢ(X)。</td></tr>
  </tbody>
</table>

### 4.1 可直接用于论文的 LaTeX booktabs 源码

~~~latex
\begin{table*}[t]
\centering
\caption{基于已知长度尺子与语义地面的尺度--高程联合校准算法}
\label{tab:metric_ground_calibration}
\begin{tabular}{clp{0.72\textwidth}}
\toprule
行 & 阶段 & 操作 \\
\midrule
1 & 输入 & 读取世界点、置信度、相机外参、语义图与已知尺长 $L_0$. \\
2 & 尺子筛选 & 按面积、细长度、旋转矩形长边和边界距离筛选尺子连通域. \\
3 & 主轴估计 & 对尺子二维像素作 SVD，取第一主轴 $\mathbf a_i$. \\
4 & 端部提取 & 取轴向投影的 $[0,2]\%$ 与 $[98,100]\%$ 像素. \\
5 & 三维端点 & 对两个端部对应三维点分别取逐坐标中位数. \\
6 & 候选尺度 & 计算 $d_i=\|\mathbf q_i^+-\mathbf q_i^-\|_2$ 和 $s_i=L_0/d_i$. \\
7 & 时序门控 & 以 $3.5$-MAD 剔除异常尺度锚点；不足五帧时执行最近邻回退. \\
8 & 全局尺度 & 取接受候选的中位数 $s^\star$，恢复米制坐标. \\
9 & 轨迹先验 & 对相机中心作 PCA，得到轨迹平面法向并检测退化. \\
10 & 语义地面 & 在高置信桌面点上执行 RANSAC，并以内点 SVD 重拟合. \\
11 & 法向选择 & 依据夹角、内点数与内点率在轨迹法向和语义地面法向间选择. \\
12 & 坐标对齐 & 求 $\mathbf R_{\rm align}$，使 $\mathbf R_{\rm align}\mathbf n_g=[0,1,0]^\mathsf T$. \\
13 & 局部细化 & 逐帧迭代拟合桌面平面，并施加 $35^\circ$ 全局法向约束. \\
14 & 输出 & 输出尺度、对齐旋转、逐帧局部平面以及米制高程函数. \\
\bottomrule
\end{tabular}
\end{table*}
~~~

---

## 5. 参数三线表

| 符号或参数 | 本实验取值 | 作用 |
|---|---:|---|
| $L_0$ | 0.15 m | 尺子真实长度 |
| 抽帧间隔 | 0.5 s | 与离线主流程一致 |
| 模型网格 | 294×518 | VGGT 深度与语义测量网格 |
| 尺子最小面积 | 3000 px | 排除小误检 |
| 尺子最小细长度 | 3.5 | 排除非尺状区域 |
| 尺子最小长边 | 150 px | 排除局部碎片 |
| 边界安全距离 | 3 px | 排除裁切尺子 |
| 端部比例 | 2% / 2% | 稳健三维端点区域 |
| 时序门控 | 3.5-MAD | 剔除尺度异常帧 |
| MAD 下限 | $0.002\tilde s$ | 防止零 MAD |
| 最小保护锚点 | 5 | 时序门控回退 |
| 地面置信门限 | $0.5c_{\max}$ | 全局语义地面支持集 |
| RANSAC 次数 | 400 | 全局地面平面估计 |
| RANSAC 距离 | 0.03 VGGT unit | 平面内点判断 |
| 轨迹/地面冲突角 | 10° | 触发法向支持度比较 |
| 语义覆盖最小内点 | 100 | 强地面支持条件 |
| 语义覆盖最小内点率 | 70% | 强地面支持条件 |
| 局部桌面置信度 | 帧内 P70 | 局部平面点筛选 |
| 局部拟合轮数 | 4 | 迭代残差截断 |
| 局部法向最大偏离 | 35° | 防止局部平面翻转 |

严格意义上的三线表排版可采用上一节 LaTeX booktabs 形式；此处保留 Markdown 表格是为了在
代码仓库和 Codex 文件预览中直接阅读。

---

## 6. 算法正确性与稳健性讨论

### 6.1 尺度估计的一致性

理想相似重建满足

$$
\mathbf X^{\rm true}=s_0\mathbf X^{\rm VGGT}+\mathbf t,
$$

因此同一刚性尺子的重建端点距离满足 $L_0=s_0d_i$，从而 $s_i=L_0/d_i$ 是
$s_0$ 的一致观测。平移项在端点差中抵消，旋转不改变欧氏距离，所以尺度估计不依赖世界
坐标原点和相机朝向。

### 6.2 中位数与 MAD 的抗异常性

只要超过一半的候选尺度来自完整尺子，中位数对任意幅度异常值保持 50% 崩溃点；MAD 同样具有
50% 崩溃点。由此，局部遮挡或端点漏分割不会像均值估计那样直接拉动最终尺度。

### 6.3 尺度误差传播

任意重建长度 $d$ 的米制估计为 $\hat L=s^\star d$。一阶相对误差近似为

$$
\frac{\delta L}{L}
\approx
\frac{\delta s}{s^\star}
+
\frac{\delta d}{d}.
$$

因此尺子锚点的尺度离散只反映第一项；物体的深度、分割和遮挡误差进入第二项。本实验接受锚点
尺度 CV 为 0.739%，而物体六轴 MAPE 为 4.03%，说明主要剩余误差来自物体几何，而非单纯
尺度因子。

### 6.4 地面误差对高度的影响

若估计法向与真实法向夹角为 $\delta\theta$，水平距离为 $r$，则由倾斜产生的高度误差
一阶上界约为

$$
|\delta h_{\rm tilt}|\lesssim r\sin(\delta\theta).
$$

因此算法不直接使用单一全局平面计算所有物体高度，而是在全局方向约束下逐帧拟合物体附近桌面。
本实验局部法向相对全局法向偏差中位数为 2.83°，局部桌面残差 RMSE 中位数为 0.324 mm。

---

## 7. 实验结果三线表

| 指标 | 数值 |
|---|---:|
| 尺子候选锚点 | 13 |
| 接受 / 拒绝锚点 | 10 / 3 |
| 最终尺度 | 0.408574 m/VGGT unit |
| 接受尺度 CV | 0.739% |
| 尺子闭环 MAE | 0.090 cm |
| 尺子闭环 RMSE | 0.110 cm |
| 接受尺度最小 / 最大值 | 0.405716 / 0.414290 m/VGGT unit |
| 局部桌面有效帧 | 21 / 21 |
| 局部桌面 P2–P98 残差厚度中位数 | 1.13 mm |
| 局部桌面残差 RMSE 中位数 | 0.324 mm |
| 红盒三轴 MAPE | 2.37% |
| 红瓶三轴 MAPE | 5.69% |
| 六轴总体 MAPE | 4.03% |
| 六轴总体 RMSE | 0.446 cm |

需要强调，尺子闭环使用同一参照物，评价的是校准的内部一致性；红盒和红瓶的真实尺寸未参与
尺度估计，因此两个物体的逐轴误差才是独立的下游尺寸精度。

---

## 8. 计算复杂度

令 $N=SHW$ 为所有模型像素数，$N_r$ 为尺子像素数，$N_g$ 为地面支持点数，
$K=400$ 为 RANSAC 次数。

- 语义连通域与点筛选：$O(N)$；
- 尺子主轴和端点计算：$O(N_r)$，因为二维 SVD 的列数固定为 2；
- 地面 RANSAC：最坏 $O(KN_g)$；
- 内点 SVD 与逐帧局部平面：$O(N_g)$；
- 额外存储：除已有点云外为 $O(N)$ 的布尔掩膜和标签。

本实验尺度、地面与尺寸分析合计用时 3.65 s，峰值 RSS 约 483 MiB；VGGT 重建本身不计入
校准算法复杂度。

---

## 9. 失效模式与适用边界

1. **尺子被系统性截短**：若超过一半候选帧只包含同一段局部尺子，中位数也会产生有偏尺度。
2. **透明参照物深度偏差**：语义掩膜正确不代表深度正确；应优先让尺子两端落在纹理充分区域。
3. **运动物体**：尺子或桌面在拍摄中移动会破坏统一世界坐标假设。
4. **非平面支撑面**：明显弯曲或分层地面需要局部曲面模型，而非单平面。
5. **轨迹退化**：近似直线运动无法稳定定义轨迹平面，此时依赖语义地面或全点云回退。
6. **语义地面污染**：若桌面掩膜大面积覆盖物体，RANSAC 可能选到错误平面；内点率门控只能
   抵御少数污染，不能抵御占主导的系统性误标。
7. **分辨率迁移**：面积与像素长度门限必须归一化，不能原样用于不同模型网格。

---

## 10. 与代码实现的对应关系

| 论文步骤 | 实现位置 |
|---|---|
| 尺子连通域与严格门控 | analyze_dimensions.py 调用 select_component |
| 二维主轴与 2% 三维端点 | analyze_scale_volume.py 中 ruler_endpoint_length |
| 候选尺度 | analyze_scale_volume.py 中 calibrate_scale |
| 3.5-MAD 时序门控 | analyze_dimensions.py 中 robust_scale |
| 轨迹—语义地面级联 | ../../gravity_alignment.py 中 estimate_gravity |
| 逐帧局部桌面细化 | analyze_scale_volume.py 中 fit_frame_table_plane |
| 真值尺寸精度 | evaluate_ground_truth.py |
| 完整数值产物 | dimension_results.json 与 ground_truth_accuracy.json |

本文档描述的是当前实际运行代码，而非另行设计的理想化流程。尺度估计使用
predictions.npz 中的 world_points_from_depth、depth_conf 和 semantic_masks；所有语义与
几何均来自同一组 21 个离线 RGB 抽帧。
