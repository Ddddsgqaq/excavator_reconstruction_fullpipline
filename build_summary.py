"""Build a self-contained HTML summary of today's research investigation.
Generates two fresh charts, base64-embeds all figures, writes the HTML."""
import base64, io, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _fp in ["/mnt/c/Windows/Fonts/msyh.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=_fp).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

VY = "/home/maomaoyu/WS/vggt_yoloe"
W = f"{VY}/workspaces/session_20260611_162643_869764"


def fig_to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── Chart 1: confidence sweep (Y stable, horizontal collapses, aspect → canonical)
keep   = [100, 75, 50, 25, 10, 5]
Yext   = [0.134, 0.134, 0.132, 0.126, 0.116, 0.105]
Wmin   = [0.233, 0.104, 0.104, 0.099, 0.057, 0.041]
aspect = [0.58, 1.29, 1.27, 1.28, 2.02, 2.55]
x = list(range(len(keep)))
fig, ax1 = plt.subplots(figsize=(7.2, 4.3))
ax1.plot(x, Yext, "o-", color="#2a9d8f", lw=2.5, label="竖直 Y extent")
ax1.plot(x, Wmin, "s-", color="#e76f51", lw=2.5, label="水平 Wmin extent")
ax1.set_xticks(x); ax1.set_xticklabels([f"top{k}%" for k in keep])
ax1.set_xlabel("保留的高置信点比例 (置信度收紧 →)"); ax1.set_ylabel("extent (VGGT units)")
ax1.legend(loc="upper right", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(x, aspect, "^--", color="#264653", lw=2, label="aspect 高/宽")
ax2.axhline(1.3, ls=":", color="gray"); ax2.text(0.1, 1.35, "canonical ≈ 1.3", color="gray", fontsize=8)
ax2.set_ylabel("aspect 高/宽", color="#264653")
ax1.set_title("置信度扫描：竖直Y几乎不动，水平宽度坍塌 → aspect 回到 canonical", fontsize=10)
chart_sweep = fig_to_b64(fig)

# ── Chart 2: anchor robustness (length spread %)
labels = ["人·两端点\n(naive)", "人·语义mask", "挖掘机·语义mask"]
spread = [83.2, 5.5, 1.9]
fig, ax = plt.subplots(figsize=(6.2, 4.0))
bars = ax.bar(labels, spread, color=["#e76f51", "#e9c46a", "#2a9d8f"])
for b, v in zip(bars, spread):
    ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("尺度长度相对跨度 p5–p95 (%)")
ax.set_title("语义 mask 锚定把尺度方差压了 15–40×", fontsize=11)
ax.set_ylim(0, 95)
chart_robust = fig_to_b64(fig)

# ── existing figures
scene_img = file_to_b64(f"{W}/images/000000.png")
halo_img  = file_to_b64(f"{W}/halo_viz2.png")

# ── 2026-06-12 figures
pit_judge_img = file_to_b64(f"{W}/pit_judgment.png")
vert_rec_img  = file_to_b64(f"{W}/vertical_recovery.png")

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VGGT土方量测 — 科研工作日志（持续更新）</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;max-width:960px;margin:0 auto;
   padding:32px 28px;color:#222;line-height:1.7;background:#fafafa}}
 h1{{font-size:26px;border-bottom:3px solid #2a9d8f;padding-bottom:10px}}
 h2{{font-size:21px;color:#264653;margin-top:38px;border-left:5px solid #2a9d8f;padding-left:12px}}
 h3{{font-size:17px;color:#2a9d8f;margin-top:24px}}
 .meta{{color:#777;font-size:13px}}
 .card{{background:#fff;border:1px solid #e4e4e4;border-radius:8px;padding:16px 20px;margin:16px 0;
   box-shadow:0 1px 3px rgba(0,0,0,.04)}}
 .key{{background:#eef7f5;border-left:4px solid #2a9d8f;padding:12px 16px;margin:14px 0;border-radius:4px}}
 .warn{{background:#fdf3ec;border-left:4px solid #e76f51;padding:12px 16px;margin:14px 0;border-radius:4px}}
 table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}}
 th,td{{border:1px solid #ddd;padding:7px 10px;text-align:center}}
 th{{background:#264653;color:#fff}}
 tr:nth-child(even){{background:#f6f6f6}}
 img{{max-width:100%;border-radius:6px;border:1px solid #e0e0e0;margin:8px 0}}
 figcaption{{font-size:13px;color:#666;text-align:center;margin-bottom:18px}}
 code{{background:#eee;padding:1px 5px;border-radius:3px;font-size:13px}}
 .pill{{display:inline-block;background:#2a9d8f;color:#fff;border-radius:12px;padding:1px 10px;font-size:12px}}
 .strike{{color:#999}}
 .daynav{{background:#264653;color:#fff;border-radius:8px;padding:14px 20px;margin:16px 0}}
 .daynav a{{color:#8ed1c7;text-decoration:none;font-weight:bold}}
 .daynav a:hover{{text-decoration:underline}}
 .daysep{{margin:54px 0 0;border:none;border-top:2px dashed #2a9d8f}}
 .dayhead{{font-size:24px;color:#fff;background:#2a9d8f;border-radius:8px;padding:12px 20px;margin-top:18px}}
</style></head><body>

<h1>单目土方量测：VGGT 失效模式诊断 · 科研工作日志</h1>
<p class="meta">项目 vggt_yoloe（VGGT + YOLOe 挖掘机三维重建与地形/体积量测）· 持续更新的工作日志</p>

<div class="daynav">📒 日志时间线：
&nbsp;<a href="#day-20260611">2026-06-11 · 光晕失效模式诊断</a>
&nbsp;|&nbsp;<a href="#day-20260612">2026-06-12 · 竖直压缩与多帧不恢复</a>
</div>

<hr class="daysep" id="day-20260611">
<h2 class="dayhead">📅 2026-06-11 · 光晕失效模式诊断</h2>

<div class="card">
<b>一句话：</b>今天从"如何对齐绝对尺度"出发，一路追到了项目真正的科学内核——
<b>单目前馈重建在深度不连续处产生的"低置信水平光晕"会污染 DEM 与土方量测</b>。
途中推翻了一个看似成立的中间结论（"VGGT 压扁竖直"），最终确立了一条
"现象 → 机理 → 对应用的影响 → 纠正方法"的完整论文链条。
</div>

<h2>1. 研究困境（出发点）</h2>
<p>系统主体几何输出来自现成的 VGGT，团队的核心焦虑是三件事：
<b>(a)</b> 尺度——VGGT 输出是相对尺度，体积无法换算米制；
<b>(b)</b> 精度——一切都依赖 VGGT；
<b>(c)</b> 工作量——"如果主体是 VGGT，本工作的贡献在哪？"</p>
<div class="key"><b>定位：</b>VGGT 给的是一团"无尺度、有噪声、单次前馈"的相对点云。
<b>从这团点云到"可信的米制量测"是一整层 VGGT 不负责的问题——这一层就是本工作的地盘。</b></div>

<h2>2. 实验场景</h2>
<figure>
<img src="data:image/png;base64,{scene_img}">
<figcaption>无人机斜视（约距正下方 66°）拍摄：一台小型挖掘机、一个站立的人（天然 ~1.7m 标尺）、
右下角一个已挖的坑（量测目标）。known-size 锚点与量测目标同框。</figcaption>
</figure>

<h2>3. 调查链路与关键结果</h2>

<h3>3.1 尺度标定 + 语义 mask 锚定 <span class="pill">已完成</span></h3>
<p>搭建了尺度标定 harness（<code>scale_calibration.py</code>）：robust 像素→3D 采样、已知尺寸物求 scale、
多锚点一致性、敏感度分析。用画面中的人（暂设 1.70m）端到端跑出
<b>scale ≈ 2.99 m/单位，场景 ≈ 20.8 × 8.3 × 19.2 m</b>（量级合理）。</p>
<p>随后对比"两端点 naive 锚定"与"语义 mask 锚定"的稳定性：</p>
<figure><img src="data:image/png;base64,{chart_robust}">
<figcaption>±2px 抖动下，naive 两端点法尺度跨度 83%；语义 mask 约束采样把方差压到 2–5%（15–40×）。</figcaption></figure>
<div class="key">推论：<b>体积相对误差 ≈ 3 × 尺度相对误差</b>（一阶），所以尺度稳定性对方量至关重要。</div>

<h3>3.2 一个被推翻的中间结论：所谓"竖直压扁" <span class="pill" style="background:#e76f51">已修正</span></h3>
<p>用"已知长宽比物体"做尺度无关探针，多场景测得物体重建 <b>高/宽</b> 比远低于应有值
（人 ~1.2 vs 3.8；挖掘机 0.578±0.011 vs ~1.3）。一度据此推断
<span class="strike">"VGGT 系统性压扁竖直，方量将偏小 2–3×"</span>。</p>
<div class="warn"><b>但"先诊断为什么"救了这个结论。</b>置信度扫描显示：
<b>竖直 Y 几乎不动，坍塌的是水平宽度。</b>看着扁，是因为分母（水平 footprint）被吹大了，不是高度变矮。</div>
<figure><img src="data:image/png;base64,{chart_sweep}">
<figcaption>挖掘机：随置信度收紧，竖直 Y extent 基本不变，水平 Wmin 从 0.233 坍到 0.10 以下，
aspect 从 0.58 回到 ≈ canonical 1.3。两个 VGGT 头（pointmap / depth）给出几乎相同的 aspect → 非单一分支 bug。</figcaption></figure>

<h3>3.3 真正的机理：深度不连续处的"水平光晕" <span class="pill">已确认</span></h3>
<p>把挖掘机的低置信点（最低 25%）标红可视化：</p>
<figure><img src="data:image/png;base64,{halo_img}">
<figcaption>左：图像空间，红色光晕点精准勾在<b>轮廓边缘</b>；
右：俯视，绿色"可靠核心"紧致，红色光晕<b>沿地面甩成长尾、横向喷射</b>。</figcaption></figure>
<table>
<tr><th>footprint</th><th>X 向</th><th>Z 向</th></tr>
<tr><td>可靠核心（top75%）</td><td>0.231</td><td><b>0.104</b></td></tr>
<tr><td>含光晕（全部点）</td><td>0.233</td><td><b>0.363</b></td></tr>
</table>
<div class="key"><b>三重确认：</b>(1) 光晕点的局部深度梯度是可靠点的 <b>2.5×</b>（0.0037 vs 0.0015）；
(2) corr(置信度, |∇depth|) = <b>−0.32</b>；(3) 光晕只沿视线/深度方向甩，把 Z 向 footprint 吹大 <b>3.5×</b>。
机理：<b>深度不连续处（物体边界、坑沿）的边界像素，深度不可靠，沿视线方向落到地面上，形成单向水平裙边。</b></div>

<h2>4. 论文思路（最终版）</h2>
<div class="card">
<b>核心命题：</b>单目（VGGT 类）前馈重建在深度不连续处（物体边界、坑沿）产生<b>低置信度的水平光晕点</b>；
naive 地用于建 DEM / 算土方量会<b>抹宽 footprint、抹平坑沿、系统性错算体积</b>。
本工作<b>诊断该失效模式</b>，并提出<b>语义引导 + 置信度/空间一致性的鲁棒地形量测</b>，
在保证建面点覆盖度的前提下去除光晕，恢复正确几何与方量。
</div>
<h3>三个递进且可独立成文的贡献</h3>
<table>
<tr><th>贡献</th><th>内容</th><th>可量化 Δ</th></tr>
<tr><td>① 语义尺度锚定</td><td>已知尺寸语义物 + mask 约束采样恢复绝对米制</td><td>尺度方差 ↓15–40×</td></tr>
<tr><td>② 抗光晕鲁棒量测（核心）</td><td>诊断光晕失效模式；空间一致性/置信度滤波去光晕、保覆盖</td><td>footprint/坑沿/方量误差 ↓（待量化）</td></tr>
<tr><td>③ ExcaVol 基准</td><td>带 GT 的单目土方量测评测集与协议</td><td>首个基准</td></tr>
</table>
<div class="key"><b>如何回答"工作量"：</b>(1) 贡献全在 VGGT 之上、与 backbone 正交（可在 Depth-Anything-3 上复验）；
(2) 每个模块对 raw-VGGT baseline 报 Δ；(3) <b>发现 + 可视化 + 量化 + 纠正</b>一个会破坏下游量测的失效模式，
是完整的研究闭环，而非"调用 VGGT"。</div>

<h2>5. 诚实的边界</h2>
<ul>
<li>绝对尺度用了临时的人身高（1.70m），需可信已知尺寸 + 多场景复算坐实；</li>
<li>canonical 高/宽是名义值，一次<b>带已知尺寸标定物的可控实拍</b>能把结论钉成铁案；</li>
<li>不同挖掘机样本主要来自少数视频，需补不同机型/航线；</li>
<li>小物体（人）置信滤波只能部分恢复 aspect，可能叠加真实欠重建——大挖掘机是干净探针。</li>
</ul>

<h2>6. 产物</h2>
<p>均可复用：<code>scale_calibration.py</code>（标定 harness）、<code>vertical_fidelity_study.py</code>（aspect 探针）、
<code>diagnose_flattening.py</code>（根因诊断）、<code>viz_halo.py</code>（光晕可视化）。结果图与 JSON 存于对应 workspace。</p>

<hr class="daysep" id="day-20260612">
<h2 class="dayhead">📅 2026-06-12 · 竖直压缩与多帧不恢复</h2>

<div class="card">
<b>一句话：</b>本想"把光晕的账算到右下角那个坑上"，结果一探针就发现
<b>这个场景根本没有可测的坑</b>，顺势把真正的主线挖了出来：
<b>VGGT 把竖直落差系统性压缩约 8×，且这个压缩是预测本身固有的——两个预测头一致、多帧聚合也救不回来</b>。
多帧只能补覆盖/补洞，治不了高度。
</div>

<h3>1. 出发点与转折：先判断"坑在哪"</h3>
<p>计划复用前一日的诊断，对比"naive DEM vs 去光晕"的坑沿与方量。第一步要先把坑框出来。
<b>判断坑的方法：</b>拟合参考地平面后，逐像素计算
<code>depression = ground_Y − Y</code>（正值=低于地面=坑，负值=高出地面=物体/土堆）。</p>
<figure><img src="data:image/png;base64,{pit_judge_img}">
<figcaption><b>坑判定图（pit_judgment.png）：</b>
A — frame 0 上黄框是候选坑 ROI（右下角），红=挖掘机、青=人；
B — 高程图，<b>全场竖直跨度不足 0.4m</b>；
C — depression 深度图，<b>红色才是坑</b>，本场景几乎全是近 0（白）与蓝（高出地面的物体），
&gt;0.2m 的坑像素 = <b>0 个</b>；
D — 穿过 ROI 的横切，重建地表紧贴参考地平面、没有下凹。</figcaption></figure>
<div class="warn"><b>结论：这帧没有可测的坑。</b>候选 ROI 的中位"低于地面深度"= <b>−0.01m</b>（即不凹陷）。
硬套一个"坑切方"数字会是<b>假数据</b>，因此放弃在本场景算方量，转而追问"为什么场景这么平"。</div>

<h3>2. 主线发现：竖直被压缩约 8×（挖掘机探针，真高 3.0m）</h3>
<p>三方交叉验证<b>排除了重力对齐错误</b>：轨迹法向与点云主平面法向夹角仅 <b>0.2°</b>，
地面确实垂直于估计的"上"。所以"压平"是 VGGT 的真实行为，不是坐标 bug。</p>
<figure><img src="data:image/png;base64,{vert_rec_img}">
<figcaption><b>竖直压缩与多帧实验图（vertical_recovery.png）：</b>
A — 挖掘机侧视，重建高度仅 ≈0.38m（红条=真高 3.0m）；
B — pointmap 头(0.38m)与 depth 头(0.37m)<b>一致压缩</b>，故非单一分支 bug；
C — 各帧顶高 0.01–0.41m，<b>多帧融合 0.40m ≈ 最好单帧</b>，离真高 3m 仍差 ~8×；
D — 但多帧把点数从 4135 提到 66636（<b>覆盖 ↑16×</b>）。</figcaption></figure>
<table>
<tr><th>结论</th><th>数字</th><th>含义</th></tr>
<tr><td>竖直压缩 ≈ 7.9×</td><td>重建 0.38m vs 真 3.0m</td><td>重力已验证 0.2°，非 bug</td></tr>
<tr><td>两个头一致压缩</td><td>pointmap 0.378m / depth 0.372m</td><td>depth 再三角化救不回</td></tr>
<tr><td>多帧不恢复高度</td><td>融合 0.40m = 最好单帧</td><td>压缩焊死在单次联合推理</td></tr>
<tr><td>多帧只补覆盖</td><td>4135 → 66636 点（↑16×）</td><td>补洞有效，治不了高度</td></tr>
</table>
<div class="key"><b>对应用的意义：</b>多帧聚合能提覆盖/补遮挡空洞，但
<b>无法修复竖直压缩</b>——压缩已被焊死在 VGGT 的单次联合前馈里。
要量准土方，必须有<b>显式的竖直纠正</b>（语义锚定反压缩，或绕开 world_points 的基线几何重三角化）。</div>

<h3>3. 与前一日"光晕"结论的关系（对齐，非矛盾）</h3>
<p>06-11 的"竖直 Y 几乎不变、光晕只吹大水平 footprint"在<b>置信滤波清裙边</b>这一层依然成立
（Z 向 1.08m→0.17m）。但今天补上的是<b>绝对竖直</b>：连最干净的核心也只有 0.35–0.38m，
即<b>叠加在光晕之上的第二个失效模式</b>，去光晕治不了。两者并存。</p>

<div class="warn"><b>诚实的边界（留给下次）：</b>当前还无法 100% 分清这是"竖直专属压缩"还是"整体欠尺度"——
scale=2.99 是用人(测量方向主要为水平)临时标定，挖掘机水平宽度 ~0.6m 看着也偏小。
<b>需要一个已知水平尺寸的标定物</b>来分离"各向异性竖直压缩 vs 均匀缩小"，才能正式定性。</div>

<h3>4. 本日产物与下一步</h3>
<p>新脚本 <code>vertical_recovery_study.py</code>（含坑判定 + 压缩/恢复实验）；
输出 <code>pit_judgment.png</code> / <code>vertical_recovery.png</code> / <code>vertical_recovery_result.json</code>。</p>
<ul>
<li><b>第一优先（贡献②升级）：</b>竖直压缩的<b>显式纠正方法</b>——语义锚定标定压缩因子并反压缩，或基线几何重三角化；并测"压缩因子是否随高度/距离恒定"以定纠正模型。</li>
<li><b>保留：</b>把账算到真实的坑——<b>需换场景</b>，在其余 workspace（尤其 05-22 dji_fly 组）用本日的 depression 判据先验证"有没有坑"，再算方量（且必须先有竖直纠正）。</li>
</ul>

<p class="meta" style="margin-top:40px">— 工作日志持续更新 · 最近更新 2026-06-12 · 下一步详见 RESEARCH_PROGRESS.md</p>
</body></html>"""

out = f"{VY}/research_summary_20260611.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out, f"({len(HTML)//1024} KB)")
