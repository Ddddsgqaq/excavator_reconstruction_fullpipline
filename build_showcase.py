"""
build_showcase.py — Generate a research-draft showcase webpage for the
monocular earthwork-volume project: full pipeline (video → reconstruction →
calibration → quantification), a technical framework diagram, all process
figures, scientific narrative, a related-work comparison, and open problems.

Self-contained: regenerates the small charts, draws the framework diagram,
base64-embeds every figure, writes experiment_showcase.html.
Run inside the `vggt` conda env.
"""
import base64, io, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for _fp in ["/mnt/c/Windows/Fonts/msyh.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=_fp).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

VY = "/home/maomaoyu/WS/vggt_yoloe"
W = f"{VY}/workspaces/session_20260611_162643_869764"


def fig_to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── Technical framework diagram ──────────────────────────────────────────────
def draw_framework():
    fig, ax = plt.subplots(figsize=(11.5, 13))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15.2); ax.axis("off")

    C = dict(input="#5c6b73", vggt="#3a86b8", done="#2a9d8f",
             fail="#e76f51", todo="#b0b0b0", out="#264653")

    def box(x, y, w, h, text, color, dashed=False, fc=None, tcolor="white", fs=11):
        style = "round,pad=0.02,rounding_size=0.12"
        p = FancyBboxPatch((x, y), w, h, boxstyle=style,
                           linewidth=2, edgecolor=color,
                           facecolor=fc if fc else color,
                           linestyle="--" if dashed else "-", zorder=2)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tcolor, zorder=3, wrap=True)
        return (x + w/2, y, x + w/2, y + h)  # bottom & top centers

    def arrow(top_of_lower, bot_of_upper, color="#444", text=None):
        # connect bottom-center of upper box to top-center of lower box
        a = FancyArrowPatch((bot_of_upper[0], bot_of_upper[1]),
                            (top_of_lower[0], top_of_lower[3]),
                            arrowstyle="-|>", mutation_scale=18,
                            linewidth=2, color=color, zorder=1)
        ax.add_patch(a)
        if text:
            ax.text((bot_of_upper[0]+top_of_lower[0])/2 + 0.15,
                    (bot_of_upper[1]+top_of_lower[3])/2, text,
                    fontsize=8.5, color="#666", va="center", ha="left")

    cx, w = 2.0, 6.0
    b_in   = box(cx, 14.0, w, 0.95, "① 输入：无人机斜视视频\n(单目, ~66° 离正下方, 含已知尺寸语义物)", C["input"])
    b_vggt = box(cx, 12.5, w, 0.95, "② VGGT 前馈重建 (CVPR'25)\n每帧 world_points / depth / 相机 / 置信度 · 无尺度", C["vggt"])
    b_grav = box(cx, 11.0, w, 0.95, "③ 重力对齐 (轨迹 PCA → R_align, Y=上)\n复用 gravity_alignment.py · 验证 0.2°", C["done"])
    b_yolo = box(cx,  9.5, w, 0.95, "④ YOLOe 开放词表分割\n语义掩膜：挖掘机 / 人 / 地面", C["done"])
    b_scale= box(cx,  8.0, w, 0.95, "⑤ 语义尺度锚定 (已知尺寸物 → 米制 scale)\nscale_calibration.py · 方差 ↓15–40×", C["done"])

    # diagnostics — two failure modes side by side
    b_h1 = box(0.4, 6.0, 4.5, 1.15, "⑥a 失效模式·水平光晕\n深度不连续处边界点沿视线甩到地面\nfootprint ↑3.5× · 置信滤波可去", C["fail"], fs=9.5)
    b_h2 = box(5.1, 6.0, 4.5, 1.15, "⑥b 失效模式·竖直压缩 ~8×\n挖掘机 3.0m→0.38m · 两头一致\n多帧不恢复(↑覆盖16×) · 去光晕治不了", C["fail"], fs=9.5)

    b_robust= box(cx, 4.0, w, 1.1, "⑦ 鲁棒地形/DEM + 体积\n去光晕(已) 多帧补覆盖(已) | 竖直纠正(待) 大坑GT(待)", C["todo"], fc="#f3f3f3", tcolor="#333", fs=10)
    b_out  = box(cx, 2.3, w, 0.95, "⑧ 输出：米制土方量 (cut / fill)\n目标精度对标 UAV 摄影测量 <1%", C["out"])

    arrow(b_vggt, b_in); arrow(b_grav, b_vggt); arrow(b_yolo, b_grav); arrow(b_scale, b_yolo)
    # scale -> diagnostics (split)
    a1 = FancyArrowPatch((b_scale[0], b_scale[1]), (b_h1[0], b_h1[3]), arrowstyle="-|>",
                         mutation_scale=16, lw=2, color="#444", zorder=1); ax.add_patch(a1)
    a2 = FancyArrowPatch((b_scale[0], b_scale[1]), (b_h2[0], b_h2[3]), arrowstyle="-|>",
                         mutation_scale=16, lw=2, color="#444", zorder=1); ax.add_patch(a2)
    a3 = FancyArrowPatch((b_h1[0], b_h1[1]), (b_robust[0]-1.0, b_robust[3]), arrowstyle="-|>",
                         mutation_scale=16, lw=2, color="#444", zorder=1); ax.add_patch(a3)
    a4 = FancyArrowPatch((b_h2[0], b_h2[1]), (b_robust[0]+1.0, b_robust[3]), arrowstyle="-|>",
                         mutation_scale=16, lw=2, color="#444", zorder=1); ax.add_patch(a4)
    arrow(b_out, b_robust)

    # legend
    leg = [("#2a9d8f","已完成 / 复用"), ("#e76f51","已诊断失效模式"),
           ("#3a86b8","现成 backbone"), ("#b0b0b0","待完成 (虚线)")]
    for i,(c,t) in enumerate(leg):
        ax.add_patch(FancyBboxPatch((0.4+i*2.45, 0.55), 0.32, 0.32, boxstyle="round,pad=0.02",
                     facecolor=c, edgecolor=c));
        ax.text(0.8+i*2.45, 0.71, t, fontsize=9, va="center")
    ax.text(5.0, 15.0, "技术框架：从视频到米制土方量", fontsize=14, weight="bold", ha="center", color="#264653")
    fig.savefig("/tmp/framework_preview.png", dpi=120, bbox_inches="tight")
    return fig_to_b64(fig)


# ── Chart: anchor robustness ─────────────────────────────────────────────────
def chart_robustness():
    labels = ["人·两端点\n(naive)", "人·语义mask", "挖掘机·语义mask"]
    spread = [83.2, 5.5, 1.9]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.bar(labels, spread, color=["#e76f51", "#e9c46a", "#2a9d8f"])
    for b, v in zip(bars, spread):
        ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("尺度长度相对跨度 p5–p95 (%)")
    ax.set_title("语义 mask 锚定把尺度方差压了 15–40×", fontsize=11); ax.set_ylim(0, 95)
    return fig_to_b64(fig)


# ── Chart: confidence sweep ──────────────────────────────────────────────────
def chart_sweep():
    keep = [100, 75, 50, 25, 10, 5]
    Yext = [0.134, 0.134, 0.132, 0.126, 0.116, 0.105]
    Wmin = [0.233, 0.104, 0.104, 0.099, 0.057, 0.041]
    aspect = [0.58, 1.29, 1.27, 1.28, 2.02, 2.55]
    x = list(range(len(keep)))
    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    ax1.plot(x, Yext, "o-", color="#2a9d8f", lw=2.5, label="竖直 Y extent")
    ax1.plot(x, Wmin, "s-", color="#e76f51", lw=2.5, label="水平 Wmin extent")
    ax1.set_xticks(x); ax1.set_xticklabels([f"top{k}%" for k in keep])
    ax1.set_xlabel("保留的高置信点比例 (置信度收紧 →)"); ax1.set_ylabel("extent (VGGT units)")
    ax1.legend(loc="upper right", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(x, aspect, "^--", color="#264653", lw=2)
    ax2.axhline(1.3, ls=":", color="gray"); ax2.text(0.1, 1.35, "canonical ≈ 1.3", color="gray", fontsize=8)
    ax2.set_ylabel("aspect 高/宽", color="#264653")
    ax1.set_title("置信度扫描：竖直Y几乎不动，水平宽度坍塌", fontsize=10)
    return fig_to_b64(fig)


print("drawing framework ...")
framework = draw_framework()
robust = chart_robustness()
sweep = chart_sweep()
scene_img = file_to_b64(f"{W}/images/000000.png")
halo_img = file_to_b64(f"{W}/halo_viz2.png")
pit_img = file_to_b64(f"{W}/pit_judgment.png")
vert_img = file_to_b64(f"{W}/vertical_recovery.png")
gauge_img = file_to_b64(f"{W}/vertical_gauge_test.png")
e1a_img = file_to_b64(f"{W}/vertical_correct_e1a.png")
e1b_img = file_to_b64(f"{W}/vertical_correct_e1b.png")
q2sig_img = file_to_b64(f"{VY}/q2_signature_stability.png")
q2ratio_img = file_to_b64(f"{VY}/q2b_vertical_ratio.png")

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>单目无人机土方量测 — 实验展示 / 科研初稿</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;max-width:1000px;margin:0 auto;
   padding:36px 30px;color:#222;line-height:1.75;background:#fafafa}}
 h1{{font-size:27px;border-bottom:3px solid #2a9d8f;padding-bottom:10px;color:#1d3a3a}}
 h2{{font-size:21px;color:#264653;margin-top:40px;border-left:5px solid #2a9d8f;padding-left:12px}}
 h3{{font-size:17px;color:#2a9d8f;margin-top:26px}}
 .meta{{color:#777;font-size:13px}}
 .abs{{background:#eef7f5;border:1px solid #cfe7e2;border-radius:8px;padding:18px 22px;margin:20px 0}}
 .card{{background:#fff;border:1px solid #e4e4e4;border-radius:8px;padding:16px 20px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
 .key{{background:#eef7f5;border-left:4px solid #2a9d8f;padding:12px 16px;margin:14px 0;border-radius:4px}}
 .warn{{background:#fdf3ec;border-left:4px solid #e76f51;padding:12px 16px;margin:14px 0;border-radius:4px}}
 .todo{{background:#f4f4f4;border-left:4px solid #999;padding:12px 16px;margin:14px 0;border-radius:4px}}
 table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}}
 th,td{{border:1px solid #ddd;padding:7px 10px;text-align:center;vertical-align:middle}}
 th{{background:#264653;color:#fff}}
 tr:nth-child(even){{background:#f6f6f6}}
 img{{max-width:100%;border-radius:6px;border:1px solid #e0e0e0;margin:8px 0}}
 figure{{margin:18px 0}}
 figcaption{{font-size:13px;color:#666;text-align:center;margin-top:6px}}
 code{{background:#eee;padding:1px 5px;border-radius:3px;font-size:13px}}
 .pill{{display:inline-block;background:#2a9d8f;color:#fff;border-radius:12px;padding:1px 10px;font-size:12px}}
 .pill.todo{{background:#999}} .pill.diag{{background:#e76f51}}
 .plain{{background:#fff7ec;border:1px solid #f0d9b5;border-left:4px solid #e9a23b;border-radius:6px;padding:12px 16px;margin:14px 0}}
 .plain b{{color:#b5701a}}
 .readfig{{background:#f2f7fb;border:1px solid #d3e3ef;border-radius:6px;padding:10px 16px;margin:8px 0 18px;font-size:13.5px}}
 .readfig b{{color:#2b6ca3}}
 .toc{{background:#264653;color:#fff;border-radius:8px;padding:14px 22px;margin:18px 0;font-size:14px}}
 .toc a{{color:#8ed1c7;text-decoration:none;margin-right:6px}} .toc a:hover{{text-decoration:underline}}
 .ref{{font-size:13.5px;color:#333}} .ref li{{margin:6px 0}}
 .strike{{color:#999}}
</style></head><body>

<h1>单目无人机土方量测：VGGT 失效模式诊断与鲁棒量测框架</h1>
<p class="meta">实验展示 / 科研初稿 · 项目 vggt_yoloe（VGGT + YOLOe）· 整理于 2026-06-12 · 工作日志见 research_summary_20260611.html</p>

<div class="toc"><b>目录：</b>
<a href="#abs">摘要</a>·<a href="#intro">1 引言</a>·<a href="#related">2 相关工作</a>·
<a href="#method">3 方法与流程</a>·<a href="#setup">4 实验场景</a>·<a href="#results">5 关键结果</a>·
<a href="#disc">6 讨论</a>·<a href="#future">7 未完成工作</a>·<a href="#ref">参考文献</a>
</div>

<div class="abs" id="abs">
<h3 style="margin-top:0">摘要</h3>
<p>我们研究<b>从单目无人机视频直接量测挖掘机土方（挖/填）体积</b>的问题。几何主体采用现成的前馈重建模型
<b>VGGT</b>[1]（CVPR'25 最佳论文），语义由开放词表分割 <b>YOLOe</b> 提供。VGGT 输出是<b>无尺度、单次前馈、不含量测</b>的相对点云，
从这团点云到"可信的米制土方量"是一整层 VGGT 不负责的问题——这一层是本工作的贡献所在。
我们(i)用<b>语义尺度锚定</b>把尺度方差压低 15–40×；(ii)<b>诊断出两类会破坏下游量测的失效模式</b>：
深度不连续处的<b>水平光晕</b>（footprint 被吹大 3.5×，置信/一致性滤波可去）与<b>系统性竖直压缩 ~8×</b>
（挖掘机 3.0m 仅重建 0.38m，<b>两个预测头一致、多帧聚合也不恢复</b>，因此去光晕与多帧都治不了）；
(iii)给出从视频到体积的完整框架与可复用代码。
传统多视图 UAV 摄影测量在地面控制点(GCP)/RTK 下可达 <1% 的体积误差[5,6]；本工作的目标是在
<b>单目、无 GCP、单次前馈</b>的更弱条件下逼近可用精度，核心障碍是尺度模糊与竖直压缩，已分别定位并给出纠正路线。
进一步地，我们证明竖直压缩是<b>深度无关的仿射畸变</b>(E0)、且<b>跨场景高度稳定可复现(波动 1–2%)</b>(Q2)——
意味着纠正只需<b>低自由度全局变换</b>而非逐像素学习场，并使<b>预标定 / 学习式纠正器</b>成为可行路线。</p>
</div>

<h2 id="intro">1 引言：动机与困境</h2>
<p>挖掘机土方量是施工计量、进度与结算的核心指标。常规做法是多视图 UAV 摄影测量/LiDAR + DEM 体积积分，
精度高但依赖多航带重叠影像、GCP/RTK 与离线 SfM 重建[5,6]。我们想探索一条更轻的路径：
<b>单目视频 + 前馈大模型一次重建</b>。但这带来三个困境：</p>
<ul>
<li><b>(a) 尺度</b>：VGGT/DUSt3R 类模型输出相对尺度，体积无法直接换算米制；</li>
<li><b>(b) 精度</b>：一切几何都依赖 VGGT，其误差结构未知；</li>
<li><b>(c) 工作量/贡献</b>：若主体是现成 VGGT，本工作的科学贡献在哪？</li>
</ul>
<div class="key"><b>定位：</b>VGGT 给的是一团"无尺度、有噪声、单次前馈"的相对点云。
<b>从这团点云到"可信的米制量测"是一整层 VGGT 不负责的问题——这一层就是本工作的地盘。</b>
我们对每个模块都给出相对 raw-VGGT 基线的可量化 Δ，并发现+量化+纠正失效模式，构成完整研究闭环。</div>

<h2 id="related">2 相关工作与对比</h2>
<p><b>前馈三维重建：</b>DUSt3R[2] 以 pointmap 回归统一了无位姿多视图重建；VGGT[1] 进一步用大 Transformer 单次前馈
预测相机/深度/点图/点轨迹，<1 秒重建且在多任务 SOTA，并指出"由深度+相机间接得到的点图常优于直接点图头"。
<b>单目深度：</b>Depth Anything V2[3]、MoGe 等给出强泛化的相对/度量深度。这些模型共有的软肋是
<b>绝对尺度模糊</b>与在斜视/边界处的几何偏差——正是土方量测的痛点。</p>
<p><b>UAV 土方/料堆体积：</b>成熟的摄影测量管线在 GCP/RTK 下体积误差可低至 0.3–1% 甚至更小[5,6]，但需要多航带、控制点与离线优化。</p>
<table>
<tr><th>维度</th><th>传统 UAV 摄影测量 (SfM+MVS)</th><th>本工作：单目前馈 (VGGT)</th></tr>
<tr><td>输入</td><td>多航带重叠影像 + GCP/RTK</td><td>单目斜视视频，无 GCP</td></tr>
<tr><td>重建</td><td>离线 SfM/MVS（分钟~小时）</td><td>单次前馈（&lt;秒级）</td></tr>
<tr><td>尺度</td><td>由 GCP/RTK 给定绝对米制</td><td><b>无尺度</b> → 需语义锚定</td></tr>
<tr><td>体积精度</td><td><b>&lt;1%</b>（成熟）[5,6]</td><td>待定；受尺度模糊 + <b>竖直压缩~8×</b> 制约</td></tr>
<tr><td>主要风险</td><td>植被/弱纹理、控制点布设</td><td><b>竖直压缩、深度不连续光晕</b></td></tr>
</table>
<div class="key"><b>研究缺口：</b>前馈单目重建是否、以及在多大程度上可用于<b>米制土方量测</b>，其失效模式与纠正方法此前未被系统刻画。本工作针对这一缺口。</div>

<h2 id="method">3 方法与完整流程</h2>
<p>完整管线分八步，从视频到米制体积；绿色为已完成/复用模块，橙色为已诊断的失效模式，灰色虚线为待完成项。</p>
<figure><img src="data:image/png;base64,{framework}">
<figcaption>图 1 · 技术框架：视频 → VGGT 前馈重建 → 重力对齐 → 语义分割 → 尺度锚定 → 失效模式诊断 → 鲁棒地形/体积 → 米制土方量。</figcaption></figure>
<h3>逐步说明</h3>
<ol>
<li><b>输入</b>：无人机斜视（~66° 离正下方）单目视频；画面内含已知尺寸语义物（人/挖掘机）作天然标尺。</li>
<li><b>VGGT 前馈重建</b>[1]：26 帧一次前馈，输出每帧 <code>world_points</code>（pointmap 头）、<code>world_points_from_depth</code>（depth 头）、相机内外参、逐点置信度；全局共享世界系但<b>无米制尺度</b>。</li>
<li><b>重力对齐</b>（<code>gravity_alignment.py</code>）：对相机中心做 PCA 估计重力方向，旋转到 +Y=上；轨迹法向与点云主平面法向夹角仅 <b>0.2°</b>，对齐可靠。</li>
<li><b>YOLOe 开放词表分割</b>：得到挖掘机/人/地面语义掩膜，用于锚定与剔除动态物体。</li>
<li><b>语义尺度锚定</b>（<code>scale_calibration.py</code>）：用已知尺寸语义物的 mask 约束 3D 采样求米制 scale，较 naive 两端点法把方差压低 15–40×（见 5.1）。</li>
<li><b>失效模式诊断</b>：(a) 深度不连续处<b>水平光晕</b>；(b) 系统性<b>竖直压缩 ~8×</b>（见 5.2–5.3）。</li>
<li><b>鲁棒地形/DEM + 体积</b>：去光晕 + 多帧补覆盖已验证有效；<b>竖直纠正</b>与<b>大坑 GT 标定</b>为待完成核心项。</li>
<li><b>输出</b>：米制 cut/fill 体积，目标对标 UAV 摄影测量 &lt;1%。</li>
</ol>

<h2 id="setup">4 实验场景与数据</h2>
<figure><img src="data:image/png;base64,{scene_img}">
<figcaption>图 2 · 测试场景（<code>session_20260611_162643_869764</code>）：一台小型挖掘机、一个站立的人（天然 ~1.7m 标尺）、右下角一个<b>小坑</b>。
已与现场核实：该视频确实仅含此一个小坑，且自动坑判定的位置正确（见 5.4）。</figcaption></figure>
<p>VGGT 输出：26 帧，每帧 294×518；键含 <code>world_points (S,294,518,3)</code>、<code>world_points_conf</code>、
<code>world_points_from_depth</code>、<code>depth</code>、<code>extrinsic/intrinsic</code>、<code>images</code>。
绝对尺度暂用人身高 1.70m 标定（<b>provisional</b>），故绝对米制数字为暂定值，<b>比例/相对结论为硬结论</b>。</p>

<h2 id="results">5 关键结果</h2>

<h3>5.1 语义尺度锚定 <span class="pill">已完成</span></h3>
<p>对比"两端点 naive 锚定"与"语义 mask 约束采样"在 ±2px 抖动下的尺度稳定性：</p>
<figure><img src="data:image/png;base64,{robust}">
<figcaption>图 3 · naive 两端点法尺度跨度 83%；语义 mask 约束把方差压到 2–5%（15–40×）。
一阶关系 <b>体积相对误差 ≈ 3 × 尺度相对误差</b>，故尺度稳定性对方量至关重要。</figcaption></figure>

<h3>5.2 失效模式①：深度不连续处的水平光晕 <span class="pill diag">已诊断</span></h3>
<p>一度误判为"VGGT 压扁竖直"；置信度扫描纠正了它——<b>竖直 Y 几乎不动，坍塌的是水平宽度</b>（看着扁是分母被吹大）。</p>
<figure><img src="data:image/png;base64,{sweep}">
<figcaption>图 4 · 随置信度收紧，竖直 Y extent 基本不变，水平 Wmin 从 0.233 坍到 0.10 以下，aspect 从 0.58 回到 canonical≈1.3。</figcaption></figure>
<figure><img src="data:image/png;base64,{halo_img}">
<figcaption>图 5 · 光晕可视化：左=图像空间红色光晕点勾在轮廓边缘；右=俯视，绿色可靠核心紧致，红色光晕沿地面横向喷射。
三重确认：光晕点深度梯度 2.5×、corr(置信度,∇depth)=−0.32、Z 向 footprint ↑3.5×。机理：边界像素深度不可靠，沿视线落到地面形成单向水平裙边。</figcaption></figure>

<h3>5.3 失效模式②：竖直压缩 ~8× 且多帧不恢复 <span class="pill diag">已诊断</span></h3>
<p>叠加在光晕之上的<b>第二个失效模式</b>：绝对竖直被系统性压缩。以挖掘机为干净探针（真高 ~3.0m）：</p>
<figure><img src="data:image/png;base64,{vert_img}">
<figcaption>图 6 · A 侧视重建仅 ≈0.38m vs 真 3.0m；B pointmap(0.38m)与 depth(0.37m)两头一致压缩；
C 各帧顶高 0.01–0.41m，多帧融合 0.40m ≈ 最好单帧、离 3m 仍差 ~8×；D 多帧把点数 4135→66636（覆盖 ↑16×）。</figcaption></figure>
<table>
<tr><th>结论</th><th>数字</th><th>含义</th></tr>
<tr><td>竖直压缩 ≈ 7.9×</td><td>重建 0.38m vs 真 3.0m</td><td>重力已验证 0.2°，非对齐 bug</td></tr>
<tr><td>两个头一致压缩</td><td>pointmap 0.378m / depth 0.372m</td><td>depth 再三角化救不回</td></tr>
<tr><td>多帧不恢复高度</td><td>融合 0.40m = 最好单帧</td><td>压缩焊死在单次联合推理</td></tr>
<tr><td>多帧只补覆盖</td><td>4135 → 66636 点（↑16×）</td><td>补洞/补覆盖有效，治不了高度</td></tr>
</table>
<div class="warn"><b>对量测的意义：</b>多帧聚合能提覆盖、补遮挡空洞，但无法修复竖直压缩；要量准土方必须有<b>显式竖直纠正</b>
（语义锚定反压缩，或绕开 world_points 的基线几何重三角化）。</div>

<h3>5.4 坑判定：方法与正确性 <span class="pill">已验证</span></h3>
<p>判断坑的方法：拟合参考地平面后逐像素计算 <code>depression = ground_Y − Y</code>（正=低于地面=坑）。</p>
<figure><img src="data:image/png;base64,{pit_img}">
<figcaption>图 7 · 坑判定：A 候选 ROI（黄框，右下角）；B 全场竖直跨度&lt;0.4m；C depression 图，红=坑、本场景几无红，&gt;0.2m 坑像素=0；D 横切显示地表紧贴参考地面。</figcaption></figure>
<div class="key"><b>与现场核实一致：</b>该视频确实仅有一个<b>小坑</b>，自动判定的位置正确。坑深小于 VGGT ~8× 竖直压缩的噪声底，
故"无<b>可测</b>坑"是真实情况而非方法错误——这恰好印证了竖直压缩对小尺度土方量测的致命性。</div>

<h3>5.5 竖直纠正初探：根因定性(E0) + 单标量纠正(E1a) <span class="pill">进行中</span></h3>

<div class="plain"><b>先用大白话说清楚这一节在干嘛。</b>
前面 5.3 发现：VGGT 把所有竖直的东西都压扁了约 8 倍（挖掘机 3m 重建成 0.38m）。
<b>为什么这对土方量是致命的？</b>坑的深度也是"竖直方向"的量——竖直压扁 8 倍，坑就显得浅 8 倍，算出来的方量也跟着错。所以<b>想量准土方，必须先把竖直压扁修回来</b>。<br>
要修，得先搞清楚<b>压扁是"有规律"还是"乱来"</b>，这决定修法难易：
<ul style="margin:6px 0">
<li><b>情况A（好修）：均匀压扁</b>——像复印机把竖直方向统一缩到 12%，不管东西远近都一样。那只要"反向放大"乘回去就行（一个数）。术语叫<b>仿射 / affine</b>。</li>
<li><b>情况B（难修）：压扁程度随距离变</b>——近处压一点、远处压更多，没有统一比例。那得按位置逐点纠正，可能要训一个模型。术语叫<b>投影 / projective</b>。</li>
</ul>
<b>E0 这个实验，就是去判断到底是 A 还是 B。</b>（文献支持：不确定相机的前馈重建本来就只能恢复到差一个这种形变[7]，且斜视小基线的航空场景正是它最容易出错的地方[8]。）</div>

<p>判断方法：拿两个<b>已知真实高度</b>的物体当"标尺"——挖掘机(3.0m)和人(1.7m)，量它们被重建成多高，算<b>压缩因子 k = 真实高度 ÷ 重建高度</b>。再把挖掘机<b>切成"离相机近的一半"和"远的一半"</b>，分别算 k：如果近、远的 k 一样，就是情况A；如果差很多，就是情况B。</p>

<figure><img src="data:image/png;base64,{gauge_img}"></figure>
<div class="readfig"><b>图 8 怎么看：</b>
<b>左 / 中两图</b>是挖掘机、人的"侧视证件照"——<span style="color:#c0392b">红色竖条</span>=它们真实该有的高度(3.0m / 1.7m)，底部那一小坨彩色点=VGGT 实际重建出来的（才 0.38m / 0.16m），<b>红条和小坨之间的巨大落差就是被压扁的量</b>。（黑/橙箭头是次要诊断：物体重建朝向 vs 相机视线方向。）<br>
<b>右图是关键结论</b>：四根柱子是压缩因子 k。重点比"<b>exc-near</b>"和"<b>exc-far</b>"（挖掘机近/远两半）——它俩几乎一样高(7.9 vs 8.0)，<b>说明压扁程度跟远近无关 → 是情况A（均匀/仿射），好修</b>。但"excavator"和"person"两根不一样高(7.9 vs 10.8)，说明<b>光靠一个数还不够精确</b>。</div>

<p>既然 E0 说是"情况A（均匀压扁）"，<b>E1a 就试最简单的修法</b>：绕地面把所有高程<b>统一放大 k 倍</b>（公式 <code>Y' = 地面 + k×(Y − 地面)</code>）。
关键是验证它<b>通不通用</b>：用挖掘机标定出 k，然后拿这个 k 去修人，看修完的人高度对不对（这叫<b>留一验证</b>——用一个物体定标、另一个物体检验）。</p>

<figure><img src="data:image/png;base64,{e1a_img}"></figure>
<div class="readfig"><b>图 9 怎么看：</b>每幅图里 <span style="color:#999">灰点</span>=修之前(矮)，<span style="color:#2a9d8f">绿点</span>=放大 k 倍之后(高)，<span style="color:#c0392b">红条</span>=真实高度。<br>
<b>左（挖掘机，用来定标）</b>：灰 0.38m 被放大成绿 3.0m，正好顶到红条——因为 k 就是用它算的，必然对齐。<br>
<b>中（人，用来检验）</b>：同一个 k 把人从 0.16m 放大到 1.25m，但真实是 1.7m，<b>绿点没够到红条，差了 27%</b>——这就是"一个数不够通用"的证据。<br>
<b>右（汇总柱状）</b>：红=真实、灰=VGGT 原始、绿=纠正后。挖掘机 +0%(定标物)、人 −27%(检验物)。</div>

<div class="key"><b>E0 + E1a 一句话结论：</b>压扁是<b>"均匀的"（仿射、与远近无关）</b>，所以<b>修法不需要训练复杂的逐点模型，乘一个全局的几何变换就行</b>——这大大降低了"训大模型"的必要性。
但<b>只乘一个数还差 ~15–30%</b>，需要升级成<b>低自由度的仿射（几个参数）并配几个已知高度的标尺</b>来定准。
→ 给下次大坑实验的明确清单：<b>在坑周围插 3–4 根已知高度的标定杆</b>，就能把这个纠正定标准、并验证土方精度。</div>

<h3>5.6 纠正模型该用哪种？(E1b) <span class="pill">进行中</span></h3>
<div class="plain"><b>这一节在问：</b>既然要"放大高度"，到底按什么公式放大？我们比了三种由真实高度 H 推算重建高度 h 的模型：
<b>M0</b> 只用挖掘机定一个倍数；<b>M1</b> 用两个物体拟合一条过原点的比例线；<b>M2</b> 允许有个"固定高度损失"的截距(h = 斜率·H − 0.13m)。
M2 能解释"<b>越小的物体压得越狠</b>"——因为固定损失对小个子占比更大。</div>
<figure><img src="data:image/png;base64,{e1b_img}"></figure>
<div class="readfig"><b>图 10 怎么看：</b>
<b>左</b>=真实高度(横)对重建高度(竖)，两个橙点(挖掘机/人)都远在"理想黑虚线"下方=被压得很惨；三条拟合线几乎贴在底部。
<b>中</b>=两物体各自的压缩倍数 k：挖掘机 7.9、人 10.8，<b>小的压更狠</b>。
<b>右(关键)</b>=把一个"重建出来只有这么深"的特征反推成"真实多深"：三条线<b>在控制物附近(0.16–0.38m)重合</b>，<b>一旦外推就分叉</b>（如反推一个很浅的特征，不同模型差好几倍）。</div>
<div class="key"><b>结论：</b>2 个控制物只能把曲线"钉"在它们附近，<b>越往外推越不确定</b>。而真实坑深往往比这两个物体更深、落在外推区——所以
<b>大坑实验的标定杆必须"高度跨度大、覆盖目标坑深量程"</b>（矮/中/高 3–4 种），不能只插一种高度。</div>

<h3>5.7 这个畸变稳定吗？能不能"学"出来？(Q2) <span class="pill">已验证</span></h3>
<div class="plain"><b>这是决定整条路线的问题：</b>如果 VGGT 的压扁<b>每次飞行都不一样</b>，那只能每次现场插标定杆硬标；
如果它是<b>稳定、可复现</b>的，就能<b>预先标定一次、甚至训一个小模型自动纠正</b>（不必每次都标）。我们用已有的 10 个场景的探针结果来检验稳定性。</div>
<figure><img src="data:image/png;base64,{q2sig_img}"></figure>
<div class="readfig"><b>图 11 怎么看：</b>纵轴是"畸变签名"(竖直被压相对水平的程度，数值越大越扁)。
<b>绿点(挖掘机)在 10 个场景里几乎连成一条水平线</b>(2.20–2.34，波动仅 1.9%)→ <b>极其稳定</b>；
黄点(人)上下乱跳——右图说明这些乱跳的都来自"小 mask"的场景(人太小、点太少不可靠)。</div>
<figure><img src="data:image/png;base64,{q2ratio_img}"></figure>
<div class="readfig"><b>图 12 怎么看（更干净的佐证）：</b>纵轴是挖掘机与人的<b>竖直高度之比 R</b>(竖直/竖直，几乎不含光晕干扰)。
当人足够大、探针可靠时(绿点)，<b>R 在 5 个场景里稳定在 3.00，波动仅 1.1%</b>；红虚线 1.76 是"若压缩均匀"该有的值——
实测 3.00 高于它，说明<b>人(小个子)比挖掘机多压了 1.70 倍</b>，再次印证 5.6 的"小物体压更狠"。</div>
<div class="key"><b>Q2 结论（路线判定）：</b>VGGT 的竖直畸变<b>跨场景高度稳定(波动 1–2%)、系统可复现</b>——
所以<b>"方向二：预标定一次 / 训一个小纠正器"是可行的</b>，不必每次飞行重新标定。
<b>诚实边界：</b>这里证的是"<b>相对/签名</b>稳定"；要证"<b>绝对</b>压缩倍数也稳定"需要每个场景的尺度，而绝对倍数无法只靠竖直已知量标定(尺度与压缩纠缠)，须<b>外部水平基准或遥测</b>——即大坑标定杆要补的。</div>

<h2 id="disc">6 讨论</h2>
<p>两类失效模式性质不同、需不同对策：<b>光晕</b>是边界处低置信外点，靠置信/空间一致性滤波可去，但要权衡
<b>覆盖度 vs 干净度</b>（过滤过狠会留不下足够建面点）；<b>竖直压缩</b>是 VGGT 预测本身的各向异性偏差，
两头一致、多帧不恢复，必须显式纠正。二者叠加解释了为何 naive 地把 VGGT 点云直接做 DEM/体积会系统性出错。</p>
<p><b>诚实的边界：</b>当前仍无法 100% 分清"竖直专属压缩"与"整体欠尺度"——scale 由人(测量方向偏水平)临时标定，
挖掘机水平宽度 ~0.6m 看着也偏小。需一个<b>已知水平尺寸的标定物</b>来分离各向异性 vs 均匀缩小。</p>

<h2 id="future">7 未完成工作（Open Problems）</h2>
<div class="todo">
<ol>
<li><b>大坑 GT 实验（下一步将做）</b>：现场<b>开挖一个已知体积的大坑</b>作绝对方量真值；并按 E1b 结论<b>撒 3–4 根"高度跨度大、覆盖目标坑深量程"的标定杆</b>（矮/中/高），用于过定竖直仿射、分清 M1/M2 并提供绝对标定的外部基准。<span class="pill todo">计划中</span></li>
<li><b>E1b-full 低自由度仿射（核心贡献②）</b>：E0 已证压缩为<b>深度无关的仿射</b>，E1b 已搭好模型框架；待多高度标定杆到位后过定拟合、把 E1a 的 ~30% 残差钉死。</li>
<li><b>方向二原型（Q2 已证可行）</b>：Q2 显示畸变跨场景稳定(CV 1–2%)，故可仿 AMB3R[9] 在<b>冻结 VGGT 特征</b>上训一个小头，<b>预测竖直纠正参数</b>——无控制物时也能自动纠正。</li>
<li><b>Q2-absolute / 遥测增强</b>：绝对压缩倍数无法只靠竖直已知量标定(尺度与压缩纠缠)，需外部水平基准；若能导出 DJI GPS/IMU/云台 → 绝对尺度+重力 → 度量 BA/重三角化，从原理上消 stretch[7]。</li>
<li><b>抗光晕滤波方法本体</b>：空间一致性/法向/置信度联合去裙边，给出覆盖度–干净度权衡曲线，对比钝百分位阈值基线。</li>
<li><b>backbone 复验</b>：在 Depth-Anything-3 / DUSt3R 上复跑同样探针，确认失效模式是 VGGT 特有还是单目前馈通病，决定贡献表述。</li>
<li><b>ExcaVol 基准</b>：构建带 GT 的单目土方量测评测集与协议（含上面的大坑实验）。</li>
<li><b>绝对尺度坐实</b>：用可信已知尺寸标定物替换临时人身高，多场景复算。</li>
</ol>
</div>

<h2 id="ref">参考文献</h2>
<ol class="ref">
<li>[1] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny. <b>VGGT: Visual Geometry Grounded Transformer</b>. CVPR 2025 (Best Paper). arXiv:2503.11651. <a href="https://arxiv.org/abs/2503.11651">arxiv.org/abs/2503.11651</a></li>
<li>[2] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, J. Revaud. <b>DUSt3R: Geometric 3D Vision Made Easy</b>. CVPR 2024. arXiv:2312.14132. <a href="https://arxiv.org/abs/2312.14132">arxiv.org/abs/2312.14132</a></li>
<li>[3] L. Yang, B. Kang, Z. Huang, et al. <b>Depth Anything V2</b>. NeurIPS 2024. arXiv:2406.09414. <a href="https://arxiv.org/abs/2406.09414">arxiv.org/abs/2406.09414</a></li>
<li>[4] YOLOe — 开放词表实时分割（项目所用，权重 HF: jameslahm/yoloe）。</li>
<li>[5] UAV 摄影测量料堆/土方体积评估（GCP/RTK 下误差 0.3–1%）：综述与案例。<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6721121/">GCP-free UAV stockpile volume (PMC6721121)</a></li>
<li>[6] 商用无人机方量精度实务（cm 级、&lt;1%）。<a href="https://www.propelleraero.com/blog/how-stockpile-volume-measurement-works-in-drone-surveying-with-propeller/">Propeller Aero</a></li>
<li>[7] D. Maggio, H. Lim, L. Carlone. <b>VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold</b>. 2025. arXiv:2505.12549 — 不确定相机重建只到 15-DOF 投影变换(含 stretch)，需 SL(4) 单应矫正。<a href="https://arxiv.org/abs/2505.12549">arxiv.org/abs/2505.12549</a></li>
<li>[8] X. Wu, S. Landgraf, M. Ulrich, R. Qin. <b>An Evaluation of DUSt3R/MASt3R/VGGT 3D Reconstruction on Photogrammetric Aerial Blocks</b>. 2025. arXiv:2507.14798 — 航空小基线/正射是失效区，VGGT 点云误差达 6m、位姿漂移 42m。<a href="https://arxiv.org/abs/2507.14798">arxiv.org/abs/2507.14798</a></li>
<li>[9] AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction. 2025. arXiv:2511.20343 — 在冻结 VGGT 特征上加 scale head 恢复度量尺度。<a href="https://arxiv.org/abs/2511.20343">arxiv.org/abs/2511.20343</a></li>
</ol>

<p class="meta" style="margin-top:40px">— 科研初稿 · 生成于 2026-06-12 · 代码：scale_calibration / gravity_alignment / vertical_recovery_study / viz_halo / build_showcase</p>
</body></html>"""

out = f"{VY}/experiment_showcase.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out, f"({len(HTML)//1024} KB)")
