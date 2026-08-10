#!/usr/bin/env python
"""
make_vlm_report.py — 读取 test_terrain_vlm.py 批量产出的 manifest.json，
生成一份自包含的 HTML 可视化报告（图片内嵌 base64，可整份拷走）。

每个 session 一张卡片：
  左 = BEV 作业地图 + VLM 决策叠加图
  右 = VLM 结构化决策（地形概述/分区/风险/作业顺序/下一步动作/置信度）
     + 发给 VLM 的结构化上下文（可折叠）

用法：
  python make_vlm_report.py vlm_report/manifest.json [--out vlm_report/report.html]
"""
import argparse, base64, html, json, os


def b64img(path):
    if not path or not os.path.exists(path):
        return ""
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def esc(x):
    return html.escape(str(x))


def li_list(items):
    if not items:
        return "<li class='muted'>—</li>"
    return "".join(f"<li>{esc(i)}</li>" for i in items)


def zones_rows(zr):
    if not zr:
        return "<tr><td class='muted' colspan=2>—</td></tr>"
    out = []
    for z in zr:
        out.append(f"<tr><td class='zone zone-{esc(z.get('zone',''))}'>"
                   f"{esc(z.get('zone',''))}</td><td>{esc(z.get('note',''))}</td></tr>")
    return "".join(out)


def card(entry):
    d = entry.get("decision", {})
    na = d.get("next_action", {}) or {}
    ctx = entry.get("context", {})
    conf = d.get("confidence", "-")
    return f"""
<section class="card">
  <h2>{esc(entry.get('session',''))} <span class="model">{esc(entry.get('model',''))}</span>
      <span class="conf conf-{esc(conf)}">confidence: {esc(conf)}</span></h2>
  <div class="row">
    <div class="imgs">
      <figure><img src="{b64img(entry.get('bev'))}"><figcaption>Worksite BEV (fed to VLM)</figcaption></figure>
      <figure><img src="{b64img(entry.get('overlay'))}"><figcaption>VLM decision overlay (magenta ★ = VLM next_action)</figcaption></figure>
    </div>
    <div class="info">
      <h3>Terrain summary</h3>
      <p>{esc(d.get('terrain_summary','—'))}</p>

      <h3>Next action</h3>
      <p class="action"><b>{esc(na.get('action','?'))}</b> @ {esc(na.get('target_xz','?'))}
         &nbsp;heading {esc(na.get('heading_deg','?'))}°</p>
      <p class="reason">{esc(na.get('reason',''))}</p>

      <h3>Zones readout</h3>
      <table>{zones_rows(d.get('zones_readout'))}</table>

      <h3>Work order</h3>
      <ol>{li_list(d.get('work_order'))}</ol>

      <h3>Risks</h3>
      <ul class="risks">{li_list(d.get('risks'))}</ul>

      <details><summary>Structured context sent to VLM</summary>
        <pre>{esc(json.dumps(ctx, ensure_ascii=False, indent=2))}</pre></details>
      <details><summary>Raw decision JSON</summary>
        <pre>{esc(json.dumps(d, ensure_ascii=False, indent=2))}</pre></details>
    </div>
  </div>
</section>"""


CSS = """
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;--acc:#58a6ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font-family:-apple-system,Segoe UI,Roboto,"PingFang SC","Microsoft YaHei",sans-serif;}
header{padding:22px 28px;border-bottom:1px solid var(--line);}
header h1{margin:0;font-size:20px}
header p{margin:6px 0 0;color:var(--mut);font-size:13px}
.card{margin:22px 28px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;}
.card h2{margin:0 0 14px;font-size:16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.model{color:var(--mut);font-size:12px;font-weight:400}
.conf{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600}
.conf-high{background:#12351f;color:#3fb950}.conf-medium{background:#3a2f10;color:#d29922}
.conf-low{background:#3a1214;color:#f85149}
.row{display:flex;gap:20px;flex-wrap:wrap}
.imgs{flex:1 1 420px;min-width:360px}
.imgs figure{margin:0 0 14px}
.imgs img{width:100%;border:1px solid var(--line);border-radius:6px;display:block}
.imgs figcaption{color:var(--mut);font-size:12px;margin-top:5px}
.info{flex:1 1 360px;min-width:320px}
.info h3{color:var(--acc);font-size:13px;margin:14px 0 5px;text-transform:uppercase;letter-spacing:.5px}
.info p{margin:0 0 6px;line-height:1.55;font-size:14px}
.action{font-size:15px}.reason{color:var(--mut);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}
td{border:1px solid var(--line);padding:5px 8px;vertical-align:top}
.zone{font-weight:600;white-space:nowrap;width:1%}
.zone-dig{color:#f59a2b}.zone-dump{color:#4d9bff}.zone-pile{color:#4ec86a}
.zone-flat{color:#c9c9c9}.zone-hazard{color:#f85149}.zone-obstacle{color:#9aa}
ol,ul{margin:4px 0;padding-left:20px}li{margin:3px 0;line-height:1.5;font-size:13.5px}
.risks li{color:#f0a58a}.muted{color:var(--mut)}
details{margin-top:10px}summary{cursor:pointer;color:var(--mut);font-size:12px}
pre{background:#0b0f14;border:1px solid var(--line);border-radius:6px;padding:10px;
 overflow:auto;font-size:11.5px;max-height:320px}
"""


def build(manifest, out_path, model_note=""):
    cards = "".join(card(e) for e in manifest)
    doc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLM Terrain Decision Report</title><style>{CSS}</style></head><body>
<header>
  <h1>VLM 地形理解 &amp; 决策报告</h1>
  <p>把「BEV 作业地图 + 结构化地形上下文」喂给 VLM，输出结构化挖掘决策。
     {esc(len(manifest))} 个 session · 模型 {esc(model_note)} ·
     洋红 ★ = VLM 建议下铲点，白 ★ = 几何启发式下铲点。
     注意：VLM 的定量数值可能有偏差，定性理解/作业逻辑更可信。</p>
</header>
{cards}
</body></html>"""
    open(out_path, "w").write(doc)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))
    out = args.out or os.path.join(os.path.dirname(args.manifest), "report.html")
    model = manifest[0]["model"] if manifest else ""
    build(manifest, out, model_note=model)
    print(f"report: {out}  ({len(manifest)} sessions)")
