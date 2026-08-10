#!/usr/bin/env python
"""
test_terrain_vlm.py — 独立测试：把地形表示喂给 VLM，输出挖掘机的结构化决策。

思路（验证“地形表示是否足以支撑决策”）：
  1. 读某个 session 的 predictions.npz → analyze_terrain 得到 work-zone 分层 + 区域 + 下铲点。
  2. 渲染 BEV 作业地图 PNG（give the VLM a picture）。
  3. 把「BEV 图 + 精简的结构化上下文(zones/machines/next_scoop/参数)」一起发给 VLM，
     要求它返回**结构化决策 JSON**：地形理解 + 作业顺序 + 风险 + 下一步动作。
  4. 打印并保存 decision.json。

VLM 走 OpenAI 兼容接口（默认通义千问 qwen3-vl-plus，可换）。
不接入服务/界面，纯命令行测试。

用法：
  python test_terrain_vlm.py <predictions.npz> \
      [--model qwen3-vl-plus] [--scale 28.0] [--tau-frac 0.1] \
      [--out-dir vlm_tests] [--base-url ...] [--api-key ...]
env 也可提供：VLM_BASE_URL / VLM_API_KEY
"""
import argparse, base64, json, os, urllib.request
import numpy as np

from gravity_alignment import estimate_gravity
import terrain_analysis as ta

DEFAULT_BASE = "https://api.silra.cn/v1/chat/completions"

# ── 让 VLM 返回的结构化决策 schema（写进 prompt 里约束）────────────────────────
DECISION_SCHEMA = {
    "terrain_summary": "一句话概述当前地形（有几种作业区、整体走向）",
    "zones_readout": [{"zone": "dig|dump|pile|flat|hazard|obstacle",
                       "note": "该区在场内的位置与作业含义"}],
    "machines": [{"label": "类名", "note": "位置/朝向的作业含义"}],
    "risks": ["风险点描述（陡坎/障碍临近/坑壁失稳等）"],
    "work_order": ["建议的作业步骤，按先后顺序"],
    "next_action": {
        "action": "dig|move|dump|idle",
        "target_xz": [0.0, 0.0],
        "heading_deg": 0.0,
        "reason": "为什么是这个动作/这个点"
    },
    "confidence": "high|medium|low"
}


def analyze(npz_path, grid, tau_frac, conf=50.0):
    d = np.load(npz_path)
    p = {k: np.array(d[k]) for k in d.files}
    pts, cf, sem = p["world_points_from_depth"], p["depth_conf"], p.get("semantic_masks")
    gmask = (sem == 1) if sem is not None else None
    grav = estimate_gravity(extrinsic=p["extrinsic"], world_points=pts,
                            ground_mask=gmask, conf=cf, conf_thres=conf / 100.0)
    pf, cff = pts.reshape(-1, 3), cf.reshape(-1).astype(np.float32)
    keep = np.isfinite(pf).all(1) & (cff >= (conf / 100.0) * cff.max())
    pa = pf[keep] @ grav.R_align.T
    semf = sem.reshape(-1)[keep] if sem is not None else None
    gf = (semf == 1) if semf is not None else None

    # 读类名映射（从 session 的 yoloe_runs meta.json）
    id_to_name = _resolve_id_to_name(os.path.dirname(npz_path))
    res = ta.analyze_terrain(pa, semf, gf, id_to_name=id_to_name,
                             grid_res=grid, tau_frac=tau_frac)
    res["sem_id_map"] = {v: k for k, v in id_to_name.items()}
    return res


def _resolve_id_to_name(session_dir):
    import glob
    metas = sorted(glob.glob(os.path.join(session_dir, "yoloe_runs", "*", "meta.json")))
    for m in reversed(metas):
        try:
            j = json.load(open(m))
            if j.get("semantic_id_map"):
                return {int(v): k for k, v in j["semantic_id_map"].items()}
        except Exception:
            continue
    return {}


def compact_context(result, scale):
    """从完整结果里抽出给 VLM 的精简结构化上下文（不含大栅格数组，省 token）。"""
    ws = result["worksite"]
    g = result["grid"]
    sf = scale
    return {
        "scale": f"1 grid-unit = {sf} m",
        "grid_bounds_units": {k: round(g[k], 3) for k in ("x_min", "x_max", "z_min", "z_max")},
        "tau_units": round(result["params"]["tau"], 4),
        "zones_area_m2": {z["name"]: round(z["area"] * sf * sf, 2) for z in ws["zones"]},
        "machines": [{"label": m["label"],
                      "centroid_units": [round(c, 3) for c in m["centroid"]],
                      "area_m2": round(m["area"] * sf * sf, 2)} for m in ws["machines"]],
        "next_scoop_geom": (None if not ws["next_scoop"] else {
            "xz_units": [round(c, 3) for c in ws["next_scoop"]["xz"]],
            "zone": ws["next_scoop"]["zone"],
            "depth_m": round(ws["next_scoop"]["depth"] * sf, 3),
            "heading_deg": round(ws["next_scoop"]["heading_deg"], 1)
                if ws["next_scoop"]["heading_deg"] is not None else None}),
        "note": "geometry-derived next_scoop is a heuristic (deepest-pit edge); "
                "you may agree or propose a better action.",
    }


def call_vlm(base_url, api_key, model, image_path, context):
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    sys_prompt = (
        "You are the terrain-reasoning module of an autonomous excavator. "
        "You receive a top-down BEV work-zone map (image) plus structured geometry "
        "context. Reason about the site and output a SINGLE JSON object, no prose, "
        "matching exactly this schema:\n" + json.dumps(DECISION_SCHEMA, ensure_ascii=False)
        + "\nCoordinates are in grid-units (see scale). Be concrete and use the "
        "provided geometry; the heuristic next_scoop is a suggestion you may override."
    )
    user_text = ("Structured context:\n" + json.dumps(context, ensure_ascii=False, indent=2)
                 + "\n\nRead the BEV map and the context, then output the decision JSON.")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
            ]},
        ],
        "max_tokens": 1500,
        "temperature": 0.2,
    }
    req = urllib.request.Request(base_url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return r["choices"][0]["message"]["content"]


def parse_json_loose(text):
    """从可能带 ```json 包裹的返回里抽出 JSON。"""
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        # 退回：找第一个 { 到最后一个 }
        s, e = t.find("{"), t.rfind("}")
        return json.loads(t[s:e + 1]) if s >= 0 else {"_raw": text}


def render_decision_overlay(result, decision, out_path, title="", scale_factor=1.0):
    """在 BEV 作业地图上叠加 VLM 的决策：next_action 点+朝向、风险标注、
    右侧列 work_order/risks 文本。让「地形→VLM决策」一图可读。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle, FancyArrow
    from matplotlib.colors import ListedColormap, BoundaryNorm

    sf = float(scale_factor) if scale_factor else 1.0
    g = result["grid"]; res = g["res"]
    cx = (g["x_min"] + g["x_max"]) / 2.0; cz = (g["z_min"] + g["z_max"]) / 2.0
    ex0 = (g["x_min"] - cx) * sf; ex1 = (g["x_max"] - cx) * sf
    ez0 = (g["z_min"] - cz) * sf; ez1 = (g["z_max"] - cz) * sf
    to_m = lambda px, pz: ((px - cx) * sf, (pz - cz) * sf)

    zone = np.array(result["layers"]["zone_map"], dtype=np.int32)
    order = [ta.ZONE_FLAT, ta.ZONE_DIG, ta.ZONE_DUMP, ta.ZONE_PILE, ta.ZONE_HAZARD, ta.ZONE_OBSTACLE]
    present = [z for z in order if (zone == z).any()]
    cmap = ListedColormap([ta.ZONE_COLORS[z] for z in present])
    norm = BoundaryNorm(np.arange(len(present) + 1) - 0.5, cmap.N)
    remap = np.full(zone.shape, np.nan)
    for k, z in enumerate(present):
        remap[zone == z] = k

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    fig.suptitle(title or "VLM Decision on BEV", fontsize=14, color="#e6edf3")

    ax.imshow(np.ma.masked_invalid(remap), extent=[ex0, ex1, ez0, ez1],
              origin="lower", cmap=cmap, norm=norm, interpolation="nearest",
              alpha=0.92, zorder=1)
    stride = max(1, int(round(res / 32)))
    step = (ex1 - ex0) / res * stride
    for gx in np.linspace(ex0, ex1, res + 1)[::stride]:
        ax.axvline(gx, color="#30363d", lw=0.5, zorder=2)
    for gz in np.linspace(ez0, ez1, res + 1)[::stride]:
        ax.axhline(gz, color="#30363d", lw=0.5, zorder=2)

    ws = result["worksite"]
    # 机械黄框
    for mc in ws.get("machines", []):
        mx, mz = to_m(*mc["centroid"])
        side = max(np.sqrt(mc["area"]) * sf, step * 0.3)
        ax.add_patch(Rectangle((mx - side/2, mz - side/2), side, side,
                     fill=False, ec="#ffd400", lw=2, zorder=6))
        ax.text(mx, mz + side/2, mc["label"], color="#ffd400", fontsize=10,
                weight="bold", ha="center", va="bottom", zorder=7)

    # VLM 的 next_action（洋红星标 + 朝向），与几何 next_scoop 区分开
    na = (decision or {}).get("next_action") or {}
    tx = na.get("target_xz")
    if isinstance(tx, (list, tuple)) and len(tx) == 2:
        px, pz = to_m(float(tx[0]), float(tx[1]))
        ax.plot(px, pz, marker="*", color="#ff4dd2", ms=26, mec="#111", mew=1.5, zorder=8)
        ax.text(px, pz - step*0.2, f"VLM: {na.get('action','?')}", color="#ff4dd2",
                fontsize=11, weight="bold", ha="center", va="top", zorder=8,
                bbox=dict(boxstyle="round", fc="#161b22", ec="#ff4dd2", alpha=0.9))
        hd = na.get("heading_deg")
        if hd is not None:
            import math
            L = step * 1.2
            ax.add_patch(FancyArrow(px, pz, L*math.cos(math.radians(hd)),
                         L*math.sin(math.radians(hd)), width=step*0.04,
                         head_width=step*0.18, color="#ff4dd2", zorder=8,
                         length_includes_head=True))

    ax.set_xlim(ex0, ex1); ax.set_ylim(ez0, ez1); ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#e6edf3"); ax.set_ylabel("Z (m)", color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    for s in ax.spines.values(): s.set_color("#30363d")
    handles = [Patch(color=ta.ZONE_COLORS[z], label=ta.ZONE_NAMES[z]) for z in present]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=9, title="Zones",
                    framealpha=0.9, facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#e6edf3")
    leg.get_title().set_color("#e6edf3")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=115, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def run_one(npz, out_dir, model, grid, tau_frac, scale, base_url, api_key, tag=""):
    """跑单个 session：analyze → BEV → VLM → decision + overlay。返回 manifest 条目。"""
    os.makedirs(out_dir, exist_ok=True)
    result = analyze(npz, grid, tau_frac)
    bev_path = os.path.join(out_dir, "bev_for_vlm.png")
    ta.render_worksite_bev(result, bev_path, title=f"Worksite BEV {tag}", scale_factor=scale)

    ctx = compact_context(result, scale)
    json.dump(ctx, open(os.path.join(out_dir, "context_sent.json"), "w"),
              ensure_ascii=False, indent=2)
    raw = call_vlm(base_url, api_key, model, bev_path, ctx)
    decision = parse_json_loose(raw)
    json.dump(decision, open(os.path.join(out_dir, "decision.json"), "w"),
              ensure_ascii=False, indent=2)

    overlay_path = os.path.join(out_dir, "decision_overlay.png")
    render_decision_overlay(result, decision, overlay_path,
                            title=f"VLM Decision {tag}", scale_factor=scale)

    return {
        "session": tag or os.path.basename(os.path.dirname(npz)),
        "bev": bev_path,
        "overlay": overlay_path,
        "context": ctx,
        "decision": decision,
        "model": model,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="*", help="one or more predictions.npz (batch)")
    ap.add_argument("--model", default="qwen3-vl-plus")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--tau-frac", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=28.0)
    ap.add_argument("--out-dir", default="vlm_tests")
    ap.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", DEFAULT_BASE))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for i, npz in enumerate(args.npz):
        sess = os.path.basename(os.path.dirname(npz))
        sub = os.path.join(args.out_dir, sess)
        print(f"[{i+1}/{len(args.npz)}] {sess} ...", flush=True)
        try:
            entry = run_one(npz, sub, args.model, args.grid, args.tau_frac,
                            args.scale, args.base_url, args.api_key, tag=sess)
            manifest.append(entry)
            na = entry["decision"].get("next_action", {})
            print(f"    -> action={na.get('action')} @ {na.get('target_xz')} "
                  f"conf={entry['decision'].get('confidence')}")
        except Exception as e:
            print(f"    ERROR: {e}")

    man_path = os.path.join(args.out_dir, "manifest.json")
    json.dump(manifest, open(man_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nmanifest: {man_path}  ({len(manifest)} sessions)")

