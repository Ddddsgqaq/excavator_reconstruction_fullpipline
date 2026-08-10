"""
DINOv2 patch 特征 K-means 无监督分割 on 全场景图.
目的: 看 DINOv2 对整幅场景能无监督分出哪些类, 重点是能否区分地形.
对每帧扫一组 K=(4,6,8,10), 每帧独立聚类, 输出:
  1) seg/<idx>_k<K>.png     单帧单 K 的 [原图 | 上色分割] 并列图
  2) seg_grid/<idx>.png      单帧 4 个 K 的横排对比
  3) kmeans_contact_sheet_k<K>.png  某个 K 下所有帧的汇总
特征: 沿用 vits14 (和 PCA/相似度图同一份). K-means 用 torch 自己实现, 不依赖 sklearn.
环境: vggt (torch 2.3.1). DINOv2 沿用 CPU (该 GPU kernel 不兼容).
"""
import os, glob
import numpy as np
import torch
import cv2

ROOT = "/home/maomaoyu/WS/vggt_yoloe"
SESS = f"{ROOT}/workspaces/session_20260629_100116_092814"
OUT  = f"{ROOT}/experiments/dinov2_test"
os.makedirs(f"{OUT}/seg", exist_ok=True)
os.makedirs(f"{OUT}/seg_grid", exist_ok=True)
os.environ["TORCH_HOME"] = f"{OUT}/torch_hub"

# DINOv2 沿用 CPU (和之前实验一致, 避开该 GPU 的 kernel 不兼容)
device = "cpu"
KS = [4, 6, 8, 10]

print("loading dinov2_vits14 (local source) ...")
import sys
sys.path.insert(0, "/home/maomaoyu/WS/vggt/vggt/dependency/dinov2")
wpath = f"{OUT}/torch_hub/checkpoints/dinov2_vits14_pretrain.pth"
from dinov2.models.vision_transformer import vit_small
model = vit_small(patch_size=14, img_size=518, init_values=1.0,
                  block_chunks=0, num_register_tokens=0)
model.load_state_dict(torch.load(wpath, map_location="cpu"), strict=True)
model = model.to(device).eval()
P = 14

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def load_img(path, target_w=518):
    bgr = cv2.imread(path); rgb = bgr[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    scale = target_w / w
    nw = target_w; nh = int(round(h * scale))
    nh = (nh // P) * P; nw = (nw // P) * P
    rgb_r = cv2.resize(rgb, (nw, nh))
    t = torch.from_numpy(rgb_r).float().permute(2, 0, 1)[None] / 255.
    t = (t - MEAN) / STD
    return t.to(device), rgb_r, (nh // P, nw // P)

@torch.no_grad()
def feats(path):
    t, rgb_r, (gh, gw) = load_img(path)
    out = model.forward_features(t)
    f = out["x_norm_patchtokens"][0]        # (gh*gw, C)
    return f, rgb_r, gh, gw

def kmeans_torch(x, k, iters=50, seed=0):
    """torch 版 K-means. x:(N,C) 已 L2 归一化建议. 返回 labels:(N,)"""
    N = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    # k-means++ 简化: 随机取第一个, 其余按到已选中心的最远概率取
    idx = [int(torch.randint(0, N, (1,), generator=g))]
    for _ in range(1, k):
        c = x[idx]                                  # (m,C)
        d2 = torch.cdist(x, c).min(1).values ** 2   # (N,) 到最近中心的平方距离
        probs = d2 / (d2.sum() + 1e-9)
        idx.append(int(torch.multinomial(probs, 1, generator=g)))
    cent = x[idx].clone()                           # (k,C)
    labels = torch.zeros(N, dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(x, cent)                    # (N,k)
        new = d.argmin(1)
        if torch.equal(new, labels):
            labels = new; break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                cent[j] = x[m].mean(0)
    return labels

# 固定调色板, 保证同一 K 内颜色稳定 (BGR)
PALETTE = np.array([
    [ 60, 76,231],[231,180, 22],[ 96,174, 39],[199, 86,224],
    [ 39,127,255],[220,220, 60],[180,120,255],[ 90,200,200],
    [255,120, 40],[ 40,220,140],[200, 60,120],[120,120,120],
], dtype=np.uint8)

def colorize(labels, gh, gw, rgb_r):
    lab = labels.reshape(gh, gw).cpu().numpy()
    seg = PALETTE[lab % len(PALETTE)]                       # (gh,gw,3)
    seg = cv2.resize(seg, (rgb_r.shape[1], rgb_r.shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    blend = (0.45 * rgb_r[:, :, ::-1] + 0.55 * seg).astype(np.uint8)
    return blend

imgs = sorted(glob.glob(f"{SESS}/images/*.png"))
print(f"images: {len(imgs)}")

by_k = {k: [] for k in KS}                                  # 收集每帧每 K 的并列图, 供 contact sheet

for p in imgs:
    idx = int(os.path.basename(p)[:6])
    f, rgb_r, gh, gw = feats(p)
    fn = torch.nn.functional.normalize(f, dim=1)            # 余弦几何下聚类
    panels = []                                             # 该帧 4 个 K 的分割图 (纯分割, 不含原图)
    for k in KS:
        lab = kmeans_torch(fn, k, seed=0)
        blend = colorize(lab, gh, gw, rgb_r)
        side = np.concatenate([rgb_r[:, :, ::-1], blend], 1)  # 原图 | 分割
        cv2.imwrite(f"{OUT}/seg/{idx:06d}_k{k}.png", side)
        by_k[k].append(side)
        # 给 grid 用: 分割图上标 K
        tag = blend.copy()
        cv2.putText(tag, f"K={k}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        panels.append(tag)
    # 单帧 4-K 横排 (最左放原图)
    row = np.concatenate([rgb_r[:, :, ::-1]] + panels, 1)
    cv2.imwrite(f"{OUT}/seg_grid/{idx:06d}.png", row)
    print(f"frame {idx}: grid={gh}x{gw}, done K={KS}")

# 每个 K 一张 contact sheet
for k in KS:
    tiles = by_k[k]
    h, w = tiles[0].shape[:2]; cols, pad, lab = 2, 6, 18
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (h + lab + pad) + pad, cols * (w + pad) + pad, 3), 40, np.uint8)
    for i, im in enumerate(tiles):
        rr, cc = divmod(i, cols); y = pad + rr * (h + lab + pad); x = pad + cc * (w + pad)
        cv2.putText(sheet, f"frame {i} | RGB | K={k}", (x + 4, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sheet[y + lab:y + lab + h, x:x + w] = im
    cv2.imwrite(f"{OUT}/kmeans_contact_sheet_k{k}.png", sheet)
    print(f"wrote kmeans_contact_sheet_k{k}.png")
print("done")
