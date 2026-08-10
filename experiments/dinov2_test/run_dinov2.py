"""
DINOv2 特征可视化 on excavator images.
两种输出:
  1) PCA 特征上色 (patch features -> 3D via SVD -> RGB)
  2) 特征相似度热力图 (点选铲斗/大臂 patch, 看哪些区域特征相似)
环境: vggt (torch 2.3.1). 权重经 torch.hub 下到本工程缓存, 不污染 home。
"""
import os, glob
import numpy as np
import torch
import torch.nn.functional as F
import cv2

ROOT = "/home/maomaoyu/WS/vggt_yoloe"
SESS = f"{ROOT}/workspaces/session_20260629_100116_092814"
OUT  = f"{ROOT}/experiments/dinov2_test"
os.makedirs(f"{OUT}/pca", exist_ok=True)
os.makedirs(f"{OUT}/sim", exist_ok=True)
# torch.hub 缓存到工程目录
os.environ["TORCH_HOME"] = f"{OUT}/torch_hub"
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("loading dinov2_vits14 (local source, weights from fbaipublicfiles) ...")
# 用 vggt 内置 dinov2 源码本地构建, 绕开 GitHub API 限流
import sys
sys.path.insert(0, "/home/maomaoyu/WS/vggt/vggt/dependency/dinov2")
WCACHE = f"{OUT}/torch_hub/checkpoints"
os.makedirs(WCACHE, exist_ok=True)
wpath = f"{WCACHE}/dinov2_vits14_pretrain.pth"
if not os.path.exists(wpath):
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
    print("downloading weights ..."); urllib.request.urlretrieve(url, wpath)
from dinov2.models.vision_transformer import vit_small
model = vit_small(patch_size=14, img_size=518, init_values=1.0,
                  block_chunks=0, num_register_tokens=0)
sd = torch.load(wpath, map_location="cpu")
model.load_state_dict(sd, strict=True)
model = model.to(device).eval()
P = 14  # patch size

MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

def load_img(path, target_w=518):
    bgr = cv2.imread(path); rgb = bgr[:,:,::-1].copy()
    h,w = rgb.shape[:2]
    # resize 到宽 518, 高取 14 的整数倍
    scale = target_w / w
    nw = target_w; nh = int(round(h*scale))
    nh = (nh//P)*P; nw = (nw//P)*P
    rgb_r = cv2.resize(rgb,(nw,nh))
    t = torch.from_numpy(rgb_r).float().permute(2,0,1)[None]/255.
    t = (t-MEAN)/STD
    return t.to(device), rgb_r, (nh//P, nw//P)

@torch.no_grad()
def feats(path):
    t, rgb_r, (gh,gw) = load_img(path)
    out = model.forward_features(t)
    f = out["x_norm_patchtokens"][0]        # (gh*gw, C)
    return f, rgb_r, gh, gw

def pca_rgb(f, gh, gw):
    fc = f - f.mean(0, keepdim=True)
    U,S,V = torch.linalg.svd(fc, full_matrices=False)
    proj = fc @ V[:3].T                     # (N,3)
    p = proj.reshape(gh,gw,3).cpu().numpy()
    for c in range(3):
        lo,hi = np.percentile(p[...,c],2), np.percentile(p[...,c],98)
        p[...,c] = np.clip((p[...,c]-lo)/(hi-lo+1e-9),0,1)
    return (p*255).astype(np.uint8)

imgs = sorted(glob.glob(f"{SESS}/images/*.png"))
print(f"images: {len(imgs)}")

# 缓存 f5 用于相似度图的取点
sim_targets = {"bucket":None, "boom":None}

for p in imgs:
    f, rgb_r, gh, gw = feats(p)
    idx = int(os.path.basename(p)[:6])
    # --- PCA ---
    rgbpca = pca_rgb(f, gh, gw)
    rgbpca = cv2.resize(rgbpca,(rgb_r.shape[1],rgb_r.shape[0]),interpolation=cv2.INTER_NEAREST)
    side = np.concatenate([rgb_r[:,:,::-1], rgbpca[:,:,::-1]],1)
    cv2.imwrite(f"{OUT}/pca/{idx:06d}_pca.png", side)
    if idx==5: f5,gh5,gw5,rgb5 = f,gh,gw,rgb_r

# --- 相似度图 on f5: 取铲斗端 & 大臂中部的 patch ---
# 用之前几何法已知: 铲斗在图左侧臂末端, 大臂在中部。按比例取 grid 坐标。
def sim_map(f, gh, gw, gy, gx, rgb_r, tag, idx=5):
    q = f[gy*gw+gx]                         # 目标 patch 特征
    fn = F.normalize(f, dim=1); qn = F.normalize(q,dim=0)
    sim = (fn @ qn).reshape(gh,gw).cpu().numpy()
    sim = np.clip((sim+1)/2,0,1)
    heat = cv2.applyColorMap((sim*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat,(rgb_r.shape[1],rgb_r.shape[0]),interpolation=cv2.INTER_LINEAR)
    blend = (0.5*rgb_r[:,:,::-1]+0.5*heat).astype(np.uint8)
    # 标记取点
    py = int((gy+0.5)/gh*rgb_r.shape[0]); px=int((gx+0.5)/gw*rgb_r.shape[1])
    cv2.drawMarker(blend,(px,py),(255,255,255),cv2.MARKER_CROSS,14,2)
    cv2.imwrite(f"{OUT}/sim/{idx:06d}_sim_{tag}.png", blend)
    print(f"sim map [{tag}] grid=({gy},{gx})")

# 铲斗: 左侧约 18% 宽、60% 高; 大臂: 约 38% 宽、45% 高 (按 f5 观察)
sim_map(f5,gh5,gw5, int(0.62*gh5), int(0.20*gw5), rgb5, "bucket")
sim_map(f5,gh5,gw5, int(0.42*gh5), int(0.40*gw5), rgb5, "boom")
sim_map(f5,gh5,gw5, int(0.45*gh5), int(0.62*gw5), rgb5, "body")

# PCA contact sheet
files = sorted(glob.glob(f"{OUT}/pca/*_pca.png"), key=lambda x:int(os.path.basename(x)[:6]))
tiles=[cv2.imread(f) for f in files]
h,w=tiles[0].shape[:2]; cols,pad,lab=2,6,18; rows=(len(tiles)+cols-1)//cols
sheet=np.full((rows*(h+lab+pad)+pad,cols*(w+pad)+pad,3),40,np.uint8)
for i,im in enumerate(tiles):
    rr,cc=divmod(i,cols); y=pad+rr*(h+lab+pad); x=pad+cc*(w+pad)
    cv2.putText(sheet,f"frame {i} | RGB | PCA",(x+4,y+13),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
    sheet[y+lab:y+lab+h,x:x+w]=im
cv2.imwrite(f"{OUT}/pca_contact_sheet.png",sheet)
print(f"wrote {OUT}/pca_contact_sheet.png")
print("done")
