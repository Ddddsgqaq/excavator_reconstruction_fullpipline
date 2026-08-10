"""
多关节运动状态分析 (扩展自单点 run_motion).
思路: 不依赖 approxPolyDP 顶点(跨帧不稳), 改沿骨架最长路径按*弧长比例*取固定锚点:
  J0 铲斗尖(0%) · J1 铲斗-小臂关节(~35%) · J2 肘(~70%) · J3 根(100%)
每个锚点 -> 局部窗口内 mask 像素的 3D 世界质心 -> 沿地面法向离地高度 -> vZ/aZ/state.
弧长比例比"第几个顶点"跨帧语义更稳, 是当前无训练前提下能做到的多关节速度估计。
环境: vggt (CPU)。
"""
import os, json
import numpy as np, cv2
from numpy.random import default_rng
from scipy.ndimage import label
from skimage.morphology import skeletonize
from collections import deque

ROOT="/home/maomaoyu/WS/vggt_yoloe"
SESS=f"{ROOT}/workspaces/session_20260629_173356_784627"
OUT =f"{ROOT}/experiments/arm_motion_state/joints"
os.makedirs(f"{OUT}/frames",exist_ok=True)

d=np.load(f"{SESS}/predictions.npz")
wp=d["world_points_from_depth"]; imgs=d["images"]
N,H,W=wp.shape[:3]; FPS=30.0/6.0; DT=1.0/FPS

# --- 垂直轴 = 地面法向 (与单点版一致) ---
def ground_normal(P,iters=600,thr=0.01):
    rng=default_rng(0); best=(0,None,None); s=P[rng.choice(P.shape[0],20000,replace=False)]
    for _ in range(iters):
        i=rng.choice(s.shape[0],3,replace=False); p0,p1,p2=s[i]
        n=np.cross(p1-p0,p2-p0); ln=np.linalg.norm(n)
        if ln<1e-9: continue
        n/=ln; o=-n.dot(p0); inl=(np.abs(s.dot(n)+o)<thr).sum()
        if inl>best[0]: best=(inl,n,o)
    return best[1],best[2]
ng,_=ground_normal(wp[5].reshape(-1,3)); off=-np.median(wp[5].reshape(-1,3).dot(ng))
h5=wp[5].reshape(-1,3).dot(ng)+off
if abs(np.percentile(h5,1))>abs(np.percentile(h5,99)): ng=-ng; off=-off
print("ground normal:",ng.round(3))

def hmap(f): return (wp[f].reshape(-1,3).dot(ng)+off).reshape(H,W)

# --- 骨架最长测地路径 (BFS 两次) ---
def longest_path(sk):
    pts=np.argwhere(sk>0)  # (y,x)
    if len(pts)<2: return None
    idx={(y,x):i for i,(y,x) in enumerate(pts)}
    nb=[[] for _ in pts]
    for i,(y,x) in enumerate(pts):
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                if dy==0 and dx==0: continue
                j=idx.get((y+dy,x+dx))
                if j is not None: nb[i].append(j)
    def bfs(s):
        dist=[-1]*len(pts); dist[s]=0; q=deque([s]); far=s
        while q:
            u=q.popleft()
            if dist[u]>dist[far]: far=u
            for v in nb[u]:
                if dist[v]<0: dist[v]=dist[u]+1; q.append(v)
        return far,dist
    a,_=bfs(0); b,dist=bfs(a)
    # 回溯 a..b
    path=[b]; cur=b
    while cur!=a:
        cur=min(nb[cur],key=lambda v:dist[v]); path.append(cur)
    path=path[::-1]
    return pts[path]  # (L,2) y,x along path

# --- 每帧: mask -> 骨架 -> 路径 -> 定向(铲斗端=DT小) -> 4个弧长锚点 ---
RATIOS=[0.0,0.35,0.70,1.0]; NAMES=["tip","bkt_joint","elbow","base"]
masks=[]; jpx=[[] for _ in RATIOS]; j3d=[[] for _ in RATIOS]; jH=[[] for _ in RATIOS]; ok=[]
for f in range(N):
    hh=hmap(f); m=(hh>0.02).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    lab,nl=label(m)
    if nl>1:
        sz=[(lab==i).sum() for i in range(1,nl+1)]; m=(lab==(np.argmax(sz)+1)).astype(np.uint8)
    masks.append(m)
    sk=skeletonize(m>0)
    path=longest_path(sk)
    if path is None or len(path)<8:
        ok.append(False)
        for k in range(len(RATIOS)): jpx[k].append((0,0)); j3d[k].append(np.full(3,np.nan)); jH[k].append(np.nan)
        continue
    # 定向: 铲斗端 DT 小. 比较两端各 K 点的 DT 均值
    dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
    K=max(3,len(path)//6)
    dt0=dt[path[:K,0],path[:K,1]].mean(); dt1=dt[path[-K:,0],path[-K:,1]].mean()
    if dt0>dt1: path=path[::-1]   # 让 path[0] = 铲斗端(DT小)
    L=len(path); ok.append(True)
    for k,r in enumerate(RATIOS):
        i=int(round(r*(L-1))); py,px=path[i]
        # 局部窗口取 3D 质心(更稳), 半径 6px
        y0,y1=max(0,py-6),min(H,py+7); x0,x1=max(0,px-6),min(W,px+7)
        win=m[y0:y1,x0:x1]>0
        ys,xs=np.where(win)
        ys=ys+y0; xs=xs+x0
        jpx[k].append((int(px),int(py)))
        j3d[k].append(wp[f,ys,xs].mean(0))
        jH[k].append(float(hh[ys,xs].mean()))

# numpy
jH=[np.array(a) for a in jH]; j3d=[np.array(a) for a in j3d]
print("valid frames:",sum(ok),"/",N)

# --- 速度/加速度/状态 (NaN 安全: 缺帧用线性插值补) ---
def fill_nan(a):
    a=a.copy(); idx=np.arange(len(a)); good=~np.isnan(a)
    if good.sum()<2: return a
    a[~good]=np.interp(idx[~good],idx[good],a[good]); return a
def smooth(a,k=3): return a.copy() if len(a)<k else np.convolve(a,np.ones(k)/k,mode='same')
def analyze(Y):
    Y=fill_nan(Y); Ys=smooth(Y,3); v=np.gradient(Ys)/DT; a=np.gradient(smooth(v,3))/DT
    eps=max(0.012,np.nanmedian(np.abs(np.diff(Y)))/DT*0.5)
    st=["up" if vi>eps else ("down" if vi<-eps else "static") for vi in v]
    return Ys,v,a,st,eps

J={}
for k,nm in enumerate(NAMES):
    Ys,v,a,st,eps=analyze(jH[k])
    J[nm]={"H":jH[k].tolist(),"vZ":v.tolist(),"aZ":a.tolist(),"state":st,"eps":eps}
    print(f"{nm:10s} eps={eps:.3f}  states={st}")

json.dump({"fps":FPS,"vertical_normal":ng.tolist(),"names":NAMES,"ratios":RATIOS,
           "valid":ok,"joints":J}, open(f"{OUT}/joints_motion.json","w"),indent=2)

# --- 曲线图: 4关节高度 + 速度 ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t=np.arange(N)*DT; CMAP=['#e74c3c','#f39c12','#27ae60','#2980b9']
def hex_bgr(h): h=h.lstrip('#'); return (int(h[4:6],16),int(h[2:4],16),int(h[0:2],16))
BGR=[hex_bgr(c) for c in CMAP]
fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
for k,nm in enumerate(NAMES):
    ax[0].plot(t,fill_nan(jH[k]),'o-',color=CMAP[k],label=nm,alpha=.85)
    ax[1].plot(t,J[nm]["vZ"],'o-',color=CMAP[k],label=nm,alpha=.85)
ax[0].set_ylabel("height above ground (m)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title("Multi-joint vertical motion (arc-length anchors on skeleton)")
ax[1].axhline(0,color='gray',lw=.6); ax[1].set_ylabel("vertical velocity (m/s)")
ax[1].set_xlabel("time (s)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/joints_curves.png",dpi=110); plt.close()

# --- 每帧叠加: 4关节点 + 各自速度小箭头 + 颜色=state ---
COL={"up":(0,200,0),"down":(0,0,255),"static":(170,170,170)}
for f in range(N):
    rgb=(imgs[f].transpose(1,2,0)*255).astype(np.uint8)[:,:,::-1].copy()
    rgb[masks[f]>0]=(0.75*rgb[masks[f]>0]+0.25*np.array([0,140,255])).astype(np.uint8)
    if ok[f]:
        for k,nm in enumerate(NAMES):
            px,py=jpx[k][f]; vz=J[nm]["vZ"][f]; st=J[nm]["state"][f]
            c=BGR[k]
            cv2.circle(rgb,(px,py),4,c,-1); cv2.circle(rgb,(px,py),5,(0,0,0),1)
            # 速度箭头(竖直, 屏幕向上=负y)
            tip=(px,int(py-np.sign(vz)*min(abs(vz)*120,28)))
            if abs(vz)>J[nm]["eps"]:
                cv2.arrowedLine(rgb,(px,py),tip,COL[st],2,cv2.LINE_AA,tipLength=0.4)
            cv2.putText(rgb,nm[:3],(px+5,py),cv2.FONT_HERSHEY_SIMPLEX,0.32,c,1,cv2.LINE_AA)
    cv2.rectangle(rgb,(0,0),(W,15),(0,0,0),-1)
    cv2.putText(rgb,f"f{f} t={f*DT:.2f}s"+("" if ok[f] else "  [skel-fail]"),(4,11),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,(230,230,230),1,cv2.LINE_AA)
    cv2.imwrite(f"{OUT}/frames/f{f:02d}.png",rgb)

# contact sheet
import glob
files=sorted(glob.glob(f"{OUT}/frames/f*.png")); tiles=[cv2.imread(x) for x in files]
h,w=tiles[0].shape[:2]; cols,pad=5,6; rows=(len(tiles)+cols-1)//cols
sheet=np.full((rows*(h+pad)+pad,cols*(w+pad)+pad,3),28,np.uint8)
for i,im in enumerate(tiles):
    rr,cc=divmod(i,cols); y=pad+rr*(h+pad); x=pad+cc*(w+pad); sheet[y:y+h,x:x+w]=im
cv2.imwrite(f"{OUT}/contact_sheet.png",sheet)
print("wrote",OUT)
