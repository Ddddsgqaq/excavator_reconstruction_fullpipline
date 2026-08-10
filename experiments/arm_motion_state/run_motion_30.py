"""
挖掘机手臂运动状态分析 v2 (粗粒度 up/down/static).
修复 v1 两个 bug:
  1) 垂直方向: 改用 *地面平面法向* (RANSAC) 而非相机轨迹重力对齐
     —— 该 session 相机轨迹不可靠, 轨迹法把垂直轴估歪了。
  2) 代表点: 改用 *铲斗区域平均高度* (臂 mask 中靠铲斗端那批高点),
     不再用帧间乱跳的"最远点/最高点"。
信号: 沿地面法向的离地高度, 相机运动天然消除(world_points 在世界系)。
输出: motion_curves.png + frames/ + motion_state.json
环境: vggt (CPU)。
"""
import os, sys, glob, json
import numpy as np
import cv2
from numpy.random import default_rng
from scipy.ndimage import label

ROOT = "/home/maomaoyu/WS/vggt_yoloe"
SESS = f"{ROOT}/workspaces/session_20260629_173356_784627"
OUT  = f"{ROOT}/experiments/arm_motion_state/run30"
os.makedirs(f"{OUT}/frames", exist_ok=True)

d = np.load(f"{SESS}/predictions.npz")
wp   = d["world_points_from_depth"]      # (N,H,W,3) world frame
imgs = d["images"]
N,H,W = wp.shape[:3]
FPS = 30.0/6.0                            # 0-6s, 14帧 (粗略量纲)
DT = 1.0/FPS

# --- 1. 地面平面法向 (RANSAC on f5), 作垂直方向 ---
def ground_normal(P, iters=600, thr=0.01):
    rng=default_rng(0); best=(0,None,None)
    s=P[rng.choice(P.shape[0],20000,replace=False)]
    for _ in range(iters):
        i=rng.choice(s.shape[0],3,replace=False); p0,p1,p2=s[i]
        n=np.cross(p1-p0,p2-p0); ln=np.linalg.norm(n)
        if ln<1e-9: continue
        n/=ln; o=-n.dot(p0); inl=(np.abs(s.dot(n)+o)<thr).sum()
        if inl>best[0]: best=(inl,n,o)
    return best[1], best[2]
ng,_ = ground_normal(wp[5].reshape(-1,3))
off = -np.median(wp[5].reshape(-1,3).dot(ng))
# 定向: 臂是离地面最远的一端。比较两侧极端高度的绝对值, 让臂(尖端伸出那侧)为正。
h5 = wp[5].reshape(-1,3).dot(ng) + off
if abs(np.percentile(h5,1)) > abs(np.percentile(h5,99)):
    # 极端值在负侧 → 臂在负侧, 翻号
    ng=-ng; off=-off
print(f"ground normal (vertical) = {ng.round(3)}")

def height_map(f):
    h = wp[f].reshape(-1,3).dot(ng) + off
    return h.reshape(H,W)

# --- 2. 每帧: 整机mask + 铲斗端高度 + 臂质心高度 ---
bucketH=[]; centH=[]; bpx=[]; cpx=[]; masks=[]
for f in range(N):
    hh = height_map(f)
    mask = (hh>0.02).astype(np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    lab,nl=label(mask)
    if nl>1:
        sizes=[(lab==i).sum() for i in range(1,nl+1)]
        mask=(lab==(np.argmax(sizes)+1)).astype(np.uint8)
    masks.append(mask)
    ys,xs=np.where(mask>0)
    # 臂质心高度
    centH.append(float(hh[ys,xs].mean()))
    cpx.append((int(xs.mean()),int(ys.mean())))
    # 铲斗端: 图像左侧 12% 列里的高点 (铲斗在该 session 偏左) → 取其平均离地高度
    thr_x = np.percentile(xs,12)
    sel = xs < thr_x
    bx,by = xs[sel],ys[sel]
    bucketH.append(float(hh[by,bx].mean()))
    bpx.append((int(bx.mean()),int(by.mean())))

bucketH=np.array(bucketH); centH=np.array(centH)

# --- 3. 平滑 + 速度/加速度 + 状态机 ---
def smooth(a,k=3):
    if len(a)<k: return a.copy()
    return np.convolve(a,np.ones(k)/k,mode='same')
def analyze(Y):
    Ys=smooth(Y,3)
    v=np.gradient(Ys)/DT
    a=np.gradient(smooth(v,3))/DT
    eps=max(0.012, np.median(np.abs(np.diff(Y)))/DT*0.5)
    st=["up" if vi>eps else ("down" if vi<-eps else "static") for vi in v]
    return Ys,v,a,st,eps
bYs,bV,bA,bSt,bEps = analyze(bucketH)
cYs,cV,cA,cSt,cEps = analyze(centH)
print("bucketH:",bucketH.round(3).tolist())
print("eps bucket/cent:",round(bEps,3),round(cEps,3))
print("bucket states:",bSt)
print("cent   states:",cSt)

# --- 4. 曲线图 ---
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
t=np.arange(N)*DT
fig,ax=plt.subplots(3,1,figsize=(9,8),sharex=True)
ax[0].plot(t,bucketH,'o-',alpha=.4,label='bucket region H (raw)')
ax[0].plot(t,bYs,'-',lw=2,label='bucket H (smooth)')
ax[0].plot(t,centH,'s-',alpha=.4,label='arm centroid H (raw)')
ax[0].plot(t,cYs,'-',lw=2,label='centroid H (smooth)')
ax[0].set_ylabel('height above ground (m)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title('Arm/bucket vertical motion (ground-normal frame)')
ax[1].plot(t,bV,'o-',label='bucket vZ'); ax[1].plot(t,cV,'s-',label='centroid vZ')
ax[1].axhline(bEps,color='gray',ls='--',lw=.7); ax[1].axhline(-bEps,color='gray',ls='--',lw=.7)
ax[1].set_ylabel('vertical velocity (m/s)'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[2].plot(t,bA,'o-',label='bucket aZ'); ax[2].plot(t,cA,'s-',label='centroid aZ')
ax[2].set_ylabel('vertical accel (m/s^2)'); ax[2].set_xlabel('time (s)'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/motion_curves.png",dpi=110); plt.close()

# --- 5. 每帧叠加 ---
COL={"up":(0,200,0),"down":(0,0,255),"static":(160,160,160)}
for f in range(N):
    rgb=(imgs[f].transpose(1,2,0)*255).astype(np.uint8)[:,:,::-1].copy()
    rgb[masks[f]>0]=(0.65*rgb[masks[f]>0]+0.35*np.array([0,140,255])).astype(np.uint8)
    cv2.circle(rgb,bpx[f],4,(255,255,0),-1)   # 铲斗端 黄
    cv2.circle(rgb,cpx[f],4,(255,0,255),-1)   # 质心 品红
    cv2.rectangle(rgb,(0,0),(W,16),(0,0,0),-1)
    cv2.putText(rgb,f"f{f} bucket:{bSt[f]} (H={bucketH[f]:.2f})",(4,12),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,COL[bSt[f]],1,cv2.LINE_AA)
    cv2.imwrite(f"{OUT}/frames/f{f:02d}.png",rgb)

# contact sheet
files=sorted(glob.glob(f"{OUT}/frames/f*.png"))
tiles=[cv2.imread(f) for f in files]; h,w=tiles[0].shape[:2]
cols,pad=4,6; rows=(len(tiles)+cols-1)//cols
sheet=np.full((rows*(h+pad)+pad,cols*(w+pad)+pad,3),40,np.uint8)
for i,im in enumerate(tiles):
    rr,cc=divmod(i,cols); y=pad+rr*(h+pad); x=pad+cc*(w+pad)
    sheet[y:y+h,x:x+w]=im
cv2.imwrite(f"{OUT}/contact_sheet.png",sheet)

json.dump({"fps":FPS,"vertical_normal":ng.tolist(),
 "bucket_H":bucketH.tolist(),"centroid_H":centH.tolist(),
 "bucket_state":bSt,"centroid_state":cSt,
 "bucket_vZ":bV.tolist()}, open(f"{OUT}/motion_state.json","w"),indent=2)
print(f"wrote {OUT}/motion_curves.png , contact_sheet.png , motion_state.json")
