"""
多关节运动状态仪表盘 + MP4.
读 joints/joints_motion.json (run_joints_motion.py 产出), 单帧叠加:
  - 4 个关节锚点 (tip/bkt_joint/elbow/base, 各自固定色)
  - 每关节速度箭头(实线, 竖直=屏幕方向, 色=升绿/降红/静灰), 加速度箭头(虚线橙)
  - 右下文字面板: 每关节 H / vZ / state
  - 底部 4 条高度时间轴 + 当前帧游标
输出 dashboard/fXX.png + dashboard_sheet.png + joints_dashboard.mp4
环境: vggt (CPU)。
"""
import os, json, glob
import numpy as np, cv2
from numpy.random import default_rng
from scipy.ndimage import label
from skimage.morphology import skeletonize
from collections import deque

ROOT="/home/maomaoyu/WS/vggt_yoloe"
SESS=f"{ROOT}/workspaces/session_20260629_173356_784627"
OUT =f"{ROOT}/experiments/arm_motion_state/joints"
os.makedirs(f"{OUT}/dashboard",exist_ok=True)

d=np.load(f"{SESS}/predictions.npz")
wp=d["world_points_from_depth"]; imgs=d["images"]
N,H,W=wp.shape[:3]; FPS=30.0/6.0; DT=1.0/FPS

J=json.load(open(f"{OUT}/joints_motion.json"))
NAMES=J["names"]; RATIOS=J["ratios"]; ok=J["valid"]; ng=np.array(J["vertical_normal"])
off=-np.median(wp[5].reshape(-1,3).dot(ng))
h5=wp[5].reshape(-1,3).dot(ng)+off
if abs(np.percentile(h5,1))>abs(np.percentile(h5,99)): off=-off  # ng 已定向, 仅校 off
def hmap(f): return (wp[f].reshape(-1,3).dot(ng)+off).reshape(H,W)

# --- 重建每帧 mask + 锚点像素 (与 run_joints 同逻辑) ---
def longest_path(sk):
    pts=np.argwhere(sk>0)
    if len(pts)<2: return None
    idx={(y,x):i for i,(y,x) in enumerate(pts)}
    nb=[[] for _ in pts]
    for i,(y,x) in enumerate(pts):
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                if dy==0 and dx==0: continue
                j=idx.get((y+dy,x+dx));
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
    path=[b]; cur=b
    while cur!=a: cur=min(nb[cur],key=lambda v:dist[v]); path.append(cur)
    return pts[path[::-1]]

masks=[]; jpx=[[] for _ in RATIOS]
for f in range(N):
    hh=hmap(f); m=(hh>0.02).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    lab,nl=label(m)
    if nl>1:
        sz=[(lab==i).sum() for i in range(1,nl+1)]; m=(lab==(np.argmax(sz)+1)).astype(np.uint8)
    masks.append(m); sk=skeletonize(m>0); path=longest_path(sk)
    if path is None or len(path)<8 or not ok[f]:
        for k in range(len(RATIOS)): jpx[k].append((0,0)); continue
    dt=cv2.distanceTransform(m,cv2.DIST_L2,5); K=max(3,len(path)//6)
    if dt[path[:K,0],path[:K,1]].mean()>dt[path[-K:,0],path[-K:,1]].mean(): path=path[::-1]
    L=len(path)
    for k,r in enumerate(RATIOS):
        i=int(round(r*(L-1))); py,px=path[i]; jpx[k].append((int(px),int(py)))

CMAP=['#e74c3c','#f39c12','#27ae60','#2980b9']
def hex_bgr(h): h=h.lstrip('#'); return (int(h[4:6],16),int(h[2:4],16),int(h[0:2],16))
BGR=[hex_bgr(c) for c in CMAP]
COL={"up":(0,200,0),"down":(0,0,255),"static":(170,170,170)}

def draw_arrow(img,p,dy,dx,color,scale,dashed=False):
    p=np.array(p,float); vec=np.array([dx,dy])*scale; tip=p+vec
    if np.linalg.norm(vec)<2: return
    p2=tuple(p.astype(int)); t2=tuple(tip.astype(int))
    if dashed:
        for i in range(7):
            a=p+(tip-p)*i/7; b=p+(tip-p)*(i+0.5)/7
            cv2.line(img,tuple(a.astype(int)),tuple(b.astype(int)),color,1,cv2.LINE_AA)
        cv2.circle(img,t2,2,color,-1)
    else:
        cv2.arrowedLine(img,p2,t2,color,2,cv2.LINE_AA,tipLength=0.35)

Hall=np.array([J["joints"][nm]["H"] for nm in NAMES])
Hmin,Hmax=np.nanmin(Hall)-0.01,np.nanmax(Hall)+0.01

def render(f):
    rgb=(imgs[f].transpose(1,2,0)*255).astype(np.uint8)[:,:,::-1].copy()
    rgb[masks[f]>0]=(0.78*rgb[masks[f]>0]+0.22*np.array([0,140,255])).astype(np.uint8)
    canvas=np.full((H+150,W,3),28,np.uint8); canvas[:H]=rgb
    if ok[f]:
        for k,nm in enumerate(NAMES):
            px,py=jpx[k][f]; vz=J["joints"][nm]["vZ"][f]; az=J["joints"][nm]["aZ"][f]; st=J["joints"][nm]["state"][f]
            draw_arrow(canvas,(px,py),-vz,0,COL[st],scale=120.0)        # 速度(竖直)
            draw_arrow(canvas,(px,py),-az,0,(255,180,0),scale=60.0,dashed=True)  # 加速度
            cv2.circle(canvas,(px,py),4,BGR[k],-1); cv2.circle(canvas,(px,py),5,(0,0,0),1)
    # 文字面板(右下)
    panel=[f"frame {f}  t={f*DT:.2f}s"]
    for nm in NAMES:
        jj=J["joints"][nm]
        panel.append(f"{nm[:9]:9s} {jj['state'][f][:4]:4s} H{jj['H'][f]:+.2f} v{jj['vZ'][f]:+.2f}")
    pw,ph=232,len(panel)*15+6; px0,py0=W-pw-2,H-ph-2
    cv2.rectangle(canvas,(px0,py0),(px0+pw,py0+ph),(0,0,0),-1)
    for i,s in enumerate(panel):
        c=(235,235,235) if i==0 else BGR[i-1]
        cv2.putText(canvas,s,(px0+5,py0+13+i*15),cv2.FONT_HERSHEY_SIMPLEX,0.36,c,1,cv2.LINE_AA)
    # 图例(左上)
    for k,nm in enumerate(NAMES):
        cv2.putText(canvas,f"{nm}",(8,14+k*15),cv2.FONT_HERSHEY_SIMPLEX,0.4,BGR[k],1,cv2.LINE_AA)
    # 底部 4 条高度时间轴
    y0=H+22; gh=110; pad=44
    cv2.putText(canvas,"joint height timelines",(pad,y0-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1,cv2.LINE_AA)
    def XY(k,i):
        Hser=np.array(J["joints"][NAMES[k]]["H"])
        x=int(pad+(W-2*pad)*i/(N-1)); yv=int(y0+gh-(gh-6)*(Hser[i]-Hmin)/(Hmax-Hmin)); return x,yv
    for k in range(len(NAMES)):
        for i in range(N-1):
            cv2.line(canvas,XY(k,i),XY(k,i+1),BGR[k],1,cv2.LINE_AA)
    cx=int(pad+(W-2*pad)*f/(N-1)); cv2.line(canvas,(cx,y0),(cx,y0+gh),(0,255,255),1)
    for k in range(len(NAMES)):
        cv2.circle(canvas,XY(k,f),3,BGR[k],-1)
    return canvas

for f in range(N):
    cv2.imwrite(f"{OUT}/dashboard/f{f:02d}.png",render(f))

# contact sheet
files=sorted(glob.glob(f"{OUT}/dashboard/f*.png")); tiles=[cv2.imread(x) for x in files]
h,w=tiles[0].shape[:2]; cols,pad=5,6; rows=(len(tiles)+cols-1)//cols
sheet=np.full((rows*(h+pad)+pad,cols*(w+pad)+pad,3),28,np.uint8)
for i,im in enumerate(tiles):
    rr,cc=divmod(i,cols); y=pad+rr*(h+pad); x=pad+cc*(w+pad); sheet[y:y+h,x:x+w]=im
cv2.imwrite(f"{OUT}/dashboard_sheet.png",sheet)

# MP4: 每帧 hold 4 张 @12fps
vw=cv2.VideoWriter(f"{OUT}/joints_dashboard.mp4",cv2.VideoWriter_fourcc(*'mp4v'),12.0,(w,h))
for im in tiles:
    for _ in range(4): vw.write(im)
for _ in range(12): vw.write(tiles[-1])
vw.release()
print("wrote dashboard/, dashboard_sheet.png, joints_dashboard.mp4")
