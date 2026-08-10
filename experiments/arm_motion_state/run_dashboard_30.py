"""
运动状态仪表盘可视化.
单张图(每帧)叠加:
  - 整机分割 + 铲斗代表点
  - 速度矢量箭头(图像平面真实方向, 由相邻帧像素位移得到; 颜色=升/降)
  - 加速度矢量箭头(虚线)
  - 文字面板: 离地高度H, 竖直速度vZ, 竖直加速度aZ, 运动状态, 3D速度大小
  - 底部高度时间轴 + 当前帧游标
依赖 v2 的分割/高度逻辑(地面法向当垂直)。环境: vggt (CPU)。
"""
import os, sys, json, glob
import numpy as np, cv2
from numpy.random import default_rng
from scipy.ndimage import label

ROOT="/home/maomaoyu/WS/vggt_yoloe"
SESS=f"{ROOT}/workspaces/session_20260629_173356_784627"
OUT =f"{ROOT}/experiments/arm_motion_state/run30"
os.makedirs(f"{OUT}/dashboard",exist_ok=True)

d=np.load(f"{SESS}/predictions.npz")
wp=d["world_points_from_depth"]; imgs=d["images"]
N,H,W=wp.shape[:3]; FPS=30.0/6.0; DT=1.0/FPS

# 地面法向(垂直)
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

def hmap(f): return (wp[f].reshape(-1,3).dot(ng)+off).reshape(H,W)

# 每帧: mask, 铲斗代表点(2D像素 + 3D世界), 离地高度
masks=[]; bpx=[]; b3d=[]; bH=[]
for f in range(N):
    hh=hmap(f); m=(hh>0.02).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    lab,nl=label(m)
    if nl>1:
        sz=[(lab==i).sum() for i in range(1,nl+1)]; m=(lab==(np.argmax(sz)+1)).astype(np.uint8)
    masks.append(m); ys,xs=np.where(m>0)
    sel=xs<np.percentile(xs,12); bx,by=xs[sel],ys[sel]
    px=(int(bx.mean()),int(by.mean())); bpx.append(px)
    bH.append(float(hh[by,bx].mean()))
    b3d.append(wp[f,by,bx].mean(0))            # 该区域 3D 质心(世界系)
bH=np.array(bH); b3d=np.array(b3d); bpx=np.array(bpx)

def smooth(a,k=3): return a.copy() if len(a)<k else np.convolve(a,np.ones(k)/k,mode='same')
Hs=smooth(bH,3); vZ=np.gradient(Hs)/DT; aZ=np.gradient(smooth(vZ,3))/DT
eps=max(0.012,np.median(np.abs(np.diff(bH)))/DT*0.5)
state=["up" if v>eps else ("down" if v<-eps else "static") for v in vZ]

# 图像平面 2D 速度/加速度矢量——用真实物理量驱动方向, 不用乱跳的像素位移:
#   竖直分量: 直接由 vZ/aZ 符号决定 (屏幕向上 = 负y), 保证与 state 一致
#   水平分量: 3D 速度沿"水平面内"的位移投到图像 x (小幅, 体现摆臂)
vel3d=np.gradient(b3d,axis=0)/DT                 # 世界系 3D 速度 (m/s)
spd3d=np.linalg.norm(vel3d,axis=1)
# 水平方向 = 去掉沿 ng 的分量
ngu=ng/np.linalg.norm(ng)
vel_horiz=vel3d-(vel3d@ngu)[:,None]*ngu[None,:]
# 把水平速度投到图像 x: 用 b3d 的世界 X 变化近似 (符号即可, 量级统一缩放)
horiz_x=np.gradient(b3d[:,0])                     # 世界X逐帧变化
def motion_vec(arr_v, hx):
    # 返回图像平面矢量 (dx, dy): dy 向上为负
    return np.stack([np.sign(hx)*np.minimum(np.abs(hx)*1.0,1.0), -arr_v], 1)
vel2d_disp=motion_vec(vZ, horiz_x)
acc2d_disp=motion_vec(aZ, np.gradient(horiz_x))

COL={"up":(0,200,0),"down":(0,0,255),"static":(170,170,170)}
def draw_arrow(img,p,vec,color,scale,thick=2,dashed=False,label=""):
    p=np.array(p,float); tip=p+vec*scale
    if np.linalg.norm(vec*scale)<2: return
    p2=tuple(p.astype(int)); t2=tuple(tip.astype(int))
    if dashed:
        n=8;
        for i in range(n):
            a=p+(tip-p)*i/n; b=p+(tip-p)*(i+0.5)/n
            cv2.line(img,tuple(a.astype(int)),tuple(b.astype(int)),color,thick,cv2.LINE_AA)
        cv2.circle(img,t2,3,color,-1)
    else:
        cv2.arrowedLine(img,p2,t2,color,thick,cv2.LINE_AA,tipLength=0.3)
    if label: cv2.putText(img,label,(t2[0]+3,t2[1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)

# 高度时间轴小图参数
def render(f):
    rgb=(imgs[f].transpose(1,2,0)*255).astype(np.uint8)[:,:,::-1].copy()
    rgb[masks[f]>0]=(0.7*rgb[masks[f]>0]+0.3*np.array([0,140,255])).astype(np.uint8)
    canvas=np.full((H+120,W,3),28,np.uint8); canvas[:H]=rgb
    p=bpx[f]
    cv2.circle(canvas,tuple(p),5,(0,255,255),-1); cv2.circle(canvas,tuple(p),6,(0,0,0),1)
    # 速度箭头(实线), 加速度箭头(虚线). 方向由真实物理量驱动(竖直符合 vZ)
    draw_arrow(canvas,p,vel2d_disp[f],COL[state[f]],scale=110.0,thick=3,label="v")
    draw_arrow(canvas,p,acc2d_disp[f],(255,180,0),scale=200.0,thick=2,dashed=True,label="a")
    # 文字面板(右下角, 避免挡住挖掘机)
    panel=[f"frame {f}   t={f*DT:.2f}s",
           f"state: {state[f].upper()}",
           f"height H = {bH[f]:+.3f} m",
           f"v_vert  = {vZ[f]:+.3f} m/s",
           f"a_vert  = {aZ[f]:+.3f} m/s2",
           f"|v_3D|  = {spd3d[f]:.3f} m/s"]
    pw,ph=192,len(panel)*16+8
    px0,py0=W-pw-2, H-ph-2
    cv2.rectangle(canvas,(px0,py0),(px0+pw,py0+ph),(0,0,0),-1)
    for i,s in enumerate(panel):
        c=COL[state[f]] if i==1 else (235,235,235)
        cv2.putText(canvas,s,(px0+5,py0+14+i*16),cv2.FONT_HERSHEY_SIMPLEX,0.42,c,1,cv2.LINE_AA)
    # 图例(左上)
    cv2.putText(canvas,"v: velocity",(8,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,COL[state[f]],1,cv2.LINE_AA)
    cv2.putText(canvas,"a: accel(dash)",(8,30),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,180,0),1,cv2.LINE_AA)
    # 底部高度时间轴
    y0=H+18; gh=88; pad=40
    cv2.putText(canvas,"bucket height timeline",(pad,y0-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1,cv2.LINE_AA)
    Hmin,Hmax=bH.min()-0.01,bH.max()+0.01
    def XY(i):
        x=int(pad+(W-2*pad)*i/(N-1)); yv=int(y0+gh-(gh-6)*(bH[i]-Hmin)/(Hmax-Hmin)); return x,yv
    for i in range(N-1):
        cv2.line(canvas,XY(i),XY(i+1),(120,160,255),2,cv2.LINE_AA)
    for i in range(N):
        c=COL[state[i]]; cv2.circle(canvas,XY(i),3,c,-1)
    cx,cyv=XY(f); cv2.line(canvas,(cx,y0),(cx,y0+gh),(0,255,255),1)
    cv2.circle(canvas,XY(f),5,(0,255,255),-1); cv2.circle(canvas,XY(f),6,(0,0,0),1)
    return canvas

for f in range(N):
    cv2.imwrite(f"{OUT}/dashboard/f{f:02d}.png",render(f))

# contact sheet
files=sorted(glob.glob(f"{OUT}/dashboard/f*.png")); tiles=[cv2.imread(x) for x in files]
h,w=tiles[0].shape[:2]; cols,pad=3,6; rows=(len(tiles)+cols-1)//cols
sheet=np.full((rows*(h+pad)+pad,cols*(w+pad)+pad,3),28,np.uint8)
for i,im in enumerate(tiles):
    rr,cc=divmod(i,cols); y=pad+rr*(h+pad); x=pad+cc*(w+pad); sheet[y:y+h,x:x+w]=im
cv2.imwrite(f"{OUT}/dashboard_sheet.png",sheet)
print("wrote dashboard/ and dashboard_sheet.png")
print("states:",state)
