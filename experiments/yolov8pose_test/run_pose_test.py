"""
YOLOv8-Pose sanity test on excavator images.
- 用官方 COCO 人体 17-kpt 预训练权重，直接在挖掘机图上推理。
- 目的: 实证「预训练人体姿态模型对挖掘机无效」，作为自训练的佐证图。
- 不污染任何环境: 在 yoloe 环境跑现有 ultralytics; 权重下到本工程 weights/ 下。
"""
import os, glob, sys
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = "/home/maomaoyu/WS/vggt_yoloe"
SESS = f"{ROOT}/workspaces/session_20260629_100116_092814"
OUT  = f"{ROOT}/experiments/yolov8pose_test"
WEIGHTS_DIR = f"{OUT}/weights"
os.makedirs(f"{OUT}/pred", exist_ok=True)

# 权重放到工程目录，避免下到 home / cwd 乱放
os.environ.setdefault("YOLO_CONFIG_DIR", f"{OUT}/.ultralytics")
weight_path = f"{WEIGHTS_DIR}/yolov8n-pose.pt"
model = YOLO(weight_path if os.path.exists(weight_path) else "yolov8n-pose.pt")
# 若是首次从名字加载，ultralytics 会下到 cwd；移动到 weights/
if not os.path.exists(weight_path) and os.path.exists("yolov8n-pose.pt"):
    os.replace("yolov8n-pose.pt", weight_path)

imgs = sorted(glob.glob(f"{SESS}/images/*.png"))
print(f"images: {len(imgs)}")

det_count = []
for p in imgs:
    r = model(p, verbose=False, conf=0.25)[0]
    n = 0 if r.keypoints is None else len(r.keypoints)
    det_count.append(n)
    # 保存可视化（即使 0 检出也存原图，便于对比）
    vis = r.plot()
    name = os.path.basename(p).replace(".png", "_pose.png")
    cv2.imwrite(f"{OUT}/pred/{name}", vis)

print("per-frame person detections:", det_count)
print(f"total detections across {len(imgs)} frames: {sum(det_count)}")
print(f"frames with >=1 detection: {sum(1 for c in det_count if c>0)}/{len(imgs)}")

# contact sheet
files = sorted(glob.glob(f"{OUT}/pred/*_pose.png"),
               key=lambda x: int(os.path.basename(x)[:6]))
tiles = [cv2.imread(f) for f in files]
if tiles:
    h, w = tiles[0].shape[:2]
    cols, pad, lab = 4, 6, 18
    rows = (len(tiles)+cols-1)//cols
    sheet = np.full((rows*(h+lab+pad)+pad, cols*(w+pad)+pad, 3), 40, np.uint8)
    for i, im in enumerate(tiles):
        rr, cc = divmod(i, cols)
        y = pad+rr*(h+lab+pad); x = pad+cc*(w+pad)
        cv2.putText(sheet, f"frame {i}", (x+4, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        sheet[y+lab:y+lab+h, x:x+w] = im
    cv2.imwrite(f"{OUT}/contact_sheet.png", sheet)
    print(f"wrote {OUT}/contact_sheet.png")
