"""Walkthrough figures for Seminar 3, on the Protea test image used before (seed 0).
Grounding DINO only: no human labels are drawn. Every image is the same size (W wide).

walk_input.jpg          1. raw image
walk_gd_all.jpg         2. every Grounding DINO box (score >= 0.15 shown), all red
walk_gd_thr.jpg         3. same boxes with scores: red = 0.3 and above (kept), grey = below (dropped)
walk_train_labels.jpg   4. the kept boxes only, as they go into training (red)
walk_detector.jpg       5. YOLOv11x trained on GD-tiny labels, its boxes on the same image (red)
"""
import glob
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLO

RUNS = "/mnt/LinuxBackup/jo/runs"
DATA = "/mnt/LinuxBackup/jo/datasets"
OUT = "/mnt/LinuxBackup/jo/results"
SUBS = ["aldabra", "tanganyika", "pondoland", "protea"]
GD_THR, CAND_THR, DET_CONF = 0.3, 0.15, 0.25
W = 1280
GD_RED, GREY = (40, 39, 214), (170, 170, 170)      # BGR; GD_RED = #D62728, same red as the slides
Q = [cv2.IMWRITE_JPEG_QUALITY, 92]


def images(d):
    return sorted(f for f in glob.glob(f"{d}/*") if f.lower().endswith((".jpg", ".jpeg", ".png")))


random.seed(0)
picks = {s: random.choice(images(f"{DATA}/{s}_hbb/test/images")) for s in SUBS}
path = picks["protea"]
img = cv2.imread(path)
h, w = img.shape[:2]
s = W / w
print("image:", os.path.basename(path))


def canvas():
    return cv2.resize(img, (W, int(h * s)))


def box(c, b, col, t=3):
    x1, y1, x2, y2 = (np.array(b) * s).astype(int)
    cv2.rectangle(c, (x1, y1), (x2, y2), (0, 0, 0), t + 2)   # dark outline for contrast
    cv2.rectangle(c, (x1, y1), (x2, y2), col, t)
    return x1, y1


def score(c, x1, y1, sc, col):
    p = (x1 + 3, max(y1 - 5, 14))
    cv2.putText(c, f"{sc:.2f}", p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(c, f"{sc:.2f}", p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)


# 1. input
cv2.imwrite(f"{OUT}/walk_input.jpg", canvas(), Q)

# Grounding DINO, once
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(dev).eval()
pil = Image.open(path).convert("RGB")
inputs = proc(images=pil, text="fish.", return_tensors="pt").to(dev)
with torch.no_grad():
    out = gd(**inputs)
r = proc.post_process_grounded_object_detection(out, input_ids=inputs.input_ids, threshold=CAND_THR,
                                                target_sizes=[(h, w)])[0]
cb, cs = r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()
keep = cs > GD_THR

# 2. all boxes, one colour
c = canvas()
for b in cb:
    box(c, b, GD_RED)
cv2.imwrite(f"{OUT}/walk_gd_all.jpg", c, Q)

# 3. same boxes split at 0.3, with scores (dropped drawn first, kept on top)
c = canvas()
for b, sc in zip(cb[~keep], cs[~keep]):
    score(c, *box(c, b, GREY, 2), sc, GREY)
for b, sc in zip(cb[keep], cs[keep]):
    score(c, *box(c, b, GD_RED), sc, GD_RED)
cv2.imwrite(f"{OUT}/walk_gd_thr.jpg", c, Q)

# 4. kept boxes only = this image's training labels
c = canvas()
for b in cb[keep]:
    box(c, b, GD_RED)
cv2.imwrite(f"{OUT}/walk_train_labels.jpg", c, Q)

stem = os.path.basename(path).rsplit(".", 1)[0]
file_lbl = glob.glob(f"{DATA}/combined_gd_tiny/*/labels/{stem}.txt")
n_file = sum(1 for l in open(file_lbl[0]) if l.strip()) if file_lbl else -1
print(f"GD boxes {len(cb)}, kept {keep.sum()}, dropped {(~keep).sum()} "
      f"(pseudo-label file has {n_file} boxes; kept should match)")

# 5. the detector trained on GD-tiny labels, on the same image
det = YOLO(f"{RUNS}/v11x_gdtiny/weights/best.pt").predict(path, conf=DET_CONF, verbose=False)[0].boxes
c = canvas()
for b in det.xyxy.cpu().numpy():
    box(c, b, GD_RED)
cv2.imwrite(f"{OUT}/walk_detector.jpg", c, Q)
print(f"trained detector: {len(det)} boxes")
