"""Grounding DINO (tiny) threshold sweep against human labels.

Runs GD-tiny once per image at the lowest threshold, then filters the boxes at each
threshold and matches them to the human boxes at IoU 0.5 (greedy, highest score first).

  python gd_threshold_sweep.py --split test    # check: 0.30 row should give 10,483 boxes, P .852, R .643
  python gd_threshold_sweep.py --split valid   # the sweep to report
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DATA = "/mnt/LinuxBackup/jo/datasets/combined_hbb"
OUT = "/mnt/LinuxBackup/jo/results"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "fish."
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
IOU = 0.5

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="valid")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(dev).eval()


def human_boxes(lbl, w, h):
    boxes = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) < 5:
                continue
            v = np.array(list(map(float, p[1:])))
            if len(p) == 5:
                cx, cy, bw, bh = v
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            else:
                xy = v.reshape(-1, 2)
                boxes.append([xy[:, 0].min() * w, xy[:, 1].min() * h, xy[:, 0].max() * w, xy[:, 1].max() * h])
    return np.array(boxes).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def count_tp(pred, scores, gt):
    order = np.argsort(-scores)
    ious = iou_matrix(pred[order], gt)
    used, tp = set(), 0
    for i in range(len(order)):
        if ious.shape[1] == 0:
            break
        cand = [(ious[i, j], j) for j in range(ious.shape[1]) if j not in used and ious[i, j] >= IOU]
        if cand:
            used.add(max(cand)[1]); tp += 1
    return tp


imgs = sorted(f for f in glob.glob(f"{DATA}/{args.split}/images/*") if f.lower().endswith((".jpg", ".jpeg", ".png")))
print(f"{len(imgs)} images in {args.split}")
stats = {t: [0, 0, 0] for t in THRESHOLDS}   # tp, n_pred, n_gt
for k, p in enumerate(imgs):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    inputs = proc(images=im, text=PROMPT, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    r = proc.post_process_grounded_object_detection(out, input_ids=inputs.input_ids, threshold=min(THRESHOLDS),
                                                    target_sizes=[(h, w)])[0]
    boxes, scores = r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()
    gt = human_boxes(p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt", w, h)
    for t in THRESHOLDS:
        keep = scores > t
        stats[t][0] += count_tp(boxes[keep], scores[keep], gt)
        stats[t][1] += int(keep.sum())
        stats[t][2] += len(gt)
    if (k + 1) % 250 == 0:
        print(f"  {k + 1}/{len(imgs)}")

rows = []
print(f"\nhuman boxes: {stats[THRESHOLDS[0]][2]}")
print(f"{'thr':>5} {'boxes':>7} {'box/img':>8} {'P':>6} {'R':>6} {'F1':>6}")
for t in THRESHOLDS:
    tp, n, g = stats[t]
    P, R = tp / max(n, 1), tp / max(g, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    rows.append([t, n, round(n / len(imgs), 2), round(P, 3), round(R, 3), round(F1, 3)])
    print(f"{t:5.2f} {n:7d} {n / len(imgs):8.2f} {P:6.3f} {R:6.3f} {F1:6.3f}")

path = f"{OUT}/threshold_sweep_{args.split}.csv"
with open(path, "w", newline="") as f:
    csv.writer(f).writerows([["threshold", "boxes", "boxes_per_img", "precision", "recall", "f1"]] + rows)
print(f"saved {path}")
