"""Failure-case figure for Seminar 3 (failure_slide.jpg).

For Aldabra, Pondoland and Protea: the test image where YOLOv11x (GD-tiny) predicts the
fewest boxes relative to the human labels (same choice as qual_failure.jpg).
Top row: human labels (blue). Bottom row: GD-trained predictions (red) and the human
boxes they missed (white, no prediction with IoU >= 0.5). Same colours as the rest of the slides.
"""
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

RUNS = "/mnt/LinuxBackup/jo/runs"
DATA = "/mnt/LinuxBackup/jo/datasets"
OUT = "/mnt/LinuxBackup/jo/results"
SUBS = ["aldabra", "pondoland", "protea"]
CONF, IOU = 0.25, 0.5
PW, PH, HEAD, GAP = 800, 600, 46, 14
HUM_BLUE, GD_RED, WHITE = (194, 120, 74), (40, 39, 214), (255, 255, 255)   # BGR: #4A78C2, #D62728

m_gd = YOLO(f"{RUNS}/v11x_gdtiny/weights/best.pt")


def human_boxes(p, w, h):
    lbl = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    boxes = []
    if os.path.exists(lbl):
        for line in open(lbl):
            q = line.split()
            if len(q) < 5:
                continue
            v = np.array(list(map(float, q[1:])))
            if len(q) == 5:
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


def match(pred, scores, gt):
    ious = iou_matrix(pred, gt)
    tp, hit = np.zeros(len(pred), bool), np.zeros(len(gt), bool)
    for i in np.argsort(-scores):
        if len(gt) == 0:
            break
        cand = np.where((~hit) & (ious[i] >= IOU))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            tp[i], hit[j] = True, True
    return tp, hit


def panel(img, items, text):
    s = min(PW / img.shape[1], PH / img.shape[0])
    im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    for boxes, col, t in items:
        for x1, y1, x2, y2 in (boxes * s).astype(int):
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 0), t + 2)
            cv2.rectangle(im, (x1, y1), (x2, y2), col, t)
    c = np.full((PH + HEAD, PW, 3), 255, np.uint8)
    y0, x0 = HEAD + (PH - im.shape[0]) // 2, (PW - im.shape[1]) // 2
    c[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    cv2.putText(c, text, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return c


top, bottom = [], []
for sub in SUBS:
    best_gap, best_p = -1e9, None
    for r in m_gd.predict(f"{DATA}/{sub}_hbb/test/images", conf=CONF, verbose=False, stream=True, batch=1):
        hh, ww = r.orig_shape
        g = len(human_boxes(r.path, ww, hh)) - len(r.boxes)
        if g > best_gap:
            best_gap, best_p = g, r.path
    img = cv2.imread(best_p)
    h, w = img.shape[:2]
    gt = human_boxes(best_p, w, h)
    det = m_gd.predict(best_p, conf=CONF, verbose=False)[0].boxes
    pred, sc = det.xyxy.cpu().numpy(), det.conf.cpu().numpy()
    tp, hit = match(pred, sc, gt)
    top.append(panel(img, [(gt, HUM_BLUE, 3)], f"{sub.capitalize()} - Human labels: {len(gt)} fish"))
    bottom.append(panel(img, [(gt[~hit], WHITE, 3), (pred, GD_RED, 3)],
                        f"{sub.capitalize()} - GD-trained: {len(pred)} boxes, {(~hit).sum()} missed"))
    print(f"{sub:10s} {os.path.basename(best_p)[:40]:40s} human {len(gt)}, TP {tp.sum()}, FP {(~tp).sum()}, FN {(~hit).sum()}")

gap_v = np.full((PH + HEAD, GAP, 3), 255, np.uint8)
row = lambda ps: np.hstack([ps[0], gap_v, ps[1], gap_v, ps[2]])
fig = np.vstack([row(top), np.full((GAP, 3 * PW + 2 * GAP, 3), 255, np.uint8), row(bottom)])
cv2.imwrite(f"{OUT}/failure_slide.jpg", fig, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved failure_slide.jpg")
