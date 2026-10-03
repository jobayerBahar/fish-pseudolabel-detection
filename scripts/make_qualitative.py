"""Qualitative figures for Chapter 4.

qual_random.jpg  : one random test image per sub-collection (seed 0)
qual_failure.jpg : per sub-collection, the test image where YOLOv11x (GD-tiny)
                   predicts the fewest boxes relative to the human labels
Columns: human labels | YOLOv11x (Human) | YOLOv11x (GD-tiny), conf 0.25.

Edit the paths below if the script cannot find something; it stops with a clear message.
Send back both images and the printed table.
"""
import glob
import os
import random
import sys

import cv2
import numpy as np
from ultralytics import YOLO

RUNS = "/mnt/LinuxBackup/jo/runs"
DATA = "/mnt/LinuxBackup/jo/datasets"
OUT = "/mnt/LinuxBackup/jo/results"
SUBS = ["aldabra", "tanganyika", "pondoland", "protea"]
CONF = 0.25
PANEL_W, PANEL_H = 640, 480
HEADER = 40


def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    hits = [h for h in hits if "deepfish" not in h.lower() and "_ft" not in h.lower()]
    if not hits:
        sys.exit(f"Could not find {what} with pattern {pattern}")
    if len(hits) > 1:
        print(f"Several matches for {what}, using the first:\n  " + "\n  ".join(hits))
    return hits[0]


w_human = find_one(f"{RUNS}/**/*v11x_human*/weights/best.pt", "v11x_human best.pt")
w_gd = find_one(f"{RUNS}/**/*v11x_gdtiny*/weights/best.pt", "v11x_gdtiny best.pt")
print(f"Human model : {w_human}\nGD-tiny model: {w_gd}")
m_human, m_gd = YOLO(w_human), YOLO(w_gd)

test_dirs = {}
for s in SUBS:
    test_dirs[s] = find_one(f"{DATA}/**/{s}*hbb*/test/images", f"{s} human test images")
    print(f"{s:11s}: {test_dirs[s]}")


def human_boxes(img_path, w, h):
    lbl = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    boxes = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) == 5:                      # class cx cy w h
                cx, cy, bw, bh = map(float, p[1:])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
            elif len(p) > 5:                     # polygon / OBB: min/max of vertices
                xy = np.array(list(map(float, p[1:]))).reshape(-1, 2)
                boxes.append([xy[:, 0].min() * w, xy[:, 1].min() * h,
                              xy[:, 0].max() * w, xy[:, 1].max() * h])
    return np.array(boxes).reshape(-1, 4)


def predict(model, img_path):
    r = model.predict(img_path, conf=CONF, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy()


def panel(img, boxes, colour, text):
    im = img.copy()
    t = max(3, round(im.shape[1] / 250))
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(im, (x1, y1), (x2, y2), colour, t)
    s = min(PANEL_W / im.shape[1], PANEL_H / im.shape[0])
    im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
    canvas = np.full((PANEL_H, PANEL_W, 3), 255, np.uint8)
    y0, x0 = (PANEL_H - im.shape[0]) // 2, (PANEL_W - im.shape[1]) // 2
    canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    cv2.rectangle(canvas, (0, 0), (len(text) * 14 + 10, 34), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def make_figure(chosen, out_path):
    titles = ["Human labels", "YOLOv11x (Human)", "YOLOv11x (GD-tiny)"]
    head = np.full((HEADER, PANEL_W * 3 + 20, 3), 255, np.uint8)
    for i, t in enumerate(titles):
        cv2.putText(head, t, (i * (PANEL_W + 10) + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    rows = [head]
    print(f"\n{out_path}")
    print(f"{'sub':11s} {'image':45s} {'human':>6s} {'v11x_h':>7s} {'v11x_gd':>8s}")
    for s, p in chosen.items():
        img = cv2.imread(p)
        h, w = img.shape[:2]
        gt, ph, pg = human_boxes(p, w, h), predict(m_human, p), predict(m_gd, p)
        print(f"{s:11s} {os.path.basename(p)[:45]:45s} {len(gt):6d} {len(ph):7d} {len(pg):8d}")
        gap = np.full((PANEL_H, 10, 3), 255, np.uint8)
        rows.append(np.hstack([panel(img, gt, (0, 200, 0), f"{s.capitalize()}: {len(gt)}"), gap,
                               panel(img, ph, (0, 230, 255), f"{len(ph)} boxes"), gap,
                               panel(img, pg, (0, 0, 255), f"{len(pg)} boxes")]))
        rows.append(np.full((10, PANEL_W * 3 + 20, 3), 255, np.uint8))
    cv2.imwrite(out_path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def images(d):
    return sorted(f for f in glob.glob(f"{d}/*") if f.lower().endswith((".jpg", ".jpeg", ".png")))


# Random: seed 0, one per sub-collection
random.seed(0)
make_figure({s: random.choice(images(test_dirs[s])) for s in SUBS},
            f"{OUT}/qual_random.jpg")

# Failure: largest (human count - GD-tiny prediction count)
worst = {}
for s in SUBS:
    best_gap, best_p = -1e9, None
    for r in m_gd.predict(test_dirs[s], conf=CONF, verbose=False, stream=True, batch=1):
        h, w = r.orig_shape
        gap = len(human_boxes(r.path, w, h)) - len(r.boxes)
        if gap > best_gap:
            best_gap, best_p = gap, r.path
    worst[s] = best_p
make_figure(worst, f"{OUT}/qual_failure.jpg")
