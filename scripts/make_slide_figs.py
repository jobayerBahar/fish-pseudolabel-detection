"""Slide figures for Seminar 3.

demo_pipeline.jpg       : walkthrough on one test image (the Protea image from qual_random)
                          input | Grounding DINO pseudo-labels | GD-trained detector | human labels
qual_failure_slide.jpg  : the failure cases, laid out wide (4 columns x 3 rows) to fit a 16:9 slide
"""
import glob
import os
import random

import cv2
import numpy as np
from ultralytics import YOLO

RUNS = "/mnt/LinuxBackup/jo/runs"
DATA = "/mnt/LinuxBackup/jo/datasets"
OUT = "/mnt/LinuxBackup/jo/results"
SUBS = ["aldabra", "tanganyika", "pondoland", "protea"]
CONF = 0.25
PW, PH = 640, 480

m_human = YOLO(f"{RUNS}/v11x_human/weights/best.pt")
m_gd = YOLO(f"{RUNS}/v11x_gdtiny/weights/best.pt")
test_dirs = {s: f"{DATA}/{s}_hbb/test/images" for s in SUBS}


def images(d):
    return sorted(f for f in glob.glob(f"{d}/*") if f.lower().endswith((".jpg", ".jpeg", ".png")))


def read_boxes(lbl, w, h):
    boxes = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) < 5:
                continue
            xy = np.array(list(map(float, p[1:])))
            if len(p) == 5:
                cx, cy, bw, bh = xy
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            else:
                xy = xy.reshape(-1, 2)
                boxes.append([xy[:, 0].min() * w, xy[:, 1].min() * h, xy[:, 0].max() * w, xy[:, 1].max() * h])
    return np.array(boxes).reshape(-1, 4)


def human_boxes(p, w, h):
    return read_boxes(p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt", w, h)


def gd_boxes(p, w, h):
    stem = os.path.basename(p).rsplit(".", 1)[0]
    hits = glob.glob(f"{DATA}/combined_gd_tiny/*/labels/{stem}.txt")
    if not hits:
        print(f"!! no GD-tiny label found for {stem}")
    return read_boxes(hits[0], w, h) if hits else np.zeros((0, 4))


def predict(model, p):
    return model.predict(p, conf=CONF, verbose=False)[0].boxes.xyxy.cpu().numpy()


def panel(img, boxes, colour, text):
    im = img.copy()
    t = max(3, round(im.shape[1] / 250))
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(im, (x1, y1), (x2, y2), colour, t)
    s = min(PW / im.shape[1], PH / im.shape[0])
    im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
    c = np.full((PH + 44, PW, 3), 255, np.uint8)
    y0, x0 = 44 + (PH - im.shape[0]) // 2, (PW - im.shape[1]) // 2
    c[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    cv2.putText(c, text, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    return c


def grid(rows):
    gap_v = np.full((PH + 44, 12, 3), 255, np.uint8)
    out = []
    for r in rows:
        line = []
        for i, p in enumerate(r):
            line += [p] + ([gap_v] if i < len(r) - 1 else [])
        out.append(np.hstack(line))
        out.append(np.full((12, out[-1].shape[1], 3), 255, np.uint8))
    return np.vstack(out[:-1])


# Walkthrough: same seed as qual_random, take the Protea image
random.seed(0)
picks = {s: random.choice(images(test_dirs[s])) for s in SUBS}
p = picks["protea"]
img = cv2.imread(p)
h, w = img.shape[:2]
gt, gd, pg = human_boxes(p, w, h), gd_boxes(p, w, h), predict(m_gd, p)
demo = grid([
    [panel(img, np.zeros((0, 4)), (0, 0, 0), '1. Input image + prompt "fish."'),
     panel(img, gd, (0, 165, 255), f"2. Grounding DINO labels: {len(gd)}")],
    [panel(img, pg, (0, 0, 255), f"3. Detector trained on them: {len(pg)}"),
     panel(img, gt, (0, 200, 0), f"Human labels (reference): {len(gt)}")],
])
cv2.imwrite(f"{OUT}/demo_pipeline.jpg", demo, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"demo: {os.path.basename(p)}  GD labels {len(gd)}, GD-trained {len(pg)}, human {len(gt)}")

# Failure cases, wide layout
worst = {}
for s in SUBS:
    best_gap, best_p = -1e9, None
    for r in m_gd.predict(test_dirs[s], conf=CONF, verbose=False, stream=True, batch=1):
        hh, ww = r.orig_shape
        g = len(human_boxes(r.path, ww, hh)) - len(r.boxes)
        if g > best_gap:
            best_gap, best_p = g, r.path
    worst[s] = best_p
rows = [[], [], []]
for s in SUBS:
    p = worst[s]
    img = cv2.imread(p)
    h, w = img.shape[:2]
    gt, ph, pg = human_boxes(p, w, h), predict(m_human, p), predict(m_gd, p)
    rows[0].append(panel(img, gt, (0, 200, 0), f"{s.capitalize()} - human labels: {len(gt)}"))
    rows[1].append(panel(img, ph, (0, 230, 255), f"Human-trained: {len(ph)}"))
    rows[2].append(panel(img, pg, (0, 0, 255), f"GD-trained: {len(pg)}"))
cv2.imwrite(f"{OUT}/qual_failure_slide.jpg", grid(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved demo_pipeline.jpg and qual_failure_slide.jpg")
