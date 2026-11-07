"""How GD-tiny boxes differ from human boxes on the same fish (box_diff.jpg).

Test split, all four environments. Each GD-tiny pseudo-label box is matched to a human box
(greedy, highest IoU first, IoU >= 0.5: the same fish counted as found at mAP50).
Prints statistics over all matched pairs, then draws one typical pair per environment
(the pair whose IoU is closest to the median), zoomed in: blue = human, red = GD-tiny.
No model is run; it reads the label files only.
"""
import glob
import os

import cv2
import numpy as np

DATA = "/mnt/LinuxBackup/jo/datasets"
OUT = "/mnt/LinuxBackup/jo/results"
SUBS = ["aldabra", "tanganyika", "pondoland", "protea"]
SPLIT, IOU, MIN_W = "test", 0.5, 60          # MIN_W: human box at least 60 px wide, so the crop is readable
P = 360                                      # panel size (px)
HUM_BLUE, GD_RED = (194, 120, 74), (40, 39, 214)   # BGR: #4A78C2, #D62728


def read_boxes(lbl, w, h):
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


def match(hum, gd):
    """Greedy one-to-one matching, highest IoU first. Returns (i_hum, j_gd, iou) triples."""
    m = iou_matrix(hum, gd)
    pairs, used_h, used_g = [], set(), set()
    for k in np.argsort(-m, axis=None):
        i, j = np.unravel_index(k, m.shape)
        if m[i, j] < IOU:
            break
        if i in used_h or j in used_g:
            continue
        used_h.add(i); used_g.add(j)
        pairs.append((i, j, m[i, j]))
    return pairs


area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
all_pairs, per_env = [], {}
for sub in SUBS:
    env = []
    for p in sorted(glob.glob(f"{DATA}/{sub}_hbb/{SPLIT}/images/*")):
        stem = os.path.basename(p).rsplit(".", 1)[0]
        gl = glob.glob(f"{DATA}/combined_gd_tiny/*/labels/{stem}.txt")
        if not gl:
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        hum = read_boxes(p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt", w, h)
        gd = read_boxes(gl[0], w, h)
        for i, j, v in match(hum, gd):
            rec = dict(path=p, hum=hum[i], gd=gd[j], iou=v, ratio=area(gd[j]) / max(area(hum[i]), 1))
            env.append(rec); all_pairs.append(rec)
    per_env[sub] = env

ious = np.array([r["iou"] for r in all_pairs])
ratios = np.array([r["ratio"] for r in all_pairs])
print(f"matched pairs (IoU >= 0.5): {len(all_pairs)}")
print(f"median IoU {np.median(ious):.2f}, mean {ious.mean():.2f}")
for t in (0.75, 0.9):
    print(f"  pairs with IoU >= {t}: {(ious >= t).mean() * 100:.0f}%")
print(f"median GD/human box area {np.median(ratios):.2f}; GD box larger in {(ratios > 1).mean() * 100:.0f}% of pairs")
for sub, env in per_env.items():
    e = np.array([r["iou"] for r in env])
    r_ = np.array([r["ratio"] for r in env])
    print(f"  {sub:10s} pairs {len(env):5d}  median IoU {np.median(e):.2f}  median area ratio {np.median(r_):.2f}")

# one typical pair per environment: IoU closest to the overall median, readable size
med = np.median(ious)
panels = []
for sub in SUBS:
    cand = [r for r in per_env[sub] if r["hum"][2] - r["hum"][0] >= MIN_W]
    r = min(cand, key=lambda z: abs(z["iou"] - med))
    img = cv2.imread(r["path"])
    h, w = img.shape[:2]
    u = np.r_[np.minimum(r["hum"][:2], r["gd"][:2]), np.maximum(r["hum"][2:], r["gd"][2:])]
    cx, cy = (u[0] + u[2]) / 2, (u[1] + u[3]) / 2
    half = max(u[2] - u[0], u[3] - u[1]) * 0.75
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
    crop = img[y0:y1, x0:x1].copy()
    s = P / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
    for b, col in ((r["hum"], HUM_BLUE), (r["gd"], GD_RED)):
        bx = ((b - [x0, y0, x0, y0]) * s).astype(int)
        cv2.rectangle(crop, tuple(bx[:2]), tuple(bx[2:]), (0, 0, 0), 5)
        cv2.rectangle(crop, tuple(bx[:2]), tuple(bx[2:]), col, 3)
    c = np.full((P + 40, P, 3), 255, np.uint8)
    oy, ox = 40 + (P - crop.shape[0]) // 2, (P - crop.shape[1]) // 2
    c[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    cv2.putText(c, f"{sub.capitalize()}  IoU {r['iou']:.2f}", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    panels.append(c)
    print(f"picked {sub}: {os.path.basename(r['path'])[:40]} IoU {r['iou']:.2f}")

g = np.full((P + 40, 12, 3), 255, np.uint8)
top = np.hstack([panels[0], g, panels[1]])
bot = np.hstack([panels[2], g, panels[3]])
fig = np.vstack([top, np.full((12, top.shape[1], 3), 255, np.uint8), bot])
cv2.imwrite(f"{OUT}/box_diff.jpg", fig, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("saved box_diff.jpg")
