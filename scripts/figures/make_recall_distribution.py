"""
Per-image recall on the SAIAB test split, YOLOv11x (Human) vs YOLOv11x (GD-tiny).

For every test image: share of human-labelled fish matched one-to-one by a prediction
(IoU >= 0.5, confidence >= 0.25, as in Figures 4.2 and 4.3).
Plots the share of fish found, grouped by how many fish an image contains, and prints
where the images of Figure 4.2 (random, seed 0) and Figure 4.3 (largest shortfall) fall.

Usage (server, inside ~/fish_env):
  python3 make_recall_distribution.py          # runs inference once (GPU), then plots
  python3 make_recall_distribution.py plot     # re-plot from the saved CSV only
Outputs: results/recall_distribution.csv, results/recall_distribution.pdf
"""
import csv, glob, os, random, sys
import numpy as np

J = "/mnt/LinuxBackup/jo"
SUBS = ["aldabra", "tanganyika", "pondoland", "protea"]
NAMES = {"aldabra": "Aldabra", "tanganyika": "Tanganyika", "pondoland": "Pondoland", "protea": "Protea"}
MODELS = {"human": f"{J}/runs/v11x_human/weights/best.pt",
          "gdtiny": f"{J}/runs/v11x_gdtiny/weights/best.pt"}
CONF, IOU = 0.25, 0.5
OUT_CSV = f"{J}/results/recall_distribution.csv"
OUT_PDF = f"{J}/results/recall_distribution.pdf"
HUMAN_C, GD_C = "#4A78C2", "#D62728"


def images(sub):
    d = f"{J}/datasets/{sub}_hbb/test/images"
    return sorted(f for f in glob.glob(f"{d}/*") if f.lower().endswith((".jpg", ".jpeg", ".png")))


def human_boxes(img_path, w, h):
    lbl = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    boxes = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) == 5:
                cx, cy, bw, bh = map(float, p[1:])
                boxes.append([(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h])
            elif len(p) > 5:
                xy = np.array(list(map(float, p[1:]))).reshape(-1, 2)
                boxes.append([xy[:, 0].min()*w, xy[:, 1].min()*h, xy[:, 0].max()*w, xy[:, 1].max()*h])
    return np.array(boxes).reshape(-1, 4)


def matched(gt, pred):
    """Greedy one-to-one matching, highest IoU first. Returns number of matched human boxes."""
    if len(gt) == 0 or len(pred) == 0:
        return 0
    x1 = np.maximum(gt[:, None, 0], pred[None, :, 0]); y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pred[None, :, 2]); y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]); ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    iou = inter / (ag[:, None] + ap[None, :] - inter + 1e-9)
    n, used_g, used_p = 0, set(), set()
    for gi, pi in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if iou[gi, pi] < IOU:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi); n += 1
    return n


def run_inference():
    import cv2
    from ultralytics import YOLO
    models = {k: YOLO(v) for k, v in MODELS.items()}
    rows = []
    for s in SUBS:
        for p in images(s):
            h, w = cv2.imread(p).shape[:2]
            gt = human_boxes(p, w, h)
            row = {"sub": s, "image": os.path.basename(p), "n_gt": len(gt)}
            for k, m in models.items():
                pred = m.predict(p, conf=CONF, imgsz=640, verbose=False, device=0)[0].boxes.xyxy.cpu().numpy()
                row[f"pred_{k}"] = len(pred)
                row[f"match_{k}"] = matched(gt, pred)
            rows.append(row)
        print(f"{s}: {sum(r['sub'] == s for r in rows)} images", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def load():
    rows = list(csv.DictReader(open(OUT_CSV)))
    for r in rows:
        for k in r:
            if k not in ("sub", "image"):
                r[k] = int(r[k])
    return rows


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BINS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10**6)]
    LABELS = ["1-2", "3-5", "6-10", "11-20", "21-40", "> 40"]
    rows = [r for r in rows if r["n_gt"] > 0]
    stats = {"human": [], "gdtiny": []}
    print(f"\n{'fish/image':>10s} {'images':>6s} {'fish':>6s} {'human':>6s} {'gdtiny':>6s}  images per sub-collection")
    for (lo, hi), lab in zip(BINS, LABELS):
        b = [r for r in rows if lo <= r["n_gt"] <= hi]
        fish = sum(r["n_gt"] for r in b)
        for k in stats:
            stats[k].append(100 * sum(r[f"match_{k}"] for r in b) / fish)
        print(f"{lab:>10s} {len(b):6d} {fish:6d} {stats['human'][-1]:5.0f}% {stats['gdtiny'][-1]:5.0f}%  "
              + ", ".join(f"{NAMES[s]} {sum(r['sub'] == s for r in b)}" for s in SUBS))
    counts = [sum(lo <= r["n_gt"] <= hi for r in rows) for lo, hi in BINS]

    # where the images of Figures 4.2 and 4.3 fall (chosen as make_qualitative.py does)
    random.seed(0)
    for s in SUBS:
        allr = [r for r in load() if r["sub"] == s]
        rnd = os.path.basename(random.choice(images(s)))
        worst = max(allr, key=lambda r: r["n_gt"] - r["pred_gdtiny"])
        r = next(x for x in allr if x["image"] == rnd)
        print(f"{NAMES[s]:11s} Fig 4.2: {r['n_gt']} fish, GD found {r['match_gdtiny']}, human-trained found {r['match_human']} | "
              f"Fig 4.3: {worst['n_gt']} fish, GD found {worst['match_gdtiny']}, human-trained found {worst['match_human']}")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(BINS)); w = 0.36
    for k, c, off, name in (("human", HUMAN_C, -w/2 - 0.01, "YOLOv11x (Human)"),
                            ("gdtiny", GD_C, w/2 + 0.01, "YOLOv11x (GD-tiny)")):
        bars = ax.bar(x + off, stats[k], width=w, color=c, label=name)
        for xb, v in zip(x + off, stats[k]):
            ax.text(xb, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(x, [f"{l}\n({n} images)" for l, n in zip(LABELS, counts)])
    ax.set_xlabel("Human-labelled fish per test image")
    ax.set_ylabel("Fish found (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print(f"\nSaved {OUT_PDF}")


if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "plot" and os.path.exists(OUT_CSV)):
        run_inference()
    plot(load())
