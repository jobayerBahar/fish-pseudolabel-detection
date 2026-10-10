"""
DeepFish background-image analysis.

Scores the existing models (no retraining) on three versions of the DeepFish validation set:
  A  fish only          909 images
  B  fish + 50% empty   909 + half of the background images, sampled per habitat (seed 0)
  C  fish + all empty   2,921 images (as reported in the thesis)
and counts false alarms on the 2,012 background images (confidence >= 0.3).

Usage (on the server, inside ~/fish_env):
  python3 deepfish_negatives_eval.py rfdetr   # CPU only, uses the saved RF-DETR prediction cache
  python3 deepfish_negatives_eval.py yolo     # needs the GPU
Results are appended to results/deepfish_negatives/summary.csv
"""
import os, sys, re, csv, glob, random
from collections import defaultdict
import numpy as np

J = "/mnt/LinuxBackup/jo"
DF = f"{J}/datasets/deepfish_yolo"
IMG, LBL = f"{DF}/valid/images", f"{DF}/valid/labels"
RF_CACHE = f"{J}/results/rfdetr_reeval/deepfish_cache"
OUT = f"{J}/results/deepfish_negatives"
os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/summary.csv"
FP_CONF = 0.3

YOLO_MODELS = {
    "v8s_human":   f"{J}/runs/v8s_human/weights/best.pt",
    "v8s_gdtiny":  f"{J}/runs/v8s_gdtiny/weights/best.pt",
    "v8s_gdbase":  f"{J}/runs/v8s_gdbase/weights/best.pt",
    "v11x_human":  f"{J}/runs/v11x_human/weights/best.pt",
    "v11x_gdtiny": f"{J}/runs/v11x_gdtiny/weights/best.pt",
    "v11x_gdbase": f"{J}/runs/v11x_gdbase/weights/best.pt",
    # DeepFish-trained YOLOv11x: fill in the run folder names
    "df_only":        f"{J}/runs/v11x_deepfish_retrain/weights/best.pt",
    "df_joint":       f"{J}/runs/v11x_merged/weights/best.pt",
    "df_ft_human":    f"{J}/runs/v11x_finetune/weights/best.pt",
    "df_ft_gdtiny":   f"{J}/runs/v11x_finetune_gdtiny/weights/best.pt",
}
RFDETR_MODELS = ["human", "gdtiny", "gdbase"]


# ---------- the three image sets ----------
def is_empty(img_path):
    lbl = os.path.join(LBL, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    return (not os.path.exists(lbl)) or os.path.getsize(lbl) == 0 or not open(lbl).read().strip()

def habitat(name):
    m = re.match(r"(\d{4})", os.path.basename(name))
    return m.group(1) if m else "unknown"

def build_sets():
    imgs = sorted(glob.glob(os.path.join(IMG, "*")))
    pos = [p for p in imgs if not is_empty(p)]
    neg = [p for p in imgs if is_empty(p)]
    by_hab = defaultdict(list)
    for p in neg:
        by_hab[habitat(p)].append(p)
    rng = random.Random(0)
    half = []
    for h in sorted(by_hab):
        items = sorted(by_hab[h])
        half += rng.sample(items, round(len(items) / 2))
    sets = {"A_fish_only": pos, "B_half_empty": pos + sorted(half), "C_all_empty": imgs}
    print(f"positives {len(pos)}, negatives {len(neg)}, half {len(half)}")
    print("negatives per habitat:", {h: len(v) for h, v in sorted(by_hab.items())})
    for k, v in sets.items():
        with open(f"{OUT}/{k}.txt", "w") as f:
            f.write("\n".join(v) + "\n")
    return sets, neg

def write_row(row):
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "set", "images", "mAP50", "mAP50-95",
                        "neg_images", "neg_with_fp_pct", "fp_per_neg_image"])
        w.writerow(row)
    print(row, flush=True)

def fp_stats(box_counts):
    """box_counts: number of boxes (conf >= FP_CONF) on each background image."""
    c = np.asarray(box_counts)
    return len(c), round(100 * float((c > 0).mean()), 1), round(float(c.mean()), 3)


# ---------- RF-DETR: from the saved predictions, CPU only ----------
def run_rfdetr(sets, neg):
    import supervision as sv
    def load_gt(path, w, h):
        boxes = []
        if os.path.exists(path):
            for line in open(path):
                p = line.split()
                if len(p) < 5:
                    continue
                cx, cy, bw, bh = map(float, p[1:5])
                boxes.append([(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h])
        a = np.array(boxes) if boxes else np.empty((0, 4))
        return sv.Detections(xyxy=a, class_id=np.zeros(len(a), dtype=int))
    neg_names = {os.path.basename(p) for p in neg}
    for m in RFDETR_MODELS:
        preds, gts = {}, {}
        for f in sorted(x for x in glob.glob(f"{RF_CACHE}/{m}_*.npz") if ".tmp" not in x):
            z = np.load(f); off = 0
            for name, n, (h, w) in zip(z["names"], z["cnt"], z["hw"]):
                name = str(name)
                if n:
                    d = sv.Detections(xyxy=z["xyxy"][off:off+n].astype(float),
                                      confidence=z["conf"][off:off+n].astype(float),
                                      class_id=np.zeros(n, dtype=int))
                else:
                    d = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.array([], dtype=float),
                                      class_id=np.array([], dtype=int))
                off += n
                preds[name] = d
                gts[name] = load_gt(os.path.join(LBL, os.path.splitext(name)[0] + ".txt"), w, h)
        assert len(preds) == len(sets["C_all_empty"]), f"cache has {len(preds)} images"
        fp = fp_stats([int((preds[n].confidence >= FP_CONF).sum()) for n in neg_names])
        for s, paths in sets.items():
            names = [os.path.basename(p) for p in paths]
            r = sv.metrics.MeanAveragePrecision().update([preds[n] for n in names],
                                                         [gts[n] for n in names]).compute()
            write_row([f"rfdetr_{m}", s, len(names), round(float(r.map50), 4),
                       round(float(r.map50_95), 4), *fp])


# ---------- YOLO: Ultralytics val on each set + false alarms, GPU ----------
def run_yolo(sets, neg):
    import yaml, torch
    torch.backends.cudnn.benchmark = False
    from ultralytics import YOLO
    names = yaml.safe_load(open(f"{DF}/data.yaml"))["names"]
    yamls = {}
    for s in sets:
        y = f"{OUT}/{s}.yaml"
        yaml.safe_dump({"train": f"{OUT}/{s}.txt", "val": f"{OUT}/{s}.txt", "names": names}, open(y, "w"))
        yamls[s] = y
    for m, w in YOLO_MODELS.items():
        if not os.path.exists(w):
            print(f"skip {m}: {w} not found"); continue
        model = YOLO(w)
        counts = []
        for p in neg:
            r = model.predict(p, conf=FP_CONF, imgsz=640, verbose=False, device=0)[0]
            counts.append(len(r.boxes))
        fp = fp_stats(counts)
        for s in sets:
            v = model.val(data=yamls[s], split="val", batch=4, workers=0, device=0,
                          verbose=False, plots=False)
            write_row([m, s, len(sets[s]), round(float(v.box.map50), 4),
                       round(float(v.box.map), 4), *fp])


if __name__ == "__main__":
    sets, neg = build_sets()
    {"rfdetr": run_rfdetr, "yolo": run_yolo}[sys.argv[1]](sets, neg)
