import os, glob, csv, sys, cv2, numpy as np
import supervision as sv

J = "/mnt/LinuxBackup/jo"
R, D, OUT = f"{J}/runs", f"{J}/datasets", f"{J}/results/rfdetr_reeval"
CHECK = "--check" in sys.argv
os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/rfdetr_reeval_summary.csv"

def split(ds, s): return (f"{D}/{ds}/{s}/images", f"{D}/{ds}/{s}/labels")
SETS = {"human_test": split("combined_hbb", "test"),
        "gdtiny_test": split("combined_gd_tiny", "test"),
        "gdbase_test": split("combined_gd_base", "test"),
        "deepfish": split("deepfish_yolo", "valid")}
SUBS = ["aldabra", "tanganyika", "pondoland", "protea"]
for s in SUBS: SETS[s] = split(f"{s}_hbb", "test")

MODELS = {m: f"{R}/rfdetr_{m}/checkpoint_best_ema.pth" for m in ["human", "gdtiny", "gdbase"]}
OWN = {"human": [], "gdtiny": ["gdtiny_test"], "gdbase": ["gdbase_test"]}
# Phase 1: SAIAB evals (17). Phase 2: DeepFish (3), last because of earlier OOM.
PHASES = [{m: ["human_test"] + OWN[m] + SUBS for m in MODELS},
          {m: ["deepfish"] for m in MODELS}]

def load_gt(lbl_path, w, h):
    boxes = []
    if os.path.exists(lbl_path):
        for line in open(lbl_path):
            parts = line.strip().split()
            if len(parts) < 5: continue
            c = [float(v) for v in parts[1:]]
            if len(parts) == 5:
                cx, cy, bw, bh = c
                boxes.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
            else:
                xs = [c[i]*w for i in range(0, len(c), 2)]
                ys = [c[i]*h for i in range(1, len(c), 2)]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
    xyxy = np.array(boxes) if boxes else np.empty((0, 4))
    return sv.Detections(xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int))

def empty():
    return sv.Detections(xyxy=np.empty((0, 4)), class_id=np.array([], dtype=int),
                         confidence=np.array([], dtype=float))

done = set()
if os.path.exists(CSV):
    done = {(r["model"], r["eval_set"]) for r in csv.DictReader(open(CSV))}
else:
    csv.writer(open(CSV, "w")).writerow(["model", "eval_set", "n_images", "mAP50", "mAP50-95",
                                         "Precision", "Recall", "mAP50_at0.3", "mAP50-95_at0.3"])

for phase in PHASES:
    for m, sets in phase.items():
        todo = [s for s in sets if (f"rfdetr_{m}", s) not in done]
        if not todo: continue
        if not CHECK:
            import torch
            from rfdetr import RFDETRLarge
            model = RFDETRLarge.from_checkpoint(MODELS[m])
        for s in todo:
            img_dir, lbl_dir = SETS[s]
            imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
            if CHECK:
                n_lbl = len(glob.glob(os.path.join(lbl_dir, "*.txt")))
                print(f"rfdetr_{m:7s} {s:12s} images={len(imgs):5d} labels={n_lbl:5d} ckpt={os.path.exists(MODELS[m])}")
                continue
            p_all, p_03, gts = [], [], []
            for p in imgs:
                img = cv2.imread(p)
                if img is None: continue
                h, w = img.shape[:2]
                det = model.predict(p, threshold=0.001)
                if len(det) == 0:
                    det = empty()
                else:
                    det.class_id = np.zeros(len(det.xyxy), dtype=int)
                p_all.append(det)
                p_03.append(det[det.confidence >= 0.3])
                gts.append(load_gt(os.path.join(lbl_dir, os.path.splitext(os.path.basename(p))[0] + ".txt"), w, h))
            full = sv.metrics.MeanAveragePrecision().update(p_all, gts).compute()
            old = sv.metrics.MeanAveragePrecision().update(p_03, gts).compute()
            pr = sv.metrics.Precision().update(p_03, gts).compute()
            rc = sv.metrics.Recall().update(p_03, gts).compute()
            row = [f"rfdetr_{m}", s, len(gts), round(float(full.map50), 4), round(float(full.map50_95), 4),
                   round(float(pr.precision_at_50), 4), round(float(rc.recall_at_50), 4),
                   round(float(old.map50), 4), round(float(old.map50_95), 4)]
            with open(CSV, "a") as f: csv.writer(f).writerow(row)
            print(row, flush=True)
        if not CHECK:
            del model; torch.cuda.empty_cache()
