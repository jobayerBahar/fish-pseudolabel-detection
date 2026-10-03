import os, sys, glob, csv, numpy as np

J = "/mnt/LinuxBackup/jo"
R, D, OUT = f"{J}/runs", f"{J}/datasets", f"{J}/results/rfdetr_reeval"
CACHE = f"{OUT}/deepfish_cache"; os.makedirs(CACHE, exist_ok=True)
CSV = f"{OUT}/rfdetr_reeval_summary.csv"
IMG, LBL = f"{D}/deepfish_yolo/valid/images", f"{D}/deepfish_yolo/valid/labels"
imgs = sorted(glob.glob(os.path.join(IMG, "*")))
mode, m = sys.argv[1], sys.argv[2]

if mode == "infer":
    s, e = int(sys.argv[3]), int(sys.argv[4])
    out = f"{CACHE}/{m}_{s:05d}.npz"
    if os.path.exists(out): sys.exit(0)
    import cv2
    from rfdetr import RFDETRLarge
    model = RFDETRLarge.from_checkpoint(f"{R}/rfdetr_{m}/checkpoint_best_ema.pth")
    names, cnt, hw, xy, cf = [], [], [], [], []
    for p in imgs[s:e]:
        img = cv2.imread(p)
        if img is None: continue
        det = model.predict(p, threshold=0.001)
        n = len(det)
        names.append(os.path.basename(p)); cnt.append(n); hw.append(img.shape[:2])
        if n:
            xy.append(np.asarray(det.xyxy, dtype=np.float32))
            cf.append(np.asarray(det.confidence, dtype=np.float32))
    tmp = f"{CACHE}/{m}_{s:05d}.tmp.npz"
    np.savez(tmp, names=np.array(names), cnt=np.array(cnt), hw=np.array(hw),
             xyxy=np.concatenate(xy) if xy else np.empty((0, 4), np.float32),
             conf=np.concatenate(cf) if cf else np.empty((0,), np.float32))
    os.replace(tmp, out)
    print(f"{m} chunk {s}-{e}: {len(names)} images", flush=True)

elif mode == "metrics":
    import supervision as sv
    if any(r["model"] == f"rfdetr_{m}" and r["eval_set"] == "deepfish" for r in csv.DictReader(open(CSV))):
        sys.exit(0)
    def load_gt(path, w, h):
        boxes = []
        if os.path.exists(path):
            for line in open(path):
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
        a = np.array(boxes) if boxes else np.empty((0, 4))
        return sv.Detections(xyxy=a, class_id=np.zeros(len(a), dtype=int))
    files = sorted(f for f in glob.glob(f"{CACHE}/{m}_*.npz") if ".tmp" not in f)
    p_all, p_03, gts = [], [], []
    for f in files:
        z = np.load(f)
        off = 0
        for name, n, (h, w) in zip(z["names"], z["cnt"], z["hw"]):
            if n:
                det = sv.Detections(xyxy=z["xyxy"][off:off+n].astype(float),
                                    confidence=z["conf"][off:off+n].astype(float),
                                    class_id=np.zeros(n, dtype=int))
            else:
                det = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.array([], dtype=float),
                                    class_id=np.array([], dtype=int))
            off += n
            p_all.append(det); p_03.append(det[det.confidence >= 0.3])
            gts.append(load_gt(os.path.join(LBL, os.path.splitext(str(name))[0] + ".txt"), w, h))
    assert len(gts) == len(imgs), f"expected {len(imgs)} images, got {len(gts)}"
    full = sv.metrics.MeanAveragePrecision().update(p_all, gts).compute()
    old = sv.metrics.MeanAveragePrecision().update(p_03, gts).compute()
    pr = sv.metrics.Precision().update(p_03, gts).compute()
    rc = sv.metrics.Recall().update(p_03, gts).compute()
    row = [f"rfdetr_{m}", "deepfish", len(gts), round(float(full.map50), 4), round(float(full.map50_95), 4),
           round(float(pr.precision_at_50), 4), round(float(rc.recall_at_50), 4),
           round(float(old.map50), 4), round(float(old.map50_95), 4)]
    with open(CSV, "a") as fh: csv.writer(fh).writerow(row)
    print(row, flush=True)
