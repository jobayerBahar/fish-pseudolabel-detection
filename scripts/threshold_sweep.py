import glob, os, csv, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv

D = "/mnt/LinuxBackup/jo/datasets"
IMG, LBL = f"{D}/combined_hbb/test/images", f"{D}/combined_hbb/test/labels"
MID = "IDEA-Research/grounding-dino-tiny"
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]

proc = AutoProcessor.from_pretrained(MID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MID).to("cuda").eval()

def load_gt(path, w, h):
    boxes = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) < 5: continue
            c = [float(v) for v in p[1:]]
            if len(p) == 5:
                cx, cy, bw, bh = c
                boxes.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
            else:
                xs, ys = [v*w for v in c[0::2]], [v*h for v in c[1::2]]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
    a = np.array(boxes) if boxes else np.empty((0, 4))
    return sv.Detections(xyxy=a, class_id=np.zeros(len(a), dtype=int))

raw, gts = [], []
for p in sorted(glob.glob(os.path.join(IMG, "*"))):
    img = Image.open(p).convert("RGB"); w, h = img.size
    inp = proc(images=img, text="fish.", return_tensors="pt").to("cuda")
    with torch.no_grad(): out = model(**inp)
    try:
        r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.05, target_sizes=[(h, w)])[0]
    except TypeError:
        r = proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=0.05, target_sizes=[(h, w)])[0]
    raw.append((r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()))
    gts.append(load_gt(os.path.join(LBL, os.path.splitext(os.path.basename(p))[0] + ".txt"), w, h))

n_gt = sum(len(g) for g in gts)
rows = []
for t in THRESHOLDS:
    preds = []
    for b, s in raw:
        k = s >= t
        preds.append(sv.Detections(xyxy=b[k].reshape(-1, 4), confidence=s[k], class_id=np.zeros(k.sum(), dtype=int)))
    n = sum(len(d) for d in preds)
    m = sv.metrics.MeanAveragePrecision().update(preds, gts).compute()
    pr = sv.metrics.Precision().update(preds, gts).compute().precision_at_50
    rc = sv.metrics.Recall().update(preds, gts).compute().recall_at_50
    rows.append([t, n, round(n/len(gts), 2), round(n/n_gt, 3), round(float(pr), 4), round(float(rc), 4),
                 round(float(m.map50), 4), round(float(m.map50_95), 4)])
    print(rows[-1], flush=True)

with open("/mnt/LinuxBackup/jo/results/threshold_sweep.csv", "w") as f:
    csv.writer(f).writerows([["threshold","boxes","boxes_per_img","coverage_vs_human","precision@50","recall@50","mAP50","mAP50-95"]] + rows)
print("human test boxes:", n_gt)
