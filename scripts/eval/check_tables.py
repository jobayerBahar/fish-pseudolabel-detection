"""Prints the raw numbers behind Table 4.1 (best-epoch validation) and Table 3.4 (box counts)."""
import glob, json, os
import pandas as pd

J = "/mnt/LinuxBackup/jo"

# ---------- Table 4.1: best epoch by validation mAP50-95 ----------
print("TABLE 4.1  (YOLO from results.csv)")
print(f"{'run':14s} {'epoch':>5s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
for arch in ("v8s", "v11x"):
    for lab in ("human", "gdtiny", "gdbase"):
        f = f"{J}/runs/{arch}_{lab}/results.csv"
        if not os.path.exists(f):
            print(f"{arch}_{lab}: results.csv not found"); continue
        df = pd.read_csv(f); df.columns = [c.strip() for c in df.columns]
        r = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
        print(f"{arch+'_'+lab:14s} {int(r['epoch']):5d} {r['metrics/precision(B)']:7.4f} "
              f"{r['metrics/recall(B)']:7.4f} {r['metrics/mAP50(B)']:7.4f} {r['metrics/mAP50-95(B)']:9.4f}")

print("\nTABLE 4.1  (RF-DETR from log.txt)")
for log in sorted(glob.glob(f"{J}/runs/**/log.txt", recursive=True)):
    if "rfdetr" not in log.lower():
        continue
    best = None
    for line in open(log):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        for k in ("ema_test_coco_eval_bbox", "test_coco_eval_bbox"):
            if k in r and (best is None or r[k][0] > best[1][0]):
                best = (r.get("epoch"), r[k], k)
    if best:
        print(f"{log}\n   best epoch {best[0]}  mAP50-95 {best[1][0]:.4f}  mAP50 {best[1][1]:.4f}  ({best[2]})")

# ---------- Table 3.4: boxes per sub-collection and label source ----------
def nboxes(path):
    if not os.path.exists(path):
        return None
    return sum(1 for l in open(path) if l.strip())

print("\nTABLE 3.4  (boxes over train+valid+test)")
print(f"{'sub':12s} {'images':>7s} {'human':>8s} {'gd_tiny':>8s} {'gd_base':>8s} {'missing':>8s}")
tot = [0, 0, 0, 0]
for sub in ("aldabra", "tanganyika", "pondoland", "protea"):
    n = [0, 0, 0, 0]; missing = 0
    for split in ("train", "valid", "test"):
        for img in glob.glob(f"{J}/datasets/{sub}_hbb/{split}/images/*"):
            stem = os.path.splitext(os.path.basename(img))[0] + ".txt"
            n[0] += 1
            for i, src in enumerate(("combined_hbb", "combined_gd_tiny", "combined_gd_base"), 1):
                c = nboxes(f"{J}/datasets/{src}/{split}/labels/{stem}")
                if c is None:
                    missing += 1
                else:
                    n[i] += c
    tot = [a + b for a, b in zip(tot, n)]
    print(f"{sub:12s} {n[0]:7d} {n[1]:8d} {n[2]:8d} {n[3]:8d} {missing:8d}")
print(f"{'TOTAL':12s} {tot[0]:7d} {tot[1]:8d} {tot[2]:8d} {tot[3]:8d}")
