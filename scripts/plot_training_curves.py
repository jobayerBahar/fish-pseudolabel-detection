"""Training curves for Chapter 4 (Figure training_curves.pdf).

(a) YOLO training box loss (six models)
(b) Validation mAP50:95 of all nine models, each against its own label source
    YOLOv8s dashed, YOLOv11x solid, RF-DETR-L dotted; colour = label source.

Run from anywhere:  python plot_training_curves.py
It prints every run it finds and the columns it used, so check that output.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "/mnt/LinuxBackup/jo/runs"          # searched recursively
OUT = "/mnt/LinuxBackup/jo/results/training_curves.pdf"

ARCHS = {"v8s": ("YOLOv8s", "--"), "v11x": ("YOLOv11x", "-"), "rfdetr": ("RF-DETR-L", ":")}
LABELS = {"human": ("Human", "tab:blue"), "gdtiny": ("GD-tiny", "tab:orange"),
          "gdbase": ("GD-base", "tab:green")}


def parse_name(path):
    """Return (arch, label) from a run directory path, or None."""
    name = os.path.basename(os.path.dirname(path)).lower()
    arch = next((a for a in ("rfdetr", "v11x", "v8s") if a in name), None)
    label = next((l for l in LABELS if l in name), None)
    if arch and label and "deepfish" not in name and "ft" not in name.split("_"):
        return arch, label
    return None


def pick(cols, patterns):
    for p in patterns:
        for c in cols:
            if re.search(p, c, re.I):
                return c
    return None


def load_yolo(csv):
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    return df["epoch"], df["train/box_loss"], df["metrics/mAP50-95(B)"]


def load_rfdetr(run_dir):
    """Try metrics.csv first, then log.txt (JSON lines)."""
    csv = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        print(f"    metrics.csv columns: {list(df.columns)}")
        ep = pick(df.columns, [r"^epoch$"])
        m = pick(df.columns, [r"val.*map.?50.?95", r"map.?50.?95", r"val.*map$", r"^map$"])
        if ep and m:
            d = df[[ep, m]].dropna().groupby(ep).max().reset_index()
            print(f"    using epoch='{ep}', mAP='{m}'")
            return d[ep], d[m]
    log = os.path.join(run_dir, "log.txt")
    if os.path.exists(log):
        eps, vals = [], []
        with open(log) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = next((k for k in ("test_coco_eval_bbox", "ema_test_coco_eval_bbox")
                            if k in r), None)
                if key:
                    eps.append(r.get("epoch", len(eps)))
                    vals.append(r[key][0])            # COCO AP@[.50:.95]
        if eps:
            print(f"    using log.txt key '{key}'[0]")
            return pd.Series(eps), pd.Series(vals)
    print(f"    !! no usable metrics in {run_dir}")
    return None


runs = {}
for csv in glob.glob(f"{ROOT}/**/results.csv", recursive=True):
    k = parse_name(csv)
    if k and k[0] != "rfdetr":
        runs.setdefault(k, ("yolo", csv))
for d in glob.glob(f"{ROOT}/**/", recursive=True):
    d = d.rstrip("/")
    if os.path.exists(os.path.join(d, "metrics.csv")) or os.path.exists(os.path.join(d, "log.txt")):
        k = parse_name(os.path.join(d, "x"))
        if k and k[0] == "rfdetr":
            runs.setdefault(k, ("rfdetr", d))

print("Runs found:")
for k in sorted(runs):
    print(f"  {k}: {runs[k][1]}")
if len(runs) != 9:
    print(f"!! expected 9 runs, found {len(runs)}. Edit ROOT or parse_name().")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for (arch, label), (kind, path) in sorted(runs.items()):
    aname, ls = ARCHS[arch]
    lname, col = LABELS[label]
    tag = f"{aname} ({lname})"
    print(f"{tag}:")
    if kind == "yolo":
        ep, loss, m = load_yolo(path)
        ax1.plot(ep, loss, ls=ls, color=col, label=tag)
        ax2.plot(ep, m, ls=ls, color=col, label=tag)
        print(f"    best mAP50:95 {m.max():.4f} at epoch {int(ep[m.idxmax()])}")
    else:
        r = load_rfdetr(path)
        if r is None:
            continue
        ep, m = r
        ax2.plot(ep, m, ls=ls, color=col, lw=2, label=tag)
        print(f"    best mAP50:95 {m.max():.4f} at epoch {int(ep[m.idxmax()])}")

ax1.set(xlabel="Epoch", ylabel="Box loss", title="(a) YOLO training box loss")
ax2.set(xlabel="Epoch", ylabel="mAP$_{50:95}$", title="(b) Validation mAP$_{50:95}$")
for a in (ax1, ax2):
    a.grid(alpha=0.3)
ax2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
