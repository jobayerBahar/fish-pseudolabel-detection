"""
DeepFish benchmark: evaluate SAIAB-trained models on the external DeepFish dataset.
Tests out-of-domain generalization.
Run on server: python3 /mnt/LinuxBackup/jo/scripts/run_deepfish_benchmark.py
"""
import subprocess
import csv
import os
import glob
from pathlib import Path

RUNS = Path("/mnt/LinuxBackup/jo/runs")
DEEPFISH = Path("/mnt/LinuxBackup/jo/datasets/deepfish_yolo")
DEEPFISH_YAML = DEEPFISH / "data.yaml"
RESULTS_DIR = Path("/mnt/LinuxBackup/jo/results/deepfish_benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

YOLO_MODELS = {
    "v8s_human":   RUNS / "v8s_human/weights/best.pt",
    "v8s_gdtiny":  RUNS / "v8s_gdtiny/weights/best.pt",
    "v8s_gdbase":  RUNS / "v8s_gdbase/weights/best.pt",
    "v11x_human":  RUNS / "v11x_human/weights/best.pt",
    "v11x_gdtiny": RUNS / "v11x_gdtiny/weights/best.pt",
    "v11x_gdbase": RUNS / "v11x_gdbase/weights/best.pt",
}

RFDETR_MODELS = {
    "rfdetr_human":   RUNS / "rfdetr_human",
    "rfdetr_gdtiny":  RUNS / "rfdetr_gdtiny",
    "rfdetr_gdbase":  RUNS / "rfdetr_gdbase",
}


def eval_yolo(model_name, weights_path):
    """Run YOLO eval on DeepFish valid set."""
    print(f"  {model_name} on DeepFish...")

    result_file = RESULTS_DIR / f"{model_name}_deepfish.txt"

    script = f"""
import torch
torch.backends.cudnn.benchmark = False
from ultralytics import YOLO
model = YOLO("{weights_path}")
metrics = model.val(
    data="{DEEPFISH_YAML}",
    split="val",
    batch=8,
    workers=0,
    device=0,
    verbose=False,
)
print(f"RESULT_mAP50: {{metrics.box.map50:.4f}}")
print(f"RESULT_mAP50-95: {{metrics.box.map:.4f}}")
print(f"RESULT_Precision: {{metrics.box.mp:.4f}}")
print(f"RESULT_Recall: {{metrics.box.mr:.4f}}")
"""
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True, text=True,
        env={**os.environ, "MLFLOW_ALLOW_FILE_STORE": "true"}
    )

    with open(result_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    metrics = {}
    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT_"):
            key, val = line.split(":", 1)
            clean_key = key.replace("RESULT_", "").strip()
            metrics[clean_key] = val.strip()

    return metrics


def eval_rfdetr(model_name, run_dir):
    """Run RF-DETR inference on DeepFish valid images and compute mAP."""
    print(f"  {model_name} on DeepFish...")

    result_file = RESULTS_DIR / f"{model_name}_deepfish.txt"

    ckpt = run_dir / "checkpoint_best_ema.pth"
    if not ckpt.exists():
        ckpt = run_dir / "checkpoint_best_total.pth"
    if not ckpt.exists():
        pths = sorted(run_dir.glob("*.pth"))
        ckpt = pths[-1] if pths else None
    if ckpt is None:
        print(f"    No checkpoint found in {run_dir}")
        return {}

    img_dir = DEEPFISH / "valid" / "images"
    lbl_dir = DEEPFISH / "valid" / "labels"

    script = f"""
import os, glob, cv2, numpy as np
from rfdetr import RFDETRLarge
import supervision as sv

model = RFDETRLarge.from_checkpoint("{ckpt}")

img_dir = "{img_dir}"
lbl_dir = "{lbl_dir}"

image_files = sorted(glob.glob(os.path.join(img_dir, "*")))
if not image_files:
    print("ERROR: No images found in " + img_dir)
    exit(1)

all_targets = []
all_preds = []

for img_path in image_files:
    detections = model.predict(img_path, threshold=0.3)

    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(lbl_dir, base + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                coords = [float(v) for v in parts[1:]]
                if len(parts) == 5:
                    cx, cy, bw, bh = coords
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    x2 = (cx + bw/2) * w
                    y2 = (cy + bh/2) * h
                else:
                    xs = [coords[i] * w for i in range(0, len(coords), 2)]
                    ys = [coords[i] * h for i in range(1, len(coords), 2)]
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                gt_boxes.append([x1, y1, x2, y2])

    gt_xyxy = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))
    gt_det = sv.Detections(
        xyxy=gt_xyxy,
        class_id=np.zeros(len(gt_xyxy), dtype=int),
    )

    if detections.xyxy.shape[0] > 0:
        detections.class_id = np.zeros(len(detections.xyxy), dtype=int)
    else:
        detections = sv.Detections(
            xyxy=np.empty((0, 4)),
            class_id=np.array([], dtype=int),
            confidence=np.array([], dtype=float),
        )

    all_targets.append(gt_det)
    all_preds.append(detections)

map_metric = sv.metrics.MeanAveragePrecision()
map_result = map_metric.update(all_preds, all_targets).compute()
map50 = float(map_result.map50)
map50_95 = float(map_result.map50_95)

prec_metric = sv.metrics.Precision()
prec_result = prec_metric.update(all_preds, all_targets).compute()
prec = float(prec_result.precision_at_50) if hasattr(prec_result, 'precision_at_50') else -1

rec_metric = sv.metrics.Recall()
rec_result = rec_metric.update(all_preds, all_targets).compute()
rec = float(rec_result.recall_at_50) if hasattr(rec_result, 'recall_at_50') else -1

print(f"RESULT_mAP50: {{map50:.4f}}")
print(f"RESULT_mAP50-95: {{map50_95:.4f}}")
print(f"RESULT_Precision: {{prec:.4f}}")
print(f"RESULT_Recall: {{rec:.4f}}")
"""
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True, text=True,
        env={**os.environ, "MLFLOW_ALLOW_FILE_STORE": "true"}
    )

    with open(result_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    metrics = {}
    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT_"):
            key, val = line.split(":", 1)
            clean_key = key.replace("RESULT_", "").strip()
            metrics[clean_key] = val.strip()

    return metrics


def main():
    all_results = []

    print(f"Running DeepFish benchmark ({len(YOLO_MODELS) + len(RFDETR_MODELS)} models)...\n")

    for model_name, weights in YOLO_MODELS.items():
        metrics = eval_yolo(model_name, str(weights))
        all_results.append({"model": model_name, **metrics})
        print(f"    -> {metrics}")

    for model_name, run_dir in RFDETR_MODELS.items():
        metrics = eval_rfdetr(model_name, run_dir)
        all_results.append({"model": model_name, **metrics})
        print(f"    -> {metrics}")

    summary_file = RESULTS_DIR / "deepfish_benchmark_summary.csv"
    fieldnames = ["model", "mAP50", "mAP50-95", "Precision", "Recall"]
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
