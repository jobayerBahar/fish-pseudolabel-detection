"""
Cross-evaluation: evaluate each trained model against both human and DINO test labels.
Run on server: python3 /mnt/LinuxBackup/jo/scripts/run_cross_eval.py
"""
import subprocess
import csv
import os
import re
from pathlib import Path

RUNS = Path("/mnt/LinuxBackup/jo/runs")
DATASETS = Path("/mnt/LinuxBackup/jo/datasets")
RESULTS_DIR = Path("/mnt/LinuxBackup/jo/results/cross_eval")
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

TEST_DATASETS = {
    "human_test":  DATASETS / "combined_hbb",
    "gdtiny_test": DATASETS / "combined_gd_tiny",
    "gdbase_test": DATASETS / "combined_gd_base",
}


def eval_yolo(model_name, weights_path, data_yaml, eval_name):
    """Run YOLO model.val() and parse only the final metric lines."""
    print(f"  Evaluating {model_name} on {eval_name}...")

    result_file = RESULTS_DIR / f"{model_name}_on_{eval_name}.txt"

    script = f"""
import torch
torch.backends.cudnn.benchmark = False
from ultralytics import YOLO
model = YOLO("{weights_path}")
metrics = model.val(
    data="{data_yaml}",
    split="test",
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

    # Parse only lines with RESULT_ prefix (ignores all YOLO progress/summary junk)
    metrics = {}
    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT_"):
            key, val = line.split(":", 1)
            clean_key = key.replace("RESULT_", "").strip()
            metrics[clean_key] = val.strip()

    return metrics


def eval_rfdetr(model_name, run_dir, dataset_dir, eval_name):
    """Run RF-DETR inference on test images and compute mAP via supervision."""
    print(f"  Evaluating {model_name} on {eval_name}...")

    result_file = RESULTS_DIR / f"{model_name}_on_{eval_name}.txt"

    # Find the best checkpoint
    ckpt = run_dir / "checkpoint_best_ema.pth"
    if not ckpt.exists():
        ckpt = run_dir / "checkpoint_best_total.pth"
    if not ckpt.exists():
        pths = sorted(run_dir.glob("*.pth"))
        ckpt = pths[-1] if pths else None
    if ckpt is None:
        print(f"    No checkpoint found in {run_dir}")
        return {}

    script = f"""
import os, glob, cv2, numpy as np
import torch
from rfdetr import RFDETRLarge
import supervision as sv

# Load model from checkpoint
model = RFDETRLarge.from_checkpoint("{ckpt}")

# Test image/label dirs
img_dir = "{dataset_dir}/test/images"
lbl_dir = "{dataset_dir}/test/labels"

image_files = sorted(glob.glob(os.path.join(img_dir, "*")))
if not image_files:
    print("ERROR: No test images found")
    exit(1)

all_targets = []
all_preds = []

for img_path in image_files:
    # Get predictions
    detections = model.predict(img_path, threshold=0.3)

    # Load ground truth from YOLO label file
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
                    # Standard bbox: class cx cy w h
                    cx, cy, bw, bh = coords
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    x2 = (cx + bw/2) * w
                    y2 = (cy + bh/2) * h
                else:
                    # Polygon: class x1 y1 x2 y2 ...
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

    # Ensure predictions have class_id and confidence
    if detections.xyxy.shape[0] > 0:
        if detections.class_id is None:
            detections.class_id = np.zeros(len(detections.xyxy), dtype=int)
        else:
            detections.class_id = np.zeros(len(detections.xyxy), dtype=int)
    else:
        detections = sv.Detections(
            xyxy=np.empty((0, 4)),
            class_id=np.array([], dtype=int),
            confidence=np.array([], dtype=float),
        )

    all_targets.append(gt_det)
    all_preds.append(detections)

# Compute mAP
map_metric = sv.metrics.MeanAveragePrecision()
map_result = map_metric.update(all_preds, all_targets).compute()
map50 = float(map_result.map50)
map50_95 = float(map_result.map50_95)

# Compute precision and recall
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

    # Build eval pairs
    eval_pairs = []
    for model_name in YOLO_MODELS:
        eval_pairs.append(("yolo", model_name, "human_test"))
        if "gdtiny" in model_name:
            eval_pairs.append(("yolo", model_name, "gdtiny_test"))
        elif "gdbase" in model_name:
            eval_pairs.append(("yolo", model_name, "gdbase_test"))

    for model_name in RFDETR_MODELS:
        eval_pairs.append(("rfdetr", model_name, "human_test"))
        if "gdtiny" in model_name:
            eval_pairs.append(("rfdetr", model_name, "gdtiny_test"))
        elif "gdbase" in model_name:
            eval_pairs.append(("rfdetr", model_name, "gdbase_test"))

    print(f"Running {len(eval_pairs)} evaluations...\n")

    for model_type, model_name, eval_name in eval_pairs:
        dataset_dir = str(TEST_DATASETS[eval_name])
        data_yaml = str(TEST_DATASETS[eval_name] / "data.yaml")

        if model_type == "yolo":
            weights = str(YOLO_MODELS[model_name])
            metrics = eval_yolo(model_name, weights, data_yaml, eval_name)
        else:
            run_dir = RFDETR_MODELS[model_name]
            metrics = eval_rfdetr(model_name, run_dir, dataset_dir, eval_name)

        all_results.append({
            "model": model_name,
            "eval_set": eval_name,
            **metrics,
        })
        print(f"    -> {metrics}")

    # Save summary CSV with fixed fieldnames
    summary_file = RESULTS_DIR / "cross_eval_summary.csv"
    fieldnames = ["model", "eval_set", "mAP50", "mAP50-95", "Precision", "Recall"]
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
