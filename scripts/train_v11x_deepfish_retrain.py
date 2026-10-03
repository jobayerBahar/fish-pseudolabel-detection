from ultralytics import YOLO
YOLO("/home/jo/yolo11x.pt").train(
    data="/mnt/LinuxBackup/jo/datasets/deepfish_yolo/data.yaml",
    epochs=100, batch=4, imgsz=640, amp=False, workers=0,
    optimizer="AdamW", lr0=0.002, momentum=0.9,
    project="/mnt/LinuxBackup/jo/runs", name="v11x_deepfish_retrain", exist_ok=False,
)
# after training: test on SAIAB human test set
r = YOLO("/mnt/LinuxBackup/jo/runs/v11x_deepfish_retrain/weights/best.pt").val(
    data="/mnt/LinuxBackup/jo/datasets/combined_hbb/data.yaml", split="test", batch=4,
    plots=False, project="/mnt/LinuxBackup/jo/runs/ood_on_saiab", name="deepfish_only")
row = f"deepfish_only,{r.box.map50:.4f},{r.box.map:.4f},{r.box.mp:.4f},{r.box.mr:.4f}"
print("RESULT", row, flush=True)
open("/mnt/LinuxBackup/jo/results/deepfish_only_on_saiab.csv", "w").write(
    "model,mAP50,mAP50-95,Precision,Recall\n" + row + "\n")
