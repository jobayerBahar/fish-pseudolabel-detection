from ultralytics import YOLO
import os

last_ckpt = "/mnt/LinuxBackup/jo/runs/v8s_gdbase/weights/last.pt"

if os.path.exists(last_ckpt):
    print(f"Resuming from {last_ckpt}")
    model = YOLO(last_ckpt)
    model.train(resume=True)
else:
    print("Starting fresh training")
    model = YOLO("/home/jo/yolov8s.pt")
    model.train(
        data="/mnt/LinuxBackup/jo/datasets/combined_gd_base/data.yaml",
        epochs=100,
        batch=16,
        workers=0,
        name="v8s_gdbase",
        project="/mnt/LinuxBackup/jo/runs",
    )
