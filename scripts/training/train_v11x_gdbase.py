import torch
torch.backends.cudnn.benchmark = False
from ultralytics import YOLO
import os

last_ckpt = "/mnt/LinuxBackup/jo/runs/v11x_gdbase/weights/last.pt"
if os.path.exists(last_ckpt):
    print(f"Resuming from {last_ckpt}")
    model = YOLO(last_ckpt)
    model.train(resume=True)
else:
    print("Starting fresh training")
    model = YOLO("/home/jo/yolo11x.pt")
    model.train(
        data="/mnt/LinuxBackup/jo/datasets/combined_gd_base/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        workers=0,
        optimizer="SGD",
        amp=False,
        name="v11x_gdbase",
        project="/mnt/LinuxBackup/jo/runs",
        device=0,
    )
