from ultralytics import YOLO
import yaml

R = "/mnt/LinuxBackup/jo/runs"
w = f"{R}/v11x_gdtiny/weights/best.pt"

a = yaml.safe_load(open(f"{R}/v11x_finetune/args.yaml"))
for k in ["save_dir", "name", "project", "model", "resume", "exist_ok", "cfg", "mode",
          "optimizer", "lr0", "momentum"]:
    a.pop(k, None)

YOLO(w).train(**a, project=R, name="v11x_finetune_gdtiny", exist_ok=False,
              optimizer="AdamW", lr0=0.002, momentum=0.9)
