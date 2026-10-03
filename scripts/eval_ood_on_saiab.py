from ultralytics import YOLO
import csv
R, D = "/mnt/LinuxBackup/jo/runs", "/mnt/LinuxBackup/jo/datasets"
models = {"joint_training": f"{R}/v11x_merged/weights/best.pt",
          "finetune_human": f"{R}/v11x_finetune/weights/best.pt",
          "finetune_gdtiny": f"{R}/v11x_finetune_gdtiny/weights/best.pt"}
rows = []
for n, w in models.items():
    r = YOLO(w).val(data=f"{D}/combined_hbb/data.yaml", split="test", batch=4,
                    plots=False, project=f"{R}/ood_on_saiab", name=n)
    rows.append([n, round(r.box.map50,4), round(r.box.map,4), round(r.box.mp,4), round(r.box.mr,4)])
    print(rows[-1], flush=True)
with open("/mnt/LinuxBackup/jo/results/ood_models_on_saiab.csv", "w") as f:
    csv.writer(f).writerows([["model","mAP50","mAP50-95","Precision","Recall"]] + rows)
