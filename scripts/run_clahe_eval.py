from ultralytics import YOLO
import csv

R = "/mnt/LinuxBackup/jo/runs"
D = "/mnt/LinuxBackup/jo/datasets"
models = {"v11x_human": f"{R}/v11x_human/weights/best.pt",
          "v11x_gdtiny": f"{R}/v11x_gdtiny/weights/best.pt"}
sets = {"original": f"{D}/aldabra_hbb/data.yaml",
        "clahe": f"{D}/aldabra_enhanced/data.yaml"}

rows = []
for m_name, w in models.items():
    for s_name, y in sets.items():
        r = YOLO(w).val(data=y, split="test", batch=4, plots=False,
                        project=f"{R}/clahe_rerun", name=f"{m_name}_{s_name}")
        rows.append([m_name, s_name, round(r.box.map50, 4), round(r.box.map, 4),
                     round(r.box.mp, 4), round(r.box.mr, 4)])
        print(rows[-1])

with open("/mnt/LinuxBackup/jo/results/clahe_eval_summary.csv", "w") as f:
    csv.writer(f).writerows([["model","condition","mAP50","mAP50-95","Precision","Recall"]] + rows)
