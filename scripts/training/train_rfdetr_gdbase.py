import os
from rfdetr import RFDETRLarge
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model = RFDETRLarge()
model.train(
    dataset_dir="/mnt/LinuxBackup/jo/datasets/combined_gd_base",
    dataset_file="yolo",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    checkpoint_interval=50,
    output_dir="/mnt/LinuxBackup/jo/runs/rfdetr_gdbase",
)
