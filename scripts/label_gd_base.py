import torch
import time
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

gdino_id  = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(gdino_id)
gdino     = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(device)
print("GD-base loaded")

BOX_THRESHOLD = 0.3
TEXT_PROMPT   = "fish."

base     = Path("/mnt/LinuxBackup/jo/datasets/combined_hbb")
out_base = Path("/mnt/LinuxBackup/jo/datasets/combined_gd_base")

splits = ["train", "valid", "test"]

total_images = 0
total_boxes  = 0
start_time   = time.time()

for split in splits:
    img_dir = base / split / "images"
    lbl_dir = out_base / split / "labels"
    img_out = out_base / split / "images"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in img_dir.glob("*")
                          if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"\nProcessing {split}: {len(image_paths)} images")

    for img_path in tqdm(image_paths):
        shutil.copy(img_path, img_out / img_path.name)

        image  = Image.open(img_path).convert("RGB")
        w, h   = image.size

        inputs = processor(
            images=image,
            text=TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = gdino(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        total_boxes  += len(boxes)
        total_images += 1

        yolo_lines = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw       = (x2 - x1) / w
            bh       = (y2 - y1) / h
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))

elapsed = time.time() - start_time

yaml = f"""train: {out_base}/train/images
val: {out_base}/valid/images
test: {out_base}/test/images
nc: 1
names: ['Fish']
"""
(out_base / "data.yaml").write_text(yaml)

print(f"\n{'='*50}")
print(f"Done")
print(f"Total images processed : {total_images}")
print(f"Total boxes generated  : {total_boxes}")
print(f"Avg boxes per image    : {total_boxes/total_images:.2f}")
print(f"Total labelling time   : {elapsed/60:.1f} minutes")
print(f"Time per image         : {elapsed/total_images:.2f} seconds")
