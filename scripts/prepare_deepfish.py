"""
Restructure DeepFish Kaggle dataset into standard YOLO format.
Run on server: python3 /mnt/LinuxBackup/jo/scripts/prepare_deepfish.py
"""
import os
import shutil
from pathlib import Path

SRC = Path("/mnt/LinuxBackup/jo/datasets/deepfish_kaggle/Deepfish")
DST = Path("/mnt/LinuxBackup/jo/datasets/deepfish_yolo")

def copy_split(split_name):
    img_dir = DST / split_name / "images"
    lbl_dir = DST / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    # Positive samples from each habitat
    for habitat in sorted(SRC.iterdir()):
        if not habitat.is_dir() or habitat.name == "Nagative_samples":
            continue
        split_dir = habitat / split_name
        if not split_dir.exists():
            continue
        for f in sorted(split_dir.iterdir()):
            if f.suffix == ".jpg":
                shutil.copy2(f, img_dir / f.name)
                count += 1
            elif f.suffix == ".txt":
                shutil.copy2(f, lbl_dir / f.name)

    # Negative samples (background images with empty labels)
    neg_dir = SRC / "Nagative_samples"
    if neg_dir.exists() and split_name == "valid":
        for habitat in sorted(neg_dir.iterdir()):
            if not habitat.is_dir():
                continue
            for f in sorted(habitat.iterdir()):
                if f.suffix == ".jpg":
                    shutil.copy2(f, img_dir / f.name)
                    count += 1
                elif f.suffix == ".txt":
                    shutil.copy2(f, lbl_dir / f.name)

    return count

print("Restructuring DeepFish into YOLO format...")
for split in ["train", "valid"]:
    n = copy_split(split)
    print(f"  {split}: {n} images")

# Create data.yaml
yaml_content = f"""train: {DST}/train/images
val: {DST}/valid/images

nc: 1
names: ['Fish']
"""
(DST / "data.yaml").write_text(yaml_content)
print(f"Created {DST / 'data.yaml'}")
print("Done!")
