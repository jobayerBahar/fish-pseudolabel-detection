import shutil
from pathlib import Path

base = Path("/mnt/LinuxBackup/jo/datasets")
out  = Path("/mnt/LinuxBackup/jo/datasets/combined_hbb")

sources = [
    base / "aldabra_hbb",
    base / "tanganyika_hbb",
    base / "pondoland_hbb",
    base / "protea_hbb",
]

splits = ["train", "valid", "test"]

for split in splits:
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)

for source in sources:
    print(f"Merging {source.name}...")
    for split in splits:
        for img in (source / split / "images").glob("*"):
            shutil.copy(img, out / split / "images" / img.name)
        for lbl in (source / split / "labels").glob("*"):
            shutil.copy(lbl, out / split / "labels" / lbl.name)

# Write data.yaml
yaml = f"""train: {out}/train/images
val: {out}/valid/images
test: {out}/test/images
nc: 1
names: ['Fish']
"""
(out / "data.yaml").write_text(yaml)

# Print counts
for split in splits:
    n = len(list((out / split / "images").glob("*")))
    print(f"{split}: {n} images")

print("Done ✅")
