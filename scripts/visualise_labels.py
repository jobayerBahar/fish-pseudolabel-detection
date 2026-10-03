import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

gdino_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(gdino_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(device)
print("Model loaded ✅")

img_dir  = Path("/home/jo/experiment2/fish_dataset/test/images")
lbl_dir  = Path("/home/jo/experiment2/fish_dataset/test/labels")
out_dir  = Path("/mnt/LinuxBackup/jo/visualisations")
out_dir.mkdir(parents=True, exist_ok=True)

image_paths = sorted([p for p in img_dir.glob("*")
                      if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
print(f"Processing {len(image_paths)} images...")

for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")
    w, h  = image.size

    # ── Left: human labels ──────────────────────────────────────────────
    human_img  = image.copy()
    draw_human = ImageDraw.Draw(human_img)
    lbl_path   = lbl_dir / (img_path.stem + ".txt")
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, xc, yc, bw, bh = map(float, parts)
                    x1 = (xc - bw/2) * w
                    y1 = (yc - bh/2) * h
                    x2 = (xc + bw/2) * w
                    y2 = (yc + bh/2) * h
                    draw_human.rectangle([x1, y1, x2, y2],
                                         outline="green", width=3)

    # ── Right: DINO labels ───────────────────────────────────────────────
    dino_img  = image.copy()
    draw_dino = ImageDraw.Draw(dino_img)
    inputs = processor(images=image, text="fish.",
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.3,
        target_sizes=[(h, w)]
    )[0]
    for box in results["boxes"].cpu().numpy():
        x1, y1, x2, y2 = box
        draw_dino.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # ── Side by side ─────────────────────────────────────────────────────
    combined = Image.new("RGB", (w * 2 + 10, h), color=(200, 200, 200))
    combined.paste(human_img, (0, 0))
    combined.paste(dino_img,  (w + 10, 0))

    # Add labels
    draw_combined = ImageDraw.Draw(combined)
    draw_combined.text((5, 5),     "Human labels", fill="green")
    draw_combined.text((w + 15, 5), "DINO labels",  fill="red")

    out_path = out_dir / (img_path.stem + ".jpg")
    combined.save(out_path)

print(f"\nDone ✅ — saved to {out_dir}")
