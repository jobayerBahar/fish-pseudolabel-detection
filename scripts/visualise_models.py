from ultralytics import YOLO
from PIL import Image, ImageDraw
from pathlib import Path

human_model = YOLO("/home/jo/experiment2/runs/human_labeled_run2/weights/best.pt")
dino_model  = YOLO("/home/jo/experiment2/runs/dino_labeled_run2/weights/best.pt")
print("Models loaded ✅")

img_dir = Path("/home/jo/experiment2/fish_dataset/test/images")
out_dir = Path("/mnt/LinuxBackup/jo/visualisations")
out_dir.mkdir(parents=True, exist_ok=True)

image_paths = sorted([p for p in img_dir.glob("*")
                      if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
print(f"Processing {len(image_paths)} images...")

for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")
    w, h  = image.size

    # ── Left: human trained model ─────────────────────────────────────
    human_img  = image.copy()
    draw_human = ImageDraw.Draw(human_img)
    results_human = human_model(img_path, verbose=False)[0]
    for box in results_human.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        draw_human.rectangle([x1, y1, x2, y2], outline="green", width=3)

    # ── Right: dino trained model ──────────────────────────────────────
    dino_img  = image.copy()
    draw_dino = ImageDraw.Draw(dino_img)
    results_dino = dino_model(img_path, verbose=False)[0]
    for box in results_dino.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        draw_dino.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # ── Side by side ──────────────────────────────────────────────────
    combined = Image.new("RGB", (w * 2 + 10, h), color=(200, 200, 200))
    combined.paste(human_img, (0, 0))
    combined.paste(dino_img,  (w + 10, 0))

    draw_combined = ImageDraw.Draw(combined)
    draw_combined.text((5, 5),      "Human-trained model", fill="green")
    draw_combined.text((w + 15, 5), "DINO-trained model",  fill="red")

    out_path = out_dir / (img_path.stem + ".jpg")
    combined.save(out_path)

print(f"\nDone ✅ — saved to {out_dir}")
