"""Animated pipeline for the website: image -> Grounding DINO boxes -> threshold -> training labels -> detector.

Uses the walkthrough images made by make_walkthrough.py. Writes:
  pipeline.gif  (for Google Sites: upload as an image, it animates)
  pipeline.mp4  (if you prefer a video, e.g. via Drive or YouTube)

  python make_pipeline_gif.py
"""
import cv2
import numpy as np
from PIL import Image

OUT = "/mnt/LinuxBackup/jo/results"
W = 960                       # output width (keeps the GIF small)
HOLD, FADE, FPS = 2.6, 0.6, 12  # seconds per step, crossfade seconds, frames per second
STEPS = [
    ("walk_input.jpg",        "1. Underwater image + the prompt \"fish\""),
    ("walk_gd_all.jpg",       "2. Grounding DINO draws a box around anything like a fish"),
    ("walk_gd_thr.jpg",       "3. Keep boxes scoring 0.3 or more (red), drop the rest (grey)"),
    ("walk_train_labels.jpg", "4. Kept boxes become training labels - no human checks them"),
    ("walk_detector.jpg",     "5. The trained detector finds fish on its own"),
    ("walk_detector.jpg",     "Not perfect: it misses some fish, and its boxes differ from a human's"),
]
STRIP = [0, 1, 2, 3, 4, 4]     # which step-bar box is lit for each frame above
SHORT = ["Image", "Grounding DINO", "Threshold 0.3", "Train", "Detect"]
INK, MUTED, BG = (34, 34, 34), (150, 150, 150), (255, 255, 255)
RED = (40, 39, 214)            # BGR, same red as the slides (#D62728)
TOP, BOT = 64, 58              # caption bar and step strip heights
FONT = cv2.FONT_HERSHEY_SIMPLEX


def frame(img, step):
    h = int(img.shape[0] * W / img.shape[1])
    im = cv2.resize(img, (W, h))
    c = np.full((TOP + h + BOT, W, 3), 255, np.uint8)
    c[TOP:TOP + h] = im
    # caption
    cv2.putText(c, STEPS[step][1], (18, 42), FONT, 0.72, INK, 2, cv2.LINE_AA)
    # step strip
    n, pad = len(SHORT), 10
    bw = (W - pad * (n + 1)) // n
    y0 = TOP + h + 10
    for i, s in enumerate(SHORT):
        x0 = pad + i * (bw + pad)
        on = i == STRIP[step]
        cv2.rectangle(c, (x0, y0), (x0 + bw, y0 + 36), RED if on else (235, 235, 235), -1)
        (tw, th), _ = cv2.getTextSize(s, FONT, 0.55, 1)
        cv2.putText(c, s, (x0 + (bw - tw) // 2, y0 + 24), FONT, 0.55,
                    (255, 255, 255) if on else MUTED, 2 if on else 1, cv2.LINE_AA)
    return c


frames_bgr = [frame(cv2.imread(f"{OUT}/{f}"), i) for i, (f, _) in enumerate(STEPS)]
hh = min(f.shape[0] for f in frames_bgr)
frames_bgr = [f[:hh] for f in frames_bgr]

seq = []
for i, f in enumerate(frames_bgr):
    seq += [f] * int(HOLD * FPS)
    nxt = frames_bgr[(i + 1) % len(frames_bgr)]
    for k in range(1, int(FADE * FPS) + 1):
        a = k / (FADE * FPS)
        m = cv2.addWeighted(f, 1 - a, nxt, a, 0)
        src = f if a < 0.5 else nxt              # captions switch at once, only the photo fades
        m[:TOP], m[hh - BOT:] = src[:TOP], src[hh - BOT:]
        seq.append(m)

vw = cv2.VideoWriter(f"{OUT}/pipeline.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, hh))
for f in seq:
    vw.write(f)
vw.release()

pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=192) for f in seq]
pil[0].save(f"{OUT}/pipeline.gif", save_all=True, append_images=pil[1:],
            duration=int(1000 / FPS), loop=0, optimize=True)
print(f"saved pipeline.gif and pipeline.mp4 ({len(seq)} frames, {len(seq) / FPS:.0f} s loop)")
