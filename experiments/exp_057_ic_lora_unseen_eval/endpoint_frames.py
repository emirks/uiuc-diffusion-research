#!/usr/bin/env python
"""exp_057 — first|last frame pairs of the endpoint clips, gridded for captioning."""
import cv2
import numpy as np
from pathlib import Path
import sys

STD = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/data/processed/transitions_std121")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

CLIPS = [
    "hero_flight_2", "hero_flight_6", "super_fast_run_2", "super_fast_run_10",
    "plasma_explosion_0", "plasma_explosion_4", "shadow_10", "shadow_2",
    "fire_element_0", "fire_element_4", "wireframe_5", "wireframe_7",
    "illustration_scene_4", "illustration_scene_7", "animalization_0", "animalization_3",
    "gas_transformation_2", "gas_transformation_7", "portal_11", "portal_12",
    "giant_grab_1", "giant_grab_4", "money_rain_2", "hole_transition_0",
]
fam = {p.stem: d.name for d in STD.iterdir() if d.is_dir() for p in d.glob("*.mp4")}

TH = 300  # thumb height
rows = []
for i, stem in enumerate(CLIPS):
    cap = cv2.VideoCapture(str(STD / fam[stem] / (stem + ".mp4")))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pair = []
    for idx in (0, n - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        assert ok, (stem, idx)
        h, w = fr.shape[:2]
        pair.append(cv2.resize(fr, (int(w * TH / h), TH)))
    cap.release()
    strip = np.hstack([pair[0], np.full((TH, 6, 3), 255, np.uint8), pair[1]])
    label = np.zeros((24, strip.shape[1], 3), np.uint8)
    cv2.putText(label, stem, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    rows.append(np.vstack([label, strip]))

for g in range(0, len(rows), 6):
    img = np.vstack(rows[g:g + 6])
    p = OUT / f"endpoints_{g//6}.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(p)
