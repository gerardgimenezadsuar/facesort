"""Write a fake 'event folder' of JPEGs from LFW to exercise the harness."""
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

RNG = np.random.default_rng(1)
OUT = Path("demo_event")
OUT.mkdir(exist_ok=True)

lfw = fetch_lfw_people(min_faces_per_person=25, color=True, resize=1.0)
ids = RNG.permutation(np.unique(lfw.target))[:12]   # 12 "guests"
n = 0
for t in ids:
    idx = RNG.permutation(np.where(lfw.target == t)[0])[:15]
    for i in idx:
        img = (lfw.images[i] * 255).astype(np.uint8)
        Image.fromarray(img).save(OUT / f"IMG_{n:04d}.jpg", quality=90)
        n += 1
print(f"Wrote {n} photos of {len(ids)} people to {OUT}/ (shuffled filenames)")
