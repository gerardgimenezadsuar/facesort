"""Spike: does detect -> embed -> cluster produce clean per-person piles?

Simulates a wedding photo set using LFW (few identities, many photos each),
hides the labels, runs InsightFace embeddings + agglomerative clustering,
and scores the recovered clusters against ground truth.
"""

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from insightface.app import FaceAnalysis

RNG = np.random.default_rng(0)
MAX_PER_PERSON = 25

print("Loading LFW subset (color)...")
lfw = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0)
names = lfw.target_names
print(f"{len(names)} identities available: {list(names[:5])}...")

# Cap photos per person so one prolific identity doesn't dominate
keep = []
for t in np.unique(lfw.target):
    idx = np.where(lfw.target == t)[0]
    keep.extend(RNG.permutation(idx)[:MAX_PER_PERSON])
keep = RNG.permutation(np.array(keep))
images = (lfw.images[keep] * 255).astype(np.uint8)  # (N, H, W, 3) RGB
labels_true = lfw.target[keep]
print(f"Test set: {len(images)} photos, {len(np.unique(labels_true))} true identities")

print("Loading InsightFace model (downloads on first run)...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))

embeddings, kept_labels, skipped = [], [], 0
for img, lab in zip(images, labels_true):
    faces = app.get(img[:, :, ::-1])  # RGB -> BGR
    if not faces:
        skipped += 1
        continue
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embeddings.append(f.normed_embedding)
    kept_labels.append(lab)
X = np.array(embeddings)
y = np.array(kept_labels)
print(f"Embedded {len(X)} faces ({skipped} photos with no detection)")

print("\nthreshold  clusters(true=%d)  ARI     purity" % len(np.unique(y)))
for thr in (0.45, 0.55, 0.65, 0.75, 0.85):
    cl = AgglomerativeClustering(
        n_clusters=None, distance_threshold=thr,
        metric="cosine", linkage="average",
    ).fit_predict(X)
    ari = adjusted_rand_score(y, cl)
    # purity: fraction of faces whose cluster's majority identity matches theirs
    purity = np.mean([
        np.max(np.bincount(y[cl == c])) for c in np.unique(cl)
        for _ in range(np.sum(cl == c))
    ] == y[np.argsort(cl)]) if False else sum(
        np.max(np.bincount(y[cl == c])) for c in np.unique(cl)
    ) / len(y)
    print(f"  {thr:.2f}     {len(np.unique(cl)):4d}            {ari:.3f}   {purity:.3f}")
