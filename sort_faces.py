"""Face-sort engine harness.

Usage: python sort_faces.py <photo_folder> [--out output] [--threshold 0.7]

Scans a folder of photos, detects every face in every photo, embeds them,
clusters faces into per-person piles, and writes:
  - <out>/report.html   visual review page (face thumbs grouped by person)
  - <out>/piles.json    machine-readable mapping person -> photos
  - <out>/embeddings.npz  cache so re-runs skip detection/embedding
Everything runs locally; nothing leaves the machine.
"""

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MIN_FACE_PX = 40          # ignore tiny background faces
SINGLETON_MAX = 1         # clusters this small go to the "occasional" bucket
THUMB = 112


def scan_photos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)


def detect_and_embed(photos: list[Path], cache: Path):
    """Returns (embeddings, records) where records[i] = {photo, bbox}."""
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        cached_names = list(data["photo_names"])
        if cached_names == [str(p) for p in photos]:
            print(f"Using cached embeddings ({len(data['embeddings'])} faces)")
            return data["embeddings"], list(data["records"])

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    # low-res photos need a smaller detector input to score above threshold
    fallback = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    fallback.prepare(ctx_id=0, det_size=(320, 320))

    embeddings, records = [], []
    for i, path in enumerate(photos):
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skip (unreadable): {path.name}")
            continue
        faces = app.get(img) or fallback.get(img)
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
                continue
            embeddings.append(f.normed_embedding)
            records.append({"photo": str(path), "bbox": [int(x1), int(y1), int(x2), int(y2)]})
        if (i + 1) % 25 == 0 or i + 1 == len(photos):
            print(f"  {i + 1}/{len(photos)} photos, {len(embeddings)} faces")

    np.savez(cache, embeddings=np.array(embeddings),
             records=np.array(records, dtype=object),
             photo_names=np.array([str(p) for p in photos]))
    return np.array(embeddings), records


def cluster(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="cosine", linkage="average",
    ).fit_predict(embeddings)


def face_thumb_b64(record) -> str:
    img = cv2.imread(record["photo"])
    x1, y1, x2, y2 = record["bbox"]
    m = int(0.25 * max(x2 - x1, y2 - y1))          # margin around the crop
    h, w = img.shape[:2]
    crop = img[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop).resize((THUMB, THUMB))
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def write_report(labels, records, out: Path):
    order = sorted(np.unique(labels), key=lambda c: -np.sum(labels == c))
    people = [c for c in order if np.sum(labels == c) > SINGLETON_MAX]
    singles = [c for c in order if np.sum(labels == c) <= SINGLETON_MAX]

    piles, sections = {}, []
    for rank, c in enumerate(people, 1):
        idx = np.where(labels == c)[0]
        photos = sorted({records[i]["photo"] for i in idx})
        piles[f"person_{rank:02d}"] = photos
        thumbs = "".join(
            f'<img src="data:image/jpeg;base64,{face_thumb_b64(records[i])}" '
            f'title="{Path(records[i]["photo"]).name}">' for i in idx[:40]
        )
        more = f"<span class=more>+{len(idx) - 40} more faces</span>" if len(idx) > 40 else ""
        sections.append(
            f"<section><h2>Person {rank} — {len(idx)} faces in {len(photos)} photos</h2>"
            f"<div class=grid>{thumbs}{more}</div></section>"
        )

    if singles:
        thumbs = "".join(
            f'<img src="data:image/jpeg;base64,{face_thumb_b64(records[np.where(labels == c)[0][0]])}" '
            f'title="{Path(records[np.where(labels == c)[0][0]]["photo"]).name}">'
            for c in singles[:60]
        )
        sections.append(
            f"<section><h2>Occasional faces — {len(singles)} clusters with ≤{SINGLETON_MAX} face(s)</h2>"
            f"<div class=grid>{thumbs}</div></section>"
        )

    html = f"""<!doctype html><meta charset=utf-8><title>Face piles</title>
<style>
 body{{font:15px system-ui;margin:2rem auto;max-width:1100px;color:#222}}
 h2{{margin:1.6rem 0 .5rem;font-size:1.05rem}}
 .grid{{display:flex;flex-wrap:wrap;gap:4px}}
 .grid img{{width:{THUMB}px;height:{THUMB}px;border-radius:6px;object-fit:cover}}
 .more{{align-self:center;color:#888;padding:0 .6rem}}
</style>
<h1>Face piles — {len(people)} people, {len(records)} faces</h1>
{"".join(sections)}"""
    (out / "report.html").write_text(html)
    (out / "piles.json").write_text(json.dumps(piles, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    photos = scan_photos(args.folder)
    if not photos:
        sys.exit(f"No photos found in {args.folder}")
    print(f"{len(photos)} photos found")
    args.out.mkdir(exist_ok=True)

    embeddings, records = detect_and_embed(photos, args.out / "embeddings.npz")
    if len(embeddings) == 0:
        sys.exit("No faces detected")
    labels = cluster(embeddings, args.threshold)
    print(f"{len(embeddings)} faces -> {len(np.unique(labels))} raw clusters")

    write_report(labels, records, args.out)
    print(f"Wrote {args.out}/report.html and piles.json")


if __name__ == "__main__":
    main()
