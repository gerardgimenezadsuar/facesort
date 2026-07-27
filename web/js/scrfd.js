// SCRFD (det_500m) post-processing: anchor decoding + NMS.
// The model outputs, per stride (8/16/32), a score, bbox-distance and
// keypoint tensor over a 2-anchor grid; we classify outputs by shape.

const DET_SIZE = 640;
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;

export function preprocessForDetection(bitmap, size = DET_SIZE) {
  const scale = Math.min(size / bitmap.width, size / bitmap.height);
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(bitmap, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, size, size);
  const n = size * size;
  const blob = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    blob[i] = (data[i * 4] - 127.5) / 128;          // R
    blob[n + i] = (data[i * 4 + 1] - 127.5) / 128;  // G
    blob[2 * n + i] = (data[i * 4 + 2] - 127.5) / 128; // B
  }
  return { blob, scale };
}

export function decodeDetections(outputs, scale, size = DET_SIZE, scoreThresh = 0.45, iouThresh = 0.4) {
  // group tensors by anchor count (stride) and by role (last dim 1/4/10)
  const byStride = {};
  for (const t of outputs) {
    const dims = t.dims;
    const last = dims[dims.length - 1];
    const count = t.data.length / last;
    const stride = size / Math.sqrt(count / NUM_ANCHORS);
    const key = Math.round(stride);
    byStride[key] = byStride[key] || {};
    byStride[key][last === 1 ? "scores" : last === 4 ? "bbox" : "kps"] = t.data;
  }

  const faces = [];
  for (const stride of STRIDES) {
    const g = byStride[stride];
    if (!g || !g.scores) continue;
    const side = size / stride;
    for (let idx = 0; idx < g.scores.length; idx++) {
      const score = g.scores[idx];
      if (score < scoreThresh) continue;
      const cell = Math.floor(idx / NUM_ANCHORS);
      const cx = (cell % side) * stride;
      const cy = Math.floor(cell / side) * stride;
      const b = idx * 4;
      const box = [
        (cx - g.bbox[b] * stride) / scale,
        (cy - g.bbox[b + 1] * stride) / scale,
        (cx + g.bbox[b + 2] * stride) / scale,
        (cy + g.bbox[b + 3] * stride) / scale,
      ];
      const kps = [];
      for (let k = 0; k < 5; k++) {
        kps.push([
          (cx + g.kps[idx * 10 + k * 2] * stride) / scale,
          (cy + g.kps[idx * 10 + k * 2 + 1] * stride) / scale,
        ]);
      }
      faces.push({ box, kps, score });
    }
  }
  return nms(faces, iouThresh);
}

function nms(faces, iouThresh) {
  faces.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const f of faces) {
    if (kept.every((k) => iou(k.box, f.box) < iouThresh)) kept.push(f);
  }
  return kept;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}
