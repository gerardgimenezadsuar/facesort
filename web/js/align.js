// ArcFace 5-landmark alignment: least-squares similarity transform
// (rotation + uniform scale + translation, via the complex-number
// formulation) mapping detected keypoints onto the 112x112 template,
// applied with a canvas transform.

export const TEMPLATE = [
  [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
  [41.5493, 92.3655], [70.7299, 92.2041],
];
export const ALIGN_SIZE = 112;

export function similarityTransform(src, dst) {
  const n = src.length;
  let mxS = 0, myS = 0, mxD = 0, myD = 0;
  for (let i = 0; i < n; i++) {
    mxS += src[i][0]; myS += src[i][1];
    mxD += dst[i][0]; myD += dst[i][1];
  }
  mxS /= n; myS /= n; mxD /= n; myD /= n;
  // c = Σ(conj(ŝ)·d̂) / Σ|ŝ|²  with points as complex numbers
  let re = 0, im = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - mxS, sy = src[i][1] - myS;
    const dx = dst[i][0] - mxD, dy = dst[i][1] - myD;
    re += sx * dx + sy * dy;
    im += sx * dy - sy * dx;
    den += sx * sx + sy * sy;
  }
  const a = re / den, b = im / den;
  return {
    a, b,
    tx: mxD - a * mxS + b * myS,
    ty: myD - b * mxS - a * myS,
  };
}

export function alignFace(bitmap, kps) {
  const t = similarityTransform(kps, TEMPLATE);
  const canvas = new OffscreenCanvas(ALIGN_SIZE, ALIGN_SIZE);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.setTransform(t.a, t.b, -t.b, t.a, t.tx, t.ty);
  ctx.drawImage(bitmap, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return canvas;
}

export function preprocessForEmbedding(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { data } = ctx.getImageData(0, 0, ALIGN_SIZE, ALIGN_SIZE);
  const n = ALIGN_SIZE * ALIGN_SIZE;
  const blob = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    blob[i] = (data[i * 4] - 127.5) / 127.5;
    blob[n + i] = (data[i * 4 + 1] - 127.5) / 127.5;
    blob[2 * n + i] = (data[i * 4 + 2] - 127.5) / 127.5;
  }
  return blob;
}

export function normalize(vec) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  s = Math.sqrt(s);
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / s;
  return out;
}
