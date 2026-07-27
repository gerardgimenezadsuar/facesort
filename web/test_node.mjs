// Parity test: run the browser pipeline's math (scrfd decode, similarity
// alignment, clustering) in Node with onnxruntime-node against the demo set.
// Canvas ops are replaced with a plain bilinear sampler.

import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import ort from "onnxruntime-node";
import jpeg from "jpeg-js";
import { decodeDetections } from "./js/scrfd.js";
import { similarityTransform, TEMPLATE, normalize } from "./js/align.js";
import { buildDendrogram, cutDendrogram } from "./js/cluster.js";

const DET = 640, ALIGN = 112;

function bilinear(img, x, y, c) {
  // constant-black border, matching cv2.warpAffine and canvas drawImage
  if (x < 0 || y < 0 || x > img.width - 1 || y > img.height - 1) return 0;
  const x0 = Math.max(0, Math.min(img.width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(img.height - 1, Math.floor(y)));
  const x1 = Math.min(img.width - 1, x0 + 1), y1 = Math.min(img.height - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;
  const p = (xx, yy) => img.data[(yy * img.width + xx) * 4 + c];
  return (p(x0, y0) * (1 - fx) + p(x1, y0) * fx) * (1 - fy) +
         (p(x0, y1) * (1 - fx) + p(x1, y1) * fx) * fy;
}

function detBlob(img, size = DET) {
  const scale = Math.min(size / img.width, size / img.height);
  const blob = new Float32Array(3 * size * size);
  const n = size * size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const sx = x / scale, sy = y / scale;
      const inside = sx < img.width && sy < img.height;
      for (let c = 0; c < 3; c++) {
        const v = inside ? bilinear(img, sx, sy, c) : 0;
        blob[c * n + y * size + x] = (v - 127.5) / 128;
      }
    }
  }
  return { blob, scale };
}

function alignBlob(img, kps) {
  const t = similarityTransform(kps, TEMPLATE);
  // invert dst = [a -b; b a]src + t to sample source pixels
  const det = t.a * t.a + t.b * t.b;
  const blob = new Float32Array(3 * ALIGN * ALIGN);
  const n = ALIGN * ALIGN;
  for (let y = 0; y < ALIGN; y++) {
    for (let x = 0; x < ALIGN; x++) {
      const dx = x - t.tx, dy = y - t.ty;
      const sx = (t.a * dx + t.b * dy) / det;
      const sy = (-t.b * dx + t.a * dy) / det;
      for (let c = 0; c < 3; c++) {
        const v = bilinear(img, sx, sy, c);
        blob[c * n + y * ALIGN + x] = (v - 127.5) / 127.5;
      }
    }
  }
  return blob;
}

const detector = await ort.InferenceSession.create("models/det_500m.onnx");
const embedder = await ort.InferenceSession.create("models/w600k_mbf.onnx");

const files = readdirSync("demo").filter((f) => f.endsWith(".jpg")).sort();
const embeddings = [];
const trueIds = [];
let faceCount = 0;
for (const [i, f] of files.entries()) {
  const img = jpeg.decode(readFileSync(`demo/${f}`), { useTArray: true });
  let faces = [];
  for (const size of [640, 320]) {
    const { blob, scale } = detBlob(img, size);
    const out = await detector.run({
      [detector.inputNames[0]]: new ort.Tensor("float32", blob, [1, 3, size, size]),
    });
    faces = decodeDetections(detector.outputNames.map((o) => out[o]), scale, size);
    if (faces.length) break;
  }
  for (const face of faces) {
    const ablob = alignBlob(img, face.kps);
    const eout = await embedder.run({
      [embedder.inputNames[0]]: new ort.Tensor("float32", ablob, [1, 3, ALIGN, ALIGN]),
    });
    embeddings.push(normalize(eout[embedder.outputNames[0]].data));
    trueIds.push(Math.floor(parseInt(f.slice(4, 8)) / 15));
    faceCount++;
  }
  if ((i + 1) % 45 === 0) console.log(`${i + 1}/${files.length} photos, ${faceCount} faces`);
}

console.log(`Total: ${files.length} photos, ${faceCount} faces`);
writeFileSync("js_embeddings.json", JSON.stringify({ ids: trueIds, embs: embeddings.map((e) => [...e]) }));
const dendro = buildDendrogram(embeddings);
for (const thr of [0.6, 0.65, 0.7, 0.75, 0.8]) {
  const labels = cutDendrogram(dendro, thr);
  const counts = {};
  for (const l of labels) counts[l] = (counts[l] || 0) + 1;
  const people = Object.values(counts).filter((c) => c > 1).length;
  const singles = Object.values(counts).filter((c) => c <= 1).length;
  const byCluster = {};
  labels.forEach((l, i) => { (byCluster[l] = byCluster[l] || []).push(trueIds[i]); });
  let misfiled = 0;
  for (const ids of Object.values(byCluster)) {
    const maj = ids.sort((a,b)=>ids.filter(v=>v===a).length-ids.filter(v=>v===b).length).pop();
    misfiled += ids.filter((v) => v !== maj).length;
  }
  console.log(`threshold ${thr}: ${people} people (+${singles} singletons), misfiled ${misfiled}/${faceCount}  [true: 12]`);
}
