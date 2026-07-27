import { preprocessForDetection, decodeDetections } from "./scrfd.js";
import { alignFace, preprocessForEmbedding, normalize, ALIGN_SIZE } from "./align.js";
import { buildDendrogram, cutDendrogram } from "./cluster.js";
import { zipStore } from "./zip.js";

const $ = (id) => document.getElementById(id);
const MIN_FACE_PX = 40;
const SINGLETON_MAX = 1;
const state = {
  faces: [],              // { photo, embedding, thumb }
  fileGetters: new Map(), // photo name -> () => Promise<Blob>
  dendrogram: null,
  names: new Map(),       // pile key (min face idx) -> user-given name
  manualMerges: [],       // [faceIdxA, faceIdxB] pairs to union after cutting
  selected: new Set(),    // pile keys ticked for merging
  people: [],             // last rendered piles (for toolbar actions)
  photoCount: 0,
};

ort.env.wasm.wasmPaths = new URL("lib/", location.href).href;
ort.env.wasm.numThreads = 1;

let detector, embedder;
async function loadModels() {
  const opts = { executionProviders: ["webgpu", "wasm"] };
  try {
    detector = await ort.InferenceSession.create("models/det_500m.onnx", opts);
    embedder = await ort.InferenceSession.create("models/w600k_mbf.onnx", opts);
  } catch {
    detector = await ort.InferenceSession.create("models/det_500m.onnx", { executionProviders: ["wasm"] });
    embedder = await ort.InferenceSession.create("models/w600k_mbf.onnx", { executionProviders: ["wasm"] });
  }
}
const modelsReady = loadModels();

function show(view) {
  $("dropzone").hidden = view !== "landing";
  $("progress").hidden = view !== "progress";
  $("toolbar").hidden = view !== "results";
}

async function detectFaces(bitmap, size = 640) {
  const { blob, scale } = preprocessForDetection(bitmap, size);
  const input = new ort.Tensor("float32", blob, [1, 3, size, size]);
  const out = await detector.run({ [detector.inputNames[0]]: input });
  const faces = decodeDetections(detector.outputNames.map((n) => out[n]), scale, size);
  // low-res photos need a smaller detector input to score above threshold
  if (!faces.length && size === 640) return detectFaces(bitmap, 320);
  return faces;
}

async function embedFace(bitmap, kps) {
  const aligned = alignFace(bitmap, kps);
  const blob = preprocessForEmbedding(aligned);
  const input = new ort.Tensor("float32", blob, [1, 3, ALIGN_SIZE, ALIGN_SIZE]);
  const out = await embedder.run({ [embedder.inputNames[0]]: input });
  return normalize(out[embedder.outputNames[0]].data);
}

function faceThumb(bitmap, box) {
  const [x1, y1, x2, y2] = box;
  const m = 0.25 * Math.max(x2 - x1, y2 - y1);
  const sx = Math.max(0, x1 - m), sy = Math.max(0, y1 - m);
  const sw = Math.min(bitmap.width, x2 + m) - sx;
  const sh = Math.min(bitmap.height, y2 + m) - sy;
  const c = new OffscreenCanvas(112, 112);
  c.getContext("2d").drawImage(bitmap, sx, sy, sw, sh, 0, 0, 112, 112);
  return c.convertToBlob({ type: "image/jpeg", quality: 0.8 });
}

async function processOne(name, blobOrFile) {
  const bitmap = await createImageBitmap(blobOrFile);
  const detections = await detectFaces(bitmap);
  for (const f of detections) {
    const [x1, y1, x2, y2] = f.box;
    if (Math.min(x2 - x1, y2 - y1) < MIN_FACE_PX * (Math.max(bitmap.width, bitmap.height) / 3000)) continue;
    const embedding = await embedFace(bitmap, f.kps);
    const thumb = URL.createObjectURL(await faceThumb(bitmap, f.box));
    state.faces.push({ photo: name, embedding, thumb });
  }
  bitmap.close();
}

async function runPipeline(entries) {
  show("progress");
  $("status").textContent = "Loading models…";
  await modelsReady;
  state.faces = [];
  state.names.clear();
  state.manualMerges = [];
  state.selected.clear();
  state.fileGetters = new Map(entries.map((e) => [e.name, e.get]));
  state.photoCount = entries.length;
  $("results").innerHTML = "";
  const t0 = performance.now();
  let done = 0;
  for (const { name, get } of entries) {
    try {
      await processOne(name, await get());
    } catch (e) {
      console.warn("skip", name, e);
    }
    done++;
    $("status").textContent = `Reading faces… ${done} of ${entries.length} photos (${state.faces.length} faces so far)`;
    $("bar").style.width = `${(100 * done) / entries.length}%`;
  }
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  $("status").textContent = `Grouping ${state.faces.length} faces…`;
  await new Promise((r) => setTimeout(r));
  state.dendrogram = buildDendrogram(state.faces.map((f) => f.embedding));
  show("results");
  render();
  console.log(`FACESORT_RESULT photos=${entries.length} faces=${state.faces.length} people=${state.people.length} secs=${secs}`);
}

function computePiles() {
  const threshold = parseFloat($("threshold").value);
  const labels = cutDendrogram(state.dendrogram, threshold);
  // apply the user's manual merges on top of the automatic cut
  const parent = new Map();
  const find = (l) => {
    while (parent.has(l) && parent.get(l) !== l) l = parent.get(l);
    return l;
  };
  for (const [a, b] of state.manualMerges) {
    const ra = find(labels[a]), rb = find(labels[b]);
    if (ra !== rb) parent.set(ra, rb);
  }
  const groups = new Map();
  labels.forEach((l, i) => {
    const root = find(l);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(i);
  });
  return [...groups.values()]
    .map((g) => ({
      key: Math.min(...g),
      faceIdxs: g,
      photos: [...new Set(g.map((i) => state.faces[i].photo))].sort(),
    }))
    .sort((a, b) => b.faceIdxs.length - a.faceIdxs.length);
}

const safeName = (s) => s.replace(/[\\/:*?"<>|]/g, "_");
function pileName(pile, rank) {
  return state.names.get(pile.key) || `Person ${rank + 1}`;
}

async function downloadZip(piles, filename, withFolders) {
  const btns = document.querySelectorAll("button");
  btns.forEach((b) => (b.disabled = true));
  const stats = $("stats");
  const prev = stats.textContent;
  try {
    const entries = [];
    const taken = new Set();
    let n = 0;
    const total = piles.reduce((s, p) => s + p.photos.length, 0);
    for (const pile of piles) {
      for (const photo of pile.photos) {
        const blob = await state.fileGetters.get(photo)();
        const base = photo.split("/").pop();
        let path = withFolders ? `${pile.zipFolder}/${base}` : base;
        if (taken.has(path)) path = path.replace(/(\.[^.]*)?$/, `_${n}$1`);
        taken.add(path);
        entries.push({ path, data: new Uint8Array(await blob.arrayBuffer()) });
        n++;
        stats.textContent = `Packing ${n}/${total}…`;
      }
    }
    const url = URL.createObjectURL(zipStore(entries));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } finally {
    stats.textContent = prev;
    btns.forEach((b) => (b.disabled = false));
  }
}

function mergeSelected() {
  const keys = [...state.selected];
  if (keys.length < 2) return;
  for (let i = 1; i < keys.length; i++) state.manualMerges.push([keys[0], keys[i]]);
  // keep the name of the first named pile among the merged ones
  const name = keys.map((k) => state.names.get(k)).find(Boolean);
  const newKey = Math.min(...keys);
  if (name) state.names.set(newKey, name);
  state.selected.clear();
  render();
}

function pileCard(pile, rank) {
  const sec = document.createElement("section");
  sec.className = "person";

  const head = document.createElement("div");
  head.className = "pile-head";

  const check = document.createElement("input");
  check.type = "checkbox";
  check.checked = state.selected.has(pile.key);
  check.title = "Select to merge with another pile";
  check.onchange = () => (check.checked ? state.selected.add(pile.key) : state.selected.delete(pile.key));

  const avatar = document.createElement("img");
  avatar.className = "avatar";
  avatar.src = state.faces[pile.faceIdxs[0]].thumb;
  avatar.alt = "";

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "name";
  nameInput.value = pileName(pile, rank);
  nameInput.title = "Click to rename";
  nameInput.onchange = () => {
    state.names.set(pile.key, nameInput.value.trim() || `Person ${rank + 1}`);
  };

  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = `${pile.photos.length} photos`;

  const spacer = document.createElement("span");
  spacer.className = "spacer";

  const dl = document.createElement("button");
  dl.className = "btn ghost small";
  dl.textContent = "Download ZIP";
  dl.onclick = () => downloadZip([pile], `${safeName(pileName(pile, rank))}.zip`, false);

  head.append(check, avatar, nameInput, meta, spacer, dl);
  sec.appendChild(head);

  const grid = document.createElement("div");
  grid.className = "grid";
  pile.faceIdxs.slice(0, 40).forEach((i) => {
    const img = document.createElement("img");
    img.src = state.faces[i].thumb;
    img.title = state.faces[i].photo;
    grid.appendChild(img);
  });
  if (pile.faceIdxs.length > 40)
    grid.insertAdjacentHTML("beforeend", `<span class="more">+${pile.faceIdxs.length - 40}</span>`);
  sec.appendChild(grid);
  return sec;
}

function render() {
  const threshold = parseFloat($("threshold").value);
  $("thval").textContent = threshold.toFixed(2);
  if (!state.dendrogram) return;
  const piles = computePiles();
  const people = piles.filter((p) => p.faceIdxs.length > SINGLETON_MAX);
  const singles = piles.filter((p) => p.faceIdxs.length <= SINGLETON_MAX);
  state.people = people;

  $("stats").innerHTML =
    `${people.length} people <small>${state.photoCount} photos · ${state.faces.length} faces</small>`;

  const el = $("results");
  el.innerHTML = "";
  people.forEach((pile, rank) => el.appendChild(pileCard(pile, rank)));

  if (singles.length) {
    const det = document.createElement("details");
    det.className = "occasional";
    const sum = document.createElement("summary");
    sum.textContent = `Occasional faces (${singles.length})`;
    det.appendChild(sum);
    const grid = document.createElement("div");
    grid.className = "grid dim";
    singles.slice(0, 60).forEach((p) => {
      const img = document.createElement("img");
      img.src = state.faces[p.faceIdxs[0]].thumb;
      img.title = state.faces[p.faceIdxs[0]].photo;
      grid.appendChild(img);
    });
    det.appendChild(grid);
    el.appendChild(det);
  }
}

$("threshold").addEventListener("input", () => {
  state.selected.clear();
  render();
});
$("mergeBtn").addEventListener("click", mergeSelected);
$("downloadAllBtn").addEventListener("click", () => {
  state.people.forEach((p, r) => (p.zipFolder = safeName(pileName(p, r))));
  downloadZip(state.people, "photos_by_person.zip", true);
});

const IMG_RE = /\.(jpe?g|png|webp|bmp)$/i;
$("dropzone").addEventListener("click", () => $("picker").click());
$("picker").addEventListener("change", (e) => {
  const files = [...e.target.files].filter((f) => IMG_RE.test(f.name));
  if (!files.length) return;
  runPipeline(files.map((f) => ({ name: f.webkitRelativePath || f.name, get: async () => f })));
});

// drag & drop a folder (or a selection of photos)
async function filesFromDataTransfer(dt) {
  const out = [];
  async function walk(entry, prefix) {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      if (IMG_RE.test(file.name)) out.push({ name: prefix + file.name, get: async () => file });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const e of batch) await walk(e, prefix + entry.name + "/");
      } while (batch.length);
    }
  }
  for (const item of dt.items) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) await walk(entry, "");
  }
  return out;
}
["dragover", "dragenter"].forEach((ev) =>
  $("dropzone").addEventListener(ev, (e) => {
    e.preventDefault();
    $("dropzone").classList.add("drag");
  })
);
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("drag"));
$("dropzone").addEventListener("drop", async (e) => {
  e.preventDefault();
  $("dropzone").classList.remove("drag");
  const entries = await filesFromDataTransfer(e.dataTransfer);
  if (entries.length) runPipeline(entries);
});

// automated test hook: ?demo loads images listed in demo/manifest.json
if (new URLSearchParams(location.search).has("demo")) {
  fetch("demo/manifest.json")
    .then((r) => r.json())
    .then((names) =>
      runPipeline(names.map((n) => ({ name: n, get: () => fetch(`demo/${n}`).then((r) => r.blob()) })))
    );
}
