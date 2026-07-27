// Average-linkage agglomerative clustering with cosine distance,
// nearest-neighbor-chain algorithm (O(n²) time, O(n²) memory).
// Returns labels for a given distance threshold; the dendrogram is
// computed once so the threshold can be re-cut instantly.

export function buildDendrogram(embeddings) {
  const n = embeddings.length;
  if (n === 0) return { n, merges: [] };
  if (n === 1) return { n, merges: [] };
  const dim = embeddings[0].length;

  // condensed full distance matrix (embeddings are L2-normalized)
  const D = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let dot = 0;
      const a = embeddings[i], b = embeddings[j];
      for (let k = 0; k < dim; k++) dot += a[k] * b[k];
      D[i * n + j] = D[j * n + i] = 1 - dot;
    }
  }

  const active = new Uint8Array(n).fill(1);
  const size = new Float64Array(n).fill(1);
  // member lists so we can emit flat labels later
  const members = Array.from({ length: n }, (_, i) => [i]);
  const merges = [];
  const chain = [];
  let remaining = n;

  while (remaining > 1) {
    if (chain.length === 0) {
      chain.push(active.indexOf(1));
    }
    const x = chain[chain.length - 1];
    // nearest active neighbor of x (prefer the previous chain element on ties)
    let best = -1, bestD = Infinity;
    const prev = chain.length > 1 ? chain[chain.length - 2] : -1;
    for (let j = 0; j < n; j++) {
      if (!active[j] || j === x) continue;
      const d = D[x * n + j];
      if (d < bestD || (d === bestD && j === prev)) { bestD = d; best = j; }
    }
    if (best === prev) {
      // reciprocal nearest neighbors -> merge prev into x
      chain.pop(); chain.pop();
      merges.push({ dist: bestD, a: x, b: prev });
      // Lance-Williams update for average linkage; cluster keeps index x
      const sx = size[x], sp = size[prev], st = sx + sp;
      for (let j = 0; j < n; j++) {
        if (!active[j] || j === x || j === prev) continue;
        const d = (sx * D[x * n + j] + sp * D[prev * n + j]) / st;
        D[x * n + j] = D[j * n + x] = d;
      }
      size[x] = st;
      members[x] = members[x].concat(members[prev]);
      active[prev] = 0;
      remaining--;
    } else {
      chain.push(best);
    }
  }
  return { n, merges, finalMembers: members };
}

export function cutDendrogram(dendrogram, threshold) {
  const { n, merges } = dendrogram;
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const sorted = [...merges].sort((a, b) => a.dist - b.dist);
  for (const m of sorted) {
    if (m.dist > threshold) break;
    parent[find(m.a)] = find(m.b);
  }
  const labels = new Int32Array(n);
  const map = new Map();
  for (let i = 0; i < n; i++) {
    const root = find(i);
    if (!map.has(root)) map.set(root, map.size);
    labels[i] = map.get(root);
  }
  return labels;
}
