// Forward pass of the 2-layer attention-only transformer, plus the readout helpers.
// Pure functions, no DOM. Loaded both by the page and by the node test.

const D_HEAD = 8;
const N_HEADS = 4;
const BOS = 10, ANS = 11, COUNT_BASE = 12;
const SYMBOLS = "abcdefghij";

function matvec(M, x) {
  const out = new Array(M.length);
  for (let i = 0; i < M.length; i++) {
    const row = M[i];
    let s = 0;
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function softmaxRow(scores) {
  const m = Math.max(...scores);
  const ex = scores.map(s => Math.exp(s - m));
  const z = ex.reduce((a, b) => a + b, 0);
  return ex.map(e => e / z);
}

function encode(letters) {
  return [BOS, ...letters.split("").map(ch => SYMBOLS.indexOf(ch)), ANS];
}

function tokenLabel(t) {
  if (t < 10) return SYMBOLS[t];
  if (t === BOS) return "BOS";
  if (t === ANS) return "ANS";
  return "#" + (t - COUNT_BASE + 1);
}

// Runs one attention head on residuals r (array of 32-vectors) with a causal mask.
function runHead(head, WO, hIndex, r) {
  const n = r.length;
  const q = r.map(x => matvec(head.WQ, x));
  const k = r.map(x => matvec(head.WK, x));
  const v = r.map(x => matvec(head.WV, x));
  const scores = [], weights = [], z = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j <= i; j++) row.push(dot(q[i], k[j]) / Math.sqrt(D_HEAD));
    const w = softmaxRow(row);
    const zi = new Array(D_HEAD).fill(0);
    for (let j = 0; j <= i; j++) for (let d = 0; d < D_HEAD; d++) zi[d] += w[j] * v[j][d];
    scores.push(row); weights.push(w); z.push(zi);
  }
  // delta_i = WO[:, 8h:8h+8] @ z_i
  const delta = z.map(zi => {
    const out = new Array(WO.length).fill(0);
    for (let a = 0; a < WO.length; a++) {
      let s = 0;
      for (let d = 0; d < D_HEAD; d++) s += WO[a][hIndex * D_HEAD + d] * zi[d];
      out[a] = s;
    }
    return out;
  });
  // ovValue_j = WO[:, slice] @ WV @ r_j : what one attended position would contribute if fully attended
  const ovValue = v.map(vj => {
    const out = new Array(WO.length).fill(0);
    for (let a = 0; a < WO.length; a++) {
      let s = 0;
      for (let d = 0; d < D_HEAD; d++) s += WO[a][hIndex * D_HEAD + d] * vj[d];
      out[a] = s;
    }
    return out;
  });
  return { scores, weights, delta, ovValue, values: v, z };
}

function forward(model, tokens) {
  let r = tokens.map(t => model.E[t].slice());
  const layers = [];
  for (let L = 0; L < model.layers.length; L++) {
    const layer = model.layers[L];
    const heads = [];
    const sum = r.map(() => new Array(r[0].length).fill(0));
    for (let h = 0; h < N_HEADS; h++) {
      const res = runHead(layer.heads[h], layer.WO, h, r);
      heads.push(res);
      for (let i = 0; i < r.length; i++) for (let a = 0; a < sum[i].length; a++) sum[i][a] += res.delta[i][a];
    }
    layers.push({ residIn: r, heads });
    r = r.map((ri, i) => ri.map((x, a) => x + sum[i][a]));
  }
  const last = r[r.length - 1];
  const logits = [];
  for (let c = 0; c < 10; c++) logits.push(dot(model.U[COUNT_BASE + c], last));
  return { layers, residOut: r, logits, prediction: argmax(logits) + 1 };
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

// Count slope: slope of the logit vector (over answers 1..10) against the answer index.
function countSlope(model, vec) {
  let s = 0;
  for (let c = 0; c < 10; c++) s += (c + 1 - 5.5) * dot(model.U[COUNT_BASE + c], vec);
  return s / 82.5;
}

function uniqueCount(letters) {
  return new Set(letters.split("")).size;
}

if (typeof module !== "undefined") {
  module.exports = { forward, encode, tokenLabel, countSlope, uniqueCount, argmax, BOS, ANS, COUNT_BASE, SYMBOLS };
}
