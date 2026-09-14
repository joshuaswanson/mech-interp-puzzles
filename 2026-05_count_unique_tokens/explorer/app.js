(function () {
  const MODEL = window.__MODEL__;
  const K = MODEL.constants;
  const HEAD_COLORS = ["var(--h0)", "var(--h1)", "var(--h2)", "var(--h3)"];
  const L1_SUBSETS = [["b"], ["f"], ["a", "d", "g", "h"], ["c", "e", "i", "j"]];
  const L0_TARGET_SYM = ["a", "h", "d", "g"];
  const L0_TARGET_ANS = ["i", "j", "c", "e"];

  const state = { letters: "abacabdace", layer: 1, head: 0, query: 11, matrixMode: "weights", walkHead: 0, out: null };
  const $ = sel => document.querySelector(sel);
  const fmt = (x, d = 2) => { const t = x.toFixed(d); return /^-0\.?0*$/.test(t) ? t.slice(1) : t; };
  const signed = x => (x < 0 ? "(−" + fmt(Math.abs(x)) + ")" : fmt(x));
  const tip = $("#tip");

  function recompute() {
    state.out = forward(MODEL, encode(state.letters));
    renderAll();
  }

  function renderAll() {
    renderEditor();
    renderMatrix();
    renderRankings();
    renderCounters();
    renderWalkthrough();
    renderReadout();
  }

  // ---------- sequence editor ----------
  function renderEditor() {
    const inp = $("#seq");
    if (inp.value !== state.letters && document.activeElement !== inp) inp.value = state.letters;
    const U = uniqueCount(state.letters);
    $("#true-count").textContent = U;
    const pv = $("#pred-count");
    pv.textContent = state.out.prediction;
    pv.className = "value " + (state.out.prediction === U ? "ok" : "bad");
  }

  function setLetters(s) {
    if (!/^[a-j]{10}$/.test(s)) return false;
    state.letters = s; recompute(); return true;
  }

  $("#seq").addEventListener("input", e => {
    const ok = setLetters(e.target.value.trim().toLowerCase());
    e.target.classList.toggle("invalid", !ok);
  });

  function segButtons(container, values, current, onPick, labelFn) {
    container.innerHTML = "";
    values.forEach(v => {
      const b = document.createElement("button");
      b.textContent = labelFn ? labelFn(v) : v;
      b.setAttribute("aria-pressed", String(v === current));
      b.addEventListener("click", () => onPick(v));
      container.appendChild(b);
    });
  }
  function goToAttention() {
    const el = $("#attention");
    if (el && typeof el.scrollIntoView === "function") el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------- layer-0 key rankings, from the score tables ----------
  const L0_RANKINGS = (() => {
    const out = [];
    for (let h = 0; h < 4; h++) {
      const head = MODEL.layers[0].heads[h];
      const S = [];
      for (let q = 0; q < 12; q++) {
        const qv = matvecLocal(head.WQ, MODEL.E[q]);
        S.push([...Array(12).keys()].map(k => dotLocal(qv, matvecLocal(head.WK, MODEL.E[k])) / Math.sqrt(8)));
      }
      const ordinary = [...Array(10).keys()].filter(q => q !== 1 && q !== 5);
      const avg = [...Array(10).keys()].map(k => ordinary.reduce((acc, q) => acc + S[q][k], 0) / ordinary.length);
      const rankSym = [...Array(10).keys()].sort((x, y) => avg[y] - avg[x]);
      const rankAns = [...Array(10).keys()].sort((x, y) => S[11][y] - S[11][x]);
      out.push({ sym: rankSym, ans: rankAns });
    }
    return out;
  })();
  function matvecLocal(M, x) { return M.map(row => row.reduce((acc, m, k) => acc + m * x[k], 0)); }
  function dotLocal(a, b) { return a.reduce((acc, v, k) => acc + v * b[k], 0); }
  function rankHtml(order, present) {
    const winner = order.find(k => present.has(SYMBOLS[k]));
    return order.map((k, idx) => {
      const ch = SYMBOLS[k];
      const cls = ["l", present.has(ch) ? "present" : "", idx === 0 ? "top" : "", k === winner ? "win" : ""].filter(Boolean).join(" ");
      return `<span class="${cls}" data-letter="${ch}" title="${ch}${present.has(ch) ? " is in the input" : " is not in the input"}${k === winner ? ", the head's pick" : ""}">${ch}</span>`;
    }).join('<span class="succ">≻</span>');
  }
  function highlightKey(letter, on) {
    $("#matrix").querySelectorAll(`[data-key="${letter}"]`).forEach(el => el.classList.toggle("hl", on));
  }
  function renderRankings() {
    const block = document.querySelector(".rankings-block");
    block.style.display = state.layer === 0 ? "" : "none";
    if (state.layer !== 0) return;
    const present = new Set(state.letters.split(""));
    let rows = "";
    for (let h = 0; h < 4; h++) {
      const sel = state.layer === 0 && state.head === h;
      rows += `<tr class="${sel ? "sel" : ""}" data-h="${h}"><td>L0H${h}</td><td class="rank">${rankHtml(L0_RANKINGS[h].sym, present)}</td><td class="rank">${rankHtml(L0_RANKINGS[h].ans, present)}</td></tr>`;
    }
    const tbl = $("#rankings");
    tbl.innerHTML = `<thead><tr><th>head</th><th>ordinary queries rank the keys</th><th>the ANS query ranks the keys</th></tr></thead><tbody>${rows}</tbody>`;
    tbl.querySelectorAll("tbody tr").forEach(tr => tr.addEventListener("click", () => { state.layer = 0; state.head = +tr.dataset.h; renderAll(); }));
    tbl.querySelectorAll(".l").forEach(el => {
      el.addEventListener("mouseenter", () => highlightKey(el.dataset.letter, true));
      el.addEventListener("mouseleave", () => highlightKey(el.dataset.letter, false));
    });
    $("#rankings-note").innerHTML = `Dark letters are in the current input, grey ones are not. The underlined letter is the first one in the ranking that is present, so it is where L0H${state.head} sends its attention once that letter has appeared (left ranking) and at ANS (right ranking). Hover a letter to highlight its key rows in the matrix above. Click a row to switch the matrix to that head. A b query reverses the ordering and attends mostly to b and BOS; an f query has almost no preference.`;
  }

  // ---------- full attention matrix for the current head (queries across, keys down) ----------
  function renderMatrix() {
    segButtons($("#matrix-layer-seg"), [0, 1], state.layer, v => { state.layer = v; renderAll(); }, v => "layer " + v);
    segButtons($("#matrix-head-seg"), [0, 1, 2, 3], state.head, v => { state.head = v; renderAll(); }, v => "head " + v);
    segButtons($("#matrix-seg"), ["weights", "scores"], state.matrixMode, v => { state.matrixMode = v; renderMatrix(); });
    const tokens = encode(state.letters), labels = tokens.map(tokenLabel);
    const head = state.out.layers[state.layer].heads[state.head];
    const useW = state.matrixMode === "weights";
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < 12; i++) for (let j = 0; j <= i; j++) {
      const v = useW ? head.weights[i][j] : head.scores[i][j];
      lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    if (useW) { lo = 0; hi = 1; }
    const span = hi - lo || 1;
    const digits = useW ? 2 : (Math.max(Math.abs(lo), Math.abs(hi)) >= 100 ? 0 : 1);
    let s = `<div></div><div></div><div class="axis-q">queries</div>`;
    s += `<div></div><div></div>`;
    for (let i = 0; i < 12; i++) s += `<div class="m-head${i === state.query ? " q" : ""}">${labels[i]}<small>${i}</small></div>`;
    s += `<div class="axis-k">keys</div>`;
    for (let j = 0; j < 12; j++) {
      s += `<div class="m-row" data-key="${labels[j]}">${labels[j]}<small>${j}</small></div>`;
      for (let i = 0; i < 12; i++) {
        if (j > i) { s += `<div class="cell masked" aria-hidden="true"></div>`; continue; }
        const v = useW ? head.weights[i][j] : head.scores[i][j];
        const pct = Math.round((v - lo) / span * 100);
        const tipText = `query ${i} (${labels[i]}) looking at key ${j} (${labels[j]}): weight ${fmt(head.weights[i][j])}, score ${fmt(head.scores[i][j], 1)}`;
        s += `<button class="cell${i === state.query ? " q" : ""}" data-i="${i}" data-key="${labels[j]}" data-tip="${tipText}" style="background: color-mix(in srgb, var(--ink) ${pct}%, var(--panel)); color: ${pct > 50 ? "var(--panel)" : "var(--ink)"}">${fmt(v, digits)}</button>`;
      }
    }
    const box = $("#matrix");
    box.innerHTML = s;
    box.querySelectorAll(".cell:not(.masked)").forEach(el => {
      el.addEventListener("click", () => { state.query = +el.dataset.i; renderAll(); });
      el.addEventListener("mousemove", e => showTip(e, el.dataset.tip));
      el.addEventListener("mouseleave", hideTip);
    });
    $("#matrix-summary").textContent = useW
      ? `Layer ${state.layer}, head ${state.head}: attention weights, 0 to 1. Each column sums to 1.`
      : `Layer ${state.layer}, head ${state.head}: raw scores, ${fmt(lo, digits)} to ${fmt(hi, digits)}.`;
  }

  // ---------- layer-1 counters ----------
  function renderCounters() {
    const tokens = encode(state.letters), labels = tokens.map(tokenLabel);
    let s = "", total = 0;
    const est = [];
    for (let h = 0; h < 4; h++) {
      const head = state.out.layers[1].heads[h];
      const w = head.weights[11];
      const slope = countSlope(MODEL, head.delta[11]);
      const n = (slope - K.intercept_slopes[h]) / K.unit;
      est.push(n); total += n;
      const trueN = L1_SUBSETS[h].filter(ch => state.letters.includes(ch)).length;
      const syms = L1_SUBSETS[h].map(ch => `<span class="${state.letters.includes(ch) ? "on" : ""}">${ch}</span>`).join("");
      const mini = labels.map((l, j) => `<div style="background: color-mix(in srgb, var(--ink) ${Math.round(w[j] * 100)}%, var(--panel)); color: ${w[j] > 0.5 ? "var(--panel)" : "var(--ink)"}" title="weight on position ${j} (${l}): ${fmt(w[j])}">${l.length > 1 ? l[0] : l}</div>`).join("");
      const top = w.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0])[0];
      s += `<div class="counter" data-h="${h}" role="button" tabindex="0" title="Click to see this head in the attention figure">
        <div class="title"><span class="dot c${h}"></span><span>L1H${h} counts</span></div>
        <div class="syms">${syms}</div>
        <div class="mini">${mini}</div>
        <div class="read">most weight on <b style="font-size:13px">${labels[top[1]]}</b> at position ${top[1]} (${fmt(top[0])})<br>
        output slope <span class="num">${fmt(slope, 1)}</span>, baseline <span class="num">${fmt(K.intercept_slopes[h], 1)}</span><br>
        (${fmt(slope, 1)} − ${fmt(K.intercept_slopes[h], 1)}) / ${fmt(K.unit, 1)} = <b>${fmt(n)}</b> &nbsp; true n<sub>${h}</sub> = <b>${trueN}</b></div>
      </div>`;
    }
    const box = $("#counters");
    box.innerHTML = s;
    box.querySelectorAll(".counter").forEach(el => {
      const go = () => { state.layer = 1; state.head = +el.dataset.h; state.query = 11; state.walkHead = +el.dataset.h; renderAll(); goToAttention(); };
      el.addEventListener("click", go);
      el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    });
    // stacked bar of estimates
    const U = uniqueCount(state.letters);
    const scale = 100 / 10.5;
    let segs = "";
    est.forEach((n, h) => {
      if (n <= 0.02) return;
      segs += `<div class="segm c${h}" style="width:${Math.max(0, n) * scale}%" title="L1H${h}: ${fmt(n)}">${n > 0.6 ? fmt(n) : ""}</div>`;
    });
    $("#sum-track").innerHTML = segs;
    $("#sum-total").innerHTML = est.map((n, h) => `<span style="color:${HEAD_COLORS[h]}">${signed(n)}</span>`).join(" + ") + ` = <b>${fmt(total)}</b>, closest whole number ${Math.round(total)}; the true count is ${U} and the model answers ${state.out.prediction}.`;
  }

  // ---------- layer-1 walkthrough ----------
  const WALK_TEXT = [
    "Looks for a <b>b</b>. Layer 0 treats a b query differently (its rankings flip), so the vector at any b position is unlike every other vector, and this head's query matches it. If there is no b the query falls back to BOS. The count lives in the <b>value</b>: pushed through this head's W<sub>OV</sub>, the b vector sits about one unit above the BOS vector on the count direction.",
    "Looks for an <b>f</b>. An f query has almost no ranking in layer 0, which again makes the f vector distinctive. Without an f the head settles on a g, d or h position, and all of those values sit at the same baseline. The count lives in the <b>value</b>.",
    "Looks for the position stamped with the most of <b>a, d, g, h</b>. The score is roughly 37·[a found] + 16·[d found] + 33·[g found] + 31·[h found], plus a large penalty on b and f tokens, so the winner is a position whose prefix already contains every a, d, g, h in the input. The count lives in the <b>value</b>: the same four stamps, pushed through W<sub>OV</sub>, each add about one unit.",
    "Does not look for a letter. Its query is built from the ANS vector, which carries the <b>c, e, i, j</b> stamps, and each stamp present lowers the score on the BOS key by about 1. Letter keys score about 1 regardless. So the weight on BOS drops one step per letter present. The count lives in the <b>attention weight</b>: the BOS value sits far down the count direction (slope about −132) and the letter values near 0, so less weight on BOS means a larger output.",
  ];

  function foundLetters(j) {
    const tokens = encode(state.letters), labels = tokens.map(tokenLabel);
    const out = [];
    for (let h = 0; h < 4; h++) {
      const w = state.out.layers[0].heads[h].weights[j];
      const byTok = {};
      for (let k = 0; k <= j; k++) byTok[labels[k]] = (byTok[labels[k]] || 0) + w[k];
      out.push(Object.entries(byTok).sort((a, b) => b[1] - a[1])[0][0]);
    }
    return out;
  }

  function renderWalkthrough() {
    segButtons($("#walk-seg"), [0, 1, 2, 3], state.walkHead, v => { state.walkHead = v; renderWalkthrough(); }, v => "L1H" + v);
    const h = state.walkHead, head = state.out.layers[1].heads[h];
    const tokens = encode(state.letters), labels = tokens.map(tokenLabel);
    const w = head.weights[11], sc = head.scores[11];
    const subset = L1_SUBSETS[h];
    $("#walk-text").innerHTML = WALK_TEXT[h];
    let rows = "", total = 0;
    for (let j = 0; j < 12; j++) {
      const stamps = foundLetters(j);
      const relevant = h === 2 ? subset : (h === 3 && j === 11 ? subset : []);
      const stampHtml = stamps.map(l => `<span class="${relevant.includes(l) ? "on" : ""}">${l.length > 1 ? l.slice(0, 1) : l}</span>`).join(" ");
      const tokOn = (h === 0 && labels[j] === "b") || (h === 1 && labels[j] === "f");
      const vs = countSlope(MODEL, head.ovValue[j]);
      const contrib = w[j] * vs;
      total += contrib;
      rows += `<tr class="${w[j] >= 0.05 ? "att" : ""}"><td>${j}</td><td class="tok${tokOn ? " on" : ""}">${labels[j]}</td><td class="stamps">${stampHtml}</td><td>${fmt(sc[j], 1)}</td><td>${fmt(w[j])}</td><td>${fmt(vs, 1)}</td><td class="contrib">${fmt(contrib, 1)}</td></tr>`;
    }
    const n = (total - K.intercept_slopes[h]) / K.unit;
    const trueN = subset.filter(ch => state.letters.includes(ch)).length;
    $("#walk-table").innerHTML = `<thead><tr><th>position</th><th>token</th><th>letters layer 0 found here (H0 H1 H2 H3)</th><th>score</th><th>weight</th><th>value: count slope of W<sub>OV</sub> × vector</th><th>weight × value</th></tr></thead><tbody>${rows}</tbody><tfoot><tr><td colspan="6">head output, count slope</td><td>${fmt(total, 1)}</td></tr><tr><td colspan="6">count = (${fmt(total, 1)} − baseline ${fmt(K.intercept_slopes[h], 1)}) / ${fmt(K.unit, 1)}</td><td>${fmt(n)}</td></tr></tfoot>`;
    $("#walk-note").innerHTML = `Rows with weight ≥ 0.05 are shaded. True count for this head's letters (${subset.join(", ")}): <b>${trueN}</b>. The "letters found" column is the detector grid, one column per row here; for BOS it is BOS itself, and for position 11 it uses the ANS rankings.`;
  }

  // ---------- readout chart ----------
  let showTable = false;
  function renderReadout() {
    const logits = state.out.logits, U = uniqueCount(state.letters), win = state.out.prediction - 1;
    const c = [...Array(10).keys()].map(k => k + 1);
    const fitRaw = c.map(cc => K.quad.lin * U * cc + K.quad.a * cc * cc + K.quad.b * cc);
    const shift = logits.reduce((a, b) => a + b, 0) / 10 - fitRaw.reduce((a, b) => a + b, 0) / 10;
    const fit = fitRaw.map(v => v + shift);
    const lo = Math.min(...logits, ...fit), hi = Math.max(...logits, ...fit);
    const pad = (hi - lo) * 0.08;
    const yMin = Math.min(0, lo - pad), yMax = hi + pad;
    const W = 760, H = 300, left = 64, right = 16, top = 16, bottom = 40;
    const pw = W - left - right, ph = H - top - bottom;
    const y = v => top + (yMax - v) / (yMax - yMin) * ph;
    const bw = pw / 10, barW = bw * 0.55;
    const xc = k => left + bw * (k + 0.5);
    let s = "";
    // gridlines: 5 ticks
    const ticks = niceTicks(yMin, yMax, 5);
    for (const t of ticks) {
      s += `<line class="grid-line" x1="${left}" x2="${W - right}" y1="${y(t)}" y2="${y(t)}"/>`;
      s += `<text x="${left - 8}" y="${y(t) + 4}" text-anchor="end">${t}</text>`;
    }
    s += `<line class="axis" x1="${left}" x2="${W - right}" y1="${y(0)}" y2="${y(0)}"/>`;
    for (let k = 0; k < 10; k++) {
      const v = logits[k], y0 = y(0), y1 = y(v);
      s += `<rect class="b${k === win ? " win" : ""}" x="${xc(k) - barW / 2}" y="${Math.min(y0, y1)}" width="${barW}" height="${Math.abs(y1 - y0)}" rx="3"/>`;
      s += `<text x="${xc(k)}" y="${H - bottom + 18}" text-anchor="middle" fill="${k === win ? "var(--ink)" : "var(--ink-2)"}">#${k + 1}</text>`;
    }
    // fit line
    s += `<path class="fit" d="${c.map((cc, k) => (k ? "L" : "M") + xc(k) + " " + y(fit[k])).join(" ")}"/>`;
    c.forEach((cc, k) => { s += `<circle class="fit-dot" cx="${xc(k)}" cy="${y(fit[k])}" r="3.5"/>`; });
    for (let k = 0; k < 10; k++) {
      s += `<rect class="hit" data-k="${k}" x="${left + bw * k}" y="${top}" width="${bw}" height="${ph}"><title>answer #${k + 1}: logit ${fmt(logits[k], 1)}, quadratic ${fmt(fit[k], 1)}</title></rect>`;
    }
    s += `<text x="${left - 8}" y="${top - 4}" text-anchor="end">logit</text>`;
    const svg = $("#readout-chart");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = s;
    svg.querySelectorAll(".hit").forEach(el => {
      el.addEventListener("mousemove", e => showTip(e, el.querySelector("title").textContent));
      el.addEventListener("mouseleave", hideTip);
    });
    $("#fit-label").innerHTML = `quadratic ${fmt(K.quad.lin, 1)}·${U}·c − ${fmt(-K.quad.a, 1)}·c² ${K.quad.b < 0 ? "−" : "+"} ${fmt(Math.abs(K.quad.b), 1)}·c + const`;
    $("#readout-summary").innerHTML = `The parabola peaks at c = (${fmt(K.quad.lin, 1)}·${U} ${K.quad.b < 0 ? "−" : "+"} ${fmt(Math.abs(K.quad.b), 1)}) / (2·${fmt(-K.quad.a, 1)}) = <span class="num">${fmt((K.quad.lin * U + K.quad.b) / (-2 * K.quad.a))}</span>. The tallest bar is <span class="num">#${win + 1}</span>.`;
    let rows = "";
    for (let k = 0; k < 10; k++) rows += `<tr><td>#${k + 1}</td><td>${fmt(logits[k], 1)}</td><td>${fmt(fit[k], 1)}</td></tr>`;
    $("#readout-table").innerHTML = `<thead><tr><th>answer</th><th>model logit</th><th>quadratic</th></tr></thead><tbody>${rows}</tbody>`;
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(st => span / st <= n + 1) || mag * 10;
    const out = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) out.push(Math.round(t));
    return out;
  }

  // ---------- tooltip ----------
  function showTip(e, text) {
    tip.textContent = text; tip.style.display = "block";
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px";
  }
  function hideTip() { tip.style.display = "none"; }

  // Optional URL state, e.g. ?seq=abacabdace&layer=0&head=3&query=6
  const qs = new URLSearchParams(location.search);
  if (/^[a-j]{10}$/.test(qs.get("seq") || "")) state.letters = qs.get("seq");
  if (["0", "1"].includes(qs.get("layer"))) state.layer = +qs.get("layer");
  if (["0", "1", "2", "3"].includes(qs.get("head"))) state.head = +qs.get("head");
  if (/^(\d|1[01])$/.test(qs.get("query") || "")) state.query = +qs.get("query");
  recompute();
})();
