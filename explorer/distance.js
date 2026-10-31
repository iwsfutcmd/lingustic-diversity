/* Distance between two ASJP varieties in the browser: PMI-LDND (the pipeline's distance) and plain LDND.
   A port of src/isodiv/pmi.py (affine-gap alignment scored by learned sound correspondences) and distance.py (LDND):
   for two word lists with shared concepts C,
     same = mean over c in C of the closest synonym pair's word distance,
     diff = mean over c != c' of the word distance between the first words of c (list A) and c' (list B),
     distance = same / diff                                  (undefined below 20 shared concepts).
   Tested against the Python implementation in tests/test_site_js.py. No DOM access here. */
(function (root) {
  "use strict";
  const NEG = -1e9;

  function makeModel(m) {                       // m: {symbols, scores (V x V), go, ge}; one extra "unknown" symbol scores 0
    const V = m.symbols.length, V1 = V + 1, S = new Float64Array(V1 * V1);
    for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) S[i * V1 + j] = m.scores[i][j];
    const idx = new Map(m.symbols.map((s, i) => [s, i]));
    return { V, V1, S, idx, go: m.go, ge: m.ge, selfCache: new Map() };
  }

  function encode(model, word) {                // word: string of ASJP segments (one code point each)
    const out = []; for (const ch of word) out.push(model.idx.has(ch) ? model.idx.get(ch) : model.V);
    return out;
  }

  function gotoh(model, a, b) {                 // best global alignment score, affine gaps (same recurrence as pmi._gotoh)
    const n1 = a.length, n2 = b.length, W = n2 + 1, size = (n1 + 1) * W, { S, V1, go, ge } = model;
    const m = new Float64Array(size).fill(NEG), x = new Float64Array(size).fill(NEG), y = new Float64Array(size).fill(NEG);
    m[0] = 0;
    for (let i = 1; i <= n1; i++) x[i * W] = go + ge * (i - 1);
    for (let j = 1; j <= n2; j++) y[j] = go + ge * (j - 1);
    for (let i = 1; i <= n1; i++) {
      const ai = a[i - 1] * V1;
      for (let j = 1; j <= n2; j++) {
        const d = (i - 1) * W + j - 1, u = (i - 1) * W + j, l = i * W + j - 1, c = i * W + j;
        m[c] = S[ai + b[j - 1]] + Math.max(m[d], x[d], y[d]);
        x[c] = Math.max(m[u] + go, x[u] + ge, y[u] + go);
        y[c] = Math.max(m[l] + go, x[l] + go, y[l] + ge);
      }
    }
    const e = n1 * W + n2;
    return Math.max(m[e], x[e], y[e]);
  }

  function selfScore(model, word, enc) {
    let s = model.selfCache.get(word);
    if (s === undefined) { s = gotoh(model, enc, enc); model.selfCache.set(word, s); }
    return s;
  }

  function levenshtein(a, b) {
    let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
    for (let i = 1; i <= a.length; i++) {
      const cur = [i];
      for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = cur;
    }
    return prev[b.length];
  }

  function wordDistance(kind, model, wa, wb) {  // wa, wb: {s: string, e: encoded, c: code points}
    if (kind === "ldnd") return levenshtein(wa.c, wb.c) / Math.max(wa.c.length, wb.c.length);
    return 1 - 2 * gotoh(model, wa.e, wb.e) / (selfScore(model, wa.s, wa.e) + selfScore(model, wb.s, wb.e));
  }

  /* A word list from the data file: "w1/w2|w3||..." with one field per concept, in the order of `concepts`. */
  function parseList(packed, concepts, model) {
    const words = new Map();
    packed.split("|").forEach((field, i) => {
      if (!field) return;
      words.set(concepts[i], field.split("/").map((s) => ({ s, c: Array.from(s), e: model ? encode(model, s) : null })));
    });
    return words;
  }

  const conceptOrder = (a, b) => {
    const da = /^\d+$/.test(a), db = /^\d+$/.test(b);
    return da !== db ? (da ? -1 : 1) : da ? Number(a) - Number(b) : a < b ? -1 : a > b ? 1 : 0;
  };

  /* Distance between two parsed lists. Returns {value, nShared, same, diff, rows}; value is NaN when fewer than
     minShared concepts are shared (or the chance baseline is zero). rows: per shared concept, the closest synonym pair. */
  function pairDistance(kind, model, A, B, minShared) {
    minShared = minShared === undefined ? 20 : minShared;
    const shared = [...A.keys()].filter((c) => B.has(c)).sort(conceptOrder);
    const n = shared.length;
    if (n < minShared || n < 2) return { value: NaN, nShared: n, same: NaN, diff: NaN, rows: [] };
    let same = 0; const rows = [];
    for (const c of shared) {
      let best = Infinity, pa = null, pb = null;
      for (const wa of A.get(c)) for (const wb of B.get(c)) {
        const d = wordDistance(kind, model, wa, wb);
        if (d < best) { best = d; pa = wa.s; pb = wb.s; }
      }
      same += best; rows.push({ concept: c, a: pa, b: pb, d: best });
    }
    let diff = 0;
    for (const c1 of shared) for (const c2 of shared) if (c1 !== c2) diff += wordDistance(kind, model, A.get(c1)[0], B.get(c2)[0]);
    same /= n; diff /= n * (n - 1);
    return { value: diff > 0 ? same / diff : NaN, nShared: n, same, diff, rows };
  }

  /* Pick doculects by Glottocode, ISO 639-1 / -3 code, doculect ID or name: the port of isodiv.codes.resolve.
     docs: [{id, name, glottocode, iso3, gname}]; iso1: {"ja": "jpn"}; macro: {"zho": ["cmn", ...]}. */
  function resolve(query, docs, iso1, macro) {
    const q = String(query).trim(), low = q.toLowerCase();
    const mk = (kind, codes, ids) => ({ kind, query: q, codes, ids });
    if (!q) return mk("name", [], []);
    if (/^[a-z]{4}\d{4}$/.test(low)) return mk("glottocode", [], docs.filter((d) => d.glottocode === low).map((d) => d.id));
    if ((low.length === 2 || low.length === 3) && /^[a-z]+$/.test(low)) {
      let code = low, kind = "iso639-3";
      if (low.length === 2) {
        if (!(low in iso1)) return mk("iso639-1", [], []);
        code = iso1[low]; kind = "iso639-1";
      }
      const ids = docs.filter((d) => d.iso3 === code).map((d) => d.id);
      if (ids.length) return mk(kind, [code], ids);
      const members = macro[code] || [];
      if (members.length) return mk("macrolanguage", members, docs.filter((d) => members.includes(d.iso3)).map((d) => d.id));
      return mk(kind, [code], []);
    }
    const exact = docs.filter((d) => d.id.toLowerCase() === low).map((d) => d.id);
    if (exact.length) return mk("name", [], exact);
    return mk("name", [], docs.filter((d) => d.name.toLowerCase().includes(low) || d.gname.toLowerCase().includes(low)).map((d) => d.id));
  }

  const api = { makeModel, encode, gotoh, levenshtein, wordDistance, parseList, pairDistance, resolve, conceptOrder };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.IsodivDistance = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
