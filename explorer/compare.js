  /* ---------- Compare varieties: two varieties by code or name, distance computed in the browser ---------- */
  const CMP = { data: null, meta: null, model: null, lists: new Map(), side: { a: null, b: null }, ran: false, inited: false, token: 0 };
  const CMP_MAX_PAIRS = 600;
  const CMP_LABEL = { pmi: "PMI-LDND", ldnd: "LDND" };
  const CMP_KIND = { glottocode: "Glottocode", "iso639-1": "ISO 639-1", "iso639-3": "ISO 639-3", macrolanguage: "ISO 639-3 macrolanguage", name: "name" };
  const cmpSel = (key) => $("cmp-" + key);

  function cmpLoad() {
    if (CMP.promise) return CMP.promise;
    $("cmp-status").textContent = "Loading the ASJP word lists (about 3 MB)…";
    CMP.promise = fetch("data/wordlists.json").then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }).then(d => {
      CMP.data = d;
      CMP.model = IsodivDistance.makeModel({ symbols: d.model.symbols, scores: d.model.scores, go: d.model.go, ge: d.model.ge });
      CMP.meta = d.docs.map(r => ({ id: r[0], name: r[1], glottocode: r[2], iso3: r[3], gname: r[4], family: r[5], packed: r[6] }));
      CMP.byId = new Map(CMP.meta.map(m => [m.id, m]));
      $("cmp-status").textContent = "";
      return d;
    }).catch(e => {
      CMP.promise = null;
      $("cmp-status").textContent = "";
      $("cmp-out").innerHTML = `<p class="verdict">The word lists could not be loaded (${esc(e.message)}). This tab needs <code>data/wordlists.json</code> next to the page: open the site on GitHub Pages, or serve the <code>docs</code> folder (for example <code>python -m http.server</code> inside it) instead of opening the file directly.</p>`;
      throw e;
    });
    return CMP.promise;
  }

  const cmpName = (m) => m.gname || m.name;
  function cmpGroups(ids) {
    const g = new Map();
    for (const id of ids) {
      const m = CMP.byId.get(id), key = m.glottocode || ("id:" + id);
      if (!g.has(key)) g.set(key, { key, glottocode: m.glottocode, name: cmpName(m), ids: [] });
      g.get(key).ids.push(id);
    }
    return [...g.values()];
  }
  const evenly = (arr, k) => k >= arr.length ? arr.slice() : Array.from({ length: k }, (_, i) => arr[Math.round(i * (arr.length - 1) / Math.max(k - 1, 1))]);

  function cmpResolve(key) {
    const q = cmpSel(key).value.trim();
    if (!q) { CMP.side[key] = null; cmpMatchView(key); return; }
    const m = IsodivDistance.resolve(q, CMP.meta, CMP.data.iso1, CMP.data.macro);
    const ids = m.ids.filter(id => CMP.byId.has(id));
    const other = key === "a" ? "b" : "a", before = CMP.side[other] ? CMP.side[other].sel.size : 0;
    CMP.side[key] = { q, m, ids, groups: cmpGroups(ids), sel: new Set(ids), edited: false, sampled: false };
    cmpAutoSelect();
    cmpMatchView(key);
    if (CMP.side[other] && CMP.side[other].sel.size !== before) cmpMatchView(other);     // thinning may have changed the other side
  }

  function cmpAutoSelect() {                      // keep nA × nB within the limit by thinning the larger untouched side
    const sides = ["a", "b"].map(k => CMP.side[k]).filter(Boolean);
    sides.forEach(s => { if (!s.edited) { s.sel = new Set(s.ids); s.sampled = false; } });
    const pairs = () => { const [a, b] = ["a", "b"].map(k => CMP.side[k] ? CMP.side[k].sel.size : 1); return a * b; };
    while (pairs() > CMP_MAX_PAIRS) {
      const open = sides.filter(s => !s.edited && s.sel.size > 1).sort((x, y) => y.sel.size - x.sel.size);
      if (!open.length) break;
      const s = open[0];
      s.sel = new Set(evenly(s.ids, Math.max(1, Math.floor(s.sel.size * 0.8)))); s.sampled = true;
    }
  }

  function cmpMatchView(key) {
    const host = $("cmp-match-" + key), s = CMP.side[key];
    if (!s) { host.innerHTML = ""; return; }
    const { m } = s;
    if (!s.ids.length) {
      const why = m.kind === "iso639-1" && !m.codes.length ? `“${esc(s.q)}” is not an ISO 639-1 code.`
        : `No ASJP word list matches “${esc(s.q)}”${m.codes.length ? ` (${esc(m.codes.join(", "))})` : ""}. ASJP has no list for every code; try the Glottocode or part of the name.`;
      host.innerHTML = `<span class="cmp-warn">${why}</span>`; return;
    }
    const via = m.kind === "macrolanguage" ? ` · ${m.codes.length} member codes` : (m.kind === "iso639-1" ? ` · ISO 639-3 ${esc(m.codes[0])}` : "");
    const n = s.sel.size;
    const groups = s.groups.slice(0, 300).map(g => {
      const on = g.ids.filter(id => s.sel.has(id)).length;
      return `<div class="cmp-grp"><label><input type="checkbox" data-g="${esc(g.key)}" ${on === g.ids.length ? "checked" : ""}> <span dir="auto">${esc(g.name)}</span> <span class="muted">${g.glottocode ? esc(g.glottocode) + " · " : ""}${on}/${g.ids.length}</span></label>
        <div class="cmp-docs">${g.ids.map(id => `<label><input type="checkbox" data-id="${esc(id)}" ${s.sel.has(id) ? "checked" : ""}> <span dir="auto">${esc(id)}</span>${CMP.data.excluded[id] ? ' <span class="muted" title="' + esc(CMP.data.excluded[id]) + '">(excluded from lineage results)</span>' : ""}</label>`).join("")}</div></div>`;
    }).join("");
    host.innerHTML = `<div><b>${s.ids.length}</b> word list${s.ids.length === 1 ? "" : "s"} · ${CMP_KIND[m.kind]}${via}${s.sampled ? ` · using a sample of ${n}` : ""}</div>
      <details><summary>${n} selected${s.groups.length > 1 ? ` from ${s.groups.length} languages` : ""}</summary><div class="cmp-list">${groups}${s.groups.length > 300 ? `<div class="muted">and ${s.groups.length - 300} more languages</div>` : ""}</div></details>`;
    host.querySelectorAll("input[data-id]").forEach(c => c.addEventListener("change", () => {
      c.checked ? s.sel.add(c.dataset.id) : s.sel.delete(c.dataset.id); s.edited = true; cmpPairNote(); cmpMatchCount(key);
    }));
    host.querySelectorAll("input[data-g]").forEach(c => c.addEventListener("change", () => {
      const g = s.groups.find(x => x.key === c.dataset.g);
      g.ids.forEach(id => c.checked ? s.sel.add(id) : s.sel.delete(id)); s.edited = true;
      host.querySelectorAll(`input[data-id]`).forEach(b => { if (g.ids.includes(b.dataset.id)) b.checked = c.checked; });
      cmpPairNote(); cmpMatchCount(key);
    }));
    cmpPairNote();
  }
  function cmpMatchCount(key) {
    const s = CMP.side[key], sum = $("cmp-match-" + key).querySelector("summary");
    if (s && sum) sum.textContent = `${s.sel.size} selected${s.groups.length > 1 ? ` from ${s.groups.length} languages` : ""}`;
  }
  function cmpPairs() {
    const a = CMP.side.a, b = CMP.side.b; if (!a || !b) return [];
    const seen = new Set(), out = [];
    for (const x of a.sel) for (const y of b.sel) {
      if (x === y) continue;
      const k = x < y ? x + "\u0000" + y : y + "\u0000" + x;
      if (!seen.has(k)) { seen.add(k); out.push([x, y]); }
    }
    return out;
  }
  function cmpPairNote() {
    const n = cmpPairs().length, go = $("cmp-go");
    go.disabled = n === 0 || n > CMP_MAX_PAIRS;
    $("cmp-status").textContent = !CMP.side.a || !CMP.side.b ? "" : n > CMP_MAX_PAIRS ? `${n} pairs is more than the limit of ${CMP_MAX_PAIRS}: deselect some word lists.`
      : n === 0 ? "Select at least one word list on each side (two different lists)." : `${n} word list pair${n === 1 ? "" : "s"} to score.`;
  }

  const cmpList = (id) => {
    if (!CMP.lists.has(id)) CMP.lists.set(id, IsodivDistance.parseList(CMP.byId.get(id).packed, CMP.data.concepts, CMP.model));
    return CMP.lists.get(id);
  };
  const median = (v) => { const s = v.slice().sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
  const sideLabel = (s) => s.groups.length === 1 ? s.groups[0].name : s.q;

  async function cmpRun() {
    if (!CMP.data) await cmpLoad();
    cmpResolveBoth();
    const pairs = cmpPairs(), kind = $("cmp-kind").value, token = ++CMP.token;
    if (!pairs.length || pairs.length > CMP_MAX_PAIRS) return;
    const res = [], out = $("cmp-out");
    out.classList.add("busy"); out.setAttribute("aria-busy", "true");          // old numbers stay visible but dimmed
    for (let i = 0; i < pairs.length; i++) {
      const [x, y] = pairs[i];
      const r = IsodivDistance.pairDistance(kind, CMP.model, cmpList(x), cmpList(y));
      res.push({ a: x, b: y, ...r });
      if (i % 25 === 24) {
        $("cmp-status").textContent = `Scoring ${i + 1} of ${pairs.length}…`;
        await new Promise(r => setTimeout(r)); if (token !== CMP.token) return;
      }
    }
    CMP.ran = true;
    out.classList.remove("busy"); out.removeAttribute("aria-busy");
    cmpPairNote();
    cmpShow(res, kind);
    try { const u = new URL(location.href); u.searchParams.set("a", CMP.side.a.q); u.searchParams.set("b", CMP.side.b.q); u.searchParams.set("d", kind); history.replaceState(null, "", u); } catch (e) { /* file:// */ }
  }
  function cmpResolveBoth() { /* selections were made when typing; re-resolve only if the text changed */
    ["a", "b"].forEach(k => { const q = cmpSel(k).value.trim(); if ((CMP.side[k] && CMP.side[k].q !== q) || (!CMP.side[k] && q)) cmpResolve(k); });
  }

  function cmpGauge(vals, med, kind) {
    const W = 640, H = 104, pad = 18, top = 14, base = 64, max = 1.1, X = v => pad + Math.min(v, max) / max * (W - 2 * pad);
    const [q10, q50, q90] = kind === "pmi" ? CMP.data.noise : CMP.data.noise_ldnd, un = CMP.data.unrelated[kind].mean;
    const un0 = kind === "pmi" ? CMP.data.unrelated.pmi.min : null, un1 = kind === "pmi" ? CMP.data.unrelated.pmi.max : null;
    let s = `<svg dir="ltr" role="img" aria-label="Distance on a scale from 0 to about 1, against the same-language range and the unrelated range" viewBox="0 0 ${W} ${H}" width="100%" style="max-inline-size:${W}px">`;
    s += `<rect x="${X(q10)}" y="${top}" width="${X(q90) - X(q10)}" height="${base - top}" fill="var(--sel)"/>`;
    s += `<line x1="${X(q50)}" x2="${X(q50)}" y1="${top}" y2="${base}" stroke="var(--accent)" stroke-dasharray="3 3"/>`;
    if (un0 != null) s += `<rect x="${X(un0)}" y="${top}" width="${X(un1) - X(un0)}" height="${base - top}" fill="var(--c-oth)" opacity=".18"/>`;
    s += `<line x1="${X(un)}" x2="${X(un)}" y1="${top}" y2="${base}" stroke="var(--c-oth)" stroke-width="2"/>`;
    s += `<line x1="${pad}" x2="${W - pad}" y1="${base}" y2="${base}" stroke="var(--rule)"/>`;
    [0, 0.25, 0.5, 0.75, 1].forEach(t => { s += `<text class="num" x="${X(t)}" y="${base + 14}" text-anchor="middle">${t}</text>`; });
    vals.forEach((v, i) => { const y = base - 8 - (vals.length > 1 ? (i * 7) % 28 : 12); s += `<circle cx="${X(v)}" cy="${y}" r="3.2" fill="var(--ink)" opacity="${vals.length > 1 ? .45 : 1}"/>`; });
    s += `<path d="M${X(med)} ${base - 2} l-7 -13 h14 z" fill="var(--accent)"/>`;
    s += `<text x="${X(q10)}" y="${H - 6}" text-anchor="start">same language: 10th–90th percentile ${fmt(q10, 2)}–${fmt(q90, 2)}</text>`;
    s += `<text x="${X(un)}" y="${H - 6}" text-anchor="end">unrelated lineages ≈ ${fmt(un, 2)}</text>`;
    return s + "</svg>";
  }

  function cmpVerdict(med, kind) {
    const [q10, q50, q90] = kind === "pmi" ? CMP.data.noise : CMP.data.noise_ldnd, un = CMP.data.unrelated[kind].mean, f = x => fmt(x, 2);
    if (med <= q50) return `At or below the typical distance between two transcriptions of the same language (${f(q50)}). On 40-item lists this cannot be told from one language written down twice.`;
    if (med <= q90) return `Inside the range of two lists of the same language (${f(q10)}–${f(q90)}, 10th–90th percentile), so the lists cannot separate this pair from transcription differences.`;
    if (med < 0.9 * un) return `Above the same-language range and well below unrelated lineages (about ${f(un)}): the vocabulary is related to some degree. This is an amount of lexical change, not a time, and no rate of change is assumed.`;
    return `In the range of unrelated lineages (they average ${f(un)}). Nothing in the basic vocabulary points to a relationship; a close pair here would be chance resemblance or contact.`;
  }

  function cmpShow(res, kind) {
    const ok = res.filter(r => Number.isFinite(r.value)), host = $("cmp-out"), L = CMP_LABEL[kind];
    const a = CMP.side.a, b = CMP.side.b;
    if (!ok.length) {
      host.innerHTML = `<p class="verdict">None of the ${res.length} pair${res.length === 1 ? "" : "s"} shares the 20 concepts a distance needs (the most shared is ${Math.max(...res.map(r => r.nShared))}). Some ASJP lists are short, so try another word list of the same language.</p>`;
      return;
    }
    const vals = ok.map(r => r.value), med = median(vals), lo = Math.min(...vals), hi = Math.max(...vals);
    const sorted = ok.slice().sort((x, y) => x.value - y.value);
    CMP.sorted = sorted; CMP.kind = kind;
    const pick = sorted.reduce((best, r) => Math.abs(r.value - med) < Math.abs(best.value - med) ? r : best, sorted[0]);
    host.innerHTML = `
      <div class="cmp-head">
        <div><div class="cmp-big num">${fmt(med)}</div><div class="muted">${L}</div></div>
        <div class="cmp-sub"><b dir="auto">${esc(sideLabel(a))}</b> ↔ <b dir="auto">${esc(sideLabel(b))}</b><br>
          <span class="muted">${ok.length === 1 ? "one pair of word lists" : `median of ${ok.length} word list pairs · range ${fmt(lo)}–${fmt(hi)}`}${ok.length < res.length ? ` · ${res.length - ok.length} pair${res.length - ok.length === 1 ? "" : "s"} left out (fewer than 20 shared concepts)` : ""}</span></div>
      </div>
      <div class="chart">${cmpGauge(vals, med, kind)}</div>
      <div class="verdict">${cmpVerdict(med, kind)}</div>
      <h3>Word list pairs</h3>
      <div class="scroll"><table class="cmp" id="cmp-pairs"><thead><tr><th>First list</th><th>Second list</th><th class="n">${L}</th><th class="n">Shared concepts</th></tr></thead><tbody></tbody></table></div>
      <div id="cmp-more"></div>
      <div id="cmp-detail"></div>`;
    cmpRows(60, pick);
    cmpDetail(pick);
  }
  function cmpRows(limit, active) {
    const body = $("cmp-pairs").querySelector("tbody"), rows = CMP.sorted;
    body.innerHTML = rows.slice(0, limit).map((r, i) => `<tr data-i="${i}" tabindex="0" ${r === active ? 'aria-selected="true"' : ""}><td dir="auto">${esc(r.a)}</td><td dir="auto">${esc(r.b)}</td><td class="n num">${fmt(r.value)}</td><td class="n num">${r.nShared}</td></tr>`).join("");
    body.querySelectorAll("tr").forEach(tr => {
      const go = () => { body.querySelectorAll("tr").forEach(x => x.removeAttribute("aria-selected")); tr.setAttribute("aria-selected", "true"); cmpDetail(rows[+tr.dataset.i]); };
      tr.addEventListener("click", go); tr.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    });
    $("cmp-more").innerHTML = rows.length > limit ? `<button type="button" class="chip" id="cmp-all">Show all ${rows.length} pairs</button>` : "";
    const all = $("cmp-all"); if (all) all.addEventListener("click", () => cmpRows(rows.length, active));
  }
  function cmpDetail(r) {
    const kind = CMP.kind, L = CMP_LABEL[kind], names = CMP.data.concept_names, ia = CMP.byId.get(r.a), ib = CMP.byId.get(r.b);
    const rows = r.rows.map(x => {
      const i = CMP.data.concepts.indexOf(x.concept);
      return `<tr><td>${esc(names[i])}</td><td class="asjp" dir="auto">${esc(x.a)}</td><td class="asjp" dir="auto">${esc(x.b)}</td><td class="n num">${fmt(x.d, 2)}</td><td><span class="cmp-bar" style="inline-size:${Math.min(100, x.d * 100)}%"></span></td></tr>`;
    }).join("");
    $("cmp-detail").innerHTML = `<h3>${esc(r.a)} ↔ ${esc(r.b)}</h3>
      <dl class="kv"><dt>${L}</dt><dd>${fmt(r.value)}</dd><dt>Same meaning, mean word distance</dt><dd>${fmt(r.same)}</dd>
        <dt>Different meaning, mean word distance (chance baseline)</dt><dd>${fmt(r.diff)}</dd><dt>Shared concepts</dt><dd>${r.nShared} of 40</dd>
        <dt>Languages</dt><dd dir="auto">${esc(cmpName(ia))} · ${esc(cmpName(ib))}</dd></dl>
      <p class="muted" style="margin:8px 0">${L} = (mean of the closest word-pair distance per concept) ÷ (mean distance between words of different concepts). Closest synonym pair shown.</p>
      <div class="scroll"><table class="cmp"><thead><tr><th>Concept</th><th>${esc(r.a)}</th><th>${esc(r.b)}</th><th class="n">Word distance</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function initCompare() {
    if (!CMP.inited) {
      CMP.inited = true;
      const timers = {};                          // one debounce per field, so editing one never drops the other's lookup
      const typed = (k) => { clearTimeout(timers[k]); timers[k] = setTimeout(() => { if (CMP.data) cmpResolve(k); }, 250); };
      ["a", "b"].forEach(k => { cmpSel(k).addEventListener("input", () => typed(k)); cmpSel(k).addEventListener("keydown", e => { if (e.key === "Enter") cmpRun(); }); });
      $("cmp-go").addEventListener("click", cmpRun);
      $("cmp-kind").addEventListener("change", () => { if (CMP.ran) cmpRun(); });
      $("cmp-swap").addEventListener("click", () => {
        const x = cmpSel("a").value; cmpSel("a").value = cmpSel("b").value; cmpSel("b").value = x;
        [CMP.side.a, CMP.side.b] = [CMP.side.b, CMP.side.a]; cmpMatchView("a"); cmpMatchView("b"); cmpPairNote(); if (CMP.ran) cmpRun();
      });
      const q = new URLSearchParams(location.search);
      if (q.get("a")) cmpSel("a").value = q.get("a");
      if (q.get("b")) cmpSel("b").value = q.get("b");
      if (q.get("d") === "ldnd") $("cmp-kind").value = "ldnd";
      cmpLoad().then(() => { ["a", "b"].forEach(k => cmpResolve(k)); if (cmpSel("a").value && cmpSel("b").value) cmpRun(); }).catch(() => {});
    }
  }
