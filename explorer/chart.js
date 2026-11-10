  /* ---------- Chart maker: grouped bar charts of D_max or depth, drawn for sharing ---------- */
  const CH_COLORS = [["#eb6834", "#f6b79b"], ["#2a78d6", "#9cc2ee"], ["#8a8984", "#cfcdc8"]];
  const CH_INK = "#0b0b0b", CH_INK2 = "#52514e", CH_MUTED = "#8a8984", CH_GRID = "#e6e5e1", CH_SURF = "#fcfcfb";
  const CH_DEFAULT_TITLE = { depth: "How far apart are the most distant varieties?", dmax: "How many languages' worth of basic vocabulary?" };
  const CH_DEFAULT_SUB = {
    depth: "Depth: share of basic vocabulary no longer shared across each group's deepest split (0% = one language).\nCalibrated on expert cognate judgements; above about 79%, 40 words can't tell.",
    dmax: "Effective number of lineages the 40-word basic vocabulary lists can't tie together. Adding more dialects doesn't raise it;\nn unrelated languages score about n.",
  };
  const CH_PRESET = [
    { name: "Usually called one language (or its \"dialects\")", items: [["Sinitic", "Chinese (all of Sinitic)"], ["Chinese without Min", "Chinese without Min"], ["Min", "Min"], ["Wu", "Wu"], ["Mandarin", "Mandarin"], ["Yue", "Yue (Cantonese etc.)"], ["Arabic", "Arabic"], ["German dialects", "German dialects"], ["Quechuan", "Quechua"], ["English", "English (incl. Scots)"], ["Basque", "Basque"]] },
    { name: "Families of recognised separate languages", items: [["Indo-European", "Indo-European"], ["Turkic", "Turkic"], ["Romance", "Romance"], ["Germanic", "Germanic"], ["Japonic", "Japonic"], ["Slavic", "Slavic"], ["Koreanic", "Korean + Jeju"]] },
    { name: "Spoilers: random mixes of unrelated languages", items: [["Random unrelated, n=5", "5 random languages"], ["Random unrelated, n=80", "80 random languages"]] },
  ];
  const CH = { inited: false, measure: "depth", sort: "value", title: "", sub: "", groups: [] };
  const CH_KEY = "isodiv-chart-v2";            // v2: depth = 100% − shared

  function chSave() { try { localStorage.setItem(CH_KEY, JSON.stringify({ measure: CH.measure, sort: CH.sort, title: CH.title, sub: CH.sub, groups: CH.groups })); } catch (e) { /* storage unavailable */ } }
  function chLoadState() {
    try {
      const s = JSON.parse(localStorage.getItem(CH_KEY) || "null");
      if (s && Array.isArray(s.groups) && s.groups.length === 3) { Object.assign(CH, s); return true; }
    } catch (e) { /* storage unavailable */ }
    return false;
  }
  function chPreset() {
    CH.groups = CH_PRESET.map(g => ({ name: g.name, items: g.items.filter(([n]) => byName.has(n)).map(([n, l]) => ({ name: n, label: l })) }));
    CH.title = CH_DEFAULT_TITLE[CH.measure]; CH.sub = CH_DEFAULT_SUB[CH.measure];
  }

  /* value of one row: {t, t2, txt, beyond} on the measure's scale, or null when the lineage has no value */
  function chValue(d) {
    if (CH.measure === "dmax") {
      if (d.d_max == null) return null;
      return { v: d.d_max, txt: d.d_max > 2.9 ? d.d_max.toFixed(1) : d.d_max.toFixed(2) };
    }
    if (d.ret == null) return null;
    // depth = share of basic vocabulary no longer shared across the deepest split (100% − retention)
    const HZ = 1 - Math.exp(-DATA.depth_cal.horizon_depth), pct = r => Math.round(r * 100) + "%";
    if (d.ret_beyond) return { t: HZ, t2: HZ, txt: "≥" + pct(HZ), beyond: true };
    const t = 1 - d.ret, t2 = d.ret_lone && d.ret2 != null ? 1 - d.ret2 : t;
    return { t, t2, txt: pct(t) + (t2 < t - 1e-9 ? `  (${pct(t2)} without one outlier)` : ""), lone: t2 < t - 1e-9 ? d.ret_lone : null };
  }

  const xesc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c]));
  const nice = id => id.replace(/_/g, " ").toLowerCase().replace(/\b\p{L}/gu, c => c.toUpperCase());

  function chSvg() {
    const W = 1600, padL = 64, right = 48, rowH = 50, headH = 58, gap = 28, barH = 30, fs = 28;
    const groups = CH.groups.map((g, gi) => {
      let rows = g.items.map(it => ({ ...it, d: byName.get(it.name) })).filter(r => r.d).map(r => ({ ...r, val: chValue(r.d) }));
      if (CH.sort !== "entry") rows.sort((a, b) => CH.measure === "dmax" ? (b.val?.v ?? -1) - (a.val?.v ?? -1)
        : CH.sort === "value2" ? (b.val?.t2 ?? -1) - (a.val?.t2 ?? -1) : (b.val?.t ?? -1) - (a.val?.t ?? -1));
      return { ...g, gi, rows };
    }).filter(g => g.rows.length);
    const labels = groups.flatMap(g => g.rows.map(r => r.label || r.name));
    const labW = Math.min(620, Math.max(280, 20 + Math.max(0, ...labels.map(s => s.length)) * fs * 0.52));
    const x0 = padL + labW + 18, x1 = W - right - 150;
    // wrap the subtitle to the chart width (about 0.5 em per character at 24 px)
    const perLine = Math.floor((W - 2 * padL) / (24 * 0.5));
    const subLines = (CH.sub || "").split("\n").flatMap(par => {
      const out = []; let line = "";
      for (const w of par.split(/\s+/).filter(Boolean)) {
        if ((line + " " + w).trim().length > perLine) { out.push(line); line = w; } else line = (line + " " + w).trim();
      }
      if (line) out.push(line);
      return out;
    });
    let y = 64 + 26 + subLines.length * 34 + 44;
    const top = y;
    const depth = CH.measure === "depth", hz = 1 - Math.exp(-DATA.depth_cal.horizon_depth), tmax = 1;
    const X = depth ? (t => x0 + Math.min(t, tmax) / tmax * (x1 - x0 + 120)) : (v => x0 + Math.min(v, 3.15) / 3.15 * (x1 - x0 + 120));
    const body = [], lone = [];
    groups.forEach((g, k) => {
      if (k) y += gap;
      const [c] = CH_COLORS[g.gi];
      body.push(`<rect x="${padL}" y="${y + headH / 2 - 10}" width="20" height="20" fill="${c}"/><text x="${padL + 30}" y="${y + headH / 2 + 10}" font-size="${fs}" font-weight="700" fill="${CH_INK}" style="unicode-bidi:plaintext">${xesc(g.name)}</text>`);
      y += headH;
      g.rows.forEach(r => {
        const cy = y + rowH / 2, [c, lc] = CH_COLORS[g.gi], lab = r.label || r.name, v = r.val;
        body.push(`<text x="${padL + labW}" y="${cy + 10}" text-anchor="end" font-size="${fs}" fill="${CH_INK}" style="unicode-bidi:plaintext">${xesc(lab)}</text>`);
        if (!v) { body.push(`<text x="${x0 + 10}" y="${cy + 10}" font-size="${fs * .8}" fill="${CH_MUTED}">no value</text>`); y += rowH; return; }
        if (depth) {
          body.push(`<rect x="${x0}" y="${cy - barH / 2}" width="${Math.max(0, X(v.t2) - x0)}" height="${barH}" fill="${c}"/>`);
          if (v.t > v.t2 + 1e-9) {
            body.push(`<rect x="${X(v.t2)}" y="${cy - barH / 2}" width="${X(v.t) - X(v.t2)}" height="${barH}" fill="${lc}"/><rect x="${X(v.t2)}" y="${cy - barH / 2}" width="${X(v.t) - X(v.t2)}" height="${barH}" fill="url(#hatch${g.gi})"/>`);
            const short = lab.split(" (")[0], first = short.split(" ")[0].toLowerCase();
            lone.push(`${short}: ${nice(v.lone).split(" ").filter((w, i) => i || w.toLowerCase() !== first).join(" ")}`);
          }
          body.push(`<text x="${X(v.t) + 14}" y="${cy + 10}" font-size="${fs * .9}" fill="${CH_INK}">${xesc(v.txt)}</text>`);
        } else {
          const broken = v.v > 2.9, end = broken ? X(2.9) : X(v.v);
          body.push(`<rect x="${x0}" y="${cy - barH / 2}" width="${Math.max(0, end - x0)}" height="${barH}" fill="${c}"/>`);
          if (broken) for (const b of [2.62, 2.68]) body.push(`<line x1="${X(b - .03)}" y1="${cy + barH / 2 + 6}" x2="${X(b + .03)}" y2="${cy - barH / 2 - 6}" stroke="${CH_SURF}" stroke-width="9"/>`);
          body.push(`<text x="${end + 14}" y="${cy + 10}" font-size="${fs * .9}" fill="${CH_INK}">${xesc(v.txt)}</text>`);
        }
        y += rowH;
      });
    });
    const bottom = y;
    // axis, grid and reference marks
    const grid = [];
    if (depth) {
      grid.push(`<rect x="${X(hz)}" y="${top - 34}" width="${X(tmax) - X(hz)}" height="${bottom - top + 34}" fill="${CH_GRID}"/><text x="${(X(hz) + X(tmax)) / 2}" y="${top - 12}" text-anchor="middle" font-size="20" fill="${CH_MUTED}">beyond 40 words</text>`);
      for (const r of [0, .25, .5, .75, 1]) {
        const t = r;
        grid.push(`<line x1="${X(t)}" x2="${X(t)}" y1="${top}" y2="${bottom}" stroke="${CH_GRID}" stroke-width="2"/><text x="${X(t)}" y="${bottom + 38}" text-anchor="middle" font-size="24" fill="${CH_MUTED}">${Math.round(r * 100)}%</text>`);
      }
    } else {
      for (const t of [0, 1, 2, 3]) grid.push(`<line x1="${X(t)}" x2="${X(t)}" y1="${top}" y2="${bottom}" stroke="${CH_GRID}" stroke-width="2"/><text x="${X(t)}" y="${bottom + 38}" text-anchor="middle" font-size="24" fill="${CH_MUTED}">${t}</text>`);
      grid.push(`<line x1="${X(1)}" x2="${X(1)}" y1="${top - 20}" y2="${bottom}" stroke="${CH_INK2}" stroke-width="3" stroke-dasharray="9 9"/><text x="${X(1) + 12}" y="${top - 8}" font-size="22" fill="${CH_INK2}">1 = one language's worth</text>`);
    }
    const foot = [];
    if (lone.length) foot.push("Hatched: depth that rests on one outlier word list; the bracketed number leaves it out.", "Outliers: " + lone.join("; ") + ".");
    foot.push(depth ? "Data: ASJP v21 40-item word lists · PMI-LDND distances mapped to cognate retention (Lexibank expert coding) · iwsfutcmd.github.io/lingustic-diversity"
                    : "Data: ASJP v21 word lists · sound-correspondence-aware distances (PMI-LDND) · Leinster–Meckes maximum diversity · iwsfutcmd.github.io/lingustic-diversity");
    const H = bottom + 70 + foot.length * 30 + 24;
    const hatch = CH_COLORS.map(([c], i) => `<pattern id="hatch${i}" width="16" height="16" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="16" stroke="${c}" stroke-width="3"/></pattern>`).join("");
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="Inter, 'Helvetica Neue', Arial, system-ui, sans-serif" role="img" aria-label="${xesc(CH.title || "Chart")}">
      <defs>${hatch}</defs><rect width="${W}" height="${H}" fill="${CH_SURF}"/>
      <text x="${padL}" y="64" font-size="46" font-weight="700" fill="${CH_INK}" style="unicode-bidi:plaintext">${xesc(CH.title)}</text>
      ${subLines.map((s, i) => `<text x="${padL}" y="${110 + i * 34}" font-size="24" fill="${CH_INK2}" style="unicode-bidi:plaintext">${xesc(s)}</text>`).join("")}
      ${grid.join("")}${body.join("")}
      ${foot.map((s, i) => `<text x="${padL}" y="${bottom + 84 + i * 30}" font-size="19" fill="${CH_MUTED}" style="unicode-bidi:plaintext">${xesc(s)}</text>`).join("")}
    </svg>`;
  }

  function chToPng() {
    const svg = chSvg(), m = svg.match(/width="(\d+)" height="(\d+)"/);
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const cv = document.createElement("canvas"); cv.width = +m[1]; cv.height = +m[2];
        cv.getContext("2d").drawImage(img, 0, 0);
        cv.toBlob(b => b ? resolve(b) : reject(new Error("no image")), "image/png");
      };
      img.onerror = () => reject(new Error("could not draw the chart"));
      img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
    });
  }

  function chRenderControls() {
    $("ch-groups").innerHTML = CH.groups.map((g, gi) => `
      <fieldset class="chart-group" data-g="${gi}">
        <legend><span class="dot" style="background:${CH_COLORS[gi][0]}"></span> Group ${gi + 1}</legend>
        <label>Heading <input type="text" class="ch-gname" dir="auto" value="${esc(g.name)}"></label>
        <ol class="chart-items">${g.items.map((it, ii) => `<li data-i="${ii}"><input type="text" class="ch-label" dir="auto" value="${esc(it.label || it.name)}" aria-label="Label for ${esc(it.name)}"><span class="muted" dir="auto">${esc(it.name)}</span><button type="button" class="chip ch-rm" aria-label="Remove ${esc(it.name)}">×</button></li>`).join("")}</ol>
        <div class="chart-add"><input type="text" class="ch-add" list="ch-names" dir="auto" placeholder="Add a lineage…" aria-label="Add a lineage to group ${gi + 1}"><button type="button" class="chip ch-addbtn">Add</button></div>
      </fieldset>`).join("");
  }
  function chDraw() {
    $("ch-preview").innerHTML = chSvg();
    chSave();
    chToPng().then(b => { const old = $("ch-img").src; $("ch-img").src = URL.createObjectURL(b); if (old.startsWith("blob:")) URL.revokeObjectURL(old); }).catch(() => {});
  }
  function initChart() {
    if (!CH.inited) {
      CH.inited = true;
      $("ch-names").innerHTML = L.map(d => `<option value="${esc(d.name)}">`).join("");
      if (!chLoadState()) chPreset();
      $("ch-measure").value = CH.measure; $("ch-sort").value = CH.sort; $("ch-title").value = CH.title; $("ch-sub").value = CH.sub;
      $("ch-measure").addEventListener("change", e => {
        const was = CH.measure; CH.measure = e.target.value;
        if (CH.title === CH_DEFAULT_TITLE[was]) { CH.title = CH_DEFAULT_TITLE[CH.measure]; $("ch-title").value = CH.title; }
        if (CH.sub === CH_DEFAULT_SUB[was]) { CH.sub = CH_DEFAULT_SUB[CH.measure]; $("ch-sub").value = CH.sub; }
        chDraw();
      });
      $("ch-sort").addEventListener("change", e => { CH.sort = e.target.value; chDraw(); });
      $("ch-title").addEventListener("input", e => { CH.title = e.target.value; chDraw(); });
      $("ch-sub").addEventListener("input", e => { CH.sub = e.target.value; chDraw(); });
      $("ch-preset").addEventListener("click", () => { chPreset(); $("ch-title").value = CH.title; $("ch-sub").value = CH.sub; chRenderControls(); chDraw(); });
      $("ch-clear").addEventListener("click", () => { CH.groups.forEach(g => { g.items = []; }); chRenderControls(); chDraw(); });
      const host = $("ch-groups");
      host.addEventListener("input", e => {
        const fs = e.target.closest("fieldset"); if (!fs) return;
        const g = CH.groups[+fs.dataset.g];
        if (e.target.classList.contains("ch-gname")) { g.name = e.target.value; chDraw(); }
        if (e.target.classList.contains("ch-label")) { g.items[+e.target.closest("li").dataset.i].label = e.target.value; chDraw(); }
      });
      const add = fs => {
        const inp = fs.querySelector(".ch-add"), name = inp.value.trim(), g = CH.groups[+fs.dataset.g];
        if (!byName.has(name)) { $("ch-status").textContent = name ? `No lineage called "${name}".` : ""; return; }
        g.items.push({ name, label: name }); inp.value = ""; $("ch-status").textContent = "";
        chRenderControls(); chDraw();
      };
      host.addEventListener("click", e => {
        const fs = e.target.closest("fieldset"); if (!fs) return;
        if (e.target.classList.contains("ch-rm")) { CH.groups[+fs.dataset.g].items.splice(+e.target.closest("li").dataset.i, 1); chRenderControls(); chDraw(); }
        if (e.target.classList.contains("ch-addbtn")) add(fs);
      });
      host.addEventListener("keydown", e => { if (e.key === "Enter" && e.target.classList.contains("ch-add")) { e.preventDefault(); add(e.target.closest("fieldset")); } });
      $("ch-png").addEventListener("click", () => chToPng().then(async b => {
        // inside a Claude artifact, saving goes through the viewer's downloads capability (it asks the viewer first);
        // on an ordinary page, a plain download link
        let dl = null;
        try { dl = window.claude && typeof window.claude.use === "function" ? await window.claude.use("downloads") : null; } catch (e) { dl = null; }
        if (dl) {
          try { await dl.save({ filename: "isodiv-chart.png", data: b }); $("ch-status").textContent = "Saved."; }
          catch (e) { $("ch-status").textContent = e && e.code === "declined" ? "Not saved." : "Saving is not available here: use “Copy image” or the image under “Image file” below."; }
          return;
        }
        const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "isodiv-chart.png";
        document.body.appendChild(a); a.click(); a.remove();
        $("ch-status").textContent = "If no file appeared, use “Copy image” or the image under “Image file” below.";
      }).catch(err => { $("ch-status").textContent = err.message; }));
      $("ch-copy").addEventListener("click", async () => {
        try { const b = await chToPng(); await navigator.clipboard.write([new ClipboardItem({ "image/png": b })]); $("ch-status").textContent = "Copied."; }
        catch (e) { $("ch-status").textContent = "Copying an image is blocked here: use the image under “Image file” below."; }
      });
      chRenderControls();
    }
    chDraw();
  }
