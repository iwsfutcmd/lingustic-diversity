"""Assemble the GitHub Pages site in docs/ from the results.

    python scripts/build_site.py            # needs results/ (run the pipeline and build_explorer.py first)

Writes docs/index.html, docs/explorer.html, docs/explainer.html, docs/data/wordlists.json (every ASJP word list, the PMI
model and the code tables, for the Compare varieties tab) and docs/data/*.csv. Serve docs/ from the main branch
(Settings -> Pages -> Deploy from a branch -> main, /docs).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from isodiv.asjp import core_concepts, load, load_exclusions
from isodiv.codes import load_iso1, load_macro
from isodiv.metric import Metric, load_model

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


def unrelated_reference(results: Path) -> dict:
    """Mean distance between unrelated lineages (cross_family.json), the other end of the scale from the noise floor."""
    import numpy as np
    pairs = json.loads((results / "cross_family.json").read_text(encoding="utf-8"))["pairs"]
    v = np.array([p["mean_between"] for p in pairs if not p["related_by_construction"] and p["mean_between"] is not None], float)
    v = v[np.isfinite(v)]
    return {"pmi": {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "pairs": int(len(v))},
            "ldnd": {"mean": 0.984, "note": "earlier 72-lineage run"}}


def export_wordlists(cldf: Path, model_path: Path, summary: dict, out: Path, summary_dir: Path = Path("results")) -> dict:
    """docs/data/wordlists.json: all doculects (nothing excluded), packed one field per concept."""
    concepts = sorted(core_concepts(cldf), key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))
    docs = load(cldf)
    langs = {r["ID"]: r for r in csv.DictReader((cldf / "languages.csv").open(encoding="utf-8"))}
    model = load_model(model_path)
    names = {r["ID"]: r["Name"].lstrip("*") for r in csv.DictReader((cldf / "parameters.csv").open(encoding="utf-8"))}
    from run_analysis import noise_floor            # same-language pairs, as in the pipeline
    import numpy as np
    ldnd_floor = np.quantile(noise_floor(cldf, Metric("ldnd")), [0.1, 0.5, 0.9]).tolist()
    rows = []
    for lid, d in docs.items():
        r = langs[lid]
        packed = "|".join("/".join("".join(w) for w in d.words[c]) if c in d.words else "" for c in concepts)
        assert "|" not in packed.replace("|", "", len(concepts) - 1) and packed.count("|") == len(concepts) - 1, lid
        rows.append([lid, r["Name"], r["Glottocode"], r["ISO639P3code"], r["Glottolog_Name"], r["Family"], packed])
    v = len(model.symbols)
    data = {
        "distance": summary.get("distance", "pmi"), "label": summary.get("distance_label", "PMI-LDND"),
        "k_cal": summary["k_cal"], "noise": summary["noise_floor_quantiles_10_50_90"],
        "noise_ldnd": ldnd_floor, "unrelated": unrelated_reference(summary_dir),
        "concepts": concepts, "concept_names": [names[c] for c in concepts],
        "model": {"symbols": list(model.symbols), "scores": [[round(float(x), 4) for x in model.scores[i, :v]] for i in range(v)],
                  "go": model.gap_open, "ge": model.gap_extend},
        "iso1": load_iso1(), "macro": load_macro(),
        "excluded": load_exclusions(ROOT / "config" / "exclusions.tsv"),
        "docs": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return data


REPO = "https://github.com/iwsfutcmd/lingustic-diversity"
DOWNLOADS = [
    ("summary.csv", "One row per lineage: doculects, distance spread, D_max at k_cal."),
    ("relative.csv", "D_max as a share of what the same number of unrelated lects score."),
    ("sensitivity.csv", "D_max with and without long-extinct and contact-restructured doculects."),
    ("rarefaction.csv", "D_max at equal numbers of distinct languages, for comparing lineages of different size."),
    ("adversarial.csv", "Random unrelated sets and proposed groupings (Altaic, Niger-Congo, …) on the same scale."),
    ("cross_family.csv", "Mean distance and combined D_max for every pair of lineages."),
    ("audit.csv", "Doculects ranked by how much they move their lineage's score."),
    ("bootstrap.json", "Concept-resampling intervals."),
    ("summary.json", "Everything run_analysis.py computed, including the nesting check."),
    ("samples.json", "The doculects of each lineage."),
    ("pmi_model_all.tsv", "The learned sound-correspondence scores behind PMI-LDND."),
]
GLANCE = ["Basque", "Koreanic", "Ainu", "Japonic", "Germanic", "Turkic", "Sinitic", "Uralic", "Formosan", "Indo-European",
          "Oceanic", "Austronesian", "Afro-Asiatic", "Nuclear Trans New Guinea", "Atlantic-Congo"]     # listed by D_max

INDEX = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>isodiv</title>
<meta name="description" content="A clock-free, dialect-count-robust measure of how internally diverse a language lineage is, from ASJP word lists.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Literata:opsz,wght@7..72,500;7..72,700&display=swap">
<style>
:root {
  --bg: #f4f7f9; --surface: #ffffff; --ink: #15212b; --muted: #5b6b78; --rule: #d5dde3; --grid: #e6ecf0;
  --accent: #0b7a75; --accent-ink: #ffffff; --sel: rgba(11,122,117,.12);
  --font-display: "Literata", Georgia, "Times New Roman", serif;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e151b; --surface: #16202a; --ink: #e5edf2; --muted: #93a3b0; --rule: #2a3946; --grid: #1f2c37;
    --accent: #3fc1b9; --accent-ink: #06201f; --sel: rgba(63,193,185,.16); color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e151b; --surface: #16202a; --ink: #e5edf2; --muted: #93a3b0; --rule: #2a3946; --grid: #1f2c37;
  --accent: #3fc1b9; --accent-ink: #06201f; --sel: rgba(63,193,185,.16); color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.6 var(--font-body); }
.wrap { max-inline-size: 980px; margin-inline: auto; padding-inline: max(16px, env(safe-area-inset-left, 0px)) max(16px, env(safe-area-inset-right, 0px)); }
header { padding-block: 44px 8px; }
h1 { font: 700 clamp(34px, 6vw, 52px)/1.05 var(--font-display); margin: 0; letter-spacing: -.02em; }
h2 { font: 700 22px/1.25 var(--font-display); margin: 0 0 12px; text-wrap: balance; }
.lede { font-size: 19px; color: var(--muted); max-inline-size: 62ch; margin: 14px 0 0; text-wrap: pretty; }
main { display: flex; flex-direction: column; gap: 44px; padding-block: 28px 64px; }
section { min-inline-size: 0; }
p { max-inline-size: 70ch; margin: 0 0 10px; }
a { color: var(--accent); }
.tools { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.tool { display: flex; flex-direction: column; gap: 6px; padding: 16px 18px; background: var(--surface); border: 1px solid var(--rule);
  border-radius: 6px; text-decoration: none; color: var(--ink); min-inline-size: 0; }
.tool:hover, .tool:focus-visible { border-color: var(--accent); outline: none; box-shadow: 0 0 0 2px var(--sel); }
.tool b { font: 700 18px/1.3 var(--font-display); color: var(--accent); }
.tool span { color: var(--muted); font-size: 14.5px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; inline-size: 100%; font-variant-numeric: tabular-nums; font-size: 14.5px; }
th, td { padding: 6px 12px; text-align: start; border-block-end: 1px solid var(--grid); white-space: nowrap; }
th { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
td.n, th.n { text-align: end; } td.n { font-family: var(--font-mono); font-size: 13.5px; }
.bar { display: inline-block; block-size: 8px; border-radius: 2px; background: var(--accent); opacity: .8; vertical-align: middle; margin-inline-start: 8px; }
ul.plain { margin: 0; padding-inline-start: 1.2em; display: flex; flex-direction: column; gap: 8px; max-inline-size: 72ch; }
.files { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px 22px; }
.files div { min-inline-size: 0; } .files a { font-family: var(--font-mono); font-size: 14px; } .files span { display: block; color: var(--muted); font-size: 13.5px; }
.note { color: var(--muted); font-size: 14px; }
code { font-family: var(--font-mono); font-size: .92em; background: var(--grid); padding: 1px 5px; border-radius: 3px; }
footer { border-block-start: 1px solid var(--rule); padding-block: 22px 40px; color: var(--muted); font-size: 13.5px; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>isodiv</h1>
  <p class="lede">How internally diverse is a language lineage? A measure that needs no clock and does not depend on how many languages or dialects you count, computed from __NDOC__ ASJP word lists.</p>
</header>
<main>
  <section aria-labelledby="tools-h">
    <h2 id="tools-h">Explore</h2>
    <div class="tools">
      <a class="tool" href="explorer.html"><b>Explorer</b><span>Diversity of __NLIN__ lineages and constructed sets, how it changes with scale, and how far apart lineages are.</span></a>
      <a class="tool" href="explorer.html#compare"><b>Compare varieties</b><span>The distance between any two word lists, picked by ISO 639-1 or -3 code, Glottocode or name. Computed in your browser.</span></a>
      <a class="tool" href="explainer.html"><b>How the numbers are made</b><span>From a 40-word list to D<sub>max</sub>, step by step, on real Japonic, English, German and Basque lists.</span></a>
    </div>
  </section>

  <section aria-labelledby="idea-h">
    <h2 id="idea-h">The idea</h2>
    <p><b>Distance in amount of change, not time.</b> Two ASJP 40-item word lists are compared with PMI-LDND: Levenshtein-style alignment scored by learned sound correspondences, divided by the same measure between words of different meanings to cancel chance resemblance. No time depth and no rate of change are assumed.</p>
    <p><b>Diversity that ignores duplicates.</b> Distances become similarities, Z = exp(−k·d), and a lineage is scored by the Leinster–Meckes maximum diversity D<sub>max</sub>(Z): the effective number of fully distinct lineages. Adding another copy of a variety already present cannot change it, so it does not matter how finely a language is split into dialects. The scale k is set so two lists of the same language are 0.9 similar (k = __KCAL__).</p>
  </section>

  <section aria-labelledby="glance-h">
    <h2 id="glance-h">Results at a glance</h2>
    <div class="scroll"><table>
      <thead><tr><th>Lineage</th><th class="n">Doculects</th><th class="n">D<sub>max</sub></th><th class="n">Share of unrelated maximum</th><th class="n">D<sub>max</sub> at k = 3</th></tr></thead>
      <tbody>
__ROWS__
      </tbody>
    </table></div>
    <p class="note" style="margin-block-start:10px">Sorted by D<sub>max</sub>, which is at the calibrated scale and uses every ASJP doculect of the lineage. The share of the unrelated maximum compares a lineage with a set of the same size whose members all come from different families (0 is one lineage, 1 is as diverse as unrelated lects can be). Lineages with two or three doculects are lower bounds. All __NLIN__ entries are in the <a href="explorer.html">explorer</a>.</p>
  </section>

  <section aria-labelledby="find-h">
    <h2 id="find-h">What it shows</h2>
    <ul class="plain">
      <li>The bands do not overlap: isolates (Basque, Haida, Koreanic, Ainu) sit near one list, genus-sized families (Japonic, Germanic, Turkic) in the middle, deep families (Uralic, Indo-European) above. Every major Sinitic group scores at genus level.</li>
      <li>D<sub>max</sub> measures diversity, not relatedness. Ten random unrelated lects score 1.48, and a union of unrelated families scores like Altaic or "Khoisan". Real families score below an unrelated set of the same size.</li>
      <li>It tracks lexical change, not age. Oceanic outscores Indo-European at every sample size, and Formosan, where Austronesian's deepest splits are, scores lowest of its groups.</li>
      <li>At the calibrated scale the deepest families all sit near the ceiling of about 1.56, and larger scales (the explorer's scale profiles) separate them.</li>
      <li>Single-list isolates (Sumerian, Etruscan, Elamite, Hruso, Kusunda, Nihali) are within chance of every lineage. There is no Elamo-Dravidian signal.</li>
    </ul>
  </section>

  <section aria-labelledby="data-h">
    <h2 id="data-h">Data</h2>
    <div class="files">
__FILES__
    </div>
  </section>

  <section aria-labelledby="src-h">
    <h2 id="src-h">Method and source</h2>
    <p>The <a href="__REPO__#readme">README</a> has the method, every result and the caveats (the same-language noise floor, sampling, contact). Related work is in <a href="__REPO__/blob/main/docs/related_work.md">docs/related_work.md</a>. Everything here is regenerated by the pipeline in <a href="__REPO__">the repository</a>; <code>python scripts/distance.py jpn eng</code> gives the same distances as the Compare tab.</p>
  </section>
</main>
<footer>
  Word lists: ASJP Database v21 (Wichmann, Holman, Brown, Dryer &amp; Ran 2025), CC-BY-4.0. Maximum diversity: Leinster &amp; Meckes 2016. LDND: Wichmann et al. 2010. Learned alignment scores after Jäger 2013, 2018. Built __DATE__.
</footer>
</div>
</body>
</html>
"""


def landing(results: Path, data: dict) -> str:
    import html
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    lin = {l["lineage"]: l for l in summary["lineages"]}
    rel = {r["lineage"]: r for r in csv.DictReader((results / "relative.csv").open(encoding="utf-8"))}
    rows = []
    for name in sorted(GLANCE, key=lambda n: lin[n]["d_max_kcal"]):
        l, r = lin[name], rel.get(name)
        share = float(r["relative"]) if r else None
        bar = f'<span class="bar" style="inline-size:{share * 64:.0f}px"></span>' if share is not None else ""
        rows.append(f'        <tr><td dir="auto">{html.escape(name)}</td><td class="n">{l["n_doculects"]}</td>'
                    f'<td class="n">{l["d_max_kcal"]:.3f}</td><td class="n">{"" if share is None else f"{share:.2f}"}{bar}</td>'
                    f'<td class="n">{l.get("d_max_k3", float("nan")):.1f}</td></tr>')
    files = []
    for name, what in DOWNLOADS:
        if (results / name).exists():
            files.append(f'      <div><a href="data/{name}" download>{name}</a><span>{html.escape(what)}</span></div>')
    import datetime
    return (INDEX.replace("__ROWS__", "\n".join(rows)).replace("__FILES__", "\n".join(files))
            .replace("__NDOC__", f"{len(data['docs']):,}").replace("__NLIN__", str(len(lin) + 14))
            .replace("__KCAL__", f"{summary['k_cal']:.3f}").replace("__REPO__", REPO)
            .replace("__DATE__", datetime.date.today().isoformat()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--site", type=Path, default=ROOT / "docs")
    ap.add_argument("--only-data", action="store_true", help="just write docs/data/wordlists.json")
    args = ap.parse_args()
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    site = args.site
    data = export_wordlists(args.cldf, args.results / "pmi_model_all.tsv", summary, site / "data" / "wordlists.json", args.results)
    print(f"docs/data/wordlists.json: {len(data['docs'])} doculects, {(site / 'data' / 'wordlists.json').stat().st_size / 1e6:.1f} MB")
    if args.only_data:
        return
    for name, _ in DOWNLOADS:                                            # data files people may want
        if (args.results / name).exists():
            shutil.copy(args.results / name, site / "data" / name)
    shutil.copy(args.results / "explorer.html", site / "explorer.html")
    shutil.copy(ROOT / "explainer" / "word-lists-to-dmax.html", site / "explainer.html")
    (site / "index.html").write_text(landing(args.results, data), encoding="utf-8")
    (site / ".nojekyll").write_text("", encoding="utf-8")                 # serve the files as they are
    print("site:", ", ".join(sorted(p.name for p in site.iterdir())))


if __name__ == "__main__":
    main()
