"""Where do single-list isolates fall? Distance from each to every unrelated lineage, against chance.

For a single list D_max is 1 by definition, so the only thing to measure is how far it is from everything else.
Each isolate is compared with a 25-doculect sample of every top-level lineage (as in cross_family.py). The chance
baseline is the nearest unrelated lineage for 60 random lists from known families.

Usage:
    PYTHONPATH=scripts python scripts/single_lists.py      # → results/single_lists.json
"""

import csv, json, numpy as np
from pathlib import Path
from run_analysis import LINEAGES, SUBFAMILIES, SINITIC_GROUPS, load_all, stratified
from isodiv.metric import get_metric
cldf = Path("data/asjp/cldf")
rows = {r["ID"]: r for r in csv.DictReader((cldf / "languages.csv").open(encoding="utf-8"))}
docs = load_all(cldf); m = get_metric("pmi")
top = [n for n in LINEAGES if n not in SUBFAMILIES and n not in SINITIC_GROUPS]
samples = {n: stratified([x for x in docs.values() if LINEAGES[n](rows[x.id])], 25) for n in top}
fam = {n: {x.family for x in samples[n]} for n in top}
def profile(x):
    pairs, owner = [], []
    for n in top:
        if x.family in fam[n]: continue
        for y in samples[n]: pairs.append((x, y)); owner.append(n)
    v = m.values(pairs); owner = np.array(owner)
    return {n: (float(np.nanmean(v[owner == n])), float(np.nanmin(v[owner == n])) if np.isfinite(v[owner == n]).any() else np.nan) for n in dict.fromkeys(owner) if np.isfinite(v[owner == n]).any()}
out = {}
for iso in ["Sumerian", "Elamite", "Etruscan", "Hruso", "Kusunda", "Nihali"]:
    x = next(d for d in docs.values() if LINEAGES[iso](rows[d.id]))
    p = profile(x); means = np.array([a for a, _ in p.values()])
    near = sorted(p.items(), key=lambda t: t[1][0])[:5]
    out[iso] = {"id": x.id, "concepts": len(x.words), "mean_all": float(means.mean()), "nearest": [(n, round(a, 3), round(b, 3)) for n, (a, b) in near],
                "dravidian": p.get("Dravidian")}
    print(iso, x.id, len(x.words), "mean to all", round(means.mean(), 3), "nearest:", out[iso]["nearest"][:4], flush=True)
    if iso == "Elamite":
        r = sorted(p, key=lambda n: p[n][0]).index("Dravidian") + 1
        print("   Elamite->Dravidian", np.round(p["Dravidian"], 3), "rank", r, "of", len(p)); out[iso]["dravidian_rank"] = [r, len(p)]
# null: nearest unrelated lineage for random single lists of known families
rng = np.random.default_rng(3)
pool = [d for d in docs.values() if d.family and len(d.words) >= 30]
null = []
for d in [pool[i] for i in rng.choice(len(pool), 60, replace=False)]:
    p = profile(d); null.append(min(a for a, _ in p.values()))
null = np.array(null)
print("null nearest-unrelated-lineage mean distance for random lists: median", round(np.median(null), 3), "5-95%", np.round(np.quantile(null, [.05, .95]), 3))
out["_null_nearest"] = {"median": float(np.median(null)), "q05": float(np.quantile(null, .05)), "q95": float(np.quantile(null, .95))}
json.dump(out, open("results/single_lists.json", "w"), indent=1)
