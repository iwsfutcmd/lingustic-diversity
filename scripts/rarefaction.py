"""Rarefaction: D_max at equal numbers of distinct languages, to separate sample size from composition and rate.

A lineage with more documented languages scores higher, so "Austronesian outscores Indo-European" could just mean
Austronesian has more languages. Here every lineage is compared at the same size: each draw takes m distinct Glottocodes
of the lineage's sample (results/samples.json), one random doculect from each, and scores D_max at k_cal and at k = 3
from the distance cache (so run run_analysis.py first). Reports the mean and 5-95% range over draws, and for chosen
pairs of lineages the probability that a draw of the first beats a draw of the second.

Usage:
    PYTHONPATH=scripts python scripts/rarefaction.py [--draws 100] [--sizes 3 5 10 20 40] [--lineages ...]
Writes results/rarefaction.{csv,json}.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_analysis import CACHE_DIR, K_FINE, load_all
from isodiv.diversity import max_diversity, similarity_matrix
from isodiv.distcache import DistCache
from isodiv.metric import get_metric

LINEAGES = ["Austronesian", "Malayo-Polynesian", "Oceanic", "Formosan", "Polynesian", "Indo-European", "Pama-Nyungan",
            "Atlantic-Congo", "Sino-Tibetan", "Afro-Asiatic"]
COMPARE = [("Austronesian", "Indo-European"), ("Oceanic", "Indo-European"), ("Formosan", "Indo-European"),
           ("Formosan", "Oceanic"), ("Formosan", "Malayo-Polynesian"), ("Oceanic", "Malayo-Polynesian"),
           ("Pama-Nyungan", "Indo-European")]


def rarefy(d: np.ndarray, by_g: dict[str, list[int]], m: int, draws: int, ks: list[float],
           rng: np.random.Generator) -> np.ndarray:
    """(draws, len(ks)) array of D_max for random sets of m distinct languages."""
    gs = list(by_g)
    out = np.empty((draws, len(ks)))
    for t in range(draws):
        pick = rng.choice(len(gs), size=m, replace=False)
        idx = np.array([rng.choice(by_g[gs[j]]) for j in pick])
        sub = d[np.ix_(idx, idx)]
        for c, k in enumerate(ks):
            out[t, c] = max_diversity(similarity_matrix(sub, k)).value
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--sizes", type=int, nargs="+", default=[3, 5, 10, 20, 40])
    ap.add_argument("--lineages", nargs="+", default=LINEAGES)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    k_cal, distance = summary["k_cal"], summary.get("distance", "pmi")
    ks = [k_cal, K_FINE]
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    metric = get_metric(distance, args.pmi_model)
    docs = load_all(args.cldf, summary.get("tokenization", "asjp"))
    cache = DistCache.load(CACHE_DIR / f'dist_{distance}_{summary.get("tokenization", "asjp")}.npy', metric, list(docs))
    rng = np.random.default_rng(args.seed)

    results: dict[tuple[str, int], np.ndarray] = {}
    rows = []
    for name in args.lineages:
        ids = samples[name]
        by_g: dict[str, list[int]] = defaultdict(list)
        for i, doc_id in enumerate(ids):
            by_g[docs[doc_id].glottocode].append(i)
        d = cache.matrix([docs[i] for i in ids])
        print(f"{name}: {len(ids)} doculects, {len(by_g)} languages", flush=True)
        for m in args.sizes:
            if m > len(by_g):
                continue
            vals = rarefy(d, by_g, m, args.draws, ks, rng)
            results[(name, m)] = vals
            for c, k in enumerate(ks):
                q05, q95 = np.quantile(vals[:, c], [0.05, 0.95])
                rows.append({"lineage": name, "languages": len(by_g), "m": m, "k": "k_cal" if c == 0 else "k3",
                             "mean": float(vals[:, c].mean()), "q05": float(q05), "q95": float(q95), "draws": args.draws})
            r = rows[-2]
            print(f"  m={m:2d}  D_max {r['mean']:.3f} ({r['q05']:.3f}-{r['q95']:.3f})   k=3 {rows[-1]['mean']:.2f}", flush=True)

    wins = []
    for a, b in COMPARE:
        for m in args.sizes:
            if (a, m) in results and (b, m) in results:
                for c, label in enumerate(["k_cal", "k3"]):
                    x, y = results[(a, m)][:, c], results[(b, m)][:, c]
                    wins.append({"a": a, "b": b, "m": m, "k": label, "p_a_greater": float((x[:, None] > y[None, :]).mean()),
                                 "mean_diff": float(x.mean() - y.mean())})

    args.out.mkdir(exist_ok=True)
    (args.out / "rarefaction.json").write_text(json.dumps(
        {"k_cal": k_cal, "distance": distance, "draws": args.draws, "sizes": args.sizes, "rows": rows, "comparisons": wins},
        indent=1, ensure_ascii=False), encoding="utf-8")
    with (args.out / "rarefaction.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lineage", "languages", "m", "k", "mean", "q05", "q95", "draws"])
        for r in rows:
            w.writerow([r["lineage"], r["languages"], r["m"], r["k"], f'{r["mean"]:.3f}', f'{r["q05"]:.3f}',
                        f'{r["q95"]:.3f}', r["draws"]])
    print("\nP(first beats second) at k_cal:")
    for x in wins:
        if x["k"] == "k_cal":
            print(f'  {x["a"]} vs {x["b"]:18s} m={x["m"]:2d}  P={x["p_a_greater"]:.2f}  mean diff {x["mean_diff"]:+.3f}')


if __name__ == "__main__":
    main()
