"""Rank doculects by how much they move their lineage's score, to shortlist lists worth reviewing.

Reads results/samples.json, summary.json and the distance cache data/cache/ (so run run_analysis.py first), and flags a doculect when
it is far from the rest of its lineage, thinly covered, or removing it changes D_max noticeably.
Nothing is excluded automatically: review a flagged list, then add its ID and reason to
config/exclusions.tsv and rerun the analysis.

Usage:
    python scripts/audit.py [--cldf data/asjp/cldf] [--out results]
Writes results/audit.{csv,json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import CACHE_DIR, EXCLUSIONS_FILE, LINEAGES  # noqa: E402

from isodiv.asjp import core_concepts, load, load_exclusions
from isodiv.distcache import DistCache
from isodiv.diversity import max_diversity, similarity_matrix
from isodiv.metric import get_metric

Z_FLAG = 2.0          # mean LDND this many SDs above the lineage's other doculects
LOO_FLAG = 0.015      # D_max drops by at least this when the doculect is removed
MIN_CONCEPTS = 25     # of the 40-item list (LDND itself needs 20 shared)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--only", nargs="+", help="audit just these lineages and merge into the existing audit files")
    args = ap.parse_args()

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    k = summary["k_cal"]
    distance = summary.get("distance", "ldnd")
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    metric = get_metric(distance, args.pmi_model)
    concepts = core_concepts(args.cldf)
    exclusions = load_exclusions(EXCLUSIONS_FILE)
    docs = load(args.cldf, keep=lambda r: any(p(r) for p in LINEAGES.values()))
    langs = {r["ID"]: r for r in csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8"))}

    cache = DistCache.load(CACHE_DIR / f'dist_{distance}_{summary.get("tokenization", "asjp")}.npy', metric, list(docs))
    rows = []
    for lineage in LINEAGES:
        if args.only and lineage not in args.only:
            continue
        ids = samples[lineage]
        d = cache.matrix([docs[i] for i in ids])
        n = len(ids)
        if n < 4:
            continue                      # too few doculects to tell an outlier from the lineage
        z = similarity_matrix(d, k)
        full = max_diversity(z)
        mean = d.sum(axis=1) / (n - 1)
        for i, doc_id in enumerate(ids):
            rest = np.delete(mean, i)
            zs = (mean[i] - rest.mean()) / (rest.std() or 1.0)
            n_conc = sum(c in concepts for c in docs[doc_id].words) if doc_id in docs else 0
            flagged = zs >= Z_FLAG or n_conc < MIN_CONCEPTS or full.weights[i] >= 0.05
            loo = None
            if flagged and full.weights[i] <= 1e-9:
                loo = 0.0                 # a doculect with no weight in the optimum can be removed without changing it
            elif flagged:                 # a leave-one-out solve is the costly step; only candidates
                keep = np.arange(n) != i
                start = np.where(keep, full.weights, 0.0)[keep]
                loo = full.value - max_diversity(similarity_matrix(d[np.ix_(keep, keep)], k),
                                                 warm=[start / start.sum()]).value
            rows.append({
                "lineage": lineage, "id": doc_id, "language": langs[doc_id]["Glottolog_Name"],
                "n_concepts": n_conc, "mean_ldnd": float(mean[i]), "z": float(zs),
                "weight": float(full.weights[i]), "loo_delta_dmax": loo,
                "review": bool(loo is not None and loo >= LOO_FLAG) or n_conc < MIN_CONCEPTS,
            })
        print(f"{lineage:26s} n={n:3d} D_max={full.value:.3f}", flush=True)

    if args.only and (args.out / "audit.json").exists():
        old = json.loads((args.out / "audit.json").read_text(encoding="utf-8"))["flagged"]
        rows += [r for r in old if r["lineage"] not in args.only]      # earlier review rows for other lineages
    rows.sort(key=lambda r: -(r["loo_delta_dmax"] or 0))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "audit.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lineage", "id", "language", "n_concepts", "mean_ldnd", "z", "weight", "loo_delta_dmax", "review"])
        for r in rows:
            if r["loo_delta_dmax"] is not None or r["review"]:
                w.writerow([r["lineage"], r["id"], r["language"], r["n_concepts"], f'{r["mean_ldnd"]:.3f}',
                            f'{r["z"]:.2f}', f'{r["weight"]:.3f}',
                            "" if r["loo_delta_dmax"] is None else f'{r["loo_delta_dmax"]:.3f}', r["review"]])
    (args.out / "audit.json").write_text(json.dumps({
        "thresholds": {"z": Z_FLAG, "loo_delta_dmax": LOO_FLAG, "min_concepts": MIN_CONCEPTS},
        "excluded": exclusions,
        "flagged": [r for r in rows if r["review"]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{sum(r['review'] for r in rows)} doculects flagged for review; {len(exclusions)} excluded")


if __name__ == "__main__":
    main()
