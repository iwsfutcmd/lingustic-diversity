"""Concept-resampling uncertainty for D_max at the noise-calibrated scale k_cal.

Each draw keeps a random 30 of the 40 ASJP concepts (delete-10 jackknife-style),
recomputes LDND, and recomputes D_max. Reports median and 5–95% interval.

Usage:
    python scripts/bootstrap.py [--draws 30] [--k-cal 0.353]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import LINEAGES, drop_nan, load_all, stratified  # noqa: E402

from isodiv.asjp import core_concepts, load
from isodiv.distance import ldnd_matrix
from isodiv.diversity import max_diversity, similarity_matrix

CAP = 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--only", nargs="+", help="recompute just these lineages and merge into the existing bootstrap.json")
    ap.add_argument("--k-cal", type=float, default=None, help="default: read from results/summary.json")
    args = ap.parse_args()

    k_cal = args.k_cal or json.loads((args.out / "summary.json").read_text(encoding="utf-8"))["k_cal"]
    concepts = sorted(core_concepts(args.cldf))
    rows = {r["ID"]: r for r in csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8"))}
    all_docs = load_all(args.cldf)
    rng = np.random.default_rng(42)
    draws = [frozenset(rng.choice(concepts, size=args.keep, replace=False)) for _ in range(args.draws)]

    path = args.out / "bootstrap.json"
    out = json.loads(path.read_text(encoding="utf-8"))["lineages"] if args.only and path.exists() else {}
    for name, pred in LINEAGES.items():
        if args.only and name not in args.only:
            continue
        docs = stratified([x for x in all_docs.values() if pred(rows[x.id])], CAP)
        vals = []
        for cs in draws:
            d = ldnd_matrix(docs, min_shared=int(args.keep * 0.5), concepts=cs)
            _, d = drop_nan(docs, d)
            vals.append(max_diversity(similarity_matrix(d, k_cal)).value)
        lo, med, hi = np.quantile(vals, [0.05, 0.5, 0.95])
        out[name] = {"median": med, "q05": lo, "q95": hi}
        print(f"{name:14s} D_max@k_cal median {med:.3f}  [{lo:.3f}, {hi:.3f}]", flush=True)

    (args.out / "bootstrap.json").write_text(
        json.dumps({"k_cal": k_cal, "draws": args.draws, "concepts_kept": args.keep, "lineages": out},
                   indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
