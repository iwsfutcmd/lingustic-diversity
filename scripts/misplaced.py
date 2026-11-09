"""Flag doculects that sit closer to a sibling group than to their own: candidate misclassifications.

Usage:
    python scripts/misplaced.py [--cldf data/asjp/cldf] [--k 3] [--out results]

For each doculect, its own group is the smallest lineage in results/samples.json that contains it (Mandarin for a
Mandarin list, not Sinitic). Its sibling groups are the other lineages of the same top-level family that share no
doculect with the own group (for Mandarin: Min, Wu, Yue, Hakka, ..., Bodic, Lolo-Burmese, ...). The doculect is
compared with each by the mean distance to its ``k`` nearest lists there; it is flagged when a sibling group is
closer than its own (``margin`` = own − sibling, in PMI-LDND units, positive when flagged).

A doculect whose own group is a whole family (no lineage inside it holds the doculect) has no siblings here; a
doculect closer to another family is audit.py's cross-family check. A flag is a list to review, not proof of error:
contact varieties (a Mandarin list surrounded by Pinghua), mixed languages and lone witnesses of a branch the
lineages do not separate can all sit nearer a sibling. Nothing is excluded automatically; see config/exclusions.tsv.

Writes results/misplaced.{csv,json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import load_all, open_world_cache  # noqa: E402

from isodiv.asjp import read_languages
from isodiv.metric import get_metric


def nearest_mean(v: np.ndarray, k: int) -> float:
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan")
    return float(np.sort(v)[: min(k, len(v))].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--k", type=int, default=3, help="nearest lists averaged in each group")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    samples = {n: set(v) for n, v in json.loads((args.out / "samples.json").read_text(encoding="utf-8")).items()}
    rows = {r["ID"]: r for r in read_languages(args.cldf)}
    docs = load_all(args.cldf)
    metric = get_metric("pmi", args.pmi_model)
    cache, _ = open_world_cache(metric, args.cldf)

    family = lambda i: rows[i]["Family"] if i in rows else ""
    # lineages by top-level family (a lineage's family: the most common Family of its doculects)
    fam_of = {n: max({family(i) for i in v}, key=[family(i) for i in v].count) for n, v in samples.items() if v}
    by_fam = defaultdict(list)
    for n, f in fam_of.items():
        by_fam[f].append(n)

    out = []
    for fam, names in sorted(by_fam.items()):
        if len(names) < 2 or not fam:
            continue
        ids = sorted(set().union(*(samples[n] for n in names)) & set(docs))
        pos = {i: j for j, i in enumerate(ids)}
        print(f"{fam}: {len(names)} lineages, {len(ids)} doculects", flush=True)
        d = cache.matrix([docs[i] for i in ids])
        np.fill_diagonal(d, np.nan)
        idx = {n: np.array([pos[i] for i in samples[n] if i in pos], dtype=int) for n in names}
        for i in ids:
            holders = [n for n in names if i in samples[n]]
            own = min(holders, key=lambda n: len(samples[n]))
            sibs = [n for n in names if not (samples[n] & samples[own])]
            if not sibs:
                continue
            row = d[pos[i]]
            own_ix = idx[own][idx[own] != pos[i]]
            if not len(own_ix):
                continue
            d_own = nearest_mean(row[own_ix], args.k)
            scores = {n: nearest_mean(row[idx[n]], args.k) for n in sibs}
            best = min(scores, key=lambda n: scores[n] if np.isfinite(scores[n]) else np.inf)
            if not np.isfinite(d_own) or not np.isfinite(scores[best]):
                continue
            j_near = idx[best][np.nanargmin(row[idx[best]])]
            out.append({
                "id": i, "name": rows[i].get("Glottolog_Name", ""), "glottocode": rows[i].get("Glottocode", ""),
                "family": fam, "own": own, "own_size": len(own_ix) + 1, "d_own_k": d_own,
                "closest_sibling": best, "d_sibling_k": scores[best], "margin": d_own - scores[best],
                "nearest_in_sibling": ids[j_near], "d_nearest_in_sibling": float(row[j_near]),
                "nearest_in_own": ids[own_ix[np.nanargmin(row[own_ix])]] if np.isfinite(row[own_ix]).any() else "",
                "flag": bool(d_own > scores[best]),
            })

    out.sort(key=lambda r: -r["margin"])
    flagged = [r for r in out if r["flag"]]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "misplaced.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    (args.out / "misplaced.json").write_text(json.dumps({
        "k": args.k, "checked": len(out), "flagged": len(flagged), "rows": flagged,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"checked {len(out)} doculects with a sibling group; {len(flagged)} closer to a sibling than to their own")
    for r in flagged[:40]:
        print(f"  {r['margin']:+.3f}  {r['id']:34} {r['own']:>14} -> {r['closest_sibling']:<14} "
              f"(own {r['d_own_k']:.3f}, sibling {r['d_sibling_k']:.3f}; nearest {r['nearest_in_sibling']})")


if __name__ == "__main__":
    main()
