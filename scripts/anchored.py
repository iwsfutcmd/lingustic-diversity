"""D_max with a similarity anchored at both ends, calibrated by the permutation test of relatedness.

Similarity is 1 at the median same-language distance (summary.json noise floor) and falls linearly to 0 at the
distance where the permutation test (scripts/regularity_eval.py) detects a related pair half the time, fitted by
logistic regression on results/regularity_eval.csv. Beyond that distance a pair counts as unrelated, so D_max of n
mutually unrelated lists is n, and a lineage's D_max is an effective number of lineages its word lists cannot tie
together. Scored on the samples of results/samples.json, with random sets of unrelated lists for reference.

Usage:
    python scripts/anchored.py
Writes results/anchored.{csv,json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import CACHE_DIR, load_all  # noqa: E402

from isodiv.distcache import DistCache
from isodiv.diversity import anchored_similarity, max_diversity
from isodiv.metric import get_metric

NONGEN = {"", "Bookkeeping", "Speech Register", "Artificial Language", "Pidgin", "Unclassifiable", "Mixed Language"}


def detection_d50(path: Path) -> tuple[float, float]:
    """Distance at which related pairs are detected half the time (logistic fit of p < 0.05 on distance)."""
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["group"] == "related"]
    d = np.array([float(r["d"]) for r in rows]); y = np.array([float(r["p_sim"]) < 0.05 for r in rows], float)

    def nll(t):
        p = np.clip(1 / (1 + np.exp(-(t[0] - t[1] * d))), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    a, b = minimize(nll, [50.0, 55.0]).x
    return float(a / b), float(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--random-draws", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(11)

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    same = summary["noise_floor_quantiles_10_50_90"][1]
    unrel, slope = detection_d50(args.out / "regularity_eval.csv")
    print(f"anchors: same-language {same:.3f}, detection limit {unrel:.3f}", flush=True)
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    metric = get_metric(summary.get("distance", "pmi"))
    tok = summary.get("tokenization", "asjp")
    docs = load_all(args.cldf, tok)
    cache = DistCache.load(CACHE_DIR / f"dist_{metric.name}_{tok}.npy", metric, list(docs))
    kcal = {r["lineage"]: r for r in summary["lineages"]}

    out = []
    for lin, ids in samples.items():
        if not ids:
            continue
        t = time.time()
        d = cache.matrix([docs[i] for i in ids])
        if np.isnan(d).any():
            continue
        r = max_diversity(anchored_similarity(d, same, unrel))
        out.append({"lineage": lin, "n": len(ids), "d_max_anchored": float(r.value),
                    "d_max_kcal": kcal.get(lin, {}).get("d_max_kcal")})
        print(f"{lin:28s} n={len(ids):5d} anchored {r.value:7.2f}  k_cal {out[-1]['d_max_kcal'] or float('nan'):.3f}  "
              f"({time.time() - t:.0f} s)", flush=True)

    # random sets of unrelated lists: one list from each of n different families
    byfam: dict[str, list[str]] = {}
    for i, x in docs.items():
        if x.family not in NONGEN and len(x.words) >= 30:
            byfam.setdefault(x.family, []).append(i)
    fams = sorted(byfam)
    rand = {}
    for n in (5, 10, 25, 40):
        vals = []
        while len(vals) < args.random_draws:
            pick = [byfam[fams[k]][rng.integers(len(byfam[fams[k]]))] for k in rng.choice(len(fams), n, replace=False)]
            dm = np.zeros((n, n)); pr = [(a, b) for a in range(n) for b in range(a + 1, n)]
            v = metric.values([(docs[pick[a]], docs[pick[b]]) for a, b in pr])
            if np.isnan(v).any():
                continue
            for (a, b), x in zip(pr, v):
                dm[a, b] = dm[b, a] = x
            vals.append(max_diversity(anchored_similarity(dm, same, unrel)).value)
        rand[n] = {"median": float(np.median(vals)), "q05": float(np.quantile(vals, 0.05)), "q95": float(np.quantile(vals, 0.95))}
        print(f"random unrelated n={n}: median {rand[n]['median']:.2f} ({rand[n]['q05']:.2f}-{rand[n]['q95']:.2f})", flush=True)

    args.out.mkdir(exist_ok=True)
    (args.out / "anchored.json").write_text(json.dumps({
        "anchors": {"same_language_median": same, "detection_limit_d50": unrel, "logistic_slope": slope},
        "random_unrelated": rand, "lineages": out}, indent=1), encoding="utf-8")
    with (args.out / "anchored.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["lineage", "n", "d_max_anchored", "d_max_kcal"])
        for r in sorted(out, key=lambda r: -r["d_max_anchored"]):
            w.writerow([r["lineage"], r["n"], f'{r["d_max_anchored"]:.3f}', "" if r["d_max_kcal"] is None else f'{r["d_max_kcal"]:.3f}'])


if __name__ == "__main__":
    main()
