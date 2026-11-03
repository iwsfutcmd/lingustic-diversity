"""How well do permutation tests detect deep relatedness that PMI-LDND scores as (nearly) unrelated?

Deep related pairs: two lists of the same top-level ASJP family whose PMI-LDND is at least --min-d (default 0.85),
sampled across families. Unrelated pairs: lists of two different families, split by distance between their
coordinates (< 1000 km, where contact can produce real correspondences, and farther). Each pair gets the permutation
tests of isodiv.regularity: plain PMI word similarity (sim), over-represented recurrent correspondences (G) and
leave-one-out recurrence (rec). Reported: share of pairs with p < 0.05 per group and distance band, and the AUC of
each z-score for related vs far unrelated pairs.

Usage:
    python scripts/regularity_eval.py [--pairs 300] [--perms 499]
Writes results/regularity_eval.{csv,json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import EXCLUSIONS_FILE  # noqa: E402

from isodiv.asjp import load, load_exclusions, read_languages
from isodiv.metric import get_metric
from isodiv.kernel import detection_limit
from isodiv.regularity import Regularity

STATS = ("sim", "g", "rec")


def km(r1, r2) -> float:
    try:
        la1, lo1, la2, lo2 = (math.radians(float(x)) for x in (r1["Latitude"], r1["Longitude"], r2["Latitude"], r2["Longitude"]))
    except ValueError:
        return float("nan")
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(min(1.0, h)))


def auc(pos: list[float], neg: list[float]) -> float:
    x = np.array(pos + neg); r = x.argsort().argsort() + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--pairs", type=int, default=300, help="pairs per group")
    ap.add_argument("--perms", type=int, default=499)
    ap.add_argument("--min-d", type=float, default=0.85)
    ap.add_argument("--min-concepts", type=int, default=30, help="lists with at least this many concepts")
    args = ap.parse_args()
    rng = np.random.default_rng(3)

    rows = {r["ID"]: r for r in read_languages(args.cldf)}
    excl = load_exclusions(EXCLUSIONS_FILE)
    docs = {i: d for i, d in load(args.cldf).items() if i not in excl and len(d.words) >= args.min_concepts}
    byfam = defaultdict(list)
    for i, d in docs.items():
        if d.family:
            byfam[d.family].append(i)
    big = sorted(f for f, v in byfam.items() if len(v) >= 10)
    metric = get_metric("pmi")
    reg = Regularity(metric.model)

    # related: within one family, PMI-LDND >= min_d; draw families evenly so a few big ones do not dominate
    cand = []
    for _ in range(args.pairs * 12):
        f = big[rng.integers(len(big))]; a, b = rng.choice(byfam[f], 2, replace=False)
        cand.append((f, a, b))
    dist = metric.values([(docs[a], docs[b]) for _, a, b in cand])
    related = [(f, a, b, d) for (f, a, b), d in zip(cand, dist) if d >= args.min_d][: args.pairs]

    fams = sorted(byfam)
    unrel_near, unrel_far = [], []
    tries = 0
    while (len(unrel_near) < args.pairs or len(unrel_far) < args.pairs) and tries < args.pairs * 400:
        tries += 1
        f, g = rng.choice(len(fams), 2, replace=False)
        a, b = rng.choice(byfam[fams[f]]), rng.choice(byfam[fams[g]])
        k = km(rows[a], rows[b])
        tgt = unrel_near if k < 1000 else unrel_far
        if len(tgt) < args.pairs:
            tgt.append((f"{fams[f]} / {fams[g]}", a, b, k))
    dn = metric.values([(docs[a], docs[b]) for _, a, b, _ in unrel_near])
    df = metric.values([(docs[a], docs[b]) for _, a, b, _ in unrel_far])

    out = []
    for group, items, ds in (("related", related, [x[3] for x in related]),
                             ("unrelated_near", unrel_near, dn), ("unrelated_far", unrel_far, df)):
        for (fam, a, b, _), d in zip(items, ds):
            r = reg.test(docs[a], docs[b], perms=args.perms)
            if r is None:
                continue
            out.append({"group": group, "family": fam, "a": a, "b": b, "d": float(d), "n": r.n,
                        "z_sim": r.z_sim, "p_sim": r.p_sim, "z_g": r.z_g, "p_g": r.p_g, "z_rec": r.z_rec, "p_rec": r.p_rec})
        print(f"{group}: {sum(o['group'] == group for o in out)} pairs", flush=True)

    summary = {"settings": vars(args) | {"cldf": str(args.cldf), "out": str(args.out)}, "groups": {}, "auc_vs_far": {}}
    bands = [(0.85, 0.90), (0.90, 0.95), (0.95, 9)]
    for group in ("related", "unrelated_near", "unrelated_far"):
        g = [o for o in out if o["group"] == group]
        rec = {"pairs": len(g)}
        for s in STATS:
            rec[f"p<0.05 {s}"] = float(np.mean([o[f"p_{s}"] < 0.05 for o in g])) if g else None
        if group == "related":
            for lo, hi in bands:
                gb = [o for o in g if lo <= o["d"] < hi]
                rec[f"d {lo}-{hi if hi < 9 else ''}"] = {"pairs": len(gb), **{s: float(np.mean([o[f'p_{s}'] < 0.05 for o in gb])) if gb else None for s in STATS}}
        summary["groups"][group] = rec
    rel = [o for o in out if o["group"] == "related"]; far = [o for o in out if o["group"] == "unrelated_far"]
    for s in STATS:
        summary["auc_vs_far"][s] = auc([o[f"z_{s}"] for o in rel], [o[f"z_{s}"] for o in far])
    fam_rate = defaultdict(list)
    for o in rel:
        fam_rate[o["family"]].append(o["p_sim"] < 0.05)
    summary["related_by_family_sim"] = {f: [len(v), float(np.mean(v))] for f, v in sorted(fam_rate.items())}

    args.out.mkdir(exist_ok=True)
    with (args.out / "regularity_eval.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    summary["detection_limit_d50"], summary["detection_slope"] = detection_limit(args.out / "regularity_eval.csv")
    print(f"detection limit (related pairs detected half the time): {summary['detection_limit_d50']:.3f}")
    (args.out / "regularity_eval.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["groups"], indent=1)); print("AUC vs far unrelated:", summary["auc_vs_far"])


if __name__ == "__main__":
    main()
