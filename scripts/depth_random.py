"""Depth of constructed groupings: random unrelated sets, proposed macro-families, and their random baselines.

Usage:
    python scripts/depth_random.py [--draws 30] [--out results]

The same sets as ``adversarial.py`` (random sets of n doculects from n different families; the union of each
proposal's member lineages; unions of as many randomly chosen unrelated families, of any depth and of shallow,
genus-level ones), scored with ``depth.py``: the shared basic vocabulary across the set's deepest split.

A set of unrelated lects should be beyond the 40-word horizon at its deepest split; a real family is not (unless
it is very deep). For a proposal, the deepest split is normally between member families: its median distance is
compared with the baselines' (``pct_any``, ``pct_shallow``: the share of random unions whose split is at least as
close), so a proposal whose members are closer across the split than unrelated families would show up there, on the
distance scale, even when both are beyond the horizon in retention.

Writes results/depth_random.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adversarial import NON_GENEALOGICAL, PROPOSALS, SIZES, union  # noqa: E402
from depth import family_depth  # noqa: E402
from run_analysis import (EXCLUSIONS_FILE, LINEAGES, NODE_GROUPS, SINITIC_GROUPS, SUBFAMILIES,  # noqa: E402
                          drop_nan, load_all, open_world_cache)

from isodiv.asjp import load, load_exclusions, read_languages
from isodiv.metric import get_metric


def score(docs, cache, cal, horizon) -> dict:
    d = cache.matrix(docs)
    docs, d = drop_nan(docs, d)
    r = family_depth(d, cal, horizon)
    a, b = r.get("_a", []), r.get("_b", [])
    fams = lambda ix: sorted({docs[i].family for i in ix})
    return {"depth": r["depth"], "retention": r["retention"], "median_d": r["median_d"],
            "beyond_horizon": bool(r["beyond_horizon"]), "split": r["split"], "families_a": fams(a),
            "families_b": fams(b)}


def summarize(rs: list[dict]) -> dict:
    md = np.array([r["median_d"] for r in rs])
    ret = np.array([r["retention"] for r in rs])
    return {"median_d": {k: float(np.quantile(md, q)) for k, q in (("q05", .05), ("median", .5), ("q95", .95))},
            "retention_median": float(np.median(ret)),
            "share_beyond_horizon": float(np.mean([r["beyond_horizon"] for r in rs])), "draws": len(rs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    cal = json.loads((args.out / "depth_calibration.json").read_text(encoding="utf-8"))
    horizon = cal["horizon_d"]
    cache, _ = open_world_cache(get_metric("pmi", args.pmi_model), args.cldf)
    rows = {r["ID"]: r for r in read_languages(args.cldf)}
    all_docs = load_all(args.cldf)
    rng = np.random.default_rng(7)
    out: dict = {"horizon_d": horizon, "horizon_retention": float(np.exp(-cal["horizon_depth"])),
                 "random_unrelated": {}, "proposals": {}, "baseline_any": {}, "baseline_shallow": {}}

    by_fam = defaultdict(list)
    for x in load(args.cldf, exclude=load_exclusions(EXCLUSIONS_FILE)).values():
        if rows[x.id]["Family"] not in NON_GENEALOGICAL and len(x.words) >= 30:
            by_fam[rows[x.id]["Family"]].append(x)
    fams = sorted(by_fam)
    for n in SIZES:
        rs = []
        for _ in range(args.draws):
            pick = rng.choice(len(fams), size=n, replace=False)
            rs.append(score([by_fam[fams[i]][rng.integers(len(by_fam[fams[i]]))] for i in pick], cache, cal, horizon))
        out["random_unrelated"][n] = summarize(rs)
        s = out["random_unrelated"][n]
        print(f"random unrelated n={n:2d}: median d across the deepest split {s['median_d']['median']:.3f} "
              f"[{s['median_d']['q05']:.3f}, {s['median_d']['q95']:.3f}], beyond the horizon in "
              f"{s['share_beyond_horizon']:.0%}", flush=True)

    lin = {l["lineage"]: l for l in summary["lineages"]}
    top = [n for n in LINEAGES if n not in SUBFAMILIES and n not in SINITIC_GROUPS and n not in NODE_GROUPS
           and n in lin and lin[n]["n_doculects"] >= 2]
    by_family = defaultdict(list)
    for n in top:
        by_family[lin[n]["family"]].append(n)
    shallow = [n for n in top if 1.24 <= lin[n]["d_max_kcal"] <= 1.36 and lin[n]["n_doculects"] >= 6]
    base_any, base_sh = {}, {}
    for name, members in PROPOSALS.items():
        k = len(members)
        if k not in base_any:
            ra, rsh = [], []
            for t in range(max(10, args.draws // 2)):
                pick = rng.choice(sorted(by_family), size=k, replace=False)
                names = [by_family[f][rng.integers(len(by_family[f]))] for f in pick]
                ra.append(score(union(names, all_docs, rows, seed=t), cache, cal, horizon))
            for t in range(max(10, args.draws // 2) * 2):
                while True:
                    names = list(rng.choice(shallow, size=k, replace=False))
                    if len({lin[x]["family"] for x in names}) == k and \
                            not any(set(names) <= set(m) for m in PROPOSALS.values()):
                        break
                rsh.append(score(union(names, all_docs, rows, seed=t), cache, cal, horizon))
            base_any[k], base_sh[k] = ra, rsh
            out["baseline_any"][k], out["baseline_shallow"][k] = summarize(ra), summarize(rsh)
        r = score(union(members, all_docs, rows), cache, cal, horizon)
        r["members"] = members
        r["pct_any"] = float(np.mean([b["median_d"] <= r["median_d"] for b in base_any[k]]))
        r["pct_shallow"] = float(np.mean([b["median_d"] <= r["median_d"] for b in base_sh[k]]))
        out["proposals"][name] = r
        print(f"{name:16} median d {r['median_d']:.3f} retention {r['retention']:.0%}"
              f"{' (beyond horizon)' if r['beyond_horizon'] else ''}; split {r['families_a']} | {r['families_b']}; "
              f"random unions at least as close: any {r['pct_any']:.0%}, shallow {r['pct_shallow']:.0%}", flush=True)

    (args.out / "depth_random.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
