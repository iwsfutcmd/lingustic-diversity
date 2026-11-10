"""Prototype: tree-free depth = how far apart a group's two most distant members are, at the lower end of each pair's
concept-resampling range.

Usage:
    python scripts/depth_pairs.py [--resamples 200] [--q 0.05] [--out results]

For each pair of lists the PMI-LDND distance is recomputed on ``--resamples`` bootstrap resamples of their shared
concepts, and the pair's value is the ``--q`` quantile of those: the distance the 40 words support with confidence,
not a lucky high draw. The resampling seed is derived from the two list IDs, so a pair has one value whatever group
it is measured in. A group's depth is the largest of these values over its member pairs, mapped to depth with the
calibration of depth.py (share of basic vocabulary no longer shared). Removing members can only remove pairs, so
nesting holds by construction, and no tree or rate assumption is made.

Exact and fast: a pair's lower value is never above its full distance, so pairs are scored in decreasing order of
full distance and the search stops once the next full distance is below the best lower value found (or the best is
past the detection limit, beyond which depth is not resolved anyway).

Writes results/depth_pairs.json and prints a comparison with results/depth.json, depth_tree.json, custom_groups.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from custom_groups import read_groups, resolve  # noqa: E402
from depth import ANCESTORS, depth_of  # noqa: E402
from run_analysis import drop_nan, load_all, open_world_cache  # noqa: E402

from isodiv.asjp import read_languages
from isodiv.metric import get_metric

BATCH = 256


def lower_value(table, resamples: int, q: float, seed: int, min_shared: int = 20) -> float:
    """``q`` quantile of the LDND-style distance over bootstrap resamples of the pair's shared concepts."""
    if table is None or len(table.concepts) < min_shared:
        return float("nan")
    n = len(table.concepts)
    rng = np.random.default_rng(seed)
    r = rng.integers(0, n, size=(resamples, n))
    same = table.same[r].mean(axis=1)
    sub = table.diff[r[:, :, None], r[:, None, :]]                       # resamples × n × n
    off = r[:, :, None] != r[:, None, :]                                 # different concepts only
    diff = (sub * off).sum(axis=(1, 2)) / np.maximum(off.sum(axis=(1, 2)), 1)
    v = np.where(diff > 0, same / np.where(diff > 0, diff, 1), np.nan)
    return float(np.nanquantile(v, q))


def pair_seed(a: str, b: str) -> int:
    a, b = sorted((a, b))
    return int.from_bytes(hashlib.sha256(f"{a}|{b}".encode()).digest()[:8], "little")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    cal = json.loads((args.out / "depth_calibration.json").read_text(encoding="utf-8"))
    horizon = cal["horizon_d"]
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    rows = {r["ID"]: r for r in read_languages(args.cldf)}
    docs = load_all(args.cldf)
    metric = get_metric("pmi", args.pmi_model)
    cache, _ = open_world_cache(metric, args.cldf)
    flagged = set()
    contact = args.out / "contact.json"
    if contact.exists():
        cj = json.loads(contact.read_text(encoding="utf-8"))
        flagged = {r["id"] for r in cj["checked"] if r.get("flagged") and r["nearest_out"] <= cj["band"]}

    groups = {n: set(v) - ANCESTORS.get(n, set()) for n, v in samples.items()}
    for g in read_groups():
        groups[g["name"]] = ((resolve(g["include"], samples, rows) - resolve(g["exclude"], samples, rows))
                             - resolve(g["not_in_depth"], samples, rows)) & set(docs)

    lv: dict[tuple[str, str], float] = {}                # pair -> lower value, shared by every group
    out = {}
    for name in sorted(groups, key=lambda n: len(groups[n])):
        ds = [docs[i] for i in sorted(groups[name]) if i in docs and i not in flagged]
        if len(ds) < 2:
            continue
        d = cache.matrix(ds)
        ds, d = drop_nan(ds, d)
        iu = np.triu_indices(len(ds), 1)
        full = d[iu]
        order = np.argsort(-full)
        best, best_pair, k, scored = -np.inf, None, 0, 0
        while k < len(order) and full[order[k]] > best and best < horizon:
            chunk = order[k:k + BATCH]
            k += len(chunk)
            pairs = [(ds[iu[0][c]], ds[iu[1][c]]) for c in chunk]
            todo = [(a, b) for a, b in pairs if tuple(sorted((a.id, b.id))) not in lv]
            if todo:
                for (a, b), t in zip(todo, metric.tables(todo)):
                    lv[tuple(sorted((a.id, b.id)))] = lower_value(t, args.resamples, args.q, pair_seed(a.id, b.id))
                scored += len(todo)
            for (a, b), c in zip(pairs, chunk):
                v = lv[tuple(sorted((a.id, b.id)))]
                if v == v and v > best:
                    best, best_pair = v, (a.id, b.id, float(full[c]))
        if best_pair is None:
            continue
        tau = float(depth_of(min(best, horizon), cal))
        out[name] = {"lists": len(ds), "lower_d": best, "full_d": best_pair[2], "pair": best_pair[:2],
                     "depth": 1 - float(np.exp(-tau)), "beyond_horizon": bool(best >= horizon), "pairs_scored": scored}
        print(f"{name:26} n={len(ds):5} lower d {best:.3f} (full {best_pair[2]:.3f}) depth {100 * out[name]['depth']:.0f}%"
              f"{'+' if best >= horizon else ''}  {best_pair[0]} ↔ {best_pair[1]}  [{scored} pairs scored]", flush=True)

    (args.out / "depth_pairs.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
