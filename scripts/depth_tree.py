"""Prototype: depth read off one tree per family, so a subgroup can never be deeper than the group containing it.

Usage:
    python scripts/depth_tree.py [--out results]

``depth.py`` builds a separate tree for every lineage and measures that tree's deepest split, so two nested groups
are judged on two different trees and a subset can come out deeper than its family. Here each top-level family gets
one average-linkage (UPGMA) tree on pairwise depth τ, built from all of its lists, and any group's depth is the height
of the node where its members meet (their most recent common ancestor in that tree). UPGMA heights never decrease
towards the root, so removing members can only move the meeting point down or leave it: nesting holds by
construction, for lineages and custom groups alike.

The height of a UPGMA node is the mean τ between the two clusters it joins; depth = 1 − exp(−τ), the share of basic
vocabulary no longer shared. A group's own ancestors are not members (as in depth.py), and lists close to another
family are left out of the family's tree. ``set_by`` names the members on the smaller side of the meeting point when
there are at most two: those few lists alone hold the group's depth up (a lone branch, or a misplaced list).

Writes results/depth_tree.json and prints a comparison with results/depth.json and results/custom_groups.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from custom_groups import read_groups, resolve  # noqa: E402
from depth import ANCESTORS, depth_of  # noqa: E402
from run_analysis import FAMILY_FIELD, LINEAGES, drop_nan, load_all, open_world_cache  # noqa: E402

from isodiv.asjp import read_languages
from isodiv.metric import get_metric


def resampled_distances(tables, concept_index: dict, weights: np.ndarray, full: np.ndarray,
                        min_shared: int = 20) -> np.ndarray:
    """PMI-LDND of each pair under each concept resample (``weights``: resamples × concepts, multinomial counts);
    the full distance where a resample leaves a pair under ``min_shared`` distinct shared concepts."""
    out = np.empty((weights.shape[0], len(tables)), dtype=np.float32)
    for k, tb in enumerate(tables):
        if tb is None:
            out[:, k] = full[k]
            continue
        idx = np.array([concept_index[c] for c in tb.concepts])
        w = weights[:, idx].astype(float)                                 # resamples × shared concepts
        sw = w.sum(1)
        same = (w @ tb.same) / np.where(sw > 0, sw, 1)
        quad = ((w @ tb.diff) * w).sum(1) - (w * w) @ np.diag(tb.diff)     # off-diagonal (different concepts) only
        den = sw * sw - (w * w).sum(1)
        diff = quad / np.where(den > 0, den, 1)
        v = np.where((diff > 0) & ((w > 0).sum(1) >= min_shared), same / np.where(diff > 0, diff, 1), np.nan)
        out[:, k] = np.where(np.isfinite(v), v, full[k])
    return out


def mrca_height(root, leaves: dict, want: frozenset):
    node = root
    while not node.is_leaf():
        l, r = node.get_left(), node.get_right()
        if want <= leaves[l.id]:
            node = l
        elif want <= leaves[r.id]:
            node = r
        else:
            break
    return node


def collect_leaves(root) -> dict:
    leaves, stack = {}, [(root, False)]
    while stack:                                                           # iterative post-order (deep trees)
        node, done = stack.pop()
        if node.is_leaf():
            leaves[node.id] = frozenset([node.id])
        elif done:
            leaves[node.id] = leaves[node.get_left().id] | leaves[node.get_right().id]
        else:
            stack += [(node, True), (node.get_right(), False), (node.get_left(), False)]
    return leaves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--boot", type=int, default=0, help="concept-resampled trees per family (0: the single tree only)")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--families", nargs="*", default=None, help="only these families (labels), for testing")
    args = ap.parse_args()
    from itertools import combinations
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform

    cal = json.loads((args.out / "depth_calibration.json").read_text(encoding="utf-8"))
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    rows = {r["ID"]: r for r in read_languages(args.cldf)}
    docs = load_all(args.cldf)
    cache, _ = open_world_cache(get_metric("pmi", args.pmi_model), args.cldf)
    flagged = set()
    contact = args.out / "contact.json"
    if contact.exists():
        cj = json.loads(contact.read_text(encoding="utf-8"))
        flagged = {r["id"] for r in cj["checked"] if r.get("flagged") and r["nearest_out"] <= cj["band"]}

    # groups: every lineage, plus the custom groups; each belongs to the family most of its members are in
    groups = {n: set(v) for n, v in samples.items()}
    for g in read_groups():
        groups[g["name"]] = (resolve(g["include"], samples, rows) - resolve(g["exclude"], samples, rows)) & set(docs)
    ancestors = dict(ANCESTORS)
    for g in read_groups():
        ancestors[g["name"]] = resolve(g["not_in_depth"], samples, rows)
    fam_of = {n: Counter(rows[i]["Family"] for i in v if i in rows).most_common(1)[0][0] for n, v in groups.items() if v}
    label = {v: k for k, v in FAMILY_FIELD.items()}
    concepts = sorted({c for x in docs.values() for c in x.words})
    concept_index = {c: i for i, c in enumerate(concepts)}
    rng = np.random.default_rng(11)
    weights = rng.multinomial(len(concepts), np.full(len(concepts), 1 / len(concepts)), size=args.boot) if args.boot else None
    metric = get_metric("pmi", args.pmi_model)

    out = {}
    for fam in sorted(set(fam_of.values())):
        if args.families and label.get(fam, fam) not in args.families:
            continue
        members = [n for n, f in fam_of.items() if f == fam]
        pool_ids = sorted(set().union(*(groups[n] for n in members)) & set(docs) - flagged)
        ds = [docs[i] for i in pool_ids]
        d = cache.matrix(ds)
        ds, d = drop_nan(ds, d)
        if len(ds) < 2:
            continue
        pos = {x.id: k for k, x in enumerate(ds)}
        wants = {n: frozenset(pos[i] for i in groups[n] if i in pos and i not in ancestors.get(n, ()))
                 for n in members}
        wants = {n: w for n, w in wants.items() if len(w) >= 2}

        def heights(dist):
            tt = depth_of(dist, cal)
            np.fill_diagonal(tt, 0)
            root = to_tree(linkage(squareform(tt, checks=False), method="average"))
            leaves = collect_leaves(root)
            res = {}
            for n, want in wants.items():
                node = mrca_height(root, leaves, want)
                sides = [want & leaves[node.get_left().id], want & leaves[node.get_right().id]]
                res[n] = (float(node.dist), sides)
            return res

        point = heights(d)
        print(f"{label.get(fam, fam)}: {len(ds)} lists, {len(wants)} groups", flush=True)
        boot = {n: [] for n in wants}
        setters = {n: Counter() for n in wants}
        if args.boot:
            iu = np.triu_indices(len(ds), 1)
            full = d[iu]
            bd = np.empty((args.boot, len(full)), dtype=np.float32)
            for s in range(0, len(full), args.chunk):
                sl = slice(s, s + args.chunk)
                tabs = metric.tables([(ds[i], ds[j]) for i, j in zip(iu[0][sl], iu[1][sl])])
                bd[:, sl] = resampled_distances(tabs, concept_index, weights, full[sl])
            for b_ in range(args.boot):
                m = np.zeros_like(d)
                m[iu] = bd[b_]
                m = m + m.T
                for n, (h, sides) in heights(m).items():
                    boot[n].append(h)
                    small = min(sides, key=len)
                    if len(small) <= 2:
                        setters[n].update(ds[k].id for k in small)
        for n, (tau, sides) in point.items():
            small = min(sides, key=len)
            row = {"family": label.get(fam, fam), "members": len(wants[n]), "tau": tau,
                   "depth": float(1 - np.exp(-tau)), "beyond_horizon": tau >= cal["horizon_depth"] - 1e-9,
                   "split": sorted((len(s) for s in sides), reverse=True),
                   "set_by": sorted(ds[k].id for k in small) if len(small) <= 2 else []}
            if args.boot:
                bt = np.array(boot[n])
                dep = 1 - np.exp(-bt)
                row.update({"boot_depth_median": float(np.median(dep)), "boot_depth_q05": float(np.quantile(dep, .05)),
                            "boot_depth_q95": float(np.quantile(dep, .95)),
                            "boot_set_by": [[i, c / args.boot] for i, c in setters[n].most_common(3)]})
            out[n] = row

    (args.out / "depth_tree.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # comparison with the per-group trees
    cur = {r["lineage"]: r for r in json.loads((args.out / "depth.json").read_text(encoding="utf-8"))["lineages"]}
    cg = {g["name"]: g for g in json.loads((args.out / "custom_groups.json").read_text(encoding="utf-8"))["groups"]}
    def old(n):
        r = cur.get(n) or cg.get(n)
        return None if r is None else 1 - r["retention"]
    print(f"\n{'group':26} {'family':16} {'lists':>5} {'now':>6} {'tree':>6} {'boot (5-95%)':>16}  split / set by")
    for n, r in sorted(out.items(), key=lambda kv: (kv[1]["family"], -kv[1]["depth"])):
        o = old(n)
        bs = (f"{100 * r['boot_depth_median']:.0f}% ({100 * r['boot_depth_q05']:.0f}-{100 * r['boot_depth_q95']:.0f})"
              if "boot_depth_median" in r else "")
        sb = "; ".join(f"{i} {f:.0%}" for i, f in r.get("boot_set_by", [])) or ", ".join(r["set_by"])
        print(f"{n:26} {r['family'][:16]:16} {r['members']:5} {('%.0f%%' % (100 * o)) if o is not None else '–':>6} "
              f"{100 * r['depth']:5.0f}%{'+' if r['beyond_horizon'] else ' '} {bs:>16}  {r['split']} {sb}")


if __name__ == "__main__":
    main()
