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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
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

    out = {}
    for fam in sorted(set(fam_of.values())):
        members = [n for n, f in fam_of.items() if f == fam]
        pool_ids = sorted(set().union(*(groups[n] for n in members)) & set(docs) - flagged)
        ds = [docs[i] for i in pool_ids]
        d = cache.matrix(ds)
        ds, d = drop_nan(ds, d)
        if len(ds) < 2:
            continue
        t = depth_of(d, cal)
        np.fill_diagonal(t, 0)
        root = to_tree(linkage(squareform(t, checks=False), method="average"))
        pos = {x.id: k for k, x in enumerate(ds)}
        # leaves under every node, once
        leaves = {}

        def collect(node):
            if node.is_leaf():
                leaves[node.id] = frozenset([node.id])
            else:
                collect(node.get_left()); collect(node.get_right())
                leaves[node.id] = leaves[node.get_left().id] | leaves[node.get_right().id]
        sys.setrecursionlimit(10000)
        collect(root)
        print(f"{label.get(fam, fam)}: {len(ds)} lists, {len(members)} groups", flush=True)
        for n in members:
            want = frozenset(pos[i] for i in groups[n] if i in pos and i not in ancestors.get(n, ()))
            if len(want) < 2:
                continue
            node = root
            while not node.is_leaf():
                l, r = node.get_left(), node.get_right()
                if want <= leaves[l.id]:
                    node = l
                elif want <= leaves[r.id]:
                    node = r
                else:
                    break
            tau = float(node.dist)
            sides = [want & leaves[node.get_left().id], want & leaves[node.get_right().id]]
            small = min(sides, key=len)
            out[n] = {"family": label.get(fam, fam), "members": len(want), "tau": tau, "depth": float(1 - np.exp(-tau)),
                      "beyond_horizon": tau >= cal["horizon_depth"] - 1e-9,
                      "split": sorted((len(s) for s in sides), reverse=True),
                      "set_by": sorted(ds[k].id for k in small) if len(small) <= 2 else []}

    (args.out / "depth_tree.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # comparison with the per-group trees
    cur = {r["lineage"]: r for r in json.loads((args.out / "depth.json").read_text(encoding="utf-8"))["lineages"]}
    cg = {g["name"]: g for g in json.loads((args.out / "custom_groups.json").read_text(encoding="utf-8"))["groups"]}
    def old(n):
        r = cur.get(n) or cg.get(n)
        return None if r is None else 1 - r["retention"]
    print(f"\n{'group':26} {'family':16} {'lists':>5} {'now':>6} {'tree':>6}  split / set by")
    for n, r in sorted(out.items(), key=lambda kv: (kv[1]["family"], -kv[1]["depth"])):
        o = old(n)
        print(f"{n:26} {r['family'][:16]:16} {r['members']:5} {('%.0f%%' % (100 * o)) if o is not None else '–':>6} "
              f"{100 * r['depth']:5.0f}%{'+' if r['beyond_horizon'] else ' '} {r['split']} {', '.join(r['set_by'])}")


if __name__ == "__main__":
    main()
