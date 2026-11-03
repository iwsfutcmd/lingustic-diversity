"""Do correspondence-based permutation tests detect deep relatedness once word lists are long?

NorthEuraLex (Lexibank, CC-BY-4.0; ~1,000 concepts for 107 languages of northern Eurasia) is converted from IPA
segments to ASJPcode (isodiv.ipa_asjp), one form per concept (the first). Each pair of languages gets the permutation
tests of isodiv.regularity (plain PMI similarity, over-represented recurrent correspondences G, leave-one-out
recurrence) on several concept lists: the ASJP-40 concepts; borrowing-resistant lists from UraLex's concept flags
(Leipzig–Jakarta, Swadesh-207, and both together = "stable"); and 100, 200, 400 and all shared concepts in a fixed
random order (shorter lists are subsets of longer ones; these include culture vocabulary, which is often borrowed).

Pairs: Saami × Samoyed (the case ASJP-40 cannot resolve), other cross-subfamily pairs within one family, and
unrelated pairs split by distance (< 1500 km, where contact can create real correspondences, and farther).

Usage:
    python scripts/longlist_eval.py [--perms 499] [--per-group 40]
Writes results/longlist_eval.{csv,json}.
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

from isodiv.asjp import core_concepts
from isodiv.ipa_asjp import convert
from isodiv.metric import get_metric
from isodiv.regularity import Regularity

LENGTHS = ("asjp40", "lj", "swadesh207", "stable", 100, 200, 400, "all")
STATS = ("sim", "g", "rec")
SAAMI = ("sma", "smj", "sme", "smn", "sms", "sjd")
SAMOYED = ("enf", "yrk", "nio", "sel")


def read(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def km(a, b) -> float:
    la1, lo1, la2, lo2 = (math.radians(float(x)) for x in (a["Latitude"], a["Longitude"], b["Latitude"], b["Longitude"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(min(1.0, h)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nel", type=Path, default=Path("data/lexibank/northeuralex/cldf"))
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--uralex", type=Path, default=Path("data/lexibank/uralex/cldf"))
    ap.add_argument("--perms", type=int, default=499)
    ap.add_argument("--per-group", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    langs = {r["ID"]: r for r in read(args.nel / "languages.csv")}
    params = {r["ID"]: r.get("Concepticon_ID", "") for r in read(args.nel / "parameters.csv")}
    asjp_params = {r["ID"]: r["Concepticon_ID"] for r in read(args.cldf / "parameters.csv")}
    c40 = {asjp_params[c] for c in core_concepts(args.cldf)}
    ura = list(read(args.uralex / "parameters.csv"))
    named = {"lj": {p["Concepticon_ID"] for p in ura if p["Leipzig_Jakarta"] == "yes"},
             "swadesh207": {p["Concepticon_ID"] for p in ura if p["Swadesh207"] == "yes"}}
    named["stable"] = named["lj"] | named["swadesh207"]
    named["asjp40"] = c40
    words: dict[str, dict[str, str]] = defaultdict(dict)
    for r in read(args.nel / "forms.csv"):
        if r["Parameter_ID"] in words[r["Language_ID"]] or not r.get("Segments"):
            continue
        w, _ = convert(r["Segments"])
        if w:
            words[r["Language_ID"]][r["Parameter_ID"]] = w
    order = list(params); rng.shuffle(order)
    rank = {p: k for k, p in enumerate(order)}

    fam = {i: r["Family"] for i, r in langs.items()}
    sub = {i: r.get("Subfamily") or r["Family"] for i, r in langs.items()}
    ids = sorted(i for i in langs if len(words[i]) >= 400)
    groups: dict[str, list[tuple[str, str]]] = {"saami_samoyed": [(a, b) for a in SAAMI for b in SAMOYED if a in ids and b in ids]}
    rel = [(a, b) for a in ids for b in ids if a < b and fam[a] == fam[b] and sub[a] != sub[b]
           and not ({a, b} & set(SAAMI) and {a, b} & set(SAMOYED))]
    groups["related_cross_branch"] = [rel[k] for k in rng.choice(len(rel), min(args.per_group, len(rel)), replace=False)]
    unrel = [(a, b) for a in ids for b in ids if a < b and fam[a] != fam[b]]
    near = [p for p in unrel if km(langs[p[0]], langs[p[1]]) < 1500]
    far = [p for p in unrel if km(langs[p[0]], langs[p[1]]) >= 1500]
    groups["unrelated_near"] = [near[k] for k in rng.choice(len(near), min(args.per_group, len(near)), replace=False)]
    groups["unrelated_far"] = [far[k] for k in rng.choice(len(far), min(args.per_group, len(far)), replace=False)]

    reg = Regularity(get_metric("pmi").model)
    out = []
    for group, pairs in groups.items():
        for a, b in pairs:
            shared = sorted(words[a].keys() & words[b].keys(), key=rank.get)
            for L in LENGTHS:
                if L in named:
                    cs = [c for c in shared if params[c] in named[L]]
                elif L == "all":
                    cs = shared
                else:
                    if len(shared) < L:
                        continue
                    cs = shared[:L]
                r = reg.test_lists([words[a][c] for c in cs], [words[b][c] for c in cs], perms=args.perms)
                out.append({"group": group, "a": a, "b": b, "family_a": fam[a], "family_b": fam[b], "length": str(L),
                            "n": r.n, "z_sim": r.z_sim, "p_sim": r.p_sim, "z_g": r.z_g, "p_g": r.p_g,
                            "z_rec": r.z_rec, "p_rec": r.p_rec})
        print(f"{group}: {len(pairs)} pairs", flush=True)

    summary = {"settings": {"perms": args.perms, "per_group": args.per_group}, "detected_p05": {}}
    for group in groups:
        summary["detected_p05"][group] = {}
        for L in LENGTHS:
            g = [o for o in out if o["group"] == group and o["length"] == str(L)]
            if g:
                summary["detected_p05"][group][str(L)] = {"pairs": len(g), "median_n": int(np.median([o["n"] for o in g])),
                                                          **{s: float(np.mean([o[f"p_{s}"] < 0.05 for o in g])) for s in STATS}}
    args.out.mkdir(exist_ok=True)
    (args.out / "longlist_eval.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (args.out / "longlist_eval.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for group, d in summary["detected_p05"].items():
        print(group)
        for L, v in d.items():
            print(f"   {L:>6s} n≈{v['median_n']:4d}  sim {v['sim']:.2f}  G {v['g']:.2f}  rec {v['rec']:.2f}  ({v['pairs']} pairs)")


if __name__ == "__main__":
    main()
