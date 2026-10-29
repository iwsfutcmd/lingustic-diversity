"""Flag likely contact-driven lects automatically, and separate them from deep branches.

Contact needs a plausible donor, so for each doculect X in a lineage F we only consider outsiders
(other Glottolog families) within ``--radius`` km. For each such neighbour D:

  gap(D) = median LDND(M, D) over relatives M of X  -  LDND(X, D)

A large gap means X resembles its neighbour D far more than its own relatives do: contact. A deep
branch that kept its inheritance (Chuvash) has no neighbour it is unusually close to.

Validation so far: across Turkic, Mongolic, Germanic, Romance, Uralic and Japonic the only doculects with a
gap above ~0.15 are Aynu (0.23, Iranian neighbours) and Moghol (0.24, Iranian neighbours); everything else
is at most 0.13.

EXPERIMENTAL: the word-level columns (``donor_like``, ``chance``, ``mean_ldnd_masked``) pool the donor's
nearby family and flag words close to it, with a shuffled-concept chance rate. On 40-item lists that
signal is not yet separable from chance (Aynu 7 vs 6 expected; Moghol 6 vs 2), so treat masked distances
as indicative only. The doculect-level ``gap`` is the useful output. Candidates only: a person should
confirm before excluding.

Usage:
    python scripts/loans.py --lineage Turkic [--lineage Germanic ...] [--out results]
    python scripts/loans.py --all
Writes results/loans_<lineage>.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import EXCLUSIONS_FILE, LINEAGES  # noqa: E402

from isodiv.asjp import core_concepts, load, load_exclusions
from isodiv.distance import ldnd

LDN = Levenshtein.normalized_distance
CLOSE = 0.34       # a donor-like word is within about one edit of the donor's word...
MARGIN = 0.20      # ...and at least this much closer than the best relative's word
GAP_FLAG = 0.25    # doculect-level: closer to a neighbour than relatives are, by this much LDND
TOP_DONORS = 3
# Not independent languages: artificial or bookkeeping entries would "donate" to their own sources.
PSEUDO_FAMILIES = frozenset({"Bookkeeping", "Artificial Language", "Speech Register", "Pidgin", "Unclassifiable"})


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (la1, lo1), (la2, lo2) = [(math.radians(x), math.radians(y)) for x, y in (a, b)]
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def best(word, others: list) -> float:
    return min((LDN(word, w) for w in others), default=float("inf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--lineage", action="append", default=[])
    ap.add_argument("--family", action="append", default=[], help="any ASJP Family value, not just the configured lineages")
    ap.add_argument("--only-ids", nargs="+", help="score just these doculects (relatives are still the whole family)")
    ap.add_argument("--cap-relatives", type=int, default=0, help="sample at most this many relatives per doculect (big families)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--radius", type=float, default=2000.0, help="km: how far away a donor may be")
    ap.add_argument("--max-donors", type=int, default=120, help="nearest outsiders considered per doculect")
    ap.add_argument("--include-excluded", action="store_true", help="also score doculects in config/exclusions.tsv")
    args = ap.parse_args()
    preds = dict(LINEAGES)
    for fam in args.family:
        preds[fam] = (lambda f: lambda r: r["Family"] == f)(fam)
    targets = list(LINEAGES) if args.all else args.lineage + args.family
    if not targets:
        ap.error("give --lineage NAME, --family NAME or --all")

    concepts = sorted(core_concepts(args.cldf))
    rows = {r["ID"]: r for r in csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8"))}
    pool = load(args.cldf, keep=lambda r: r["Family"] != "" and r["Latitude"] != "")
    excluded = set() if args.include_excluded else set(load_exclusions(EXCLUSIONS_FILE))
    coord = {i: (float(rows[i]["Latitude"]), float(rows[i]["Longitude"])) for i in pool}
    rng = np.random.default_rng(0)

    for name in targets:
        members = [d for i, d in pool.items() if preds[name](rows[i]) and i not in excluded]
        fams = {d.family for d in members}
        outsiders = [d for d in pool.values() if d.family not in fams and d.family not in PSEUDO_FAMILIES]
        ids = [d.id for d in outsiders]
        lat = np.array([coord[i][0] for i in ids]); lon = np.array([coord[i][1] for i in ids])

        @lru_cache(maxsize=None)
        def dist(a: str, b: str) -> float:
            return ldnd(pool[a], pool[b])

        report = []
        for x in members:
            if args.only_ids and x.id not in args.only_ids:
                continue
            rel = [m for m in members if m.id != x.id and m.glottocode != x.glottocode]
            if args.cap_relatives and len(rel) > args.cap_relatives:
                rel = [rel[i] for i in rng.choice(len(rel), size=args.cap_relatives, replace=False)]
            la, lo = np.radians(coord[x.id][0]), np.radians(coord[x.id][1])
            h = np.sin((np.radians(lat) - la) / 2) ** 2 + np.cos(la) * np.cos(np.radians(lat)) * np.sin((np.radians(lon) - lo) / 2) ** 2
            far = 12742 * np.arcsin(np.sqrt(h))
            order = [j for j in np.argsort(far) if far[j] <= args.radius][: args.max_donors]
            gaps = []
            for j in order:
                dd = dist(x.id, ids[j])
                if math.isnan(dd) or not rel:
                    continue
                rm = [dist(m.id, ids[j]) for m in rel]
                rm = [v for v in rm if not math.isnan(v)]
                if rm:
                    gaps.append((float(np.median(rm)) - dd, dd, float(np.median(rm)), outsiders[j]))
            gaps.sort(key=lambda t: -t[0])
            top = gaps[:TOP_DONORS]
            # word-level: pool the best donor's whole nearby family (one lect's list rarely matches
            # another's word choice); the shuffled-concept test gives the chance rate for that pool.
            hits = chance = tested = 0
            flagged, donors = set(), Counter()
            if top and top[0][0] >= GAP_FLAG / 2:
                dfam = top[0][3].family
                pool_d = [outsiders[j] for j in order if outsiders[j].family == dfam]
            else:
                pool_d = []
            for c in concepts:
                if c not in x.words or not pool_d:
                    tested += c in x.words
                    continue
                w = x.words[c][0]
                din = best(w, [m.words[c][0] for m in rel if c in m.words])
                tested += 1
                cands = [(d, LDN(w, d.words[c][0])) for d in pool_d if c in d.words]
                if cands:
                    d, dv = min(cands, key=lambda t: t[1])
                    if dv <= CLOSE and dv + MARGIN <= din:
                        hits += 1; flagged.add(c); donors[d.name] += 1
                c2 = concepts[int(rng.integers(len(concepts)))]
                if c2 != c and c2 in x.words:
                    w2 = x.words[c2][0]
                    d2 = best(w2, [m.words[c2][0] for m in rel if c2 in m.words])
                    dv2 = best(w2, [d.words[c2][0] for d in pool_d if c2 in d.words])
                    chance += int(dv2 <= CLOSE and dv2 + MARGIN <= d2)
            xm = replace(x, words={c: ws for c, ws in x.words.items() if c not in flagged})
            plain = [v for v in (ldnd(x, m) for m in rel) if not math.isnan(v)]
            masked = [v for v in (ldnd(xm, m) for m in rel) if not math.isnan(v)]
            g0 = top[0] if top else (float("nan"), float("nan"), float("nan"), None)
            report.append({
                "id": x.id, "best_donor": g0[3].name if g0[3] else "", "donor_family": g0[3].family if g0[3] else "",
                "gap": g0[0], "ldnd_to_donor": g0[1], "relatives_to_donor": g0[2],
                "donor_like": hits, "chance": chance, "tested": tested,
                "mean_ldnd": float(np.mean(plain)) if plain else float("nan"),
                "mean_ldnd_masked": float(np.mean(masked)) if masked else float("nan"),
                "top_donors": "; ".join(f"{k}x{v}" for k, v in donors.most_common(3)),
            })
        report.sort(key=lambda r: -(r["gap"] if not math.isnan(r["gap"]) else -9))
        args.out.mkdir(parents=True, exist_ok=True)
        cols = ["id", "best_donor", "donor_family", "gap", "ldnd_to_donor", "relatives_to_donor", "donor_like",
                "chance", "tested", "mean_ldnd", "mean_ldnd_masked", "top_donors"]
        with (args.out / f"loans_{name}.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in report:
                w.writerow([f"{r[c]:.3f}" if isinstance(r[c], float) else r[c] for c in cols])
        print(f"== {name} ({len(members)} doculects): largest gap to a neighbouring outsider")
        for r in report[:8]:
            print(f'  {r["id"]:22s} gap {r["gap"]:.2f} ({r["best_donor"]}, {r["donor_family"]}: X {r["ldnd_to_donor"]:.2f} vs relatives {r["relatives_to_donor"]:.2f})'
                  f'  donor-like {r["donor_like"]}/{r["tested"]} (chance {r["chance"]})  LDND {r["mean_ldnd"]:.2f} -> {r["mean_ldnd_masked"]:.2f}')


if __name__ == "__main__":
    main()
