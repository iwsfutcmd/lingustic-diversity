"""Widest pair per lineage on the distance scale, with a contact check, using no internal classification.

For every lineage (results/samples.json, distances from the cache of run_analysis.py):

* the raw widest pair: the largest distance between two of its lists;
* the robust widest pair (``isodiv.widest``): lists are replaced by their neighbourhoods (the list and up to 4 nearest
  lists inside the same-language noise band) and pairs are compared by the median over neighbourhoods, which removes the
  noise tail. Old or deep branches count whether they have one list or several;
* the contact check for lone lists (no relative inside the noise band). A lone list is either an early branch
  (Hittite, Ugaritic) or a problem list (a mixed language, a contact-restructured list). The check needs no subgrouping:
  if the nearest list from another family is closer than the nearest list of its own family, the list is flagged and
  left out of the robust widest pair (Aynu: nearest Turkic list 0.852, nearest outside 0.648, Luristani).

The only classification used is lineage membership. Writes results/widest.{csv,json} and results/contact.{csv,json};
audit.py reads the flags.

Usage:
    PYTHONPATH=scripts python scripts/widest.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_analysis import CACHE_DIR, load_all
from isodiv.asjp import load, load_exclusions
from isodiv.distcache import DistCache
from isodiv.metric import get_metric
from isodiv.widest import neighbourhoods, robust_distances, widest

NONGEN = {"", "Bookkeeping", "Speech Register", "Artificial Language", "Pidgin", "Unclassifiable", "Mixed Language"}
K = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--lineages", nargs="+", help="only these (for testing; the output files then hold only them)")
    args = ap.parse_args()

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    distance, tok = summary.get("distance", "pmi"), summary.get("tokenization", "asjp")
    band = summary["noise_floor_quantiles_10_50_90"][2]
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    rows = {r["ID"]: r for r in csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8"))}
    metric = get_metric(distance, args.pmi_model)
    docs = load_all(args.cldf, tok)
    cache = DistCache.load(CACHE_DIR / f"dist_{distance}_{tok}.npy", metric, list(docs))
    excluded = load_exclusions(Path(__file__).resolve().parent.parent / "config" / "exclusions.tsv")
    world = {i: x for i, x in load(args.cldf, tokenization=tok).items()
             if rows[i]["Family"] not in NONGEN and i not in excluded and len(x.words) >= 20}
    world_ids = list(world)

    contact: dict[str, dict] = {}

    def check(i: str, lineage_ids: list[str]) -> dict:
        """Nearest list of the same family (or lineage) against the nearest list of any other family."""
        if i in contact:
            return contact[i]
        fam = rows[i]["Family"]
        mine = set(lineage_ids) | {j for j in world_ids if fam and rows[j]["Family"] == fam}
        v = metric.values([(docs.get(i) or world[i], world[j]) for j in world_ids])
        best_in = best_out = (np.inf, "")
        for j, x in zip(world_ids, v):
            if j == i or x != x:
                continue
            if j in mine:
                best_in = min(best_in, (float(x), j))
            else:
                best_out = min(best_out, (float(x), j))
        for j in lineage_ids:                                   # lineage members missing from `world` (blank Family)
            if j != i and j not in world:
                x = float(cache.matrix([docs[i], docs[j]])[0, 1])
                if x == x:
                    best_in = min(best_in, (x, j))
        contact[i] = {"id": i, "family": fam, "nearest_in": best_in[0], "nearest_in_id": best_in[1],
                      "nearest_out": best_out[0], "nearest_out_id": best_out[1],
                      "nearest_out_family": rows[best_out[1]]["Family"] if best_out[1] else "",
                      "flagged": best_out[0] < best_in[0]}
        return contact[i]

    out = []
    for name, ids in samples.items():
        if args.lineages and name not in args.lineages:
            continue
        n = len(ids)
        row = {"lineage": name, "n": n, "raw_widest": None, "raw_pair": [], "widest": None, "pair": [], "lone": 0,
               "flagged": []}
        if n >= 2:
            d = cache.matrix([docs[i] for i in ids])
            raw = d.copy(); np.fill_diagonal(raw, np.nan)
            a, b = np.unravel_index(np.nanargmax(raw), raw.shape)
            nb = neighbourhoods(d, band, K)
            lone = [x for x in range(n) if len(nb[x]) == 1]
            allowed = np.ones(n, bool)
            for x in lone:
                if check(ids[x], ids)["flagged"]:
                    allowed[x] = False
            w, p, q = widest(robust_distances(d, nb), allowed)
            row.update(raw_widest=float(raw[a, b]), raw_pair=[ids[a], ids[b]], widest=w if w == w else None,
                       pair=[ids[p], ids[q]] if p >= 0 else [], lone=len(lone),
                       flagged=[ids[x] for x in range(n) if not allowed[x]],
                       pair_lone=[len(nb[p]) == 1, len(nb[q]) == 1] if p >= 0 else [])
        out.append(row)
        print(f'{name:26s} n={n:5d}  raw {row["raw_widest"] or 0:.3f}  robust {row["widest"] or 0:.3f} '
              f'{" - ".join(row["pair"])}  lone {row["lone"]}  flagged {",".join(row["flagged"])}', flush=True)

    args.out.mkdir(exist_ok=True)
    (args.out / "widest.json").write_text(json.dumps({"band": band, "k": K, "distance": distance, "lineages": out},
                                                     indent=1, ensure_ascii=False), encoding="utf-8")
    with (args.out / "widest.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lineage", "n", "widest", "list_a", "list_b", "raw_widest", "raw_a", "raw_b", "lone_lists", "flagged"])
        for r in out:
            w.writerow([r["lineage"], r["n"], "" if r["widest"] is None else f'{r["widest"]:.3f}', *(r["pair"] or ["", ""]),
                        "" if r["raw_widest"] is None else f'{r["raw_widest"]:.3f}', *(r["raw_pair"] or ["", ""]),
                        r["lone"], " ".join(r["flagged"])])
    checked = sorted(contact.values(), key=lambda c: c["nearest_out"] - c["nearest_in"])
    (args.out / "contact.json").write_text(json.dumps({"band": band, "checked": checked}, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    with (args.out / "contact.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "family", "nearest_in", "nearest_in_id", "nearest_out", "nearest_out_id", "nearest_out_family", "flagged"])
        for c in checked:
            w.writerow([c["id"], c["family"], f'{c["nearest_in"]:.3f}', c["nearest_in_id"], f'{c["nearest_out"]:.3f}',
                        c["nearest_out_id"], c["nearest_out_family"], c["flagged"]])
    print(f"\n{len(checked)} lone lists checked, {sum(c['flagged'] for c in checked)} flagged as closer to another family")


if __name__ == "__main__":
    main()
