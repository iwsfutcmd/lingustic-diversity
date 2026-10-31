"""Distance between two varieties, picked by ISO 639-1, ISO 639-3 or Glottocode (or a name).

    python scripts/distance.py jpn eng
    python scripts/distance.py ja en --distance ldnd
    python scripts/distance.py nucl1643 cent2126 --pairs        # every doculect pair
    python scripts/distance.py zh yue                            # a macrolanguage expands to its members

A code can match several doculects (several transcriptions of one language, or the members of a macrolanguage). Every
pair of one doculect from each side is scored, and the median is the headline, with the range; ``--pairs`` lists them.
Uses the same word lists and PMI model as the pipeline, with no exclusions, so any ASJP list can be compared. Two lists
need at least 20 shared concepts to have a distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from itertools import product
from pathlib import Path

from isodiv.asjp import load
from isodiv.codes import resolve
from isodiv.metric import DISTANCES, get_metric


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="ISO 639-1/-3 code, Glottocode, doculect ID or name")
    ap.add_argument("b")
    ap.add_argument("--distance", choices=DISTANCES, default="pmi")
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--pairs", action="store_true", help="list every doculect pair")
    ap.add_argument("--max-pairs", type=int, default=2500)
    args = ap.parse_args()

    langs = list(csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8")))
    info = {r["ID"]: r for r in langs}
    docs = load(args.cldf)                        # all doculects, nothing excluded
    sides = []
    for q in (args.a, args.b):
        m = resolve(q, langs)
        ids = [i for i in m.ids if i in docs]
        if not ids:
            raise SystemExit(f"no ASJP word list matches {q!r} ({m.kind})")
        names = sorted({info[i]["Glottolog_Name"] or info[i]["Name"] for i in ids})
        print(f"{q}: {m.kind}, {len(ids)} doculect(s): {', '.join(names[:4])}{' …' if len(names) > 4 else ''}")
        sides.append(ids)
    pairs = [(x, y) for x, y in product(*sides) if x != y]
    if len(pairs) > args.max_pairs:
        raise SystemExit(f"{len(pairs)} doculect pairs; raise --max-pairs or pick narrower codes")
    metric = get_metric(args.distance, args.pmi_model)
    vals = metric.values([(docs[x], docs[y]) for x, y in pairs])
    ok = [(p, float(v)) for p, v in zip(pairs, vals) if v == v]
    if not ok:
        raise SystemExit("no pair shares 20 concepts, so there is no distance")
    d = [v for _, v in ok]
    label = metric.label
    print(f"\n{label} distance: {statistics.median(d):.3f}"
          + (f"   (median of {len(d)} doculect pairs; range {min(d):.3f}–{max(d):.3f})" if len(d) > 1 else ""))
    if len(ok) < len(pairs):
        print(f"{len(pairs) - len(ok)} pair(s) share fewer than 20 concepts and are left out")
    summary = Path("results/summary.json")
    if summary.exists():
        s = json.loads(summary.read_text(encoding="utf-8"))
        if s.get("distance") == args.distance:
            q10, q50, q90 = s["noise_floor_quantiles_10_50_90"]
            print(f"for scale: two lists of one language differ by {q10:.2f} / {q50:.2f} / {q90:.2f} "
                  f"(10th / 50th / 90th percentile); unrelated lineages average about 0.97")
    if args.pairs or len(ok) == 1:
        print()
        for (x, y), v in sorted(ok, key=lambda t: t[1]):
            print(f"  {v:.3f}  {x} ↔ {y}")


if __name__ == "__main__":
    main()
