"""How much does masking WOLD-labelled loans change D_max for lineages that contain WOLD languages?

Usage:
    python scripts/wold_effect.py [--wold data/wold/cldf] [--cldf data/asjp/cldf] [--out results]
Writes results/wold_effect.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import LINEAGES, MAX_PER_LINEAGE, analyse, load_all, stratified  # noqa: E402

from isodiv.metric import DISTANCES, get_metric
from isodiv.wold import loan_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wold", type=Path, default=Path("data/wold/cldf"))
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--distance", choices=DISTANCES, default=None, help="default: as in results/summary.json")
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    args = ap.parse_args()
    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    k_cal = summary["k_cal"]
    metric = get_metric(args.distance or summary.get("distance", "ldnd"), args.pmi_model)
    table = loan_table(args.wold, args.cldf)
    rows = {r["ID"]: r for r in csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8"))}
    plain, masked = load_all(args.cldf), load_all(args.cldf, wold=args.wold)

    out = []
    for name, pred in LINEAGES.items():
        ids = [i for i in plain if pred(rows[i])]
        wold_ids = [i for i in ids if i in table and table[i].borrowed]
        if not wold_ids:
            continue
        a = analyse(name, stratified([plain[i] for i in ids], MAX_PER_LINEAGE), k_cal, metric)
        b = analyse(name, stratified([masked[i] for i in ids], MAX_PER_LINEAGE), k_cal, metric)
        out.append({"lineage": name, "wold_languages": ";".join(table[i].wold_language for i in wold_ids),
                    "loan_words_masked": sum(len(table[i].borrowed) for i in wold_ids),
                    "d_max": a["d_max_kcal"], "d_max_masked": b["d_max_kcal"],
                    "spread": a["spread"], "spread_masked": b["spread"]})
        print(f'{name:16s} D_max {a["d_max_kcal"]:.3f} -> {b["d_max_kcal"]:.3f}   {out[-1]["wold_languages"]}', flush=True)
    with (args.out / "wold_effect.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
