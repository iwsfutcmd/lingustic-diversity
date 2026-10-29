"""Check the contact-gap detector (loans.py) against WOLD's expert loan labels.

WOLD (Haspelmath & Tadmor 2009; github.com/lexibank/wold) labels every word of 41 languages as inherited or
borrowed. For each WOLD language with an ASJP list (matched by Glottocode) this compares
  - the share of the ASJP 40-item concepts that WOLD marks as borrowed, with
  - loans.py's ``gap``: how much closer the ASJP list is to a nearby unrelated language than its relatives are.

Usage:
    git clone --depth 1 https://github.com/lexibank/wold data/wold
    python scripts/wold_validate.py [--wold data/wold/cldf] [--cldf data/asjp/cldf] [--out results]
Writes results/wold_validation.csv.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from isodiv.wold import loan_table  # noqa: E402


def read(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wold", type=Path, default=Path("data/wold/cldf"))
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    table = loan_table(args.wold, args.cldf)
    asjp = {r["ID"]: r for r in read(args.cldf / "languages.csv")}
    pick = {t.wold_language: asjp[i] for i, t in table.items() if asjp[i]["Latitude"] and asjp[i]["Family"]}
    rate = {t.wold_language: (len(t.borrowed) / t.n_concepts, t.n_concepts) for t in table.values()}
    wold_langs = {n: {"Name": n} for n in pick}
    families = sorted({r["Family"] for r in pick.values()})
    ids = [r["ID"] for r in pick.values()]
    print(f"{len(pick)} WOLD languages have an ASJP list; running the contact check on {len(families)} families")

    out = args.out / "_wold_loans"
    cmd = [sys.executable, str(Path(__file__).parent / "loans.py"), "--cldf", str(args.cldf), "--out", str(out),
           "--include-excluded", "--cap-relatives", "60", "--only-ids", *ids]
    for f in families:
        cmd += ["--family", f]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    gaps = {}
    for f in families:
        path = out / f"loans_{f}.csv"
        if path.exists():
            for r in read(path):
                gaps[r["id"]] = r

    rows = []
    for lid, a in pick.items():
        g = gaps.get(a["ID"])
        if g and g["gap"] not in ("", "nan"):
            rows.append({"wold_language": wold_langs[lid]["Name"], "asjp_id": a["ID"], "family": a["Family"],
                         "wold_loan_share": rate[lid][0], "concepts": rate[lid][1], "gap": float(g["gap"]),
                         "best_donor": g["best_donor"], "donor_family": g["donor_family"]})
    rows.sort(key=lambda r: -r["gap"])
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "wold_validation.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    rho, p = spearmanr([r["gap"] for r in rows], [r["wold_loan_share"] for r in rows])
    print(f"n={len(rows)}  Spearman(gap, WOLD loan share) = {rho:.2f} (p={p:.3f})")
    for r in rows:
        print(f'  {r["wold_language"]:20s} {r["family"]:18s} WOLD loans {r["wold_loan_share"]:.0%} of {r["concepts"]:2d}   gap {r["gap"]:.2f}  ({r["best_donor"]}, {r["donor_family"]})')


if __name__ == "__main__":
    main()
