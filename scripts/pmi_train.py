"""Train the PMI segment scores the pipeline uses (``--distance pmi``) on all of ASJP.

Same procedure as ``pmi_eval.py --train random``, but nothing is held out: training uses no
classification, so there is nothing to leak. Screens random doculect pairs, keeps those with
LDND < 0.7, and learns PMI from their same-concept word pairs (``isodiv.pmi.fit``).

Usage:
    python scripts/pmi_train.py [--cldf data/asjp/cldf] [--out results/pmi_model_all.tsv]
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import pmi_eval
from pmi_eval import random_pairs
from isodiv.asjp import load
from isodiv.pmi import fit, word_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--random-pairs", type=int, default=1_000_000)
    ap.add_argument("--max-candidates", type=int, default=150_000)
    args = ap.parse_args()
    rng = np.random.default_rng(1)

    pmi_eval.DOCS = docs = load(args.cldf)
    pairs = random_pairs(set(docs), args.random_pairs, rng)
    with Pool() as pool:
        d = pool.map(pmi_eval._ldnd, pairs, chunksize=64)
    close = [p for p, x in zip(pairs, d) if x < 0.7]
    cand, chance = word_pairs((docs[a], docs[b]) for a, b in close)
    cand = [cand[i] for i in rng.choice(len(cand), min(args.max_candidates, len(cand)), replace=False)]
    chance = [chance[i] for i in rng.choice(len(chance), min(args.max_candidates // 3, len(chance)), replace=False)]
    print(f"training: {len(docs)} doculects, {len(pairs)} random pairs, {len(close)} with LDND < 0.7, "
          f"{len(cand)} same-concept word pairs, {len(chance)} chance pairs")
    model = fit(cand, chance, log=print)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerows(model.to_rows())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
