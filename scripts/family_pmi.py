"""Sensitivity check: D_max with sound-correspondence weights learned inside each group instead of across all ASJP.

Usage:
    python scripts/family_pmi.py [--groups Quechuan Romance ...] [--out results/family_pmi.json]

The headline distance (PMI-LDND) scores alignments with segment weights learned from random, probably related pairs
across all of ASJP, so it discounts correspondences that are common worldwide (r~l, k~g). A correspondence regular in
one group but rare elsewhere (Huanca Quechua l ~ Cusco r, High German ts ~ Low German t) still costs something. Here
each group gets its own weights, trained the same way (``isodiv.pmi.fit``) on same-concept word pairs from its own
doculect pairs, and D_max is recomputed on the same lists with the global and with the group's weights. Because training and scoring on the same word pairs flatters
the group's weights (a few thousand word pairs fit about 1,700 segment scores), the headline comparison is held out:
weights learned on a random half of the concepts, scored on the other half (``held_out_*``, mean of 4 halves).

Learning correspondences inside a group assumes it is one family, so this is for a known family's internal diversity
only, never for asking whether two groups are related. The kernel anchors (same-language median, detection limit) are
those calibrated on the global distances; PMI-LDND is a ratio to chance similarity, so its scale carries over
approximately but not exactly, and the comparison is a sensitivity check, not a replacement.

Groups are lineages from results/samples.json, plus the ad hoc ones in ``EXTRA``.
Writes results/family_pmi.{json,csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import combinations
from dataclasses import replace
from pathlib import Path

import numpy as np

from isodiv.diversity import max_diversity
from isodiv.kernel import Kernel
from isodiv.metric import Metric, get_metric
from isodiv.pmi import fit, word_pairs

from run_analysis import drop_nan, load_all

# High and Low German of Germany, Austria, Switzerland and Alsace (Glottolog classification), as on the shared chart
EXTRA = {
    "German dialects": [
        "STANDARD_GERMAN", "STANDARD_GERMAN_2", "BAVARIAN_GERMAN", "BAVARIAN_GERMAN_2", "ALSATIAN", "BERNESE_GERMAN",
        "SWABIAN", "RIPUARIAN_GERMAN", "SAXON_UPPER", "LOWER_SILESIAN", "EASTERN_FRISIAN", "NORTHERN_LOW_SAXON"],
    # the German dialects plus speech islands, Luxembourgish and the diaspora: more pairs to learn from
    "German (wide)": [
        "STANDARD_GERMAN", "STANDARD_GERMAN_2", "BAVARIAN_GERMAN", "BAVARIAN_GERMAN_2", "ALSATIAN", "BERNESE_GERMAN",
        "SWABIAN", "RIPUARIAN_GERMAN", "SAXON_UPPER", "LOWER_SILESIAN", "EASTERN_FRISIAN", "NORTHERN_LOW_SAXON",
        "CIMBRIAN", "MOCHENO", "WALSER", "WYMYSORYS", "LUXEMBOURGISH", "GERMAN_COLONIA_TOVAR", "GERMAN_PENNSYLVANIA",
        "HUNSRIK", "HUTTERITE_GERMAN", "PLAUTDIETSCH"],
}
DEFAULT = ["Quechuan", "German dialects", "German (wide)", "Arabic", "Yue", "Wu", "Min", "Mandarin", "Sinitic",
           "Romance", "Germanic", "Slavic", "Japonic", "Turkic"]


def train(docs, rng, max_doc_pairs: int, max_candidates: int):
    """Family weights from the group's own doculect pairs (all of them, or a random subset)."""
    pairs = list(combinations(range(len(docs)), 2))
    if len(pairs) > max_doc_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_doc_pairs, replace=False)]
    cand, chance = word_pairs((docs[i], docs[j]) for i, j in pairs)
    if len(cand) > max_candidates:
        cand = [cand[i] for i in rng.choice(len(cand), max_candidates, replace=False)]
    if len(chance) > max_candidates // 3:
        chance = [chance[i] for i in rng.choice(len(chance), max_candidates // 3, replace=False)]
    return fit(cand, chance), len(pairs), len(cand)


def score(metric: Metric, docs, kernel: Kernel, min_shared: int = 20):
    d = metric.matrix(docs, min_shared=min_shared)
    kept, d = drop_nan(docs, d, min_shared=min_shared)
    r = max_diversity(kernel(d))
    iu = np.triu_indices(len(kept), 1)
    return {"d_max": float(r.value), "n": len(kept), "median_d": float(np.nanmedian(d[iu])),
            "max_d": float(np.nanmax(d[iu])), "kept": [x.id for x in kept]}


def held_out(docs, glob: Metric, kernel: Kernel, rng, splits: int, max_doc_pairs: int, max_candidates: int):
    """Weights learned on a random half of the concepts, D_max scored on the other half (at least 10 shared concepts
    per pair), next to the global weights on the same half; mean over ``splits`` random halves. Training and scoring
    on the same word pairs flatters the group's own weights, most for small groups."""
    concepts = sorted({c for x in docs for c in x.words})
    out = []
    for _ in range(splits):
        perm = rng.permutation(concepts)
        a, b = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
        da = [replace(x, words={c: w for c, w in x.words.items() if c in a}) for x in docs]
        db = [replace(x, words={c: w for c, w in x.words.items() if c in b}) for x in docs]
        model = train(da, rng, max_doc_pairs, max_candidates)[0]
        out.append((score(glob, db, kernel, 10)["d_max"], score(Metric("pmi", model), db, kernel, 10)["d_max"]))
    return tuple(float(v) for v in np.mean(out, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--groups", nargs="*", default=DEFAULT)
    ap.add_argument("--max-doc-pairs", type=int, default=4000)
    ap.add_argument("--max-candidates", type=int, default=150_000)
    ap.add_argument("--splits", type=int, default=4, help="random concept halves for the held-out check")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    kernel = Kernel.from_summary(summary)
    samples = json.loads(Path("results/samples.json").read_text(encoding="utf-8"))
    all_docs = load_all(args.cldf)
    glob = get_metric("pmi", Path("results/pmi_model_all.tsv"))
    rng = np.random.default_rng(0)

    rows = []
    for name in args.groups:
        ids = EXTRA.get(name) or samples[name]
        docs = [all_docs[i] for i in ids if i in all_docs]
        t0 = time.time()
        g = score(glob, docs, kernel)
        model, n_pairs, n_cand = train(docs, rng, args.max_doc_pairs, args.max_candidates)
        f = score(Metric("pmi", model), docs, kernel)
        hg, hf = held_out(docs, glob, kernel, rng, args.splits, args.max_doc_pairs, args.max_candidates)
        row = {"group": name, "lists": len(docs), "train_doculect_pairs": n_pairs, "train_word_pairs": n_cand,
               "n_global": g["n"], "n_family": f["n"],
               "d_max_global": g["d_max"], "d_max_family": f["d_max"],
               "held_out_d_max_global": hg, "held_out_d_max_family": hf, "held_out_change": hf / hg - 1,
               "median_d_global": g["median_d"], "median_d_family": f["median_d"],
               "max_d_global": g["max_d"], "max_d_family": f["max_d"]}
        rows.append(row)
        print(f"{name:16} lists={len(docs):4} D global {g['d_max']:.2f} -> family {f['d_max']:.2f}   "
              f"held out {hg:.2f} -> {hf:.2f} ({100 * (hf / hg - 1):+.0f}%)   median d {g['median_d']:.3f} -> {f['median_d']:.3f}   max d {g['max_d']:.3f} -> {f['max_d']:.3f}   "
              f"({n_pairs} pairs, {n_cand} word pairs, {time.time() - t0:.0f}s)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "family_pmi.json").write_text(json.dumps({"kernel": kernel.to_json(), "groups": rows},
                                                         ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "family_pmi.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
