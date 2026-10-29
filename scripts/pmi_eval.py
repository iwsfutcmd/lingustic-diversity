"""Train PMI segment scores on ASJP and test whether PMI distances separate pair classes better than LDND.

Test pairs (fixed seed): two lists of the same language (same Glottocode, X / X_2), same WALS genus,
same family but different genus, and different families. A distance is better if it ranks the classes
in that order more reliably (AUC between adjacent classes) and puts unrelated pairs further above
the same-language noise floor. A fifth class, unrelated neighbours (different families, among each
other's 3 nearest doculects and within 500 km), checks that a distance does not mistake contact for
descent. Doculects in any test pair are left out of PMI training.

Training pairs use no classification and are chosen by ``--train``:

* ``random``     — random doculect pairs with LDND < 0.7 (similarity alone decides);
* ``geo``        — each doculect's 3 geographic neighbours with LDND < 0.7 (cheaper; about half are two
                   lists of one language);
* ``geo-no-dup`` — ``geo`` without pairs whose names differ only by a _N suffix.

Usage:
    python scripts/pmi_eval.py [--cldf data/asjp/cldf] [--out results] [--n 1500] [--train random|geo|geo-no-dup]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr

from isodiv.asjp import load
from isodiv.distance import ldnd
from isodiv.pmi import PMIModel, fit, ldn_erc, pmi_erc, pmi_ldnd, word_pairs

CLASSES = ["same language", "same genus", "same family, diff. genus", "unrelated"]
CONTACT = "unrelated neighbours"
NON_GENEALOGICAL = {"", "Bookkeeping", "Speech Register", "Artificial Language", "Pidgin", "Unclassifiable"}
EARTH_KM = 6371.0
METHODS = ["ldnd", "ldn_erc", "pmi_ldnd", "pmi_erc"]


def sample_classes(rows: list[dict], n: int, seed: int = 0) -> dict[str, list[tuple[str, str]]]:
    by = {r["ID"]: r for r in rows}
    rng = np.random.default_rng(seed)
    g = defaultdict(list)
    for r in rows:
        g[(re.sub(r"_\d+$", "", r["ID"]), r["Glottocode"])].append(r["ID"])
    same = [tuple(v[:2]) for v in g.values() if len(v) >= 2]
    gen = defaultdict(list)
    for r in rows:
        if r["classification_wals"] and "." in r["classification_wals"]:
            gen[r["classification_wals"]].append(r["ID"])
    fam = defaultdict(list)
    for r in rows:
        if r["Family"] and r["classification_wals"]:
            fam[r["Family"]].append(r["ID"])
    genkeys = [k for k, v in gen.items() if len(v) >= 2]
    famkeys = [k for k, v in fam.items() if len({by[i]["classification_wals"] for i in v}) >= 2]
    ids = [r["ID"] for r in rows if r["Family"]]

    def pick(draw, k):
        out = set()
        while len(out) < k:
            if p := draw():
                out.add(p)
        return list(out)

    def p_genus():
        a, b = rng.choice(gen[genkeys[rng.integers(len(genkeys))]], 2, replace=False)
        return (a, b) if by[a]["Glottocode"] != by[b]["Glottocode"] else None

    def p_family():
        a, b = rng.choice(fam[famkeys[rng.integers(len(famkeys))]], 2, replace=False)
        return (a, b) if by[a]["classification_wals"] != by[b]["classification_wals"] else None

    def p_unrel():
        a, b = rng.choice(ids, 2, replace=False)
        return (a, b) if by[a]["Family"] != by[b]["Family"] else None

    return {
        "same language": [same[i] for i in rng.choice(len(same), min(n, len(same)), replace=False)],
        "same genus": pick(p_genus, n),
        "same family, diff. genus": pick(p_family, n),
        "unrelated": pick(p_unrel, n),
    }


def neighbour_pairs(rows: list[dict], ids: set[str], k: int = 3, max_km: float = np.inf) -> list[tuple[str, str]]:
    pts = [(r["ID"], float(r["Latitude"]), float(r["Longitude"])) for r in rows
           if r["ID"] in ids and r["Latitude"] and r["Longitude"]]
    lat, lon = np.radians([p[1] for p in pts]), np.radians([p[2] for p in pts])
    xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    chord, nn = cKDTree(xyz).query(xyz, k=k + 1)
    km = 2 * EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1))
    return sorted({tuple(sorted((pts[i][0], pts[j][0])))
                   for i, row in enumerate(nn) for j, dk in zip(row[1:], km[i, 1:]) if dk <= max_km})


def contact_pairs(rows: list[dict], n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Neighbouring doculects of different, genealogically classified families."""
    by = {r["ID"]: r for r in rows}
    ok = {r["ID"] for r in rows if r["Family"] not in NON_GENEALOGICAL}
    pairs = [p for p in neighbour_pairs(rows, ok, max_km=500) if by[p[0]]["Family"] != by[p[1]]["Family"]]
    rng = np.random.default_rng(seed)
    return [pairs[i] for i in rng.choice(len(pairs), min(n, len(pairs)), replace=False)]


def random_pairs(ids: set[str], n: int, rng) -> list[tuple[str, str]]:
    ids = sorted(ids)
    a, b = rng.integers(len(ids), size=n), rng.integers(len(ids), size=n)
    return sorted({tuple(sorted((ids[i], ids[j]))) for i, j in zip(a, b) if i != j})


def same_name(a: str, b: str) -> bool:
    return re.sub(r"_\d+$", "", a) == re.sub(r"_\d+$", "", b)


DOCS: dict = {}
MODEL: PMIModel | None = None


def _ldnd(pair):
    return ldnd(DOCS[pair[0]], DOCS[pair[1]])


def _all(pair):
    a, b = DOCS[pair[0]], DOCS[pair[1]]
    return {"ldnd": ldnd(a, b), "ldn_erc": ldn_erc(a, b), "pmi_ldnd": pmi_ldnd(MODEL, a, b), "pmi_erc": pmi_erc(MODEL, a, b)}


def auc(lo: np.ndarray, hi: np.ndarray) -> float:
    """P(a random pair from the closer class scores below one from the farther class)."""
    return float(1 - mannwhitneyu(lo, hi).statistic / (len(lo) * len(hi)))


def main() -> None:
    global DOCS, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--max-candidates", type=int, default=150_000)
    ap.add_argument("--train", choices=["random", "geo", "geo-no-dup"], default="random")
    ap.add_argument("--random-pairs", type=int, default=400_000, help="doculect pairs screened by --train random")
    args = ap.parse_args()
    tag = "" if args.train == "random" else f"_{args.train}"
    rng = np.random.default_rng(1)

    rows = list(csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8")))
    classes = sample_classes(rows, args.n)
    classes[CONTACT] = contact_pairs(rows, args.n)
    test_ids = {i for v in classes.values() for p in v for i in p}
    DOCS = load(args.cldf)
    train_ids = set(DOCS) - test_ids

    # --- training data: probably related doculect pairs, chosen by similarity alone (no labels)
    nb = random_pairs(train_ids, args.random_pairs, rng) if args.train == "random" else neighbour_pairs(rows, train_ids)
    if args.train == "geo-no-dup":
        nb = [p for p in nb if not same_name(*p)]
    with Pool() as pool:
        d = pool.map(_ldnd, nb, chunksize=64)
    close = [p for p, x in zip(nb, d) if x < 0.7]
    cand, chance = word_pairs((DOCS[a], DOCS[b]) for a, b in close)
    cand = [cand[i] for i in rng.choice(len(cand), min(args.max_candidates, len(cand)), replace=False)]
    chance = [chance[i] for i in rng.choice(len(chance), min(args.max_candidates // 3, len(chance)), replace=False)]
    print(f"training: {len(train_ids)} doculects, {len(nb)} {args.train} pairs, {len(close)} with LDND < 0.7, "
          f"{len(cand)} same-concept word pairs, {len(chance)} chance pairs")
    MODEL = fit(cand, chance, log=print)
    args.out.mkdir(exist_ok=True)
    with (args.out / f"pmi_model{tag}.tsv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerows(MODEL.to_rows())

    # --- evaluation
    res: dict[str, dict[str, np.ndarray]] = {}
    with Pool() as pool:
        for c in [*CLASSES, CONTACT]:
            pairs = [p for p in classes[c] if p[0] in DOCS and p[1] in DOCS]
            out = pool.map(_all, pairs, chunksize=8)
            v = {m: np.array([o[m] for o in out]) for m in METHODS}
            ok = np.all([np.isfinite(v[m]) for m in METHODS], axis=0)
            res[c] = {m: v[m][ok] for m in METHODS}
            with (args.out / f"pmi_eval_pairs{tag}.csv").open("a" if c != CLASSES[0] else "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                if c == CLASSES[0]:
                    w.writerow(["class", "a", "b", *METHODS])
                for p, o in zip(pairs, out):
                    w.writerow([c, *p, *(f"{o[m]:.4f}" for m in METHODS)])

    summary = {"train": args.train, "training_pairs": len(close),
               "n": {c: len(res[c]["ldnd"]) for c in [*CLASSES, CONTACT]}, "quantiles_10_50_90": {}, "auc": {}, "resolution": {}}
    print("\nn:", summary["n"])
    print("\nquantiles 10/50/90")
    for m in METHODS:
        q = {c: np.quantile(res[c][m], [.1, .5, .9]).round(3).tolist() for c in [*CLASSES, CONTACT]}
        summary["quantiles_10_50_90"][m] = q
        print(f"  {m:9s}", "  ".join(f"{c[:14]:>14s} " + "/".join(f"{x:.2f}" for x in q[c]) for c in [*CLASSES, CONTACT]))
    print("\nAUC (higher = better separated)")
    for lo, hi in [*zip(CLASSES, CLASSES[1:]), ("same genus", "unrelated"),
                   ("same family, diff. genus", CONTACT), (CONTACT, "unrelated")]:
        key = f"{lo} < {hi}"
        summary["auc"][key] = {m: auc(res[lo][m], res[hi][m]) for m in METHODS}
        print(f"  {key:52s}", "  ".join(f"{m}={x:.3f}" for m, x in summary["auc"][key].items()))
    print("\nnoise-floor resolution: (median unrelated − median same language) / IQR(same language)")
    for m in METHODS:
        s = res["same language"][m]
        r = (np.median(res["unrelated"][m]) - np.median(s)) / np.subtract(*np.quantile(s, [.75, .25]))
        summary["resolution"][m] = float(r)
        print(f"  {m:9s} {r:.2f}")
    allv = {m: np.concatenate([res[c][m] for c in CLASSES]) for m in METHODS}
    summary["spearman_vs_ldnd"] = {m: float(spearmanr(allv["ldnd"], allv[m])[0]) for m in METHODS[1:]}
    print("\nSpearman vs LDND:", {m: round(x, 3) for m, x in summary["spearman_vs_ldnd"].items()})
    (args.out / f"pmi_eval{tag}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
