"""Within- and between-lineage distances, to check that unrelated lineages are far apart.

Usage:
    python scripts/cross_family.py [--cldf data/asjp/cldf] [--out results] [--cap 25]

Writes results/cross_family.{json,csv}, cross_family.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from isodiv.asjp import load
from isodiv.distance import ldnd
from isodiv.diversity import max_diversity, similarity_matrix

from run_analysis import LINEAGES, ISOLATE_CANDIDATES, stratified


def cross_matrix(a, b) -> np.ndarray:
    return np.array([[ldnd(x, y) for y in b] for x in a])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--cap", type=int, default=25, help="max doculects per lineage")
    args = ap.parse_args()

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    k_cal = summary["k_cal"]

    rows = {r["ID"]: r for r in csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8"))}
    all_docs = load(args.cldf, keep=lambda r: any(p(r) for p in LINEAGES.values()))
    names = list(LINEAGES)
    samples = {
        n: stratified([x for x in all_docs.values() if LINEAGES[n](rows[x.id])], args.cap)
        for n in names
    }

    # Nested lineages (Romance/Germanic/Slavic ⊂ Indo-European) are related by construction.
    nested = {frozenset(p) for p in [("Romance", "Indo-European"), ("Germanic", "Indo-European"),
                                      ("Slavic", "Indo-European")]}

    within_full = {n: cross_matrix(samples[n], samples[n]) for n in names}
    within = {n: d[np.triu_indices(len(d), 1)] for n, d in within_full.items()}
    blocks = {(a, b): cross_matrix(samples[a], samples[b]) for a, b in combinations(names, 2)}
    between = {k: v.ravel() for k, v in blocks.items()}

    def stat(v: np.ndarray, f) -> float:
        v = v[~np.isnan(v)]
        return float(f(v)) if len(v) else float("nan")

    n = len(names)
    mean_m = np.full((n, n), np.nan)
    for i, a in enumerate(names):
        mean_m[i, i] = stat(within[a], np.mean)
    for (a, b), v in between.items():
        i, j = names.index(a), names.index(b)
        mean_m[i, j] = mean_m[j, i] = stat(v, np.mean)

    # Union D_max: if two lineages are truly distinct, D_max(A ∪ B) ≈ D_max(A) + D_max(B).
    union_rows = []
    for a, b in combinations(names, 2):
        d = np.block([[within_full[a], blocks[(a, b)]], [blocks[(a, b)].T, within_full[b]]])
        d = _clean(d)
        union_rows.append({
            "a": a, "b": b,
            "mean_between": stat(between[(a, b)], np.mean),
            "min_between": stat(between[(a, b)], np.min),
            "median_between": stat(between[(a, b)], np.median),
            "mean_within_a": stat(within[a], np.mean),
            "mean_within_b": stat(within[b], np.mean),
            "dmax_union": max_diversity(similarity_matrix(d, k_cal)).value,
            # k_cal is coarse: unrelated pairs (LDND≈1) are still exp(-k_cal)≈0.7 similar, so also
            # report a fine scale where LDND≈1 means "fully distinct" and D_max should add up.
            "dmax_union_k3": max_diversity(similarity_matrix(d, 3.0)).value,
            "dmax_a_k3": _dmax(within_full[a], 3.0),
            "dmax_b_k3": _dmax(within_full[b], 3.0),
            "related_by_construction": frozenset((a, b)) in nested,
        })

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "cross_family.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(union_rows[0]))
        w.writeheader()
        for r in sorted(union_rows, key=lambda r: r["mean_between"]):
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    (args.out / "cross_family.json").write_text(json.dumps({
        "cap_per_lineage": args.cap, "k_cal": k_cal, "lineages": names,
        "mean_ldnd_matrix": np.round(mean_m, 4).tolist(), "pairs": union_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    jk = next(r for r in union_rows if {r["a"], r["b"]} == {"Japonic", "Kartvelian"})
    unrelated = [r["mean_between"] for r in union_rows if not r["related_by_construction"]]
    print("Japonic vs Kartvelian:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in jk.items()})
    print(f"mean between-lineage LDND over {len(unrelated)} unrelated pairs: "
          f"{np.mean(unrelated):.3f} (min {np.min(unrelated):.3f}, max {np.max(unrelated):.3f})")
    print("mean within-lineage LDND:", {k: round(stat(v, np.mean), 3) for k, v in within.items()})

    plot(names, mean_m, within, between, union_rows, args.out / "cross_family.png")


def _clean(d: np.ndarray) -> np.ndarray:
    """Drop the doculect with the most missing pairs until no NaN remains."""
    keep = np.ones(len(d), dtype=bool)
    while keep.any() and np.isnan(d[np.ix_(keep, keep)]).any():
        keep[np.flatnonzero(keep)[np.isnan(d[np.ix_(keep, keep)]).sum(axis=1).argmax()]] = False
    return d[np.ix_(keep, keep)]


def _dmax(d: np.ndarray, k: float) -> float:
    d = _clean(d)
    return max_diversity(similarity_matrix(d, k)).value if len(d) else float("nan")


def plot(names, mean_m, within, between, union_rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 11), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # 1. mean LDND heatmap; diagonal = within-lineage mean
    order = np.argsort([0 if n in ISOLATE_CANDIDATES else 1 for n in names], kind="stable")
    m = mean_m[np.ix_(order, order)]
    lab = [names[i] for i in order]
    im = ax1.imshow(m, cmap="viridis_r", vmin=0, vmax=1.05)
    ax1.set_xticks(range(len(lab)), lab, rotation=60, ha="right", fontsize=8)
    ax1.set_yticks(range(len(lab)), lab, fontsize=8)
    for i in range(len(lab)):
        for j in range(len(lab)):
            if not np.isnan(m[i, j]) and len(lab) <= 20:
                ax1.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                         color="white" if m[i, j] > 0.55 else "black")
    ax1.set_title("Mean LDND  (diagonal = within lineage, off-diagonal = between)")
    fig.colorbar(im, ax=ax1, fraction=0.04, label="LDND  (0 = identical, ≈1 = unrelated)")

    # 2. within-lineage distributions vs. one unrelated-lineage distribution
    data = [v[~np.isnan(v)] for v in (within[n] for n in lab) if len(v)]
    labs = [n for n in lab if len(within[n])]
    ax2.boxplot(data, orientation="horizontal", tick_labels=labs, showfliers=False)
    jk = between[("Japonic", "Kartvelian")]
    ax2.axvline(np.nanmean(jk), color="crimson", lw=2, label=f"Japonic ↔ Kartvelian mean = {np.nanmean(jk):.2f}")
    ax2.axvline(1.0, color="grey", ls=":", lw=1, label="unrelated ≈ 1")
    ax2.set(xlabel="pairwise LDND within a lineage", title="Within-lineage spread vs. Japonic↔Kartvelian")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.tick_params(axis="y", labelsize=8)

    # 3. Japonic ↔ Kartvelian distance distribution against each one's internal distances
    bins = np.linspace(0, 1.2, 40)
    for v, label, c in [(within["Japonic"], "within Japonic", "tab:blue"),
                        (within["Kartvelian"], "within Kartvelian", "tab:green"),
                        (jk, "Japonic ↔ Kartvelian", "crimson")]:
        v = v[~np.isnan(v)]
        ax3.hist(v, bins=bins, alpha=0.55, color=c, density=True, label=f"{label} (n={len(v)})")
    ax3.set(xlabel="pairwise LDND", ylabel="density", title="Japonic vs. Kartvelian, pair by pair")
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
