"""Adversarial check: what do *constructed* groupings of unrelated lects score?

D_max measures diversity, not relatedness, so a set of mutually unrelated lects should score at least as high as any
real family of the same size. This script measures how much higher, and where proposed or rejected macro-groupings
land on the same scale:

* random unrelated sets: n doculects, each from a different genealogical family (30 draws per n);
* proposed / rejected groupings (Altaic, Macro-Altaic, Ural-Altaic, Dene-Yeniseian, "Khoisan", "Paleo-Siberian",
  Elamo-Dravidian, Niger-Congo):
  the union of their member lineages, capped at 80 doculects like real lineages;
* two baselines for each grouping: unions of the same number of randomly chosen unrelated families, of any depth
  and of *shallow* families only (genus-level D_max, 1.24–1.36, like Turkic, Mongolic and Tungusic).

Reported at k_cal and at k = 3 (a fine scale where unrelated lects count as nearly fully distinct).

Usage:
    PYTHONPATH=scripts python scripts/adversarial.py [--draws 30]   # → results/adversarial.{json,csv}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_analysis import (EXCLUSIONS_FILE, KS, LINEAGES, MAX_PER_LINEAGE, NODE_GROUPS, SINITIC_GROUPS, SUBFAMILIES,
                          drop_nan, load_all, stratified)
from isodiv.asjp import load, load_exclusions
from isodiv.diversity import max_diversity, similarity_matrix, uniform_diversity
from isodiv.metric import get_metric

K_FINE = 3.0
N_PROFILE_DRAWS = 3     # random sets whose full D_max(k) profile is averaged into the baseline curve
SIZES = (5, 10, 25, 40, 60, 80)
NON_GENEALOGICAL = {"", "Bookkeeping", "Speech Register", "Artificial Language", "Pidgin", "Unclassifiable",
                    "Mixed Language"}
PROPOSALS = {
    "Altaic": ["Turkic", "Mongolic", "Tungusic"],
    "Macro-Altaic": ["Turkic", "Mongolic", "Tungusic", "Koreanic", "Japonic"],
    "Ural-Altaic": ["Uralic", "Turkic", "Mongolic", "Tungusic"],
    "Dene-Yeniseian": ["Athabaskan-Eyak-Tlingit", "Yeniseian"],
    "Khoisan": ["Khoe-Kwadi", "Tuu", "Kxa"],
    "Paleo-Siberian": ["Chukotko-Kamchatkan", "Yukaghir", "Nivkh", "Yeniseian", "Ainu"],
    "Elamo-Dravidian": ["Elamite", "Dravidian"],
    "Niger-Congo": ["Atlantic-Congo", "Mande", "Kru", "Ijoid", "Dogon", "Heibanic", "Narrow Talodi", "Katla-Tima", "Rashad"],
}


def mean_offdiag(z: np.ndarray) -> float:
    n = len(z)
    return float((z.sum() - n) / (n * (n - 1))) if n > 1 else 1.0


def score(docs, metric, k_cal, profile: bool = False) -> dict:
    d = metric.matrix(docs)
    docs, d = drop_nan(docs, d)
    r = max_diversity(similarity_matrix(d, k_cal))
    out = {
        "n": len(docs), "d_max": r.value, "d_max_k3": max_diversity(similarity_matrix(d, K_FINE)).value,
        "uniform": uniform_diversity(similarity_matrix(d, k_cal)),
        "zbar_kcal": mean_offdiag(similarity_matrix(d, k_cal)),
        "mean_within": float(d[np.triu_indices(len(d), 1)].mean()) if len(d) > 1 else 0.0,
        "max_within": float(d.max()) if len(d) > 1 else 0.0,
        "reps": sorted(((docs[i].id, round(float(w), 3)) for i, w in enumerate(r.weights) if w > 1e-3),
                       key=lambda t: -t[1])[:8],
    }
    if profile:
        zs = [similarity_matrix(d, k) for k in KS]
        out["profile_d_max"] = [max_diversity(z).value for z in zs]
        out["profile_uniform"] = [uniform_diversity(z) for z in zs]
        out["profile_zbar"] = [mean_offdiag(z) for z in zs]
    return out


def union(names, all_docs, rows, cap=MAX_PER_LINEAGE, seed=0):
    per = math.ceil(cap / len(names))
    docs = []
    for n in names:
        docs += stratified([x for x in all_docs.values() if LINEAGES[n](rows[x.id])], per, seed=seed)
    return docs


def summarize(vals: list[float]) -> dict:
    lo, med, hi = np.quantile(vals, [0.05, 0.5, 0.95])
    return {"median": float(med), "q05": float(lo), "q95": float(hi)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    args = ap.parse_args()

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    k_cal = summary["k_cal"]
    metric = get_metric(summary.get("distance", "ldnd"), args.pmi_model)
    rows = {r["ID"]: r for r in csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8"))}
    all_docs = load_all(args.cldf)
    rng = np.random.default_rng(7)
    result: dict = {"ks": KS.tolist(), "k_cal": k_cal, "k_fine": K_FINE, "distance": metric.label, "random_unrelated": {},
                    "proposals": {}, "matched_baseline": {}}

    # 1. random sets of unrelated lects, one doculect per family (well-covered lists only), from all of ASJP
    by_fam = defaultdict(list)
    for x in load(args.cldf, exclude=load_exclusions(EXCLUSIONS_FILE)).values():
        if rows[x.id]["Family"] not in NON_GENEALOGICAL and len(x.words) >= 30:
            by_fam[rows[x.id]["Family"]].append(x)
    fams = sorted(by_fam)
    print(f"{len(fams)} genealogical families available", flush=True)
    for n in SIZES:
        draws = []
        for t in range(args.draws):
            pick = rng.choice(len(fams), size=n, replace=False)
            docs = [by_fam[fams[i]][rng.integers(len(by_fam[fams[i]]))] for i in pick]
            draws.append(score(docs, metric, k_cal, profile=(t < N_PROFILE_DRAWS)))
        med = sorted(draws, key=lambda r: r["d_max"])[len(draws) // 2]
        result["random_unrelated"][n] = {
            "d_max": summarize([r["d_max"] for r in draws]), "d_max_k3": summarize([r["d_max_k3"] for r in draws]),
            "mean_within": float(np.mean([r["mean_within"] for r in draws])),
            "uniform": float(np.median([r["uniform"] for r in draws])),
            "profile_d_max": np.mean([r["profile_d_max"] for r in draws if "profile_d_max" in r], axis=0).tolist(),
            "profile_uniform": np.mean([r["profile_uniform"] for r in draws if "profile_uniform" in r], axis=0).tolist(),
            "zbar_profile": np.mean([r["profile_zbar"] for r in draws if "profile_zbar" in r], axis=0).tolist(),
            "zbar_kcal": float(np.mean([r["zbar_kcal"] for r in draws])),
            "example": med["reps"]}
        s = result["random_unrelated"][n]
        print(f"random unrelated n={n:2d}: D_max {s['d_max']['median']:.3f} [{s['d_max']['q05']:.3f}, {s['d_max']['q95']:.3f}]"
              f"  k=3 {s['d_max_k3']['median']:.2f}", flush=True)

    # 2. proposed / rejected groupings, and 3. random unions of as many unrelated top-level families
    lin = {l["lineage"]: l for l in summary["lineages"]}
    top = [n for n in LINEAGES if n not in SUBFAMILIES and n not in SINITIC_GROUPS and n not in NODE_GROUPS and n in lin and lin[n]["n_doculects"] >= 2]
    by_family = defaultdict(list)
    for n in top:
        by_family[lin[n]["family"]].append(n)
    shallow = [n for n in top if 1.24 <= lin[n]["d_max_kcal"] <= 1.36 and lin[n]["n_doculects"] >= 6]
    result["shallow_pool"] = shallow
    result["shallow_baseline"] = {}
    for name, members in PROPOSALS.items():
        r = score(union(members, all_docs, rows), metric, k_cal, profile=True)
        r["members"] = members
        result["proposals"][name] = r
        k = len(members)
        if k not in result["matched_baseline"]:
            vals, vals3 = [], []
            for t in range(max(10, args.draws // 2)):
                fam_pick = rng.choice(sorted(by_family), size=k, replace=False)
                names = [by_family[f][rng.integers(len(by_family[f]))] for f in fam_pick]
                b = score(union(names, all_docs, rows, seed=t), metric, k_cal)
                vals.append(b["d_max"]); vals3.append(b["d_max_k3"])
            result["matched_baseline"][k] = {"d_max": summarize(vals), "d_max_k3": summarize(vals3)}
            vals, vals3 = [], []
            for t in range(max(10, args.draws // 2) * 2):
                while True:                      # distinct families, and not one of the proposals itself
                    names = list(rng.choice(shallow, size=k, replace=False))
                    if len({lin[x]["family"] for x in names}) == k and \
                            not any(set(names) <= set(m) for m in PROPOSALS.values()):
                        break
                b = score(union(names, all_docs, rows, seed=t), metric, k_cal)
                vals.append(b["d_max"]); vals3.append(b["d_max_k3"])
            result["shallow_baseline"][k] = {"d_max": summarize(vals), "d_max_k3": summarize(vals3)}
        b = result["matched_baseline"][k]
        sb = result["shallow_baseline"][k]
        print(f"{name:15s} n={r['n']:2d} D_max {r['d_max']:.3f}  k=3 {r['d_max_k3']:.2f}  | random {k}-family unions "
              f"{b['d_max']['median']:.3f} [{b['d_max']['q05']:.3f}, {b['d_max']['q95']:.3f}]; shallow {sb['d_max']['median']:.3f} "
              f"[{sb['d_max']['q05']:.3f}, {sb['d_max']['q95']:.3f}], k=3 {sb['d_max_k3']['median']:.2f} "
              f"[{sb['d_max_k3']['q05']:.2f}, {sb['d_max_k3']['q95']:.2f}]", flush=True)

    # real lineages at the fine scale (computed by run_analysis.py)
    result["real_k3"] = {n: l["d_max_k3"] for n, l in lin.items() if "d_max_k3" in l}

    args.out.mkdir(exist_ok=True)
    (args.out / "adversarial.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    with (args.out / "adversarial.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "n", "d_max", "d_max_q05", "d_max_q95", "d_max_k3"])
        for n, s in result["random_unrelated"].items():
            w.writerow([f"random unrelated (n={n})", n, *(f"{s['d_max'][q]:.3f}" for q in ("median", "q05", "q95")),
                        f"{s['d_max_k3']['median']:.3f}"])
        for name, r in result["proposals"].items():
            b = result["matched_baseline"][len(r["members"])]
            w.writerow([name, r["n"], f"{r['d_max']:.3f}", "", "", f"{r['d_max_k3']:.3f}"])
            w.writerow([f"random {len(r['members'])}-family union", "", *(f"{b['d_max'][q]:.3f}" for q in ("median", "q05", "q95")),
                        f"{b['d_max_k3']['median']:.3f}"])
            sb = result["shallow_baseline"][len(r["members"])]
            w.writerow([f"random {len(r['members'])}-shallow-family union", "",
                        *(f"{sb['d_max'][q]:.3f}" for q in ("median", "q05", "q95")), f"{sb['d_max_k3']['median']:.3f}"])


if __name__ == "__main__":
    main()
