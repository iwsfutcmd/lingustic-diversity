"""Sensitivity of D_max to contested doculects: report each lineage with and without them.

Two scenarios, each removing doculects from the lineage samples of results/samples.json and re-solving D_max at k_cal
and at the fine scale k = 3 from the distance cache (so run run_analysis.py first):

* ``long_extinct``: doculects ASJP flags ``long_extinct`` (Old Chinese, Middle Chinese, Proto-Germanic, Gothic, ...),
  which only lineages with a written past have;
* ``contact``: the doculects listed in config/sensitivity.tsv (``id<TAB>scenario<TAB>reason``), e.g. Wutun, a
  Chinese-based language restructured by Tibetan and Mongolic contact.

Nothing is excluded from the main results; this shows how much each choice would matter. Only lineages that contain a
removed doculect appear. The last column puts the change next to the width of the concept-resampling interval.

Usage:
    PYTHONPATH=scripts python scripts/sensitivity.py [--cldf data/asjp/cldf] [--out results]
Writes results/sensitivity.{csv,json}.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_analysis import CACHE_DIR, K_FINE, load_all
from isodiv.diversity import max_diversity, similarity_matrix
from isodiv.distcache import DistCache
from isodiv.metric import get_metric

SENS_FILE = Path(__file__).resolve().parent.parent / "config" / "sensitivity.tsv"


def scenarios(languages: Path) -> dict[str, dict[str, str]]:
    """scenario -> {doculect id: reason}."""
    out: dict[str, dict[str, str]] = {"long_extinct": {}}
    for r in csv.DictReader(languages.open(encoding="utf-8")):
        if r["long_extinct"] == "True":
            out["long_extinct"][r["ID"]] = "ASJP long_extinct"
    for r in csv.DictReader(SENS_FILE.open(encoding="utf-8"), delimiter="\t"):
        out.setdefault(r["scenario"], {})[r["id"]] = r["reason"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    args = ap.parse_args()

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    k_cal, distance = summary["k_cal"], summary.get("distance", "pmi")
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    lineages = {l["lineage"]: l for l in summary["lineages"]}
    boot = json.loads((args.out / "bootstrap.json").read_text(encoding="utf-8"))["lineages"]
    metric = get_metric(distance, args.pmi_model)
    docs = load_all(args.cldf, summary.get("tokenization", "asjp"))
    cache = DistCache.load(CACHE_DIR / f'dist_{distance}_{summary.get("tokenization", "asjp")}.npy', metric, list(docs))

    rows = []
    for scenario, removed in scenarios(args.cldf / "languages.csv").items():
        for name, ids in samples.items():
            gone = [i for i in ids if i in removed]
            if not gone:
                continue
            keep = [i for i in ids if i not in removed]
            full_d = cache.matrix([docs[i] for i in ids])
            pos = {i: j for j, i in enumerate(ids)}
            sub = np.array([pos[i] for i in keep], dtype=int)
            out = {}
            for label, k in (("kcal", k_cal), ("k3", K_FINE)):
                z = similarity_matrix(full_d, k)
                full = max_diversity(z)
                if not keep:
                    without = None                    # every doculect of the lineage is removed
                elif len(keep) < 2:
                    without = 1.0
                else:
                    warm = np.zeros(len(keep))
                    w = full.weights[sub]
                    if w.sum() > 0:
                        warm = w / w.sum()
                    without = max_diversity(z[np.ix_(sub, sub)], warm=[warm] if warm.sum() > 0 else ()).value
                out[label] = (full.value, without)
            b = boot.get(name)
            rows.append({
                "lineage": name, "scenario": scenario, "n": len(ids), "n_removed": len(gone), "removed": gone[:12],
                "d_max": out["kcal"][0], "d_max_without": out["kcal"][1],
                "delta": None if out["kcal"][1] is None else out["kcal"][1] - out["kcal"][0],
                "d_max_k3": out["k3"][0], "d_max_k3_without": out["k3"][1],
                "resampling_width": (b["q95"] - b["q05"]) if b else None,
            })
            r = rows[-1]
            if r["d_max_without"] is None:
                print(f'{scenario:12s} {name:26s} n={r["n"]:4d} removed {r["n_removed"]:3d}  no doculects left', flush=True)
            else:
                print(f'{scenario:12s} {name:26s} n={r["n"]:4d} removed {r["n_removed"]:3d}  D_max {r["d_max"]:.3f} → '
                      f'{r["d_max_without"]:.3f} ({r["delta"]:+.3f})  k=3 {r["d_max_k3"]:.2f} → {r["d_max_k3_without"]:.2f}',
                      flush=True)

    rows.sort(key=lambda r: (r["scenario"], 0 if r["delta"] is None else r["delta"]))
    args.out.mkdir(exist_ok=True)
    (args.out / "sensitivity.json").write_text(json.dumps({"k_cal": k_cal, "distance": distance, "rows": rows},
                                                          indent=1, ensure_ascii=False), encoding="utf-8")
    with (args.out / "sensitivity.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "lineage", "n", "n_removed", "d_max", "d_max_without", "delta", "d_max_k3",
                    "d_max_k3_without", "resampling_width"])
        for r in rows:
            gone = r["d_max_without"] is None
            w.writerow([r["scenario"], r["lineage"], r["n"], r["n_removed"], f'{r["d_max"]:.3f}',
                        "" if gone else f'{r["d_max_without"]:.3f}', "" if gone else f'{r["delta"]:+.3f}',
                        f'{r["d_max_k3"]:.2f}', "" if gone else f'{r["d_max_k3_without"]:.2f}',
                        "" if r["resampling_width"] is None else f'{r["resampling_width"]:.3f}'])


if __name__ == "__main__":
    main()
