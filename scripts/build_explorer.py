"""Bundle results/*.json into one self-contained explorer page.

Usage:
    python scripts/build_explorer.py [--results results] [--out results/explorer.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import CORE_LINEAGES, ISOLATE_CANDIDATES  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent.parent / "explorer" / "template.html"


def category(name: str) -> str:
    if name in ISOLATE_CANDIDATES:
        return "iso"
    return "ctl" if name in CORE_LINEAGES else "oth"


def clean(x):
    """JSON has no NaN: map NaN to None."""
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def excluded_by_lineage(excluded: dict[str, str], results: Path) -> dict[str, list[dict]]:
    """Which lineage each excluded ID belongs to, using the lineage predicates on languages.csv."""
    import csv
    from run_analysis import LINEAGES

    cldf = results.parent / "data" / "asjp" / "cldf" / "languages.csv"
    if not excluded or not cldf.exists():
        return {}
    rows = {x["ID"]: x for x in csv.DictReader(cldf.open(encoding="utf-8"))}
    out: dict[str, list[dict]] = {}
    for doc_id, reason in excluded.items():
        for name, pred in LINEAGES.items():
            if doc_id in rows and pred(rows[doc_id]):
                out.setdefault(name, []).append({"id": doc_id, "reason": reason})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/explorer.html"))
    args = ap.parse_args()
    r = args.results

    summary = json.loads((r / "summary.json").read_text(encoding="utf-8"))
    profiles = json.loads((r / "profiles.json").read_text(encoding="utf-8"))
    cross = json.loads((r / "cross_family.json").read_text(encoding="utf-8"))
    boot = json.loads((r / "bootstrap.json").read_text(encoding="utf-8"))["lineages"]

    audit_path = r / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"excluded": {}, "flagged": []}
    names = cross["lineages"]
    matrix = [[clean(v) for v in row] for row in cross["mean_ldnd_matrix"]]
    within = {n: matrix[i][i] for i, n in enumerate(names)}

    lineages = []
    for s in summary["lineages"]:
        n = s["lineage"]
        b = boot.get(n, {})
        lineages.append({
            "name": n, "cat": category(n),
            "n": s["n_doculects"], "glotto": s["n_glottocodes"], "max_ldnd": s["max_ldnd"],
            "spread": s["spread"], "rare3": clean(s["spread_rarefied_m3"]),
            "d_max": s["d_max_kcal"], "uniform": s["uniform_d2_kcal"],
            "lo": b.get("q05"), "hi": b.get("q95"), "within": within.get(n),
            "reps": s["representatives_kcal"][:8],
        })

    pairs = [{
        "a": p["a"], "b": p["b"], "related": p["related_by_construction"],
        **{k: clean(p[k]) for k in ("mean_between", "min_between", "median_between", "mean_within_a",
                                    "mean_within_b", "dmax_union", "dmax_union_k3", "dmax_a_k3", "dmax_b_k3")},
    } for p in cross["pairs"]]
    unrel = [p["mean_between"] for p in pairs if not p["related"] and p["mean_between"] is not None]

    data = {
        "k_cal": summary["k_cal"], "noise": summary["noise_floor_quantiles_10_50_90"],
        "lineages": lineages, "names": names, "matrix": matrix, "pairs": pairs,
        "unrelated": {"mean": float(np.mean(unrel)), "min": float(np.min(unrel)), "max": float(np.max(unrel))},
        "audit": {
            "excluded_by_lineage": excluded_by_lineage(summary.get("excluded", {}), r),
            "flagged": [{k: clean(f[k]) for k in ("lineage", "id", "mean_ldnd", "n_concepts", "loo_delta_dmax")}
                        for f in audit["flagged"]],
        },
        "ks": profiles["ks"], "profiles": profiles["profiles"],
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", blob)
    args.out.write_text(html, encoding="utf-8")
    print(f"{args.out}: {len(html) / 1024:.0f} KB, {len(lineages)} lineages, {len(pairs)} pairs")


if __name__ == "__main__":
    main()
