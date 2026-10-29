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
from run_analysis import CORE_LINEAGES, EXCLUSIONS_FILE, ISOLATE_CANDIDATES, SUBFAMILIES  # noqa: E402

from isodiv.asjp import load_exclusions

TEMPLATE = Path(__file__).resolve().parent.parent / "explorer" / "template.html"
# "LDND" in the template is replaced by the distance actually used (summary.json "distance_label").
METHOD = {
    "LDND": "LDND on ASJP 40-item lists (Wichmann et al. 2010): Levenshtein distance normalized by word length, "
            "divided by the same measure for different meanings to remove chance resemblance. About 0 for identical "
            "lists and about 1 for unrelated languages.",
    "PMI-LDND": "PMI-LDND on ASJP 40-item lists: LDND (Wichmann et al. 2010) with word distances from alignments "
                "scored by learned sound correspondences (pointwise mutual information, after Jäger 2013, 2018) "
                "instead of a flat cost per changed segment, so regular correspondences such as Japonic h : p count as "
                "close. Divided by the same measure for different meanings to remove chance resemblance. About 0 for "
                "identical lists and about 1 for unrelated languages.",
}


def category(name: str) -> str:
    if name in ISOLATE_CANDIDATES:
        return "iso"
    if name in SUBFAMILIES:
        return "sub"
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
            "name": n, "cat": category(n), "family": s.get("family", ""),
            "n": s["n_doculects"], "glotto": s["n_glottocodes"], "max_ldnd": s["max_ldnd"],
            "spread": s["spread"], "rare3": clean(s["spread_rarefied_m3"]),
            "d_max": s["d_max_kcal"], "uniform": s["uniform_d2_kcal"],
            # The bootstrap samples at most 40 doculects (the main run up to 80), so its absolute level sits
            # lower for big lineages. Keep its width (concept-resampling spread) and centre it on the estimate.
            "lo": s["d_max_kcal"] + (b["q05"] - b["median"]) if b else None,
            "hi": s["d_max_kcal"] + (b["q95"] - b["median"]) if b else None,
            "within": within.get(n),
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
            "excluded_by_lineage": excluded_by_lineage(load_exclusions(EXCLUSIONS_FILE), r),
            "flagged": [{k: clean(f[k]) for k in ("lineage", "id", "mean_ldnd", "n_concepts", "loo_delta_dmax")}
                        for f in audit["flagged"]],
        },
        "ks": profiles["ks"], "profiles": profiles["profiles"],
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    label = summary.get("distance_label", "LDND")
    html = (TEMPLATE.read_text(encoding="utf-8").replace("LDND", label)
            .replace("<!--__METHOD__-->", METHOD[label]).replace("/*__DATA__*/", blob))
    args.out.write_text(html, encoding="utf-8")
    print(f"{args.out}: {len(html) / 1024:.0f} KB, {len(lineages)} lineages, {len(pairs)} pairs")


if __name__ == "__main__":
    main()
