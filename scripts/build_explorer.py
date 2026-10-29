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
from run_analysis import CORE_LINEAGES, EXCLUSIONS_FILE, ISOLATE_CANDIDATES, NODE_GROUPS, SINITIC_GROUPS, SUBFAMILIES  # noqa: E402

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
    if name in SUBFAMILIES or name in SINITIC_GROUPS or name in NODE_GROUPS:
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

    single_path = r / "single_lists.json"
    single = json.loads(single_path.read_text(encoding="utf-8")) if single_path.exists() else {}

    def single_note(name: str) -> str:
        note = "One list only: internal diversity cannot be measured (D_max is 1 by definition)."
        x, null = single.get(name), single.get("_null_nearest")
        if x and null:
            near = ", ".join(f"{n} {a:.3f}" for n, a, _ in x["nearest"][:3])
            best = x["nearest"][0][1]
            note += (f" Nearest unrelated lineages (mean distance): {near}. For a random list from a known family the"
                     f" nearest unrelated lineage sits at {null['median']:.3f} ({null['q05']:.3f}–{null['q95']:.3f}, 5–95%),"
                     + (" so this is within chance." if best >= null["q05"] else
                        " so the closest match is in the bottom 5%, which one of several isolates tested is expected to"
                        " reach by chance."))
            if x.get("dravidian_rank"):
                note += (f" Dravidian: {x['dravidian'][0]:.3f}, rank {x['dravidian_rank'][0]} of"
                         f" {x['dravidian_rank'][1]} (Elamo-Dravidian would predict it near the top).")
        return note

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
            **({"note": single_note(n)} if s["n_doculects"] == 1 else {}),
        })

    adv_path = r / "adversarial.json"
    profiles_all = dict(profiles["profiles"])
    if adv_path.exists():
        adv = json.loads(adv_path.read_text(encoding="utf-8"))
        for n, x in adv["random_unrelated"].items():
            name = f"Random unrelated, n={n}"
            lineages.append({
                "name": name, "cat": "adv", "family": "", "n": int(n), "glotto": int(n), "max_ldnd": None,
                "spread": None, "rare3": None, "d_max": x["d_max"]["median"], "uniform": x["uniform"],
                "lo": x["d_max"]["q05"], "hi": x["d_max"]["q95"], "range_label": "5–95% over random draws",
                "d_max_k3": x["d_max_k3"]["median"], "within": x["mean_within"], "reps": x["example"],
                "note": f"{n} doculects, each from a different family. Median of the draws; representatives are from one draw.",
            })
            profiles_all[name] = {"d_max": x["profile_d_max"], "uniform": x["profile_uniform"]}
        for n, x in adv["proposals"].items():
            bl = adv["matched_baseline"][str(len(x["members"]))]
            sb = adv.get("shallow_baseline", {}).get(str(len(x["members"])))
            name = f"{n} (constructed)"
            lineages.append({
                "name": name, "cat": "adv", "family": "", "n": x["n"], "glotto": None, "max_ldnd": x["max_within"],
                "spread": None, "rare3": None, "d_max": x["d_max"], "uniform": x["uniform"], "lo": None, "hi": None,
                "d_max_k3": x["d_max_k3"], "within": x["mean_within"], "reps": x["reps"],
                "note": f"Union of {', '.join(x['members'])}. Random unions of {len(x['members'])} unrelated families "
                        f"score {bl['d_max']['median']:.3f} ({bl['d_max']['q05']:.3f}–{bl['d_max']['q95']:.3f})"
                        + (f"; unions of {len(x['members'])} shallow (genus-level) unrelated families score "
                           f"{sb['d_max']['median']:.3f} ({sb['d_max']['q05']:.3f}–{sb['d_max']['q95']:.3f})" if sb else "") + ".",
            })
            profiles_all[name] = {"d_max": x["profile_d_max"], "uniform": x["profile_uniform"]}

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
        "ks": profiles["ks"], "profiles": profiles_all,
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    label = summary.get("distance_label", "LDND")
    html = (TEMPLATE.read_text(encoding="utf-8").replace("LDND", label)
            .replace("<!--__METHOD__-->", METHOD[label]).replace("/*__DATA__*/", blob))
    args.out.write_text(html, encoding="utf-8")
    print(f"{args.out}: {len(html) / 1024:.0f} KB, {len(lineages)} lineages, {len(pairs)} pairs")


if __name__ == "__main__":
    main()
