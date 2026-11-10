"""Is a lineage's one-list deepest branch a real lone branch, a fast-changing member, or a doubtful list?

Usage:
    python scripts/lone_branches.py [--cldf data/asjp/cldf] [--out results]

``depth.py`` reports a lineage whose deepest split is a single list against the rest (``lone_branch``). That list
can be a real old branch with no sampled relatives (Cherokee, Tlingit, Kwadi, Svan), a member whose vocabulary
changed fast (Polabian under German), or a doubtful list. Three independent kinds of evidence:

1. **Classification.** The lineage's node is the deepest Glottolog node shared by all its members; each member's
   *primary branch* is the next node on its path below it (or its own language when nothing comes below).
   ``primary`` = no other member is on the list's primary branch, i.e. Glottolog also makes it a lone branch.
   Otherwise ``sharing`` counts the members on its branch and ``same_language`` those with its Glottocode.
2. **Relative rate.** With an outgroup in range (the rest of the smallest larger lineage, e.g. Baltic for Slavic,
   the rest of Sinitic for Yue), every member of a lineage is the same evolutionary distance from it, so a real
   early branch is no farther from the outgroup than its sisters are. A fast-changing or noisy list is. ``rate_z``
   is the list's median distance to the outgroup in standard deviations above its sisters' (robust: median and MAD).
   Skipped when the outgroup is beyond the detection limit, where distances saturate.
3. **The list itself.** Concept coverage; the distance to another list of the same language in the lineage (beyond
   the same-language band, one of the two is off); and the nearest list outside the lineage, if it is closer than
   the nearest list inside (contact, mixing or a mislabel).

Verdict, in this order: ``doubtful: another family's list is nearer`` (within the same-language band and nearer
than any list of its own lineage: a mislabel or mixing); ``doubtful: disagrees with another list of its language``
(farther from every other list of its language than the same-language band, where those other lists agree with
each other); ``closer to a sibling group`` (a list of a sibling lineage
of the same family is nearer than any of its own: misfiled or transitional); ``relatives missing from the sample``
(its nearest relative is a same-family list in no lineage); ``real lone branch`` (Glottolog primary branch and no
rate excess, z < 2.5); ``fast-changing member`` (nested inside a sampled branch, with rate excess); ``nested, no
rate test`` (nested, no outgroup in range); ``primary branch, faster than sisters`` (a long branch and fast change
both produce it); ``nested, same rate`` (nested yet not faster: Glottolog and the lists disagree).

Writes results/lone_branches.{json,csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from depth import ANCESTORS  # noqa: E402
from run_analysis import load_all, open_world_cache  # noqa: E402

from isodiv.asjp import read_languages
from isodiv.metric import get_metric

RATE_Z = 2.5


def path_of(row: dict) -> list[str]:
    return [p for p in (row.get("classification_glottolog") or "").split(",") if p]


def primary_branch(path: list[str], glottocode: str, depth: int) -> str:
    return path[depth] if len(path) > depth else f"lang:{glottocode}"


def robust_z(x: float, others: np.ndarray) -> float:
    others = others[np.isfinite(others)]
    if len(others) < 3 or not np.isfinite(x):
        return float("nan")
    med = np.median(others)
    mad = 1.4826 * np.median(np.abs(others - med))
    return float((x - med) / max(mad, 0.02))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    band = summary["noise_floor_quantiles_10_50_90"][2]          # 90th percentile of same-language distances
    horizon = summary["kernel"]["unrelated"]
    samples = {n: [i for i in v if i not in ANCESTORS.get(n, ())]
               for n, v in json.loads((args.out / "samples.json").read_text(encoding="utf-8")).items()}
    depth = {r["lineage"]: r for r in json.loads((args.out / "depth.json").read_text(encoding="utf-8"))["lineages"]}
    rows = {r["ID"]: r for r in read_languages(args.cldf)}
    docs = load_all(args.cldf)
    cache, _ = open_world_cache(get_metric("pmi", args.pmi_model), args.cldf)
    world = [x for x in docs.values()]

    out = []
    for name, dr in depth.items():
        x_id = dr.get("lone_branch")
        if not x_id or dr["lists"] <= 2:
            continue
        members = [i for i in samples[name] if i in docs]
        sisters = [i for i in members if i != x_id]
        paths = {i: path_of(rows[i]) for i in members}
        lca = 0
        while all(len(p) > lca for p in paths.values()) and len({p[lca] for p in paths.values()}) == 1:
            lca += 1
        br = {i: primary_branch(paths[i], rows[i]["Glottocode"], lca) for i in members}
        sharing = [i for i in sisters if br[i] == br[x_id]]
        same_lang = [i for i in sisters if rows[i]["Glottocode"] == rows[x_id]["Glottocode"]]
        x = docs[x_id]

        # relative rate against the rest of the smallest larger lineage
        mset = set(members)
        parents = [n for n, v in samples.items() if n != name and mset < set(v)]
        rate_z, out_name, d_out_x, d_out_sis = float("nan"), "", float("nan"), float("nan")
        if parents:
            out_name = min(parents, key=lambda n: len(samples[n]))
            og = [docs[i] for i in samples[out_name] if i not in mset and i in docs]
            if og:
                sis = [docs[i] for i in sisters]
                m = cache.matrix([x, *sis, *og])
                k = 1 + len(sis)
                med_og = np.nanmedian(m[:k, k:], axis=1)
                d_out_x, d_out_sis = float(med_og[0]), float(np.nanmedian(med_og[1:]))
                if d_out_sis < horizon:
                    rate_z = robust_z(med_og[0], med_og[1:])

        # the list itself
        inside = cache.matrix([x, *[docs[i] for i in sisters]])[0, 1:]
        near_in = float(np.nanmin(inside)) if np.isfinite(inside).any() else float("nan")
        same_d = [float(inside[sisters.index(i)]) for i in same_lang]
        # the same-language test only means something where the language's other lists agree with each other (a
        # Chinese dialect-group Glottocode such as Jin or Gan covers varieties farther apart than that)
        coherent = False
        if len(same_lang) >= 2:
            sm = cache.matrix([docs[i] for i in same_lang])
            coherent = float(np.nanmedian(sm[np.triu_indices(len(same_lang), 1)])) <= band
        elif len(same_lang) == 1:
            coherent = True
        fam = x.family
        outside = [y for y in world if y.id not in mset]
        vo = cache.values([(x, y) for y in outside])

        def nearest(pred):
            ix = [k for k, y in enumerate(outside) if pred(y) and np.isfinite(vo[k])]
            if not ix:
                return float("nan"), ""
            k = min(ix, key=lambda k: vo[k])
            return float(vo[k]), outside[k].id
        near_other_fam, near_other_fam_id = nearest(lambda y: y.family != fam)
        near_same_fam, near_same_fam_id = nearest(lambda y: y.family == fam)
        # where the nearest same-family outsider lives: a sibling lineage of this family, or no lineage at all
        sib_of = ""
        if near_same_fam_id:
            holders = [n for n, v in samples.items() if near_same_fam_id in v and not (set(v) & mset)]
            sib_of = min(holders, key=lambda n: len(samples[n])) if holders else "(no lineage)"

        primary = not sharing
        rate_excess = np.isfinite(rate_z) and rate_z >= RATE_Z
        if near_other_fam < min(near_in, band):
            verdict = "doubtful: another family's list is nearer"
        elif same_d and min(same_d) > band and coherent:
            verdict = "doubtful: disagrees with another list of its language"
        elif np.isfinite(near_same_fam) and near_same_fam < near_in - 0.02:
            verdict = ("relatives missing from the sample" if sib_of == "(no lineage)"
                       else "closer to a sibling group (misfiled or transitional)")
        elif primary and not rate_excess:
            verdict = "real lone branch"
        elif not primary and rate_excess:
            verdict = "fast-changing member"
        elif not primary and not np.isfinite(rate_z):
            verdict = "nested, no rate test"
        elif primary:
            verdict = "primary branch, faster than sisters"
        else:
            verdict = "nested, same rate"
        near_out, near_out_id = near_other_fam, near_other_fam_id
        out.append({
            "lineage": name, "lone_list": x_id, "language": rows[x_id].get("Glottolog_Name", ""),
            "retention": dr["retention"], "retention_2": dr["retention_2"],
            "glottolog_primary": primary, "branch": br[x_id], "members_on_branch": len(sharing),
            "same_language_lists": len(same_lang),
            "min_d_same_language": min(same_d) if same_d else float("nan"),
            "concepts": len(x.words), "outgroup": out_name, "d_outgroup": d_out_x, "d_outgroup_sisters": d_out_sis,
            "rate_z": rate_z, "nearest_inside": near_in,
            "nearest_same_family_outside": near_same_fam, "nearest_same_family_outside_id": near_same_fam_id,
            "its_lineage": sib_of, "nearest_other_family": near_out, "nearest_other_family_id": near_out_id,
            "verdict": verdict,
        })
        print(f"{name:26} {x_id:28} {verdict:30} primary={primary!s:5} on-branch={len(sharing):3} "
              f"rate_z={rate_z:6.2f} ({out_name}) cov={len(x.words)} in={near_in:.3f} "
              f"famout={near_same_fam:.3f} {near_same_fam_id} [{sib_of}] other={near_out:.3f}", flush=True)

    out.sort(key=lambda r: (r["verdict"], r["lineage"]))
    (args.out / "lone_branches.json").write_text(json.dumps({"rate_z": RATE_Z, "band": band, "rows": out},
                                                            ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "lone_branches.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
