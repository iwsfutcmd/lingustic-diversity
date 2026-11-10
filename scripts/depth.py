"""Depth: how closely related two varieties are, and how deep a family's deepest split is, in one unit.

Usage:
    python scripts/depth.py [--cldf data/asjp/cldf] [--lexibank data/lexibank] [--out results]

**Pairwise.** PMI-LDND on the 40-item lists is mapped to *retention*: the share of basic vocabulary the two varieties
still have as cognates, relative to two lists of the same language. The map is fitted on expert cognate coding
(the Lexibank datasets of ``cognate_check.py``): for each dataset and distance bin, the median log cognate share;
datasets differ in how strictly they draw cognate sets, which shifts their shares by a constant factor, so each gets
an offset and one common curve is fitted (alternating medians), then made monotone. Retention is 1 at the
same-language median distance. **Depth** is ``τ = −ln retention``: additive along a tree if replacement in the basic
vocabulary is a Poisson process, with no rate and no dates assumed (0.69 = half the basic vocabulary replaced along
the path between the two, 1.39 = three quarters). Distances at or beyond the detection limit (the kernel's
``unrelated`` anchor) are past the 40-word horizon: depth there is reported as a lower bound.

**Family.** The depth of a family's deepest split: average-linkage (UPGMA) clustering on pairwise depth, using no
classification; the root splits the family into two branches, and the family depth is the median pairwise depth
across that split. Dialect counts do not raise it (extra lists of one branch sit inside that branch), and more
sampling only raises it when it finds a deeper branch. Lists flagged by the contact check (closer to another family
than to their own, results/contact.json) are left out, so one mislabelled list cannot set a family's depth. A family
whose median across the root is at or beyond the detection limit is marked ``beyond_horizon``: its depth is at least
the horizon depth and 40 words cannot say more. The root branch can be a single list: often a real lone branch
(Eyak, Kwadi, Svan, Cherokee), sometimes a fast-changing or noisy list; ``depth_2`` is the deepest split with at least
two lists on each side, and ``lone_branch`` names the single list when there is one.

Writes results/depth_calibration.json, results/depth.{json,csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cognate_check import DATASETS, MIN_CONCEPTS, cognate_matrices, read  # noqa: E402
from run_analysis import drop_nan, load_all, open_world_cache  # noqa: E402

from isodiv.asjp import load, load_exclusions
from isodiv.metric import get_metric

# ancestors of (nearly) the whole lineage: an ancestor is not a branch, so it is left out of that lineage's depth (it
# stays in its D_max and in higher lineages, where it is a member of one branch). Ancient lists that are branches
# (Gothic, Hittite, Akkadian, Avestan, Ge'ez, Old Church Slavonic) are kept.
ANCESTORS = {
    "Sinitic": {"OLD_CHINESE", "MIDDLE_CHINESE"},
    "Germanic": {"PROTO_GERMANIC"},
    "Romance": {"LATIN", "LATIN_2"},
    "Indo-Aryan": {"SANSKRIT"},
    "Arabic": {"CLASSICAL_ARABIC", "ARABIC_QURANIC", "STANDARD_ARABIC", "STANDARD_ARABIC_2"},
    "Aramaic": {"ARAMAIC_ANCIENT"},
}
FIT_EXCLUDE = {"bowernpny", "leejaponic"}     # Pama-Nyungan coding differs in shape, not just level; Japonic: 3 langs
EDGES = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.1])
MIN_BIN = 8


# ---- calibration --------------------------------------------------------------------------------------------------

def calibration_pairs(cldf: Path, lexibank: Path, cache, max_pairs: int, rng) -> list[tuple[str, float, float]]:
    """(dataset, PMI-LDND, cognate share on the dataset's whole list) for Glottocode pairs matched to ASJP."""
    excl = load_exclusions(Path(__file__).resolve().parent.parent / "config" / "exclusions.tsv")
    asjp = {i: x for i, x in load(cldf).items() if i not in excl and len(x.words) >= 20}
    byg = defaultdict(list)
    for i in sorted(asjp, key=lambda i: -len(asjp[i].words)):
        byg[asjp[i].glottocode].append(i)
    out = []
    for name in DATASETS:
        ds = lexibank / name
        if not (ds / "cldf" / "cognates.csv").exists():
            continue
        langs = [l for l in read(ds / "cldf" / "languages.csv") if l.get("Glottocode") in byg]
        nf = defaultdict(int)
        for f in read(ds / "cldf" / "forms.csv"):
            nf[f["Language_ID"]] += 1
        one = {}
        for l in sorted(langs, key=lambda l: -nf[l["ID"]]):
            one.setdefault(l["Glottocode"], l["ID"])
        gl = sorted(one)
        c, n = cognate_matrices(ds, [one[g] for g in gl], None)
        c[n < MIN_CONCEPTS] = np.nan
        pairs = list(combinations(range(len(gl)), 2))
        if len(pairs) > max_pairs:
            pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
        flat, own = [], []
        for k, (a, b) in enumerate(pairs):
            for x in byg[gl[a]][:3]:
                for y in byg[gl[b]][:3]:
                    flat.append((asjp[x], asjp[y])); own.append(k)
        acc = defaultdict(list)
        for k, v in zip(own, cache.values(flat)):
            if v == v:
                acc[k].append(v)
        for k, (a, b) in enumerate(pairs):
            if acc[k] and np.isfinite(c[a, b]) and c[a, b] < 1:
                out.append((name, float(np.median(acc[k])), float(1 - c[a, b])))
    return out


def fit_curve(pairs, same: float) -> dict:
    """Common log-retention curve over distance bins, with a level offset per dataset."""
    names = sorted({p[0] for p in pairs if p[0] not in FIT_EXCLUDE})
    mids = (EDGES[1:] + EDGES[:-1]) / 2

    def binned(ds):
        v = [[np.log(p[2]) for p in pairs if p[0] == ds and lo <= p[1] < hi] for lo, hi in zip(EDGES, EDGES[1:])]
        return np.array([np.median(x) if len(x) >= MIN_BIN else np.nan for x in v])

    t = np.array([binned(n) for n in names])
    a = np.zeros(len(names))
    for _ in range(100):
        g = np.nanmedian(t - a[:, None], axis=0)
        a = np.nanmedian(t - g[None, :], axis=1)
    ok = np.isfinite(g)
    x, g = mids[ok], g[ok]
    g = np.minimum.accumulate(g)                         # retention never rises with distance
    g0 = np.interp(same, x, g)                           # retention 1 at the same-language median distance
    tau = np.maximum(-(g - g0), 0.0)
    resid = {n: [float(v) if np.isfinite(v) else None for v in (t[i] - a[i])[ok] - (g + 0)]
             for i, n in enumerate(names)}
    return {"d": x.tolist(), "tau": tau.tolist(), "offsets": dict(zip(names, map(float, a))),
            "residuals": resid, "same": same, "excluded_from_fit": sorted(FIT_EXCLUDE)}


def depth_of(d, cal: dict) -> np.ndarray:
    """Pairwise depth τ for distances d (beyond the last bin: the last bin's depth, i.e. a lower bound)."""
    return np.interp(np.asarray(d, dtype=float), cal["d"], cal["tau"])


# ---- family depth -------------------------------------------------------------------------------------------------

def family_depth(d: np.ndarray, cal: dict, horizon: float) -> dict:
    """Deepest split by UPGMA on pairwise depth; median depth across it."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(d)
    if n < 2:
        return {"depth": 0.0, "retention": 1.0, "median_d": 0.0, "beyond_horizon": False, "split": [n, 0],
                "supported": None}
    t = depth_of(d, cal)
    np.fill_diagonal(t, 0)
    z = linkage(squareform(t, checks=False), method="average")
    lab = fcluster(z, 2, criterion="maxclust")
    a, b = np.flatnonzero(lab == 1), np.flatnonzero(lab == 2)
    cross_d = d[np.ix_(a, b)].ravel()
    md = float(np.median(cross_d))
    tau = float(np.median(t[np.ix_(a, b)]))
    # the deepest split with at least two lists on each side: walk down from the root past one-list branches
    from scipy.cluster.hierarchy import to_tree
    node = to_tree(z)
    while node is not None and not node.is_leaf():
        l, r = node.get_left(), node.get_right()
        if l.get_count() >= 2 and r.get_count() >= 2:
            break
        node = l if l.get_count() >= r.get_count() else r
    sup = None
    if node is not None and not node.is_leaf() and node.get_left().get_count() >= 2:
        la, lb = np.array(node.get_left().pre_order()), np.array(node.get_right().pre_order())
        sup = {"depth": float(np.median(t[np.ix_(la, lb)])), "median_d": float(np.median(d[np.ix_(la, lb)])),
               "split": sorted([len(la), len(lb)], reverse=True)}
    return {"depth": tau, "retention": float(np.exp(-tau)), "median_d": md, "beyond_horizon": md >= horizon,
            "split": sorted([len(a), len(b)], reverse=True), "_a": a, "_b": b, "supported": sup}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--lexibank", type=Path, default=Path("data/lexibank"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--max-pairs", type=int, default=6000)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    rng = np.random.default_rng(5)

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    same, horizon = summary["kernel"]["same"], summary["kernel"]["unrelated"]
    metric = get_metric("pmi", args.pmi_model)
    cache, _ = open_world_cache(metric, args.cldf)

    pairs = calibration_pairs(args.cldf, args.lexibank, cache, args.max_pairs, rng)
    cal = fit_curve(pairs, same)
    cal["horizon_d"] = horizon
    cal["horizon_depth"] = float(depth_of(horizon, cal))
    cal["pairs_per_dataset"] = {n: sum(1 for p in pairs if p[0] == n) for n in sorted({p[0] for p in pairs})}
    (args.out / "depth_calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print("depth curve:", ", ".join(f"d {x:.3f} → τ {y:.2f} ({np.exp(-y):.0%})" for x, y in zip(cal["d"], cal["tau"])))
    print(f"horizon: d {horizon:.3f} → τ ≥ {cal['horizon_depth']:.2f} (retention ≤ {np.exp(-cal['horizon_depth']):.0%})")

    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    # drop only lists *close* to another family (within the same-language band): a mislabelled or mixed list. A lone
    # deep witness whose nearest outsider is merely noise at the horizon (Cherokee) is a real branch and stays.
    flagged = set()
    contact = args.out / "contact.json"
    if contact.exists():
        cj = json.loads(contact.read_text(encoding="utf-8"))
        flagged = {r["id"] for r in cj["checked"] if r.get("flagged") and r["nearest_out"] <= cj["band"]}
    docs = load_all(args.cldf)
    rows = []
    for name, ids in samples.items():
        ds = [docs[i] for i in ids if i in docs and i not in flagged and i not in ANCESTORS.get(name, ())]
        d = cache.matrix(ds)
        ds, d = drop_nan(ds, d)
        r = family_depth(d, cal, horizon)
        a, b = r.get("_a", []), r.get("_b", [])
        if len(a) < len(b):
            a, b = b, a
        rows.append({"lineage": name, "lists": len(ds), "depth": r["depth"], "retention": r["retention"],
                     "median_d_across_root": r["median_d"], "beyond_horizon": r["beyond_horizon"],
                     "root_split_sizes": r["split"], "lone_branch": ds[b[0]].id if len(b) == 1 else "",
                     # deepest split with >= 2 lists on each side (equals the above unless the root branch is one list)
                     "depth_2": (r["supported"] or {}).get("depth", float("nan")),
                     "retention_2": float(np.exp(-(r["supported"] or {}).get("depth", float("nan")))),
                     "beyond_horizon_2": bool((r["supported"] or {}).get("median_d", 0) >= horizon),
                     "smaller_branch": sorted(ds[i].id for i in b)[:6]})
        print(f"{name:26} n={len(ds):5} depth {r['depth']:.2f} retention {r['retention']:.0%} "
              f"(d {r['median_d']:.3f}{', beyond horizon' if r['beyond_horizon'] else ''}) split {r['split']} "
              f"{sorted(ds[i].id for i in b)[:4]}",
              flush=True)

    rows.sort(key=lambda r: r["depth"])
    (args.out / "depth.json").write_text(json.dumps({"calibration": "depth_calibration.json", "horizon_d": horizon,
                                                     "horizon_depth": cal["horizon_depth"], "lineages": rows},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "depth.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else (";".join(map(str, v)) if isinstance(v, list) else v))
                        for k, v in r.items()})


if __name__ == "__main__":
    main()
