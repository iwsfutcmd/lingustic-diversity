"""Do our automatic distances agree with expert cognate judgements? An independent check on PMI-LDND and on diversity.

For each cognate-coded Lexibank dataset (git clone --depth 1 https://github.com/lexibank/<name> data/lexibank/<name>;
IE-CoR is lexibank/iecor), languages are matched to ASJP by Glottocode (one per Glottocode), and:

* cognate distance between two languages = share of the concepts both have (loans left out) on which they share no
  expert cognate set; computed on the dataset's whole concept list and on the concepts it shares with ASJP's 40;
* the same pairs get PMI-LDND and LDND from their ASJP lists (median over up to 3 ASJP lists per Glottocode);
* pair level: Spearman correlation of each automatic distance with cognate distance;
* lineage level: for every lineage of the pipeline with at least 5 matched languages, ``spread`` (the scale-free
  diversity of run_analysis.py, in the units of its own distance) from cognate distances and from PMI-LDND over the same
  languages; Spearman correlation across lineages, per dataset and pooled.

Nothing from the datasets is redistributed; results/cognate_check.{json,csv} hold only the derived statistics.

Usage:
    PYTHONPATH=scripts python scripts/cognate_check.py [--datasets iecor abvd ...] [--max-pairs 20000]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from run_analysis import CACHE_DIR, LINEAGES, spread
from isodiv.asjp import core_concepts, load, load_exclusions
from isodiv.distcache import DistCache
from isodiv.metric import get_metric

DATASETS = ["iecor", "abvd", "bowernpny", "sagartst", "kitchensemitic", "savelyevturkic", "leejaponic", "dravlex",
            "grollemundbantu", "utoaztecan"]
MIN_CONCEPTS = 30          # comparable concepts for a cognate distance on the whole list
MIN_CONCEPTS_40 = 20       # ... on the ASJP-40 overlap
MIN_LANGS = 5              # matched languages for a lineage-level comparison
MAX_LINEAGE = 120          # languages per lineage-level comparison (random subset above this)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def cognate_matrices(ds: Path, lang_ids: list[str], keep_concepts: set[str] | None) -> tuple[np.ndarray, np.ndarray]:
    """(distance, comparable-concept count) between the given dataset languages."""
    pos = {l: i for i, l in enumerate(lang_ids)}
    forms = {f["ID"]: f for f in read(ds / "cldf" / "forms.csv") if f["Language_ID"] in pos
             and f.get("Loan", "").lower() != "true" and (keep_concepts is None or f["Parameter_ID"] in keep_concepts)}
    sets: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))       # concept -> lang -> cognate sets
    for c in read(ds / "cldf" / "cognates.csv"):
        f = forms.get(c["Form_ID"])
        if f and c["Cognateset_ID"]:
            sets[f["Parameter_ID"]][pos[f["Language_ID"]]].add(c["Cognateset_ID"])
    n = len(lang_ids)
    shared = np.zeros((n, n)); comparable = np.zeros((n, n))
    for by_lang in sets.values():
        langs = sorted(by_lang)
        ids = {s: k for k, s in enumerate(sorted({s for v in by_lang.values() for s in v}))}
        m = np.zeros((len(langs), len(ids)), dtype=np.float32)
        for r, l in enumerate(langs):
            m[r, [ids[s] for s in by_lang[l]]] = 1
        idx = np.array(langs)
        comparable[np.ix_(idx, idx)] += 1
        shared[np.ix_(idx, idx)] += (m @ m.T) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1 - shared / comparable
    np.fill_diagonal(d, 0)
    return d, comparable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--lexibank", type=Path, default=Path("data/lexibank"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--max-pairs", type=int, default=20000)
    args = ap.parse_args()
    rng = np.random.default_rng(5)

    asjp_rows = {r["ID"]: r for r in read(args.cldf / "languages.csv")}
    excluded = load_exclusions(Path(__file__).resolve().parent.parent / "config" / "exclusions.tsv")
    asjp = {i: x for i, x in load(args.cldf).items() if i not in excluded and len(x.words) >= 20}
    by_glotto: dict[str, list[str]] = defaultdict(list)
    for i in sorted(asjp, key=lambda i: -len(asjp[i].words)):
        by_glotto[asjp[i].glottocode].append(i)
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    lineage_glottos = {n: {asjp_rows[i]["Glottocode"] for i in ids} for n, ids in samples.items()}
    asjp40 = {r["Concepticon_ID"] for r in read(args.cldf / "parameters.csv") if r["ID"] in core_concepts(args.cldf)}
    pmi = get_metric("pmi", args.pmi_model)
    ldnd = get_metric("ldnd")
    cache = DistCache.load(CACHE_DIR / "dist_pmi_asjp.npy", pmi, list(asjp))

    def asjp_distance(metric, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Median over up to 3 ASJP lists per Glottocode, for Glottocode pairs."""
        flat, owner = [], []
        for k, (g, h) in enumerate(pairs):
            for a in by_glotto[g][:3]:
                for b in by_glotto[h][:3]:
                    flat.append((asjp[a], asjp[b])); owner.append(k)
        v = metric.values(flat)
        out = [[] for _ in pairs]
        for k, x in zip(owner, v):
            if x == x:
                out[k].append(x)
        return np.array([np.median(o) if o else np.nan for o in out])

    report = {"datasets": {}, "lineages": []}
    for name in args.datasets:
        ds = args.lexibank / name
        if not (ds / "cldf" / "cognates.csv").exists():
            print(f"{name}: not found, skipped"); continue
        langs = [l for l in read(ds / "cldf" / "languages.csv") if l.get("Glottocode") in by_glotto]
        nforms = defaultdict(int)
        for f in read(ds / "cldf" / "forms.csv"):
            nforms[f["Language_ID"]] += 1
        one = {}
        for l in sorted(langs, key=lambda l: -nforms[l["ID"]]):          # one dataset language per Glottocode
            one.setdefault(l["Glottocode"], l["ID"])
        glottos = sorted(one); lang_ids = [one[g] for g in glottos]; n = len(glottos)
        params = {p["ID"]: p.get("Concepticon_ID", "") for p in read(ds / "cldf" / "parameters.csv")}
        c_all, n_all = cognate_matrices(ds, lang_ids, None)
        c_40, n_40 = cognate_matrices(ds, lang_ids, {p for p, cid in params.items() if cid in asjp40})
        c_all[n_all < MIN_CONCEPTS] = np.nan; c_40[n_40 < MIN_CONCEPTS_40] = np.nan
        np.fill_diagonal(c_all, 0); np.fill_diagonal(c_40, 0)
        pairs = list(combinations(range(n), 2))
        if len(pairs) > args.max_pairs:
            pairs = [pairs[k] for k in rng.choice(len(pairs), args.max_pairs, replace=False)]
        gp = [(glottos[a], glottos[b]) for a, b in pairs]
        d_pmi, d_ldnd = asjp_distance(pmi, gp), asjp_distance(ldnd, gp)
        cog = np.array([c_all[a, b] for a, b in pairs]); cog40 = np.array([c_40[a, b] for a, b in pairs])
        def rho(x, y):
            ok = np.isfinite(x) & np.isfinite(y)
            return (float(spearmanr(x[ok], y[ok])[0]), int(ok.sum())) if ok.sum() > 10 else (None, int(ok.sum()))
        row = {"languages": n, "pairs": len(pairs), "concepts": len(set(params)),
               "rho_pmi": rho(cog, d_pmi), "rho_ldnd": rho(cog, d_ldnd),
               "rho_pmi_40": rho(cog40, d_pmi), "rho_ldnd_40": rho(cog40, d_ldnd)}
        report["datasets"][name] = row
        def f(r):
            return "  n/a" if r[0] is None else f"{r[0]:.3f}"
        print(f'{name:16s} {n:4d} languages, {len(pairs):6d} pairs | Spearman with cognate distance: '
              f'PMI-LDND {f(row["rho_pmi"])}  LDND {f(row["rho_ldnd"])} ({row["rho_pmi"][1]} usable) '
              f'| ASJP-40 concepts only: PMI-LDND {f(row["rho_pmi_40"])}  LDND {f(row["rho_ldnd_40"])} '
              f'({row["rho_pmi_40"][1]} usable)', flush=True)

        # lineage level: diversity from cognates vs from PMI-LDND over the same languages
        for lin, gs in lineage_glottos.items():
            members = [k for k, g in enumerate(glottos) if g in gs]
            if len(members) < MIN_LANGS:
                continue
            if len(members) > MAX_LINEAGE:
                members = sorted(rng.choice(members, MAX_LINEAGE, replace=False).tolist())
            sub = c_all[np.ix_(members, members)]
            keep = np.ones(len(members), bool)
            while np.isnan(sub[np.ix_(keep, keep)]).any():          # languages with too little overlap
                idx = np.flatnonzero(keep); keep[idx[np.isnan(sub[np.ix_(idx, idx)]).sum(1).argmax()]] = False
            members = [m for m, k in zip(members, keep) if k]
            if len(members) < MIN_LANGS:
                continue
            mp = list(combinations(range(len(members)), 2))
            v = asjp_distance(pmi, [(glottos[members[a]], glottos[members[b]]) for a, b in mp])
            dp = np.zeros((len(members), len(members)))
            for (a, b), x in zip(mp, v):
                dp[a, b] = dp[b, a] = x
            if np.isnan(dp).any():
                continue
            dc = c_all[np.ix_(members, members)]
            rec = {"dataset": name, "lineage": lin, "languages": len(members),
                   "spread_cognate": spread(dc), "spread_pmi": spread(dp),
                   "mean_cognate": float(dc[np.triu_indices(len(members), 1)].mean()),
                   "mean_pmi": float(dp[np.triu_indices(len(members), 1)].mean())}
            report["lineages"].append(rec)

    by_ds = defaultdict(list)
    for r in report["lineages"]:
        by_ds[r["dataset"]].append(r)
    print("\nLineage level (spread from cognates vs from PMI-LDND, same languages):")
    for name, rs in by_ds.items():
        if len(rs) >= 4:
            r = spearmanr([x["spread_cognate"] for x in rs], [x["spread_pmi"] for x in rs])[0]
            report["datasets"][name]["rho_lineages"] = (float(r), len(rs))
            print(f'  {name:16s} {len(rs):2d} lineages  Spearman {r:.3f}')
        else:
            print(f'  {name:16s} {len(rs):2d} lineages  (too few for a correlation)')
        for x in sorted(rs, key=lambda x: x["spread_cognate"]):
            print(f'     {x["lineage"]:28s} n={x["languages"]:3d}  spread cognate {x["spread_cognate"]:.3f}  PMI-LDND {x["spread_pmi"]:.3f}')
    rs = report["lineages"]
    if len(rs) >= 4:      # every dataset's lineages together; each dataset's cognate scale differs a little
        r = spearmanr([x["spread_cognate"] for x in rs], [x["spread_pmi"] for x in rs])[0]
        report["pooled_rho_lineages"] = (float(r), len(rs))
        print(f'  all datasets     {len(rs):2d} lineages  Spearman {r:.3f}')
    args.out.mkdir(exist_ok=True)
    (args.out / "cognate_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with (args.out / "cognate_check.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "lineage", "languages", "spread_cognate", "spread_pmi", "mean_cognate", "mean_pmi"])
        for r in report["lineages"]:
            w.writerow([r["dataset"], r["lineage"], r["languages"], f'{r["spread_cognate"]:.3f}', f'{r["spread_pmi"]:.3f}',
                        f'{r["mean_cognate"]:.3f}', f'{r["mean_pmi"]:.3f}'])


if __name__ == "__main__":
    main()
