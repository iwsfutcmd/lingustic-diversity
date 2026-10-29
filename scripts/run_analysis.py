"""Diversity profiles for isolate candidates and control lineages from ASJP.

Usage:
    python scripts/run_analysis.py [--cldf data/asjp/cldf] [--out results] [--tokens asjp|segments]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import re
from itertools import combinations

import numpy as np
from scipy.optimize import brentq

from isodiv.asjp import Doculect, load, load_exclusions
from isodiv.distance import ldnd, ldnd_matrix
from isodiv.diversity import max_diversity, similarity_matrix, uniform_diversity

# lineage label -> predicate on a languages.csv row
IE_GENUS = lambda g: (lambda r: r["classification_wals"] == f"IE.{g}")
LINEAGES: dict[str, callable] = {
    # isolate candidates (top-level in ASJP's Glottolog-based Family field)
    "Basque": lambda r: r["Family"] == "Basque",
    "Ainu": lambda r: r["Family"] == "Ainu",
    "Japonic": lambda r: r["Family"] == "Japonic",
    "Koreanic": lambda r: r["Family"] == "Koreanic",
    "Burushaski": lambda r: r["Family"] == "Burushaski",
    "Nivkh": lambda r: r["Family"] == "Nivkh",
    "Haida": lambda r: r["Family"] == "Haida",
    "Yeniseian": lambda r: r["Family"] == "Yeniseian",
    "Kartvelian": lambda r: r["Family"] == "Kartvelian",
    # controls: single genera, then deep families
    "Romance": IE_GENUS("ROMANCE"),
    "Germanic": IE_GENUS("GERMANIC"),
    "Slavic": IE_GENUS("SLAVIC"),
    "Turkic": lambda r: r["Family"] == "Turkic",
    "Mongolic": lambda r: r["Family"] == "Mongolic-Khitan",
    "Uralic": lambda r: r["Family"] == "Uralic",
    "Indo-European": lambda r: r["Family"] == "Indo-European",
}
ISOLATE_CANDIDATES = list(LINEAGES)[:9]
CORE_LINEAGES = list(LINEAGES)  # the original 16; everything added below is an extra comparison family
# More ASJP top-level families (Glottolog-based Family column), small and large, for the explorer.
EXTRA_FAMILIES = [
    "Chukotko-Kamchatkan", "Yukaghir", "Eskimo-Aleut", "Tungusic", "Abkhaz-Adyge",
    "Nakh-Daghestanian", "Dravidian", "Great Andamanese", "Khoe-Kwadi", "Tuu", "Kxa", "Kuliak",
    "Tarascan", "Araucanian", "Quechuan", "Mayan", "Algic", "Salishan", "Iroquoian", "Siouan",
    "Athabaskan-Eyak-Tlingit", "Uto-Aztecan", "Hmong-Mien", "Tai-Kadai", "Austroasiatic",
    "Pama-Nyungan", "Austronesian", "Sino-Tibetan", "Afro-Asiatic", "Atlantic-Congo",
    "Nuclear Trans New Guinea", "Otomanguean",
]
for _fam in EXTRA_FAMILIES:
    LINEAGES[_fam] = (lambda f: lambda r: r["Family"] == f)(_fam)
MAX_PER_LINEAGE = 80          # stratified cap keeps pairwise LDND tractable
KS = np.geomspace(0.1, 40, 48)
RAREFY_M = 3          # compare lineages at a fixed number of distinct-Glottocode doculects
RAREFY_DRAWS = 40


EXCLUSIONS_FILE = Path(__file__).resolve().parent.parent / "config" / "exclusions.tsv"


def load_all(cldf: Path, tokenization: str = "asjp", wold: Path | None = None) -> dict[str, Doculect]:
    """All doculects of every configured lineage, minus those in config/exclusions.tsv; with ``wold``
    (a WOLD CLDF directory) the words WOLD marks as loans are dropped for the languages it covers."""
    mask = None
    if wold is not None:
        from isodiv.wold import loan_table
        mask = {i: t.borrowed for i, t in loan_table(wold, cldf).items()}
    return load(cldf, keep=lambda r: any(p(r) for p in LINEAGES.values()), tokenization=tokenization,
                exclude=load_exclusions(EXCLUSIONS_FILE), mask=mask)


def stratified(docs: list[Doculect], cap: int, seed: int = 0) -> list[Doculect]:
    """Cap a lineage's size without losing breadth: round-robin over WALS genus/glottocode."""
    if len(docs) <= cap:
        return docs
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[Doculect]] = defaultdict(list)
    for doc in docs:
        buckets[doc.wals + "|" + doc.glottocode].append(doc)
    pools = [list(rng.permutation(b)) for b in buckets.values()]
    out: list[Doculect] = []
    while len(out) < cap:
        for pool in pools:
            if pool and len(out) < cap:
                out.append(pool.pop())
    return out


def drop_nan(docs: list[Doculect], d: np.ndarray) -> tuple[list[Doculect], np.ndarray]:
    keep = np.ones(len(docs), dtype=bool)
    while np.isnan(d[np.ix_(keep, keep)]).any():
        sub = d[np.ix_(keep, keep)]
        worst = np.flatnonzero(keep)[np.isnan(sub).sum(axis=1).argmax()]
        keep[worst] = False
    return [x for x, k in zip(docs, keep) if k], d[np.ix_(keep, keep)]


def spread(d: np.ndarray, level: float = 1.5) -> float:
    """Clock-free 'diversity radius' in LDND units.

    k* = the scale at which D_max first reaches `level` effective lineages;
    spread = ln 3 / k*. For a two-doculect lineage at distance δ this is exactly δ,
    so it reads like a distance — but it is a property of the whole similarity
    space, robust to how many dialects were sampled. A singleton has spread 0.
    """
    if d.shape[0] < 2 or np.nanmax(d) == 0:
        return 0.0
    f = lambda logk: max_diversity(similarity_matrix(d, np.exp(logk))).value - level
    lo, hi = np.log(1e-3), np.log(1e4)
    if f(hi) < 0:
        return 0.0
    return float(np.log(3) / np.exp(brentq(f, lo, hi, xtol=1e-4)))


def noise_floor(cldf: Path, n_pairs: int = 400, seed: int = 0) -> np.ndarray:
    """LDND between doculects of the *same* language (same Glottocode, names X / X_2):
    differences here are transcription/source/idiolect noise, not lineage diversity."""
    rows = list(csv.DictReader((cldf / "languages.csv").open(encoding="utf-8")))
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        groups[(re.sub(r"_\d+$", "", r["ID"]), r["Glottocode"])].append(r["ID"])
    pairs = [g[:2] for g in groups.values() if len(g) >= 2]
    rng = np.random.default_rng(seed)
    pairs = [pairs[i] for i in rng.choice(len(pairs), size=min(n_pairs, len(pairs)), replace=False)]
    ids = {i for p in pairs for i in p}
    docs = load(cldf, keep=lambda r: r["ID"] in ids)
    v = np.array([ldnd(docs[a], docs[b]) for a, b in pairs if a in docs and b in docs])
    return v[~np.isnan(v)]


def rarefied_spread(docs: list[Doculect], d: np.ndarray, m: int = RAREFY_M,
                    draws: int = RAREFY_DRAWS, seed: int = 0) -> float:
    """Mean spread over random m-subsets with m distinct Glottocodes (fixed sample size)."""
    by_g: dict[str, list[int]] = defaultdict(list)
    for i, x in enumerate(docs):
        by_g[x.glottocode].append(i)
    if len(by_g) < m:
        return float("nan")
    rng = np.random.default_rng(seed)
    gs = list(by_g)
    combos = list(combinations(range(len(gs)), m))
    picks = [combos[i] for i in rng.choice(len(combos), size=min(draws, len(combos)), replace=False)]
    vals = []
    for c in picks:
        idx = np.array([rng.choice(by_g[gs[j]]) for j in c])
        vals.append(spread(d[np.ix_(idx, idx)]))
    return float(np.mean(vals))


def analyse(name: str, docs: list[Doculect], k_cal: float) -> dict:
    d = ldnd_matrix(docs)
    docs, d = drop_nan(docs, d)
    zs = [similarity_matrix(d, k) for k in KS]
    maxes = [max_diversity(z) for z in zs]
    k_ref = k_cal
    m_ref = max_diversity(similarity_matrix(d, k_ref))
    return {
        "lineage": name,
        "n_doculects": len(docs),
        "n_glottocodes": len({x.glottocode for x in docs}),
        "max_ldnd": float(d.max()) if len(docs) > 1 else 0.0,
        "spread": spread(d),
        "spread_rarefied_m3": rarefied_spread(docs, d),
        "d_max_kcal": m_ref.value,
        "uniform_d2_kcal": uniform_diversity(similarity_matrix(d, k_ref)),
        "representatives_kcal": sorted(
            ((docs[i].id, round(float(w), 3)) for i, w in enumerate(m_ref.weights) if w > 1e-3),
            key=lambda t: -t[1],
        ),
        "profile_d_max": [m.value for m in maxes],
        "profile_uniform": [uniform_diversity(z) for z in zs],
        "kkt_all_ok": all(m.kkt_ok for m in maxes),
        "_d": d,
        "_docs": docs,
    }


def sampling_robustness(docs: list[Doculect], k_cal: float) -> list[dict]:
    """Same lineage, different 'dialect counting' choices."""
    by_glotto: dict[str, Doculect] = {}
    for x in docs:
        by_glotto.setdefault(x.glottocode, x)
    variants = {
        "all doculects": docs,
        "one per Glottocode": list(by_glotto.values()),
        "Tokyo + Shuri only": [x for x in docs if x.id in {"TOKYO_JAPANESE", "SHURI"}],
        "all + 3x duplicated Tokyo": docs + [x for x in docs if x.id == "TOKYO_JAPANESE"] * 3,
    }
    out = []
    for label, sub in variants.items():
        d = ldnd_matrix(sub)
        sub, d = drop_nan(sub, d)
        z = similarity_matrix(d, k_cal)
        out.append({
            "sample": label,
            "n": len(sub),
            "d_max_kcal": max_diversity(z).value,
            "uniform_d2_kcal": uniform_diversity(z),
            "spread": spread(d),
        })
    return out


def plot(results: list[dict], robust: list[dict], out: Path, k_cal: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Noto fallbacks so any non-Latin labels added later render (no bidi/shaping in mpl).
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    wanted = ["DejaVu Sans", "Noto Sans", "Noto Sans CJK JP", "Noto Sans Arabic", "Noto Sans Hebrew"]
    plt.rcParams["font.family"] = [f for f in wanted if f in installed] or ["sans-serif"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(results):
        iso = r["lineage"] in ISOLATE_CANDIDATES
        if r["lineage"] not in CORE_LINEAGES:
            ax1.plot(KS, r["profile_d_max"], lw=0.7, color="0.65", zorder=0)
            continue
        ax1.plot(KS, r["profile_d_max"], lw=2.2 if iso else 1.2, ls="-" if iso else "--",
                 color=cmap(i % 20), label=f'{r["lineage"]} (n={r["n_doculects"]})')
    ax1.axhline(1.5, color="grey", lw=0.8, ls=":")
    ax1.axvline(k_cal, color="crimson", lw=0.8, ls=":")
    ax1.text(k_cal * 1.08, 2.6, f"k_cal = {k_cal:.2f}\n(noise floor ≈ 0.9 similar)", color="crimson", fontsize=8)
    ax1.set(xscale="log", yscale="log", xlabel="scale k  (Z = exp(−k · LDND))",
            ylabel="maximum diversity D_max (effective # of distinct lineages)",
            title="Diversity profiles — solid: isolate candidates, dashed: controls, grey: other families")
    ax1.legend(fontsize=8, ncol=2)

    labels = [r["sample"] for r in robust]
    x = np.arange(len(labels))
    ax2.bar(x - 0.27, [r["d_max_kcal"] for r in robust], 0.27, label="D_max at k_cal (Leinster–Meckes)")
    ax2.bar(x, [r["uniform_d2_kcal"] for r in robust], 0.27, label="naive D_2 at k_cal, equal weights")
    ax2.bar(x + 0.27, [r["spread"] for r in robust], 0.27, label="spread (LDND units)")
    for xi, r in zip(x, robust):
        ax2.text(xi, max(r["d_max_kcal"], r["spread"]) + 0.04, f'n={r["n"]}', ha="center", fontsize=8)
    ax2.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
    ax2.set(title="Japonic: same lineage, different dialect counts")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--wold", type=Path, default=None, help="WOLD CLDF dir (e.g. data/wold/cldf): mask words WOLD marks as loans")
    ap.add_argument("--tokens", choices=["asjp", "segments"], default="asjp")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_docs = load_all(args.cldf, args.tokens, args.wold)
    rows = {r["ID"]: r for r in csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8"))}

    floor = noise_floor(args.cldf)
    k_cal = float(-np.log(0.9) / np.median(floor))
    print(f"noise floor LDND quantiles 10/50/90: {np.round(np.quantile(floor, [.1, .5, .9]), 3)}; k_cal={k_cal:.3f}")

    results = []
    for name, pred in LINEAGES.items():
        docs = stratified([x for x in all_docs.values() if pred(rows[x.id])], MAX_PER_LINEAGE)
        r = analyse(name, docs, k_cal)
        results.append(r)
        print(f'{name:14s} n={r["n_doculects"]:3d} glotto={r["n_glottocodes"]:3d} '
              f'maxLDND={r["max_ldnd"]:.3f} spread={r["spread"]:.3f} rare3={r["spread_rarefied_m3"]:.3f} '
              f'Dmax@kcal={r["d_max_kcal"]:.3f} naive@kcal={r["uniform_d2_kcal"]:.3f} kkt={r["kkt_all_ok"]}')

    japonic = [x for x in all_docs.values() if rows[x.id]["Family"] == "Japonic"]
    robust = sampling_robustness(japonic, k_cal)
    for r in robust:
        print(r)

    plot(results, robust, args.out / "diversity_profiles.png", k_cal)

    summary = [{k: v for k, v in r.items() if not k.startswith(("_", "profile"))} for r in results]
    (args.out / "summary.json").write_text(
        json.dumps({"tokenization": args.tokens, "k_cal": k_cal,
                    "noise_floor_quantiles_10_50_90": np.quantile(floor, [.1, .5, .9]).tolist(),
                    "excluded": load_exclusions(EXCLUSIONS_FILE),
                    "lineages": summary, "japonic_sampling": robust},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "profiles.json").write_text(json.dumps(
        {"ks": KS.tolist(), "k_cal": k_cal,
         "profiles": {r["lineage"]: {"d_max": r["profile_d_max"], "uniform": r["profile_uniform"]} for r in results}},
        ensure_ascii=False), encoding="utf-8")
    with (args.out / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lineage", "n_doculects", "n_glottocodes", "max_ldnd", "spread",
                    "spread_rarefied_m3", "d_max_kcal", "uniform_d2_kcal"])
        for r in sorted(summary, key=lambda r: r["spread"]):
            w.writerow([r["lineage"], r["n_doculects"], r["n_glottocodes"], f'{r["max_ldnd"]:.3f}',
                        f'{r["spread"]:.3f}', f'{r["spread_rarefied_m3"]:.3f}',
                        f'{r["d_max_kcal"]:.3f}', f'{r["uniform_d2_kcal"]:.3f}'])
    for r in results:
        np.savetxt(args.out / f'ldnd_{r["lineage"]}.tsv', r["_d"], delimiter="\t", fmt="%.4f",
                   header="\t".join(x.id for x in r["_docs"]), encoding="utf-8")


if __name__ == "__main__":
    main()
