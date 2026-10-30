"""Diversity profiles for isolate candidates and control lineages from ASJP.

Usage:
    python scripts/run_analysis.py [--cldf data/asjp/cldf] [--out results] [--tokens asjp|segments]
                                   [--distance pmi|ldnd] [--pmi-model results/pmi_model_all.tsv]
"""

from __future__ import annotations

import os

# One BLAS thread per process: D_max solves are many small matrix products, which multithreaded BLAS slows down
# (2x on a 660-doculect matrix). Parallelism comes from solving the scales in separate processes instead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
from isodiv.distcache import DistCache
from isodiv.diversity import max_diversity, similarity_matrix, uniform_diversity
from isodiv.metric import DISTANCES, Metric, get_metric

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

# Isolates with one to three lists, ancient (Sumerian, Elamite, Etruscan) and modern. A single list has no measurable
# internal diversity (D_max = 1); they are here for the between-lineage distances. Hurro-Urartian has no ASJP list.
SMALL_ISOLATES = ["Sumerian", "Elamite", "Etruscan", "Kusunda", "Nihali", "Hruso"]
# Small families from other macroareas, as further points of comparison, and the branches of the proposed
# Niger-Congo that Glottolog keeps apart from Atlantic-Congo (Mande, Kru, Ijoid, Dogon, the Kordofanian groups).
SMALL_FAMILIES = ["Jarawa-Onge", "Nubian", "Songhay", "Saharan", "Dogon", "Kru", "Ijoid", "Mixe-Zoque", "Totonacan",
                  "Cochimi-Yuman", "Miwok-Costanoan", "Mande", "Heibanic", "Narrow Talodi", "Katla-Tima", "Rashad"]
for _fam in SMALL_ISOLATES + SMALL_FAMILIES:
    LINEAGES[_fam] = (lambda f: lambda r: r["Family"] == f)(_fam)
ISOLATE_CANDIDATES += SMALL_ISOLATES

# Subfamilies, picked by node in ASJP's Glottolog classification path (label -> node). Each is nested in a
# top-level family above, so cross-family comparisons treat pairs that share a Family as related.
SUBFAMILIES = {
    "Semitic": "Semitic", "Arabic": "Arabic", "Chadic": "Chadic", "Cushitic": "Cushitic", "Berber": "Berber",
    "Sinitic": "Sinitic", "Min": "Min", "Lolo-Burmese": "Lolo-Burmese", "Bodic": "Bodic", "Kuki-Chin": "Kuki-Chin",
    "Indo-Aryan": "Indo-Aryan", "Iranian": "Iranian", "Celtic": "Celtic", "Balto-Slavic": "Balto-Slavic",
    "Oceanic": "Oceanic", "Polynesian": "Polynesian", "Finnic": "Finnic", "Saami": "Saami", "Samoyedic": "Samoyedic",
    "Oghuz": "Oghuz", "Kipchak": "Kipchak", "South Dravidian": "SouthDravidian",
    "Quichean-Mamean": "Quichean-Mamean", "Kam-Tai": "Kam-Tai",
}
for _label, _node in SUBFAMILIES.items():
    LINEAGES[_label] = (lambda n: lambda r: r["Family"] != "Bookkeeping" and n in r["classification_glottolog"].split(","))(_node)

# The major Sinitic groups. Glottolog treats Yue, Hakka, Gan, Xiang, Jin, Hui and Wu each as one language with many
# dialect doculects, so those are picked by Glottocode; Mandarin and Pinghua by classification node. Min is above.
SINITIC_GROUPS = {
    "Mandarin": ("node", "Mandarinic"), "Jin": ("glottocode", "jiny1235"), "Wu": ("glottocode", "wuch1236"),
    "Hui": ("glottocode", "huiz1242"), "Gan": ("glottocode", "ganc1239"), "Xiang": ("glottocode", "xian1251"),
    "Hakka": ("glottocode", "hakk1236"), "Yue": ("glottocode", "yuec1235"), "Pinghua": ("node", "Pinghua"),
}
for _label, (_by, _key) in SINITIC_GROUPS.items():
    LINEAGES[_label] = (
        (lambda n: lambda r: r["Family"] != "Bookkeeping" and n in r["classification_glottolog"].split(","))(_key)
        if _by == "node" else (lambda g: lambda r: r["Family"] != "Bookkeeping" and r["Glottocode"] == g)(_key))
MAX_PER_LINEAGE = 80          # per-lineage cap for --cap 80 (and wold_effect.py); the default is no cap: whole pools
K_FINE = 3.0                  # fine scale where unrelated lects count as nearly fully distinct
SAVE_MATRIX_MAX_N = 200       # per-lineage ldnd_*.tsv only up to this size; larger ones stay in the distance cache
CACHE_DIR = Path("data/cache")
NEST_TOL = 0.02               # a lineage nests in another when at most this share of its pool lies outside it
KS = np.geomspace(0.1, 40, 48)
RAREFY_M = 3          # compare lineages at a fixed number of distinct-Glottocode doculects
RAREFY_DRAWS = 40

# Subgroups by Glottolog node: (nodes the path must contain one of, nodes it must not contain). Groupings Glottolog
# has no node for (South Semitic, North Ethiopic, Formosan) are built from nodes; Formosan is not a clade but
# "Austronesian outside Malayo-Polynesian".
NODE_GROUPS = {
    # Austronesian
    "Malayo-Polynesian": (["Malayo-Polynesian"], []),
    "Formosan": (["Austronesian"], ["Malayo-Polynesian"]),
    "Eastern Malayo-Polynesian": (["EasternMalayo-Polynesian"], []),
    "Greater Central Philippine": (["GreaterCentralPhilippine"], []),
    "Malayic": (["Malayic"], []),
    "Celebic": (["Celebic"], []),
    # Semitic
    "East Semitic": (["EastSemitic"], []),
    "West Semitic": (["WestSemitic"], []),
    "Central Semitic": (["CentralSemitic"], []),
    "Northwest Semitic": (["NorthwestSemitic"], []),
    "Aramaic": (["Aramaic"], []),
    "North-Eastern Neo-Aramaic": (["North-EasternNeo-Aramaic"], []),
    "Canaanite": (["Canaanite"], []),
    "South Semitic": (["Ethiosemitic", "ModernSouthArabian", "Sayhadic"], []),
    "Ethiosemitic": (["Ethiosemitic"], []),
    "North Ethiopic": (["Ethiosemitic"], ["SouthEthiopic"]),
    "South Ethiopic": (["SouthEthiopic"], []),
    "Outer South Ethiopic": (["OuterSouthEthiopic"], []),
    "Modern South Arabian": (["ModernSouthArabian"], []),
    "Ancient South Arabian": (["Sayhadic"], []),
    "Peninsular Arabic": (["ArabianPeninsulaArabic"], []),
    "Egyptic Arabic": (["EgypticArabic"], []),
    "North African Arabic": (["NorthAfricanArabic"], []),
    "Eastern Arabic": (["EasternArabic"], []),
    "Levantine Arabic": (["Levantine-CypriotArabic"], []),
}
for _label, (_any, _none) in NODE_GROUPS.items():
    LINEAGES[_label] = (lambda a, z: lambda r: r["Family"] != "Bookkeeping" and
                        bool(set(a) & set(p := r["classification_glottolog"].split(","))) and not set(z) & set(p))(_any, _none)


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


def nested_samples(pools: dict[str, frozenset[str]], docs: dict[str, Doculect], cap: int = MAX_PER_LINEAGE,
                   tol: float = NEST_TOL) -> tuple[dict[str, list[Doculect]], dict[str, list[str]]]:
    """One sample per lineage such that a lineage nested in another is a subset of its sample.

    D_max can only grow when points are added, so a parent whose sample contains its children's samples can never
    score below a child; independent caps (the earlier scheme) let a child with better coverage of its fast-changing
    part outscore its parent. Built bottom-up: sample(L) = the samples of L's maximal children, topped up from the rest
    of L's pool by ``stratified`` to ``cap`` (a parent is therefore larger than ``cap`` when its children already are).
    Child means at most ``tol`` of its pool outside L's pool, which absorbs ASJP rows with a blank Family. Lineages
    with identical pools share one sample. Returns (samples, maximal children by lineage).
    """
    groups: dict[frozenset[str], list[str]] = defaultdict(list)
    for name, pool in pools.items():
        groups[pool].append(name)
    ordered = sorted(groups, key=len)
    eff: dict[frozenset[str], frozenset[str]] = {}
    chosen: dict[frozenset[str], list[Doculect]] = {}
    kids: dict[frozenset[str], list[frozenset[str]]] = {}
    for pool in ordered:
        cand = [c for c in chosen if len(eff[c] - pool) <= tol * len(eff[c]) and len(eff[c] | pool) > len(eff[c])]
        maximal = [c for c in cand if not any(c is not o and len(eff[c] - eff[o]) <= tol * len(eff[c])
                                              and len(eff[o]) > len(eff[c]) for o in cand)]
        eff[pool] = pool.union(*(eff[c] for c in cand))
        have = {x.id: x for c in maximal for x in chosen[c]}
        rest = [docs[i] for i in sorted(eff[pool] - have.keys()) if i in docs]
        extra = stratified(rest, cap - len(have)) if len(have) < cap else []
        chosen[pool] = sorted([*have.values(), *extra], key=lambda x: x.id)
        kids[pool] = maximal
    samples = {name: chosen[pool] for pool, names in groups.items() for name in names}
    children = {name: [n for c in kids[pool] for n in groups[c][:1]] for pool, names in groups.items() for name in names}
    return samples, children


def drop_nan(docs: list[Doculect], d: np.ndarray, protect: frozenset[str] = frozenset()
             ) -> tuple[list[Doculect], np.ndarray]:
    """Drop doculects until no pair is NaN (too few shared concepts), most-affected first. Ids in ``protect`` go
    last, so a parent keeps every doculect that survived in one of its children."""
    keep = np.ones(len(docs), dtype=bool)
    prot = np.array([x.id in protect for x in docs])
    while np.isnan(d[np.ix_(keep, keep)]).any():
        sub = d[np.ix_(keep, keep)]
        idx = np.flatnonzero(keep)
        score = np.isnan(sub).sum(axis=1) * np.where(prot[idx], 1e-3, 1.0)   # clean doculects score 0
        keep[idx[score.argmax()]] = False
    return [x for x, k in zip(docs, keep) if k], d[np.ix_(keep, keep)]


def spread(d: np.ndarray, level: float = 1.5, bracket: tuple[float, float] | None = None) -> float:
    """Clock-free 'diversity radius' in distance units (LDND or PMI-LDND).

    k* = the scale at which D_max first reaches `level` effective lineages;
    spread = ln 3 / k*. For a two-doculect lineage at distance δ this is exactly δ,
    so it reads like a distance — but it is a property of the whole similarity
    space, robust to how many dialects were sampled. A singleton has spread 0.
    """
    if d.shape[0] < 2 or np.nanmax(d) == 0:
        return 0.0
    f = lambda logk: max_diversity(similarity_matrix(d, np.exp(logk))).value - level
    lo, hi = np.log(1e-3), np.log(1e4)
    if bracket is not None:                         # the profile already brackets the root: skip the blind search
        lo, hi = np.log(bracket[0]), np.log(bracket[1])
    elif f(hi) < 0:
        return 0.0
    try:
        root = brentq(f, lo, hi, xtol=1e-4)
    except ValueError:                              # bracket from the profile did not straddle the root
        if bracket is None or f(np.log(1e4)) < 0:
            return 0.0
        root = brentq(f, np.log(1e-3), np.log(1e4), xtol=1e-4)
    return float(np.log(3) / np.exp(root))


def noise_floor(cldf: Path, metric: Metric, n_pairs: int = 400, seed: int = 0) -> np.ndarray:
    """Distance between doculects of the *same* language (same Glottocode, names X / X_2):
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
    v = metric.values([(docs[a], docs[b]) for a, b in pairs if a in docs and b in docs])
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


def _solve_scale(task):
    """One scale of the profile; module-level so a process pool can run it."""
    d, k, warm = task
    z = similarity_matrix(d, k)
    r = max_diversity(z, warm=warm)
    return r, uniform_diversity(z)


_POOL = None
PARALLEL_MIN_N = 30           # smaller lineages solve faster in-process than the pickling costs


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        _POOL = ProcessPoolExecutor(os.cpu_count() or 1, mp_context=get_context("spawn"))
    return _POOL


def solve_scales(d: np.ndarray, ks, warms: list[list[np.ndarray]]):
    """D_max (and the equal-weight diversity) at each scale; large lineages use the process pool."""
    tasks = [(d, k, w) for k, w in zip(ks, warms)]
    if d.shape[0] >= PARALLEL_MIN_N and (os.cpu_count() or 1) > 1:
        return list(_pool().map(_solve_scale, tasks))
    return [_solve_scale(t) for t in tasks]


def _embed(child: dict, docs: list[Doculect], weights: np.ndarray) -> np.ndarray | None:
    """A child's optimal distribution on the parent's doculects (None if none of its support survives)."""
    pos = {x.id: i for i, x in enumerate(docs)}
    w = np.zeros(len(docs))
    for x, p in zip(child["_docs"], weights):
        if x.id in pos:
            w[pos[x.id]] = p
    return w / w.sum() if w.sum() > 0 else None


def _spread_bracket(profile: list[float], level: float = 1.5) -> tuple[float, float] | None:
    """Neighbouring grid scales between which the profile crosses ``level`` (D_max grows with k)."""
    above = [i for i, v in enumerate(profile) if v >= level]
    if not above:
        return None
    i = above[0]
    return (float(KS[i - 1]), float(KS[i])) if i > 0 else (1e-3, float(KS[0]))


def analyse(name: str, docs: list[Doculect], k_cal: float, metric: Metric, children: tuple[dict, ...] = (),
            cache: DistCache | None = None) -> dict:
    """``children``: results of the lineages nested in this one. Their doculects are kept through ``drop_nan`` and
    their optimal distributions seed the optimizer, so D_max here is never below a child's at any scale."""
    d = cache.matrix(docs) if cache is not None else metric.matrix(docs)
    docs, d = drop_nan(docs, d, protect=frozenset(x.id for c in children for x in c["_docs"]))

    def warm(ws: list[np.ndarray]) -> list[np.ndarray]:
        return [w for c, cw in zip(children, ws) if (w := _embed(c, docs, cw)) is not None]

    k_ref = k_cal
    scales = [*KS, k_ref, K_FINE]                # the profile, then k_cal and the fine scale
    warms = [warm([c["_weights"][i] for c in children]) for i in range(len(KS))]
    warms += [warm([c["_weights_kcal"] for c in children]), warm([c["_weights_k3"] for c in children])]
    solved = solve_scales(d, scales, warms)
    maxes, uniforms = [r for r, _ in solved[: len(KS)]], [u for _, u in solved[: len(KS)]]
    (m_ref, u_ref), (d_k3, _) = solved[len(KS)], solved[len(KS) + 1]
    return {
        "lineage": name,
        "family": max({x.family for x in docs}, key=[x.family for x in docs].count) if docs else "",
        "n_doculects": len(docs),
        "n_glottocodes": len({x.glottocode for x in docs}),
        "max_ldnd": float(d.max()) if len(docs) > 1 else 0.0,
        "spread": spread(d, bracket=_spread_bracket([m.value for m in maxes])),
        "spread_rarefied_m3": rarefied_spread(docs, d),
        "d_max_kcal": m_ref.value,
        "uniform_d2_kcal": u_ref,
        "representatives_kcal": sorted(
            ((docs[i].id, round(float(w), 3)) for i, w in enumerate(m_ref.weights) if w > 1e-3),
            key=lambda t: -t[1],
        ),
        "profile_d_max": [m.value for m in maxes],
        "profile_uniform": uniforms,
        "d_max_k3": d_k3.value,
        "kkt_all_ok": all(m.kkt_ok for m in maxes),
        "_d": d,
        "_docs": docs,
        "_weights": [m.weights for m in maxes],
        "_weights_kcal": m_ref.weights,
        "_weights_k3": d_k3.weights,
    }


def check_nesting(results: list[dict], children: dict[str, list[str]], tol: float = 1e-6) -> dict:
    """D_max of a parent must be at least that of every child at every scale (holds when the child's doculects are a
    subset of the parent's). Lists the pairs where it is not."""
    res = {r["lineage"]: r for r in results}
    pairs, bad = 0, []
    for parent, kids in children.items():
        for child in kids:
            if not {x.id for x in res[child]["_docs"]} <= {x.id for x in res[parent]["_docs"]}:
                continue
            pairs += 1
            gap = min([res[parent]["d_max_kcal"] - res[child]["d_max_kcal"],
                       *(a - b for a, b in zip(res[parent]["profile_d_max"], res[child]["profile_d_max"]))])
            if gap < -tol:
                bad.append({"parent": parent, "child": child, "worst_gap": float(gap)})
    return {"pairs": pairs, "violations": bad}


def sampling_robustness(docs: list[Doculect], k_cal: float, metric: Metric) -> list[dict]:
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
        d = metric.matrix(sub)
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


def plot(results: list[dict], robust: list[dict], out: Path, k_cal: float, label: str = "LDND") -> None:
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
    ax1.set(xscale="log", yscale="log", xlabel=f"scale k  (Z = exp(−k · {label}))",
            ylabel="maximum diversity D_max (effective # of distinct lineages)",
            title="Diversity profiles — solid: isolate candidates, dashed: controls, grey: other families")
    ax1.legend(fontsize=8, ncol=2)

    labels = [r["sample"] for r in robust]
    x = np.arange(len(labels))
    ax2.bar(x - 0.27, [r["d_max_kcal"] for r in robust], 0.27, label="D_max at k_cal (Leinster–Meckes)")
    ax2.bar(x, [r["uniform_d2_kcal"] for r in robust], 0.27, label="naive D_2 at k_cal, equal weights")
    ax2.bar(x + 0.27, [r["spread"] for r in robust], 0.27, label=f"spread ({label} units)")
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
    ap.add_argument("--distance", choices=DISTANCES, default="pmi")
    ap.add_argument("--cap", type=int, default=0,
                    help="doculects per lineage (0: whole pools, so D_max is exact for all documented data; a lineage "
                         "that contains others also keeps theirs)")
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    args = ap.parse_args()
    metric = get_metric(args.distance, args.pmi_model)
    args.out.mkdir(parents=True, exist_ok=True)

    all_docs = load_all(args.cldf, args.tokens, args.wold)
    rows = {r["ID"]: r for r in csv.DictReader((args.cldf / "languages.csv").open(encoding="utf-8"))}

    floor = noise_floor(args.cldf, metric)
    k_cal = float(-np.log(0.9) / np.median(floor))
    print(f"noise floor {metric.label} quantiles 10/50/90: {np.round(np.quantile(floor, [.1, .5, .9]), 3)}; k_cal={k_cal:.3f}")

    pools = {name: frozenset(i for i in all_docs if pred(rows[i])) for name, pred in LINEAGES.items()}
    cap = args.cap or 10**9
    samples, children = nested_samples(pools, all_docs, cap)
    cache_path = CACHE_DIR / f"dist_{args.distance}_{args.tokens}.npy"
    cache = DistCache.load(cache_path, metric, list(all_docs))
    by_name: dict[str, dict] = {}

    def run(name: str) -> None:
        if name in by_name:
            return
        for c in children[name]:
            run(c)                                                    # children first
        r = analyse(name, samples[name], k_cal, metric, tuple(by_name[c] for c in children[name]), cache)
        by_name[name] = r
        if r["n_doculects"] > 300:
            cache.save(cache_path)                                    # keep the slow part if the run dies
        print(f'{name:14s} n={r["n_doculects"]:3d} glotto={r["n_glottocodes"]:3d} '
              f'maxLDND={r["max_ldnd"]:.3f} spread={r["spread"]:.3f} rare3={r["spread_rarefied_m3"]:.3f} '
              f'Dmax@kcal={r["d_max_kcal"]:.3f} naive@kcal={r["uniform_d2_kcal"]:.3f} kkt={r["kkt_all_ok"]}', flush=True)

    for name in sorted(LINEAGES, key=lambda n: len(samples[n])):
        run(name)
    cache.save(cache_path)
    results = [by_name[n] for n in LINEAGES]
    nesting = check_nesting(results, children)
    print(f"nesting: {nesting['pairs']} parent/child pairs, {len(nesting['violations'])} violations")

    japonic = [x for x in all_docs.values() if rows[x.id]["Family"] == "Japonic"]
    robust = sampling_robustness(japonic, k_cal, metric)
    for r in robust:
        print(r)

    plot(results, robust, args.out / "diversity_profiles.png", k_cal, metric.label)

    summary = [{k: v for k, v in r.items() if not k.startswith(("_", "profile"))} for r in results]
    (args.out / "summary.json").write_text(
        json.dumps({"tokenization": args.tokens, "distance": args.distance, "distance_label": metric.label,
                    "k_cal": k_cal,
                    "noise_floor_quantiles_10_50_90": np.quantile(floor, [.1, .5, .9]).tolist(),
                    "excluded": load_exclusions(EXCLUSIONS_FILE),
                    "max_per_lineage": args.cap or None, "nesting": nesting,
                    "children": {n: c for n, c in children.items() if c},
                    "lineages": summary, "japonic_sampling": robust},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "profiles.json").write_text(json.dumps(
        {"ks": KS.tolist(), "k_cal": k_cal, "distance": args.distance,
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
    (args.out / "samples.json").write_text(
        json.dumps({r["lineage"]: [x.id for x in r["_docs"]] for r in results}, ensure_ascii=False), encoding="utf-8")
    for r in results:
        if r["n_doculects"] <= SAVE_MATRIX_MAX_N:
            np.savetxt(args.out / f'ldnd_{r["lineage"]}.tsv', r["_d"], delimiter="\t", fmt="%.4f",
                       header="\t".join(x.id for x in r["_docs"]), encoding="utf-8")
        else:
            (args.out / f'ldnd_{r["lineage"]}.tsv').unlink(missing_ok=True)       # stale file from a capped run


if __name__ == "__main__":
    main()
