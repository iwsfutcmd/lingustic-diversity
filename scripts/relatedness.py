"""Relatedness from union shortfall: do two lineages score less together than apart?

Usage:
    python scripts/relatedness.py [--cross results/cross_family.json] [--cap LABEL=PATH ...] [--near-km 1500]

Two lineages that the 40-word lists cannot tie together score the sum of their own D_max together: no cross pair is
within the detection limit, the union's similarity matrix is block-diagonal and D_max adds up. Every cross pair
within reach takes something off, and the union lies between max(D_a, D_b) and D_a + D_b. The shortfall

    s = (D_a + D_b − D_union) / min(D_a, D_b)

is 0 for no detectable link and 1 when the smaller lineage is entirely absorbed into the larger one.

s is calibrated against pairs of different families (the null), not against shuffles: chance resemblance, sound
symbolism and loans all shorten distances, and real unrelated pairs carry all three. Neighbours share loans, so each
pair is judged against unrelated pairs whose closest doculects are within ``--near-km`` of each other as well as
against all unrelated pairs. Sibling branches of one family (related, sharing no doculect) are the positive control;
nested pairs (one lineage's sample inside the other's) are left out. The proposals below are left out of the null.

Writes results/relatedness.{json,csv} and relatedness.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Long-range proposals, as pairs of lineages in LINEAGES. Pairs already one family in Glottolog are skipped.
PROPOSALS: dict[str, list[tuple[str, str]]] = {
    "Altaic (core)": [("Turkic", "Mongolic"), ("Mongolic", "Tungusic"), ("Turkic", "Tungusic")],
    "Macro-Altaic (+ Koreanic, Japonic)": [
        ("Koreanic", "Japonic"), ("Koreanic", "Tungusic"), ("Japonic", "Tungusic"), ("Koreanic", "Mongolic"),
        ("Japonic", "Mongolic"), ("Koreanic", "Turkic"), ("Japonic", "Turkic")],
    "Dene–Yeniseian": [("Yeniseian", "Athabaskan-Eyak-Tlingit")],
    "Uralic–Yukaghir": [("Uralic", "Yukaghir")],
    "Uralo-Siberian": [("Uralic", "Eskimo-Aleut"), ("Uralic", "Chukotko-Kamchatkan"),
                       ("Eskimo-Aleut", "Chukotko-Kamchatkan")],
    "Chukotko-Kamchatkan–Amuric": [("Nivkh", "Chukotko-Kamchatkan")],
    "Indo-Uralic / Nostratic": [("Indo-European", "Uralic"), ("Uralic", "Turkic"), ("Indo-European", "Kartvelian"),
                                ("Indo-European", "Afro-Asiatic"), ("Uralic", "Dravidian")],
    "Elamo-Dravidian": [("Elamite", "Dravidian")],
    "North Caucasian": [("Abkhaz-Adyge", "Nakh-Daghestanian")],
    "Dene–Caucasian": [("Yeniseian", "Sino-Tibetan"), ("Basque", "Abkhaz-Adyge"), ("Basque", "Nakh-Daghestanian"),
                       ("Burushaski", "Yeniseian")],
    "Khoisan": [("Khoe-Kwadi", "Tuu"), ("Khoe-Kwadi", "Kxa"), ("Tuu", "Kxa")],
    "Austric / Austro-Tai": [("Austroasiatic", "Austronesian"), ("Austronesian", "Tai-Kadai"),
                             ("Hmong-Mien", "Austroasiatic"), ("Hmong-Mien", "Tai-Kadai")],
    "Nilo-Saharan": [("Nubian", "Saharan"), ("Songhay", "Saharan"), ("Songhay", "Nubian"), ("Kuliak", "Nubian")],
    "Niger-Congo / Kordofanian": [("Mande", "Atlantic-Congo"), ("Dogon", "Atlantic-Congo"), ("Ijoid", "Atlantic-Congo"),
                                  ("Kru", "Atlantic-Congo"), ("Heibanic", "Narrow Talodi"),
                                  ("Heibanic", "Atlantic-Congo"), ("Katla-Tima", "Atlantic-Congo"),
                                  ("Rashad", "Atlantic-Congo"), ("Narrow Talodi", "Rashad"), ("Katla-Tima", "Rashad"),
                                  ("Narrow Talodi", "Katla-Tima"), ("Heibanic", "Rashad"), ("Heibanic", "Katla-Tima"),
                                  ("Narrow Talodi", "Atlantic-Congo"), ("Kru", "Mande"), ("Dogon", "Kru"),
                                  ("Dogon", "Mande"), ("Ijoid", "Mande"), ("Ijoid", "Kru"), ("Ijoid", "Dogon")],
    "Hruso in Sino-Tibetan": [("Hruso", "Sino-Tibetan")],
    "Totozoquean": [("Mixe-Zoque", "Totonacan")],
    "Great Andamanese–Ongan": [("Great Andamanese", "Jarawa-Onge")],
}


def shortfall(r: dict) -> float:
    a, b, u = r["dmax_a"], r["dmax_b"], r["dmax_union"]
    if not all(np.isfinite([a, b, u])) or min(a, b) <= 0:
        return float("nan")
    return float(np.clip((a + b - u) / min(a, b), 0.0, 1.0))


def classify(pairs: list[dict], proposals: dict[str, list[tuple[str, str]]], near_km: float) -> None:
    """Adds s, group (nested / related / proposal / unrelated) and near to each row, in place."""
    prop = {frozenset(p): name for name, ps in proposals.items() for p in ps}
    for r in pairs:
        # no cross pair has a distance (too few shared concepts): untestable, not "no link"
        r["s"] = shortfall(r) if np.isfinite(r.get("min_between", 0.0)) else float("nan")
        r["near"] = bool(np.isfinite(r.get("min_km", np.nan)) and r["min_km"] <= near_km)
        key = frozenset((r["a"], r["b"]))
        if r.get("shared_doculects", 0) > 0:
            r["group"] = "nested"
        elif r["related_by_construction"]:
            r["group"] = "related"
        elif key in prop:
            r["group"], r["proposal"] = "proposal", prop[key]
        else:
            r["group"] = "unrelated"


def exceed(s: float, null: np.ndarray) -> float:
    """Share of the null at or above s (with the pair itself counted, so never 0)."""
    if not np.isfinite(s) or not len(null):
        return float("nan")
    return float((1 + (null >= s - 1e-9).sum()) / (1 + len(null)))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a positive outscores a negative), ties counting half."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None]).mean()
    eq = (pos[:, None] == neg[None]).mean()
    return float(gt + eq / 2)


def analyse(pairs: list[dict], near_km: float, k: int = 250, k_near: int = 100) -> dict:
    classify(pairs, PROPOSALS, near_km)
    s = lambda rs: np.array([r["s"] for r in rs], dtype=float)
    unrel = [r for r in pairs if r["group"] == "unrelated" and np.isfinite(r["s"])]
    near = [r for r in unrel if r["near"]]
    rel = [r for r in pairs if r["group"] == "related" and np.isfinite(r["s"])]
    null_all, null_near = s(unrel), s(near)
    # size-matched null: a deep, diverse lineage offers more lists for a chance near-miss, so the shortfall grows with
    # the larger lineage's D_max; compare each pair with the k unrelated pairs nearest in (log larger D, log smaller D)
    size = lambda rs: np.log(np.array([[max(r["dmax_a"], r["dmax_b"]), min(r["dmax_a"], r["dmax_b"])] for r in rs]))
    xu = size(unrel)
    near_ix = np.array([u["near"] for u in unrel])
    for r in pairs:
        r["p_all"] = exceed(r["s"], null_all)
        r["p_near"] = exceed(r["s"], null_near)
        if np.isfinite(r["s"]):
            dist = np.hypot(*(xu - size([r])).T)
            if r["group"] == "unrelated":
                dist[[i for i, u in enumerate(unrel) if u is r]] = np.inf    # not against itself
            r["p_size"] = exceed(r["s"], null_all[np.argsort(dist)[:k]])
            # neighbours share loans: against size-matched unrelated neighbours (the fair null for a neighbouring pair)
            dn = np.where(near_ix, dist, np.inf)
            r["p_size_near"] = exceed(r["s"], null_all[np.argsort(dn)[:k_near]])
        else:
            r["p_size"] = r["p_size_near"] = float("nan")
    q = lambda v: {f"q{int(p)}": float(np.percentile(v, p)) for p in (50, 90, 95, 99)} if len(v) else {}
    return {
        "near_km": near_km,
        "n": {"unrelated": len(unrel), "unrelated_near": len(near), "related": len(rel),
              "nested": sum(r["group"] == "nested" for r in pairs),
              "proposal": sum(r["group"] == "proposal" for r in pairs)},
        "null_all": q(null_all), "null_near": q(null_near), "related_q": q(s(rel)),
        "auc_related_vs_unrelated": auc(s(rel), null_all),
        "auc_related_vs_unrelated_near": auc(s(rel), null_near),
        "share_zero_unrelated": float((null_all == 0).mean()) if len(null_all) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", type=Path, default=Path("results/cross_family.json"))
    ap.add_argument("--cap", action="append", default=[], metavar="LABEL=PATH",
                    help="cross_family.json from another --cap, to show how the cap moves the shortfall")
    ap.add_argument("--near-km", type=float, default=1500.0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    main_j = json.loads(args.cross.read_text(encoding="utf-8"))
    runs = {f"cap{main_j['cap_per_lineage']}": main_j["pairs"]}
    for spec in args.cap:
        label, path = spec.split("=", 1)
        runs[label] = json.loads(Path(path).read_text(encoding="utf-8"))["pairs"]
    head = next(iter(runs))
    stats = {label: analyse(pairs, args.near_km) for label, pairs in runs.items()}
    pairs = runs[head]

    other = {label: {frozenset((r["a"], r["b"])): r for r in ps} for label, ps in runs.items() if label != head}
    rows = []
    for r in sorted(pairs, key=lambda r: (r["group"], -(r["s"] if np.isfinite(r["s"]) else -1))):
        if r["group"] == "nested":
            continue
        row = {k: r.get(k, "") for k in ("a", "b", "group", "proposal", "s", "p_size_near", "p_size", "p_near", "p_all", "min_km",
                                         "min_between", "dmax_a", "dmax_b", "dmax_union", "n_a", "n_b")}
        for label, idx in other.items():
            o = idx.get(frozenset((r["a"], r["b"])))
            row[f"s_{label}"] = o["s"] if o else float("nan")
            row[f"p_size_near_{label}"] = o["p_size_near"] if o else float("nan")
        rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "relatedness.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()})
    props = [r for r in rows if r["group"] == "proposal"]
    (args.out / "relatedness.json").write_text(json.dumps({
        "measure": "s = (D_a + D_b - D_union) / min(D_a, D_b); p = share of unrelated pairs with s at least as large",
        "stats": stats, "proposals": PROPOSALS, "proposal_pairs": props,
        "top_unrelated": [r for r in rows if r["group"] == "unrelated"][:40],
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    for label, st in stats.items():
        print(label, json.dumps(st, default=float))
    print(f"\n{'proposal':36} {'pair':40} {'s':>6} {'p_sn':>6} {'p_size':>7} {'p_near':>7} {'p_all':>6} " +
          " ".join(f"{'s_' + l:>8}" for l in other))
    for r in sorted(props, key=lambda r: (r["proposal"], r["p_size_near"])):
        print(f"{r['proposal']:36} {r['a'] + ' – ' + r['b']:40} {r['s']:6.3f} {r['p_size_near']:6.3f} {r['p_size']:7.3f} {r['p_near']:7.3f} {r['p_all']:6.3f} " +
              " ".join(f"{r['s_' + l]:8.3f}" for l in other))
    plot(pairs, stats[head], args.out / "relatedness.png")


def plot(pairs: list[dict], st: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    props = sorted([r for r in pairs if r["group"] == "proposal" and np.isfinite(r["s"])], key=lambda r: r["s"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(7, 0.2 * len(props) + 1.5)),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # 1. shortfall of unrelated pairs (far / neighbours) and of sibling branches
    bins = np.linspace(0, 1, 41)
    for g, near, label, c in [("unrelated", False, "unrelated, far apart", "tab:grey"),
                              ("unrelated", True, f"unrelated, within {st['near_km']:.0f} km", "tab:blue"),
                              ("related", None, "sibling branches of one family", "tab:green")]:
        v = [r["s"] for r in pairs if r["group"] == g and np.isfinite(r["s"]) and (near is None or r["near"] == near)]
        ax1.hist(v, bins=bins, density=True, alpha=0.5, color=c, label=f"{label} (n={len(v)})")
    ax1.set_yscale("log")
    ax1.set(xlabel="union shortfall s = (D_a + D_b − D_union) / min(D_a, D_b)", ylabel="density (log)",
            title=f"Shortfall by group (AUC siblings vs unrelated {st['auc_related_vs_unrelated']:.2f})")
    ax1.legend(fontsize=8)

    # 2. proposals, coloured by the size-matched neighbour p
    y = np.arange(len(props))
    hit = np.array([r["p_size_near"] < 0.05 for r in props])
    ax2.scatter([r["s"] for r in props], y, c=np.where(hit, "crimson", "grey"), s=22, zorder=3)
    for q, ls in (("q50", ":"), ("q95", "--")):
        if q in st["null_near"]:
            ax2.axvline(st["null_near"][q], color="tab:blue", ls=ls, lw=1,
                        label=f"unrelated neighbours, {q[1:]}th percentile")
    ax2.set_yticks(y, [f"{r['a']} – {r['b']}" for r in props], fontsize=7)
    ax2.set(xlabel="union shortfall s", title="Long-range proposals (red: p < 0.05 against size-matched neighbours)")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
