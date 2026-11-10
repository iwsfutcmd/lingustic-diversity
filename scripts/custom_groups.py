"""Score ad hoc groups (config/custom/custom_groups.tsv) like lineages: D_max and depth.

Usage:
    python scripts/custom_groups.py [--out results]

Each row names a group by the pools of results/samples.json, ASJP IDs and Glottolog nodes it includes and excludes.
D_max uses the headline kernel; depth uses scripts/depth.py's calibration and family depth, without the lists in
``not_in_depth`` (the group's own ancestors). Writes results/custom_groups.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from depth import family_depth  # noqa: E402
from run_analysis import drop_nan, load_all, open_world_cache  # noqa: E402

from isodiv.asjp import read_languages
from isodiv.diversity import max_diversity
from isodiv.kernel import Kernel
from isodiv.metric import get_metric

CONFIG = Path(__file__).resolve().parent.parent / "config" / "custom" / "custom_groups.tsv"


def read_groups(path: Path = CONFIG) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.startswith("#")]
    cols = ["name", "include", "exclude", "not_in_depth", "note"]
    return [dict(zip(cols, [c.strip() for c in ln.rstrip("\n").split("\t")] + [""] * 5)) for ln in lines]


def resolve(tokens: str, samples: dict, rows: dict) -> set[str]:
    out = set()
    for tok in tokens.split():
        kind, _, val = tok.partition(":")
        if kind == "lineage":
            out |= set(samples[val])
        elif kind == "id":
            out.add(val)
        elif kind == "node":
            out |= {i for i, r in rows.items() if val in (r.get("classification_glottolog") or "").split(",")}
        else:
            raise SystemExit(f"{CONFIG}: unknown token {tok!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cldf", type=Path, default=Path("data/asjp/cldf"))
    ap.add_argument("--pmi-model", type=Path, default=Path("results/pmi_model_all.tsv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    kernel = Kernel.from_summary(summary)
    cal = json.loads((args.out / "depth_calibration.json").read_text(encoding="utf-8"))
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    rows = {r["ID"]: r for r in read_languages(args.cldf)}
    docs = load_all(args.cldf)                       # exclusions already applied
    cache, _ = open_world_cache(get_metric("pmi", args.pmi_model), args.cldf)

    out = []
    for g in read_groups():
        ids = (resolve(g["include"], samples, rows) - resolve(g["exclude"], samples, rows)) & set(docs)
        ds = [docs[i] for i in sorted(ids)]
        d = cache.matrix(ds)
        ds, d = drop_nan(ds, d)
        r = max_diversity(kernel(d))
        skip = resolve(g["not_in_depth"], samples, rows)
        dd_docs = [x for x in ds if x.id not in skip]
        dd = cache.matrix(dd_docs)
        dd_docs, dd = drop_nan(dd_docs, dd)
        f = family_depth(dd, cal, cal["horizon_d"])
        a, b = f.get("_a", []), f.get("_b", [])
        if len(a) < len(b):
            a, b = b, a
        sup = f["supported"] or {}
        lone = dd_docs[b[0]].id if len(b) == 1 and len(dd_docs) > 2 else ""
        row = {
            "name": g["name"], "note": g["note"], "lists": len(ds), "glottocodes": len({x.glottocode for x in ds}),
            "d_max": float(r.value),
            "representatives": sorted(((ds[i].id, round(float(w), 3)) for i, w in enumerate(r.weights) if w > 1e-3),
                                      key=lambda t: -t[1])[:12],
            "depth": f["depth"], "retention": f["retention"], "median_d_across_root": f["median_d"],
            "beyond_horizon": bool(f["beyond_horizon"]), "root_split_sizes": f["split"],
            "smaller_branch": sorted(dd_docs[i].id for i in b)[:6], "lone_branch": lone,
            "depth_2": sup.get("depth", float("nan")) if lone else f["depth"],
            "retention_2": float(np.exp(-sup["depth"])) if lone and sup else f["retention"],
        }
        out.append(row)
        print(f"{g['name']:22} lists {len(ds):4}  D_max {row['d_max']:.2f}  shared across the deepest split "
              f"{row['retention']:.0%}{f' ({row['retention_2']:.0%} without {lone})' if lone else ''}", flush=True)
    (args.out / "custom_groups.json").write_text(json.dumps({"groups": out}, ensure_ascii=False, indent=2,
                                                            default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
