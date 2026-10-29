"""Expert loan labels from WOLD (Haspelmath & Tadmor 2009; github.com/lexibank/wold), mapped onto ASJP.

WOLD marks every word of 41 languages as inherited or borrowed. Its concepts map to the ASJP 40-item list
through Concepticon IDs (37 of 40 have a WOLD counterpart). A concept counts as borrowed in a language when
*every* WOLD word for it is a loan; an ASJP doculect is matched to a WOLD language by Glottocode (the doculect
with the most 40-item concepts).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WoldLoans:
    wold_language: str
    asjp_id: str
    borrowed: frozenset[str]     # ASJP concept IDs WOLD marks as borrowed
    n_concepts: int              # ASJP concepts with a WOLD counterpart in this language


def _read(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def loan_table(wold_cldf: Path, asjp_cldf: Path) -> dict[str, WoldLoans]:
    """{ASJP doculect ID: WoldLoans} for every WOLD language that has an ASJP list."""
    wold_cldf, asjp_cldf = Path(wold_cldf), Path(asjp_cldf)
    core = {r["ID"]: r["Concepticon_ID"] for r in _read(asjp_cldf / "parameters.csv") if r["Name"].startswith("*")}
    wold_to_asjp = {}
    for r in _read(wold_cldf / "parameters.csv"):
        for aid, cid in core.items():
            if r["Concepticon_ID"] and r["Concepticon_ID"] == cid:
                wold_to_asjp[r["ID"]] = aid

    loans: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in _read(wold_cldf / "forms.csv"):
        if r["Parameter_ID"] in wold_to_asjp:
            loans[r["Language_ID"]][wold_to_asjp[r["Parameter_ID"]]].append(r["Loan"] == "true")

    counts: dict[str, int] = defaultdict(int)
    for r in _read(asjp_cldf / "forms.csv"):
        if r["Parameter_ID"] in core:
            counts[r["Language_ID"]] += 1
    asjp_langs = list(_read(asjp_cldf / "languages.csv"))

    out = {}
    for w in _read(wold_cldf / "languages.csv"):
        by_c = loans.get(w["ID"])
        cands = [r for r in asjp_langs if r["Glottocode"] == w["Glottocode"]]
        if not by_c or not cands:
            continue
        a = max(cands, key=lambda r: counts[r["ID"]])
        out[a["ID"]] = WoldLoans(w["Name"], a["ID"], frozenset(c for c, v in by_c.items() if all(v)), len(by_c))
    return out
