"""Load ASJP word lists from the lexibank CLDF export (github.com/lexibank/asjp).

Words are returned as tuples of *segments* (hashable tokens), never as raw
strings, so the distance code is agnostic to script and segmentation:

* ``"asjp"``     — ASJPcode with the modifiers ``" * ~ $`` stripped, one token per
                   remaining symbol (the convention behind published LDND values).
* ``"segments"`` — the CLDF ``Segments`` column (space-separated IPA segments,
                   NFC-normalized), for finer-grained comparisons.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Tokenization = Literal["asjp", "segments"]
Word = tuple[str, ...]
WordList = Mapping[str, tuple[Word, ...]]  # concept ID -> synonyms

ASJP_MODIFIERS = frozenset('"*~$')
NON_SEGMENTAL = frozenset(" -_+")


@dataclass(frozen=True, slots=True)
class Doculect:
    id: str
    name: str
    glottocode: str
    family: str
    wals: str
    words: WordList


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def tokenize(row: Mapping[str, str], how: Tokenization) -> Word:
    match how:
        case "asjp":
            return tuple(
                ch for ch in _nfc(row["Form"]) if ch not in ASJP_MODIFIERS and ch not in NON_SEGMENTAL
            )
        case "segments":
            return tuple(
                seg for seg in _nfc(row["Segments"]).split() if seg not in NON_SEGMENTAL
            )
    raise ValueError(f"unknown tokenization: {how!r}")


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def load_exclusions(path: Path | None) -> dict[str, str]:
    """{doculect ID: reason} from a TSV with columns ``id`` and ``reason`` (``#`` lines ignored)."""
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rows = csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t")
        return {r["id"].strip(): (r.get("reason") or "").strip() for r in rows if r.get("id", "").strip()}


def core_concepts(cldf_dir: Path) -> frozenset[str]:
    """The 40-item ASJP list (concept names marked with a leading '*')."""
    return frozenset(
        r["ID"] for r in _read_csv(cldf_dir / "parameters.csv") if r["Name"].startswith("*")
    )


def load(
    cldf_dir: Path,
    *,
    keep: callable = lambda lang: True,
    concepts: frozenset[str] | None = None,
    tokenization: Tokenization = "asjp",
    drop_loans: bool = True,
    exclude: Iterable[str] = (),
) -> dict[str, Doculect]:
    """Return {doculect ID: Doculect} for languages where ``keep(row)`` is true."""
    cldf_dir = Path(cldf_dir)
    concepts = core_concepts(cldf_dir) if concepts is None else concepts
    langs = {r["ID"]: r for r in _read_csv(cldf_dir / "languages.csv") if keep(r) and r["ID"] not in set(exclude)}

    words: dict[str, dict[str, list[Word]]] = defaultdict(lambda: defaultdict(list))
    for r in _read_csv(cldf_dir / "forms.csv"):
        if r["Language_ID"] not in langs or r["Parameter_ID"] not in concepts:
            continue
        if drop_loans and r.get("Loan", "").lower() == "true":
            continue
        tok = tokenize(r, tokenization)
        if tok:
            words[r["Language_ID"]][r["Parameter_ID"]].append(tok)

    return {
        lid: Doculect(
            id=lid,
            name=_nfc(row["Name"]),
            glottocode=row["Glottocode"],
            family=row["Family"],
            wals=row["classification_wals"],
            words={c: tuple(ws) for c, ws in words[lid].items()},
        )
        for lid, row in langs.items()
        if words.get(lid)
    }
