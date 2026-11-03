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


CONFIG = Path(__file__).resolve().parent.parent.parent / "config"
SUPPLEMENTS = CONFIG / "supplements.tsv"
ADDED_LANGUAGES = CONFIG / "added_languages.tsv"


def _supplements_off() -> bool:
    import os
    return bool(os.environ.get("ISODIV_NO_SUPPLEMENTS"))


def _supplement_files(path: Path) -> list[Path]:
    """config/supplements.tsv plus every config/supplements_*.tsv beside it (kept apart where licences differ)."""
    path = Path(path)
    return [x for x in [path, *sorted(path.parent.glob(f"{path.stem}_*{path.suffix}"))] if x.exists()]


def load_supplements(path: Path | None = SUPPLEMENTS) -> dict[str, dict[str, list[str]]]:
    """{doculect ID: {concept ID: [ASJPcode forms]}} from config/supplements.tsv and supplements_*.tsv (``#`` lines
    ignored).

    Set the environment variable ISODIV_NO_SUPPLEMENTS=1 (or pass path=None) to use ASJP as it is."""
    if path is None or _supplements_off():
        return {}
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for f in _supplement_files(path):
        with f.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t"):
                if r.get("asjpcode", "").strip():
                    out[r["doculect"].strip()][r["concept"].strip()].append(r["asjpcode"].strip())
    return out


def read_languages(cldf_dir: Path, added: Path | None = ADDED_LANGUAGES) -> list[dict[str, str]]:
    """ASJP's languages.csv rows plus doculects ASJP lacks (config/added_languages.tsv, same columns), whose words
    come from the supplements. The added rows go with the supplements: ISODIV_NO_SUPPLEMENTS=1 drops both."""
    rows = list(_read_csv(Path(cldf_dir) / "languages.csv"))
    if added is not None and Path(added).exists() and not _supplements_off():
        cols = rows[0].keys() if rows else ()
        with Path(added).open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t"):
                rows.append({c: (r.get(c) or "").strip() for c in cols})
    return rows


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
    mask: Mapping[str, Iterable[str]] | None = None,
    supplements: Path | None = SUPPLEMENTS,
) -> dict[str, Doculect]:
    """Return {doculect ID: Doculect} for languages where ``keep(row)`` is true.

    ``supplements`` (config/supplements.tsv) fills concepts a list lacks, never replacing an ASJP form; it holds
    ASJPcode, so it applies to the "asjp" tokenization only."""
    cldf_dir = Path(cldf_dir)
    concepts = core_concepts(cldf_dir) if concepts is None else concepts
    langs = {r["ID"]: r for r in read_languages(cldf_dir) if keep(r) and r["ID"] not in set(exclude)}

    words: dict[str, dict[str, list[Word]]] = defaultdict(lambda: defaultdict(list))
    for r in _read_csv(cldf_dir / "forms.csv"):
        if r["Language_ID"] not in langs or r["Parameter_ID"] not in concepts:
            continue
        if drop_loans and r.get("Loan", "").lower() == "true":
            continue
        if mask and r["Parameter_ID"] in mask.get(r["Language_ID"], ()):
            continue                      # externally labelled loan (e.g. WOLD)
        tok = tokenize(r, tokenization)
        if tok:
            words[r["Language_ID"]][r["Parameter_ID"]].append(tok)
    if tokenization == "asjp":
        for lid, by_concept in load_supplements(supplements).items():
            if lid in langs:
                for c, codes in by_concept.items():
                    if c in concepts and not words[lid].get(c):
                        for code in codes:
                            tok = tokenize({"Form": code}, "asjp")
                            if tok:
                                words[lid][c].append(tok)

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
