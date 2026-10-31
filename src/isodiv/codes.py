"""Pick doculects by ISO 639-1, ISO 639-3 or Glottocode.

ASJP gives each doculect a Glottocode and an ISO 639-3 code. A two-letter ISO 639-1 code is mapped to its 639-3 code
(``config/iso639_1.tsv``). A macrolanguage code (``zho``, ``ara``, ``msa``) that ASJP does not use itself stands for its
member languages (``config/iso639_macro.tsv``, from the ISO 639-3 registry), so ``zh`` finds Mandarin, Yue, Wu, Min and
the other Chinese varieties. One code can match several doculects (two transcriptions of one language, or a dialect
continuum), so ``resolve`` returns all of them.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent.parent / "config"
GLOTTOCODE = re.compile(r"^[a-z]{4}\d{4}$")


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_iso1(path: Path = CONFIG / "iso639_1.tsv") -> dict[str, str]:
    """ISO 639-1 -> ISO 639-3."""
    return {r["iso639_1"]: r["iso639_3"] for r in _tsv(path)}


def load_macro(path: Path = CONFIG / "iso639_macro.tsv") -> dict[str, list[str]]:
    """ISO 639-3 macrolanguage -> its member ISO 639-3 codes."""
    out: dict[str, list[str]] = defaultdict(list)
    for r in _tsv(path):
        out[r["macrolanguage"]].append(r["member"])
    return dict(out)


@dataclass(frozen=True)
class Match:
    kind: str                 # "glottocode", "iso639-1", "iso639-3", "macrolanguage" or "name"
    query: str
    iso639_3: tuple[str, ...]  # the 639-3 codes actually searched
    ids: tuple[str, ...]      # matching doculect IDs, in the order of the languages table


def resolve(query: str, languages: Iterable[Mapping[str, str]], iso1: Mapping[str, str] | None = None,
            macro: Mapping[str, list[str]] | None = None) -> Match:
    """Doculects matching a Glottocode, ISO 639-1/-3 code, doculect ID, or (failing those) a name.

    ``languages`` are rows of ASJP's languages.csv (ID, Name, Glottocode, Glottolog_Name, ISO639P3code).
    """
    iso1 = load_iso1() if iso1 is None else iso1
    macro = load_macro() if macro is None else macro
    rows = list(languages)
    q = query.strip()
    low = q.lower()
    if not q:
        return Match("name", q, (), ())
    if GLOTTOCODE.match(low):
        return Match("glottocode", q, (), tuple(r["ID"] for r in rows if r["Glottocode"] == low))
    if len(low) in (2, 3) and low.isalpha() and low.isascii():
        code, kind = low, "iso639-3"
        if len(low) == 2:
            if low not in iso1:
                return Match("iso639-1", q, (), ())
            code, kind = iso1[low], "iso639-1"
        ids = tuple(r["ID"] for r in rows if r["ISO639P3code"] == code)
        if ids:
            return Match(kind, q, (code,), ids)
        members = tuple(macro.get(code, ()))
        if members:
            found = tuple(r["ID"] for r in rows if r["ISO639P3code"] in members)
            return Match("macrolanguage", q, members, found)
        return Match(kind, q, (code,), ())
    exact = tuple(r["ID"] for r in rows if r["ID"].lower() == low)
    if exact:
        return Match("name", q, (), exact)
    return Match("name", q, (), tuple(r["ID"] for r in rows
                                      if low in r["Name"].lower() or low in r["Glottolog_Name"].lower()))
