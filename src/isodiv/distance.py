"""LDND — Levenshtein Distance Normalized Divided (Wichmann et al. 2010).

For doculects A, B with shared concepts C:

    LDN(a, b)  = Levenshtein(a, b) / max(|a|, |b|)
    LDN_same   = mean over c ∈ C of min over synonym pairs LDN(a_c, b_c)
    LDN_diff   = mean over c ≠ c' ∈ C of LDN(first(a_c), first(b_c'))
    LDND       = LDN_same / LDN_diff

Dividing by LDN_diff corrects for chance similarity due to shared phonotactics
and inventory size. LDND ≈ 1 for unrelated languages and 0 for identical lists.
No time depth and no rate assumption is involved anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .asjp import Doculect

_LDN = Levenshtein.normalized_distance


def ldnd(
    a: Doculect, b: Doculect, *, min_shared: int = 20, concepts: frozenset[str] | None = None
) -> float:
    keys = a.words.keys() & b.words.keys()
    if concepts is not None:
        keys &= concepts
    shared = sorted(keys, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))
    if len(shared) < min_shared:
        return float("nan")

    same = np.fromiter(
        (process.cdist(a.words[c], b.words[c], scorer=_LDN).min() for c in shared),
        dtype=float,
        count=len(shared),
    ).mean()

    firsts_a = [a.words[c][0] for c in shared]
    firsts_b = [b.words[c][0] for c in shared]
    full = process.cdist(firsts_a, firsts_b, scorer=_LDN).astype(float)
    off_diag = ~np.eye(len(shared), dtype=bool)
    diff = full[off_diag].mean()

    return float(same / diff) if diff > 0 else float("nan")


def ldnd_matrix(
    doculects: Sequence[Doculect], *, min_shared: int = 20, concepts: frozenset[str] | None = None
) -> np.ndarray:
    n = len(doculects)
    d = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d[i, j] = d[j, i] = ldnd(doculects[i], doculects[j], min_shared=min_shared, concepts=concepts)
    return d
