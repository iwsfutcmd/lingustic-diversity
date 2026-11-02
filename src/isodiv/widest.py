"""The widest pair of a lineage on the distance scale, robust to stray lists, using no internal classification.

The largest single distance between two lists is mostly the noise tail: 40-item lists are noisy, and a lineage with
hundreds of lists has tens of thousands of pairs, so the extreme is two ordinary lists that happen to land above the
unrelated level. Here each list is represented by its neighbourhood: the list and up to ``k`` nearest lists that are
within the same-language noise band of it (``band``, the 90th percentile of distances between two lists of one
language). The distance between two lists is the median over their neighbourhoods' pairs, and the widest pair is the
largest of those. A list with no relative inside the band is "lone"; whether a lone list may count is decided outside
this module (``scripts/widest.py`` admits it unless it is closer to a list of another family than to its own, the
signature of contact or mixing rather than of an old branch).
"""

from __future__ import annotations

import warnings

import numpy as np


def neighbourhoods(d: np.ndarray, band: float, k: int = 4) -> list[np.ndarray]:
    """Each list with up to ``k`` nearest lists at distance ≤ ``band`` (diagonal ignored)."""
    d = np.array(d, dtype=float)
    np.fill_diagonal(d, np.inf)
    out = []
    for x in range(d.shape[0]):
        near = [int(y) for y in np.argsort(np.nan_to_num(d[x], nan=np.inf))[:k] if d[x, y] <= band]
        out.append(np.array([x, *near], dtype=int))
    return out


def robust_distances(d: np.ndarray, nb: list[np.ndarray]) -> np.ndarray:
    """r[a, b] = median of d over N(a) × N(b), leaving out the pairs of a list with itself."""
    n = d.shape[0]
    d = np.array(d, dtype=float)
    np.fill_diagonal(d, np.nan)
    width = max(len(v) for v in nb)
    cols = np.full((n, width), -1)
    for b, v in enumerate(nb):
        cols[b, : len(v)] = v
    valid = cols >= 0
    r = np.full((n, n), np.nan)
    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)      # a list against itself
    for a in range(n):
        block = d[nb[a]][:, np.where(valid, cols, 0)]          # (|N(a)|, n, width)
        block = np.where(valid[None], block, np.nan)
        r[a] = np.nanmedian(block.reshape(len(nb[a]), n, width).transpose(1, 0, 2).reshape(n, -1), axis=1)
    np.fill_diagonal(r, np.nan)
    return r


def widest(r: np.ndarray, allowed: np.ndarray | None = None) -> tuple[float, int, int]:
    """Largest r over pairs of allowed lists, and the pair; (nan, -1, -1) when there is no pair."""
    r = np.array(r, dtype=float)
    if allowed is not None:
        r[~allowed, :] = np.nan
        r[:, ~allowed] = np.nan
    if not np.isfinite(r).any():
        return float("nan"), -1, -1
    a, b = np.unravel_index(np.nanargmax(r), r.shape)
    return float(r[a, b]), int(min(a, b)), int(max(a, b))
