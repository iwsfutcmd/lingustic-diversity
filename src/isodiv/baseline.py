"""Baseline for D_max: what a set of the same size scores when its members are mutually unrelated.

D_max is bounded by its value for an unrelated assortment: lects from different families sit at distance ≈ 1, so
Z = exp(−k·d) is about the same constant z̄ between every pair and D_max tends to 1/z̄ as the set grows. A real family
has internal distances below that, so it scores below the baseline whatever its size. Dividing by the baseline puts
lineages of different sizes on one scale: ``(D − 1) / (D_unrelated(n) − 1)`` is 0 for a single lineage and about 1 for
a set as diverse as the same number of unrelated lects.

``adversarial.py`` measures D_unrelated(n) for n up to 80 (one doculect from each of n random families). Beyond 80 the
random draws run out of families, so the n = 80 value is extended with the equal-similarity ceiling
n / (1 + (n − 1)·z̄), which tracks the measured values within 0.01 (1.555 vs 1.563 at n = 80 and k_cal).
"""

from __future__ import annotations

import numpy as np


def ceiling(n: float, zbar: np.ndarray | float) -> np.ndarray | float:
    """D_max of n points with every pairwise similarity equal to ``zbar``."""
    return n / (1.0 + (n - 1.0) * np.asarray(zbar))


def unrelated_baseline(n: int, sizes: list[int], measured: list[np.ndarray | float],
                       zbar: np.ndarray | float) -> np.ndarray | float:
    """D_max of ``n`` mutually unrelated lects, at each scale (arrays) or at one scale (floats).

    ``measured[i]`` is the median D_max of random unrelated sets of ``sizes[i]``; one lect scores exactly 1.
    Interpolated linearly in log n inside the measured range, extended with ``ceiling`` beyond it.
    """
    xs = np.log([1, *sizes])
    ys = [np.ones_like(np.asarray(measured[0], dtype=float)), *(np.asarray(m, dtype=float) for m in measured)]
    if n <= sizes[-1]:
        t = np.log(max(n, 1))
        j = min(max(int(np.searchsorted(xs, t, side="right")) - 1, 0), len(xs) - 2)
        w = (t - xs[j]) / (xs[j + 1] - xs[j])
        return (1 - w) * ys[j] + w * ys[j + 1]
    return ys[-1] * ceiling(n, zbar) / ceiling(sizes[-1], zbar)


def relative(d_max: np.ndarray | float, baseline: np.ndarray | float) -> np.ndarray | float:
    """Share of the unrelated-assortment maximum: (D − 1) / (D_unrelated − 1)."""
    return (np.asarray(d_max) - 1.0) / np.maximum(np.asarray(baseline) - 1.0, 1e-9)
