"""The similarity used for the headline D_max: anchored at both ends by evidence.

Similarity is 1 at the median distance between two lists of the same language (the noise floor) and falls linearly
to 0 at the detection limit: the distance at which the permutation test of relatedness (scripts/regularity_eval.py)
detects a related pair half the time. Pairs farther apart than that count as unrelated, so D_max of n lists that
cannot be tied together is n, and a lineage's D_max is an effective number of lineages its word lists cannot tie
together. The older exponential similarity exp(−k·d) with a calibrated k stays available for comparison.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .diversity import anchored_similarity


@dataclass(frozen=True)
class Kernel:
    same: float           # similarity 1 at and below this distance
    unrelated: float      # similarity 0 at and beyond this distance

    def __call__(self, d: np.ndarray) -> np.ndarray:
        return anchored_similarity(d, self.same, self.unrelated)

    def to_json(self) -> dict:
        return {"type": "anchored", **asdict(self)}

    @classmethod
    def from_summary(cls, summary: dict) -> "Kernel":
        k = summary.get("kernel")
        if not k:
            raise SystemExit("results/summary.json has no kernel; rerun scripts/run_analysis.py")
        return cls(k["same"], k["unrelated"])


def detection_limit(path: Path) -> tuple[float, float]:
    """(distance at which related pairs are detected half the time, logistic slope), fitted on the related pairs of
    results/regularity_eval.csv (detected = permutation p < 0.05)."""
    from scipy.optimize import minimize

    rows = [r for r in csv.DictReader(Path(path).open(encoding="utf-8")) if r["group"] == "related"]
    d = np.array([float(r["d"]) for r in rows]); y = np.array([float(r["p_sim"]) < 0.05 for r in rows], float)

    def nll(t):
        p = np.clip(1 / (1 + np.exp(-(t[0] - t[1] * d))), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    a, b = minimize(nll, [50.0, 55.0]).x
    return float(a / b), float(b)


def load_detection_limit(results: Path = Path("results")) -> float:
    """The detection limit recorded by scripts/regularity_eval.py."""
    p = Path(results) / "regularity_eval.json"
    if not p.exists():
        raise SystemExit("results/regularity_eval.json missing: run scripts/regularity_eval.py first")
    info = json.loads(p.read_text(encoding="utf-8"))
    if "detection_limit_d50" not in info:
        info["detection_limit_d50"], info["detection_slope"] = detection_limit(Path(results) / "regularity_eval.csv")
    return float(info["detection_limit_d50"])


def similarity(d: np.ndarray, k: "Kernel | float") -> np.ndarray:
    """Similarity matrix for either the headline Kernel or a scale k of the older exp(−k·d)."""
    from .diversity import similarity_matrix
    return k(d) if isinstance(k, Kernel) else similarity_matrix(d, k)
