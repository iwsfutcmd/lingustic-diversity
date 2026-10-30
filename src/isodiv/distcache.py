"""Pairwise doculect distances, each computed once.

Nested lineages (Oceanic inside Malayo-Polynesian inside Austronesian) and the later pipeline steps (audit) ask for
overlapping sets of doculects. ``DistCache`` holds one symmetric float32 matrix over a fixed list of doculect ids,
fills in only the pairs a request is missing, and can be saved and reloaded. A pair whose lists share too few concepts
is stored as NaN (computed, undefined); a pair not yet computed is stored as -1.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .asjp import Doculect
from .metric import Metric

UNSET = np.float32(-1.0)


class DistCache:
    def __init__(self, metric: Metric, ids: Sequence[str]):
        self.metric = metric
        self.ids = list(ids)
        self.pos = {i: k for k, i in enumerate(self.ids)}
        n = len(self.ids)
        self.d = np.full((n, n), UNSET, dtype=np.float32)
        np.fill_diagonal(self.d, 0.0)

    def matrix(self, docs: Sequence[Doculect]) -> np.ndarray:
        """Distance matrix of ``docs`` (float64), computing the pairs not seen before."""
        idx = np.array([self.pos[x.id] for x in docs], dtype=int)
        sub = self.d[np.ix_(idx, idx)]
        ii, jj = np.nonzero(np.triu(sub == UNSET, 1))
        if len(ii):
            v = self.metric.values([(docs[i], docs[j]) for i, j in zip(ii, jj)])
            a, b = idx[ii], idx[jj]
            self.d[a, b] = self.d[b, a] = v.astype(np.float32)
        return self.d[np.ix_(idx, idx)].astype(np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.d)
        path.with_suffix(".json").write_text(json.dumps({"distance": self.metric.name, "ids": self.ids}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, metric: Metric, ids: Sequence[str]) -> "DistCache":
        """Reload a saved cache for ``metric``; ids not in the file start empty. A cache saved with another distance
        is ignored."""
        cache = cls(metric, ids)
        meta = path.with_suffix(".json")
        if not (path.exists() and meta.exists()):
            return cache
        info = json.loads(meta.read_text(encoding="utf-8"))
        if info["distance"] != metric.name:
            return cache
        old = np.load(path, mmap_mode="r")
        common = [(cache.pos[i], k) for k, i in enumerate(info["ids"]) if i in cache.pos]
        if common:
            new, src = (np.array(x) for x in zip(*common))
            cache.d[np.ix_(new, new)] = old[np.ix_(src, src)]
        return cache
