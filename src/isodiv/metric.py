"""One switch for the doculect distance used by the pipeline scripts.

``ldnd`` is the original LDND (Wichmann et al. 2010). ``pmi`` is PMI-LDND: the same construction
with word distances from learned sound-correspondence scores (see ``pmi``), which separates
relatedness levels better on ASJP (``scripts/pmi_eval.py``).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np

from .asjp import Doculect
from .distance import PairTable, from_table, ldn_table
from .pmi import PMIModel, pmi_ldnd_values, pmi_tables

DISTANCES = ("pmi", "ldnd")
LABELS = {"pmi": "PMI-LDND", "ldnd": "LDND"}


@dataclass(frozen=True)
class Metric:
    name: str
    model: PMIModel | None = None
    workers: int | None = None

    @property
    def label(self) -> str:
        return LABELS[self.name]

    def tables(self, pairs: Sequence[tuple[Doculect, Doculect]]) -> list[PairTable | None]:
        if self.name == "pmi":
            return pmi_tables(self.model, pairs, workers=self.workers)
        return [ldn_table(a, b) for a, b in pairs]

    def values(self, pairs: Sequence[tuple[Doculect, Doculect]], *, min_shared: int = 20,
               concepts: frozenset[str] | None = None) -> np.ndarray:
        if self.name == "pmi":
            return pmi_ldnd_values(self.model, pairs, min_shared=min_shared, concepts=concepts, workers=self.workers)
        return np.array([from_table(ldn_table(a, b), min_shared=min_shared, concepts=concepts) for a, b in pairs],
                        dtype=float)

    def matrix(self, docs: Sequence[Doculect], **kw) -> np.ndarray:
        n = len(docs)
        d = np.zeros((n, n))
        idx = list(combinations(range(n), 2))
        if idx:
            v = self.values([(docs[i], docs[j]) for i, j in idx], **kw)
            i, j = np.array(idx).T
            d[i, j] = d[j, i] = v
        return d

    def cross(self, a: Sequence[Doculect], b: Sequence[Doculect], **kw) -> np.ndarray:
        return self.values(list(product(a, b)), **kw).reshape(len(a), len(b))


def matrix_from_tables(tables: Sequence[PairTable | None], n: int, **kw) -> np.ndarray:
    """Distance matrix from the tables of all ``combinations(range(n), 2)``, e.g. for a concept subset."""
    d = np.zeros((n, n))
    for (i, j), t in zip(combinations(range(n), 2), tables):
        d[i, j] = d[j, i] = from_table(t, **kw)
    return d


def load_model(path: Path) -> PMIModel:
    if not Path(path).exists():
        raise SystemExit(f"{path} not found: run `python scripts/pmi_train.py` first (or pass --distance ldnd)")
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return PMIModel.from_rows(list(csv.reader(fh, delimiter="\t")))


def get_metric(name: str, model_path: Path | None = None, workers: int | None = None) -> Metric:
    if name == "pmi":
        return Metric(name, load_model(model_path or Path("results/pmi_model_all.tsv")), workers)
    if name == "ldnd":
        return Metric(name)
    raise ValueError(f"unknown distance {name!r}; choose from {DISTANCES}")
