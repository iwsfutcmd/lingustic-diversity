"""PMI-weighted alignment of words (after Jäger 2013, 2018).

Plain Levenshtein charges 1 for every substitution. Here the score of aligning segment
a with segment b is learned from the data as pointwise mutual information

    PMI(a, b) = log p(a, b) / (q(a) q(b))

where p(a, b) is how often a and b are aligned in probable cognates and q(·) is
segment frequency. Segments that regularly *correspond* (Japonic h : p, Germanic
t : ts) score high whether or not they sound alike; chance pairings score low.
Probable cognates are found without any classification: same-concept word pairs
from similar doculects, re-selected each round by their alignment score against
the scores of different-concept (chance) pairs.

Alignment is Needleman–Wunsch with affine gaps (Gotoh), computed for a whole batch
of word pairs at once in numpy, so no compiled extension is needed.

Two doculect distances are offered, both 0 for identical lists and ≈ 1 for unrelated:

* ``pmi_ldnd`` — LDND's construction with a PMI word distance
  1 − 2·S(a, b) / (S(a, a) + S(b, b)).
* ``erc`` — per-concept empirical rank calibration (cf. Jäger 2013): rank each
  same-meaning score among that doculect pair's different-meaning scores, giving
  p_c ∈ (0, 1]; distance = 2 · mean p_c. Works with any word score (PMI or −LDN).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .asjp import Doculect, Word

GAP_OPEN = -2.49     # Jäger (2018)
GAP_EXTEND = -1.71
_NEG = -1e9
_BATCH = 4096


@dataclass(frozen=True, slots=True)
class PMIModel:
    symbols: tuple[str, ...]
    scores: np.ndarray            # (V+1, V+1); last row/column is "unknown segment", scored 0
    gap_open: float = GAP_OPEN
    gap_extend: float = GAP_EXTEND

    def encode(self, words: Sequence[Word]) -> tuple[np.ndarray, np.ndarray]:
        idx = {s: i for i, s in enumerate(self.symbols)}
        unk = len(self.symbols)
        lens = np.fromiter((len(w) for w in words), dtype=np.int64, count=len(words))
        out = np.full((len(words), max(int(lens.max(initial=0)), 1)), unk, dtype=np.int64)
        for r, w in enumerate(words):
            out[r, : len(w)] = [idx.get(s, unk) for s in w]
        return out, lens

    def to_rows(self) -> list[list[str]]:
        v = len(self.symbols)
        return [["", *self.symbols]] + [[a, *(f"{x:.4f}" for x in self.scores[i, :v])] for i, a in enumerate(self.symbols)]

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[str]], **kw) -> PMIModel:
        symbols = tuple(rows[0][1:])
        s = np.zeros((len(symbols) + 1,) * 2)
        s[:-1, :-1] = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
        return cls(symbols, s, **kw)


def _gotoh(s: np.ndarray, a: np.ndarray, la: np.ndarray, b: np.ndarray, lb: np.ndarray,
           go: float, ge: float, traceback: bool):
    """Batched global alignment with affine gaps. Returns scores, and aligned index pairs if asked."""
    n, l1, l2 = len(a), a.shape[1], b.shape[1]
    m = np.full((n, l1 + 1, l2 + 1), _NEG)
    x = np.full_like(m, _NEG)                 # a[i-1] against a gap
    y = np.full_like(m, _NEG)                 # b[j-1] against a gap
    m[:, 0, 0] = 0.0
    x[:, 1:, 0] = go + ge * np.arange(l1)
    y[:, 0, 1:] = go + ge * np.arange(l2)
    if traceback:
        pm = np.zeros((n, l1 + 1, l2 + 1), np.int8)
        px, py = np.zeros_like(pm), np.zeros_like(pm)
        px[:, 1:, 0], py[:, 0, 1:] = 1, 2
    rows = np.arange(n)
    for i in range(1, l1 + 1):
        sub = s[a[:, i - 1][:, None], b]          # (n, l2)
        prev = np.stack([m[:, i - 1, :-1], x[:, i - 1, :-1], y[:, i - 1, :-1]])
        m[:, i, 1:] = sub + prev.max(0)
        candx = np.stack([m[:, i - 1, 1:] + go, x[:, i - 1, 1:] + ge, y[:, i - 1, 1:] + go])
        x[:, i, 1:] = candx.max(0)
        if traceback:
            pm[:, i, 1:] = prev.argmax(0)
            px[:, i, 1:] = candx.argmax(0)
        for j in range(1, l2 + 1):            # y depends on its own row to the left
            c = np.stack([m[:, i, j - 1] + go, x[:, i, j - 1] + go, y[:, i, j - 1] + ge])
            y[:, i, j] = c.max(0)
            if traceback:
                py[:, i, j] = c.argmax(0)
    ends = np.stack([m[rows, la, lb], x[rows, la, lb], y[rows, la, lb]])
    score = ends.max(0)
    if not traceback:
        return score, None
    # walk back all alignments in lockstep; collect (segment a, segment b) for matched columns
    i, j, st = la.copy(), lb.copy(), ends.argmax(0)
    pairs_a, pairs_b = [], []
    while True:
        live = (i > 0) | (j > 0)
        if not live.any():
            break
        r = rows[live]
        ii, jj, ss = i[live], j[live], st[live]
        mm = ss == 0
        pairs_a.append(a[r[mm], ii[mm] - 1])
        pairs_b.append(b[r[mm], jj[mm] - 1])
        nxt = np.where(ss == 0, pm[r, ii, jj], np.where(ss == 1, px[r, ii, jj], py[r, ii, jj]))
        i[live] = ii - (ss != 2)
        j[live] = jj - (ss != 1)
        st[live] = nxt
    return score, (np.concatenate(pairs_a), np.concatenate(pairs_b))


def align(model: PMIModel, aw: Sequence[Word], bw: Sequence[Word], *, traceback: bool = False):
    """Alignment scores of aw[i] with bw[i]; with ``traceback`` also the aligned segment id pairs."""
    a, la = model.encode(aw)
    b, lb = model.encode(bw)
    order = np.lexsort((lb, la))              # similar lengths together keeps padding small
    score = np.empty(len(aw))
    pa, pb = [], []
    for k in range(0, len(order), _BATCH):
        o = order[k : k + _BATCH]
        w1, w2 = max(int(la[o].max()), 1), max(int(lb[o].max()), 1)
        sc, tb = _gotoh(model.scores, a[o, :w1], la[o], b[o, :w2], lb[o],
                        model.gap_open, model.gap_extend, traceback)
        score[o] = sc
        if tb:
            pa.append(tb[0]); pb.append(tb[1])
    if not traceback:
        return score
    return score, (np.concatenate(pa) if pa else np.zeros(0, int), np.concatenate(pb) if pb else np.zeros(0, int))


def _pmi_from_counts(counts: np.ndarray, freq: np.ndarray, pseudo: float) -> np.ndarray:
    c = counts + counts.T + pseudo            # symmetric
    p = c / c.sum()
    q = freq / freq.sum()
    s = np.zeros((len(q) + 1,) * 2)
    s[:-1, :-1] = np.log(p / np.outer(q, q))
    return s


def fit(
    candidates: Sequence[tuple[Word, Word]],
    chance: Sequence[tuple[Word, Word]],
    *,
    iterations: int = 10,
    init_ldn: float = 0.5,
    quantile: float = 0.99,
    pseudo: float = 0.1,
    gap_open: float = GAP_OPEN,
    gap_extend: float = GAP_EXTEND,
    log: callable = lambda msg: None,
) -> PMIModel:
    """Learn PMI segment scores.

    ``candidates`` are same-concept word pairs from doculects that are probably related;
    ``chance`` are different-concept pairs from the same doculects. Round 0 keeps candidates with
    normalized Levenshtein ≤ ``init_ldn``; each later round keeps those whose PMI alignment score
    beats the ``quantile`` of chance scores, then re-estimates PMI from their alignments.
    """
    from rapidfuzz.distance import Levenshtein

    symbols = tuple(sorted({s for pr in (*candidates, *chance) for w in pr for s in w}))
    v = len(symbols)
    freq = np.zeros(v)
    idx = {s: i for i, s in enumerate(symbols)}
    for pr in candidates:
        for w in pr:
            for s in w:
                freq[idx[s]] += 1
    aw = [p[0] for p in candidates]
    bw = [p[1] for p in candidates]
    keep = np.array([Levenshtein.normalized_distance(x, y) <= init_ldn for x, y in candidates])
    # round 0 aligns with a flat match/mismatch scheme on the PMI scale
    init = np.full((v + 1, v + 1), -1.0)
    np.fill_diagonal(init, 3.0)
    init[-1, :] = init[:, -1] = 0.0
    model = PMIModel(symbols, init, gap_open, gap_extend)
    for it in range(iterations):
        sel = np.flatnonzero(keep)
        _, (sa, sb) = align(model, [aw[i] for i in sel], [bw[i] for i in sel], traceback=True)
        counts = np.zeros((v, v))
        ok = (sa < v) & (sb < v)
        np.add.at(counts, (sa[ok], sb[ok]), 1)
        model = PMIModel(symbols, _pmi_from_counts(counts, freq, pseudo), gap_open, gap_extend)
        theta = np.quantile(align(model, [c[0] for c in chance], [c[1] for c in chance]), quantile)
        new = align(model, aw, bw) >= theta
        log(f"round {it}: aligned {len(sel)} pairs, threshold {theta:.2f}, next {int(new.sum())} kept")
        if it > 0 and np.array_equal(new, keep):
            break
        keep = new
    return model


# ---- doculect distances ----------------------------------------------------------------

def _shared(a: Doculect, b: Doculect, concepts: frozenset[str] | None) -> list[str]:
    keys = a.words.keys() & b.words.keys()
    if concepts is not None:
        keys &= concepts
    return sorted(keys, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))


def score_tables(model: PMIModel, a: Doculect, b: Doculect, shared: Sequence[str]):
    """(best same-concept score per concept, first-synonym score matrix, self-score lookup)."""
    aw = sorted({w for c in shared for w in a.words[c]})
    bw = sorted({w for c in shared for w in b.words[c]})
    ia = {w: i for i, w in enumerate(aw)}
    ib = {w: i for i, w in enumerate(bw)}
    grid = align(model, [x for x in aw for _ in bw], [y for _ in aw for y in bw]).reshape(len(aw), len(bw))
    self_a = dict(zip(aw, align(model, aw, aw)))
    self_b = dict(zip(bw, align(model, bw, bw)))
    return grid, ia, ib, self_a, self_b


def _tables(model, a, b, min_shared, concepts):
    shared = _shared(a, b, concepts)
    if len(shared) < min_shared:
        return None
    grid, ia, ib, sa, sb = score_tables(model, a, b, shared)
    first = np.array([[grid[ia[a.words[c][0]], ib[b.words[d][0]]] for d in shared] for c in shared])
    return shared, grid, ia, ib, sa, sb, first


def pmi_ldnd(model: PMIModel, a: Doculect, b: Doculect, *, min_shared: int = 20,
             concepts: frozenset[str] | None = None) -> float:
    t = _tables(model, a, b, min_shared, concepts)
    if t is None:
        return float("nan")
    shared, grid, ia, ib, sa, sb, _ = t
    def dist(x, y):
        return 1.0 - 2.0 * grid[ia[x], ib[y]] / (sa[x] + sb[y])
    same = np.mean([min(dist(x, y) for x in a.words[c] for y in b.words[c]) for c in shared])
    firsts = [(a.words[c][0], b.words[c][0]) for c in shared]
    n = len(shared)
    diff = np.mean([dist(firsts[i][0], firsts[j][1]) for i in range(n) for j in range(n) if i != j])
    return float(same / diff) if diff > 0 else float("nan")


def erc(same: np.ndarray, first: np.ndarray) -> float:
    """Empirical rank calibration. ``same[c]`` is the best same-meaning similarity for concept c,
    ``first`` the first-synonym similarity matrix; off-diagonal cells are the chance sample."""
    n = len(same)
    off = np.sort(first[~np.eye(n, dtype=bool)])
    ge = len(off) - np.searchsorted(off, same, side="left")    # chance pairs at least as similar
    p = (ge + 1) / (len(off) + 1)
    return float(2.0 * p.mean())


def pmi_erc(model: PMIModel, a: Doculect, b: Doculect, *, min_shared: int = 20,
            concepts: frozenset[str] | None = None) -> float:
    t = _tables(model, a, b, min_shared, concepts)
    if t is None:
        return float("nan")
    shared, grid, ia, ib, *_, first = t
    same = np.array([max(grid[ia[x], ib[y]] for x in a.words[c] for y in b.words[c]) for c in shared])
    return erc(same, first)


def ldn_erc(a: Doculect, b: Doculect, *, min_shared: int = 20, concepts: frozenset[str] | None = None) -> float:
    """The same rank calibration on plain normalized Levenshtein (similarity = −LDN), as a control."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    shared = _shared(a, b, concepts)
    if len(shared) < min_shared:
        return float("nan")
    same = -np.array([process.cdist(a.words[c], b.words[c], scorer=Levenshtein.normalized_distance).min()
                      for c in shared])
    first = -process.cdist([a.words[c][0] for c in shared], [b.words[c][0] for c in shared],
                           scorer=Levenshtein.normalized_distance).astype(float)
    return erc(same, first)


def word_pairs(docs: Iterable[tuple[Doculect, Doculect]]) -> tuple[list[tuple[Word, Word]], list[tuple[Word, Word]]]:
    """Same-concept (all synonym combinations) and different-concept (first synonyms) word pairs."""
    same, diff = [], []
    for a, b in docs:
        shared = _shared(a, b, None)
        for c in shared:
            same.extend((x, y) for x in a.words[c] for y in b.words[c])
        for c in shared:
            for d in shared:
                if c != d:
                    diff.append((a.words[c][0], b.words[d][0]))
    return same, diff
