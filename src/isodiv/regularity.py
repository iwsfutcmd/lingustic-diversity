"""Evidence of relatedness from recurrent sound correspondences, tested by permutation.

The comparative method shows two languages are related by *regular* correspondences: the same sound pair recurs
across many word pairs with the same meaning, more than it would between words of different meanings. This module
measures that directly on a pair of word lists, without a tree:

1. Align every word of list A with every word of list B (all concept pairs, global PMI scores, affine gaps), keeping
   each alignment's columns as correspondence ids (segment of A, segment of B or a gap).
2. For a matching of A's concepts to B's (the real one: same meaning), count every correspondence over the matched
   word pairs. A word pair scores the mean, over its consonant columns, of log(1 + how often that correspondence
   occurs in the *other* matched pairs). Leaving the word out means a correspondence must recur to count.
3. Shuffle which concepts are matched and recompute. The real statistic's standing against the shuffles is a
   permutation z-score and p-value (Ringe 1992; Kessler 2001). The shuffles keep each list's sounds and word shapes,
   so chance resemblance and shared phonotactics are in the baseline.

The same permutation test is applied to the plain PMI word similarity, for comparison. Vowels are left out of the
recurrence statistic: they change fast and are few, so vowel correspondences recur by chance.

Pair-specific correspondence learning (realigning with the recurrent correspondences, as LexStat does) is a possible
refinement; here the alignments use the global scores only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .asjp import Doculect
from .pmi import PMIModel

VOWELS = frozenset("ieE3auo")

try:
    import numba as _numba
except ImportError:          # pragma: no cover
    _numba = None


def _jit(f):
    return _numba.njit(cache=True)(f) if _numba is not None else f


@_jit
def _align_tb(s, a, b, go, ge, gap, out_a, out_b):
    """Global alignment of one word pair with affine gaps; writes columns (gap index for a gap) and returns
    (score, number of columns)."""
    la, lb = len(a), len(b)
    neg = -1e9
    m = np.full((la + 1, lb + 1), neg); x = np.full((la + 1, lb + 1), neg); y = np.full((la + 1, lb + 1), neg)
    pm = np.zeros((la + 1, lb + 1), np.int8); px = np.zeros((la + 1, lb + 1), np.int8)
    py = np.zeros((la + 1, lb + 1), np.int8)
    m[0, 0] = 0.0
    for i in range(1, la + 1):
        x[i, 0] = go + ge * (i - 1); px[i, 0] = 1
    for j in range(1, lb + 1):
        y[0, j] = go + ge * (j - 1); py[0, j] = 2
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            best, k = m[i - 1, j - 1], 0
            if x[i - 1, j - 1] > best:
                best, k = x[i - 1, j - 1], 1
            if y[i - 1, j - 1] > best:
                best, k = y[i - 1, j - 1], 2
            m[i, j] = best + s[a[i - 1], b[j - 1]]; pm[i, j] = k
            best, k = m[i - 1, j] + go, 0
            if x[i - 1, j] + ge > best:
                best, k = x[i - 1, j] + ge, 1
            if y[i - 1, j] + go > best:
                best, k = y[i - 1, j] + go, 2
            x[i, j] = best; px[i, j] = k
            best, k = m[i, j - 1] + go, 0
            if x[i, j - 1] + go > best:
                best, k = x[i, j - 1] + go, 1
            if y[i, j - 1] + ge > best:
                best, k = y[i, j - 1] + ge, 2
            y[i, j] = best; py[i, j] = k
    st, score = 0, m[la, lb]
    if x[la, lb] > score:
        st, score = 1, x[la, lb]
    if y[la, lb] > score:
        st, score = 2, y[la, lb]
    i, j, n = la, lb, 0
    while i > 0 or j > 0:
        if st == 0:
            out_a[n] = a[i - 1]; out_b[n] = b[j - 1]; nxt = pm[i, j]; i -= 1; j -= 1
        elif st == 1:
            out_a[n] = a[i - 1]; out_b[n] = gap; nxt = px[i, j]; i -= 1
        else:
            out_a[n] = gap; out_b[n] = b[j - 1]; nxt = py[i, j]; j -= 1
        st = nxt; n += 1
    return score, n


def _pjit(f):
    return _numba.njit(cache=True, parallel=True)(f) if _numba is not None else f


@_pjit
def _align_all(s, A, la, B, lb, go, ge, gap, nsym, vowel, width, ids, keep, length, sim):
    """Align every word of A with every word of B (one word per concept, padded rows). Pair p = i*n + j is stored
    at ids[p*width : p*width + length[p]]; sim[p] = 2·score / (self(A_i) + self(B_j))."""
    n = len(la)
    selfa = np.empty(n); selfb = np.empty(n)
    for i in range(n):
        oa = np.empty(2 * la[i] + 2, np.int64); ob = np.empty(2 * la[i] + 2, np.int64)
        selfa[i] = _align_tb(s, A[i, :la[i]], A[i, :la[i]], go, ge, gap, oa, ob)[0]
        oa = np.empty(2 * lb[i] + 2, np.int64); ob = np.empty(2 * lb[i] + 2, np.int64)
        selfb[i] = _align_tb(s, B[i, :lb[i]], B[i, :lb[i]], go, ge, gap, oa, ob)[0]
    for i in _numba.prange(n) if _numba is not None else range(n):
        oa = np.empty(width, np.int64); ob = np.empty(width, np.int64)
        for j in range(n):
            p = i * n + j
            sc, k = _align_tb(s, A[i, :la[i]], B[j, :lb[j]], go, ge, gap, oa, ob)
            den = selfa[i] + selfb[j]
            sim[p] = 2 * sc / den if den > 0 else 0.0
            length[p] = k
            base = p * width
            for t in range(k):
                ids[base + t] = oa[t] * nsym + ob[t]
                keep[base + t] = not (vowel[oa[t]] and vowel[ob[t]])


@_jit
def _stats(perm, start, length, ids, keep, sim, ncorr, out, expect):
    """For a matching ``perm`` (A concept i ↔ B concept perm[i]) return (recurrence, similarity, over-representation G).

    G = Σ O_c·log(O_c / E_c) over consonant correspondences c seen at least twice in the matched pairs, where E_c is
    c's count over all n×n concept pairs divided by n (its expected count under a random matching).

    ``start``/``length`` locate each of the n×n alignments (flattened i*n + j) in ``ids`` (correspondence ids) and
    ``keep`` (whether a column counts: at least one consonant). ``out`` is a scratch count array of size ncorr."""
    n = len(perm)
    out[:] = 0
    for i in range(n):
        p = i * n + perm[i]
        for t in range(start[p], start[p] + length[p]):
            if keep[t]:
                out[ids[t]] += 1
    rec, simsum = 0.0, 0.0
    for i in range(n):
        p = i * n + perm[i]
        tot, cnt = 0.0, 0
        for t in range(start[p], start[p] + length[p]):
            if keep[t]:
                own = 0
                for u in range(start[p], start[p] + length[p]):
                    if keep[u] and ids[u] == ids[t]:
                        own += 1
                tot += np.log1p(out[ids[t]] - own)     # occurrences in the other matched pairs
                cnt += 1
        if cnt:
            rec += tot / cnt
        simsum += sim[p]
    g = 0.0
    for c in range(ncorr):
        if out[c] >= 2:
            g += out[c] * np.log(out[c] / expect[c])
    return rec, simsum, g


@dataclass(frozen=True)
class RegularityResult:
    n: int                 # shared concepts
    rec: float             # recurrence statistic, real matching
    z_rec: float
    p_rec: float
    sim: float             # summed PMI word similarity, real matching
    z_sim: float
    p_sim: float
    g: float               # over-representation of recurrent correspondences, real matching
    z_g: float
    p_g: float


class Regularity:
    """Permutation tests of relatedness for pairs of word lists with one PMI model."""

    def __init__(self, model: PMIModel):
        self.model = model
        self.sym = {c: i for i, c in enumerate(model.symbols)}
        self.unk = len(model.symbols)
        self.gap = self.unk + 1
        self.nsym = self.unk + 2
        vow = np.zeros(self.nsym, bool)
        for c, i in self.sym.items():
            vow[i] = c in VOWELS
        self.vowel = vow
        self.vowel[self.gap] = True               # gap–vowel columns are left out too
        self._self: dict = {}

    def _enc(self, w) -> np.ndarray:
        return np.array([self.sym.get(c, self.unk) for c in w], dtype=np.int64)

    def _selfscore(self, w) -> float:
        if w not in self._self:
            e = self._enc(w); buf = np.empty(2 * len(e) + 2, np.int64)
            self._self[w] = _align_tb(self.model.scores, e, e, self.model.gap_open, self.model.gap_extend,
                                      self.gap, buf, buf.copy())[0]
        return self._self[w]

    def prepare(self, a: Doculect, b: Doculect, concepts: Sequence[str] | None = None):
        """Alignments of every concept pair (best synonym pair by PMI word similarity)."""
        shared = sorted(a.words.keys() & b.words.keys()) if concepts is None else list(concepts)
        n = len(shared)
        s, go, ge = self.model.scores, self.model.gap_open, self.model.gap_extend
        off = np.zeros(n * n + 1, np.int64)
        ids_l, keep_l, sim = [], [], np.empty(n * n)
        for i, c in enumerate(shared):
            for j, d in enumerate(shared):
                best = None
                for wa in a.words[c]:
                    for wb in b.words[d]:
                        ea, eb = self._enc(wa), self._enc(wb)
                        oa = np.empty(len(ea) + len(eb), np.int64); ob = np.empty_like(oa)
                        sc, k = _align_tb(s, ea, eb, go, ge, self.gap, oa, ob)
                        den = self._selfscore(wa) + self._selfscore(wb)
                        v = 2 * sc / den if den > 0 else 0.0
                        if best is None or v > best[0]:
                            best = (v, oa[:k].copy(), ob[:k].copy())
                v, oa, ob = best
                p = i * n + j
                sim[p] = v
                ids_l.append(oa * self.nsym + ob)
                keep_l.append(~(self.vowel[oa] & self.vowel[ob]))   # a column with at least one consonant
                off[p + 1] = off[p] + len(oa)
        return shared, off, np.concatenate(ids_l), np.concatenate(keep_l), sim

    def test(self, a: Doculect, b: Doculect, *, perms: int = 999, seed: int = 0, min_shared: int = 20,
             concepts: Sequence[str] | None = None) -> RegularityResult | None:
        """Permutation tests for two ASJP doculects (best synonym pair per concept pair)."""
        shared, off, ids, keep, sim = self.prepare(a, b, concepts)
        if len(shared) < min_shared:
            return None
        return self._permute(len(shared), off[:-1], np.diff(off), ids, keep, sim, perms, seed)

    def test_lists(self, wa: Sequence[str], wb: Sequence[str], *, perms: int = 999, seed: int = 0) -> RegularityResult:
        """Permutation tests for two aligned lists of ASJP strings, one word per concept (wa[i] and wb[i] share a
        meaning). Built for long lists: all n² alignments run compiled and in parallel."""
        n = len(wa)
        ea = [self._enc(w) for w in wa]; eb = [self._enc(w) for w in wb]
        la = np.array([len(x) for x in ea], np.int64); lb = np.array([len(x) for x in eb], np.int64)
        A = np.zeros((n, max(1, la.max())), np.int64); B = np.zeros((n, max(1, lb.max())), np.int64)
        for i in range(n):
            A[i, :la[i]] = ea[i]; B[i, :lb[i]] = eb[i]
        width = int(la.max() + lb.max())
        ids = np.zeros(n * n * width, np.int16); keep = np.zeros(n * n * width, np.bool_)
        length = np.zeros(n * n, np.int64); sim = np.empty(n * n)
        _align_all(self.model.scores, A, la, B, lb, self.model.gap_open, self.model.gap_extend, self.gap, self.nsym,
                   self.vowel, width, ids, keep, length, sim)
        start = np.arange(n * n, dtype=np.int64) * width
        return self._permute(n, start, length, ids, keep, sim, perms, seed)

    def _permute(self, n, start, length, ids, keep, sim, perms, seed) -> RegularityResult:
        rng = np.random.default_rng(seed)
        scratch = np.zeros(self.nsym * self.nsym, np.int64)
        valid = np.zeros(len(ids), np.bool_)
        for p in range(len(start)):                          # columns that belong to an alignment
            valid[start[p]:start[p] + length[p]] = True
        expect = np.bincount(ids[keep & valid].astype(np.int64), minlength=len(scratch)) / n
        rec0, sim0, g0 = _stats(np.arange(n), start, length, ids, keep, sim, len(scratch), scratch, expect)
        null = np.empty((perms, 3))
        for k in range(perms):
            null[k] = _stats(rng.permutation(n), start, length, ids, keep, sim, len(scratch), scratch, expect)

        def zp(x, col):
            mu, sd = null[:, col].mean(), null[:, col].std()
            return (x - mu) / sd if sd > 0 else 0.0, (1 + (null[:, col] >= x).sum()) / (perms + 1)

        zr, pr = zp(rec0, 0); zs, ps = zp(sim0, 1); zg, pg = zp(g0, 2)
        return RegularityResult(n, float(rec0), float(zr), float(pr), float(sim0), float(zs), float(ps),
                                float(g0), float(zg), float(pg))
