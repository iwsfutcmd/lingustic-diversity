"""Similarity-sensitive diversity (Leinster & Cobbold 2012) and its maximum
(Leinster & Meckes 2016).

Given doculects with pairwise distances d and a scale k, set Z = exp(-k·d).
For a distribution p over doculects the order-q diversity is

    D_q(p) = ( Σ_i p_i (Zp)_i^{q-1} )^{1/(1-q)}          (q ≠ 1, ∞)

Leinster & Meckes prove that for symmetric Z there is one distribution that
maximizes D_q for *every* q simultaneously, and that

    D_max(Z) = max { |Z_B| : B ⊆ doculects, Z_B has a nonnegative weighting }

where a weighting w solves Z_B w = 1 and |Z_B| = Σ w. D_max is a property of
Z alone — no abundances, no q — and duplicating a doculect (an extra "dialect"
at distance 0) cannot change it. That is the property we want: it is immune to
the language/dialect counting problem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

EXACT_MAX_N = 14
CG_MIN_N = 120      # above this, solve on a small support and grow it (column generation)


@dataclass(frozen=True, slots=True)
class MaxDiversity:
    value: float          # D_max: effective number of fully distinct lineages
    weights: np.ndarray   # maximizing distribution p (sums to 1)
    method: str           # "exact" or "qp"
    kkt_ok: bool          # first-order optimality verified


def similarity_matrix(d: np.ndarray, k: float) -> np.ndarray:
    if np.isnan(d).any():
        raise ValueError("distance matrix contains NaN; drop doculects with too little overlap first")
    z = np.exp(-k * np.clip(d, 0.0, None))
    np.fill_diagonal(z, 1.0)
    return (z + z.T) / 2


def hill_similarity(z: np.ndarray, p: np.ndarray, q: float = 2.0) -> float:
    """Leinster–Cobbold D_q^Z(p)."""
    support = p > 0
    zp = (z @ p)[support]
    ps = p[support]
    if np.isinf(q):
        return float(1.0 / zp.max())
    if np.isclose(q, 1.0):
        return float(np.exp(-np.sum(ps * np.log(zp))))
    return float(np.sum(ps * zp ** (q - 1)) ** (1.0 / (1.0 - q)))


def uniform_diversity(z: np.ndarray, q: float = 2.0) -> float:
    """D_q^Z with equal weight on every doculect — the naive, dialect-sensitive figure."""
    n = z.shape[0]
    return hill_similarity(z, np.full(n, 1.0 / n), q)


def _weighting(zb: np.ndarray) -> np.ndarray | None:
    try:
        w = np.linalg.solve(zb, np.ones(zb.shape[0]))
    except np.linalg.LinAlgError:
        return None
    return w if np.all(w >= -1e-12) else None


def _kkt(z: np.ndarray, p: np.ndarray, tol: float = 1e-6) -> bool:
    """min pᵀZp on the simplex: (Zp)_i equal on the support and ≥ that value off it."""
    zp = z @ p
    on = p > 1e-9
    lam = zp[on].mean()
    return bool(np.allclose(zp[on], lam, atol=tol) and np.all(zp[~on] >= lam - tol))


def _exact(z: np.ndarray) -> MaxDiversity:
    n = z.shape[0]
    best_val, best_p = 0.0, None
    for size in range(1, n + 1):
        for b in combinations(range(n), size):
            idx = np.array(b)
            w = _weighting(z[np.ix_(idx, idx)])
            if w is None:
                continue
            mag = float(w.sum())
            if mag > best_val + 1e-12:
                p = np.zeros(n)
                p[idx] = np.clip(w, 0, None) / mag
                best_val, best_p = mag, p
    return MaxDiversity(best_val, best_p, "exact", _kkt(z, best_p))


def _qp(z: np.ndarray, *, restarts: int = 24, seed: int = 0, warm: Sequence[np.ndarray] = ()) -> MaxDiversity:
    """Max D_2 ⇔ min pᵀZp on the simplex; by Leinster–Meckes that maximizer is
    optimal for every q. Z need not be PSD, so use multi-start + polishing.
    ``warm`` adds starting distributions (e.g. the optimum of a sub-collection, which makes the result at
    least as large as that sub-collection's D_max)."""
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    starts = [np.full(n, 1.0 / n), *rng.dirichlet(np.full(n, 0.5), size=restarts), *warm]
    cons = ({"type": "eq", "fun": lambda p: p.sum() - 1.0, "jac": lambda p: np.ones_like(p)},)

    def solve(p0: np.ndarray) -> np.ndarray:
        res = minimize(
            lambda p: p @ z @ p,
            p0,
            jac=lambda p: 2 * z @ p,
            bounds=[(0.0, 1.0)] * n,
            constraints=cons,
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 2000},
        )
        return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()

    def polish(p: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(p > 1e-6)
        w = _weighting(z[np.ix_(idx, idx)])
        if w is None:
            return p
        out = np.zeros(n)
        out[idx] = np.clip(w, 0, None) / w.sum()
        return out

    candidates = [polish(solve(p0)) for p0 in starts]
    best = min(candidates, key=lambda p: p @ z @ p)
    return MaxDiversity(float(1.0 / (best @ z @ best)), best, "qp", _kkt(z, best))


class _NotConvex(Exception):
    """The working set's Gram matrix is singular or indefinite; Wolfe's method does not apply."""


def _wolfe(z: np.ndarray, *, warm: Sequence[np.ndarray] = (), tol: float = 1e-9, batch: int = 1,
           max_iter: int = 20000) -> MaxDiversity:
    """Minimise pᵀZp on the simplex by Wolfe's min-norm-point method on the Gram matrix Z (exact when Z is positive
    semidefinite on the visited supports). Keeps a corral S of points, adds the points that violate optimality
    ((Zp)_i < pᵀZp), re-solves Z_S y = 1 on S, and steps back to the simplex when a weight would go negative. Starts
    from the best of ``warm`` (e.g. a child's optimum). Raises ``_NotConvex`` when a corral system is not usable."""
    n = z.shape[0]
    p = np.zeros(n)
    best = None
    for w in warm:
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            w = w / w.sum()
            val = w @ z @ w
            if best is None or val < best[0]:
                best = (val, w)
    if best is not None:
        p = best[1]
    else:
        p[int(np.argmin(np.diag(z)))] = 1.0
    corral = list(np.flatnonzero(p > 1e-12))
    for _ in range(max_iter):
        zp = z @ p
        lam = float(p @ zp)
        viol = np.flatnonzero(zp < lam - tol)
        if len(viol) == 0:
            return MaxDiversity(1.0 / lam, p, "wolfe", _kkt(z, p))
        in_corral = set(corral)
        corral += [int(i) for i in viol[np.argsort(zp[viol])][:batch] if int(i) not in in_corral]
        while True:                                   # minor cycles
            idx = np.array(corral)
            try:
                y = np.linalg.solve(z[np.ix_(idx, idx)], np.ones(len(idx)))
            except np.linalg.LinAlgError as e:
                raise _NotConvex from e
            if not np.isfinite(y).all() or y.sum() <= 0:
                raise _NotConvex
            y = y / y.sum()
            cur = p[idx]
            if y.min() >= -1e-13:
                p = np.zeros(n)
                p[idx] = np.clip(y, 0, None)
                break
            neg = y < 0
            theta = float(np.min(cur[neg] / (cur[neg] - y[neg])))
            cur = cur + theta * (y - cur)
            p = np.zeros(n)
            p[idx] = np.clip(cur, 0, None)
            corral = [int(i) for i, c in zip(idx, cur) if c > 1e-12]
            p = p / p.sum()
    raise _NotConvex


def _full_weighting(z: np.ndarray) -> MaxDiversity | None:
    """If Z w = 1 has a nonnegative solution on all points, p = w/Σw satisfies the optimality conditions
    ((Zp)_i = 1/Σw for every i), and D_max = Σw. Returns None when some weight is negative."""
    w = _weighting(z)
    if w is None or w.sum() <= 0:
        return None
    p = np.clip(w, 0, None) / np.clip(w, 0, None).sum()
    return MaxDiversity(float(1.0 / (p @ z @ p)), p, "weighting", _kkt(z, p)) if _kkt(z, p) else None


def _qp_cg(z: np.ndarray, *, warm: Sequence[np.ndarray] = (), seed: int = 0, batch: int = 25,
           max_rounds: int = 40) -> MaxDiversity:
    """Same optimum as ``_qp`` for large collections. The optimal distribution has a small support, so solve on a
    working set, then add every point i with (Zp)_i < pᵀZp (a violated KKT condition, i.e. a point that would
    still lower pᵀZp) until none is left. The last round runs the full multi-start on the final working set."""
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    work = set(rng.choice(n, size=min(n, 40), replace=False).tolist())
    for w in warm:
        work |= set(np.flatnonzero(w > 1e-9).tolist())
    p = np.full(n, 1.0 / n)
    for rnd in range(max_rounds):
        idx = np.array(sorted(work))
        sub_warm = [w[idx] / w[idx].sum() for w in warm if w[idx].sum() > 0]
        sub = _qp(z[np.ix_(idx, idx)], restarts=4 if rnd < max_rounds - 1 else 24, seed=seed, warm=sub_warm)
        p = np.zeros(n)
        p[idx] = sub.weights
        zp = z @ p
        lam = float(p @ zp)
        viol = [i for i in np.argsort(zp) if zp[i] < lam - 1e-7 and i not in work][:batch]
        if not viol:
            break
        work = set(np.flatnonzero(p > 1e-9).tolist()) | set(viol)
    else:
        return MaxDiversity(float(1.0 / (p @ z @ p)), p, "qp-cg", False)
    # converged: re-solve the final working set with the full multi-start and re-check against all points
    idx = np.array(sorted(work | set(np.flatnonzero(p > 1e-9).tolist())))
    sub = _qp(z[np.ix_(idx, idx)], restarts=24 if len(idx) <= 120 else 6, seed=seed, warm=[p[idx] / p[idx].sum()])
    p = np.zeros(n)
    p[idx] = sub.weights
    return MaxDiversity(float(1.0 / (p @ z @ p)), p, "qp-cg", _kkt(z, p))


def max_diversity(z: np.ndarray, *, exact_max_n: int = EXACT_MAX_N, warm: Sequence[np.ndarray] = ()) -> MaxDiversity:
    """D_max of Z. ``warm``: distributions on the same points (length n, summing to 1) that the optimizer also
    starts from, so the answer is never below pᵀZp⁻¹ for any of them."""
    if z.shape[0] <= exact_max_n:
        return _exact(z)
    if z.shape[0] <= CG_MIN_N:
        return _qp(z, warm=warm)
    full = _full_weighting(z)         # fine scales: every point is in the optimum, no search needed
    if full is not None:
        return full
    try:
        res = _wolfe(z, warm=warm)
        if res.kkt_ok:
            return res
    except _NotConvex:
        pass                          # Z not positive semidefinite on the visited supports: use the general solver
    return _qp_cg(z, warm=warm)


def diversity_profile(d: np.ndarray, ks: Sequence[float]) -> dict[str, np.ndarray]:
    zs = [similarity_matrix(d, k) for k in ks]
    maxes = [max_diversity(z) for z in zs]
    return {
        "k": np.asarray(ks, dtype=float),
        "d_max": np.array([m.value for m in maxes]),
        "d_uniform": np.array([uniform_diversity(z) for z in zs]),
        "kkt_ok": np.array([m.kkt_ok for m in maxes]),
        "weights": [m.weights for m in maxes],
    }
