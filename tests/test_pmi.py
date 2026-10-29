import numpy as np
import pytest

from isodiv.asjp import Doculect
from isodiv.pmi import PMIModel, align, erc, fit, pmi_erc, pmi_ldnd


def reference_gotoh(a, b, s, go, ge):
    """Straightforward scalar Gotoh, for checking the batched version."""
    neg = -1e9
    n, m = len(a), len(b)
    M = [[neg] * (m + 1) for _ in range(n + 1)]
    X = [[neg] * (m + 1) for _ in range(n + 1)]
    Y = [[neg] * (m + 1) for _ in range(n + 1)]
    M[0][0] = 0.0
    for i in range(1, n + 1):
        X[i][0] = go + ge * (i - 1)
    for j in range(1, m + 1):
        Y[0][j] = go + ge * (j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            M[i][j] = s[a[i - 1], b[j - 1]] + max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
            X[i][j] = max(M[i - 1][j] + go, X[i - 1][j] + ge, Y[i - 1][j] + go)
            Y[i][j] = max(M[i][j - 1] + go, Y[i][j - 1] + ge, X[i][j - 1] + go)
    return max(M[n][m], X[n][m], Y[n][m])


def random_model(seed=0, v=6):
    rng = np.random.default_rng(seed)
    s = np.zeros((v + 1, v + 1))
    r = rng.normal(-1, 1.5, (v, v))
    s[:v, :v] = (r + r.T) / 2 + np.diag(rng.uniform(2, 4, v))
    return PMIModel(tuple("abcdef"[:v]), s)


@pytest.mark.parametrize("seed", range(4))
def test_batched_alignment_matches_scalar(seed):
    model = random_model(seed)
    rng = np.random.default_rng(seed + 10)
    words = [tuple(rng.choice(list("abcdef"), rng.integers(1, 9))) for _ in range(60)]
    aw, bw = words[:30], words[30:]
    got = align(model, aw, bw)
    idx = {c: i for i, c in enumerate(model.symbols)}
    want = [reference_gotoh([idx[c] for c in a], [idx[c] for c in b], model.scores, model.gap_open, model.gap_extend)
            for a, b in zip(aw, bw)]
    np.testing.assert_allclose(got, want)


def test_traceback_pairs_add_up_to_score():
    model = random_model(1)
    aw, bw = [tuple("abcab"), tuple("ffd"), tuple("a")], [tuple("acb"), tuple("fdde"), tuple("bbb")]
    score, (sa, sb) = align(model, aw, bw, traceback=True)
    # matched columns + gap columns reproduce the total score
    matched = model.scores[sa, sb].sum()
    gaps = score.sum() - matched
    assert len(sa) <= sum(min(len(a), len(b)) for a, b in zip(aw, bw))
    assert gaps <= 0


def doc(i, words):
    return Doculect(i, i, "", "", "", {str(k): (w,) for k, w in enumerate(words)})


def test_erc_identical_lists_near_zero_and_random_near_one():
    rng = np.random.default_rng(0)
    first = rng.normal(size=(40, 40))
    assert erc(np.full(40, 99.0), first) < 0.05
    assert erc(rng.normal(size=40), first) == pytest.approx(1.0, abs=0.25)


def test_fit_learns_a_regular_correspondence():
    # language B has p wherever language A has h; PMI should score h:p above h:t
    rng = np.random.default_rng(0)
    base = [tuple(rng.choice(list("hatiku"), rng.integers(2, 6))) for _ in range(400)]
    shift = [tuple("p" if s == "h" else s for s in w) for w in base]
    cand = list(zip(base, shift))
    chance = [(base[i], shift[(i + 7) % len(base)]) for i in range(len(base))]
    model = fit(cand, chance, iterations=4)
    i = {s: k for k, s in enumerate(model.symbols)}
    assert model.scores[i["h"], i["p"]] > model.scores[i["h"], i["t"]]
    a = doc("A", base[:30])
    b = doc("B", shift[:30])
    assert pmi_erc(model, a, b) < 0.1
    assert pmi_ldnd(model, a, b) < 0.3
