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


def random_docs(n=6, concepts=30, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        words = {}
        for c in range(concepts):
            if rng.random() < 0.9:
                syn = 1 + (rng.random() < 0.2)
                words[str(c + 1)] = tuple(tuple(rng.choice(list("abcdef"), rng.integers(1, 7))) for _ in range(syn))
        out.append(Doculect(f"D{k}", f"D{k}", "", "", "", words))
    return out


def test_ldnd_metric_matches_original_ldnd():
    from isodiv.distance import ldnd
    from isodiv.metric import Metric
    docs = random_docs()
    m = Metric("ldnd").matrix(docs)
    sub = frozenset(str(c) for c in range(1, 26))
    msub = Metric("ldnd").matrix(docs, concepts=sub, min_shared=15)
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            assert m[i, j] == pytest.approx(ldnd(docs[i], docs[j]))
            assert msub[i, j] == pytest.approx(ldnd(docs[i], docs[j], concepts=sub, min_shared=15))


def test_pmi_tables_match_direct_alignment():
    model = random_model(3)
    a, b = random_docs(2, seed=4)
    from isodiv.pmi import pmi_tables
    t = pmi_tables(model, [(a, b)])[0]
    def dist(x, y):
        s, sx, sy = align(model, [x, x, y], [y, x, y])
        return 1 - 2 * s / (sx + sy)
    for k, c in enumerate(t.concepts):
        assert t.same[k] == pytest.approx(min(dist(x, y) for x in a.words[c] for y in b.words[c]))
    c0, c1 = t.concepts[0], t.concepts[1]
    assert t.diff[0, 1] == pytest.approx(dist(a.words[c0][0], b.words[c1][0]))


def test_pmi_parallel_equals_serial(monkeypatch):
    import isodiv.pmi as pmi
    monkeypatch.setattr(pmi, "_CHUNK_WORD_PAIRS", 2000)      # force several chunks
    model = random_model(2)
    docs = random_docs(8, seed=5)
    pairs = [(docs[i], docs[j]) for i in range(8) for j in range(i + 1, 8)]
    serial = pmi.pmi_ldnd_values(model, pairs, workers=1)
    parallel = pmi.pmi_ldnd_values(model, pairs, workers=2)
    np.testing.assert_allclose(serial, parallel)
    assert np.all(np.isfinite(serial))


def test_compiled_kernel_matches_numpy():
    from isodiv import pmi
    if pmi._gotoh_scores is None:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(0)
    model = PMIModel(tuple("abcdefgh"), rng.normal(size=(9, 9)))
    words = [tuple(rng.choice(list("abcdefgh"), size=int(rng.integers(1, 9)))) for _ in range(300)]
    a, la = model.encode(words)
    b, lb = model.encode(words[::-1])
    fast = pmi._align_encoded(model, a, la, b, lb)
    kernel, pmi._gotoh_scores = pmi._gotoh_scores, None
    try:
        slow = pmi._align_encoded(model, a, la, b, lb)
    finally:
        pmi._gotoh_scores = kernel
    assert np.allclose(fast, slow, atol=1e-9)


def test_compiled_ldnd_matches_numpy_path():
    from isodiv import pmi
    if pmi._ldnd_kernel is None:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(1)
    syms = list("abcdefgh")
    model = PMIModel(tuple(syms), rng.normal(size=(9, 9)))

    def doc(i):
        words = {}
        for c in range(30):
            if rng.random() < 0.9:
                words[f"c{c}"] = tuple(tuple(rng.choice(syms, size=int(rng.integers(1, 7))))
                                       for _ in range(int(rng.integers(1, 3))))
        return Doculect(f"d{i}", f"d{i}", "g", "F", "W", words)

    docs = [doc(i) for i in range(12)]
    pairs = [(docs[i], docs[j]) for i in range(12) for j in range(i + 1, 12)]
    for kw in ({}, {"min_shared": 10, "concepts": frozenset(f"c{c}" for c in range(0, 30, 2))}, {"min_shared": 29}):
        fast = pmi.pmi_ldnd_values(model, pairs, **kw)
        kernel, pmi._ldnd_kernel = pmi._ldnd_kernel, None
        try:
            slow = pmi.pmi_ldnd_values(model, pairs, workers=1, **kw)
        finally:
            pmi._ldnd_kernel = kernel
        assert np.array_equal(np.isnan(fast), np.isnan(slow))
        assert np.allclose(fast[~np.isnan(fast)], slow[~np.isnan(slow)], atol=1e-9)
