import numpy as np
import pytest

from isodiv.asjp import Doculect
from isodiv.distance import ldnd
from isodiv.diversity import _exact, _qp, max_diversity, similarity_matrix, uniform_diversity


def random_metric(n: int, dim: int = 3, seed: int = 0) -> np.ndarray:
    pts = np.random.default_rng(seed).uniform(size=(n, dim))
    return np.linalg.norm(pts[:, None] - pts[None], axis=-1)


def with_duplicate(d: np.ndarray, i: int) -> np.ndarray:
    idx = np.r_[np.arange(d.shape[0]), i]
    out = d[np.ix_(idx, idx)].copy()
    return out


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [0.5, 3.0, 20.0])
def test_duplicate_dialect_leaves_dmax_unchanged(seed, k):
    d = random_metric(7, seed=seed)
    base = max_diversity(similarity_matrix(d, k)).value
    dup = max_diversity(similarity_matrix(with_duplicate(d, 2), k)).value
    assert dup == pytest.approx(base, rel=1e-8)


def test_uniform_diversity_is_dialect_sensitive():
    # many near-copies of one variety drag naive diversity down; D_max ignores them
    d = random_metric(4, seed=1)
    for _ in range(6):
        d = with_duplicate(d, 0)
    z = similarity_matrix(d, 10.0)
    z4 = similarity_matrix(d[:4, :4], 10.0)
    assert max_diversity(z).value == pytest.approx(max_diversity(z4).value, rel=1e-8)
    assert uniform_diversity(z) < 0.8 * uniform_diversity(z4)


@pytest.mark.parametrize("seed", range(5))
def test_adding_a_doculect_never_decreases_dmax(seed):
    d = random_metric(9, seed=seed)
    z = similarity_matrix(d, 4.0)
    assert max_diversity(z).value >= max_diversity(z[:8, :8]).value - 1e-9


def test_limits():
    d = random_metric(6, seed=3)
    assert max_diversity(similarity_matrix(d, 1e-6)).value == pytest.approx(1.0, abs=1e-4)
    assert max_diversity(similarity_matrix(d, 1e4)).value == pytest.approx(6.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", [1.0, 5.0, 15.0])
def test_qp_matches_exact(seed, k):
    z = similarity_matrix(random_metric(10, seed=seed), k)
    ex, qp = _exact(z), _qp(z)
    assert qp.value == pytest.approx(ex.value, rel=1e-5)
    assert ex.kkt_ok and qp.kkt_ok


def test_ldnd_identity_and_unicode_tokens():
    # IPA segments incl. combining diacritics and non-Latin tokens are opaque tokens
    words = {str(i): ((f"t͡ɕ{i}", "a", "ʔ"),) for i in range(25)}
    words["1"] = (("ʕ", "a", "j", "n"), ("ع", "ي", "ن"))
    a = Doculect("A", "A", "", "F", "", words)
    b = Doculect("B", "B", "", "F", "", dict(words))
    assert ldnd(a, b) == pytest.approx(0.0)
    c = Doculect("C", "C", "", "F", "", {k: (("x", "y", "z", "q"),) for k in words})
    assert ldnd(a, c) > 0.9
