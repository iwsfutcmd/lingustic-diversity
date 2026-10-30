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


def test_exclusions_file_and_filter(tmp_path):
    from isodiv.asjp import load_exclusions

    f = tmp_path / "ex.tsv"
    f.write_text("id\treason\n# comment\nA\tbad list\nB\t\n", encoding="utf-8")
    assert load_exclusions(f) == {"A": "bad list", "B": ""}
    assert load_exclusions(tmp_path / "missing.tsv") == {}


def test_column_generation_matches_full_qp():
    from isodiv.diversity import _qp, _qp_cg
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(150, 6))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d /= d.max()
    z = similarity_matrix(d, 4.0)
    full, cg = _qp(z), _qp_cg(z)
    assert cg.kkt_ok
    assert abs(full.value - cg.value) < 1e-4


def test_warm_start_makes_superset_at_least_subset():
    from isodiv.diversity import max_diversity as md
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(140, 5))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d /= d.max()
    z = similarity_matrix(d, 2.0)
    sub = np.arange(40)
    child = md(z[np.ix_(sub, sub)])
    warm = np.zeros(140)
    warm[sub] = child.weights
    parent = md(z, warm=[warm])
    assert parent.value >= child.value - 1e-9


def test_wolfe_matches_exact_and_qp():
    from isodiv.diversity import _wolfe
    for seed, k in [(0, 1.0), (1, 3.0), (2, 8.0)]:
        z = similarity_matrix(random_metric(12, seed=seed), k)
        w = _wolfe(z)
        assert w.kkt_ok and abs(w.value - _exact(z).value) < 1e-8
    z = similarity_matrix(random_metric(160, dim=4, seed=9), 2.5)
    assert abs(_wolfe(z).value - _qp(z).value) < 1e-5


def test_wolfe_warm_start_never_below_child():
    from isodiv.diversity import _wolfe
    z = similarity_matrix(random_metric(150, dim=4, seed=3), 2.0)
    sub = np.arange(50)
    child = max_diversity(z[np.ix_(sub, sub)])
    warm = np.zeros(150)
    warm[sub] = child.weights
    assert _wolfe(z, warm=[warm]).value >= child.value - 1e-9


def test_indefinite_matrix_falls_back_to_general_solver():
    z = similarity_matrix(random_metric(130, dim=2, seed=4), 0.3)
    z = z - 0.4 * np.eye(130) + 0.4 * np.ones((130, 130)) * 0.0    # not positive semidefinite
    np.fill_diagonal(z, 1.0)
    res = max_diversity(z)
    assert np.isfinite(res.value) and res.weights.min() >= 0 and abs(res.weights.sum() - 1) < 1e-9
