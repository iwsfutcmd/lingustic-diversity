import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_analysis import drop_nan, nested_samples  # noqa: E402

from isodiv.asjp import Doculect
from isodiv.diversity import max_diversity, similarity_matrix


def doc(i: str, g: str = "") -> Doculect:
    return Doculect(i, i, g or i, "F", "W", {})


def world():
    docs = {f"d{i}": doc(f"d{i}", f"g{i % 7}") for i in range(300)}
    ids = list(docs)
    pools = {
        "A": frozenset(ids[:200]),                  # parent
        "B": frozenset(ids[:120]),                  # child of A
        "C": frozenset(ids[:50] + ["d250"]),        # child of B, plus one stray row outside B and A
        "B2": frozenset(ids[:120]),                 # same pool as B
        "D": frozenset(ids[200:260]),               # unrelated
    }
    return docs, pools


def test_child_sample_is_subset_of_parent_sample():
    docs, pools = world()
    samples, children = nested_samples(pools, docs, cap=30)
    ids = {n: {x.id for x in s} for n, s in samples.items()}
    assert ids["C"] <= ids["B"] <= ids["A"]
    assert not ids["D"] & ids["A"]
    assert children["A"] == ["B"] and children["B"] == ["C"] and children["D"] == []


def test_identical_pools_share_a_sample_and_caps_hold():
    docs, pools = world()
    samples, _ = nested_samples(pools, docs, cap=30)
    assert [x.id for x in samples["B"]] == [x.id for x in samples["B2"]]
    assert len(samples["D"]) == 30 and len(samples["C"]) == 30
    assert len(samples["A"]) == 30 and len(samples["B"]) == 30     # children already fill the cap


def test_stray_row_is_absorbed_by_tolerance():
    docs, pools = world()
    samples, children = nested_samples(pools, docs, cap=60, tol=0.05)
    assert "C" in children["B"] or "C" in children["A"] or children["B"] == ["C"]
    assert {x.id for x in samples["C"]} <= {x.id for x in samples["B"]}


def test_dmax_monotone_along_nesting():
    rng = np.random.default_rng(0)
    pts = rng.uniform(size=(300, 3))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    docs, pools = world()
    samples, _ = nested_samples(pools, docs, cap=30)
    idx = lambda n: [int(x.id[1:]) for x in samples[n]]
    val = lambda n, k: max_diversity(similarity_matrix(d[np.ix_(idx(n), idx(n))], k)).value
    for k in (0.5, 3.0):
        assert val("C", k) <= val("B", k) + 1e-9 <= val("A", k) + 2e-9


def test_drop_nan_keeps_protected_doculects():
    docs = [doc(x) for x in "abcd"]
    d = np.zeros((4, 4))
    d[0, 1] = d[1, 0] = np.nan                     # a-b conflict: drop the unprotected one
    kept, _ = drop_nan(docs, d, protect=frozenset("a"))
    assert [x.id for x in kept] == ["a", "c", "d"]
    kept, _ = drop_nan(docs, d, protect=frozenset("ab"))
    assert len(kept) == 3 and {"c", "d"} <= {x.id for x in kept}
