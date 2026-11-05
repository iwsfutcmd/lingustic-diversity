import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from relatedness import analyse, exceed, shortfall  # noqa: E402

from isodiv.diversity import max_diversity
from isodiv.kernel import Kernel


def test_shortfall_bounds():
    assert shortfall({"dmax_a": 2.0, "dmax_b": 3.0, "dmax_union": 5.0}) == 0.0
    assert shortfall({"dmax_a": 2.0, "dmax_b": 3.0, "dmax_union": 3.0}) == 1.0
    assert abs(shortfall({"dmax_a": 2.0, "dmax_b": 3.0, "dmax_union": 4.5}) - 0.25) < 1e-12
    assert np.isnan(shortfall({"dmax_a": float("nan"), "dmax_b": 3.0, "dmax_union": 3.0}))


def test_union_between_max_and_sum_with_undetectable_cross_pairs():
    """A cross pair at or beyond the detection limit (or missing, scored there) adds nothing: block-diagonal sum.
    One cross pair within reach pulls the union below the sum but never below the larger lineage."""
    k = Kernel(0.2, 0.9)
    a = np.array([[0, .5], [.5, 0]]); b = np.array([[0, .6, .7], [.6, 0, .4], [.7, .4, 0]])
    da, db = max_diversity(k(a)).value, max_diversity(k(b)).value
    far = np.full((2, 3), 0.9)
    u = max_diversity(k(np.block([[a, far], [far.T, b]]))).value
    assert abs(u - (da + db)) < 1e-6
    near = far.copy(); near[0, 0] = 0.3
    u = max_diversity(k(np.block([[a, near], [near.T, b]]))).value
    assert max(da, db) - 1e-6 <= u < da + db - 1e-3


def test_exceed_counts_the_pair_itself():
    assert exceed(0.5, np.array([0.1, 0.2, 0.3])) == 0.25
    assert exceed(0.0, np.array([0.1, 0.2, 0.3])) == 1.0


def _pair(a, b, s, rel=False, shared=0, km=100.0):
    return {"a": a, "b": b, "dmax_a": 2.0, "dmax_b": 2.0, "dmax_union": 4.0 - 2.0 * s,
            "related_by_construction": rel, "shared_doculects": shared, "min_km": km}


def test_groups_and_calibration():
    rng = np.random.default_rng(0)
    pairs = [_pair(f"U{i}", f"V{i}", float(x)) for i, x in enumerate(rng.uniform(0, 0.1, 300))]
    pairs += [_pair("R1", "R2", 0.6, rel=True), _pair("N1", "N2", 0.9, rel=True, shared=3),
              _pair("Turkic", "Mongolic", 0.5)]
    st = analyse(pairs, near_km=1500)
    g = {r["a"]: r for r in pairs}
    assert g["N1"]["group"] == "nested" and g["R1"]["group"] == "related" and g["Turkic"]["group"] == "proposal"
    assert st["n"]["unrelated"] == 300 and st["auc_related_vs_unrelated"] == 1.0
    assert g["Turkic"]["p_size_near"] < 0.02 and g["U0"]["p_size"] > 0.0
