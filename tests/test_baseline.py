import numpy as np

from isodiv.baseline import ceiling, relative, unrelated_baseline


def test_ceiling_limits():
    assert ceiling(1, 0.6) == 1.0
    assert abs(ceiling(10_000, 0.64) - 1 / 0.64) < 1e-2
    assert abs(ceiling(5, 0.0) - 5.0) < 1e-12


def test_baseline_interpolates_and_extends():
    sizes, meas = [5, 10, 80], [1.4, 1.48, 1.56]
    assert unrelated_baseline(1, sizes, meas, 0.64) == 1.0
    assert abs(unrelated_baseline(10, sizes, meas, 0.64) - 1.48) < 1e-12
    mid = unrelated_baseline(7, sizes, meas, 0.64)
    assert 1.4 < mid < 1.48
    assert abs(unrelated_baseline(80, sizes, meas, 0.64) - 1.56) < 1e-12
    big = unrelated_baseline(400, sizes, meas, 0.64)
    assert 1.56 < big < 1 / 0.64 + 0.05                       # continues up toward the 1/z̄ ceiling


def test_baseline_arrays_and_relative():
    sizes, meas = [5, 80], [np.array([1.0, 3.0]), np.array([1.0, 9.0])]
    b = unrelated_baseline(80, sizes, meas, np.array([0.99, 0.1]))
    assert np.allclose(b, [1.0, 9.0])
    assert np.allclose(relative(np.array([5.0]), 9.0), 0.5)
    assert float(relative(1.0, 1.5)) == 0.0


def test_baseline_below_smallest_measured_set_follows_ceiling():
    # calibrated scale: 5 unrelated lects ≈ 4.92, z̄ ≈ 0.007; two unrelated lects must score about 2, not ≈ 2.7
    b2 = unrelated_baseline(2, [5, 10], [4.92, 9.74], 0.007)
    assert abs(b2 - 2.0) < 0.05
    assert unrelated_baseline(1, [5, 10], [4.92, 9.74], 0.007) == 1.0
    assert abs(unrelated_baseline(5, [5, 10], [4.92, 9.74], 0.007) - 4.92) < 1e-12
    # saturated scale: two lects at similarity 0.64 score 2/1.64
    b2k = unrelated_baseline(2, [5, 10], [5 / (1 + 4 * 0.64), 1.48], 0.64)
    assert abs(b2k - 2 / 1.64) < 1e-9
