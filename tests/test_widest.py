import numpy as np

from isodiv.widest import neighbourhoods, robust_distances, widest


def toy():
    """Two branches of three close lists each, 0.8 apart; one noisy pair at 1.2 inside branch A."""
    d = np.full((6, 6), 0.8)
    d[:3, :3] = 0.2
    d[3:, 3:] = 0.2
    np.fill_diagonal(d, 0)
    d[0, 1] = d[1, 0] = 1.2
    return d


def test_neighbourhood_median_ignores_one_noisy_pair():
    d = toy()
    assert d.max() == 1.2                          # the raw widest pair is noise inside one branch
    nb = neighbourhoods(d, band=0.47)
    assert [len(v) for v in nb] == [2, 2, 3, 3, 3, 3]
    w, a, b = widest(robust_distances(d, nb))
    assert abs(w - 0.8) < 1e-12 and (a < 3) != (b < 3)     # the branch split


def test_lone_list_counts_and_can_be_excluded():
    d = np.pad(toy(), ((0, 1), (0, 1)), constant_values=0.95)
    d[6, 6] = 0
    nb = neighbourhoods(d, band=0.47)
    assert len(nb[6]) == 1                         # no relative inside the band: a lone list
    r = robust_distances(d, nb)
    assert widest(r)[0] == 0.95
    allowed = np.ones(7, bool)
    allowed[6] = False
    assert abs(widest(r, allowed)[0] - 0.8) < 1e-12


def test_degenerate_cases():
    assert np.isnan(widest(robust_distances(np.zeros((1, 1)), [np.array([0])]))[0])
    d = np.array([[0, 0.3], [0.3, 0]])
    assert widest(robust_distances(d, neighbourhoods(d, 0.47)))[0] == 0.3
