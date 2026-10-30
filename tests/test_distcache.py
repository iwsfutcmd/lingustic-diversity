import numpy as np

from isodiv.asjp import Doculect
from isodiv.distcache import DistCache
from isodiv.metric import Metric


def doc(i: str, words: dict) -> Doculect:
    return Doculect(i, i, i, "F", "W", {c: (tuple(w),) for c, w in words.items()})


def make():
    concepts = [f"c{k}" for k in range(25)]
    a = doc("a", {c: "kat" for c in concepts})
    b = doc("b", {c: "kit" for c in concepts})
    c = doc("c", {c: "mon" for c in concepts})
    sparse = doc("s", {"c0": "kat"})
    return a, b, c, sparse


def test_matches_metric_and_computes_each_pair_once(monkeypatch):
    a, b, c, s = make()
    m = Metric("ldnd")
    ref = m.matrix([a, b, c])
    calls = []
    orig = Metric.values

    def spy(self, pairs, **kw):
        calls.append(len(pairs))
        return orig(self, pairs, **kw)

    monkeypatch.setattr(Metric, "values", spy)
    cache = DistCache(m, ["a", "b", "c", "s"])
    d1 = cache.matrix([a, b, c])
    assert np.allclose(d1, ref, atol=1e-6)
    d2 = cache.matrix([a, b, c, s])             # only the three pairs with s are new
    assert calls == [3, 3]
    assert np.isnan(d2[0, 3]) and np.allclose(d2[:3, :3], d1)
    cache.matrix([c, a])
    assert calls == [3, 3]                       # nothing recomputed


def test_save_and_reload(tmp_path):
    a, b, c, s = make()
    m = Metric("ldnd")
    cache = DistCache(m, ["a", "b", "c", "s"])
    cache.matrix([a, b])
    cache.save(tmp_path / "x.npy")
    again = DistCache.load(tmp_path / "x.npy", m, ["c", "b", "a"])        # different order and subset
    assert again.d[again.pos["a"], again.pos["b"]] == cache.d[0, 1]
    assert again.d[again.pos["a"], again.pos["c"]] == -1.0
    other = DistCache.load(tmp_path / "x.npy", Metric("pmi", model=None), ["a", "b"])
    assert other.d[0, 1] == -1.0
