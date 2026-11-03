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


def test_a_changed_list_is_recomputed(tmp_path):
    a, b, c, s = make()
    m = Metric("ldnd")
    cache = DistCache(m, ["a", "b", "c", "s"])
    before = cache.matrix([a, b, c])
    cache.save(tmp_path / "x.npy")
    again = DistCache.load(tmp_path / "x.npy", m, ["a", "b", "c", "s"])
    assert np.allclose(again.matrix([a, b, c]), before)                 # unchanged lists come back as they were
    b2 = doc("b", {f"c{k}": "mon" for k in range(25)})                   # b's words changed
    d = again.matrix([a, b2, c])
    assert np.allclose(d, m.matrix([a, b2, c]), atol=1e-6, equal_nan=True) and not np.allclose(d, before, equal_nan=True)


def test_legacy_cache_without_fingerprints_is_not_trusted(tmp_path):
    import json
    a, b, c, s = make()
    m = Metric("ldnd")
    cache = DistCache(m, ["a", "b"])
    cache.matrix([a, b])
    cache.save(tmp_path / "x.npy")
    meta = tmp_path / "x.json"
    info = json.loads(meta.read_text())
    del info["fingerprints"]
    meta.write_text(json.dumps(info))
    assert DistCache.load(tmp_path / "x.npy", m, ["a", "b"]).d[0, 1] == -1.0


def test_values_fills_only_missing_pairs_and_agrees_with_matrix(monkeypatch):
    a, b, c, s = make()
    m = Metric("ldnd")
    cache = DistCache(m, ["a", "b", "c", "s"])
    calls = []
    orig = Metric.values

    def spy(self, pairs, **kw):
        calls.append(len(pairs))
        return orig(self, pairs, **kw)
    monkeypatch.setattr(Metric, "values", spy)
    v = cache.values([(a, b), (b, a), (a, c), (a, s)])
    assert calls == [3]                                  # (a, b) and (b, a) are one pair
    assert v[0] == v[1] and np.isnan(v[3])               # too little overlap: undefined
    assert np.allclose(cache.matrix([a, b, c])[[0, 0], [1, 2]], v[[0, 2]])
    assert calls == [3, 1]                               # matrix only adds (b, c)
    assert cache.values([(c, b)])[0] == cache.matrix([b, c])[0, 1] and calls == [3, 1]
