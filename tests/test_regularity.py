import numpy as np
import pytest

from isodiv.asjp import Doculect
from isodiv.diversity import anchored_similarity, max_diversity
from isodiv.ipa_asjp import convert, segment


def test_ipa_asjp_classes():
    assert convert("s i l m æ") == ("silmE", [])
    assert convert("t͡ʃ ɑ l ɛ b m i")[0] == "CalEbmi"            # tie-bar affricate
    assert convert("tʃː a")[0] == "Ca" and convert("pʰ aː")[0] == "pa"   # length, aspiration dropped
    assert segment("ɲ") == "5" and segment("ŋ") == "N" and segment("ʔ") == "7" and segment("ʎ") == "L"
    assert segment("ə") == "3" and segment("ɨ") == "3" and segment("ø") == "e" and segment("ɔ") == "o"
    assert segment("²") == "" and segment("+") == ""               # tone numbers, boundaries
    assert convert("a ☃ b") == ("ab", ["☃"])                       # unknowns are reported, not guessed


def test_anchored_similarity_counts_unrelated_as_distinct():
    d = np.full((5, 5), 0.97); np.fill_diagonal(d, 0)
    assert max_diversity(anchored_similarity(d, 0.228, 0.938)).value == pytest.approx(5.0)
    same = np.full((4, 4), 0.2); np.fill_diagonal(same, 0)        # one language transcribed 4 times
    assert max_diversity(anchored_similarity(same, 0.228, 0.938)).value == pytest.approx(1.0)


@pytest.fixture(scope="module")
def reg():
    from pathlib import Path
    model = Path(__file__).resolve().parent.parent / "results" / "pmi_model_all.tsv"
    if not model.exists():
        pytest.skip("needs results/pmi_model_all.tsv")
    from isodiv.metric import get_metric
    from isodiv.regularity import Regularity
    return Regularity(get_metric("pmi", model).model)


def _lists(seed=0, n=40):
    rng = np.random.default_rng(seed)
    cons, vow = list("ptkmnslrbdg"), list("aeiou")
    return ["".join(rng.choice(cons) + rng.choice(vow) for _ in range(rng.integers(1, 3))) for _ in range(n)]


def test_identical_lists_are_significant_and_paths_agree(reg):
    w = _lists()
    r = reg.test_lists(w, w, perms=199)
    assert r.p_sim <= 0.01 and r.p_g <= 0.01
    a = Doculect("A", "A", "", "", "", {str(i): (tuple(x),) for i, x in enumerate(w)})
    r2 = reg.test(a, a, perms=199)
    assert r2.sim == pytest.approx(r.sim) and r2.g == pytest.approx(r.g)


def test_unrelated_random_lists_are_not(reg):
    ps = [reg.test_lists(_lists(1 + k), _lists(100 + k), perms=99).p_g for k in range(6)]
    assert np.median(ps) > 0.1
