from pathlib import Path

import pytest

from isodiv.asjp import load

CLDF = Path(__file__).resolve().parent.parent / "data" / "asjp" / "cldf"
pytestmark = pytest.mark.skipif(not CLDF.exists(), reason="needs data/asjp")


def test_supplement_fills_gaps_and_never_overrides(tmp_path):
    sup = tmp_path / "s.tsv"
    sup.write_text("# comment\ndoculect\tconcept\tasjpcode\tsource_form\tsource\tnote\n"
                   "HITTITE\t77\tperu\tperu-\tx\t\nHITTITE\t75\tzzz\tbad\tx\tASJP already has water\n", encoding="utf-8")
    plain = load(CLDF, keep=lambda r: r["ID"] == "HITTITE", supplements=None)["HITTITE"]
    more = load(CLDF, keep=lambda r: r["ID"] == "HITTITE", supplements=sup)["HITTITE"]
    assert "77" not in plain.words and more.words["77"] == (tuple("peru"),)
    assert more.words["75"] == plain.words["75"]                      # an existing ASJP form is never replaced


def test_default_supplements_and_switch(monkeypatch):
    h = load(CLDF, keep=lambda r: r["ID"] == "HITTITE")["HITTITE"]
    assert len(h.words) == 36
    monkeypatch.setenv("ISODIV_NO_SUPPLEMENTS", "1")
    assert len(load(CLDF, keep=lambda r: r["ID"] == "HITTITE")["HITTITE"].words) == 30


def test_added_doculect_luwian(monkeypatch):
    from isodiv.asjp import read_languages
    rows = {r["ID"]: r for r in read_languages(CLDF)}
    assert rows["LUWIAN"]["Family"] == "Indo-European" and rows["LUWIAN"]["long_extinct"] == "True"
    lw = load(CLDF, keep=lambda r: r["ID"] == "LUWIAN")["LUWIAN"]
    assert len(lw.words) == 23 and lw.words["1"] == (tuple("ammu"),)
    monkeypatch.setenv("ISODIV_NO_SUPPLEMENTS", "1")                  # the added row goes with the supplements
    assert "LUWIAN" not in {r["ID"] for r in read_languages(CLDF)}
    assert load(CLDF, keep=lambda r: r["ID"] == "LUWIAN") == {}
