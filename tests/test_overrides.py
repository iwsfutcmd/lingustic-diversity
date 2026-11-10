import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_analysis import LINEAGES, load_overrides  # noqa: E402

from isodiv.asjp import read_languages


def test_shaowu_out_of_gan_but_in_sinitic():
    rows = {r["ID"]: r for r in read_languages(Path("data/asjp/cldf"))}
    r = rows["GAN_SHAOWU"]
    assert not LINEAGES["Gan"](r)
    assert LINEAGES["Sinitic"](r)
    assert LINEAGES["Gan"](rows["GAN_NANCHANG"])          # the rest of Gan is untouched


def test_override_file_parses(tmp_path):
    p = tmp_path / "o.tsv"
    p.write_text("# c\nX1\tremove\tGan\twhy\nX2\tadd\tPinghua\twhy\n", encoding="utf-8")
    remove, add = load_overrides(p)
    assert remove == {"Gan": {"X1"}} and add == {"Pinghua": {"X2"}}
