from isodiv.codes import load_iso1, load_macro, resolve

LANGS = [
    {"ID": "JAPANESE", "Name": "JAPANESE", "Glottocode": "nucl1643", "Glottolog_Name": "Japanese", "ISO639P3code": "jpn"},
    {"ID": "JAPANESE_2", "Name": "JAPANESE_2", "Glottocode": "nucl1643", "Glottolog_Name": "Japanese", "ISO639P3code": "jpn"},
    {"ID": "NAHA", "Name": "NAHA", "Glottocode": "cent2126", "Glottolog_Name": "Central Okinawan", "ISO639P3code": "ryu"},
    {"ID": "MANDARIN", "Name": "MANDARIN", "Glottocode": "mand1415", "Glottolog_Name": "Mandarin Chinese", "ISO639P3code": "cmn"},
    {"ID": "CANTONESE", "Name": "CANTONESE", "Glottocode": "yuec1235", "Glottolog_Name": "Yue Chinese", "ISO639P3code": "yue"},
    {"ID": "NOCODE", "Name": "NOCODE", "Glottocode": "", "Glottolog_Name": "", "ISO639P3code": ""},
]


def test_glottocode_iso3_iso1_agree():
    for q in ("nucl1643", "jpn", "ja", "JPN", " ja "):
        assert resolve(q, LANGS).ids == ("JAPANESE", "JAPANESE_2"), q
    assert resolve("ja", LANGS).kind == "iso639-1" and resolve("jpn", LANGS).kind == "iso639-3"
    assert resolve("nucl1643", LANGS).kind == "glottocode"


def test_macrolanguage_expands_to_members():
    m = resolve("zh", LANGS)
    assert m.kind == "macrolanguage" and set(m.ids) == {"MANDARIN", "CANTONESE"}
    assert resolve("zho", LANGS).ids == m.ids


def test_unknown_and_names():
    assert resolve("xx", LANGS).ids == ()                    # not an ISO 639-1 code
    assert resolve("qqq", LANGS).ids == ()
    assert resolve("zzzz9999", LANGS).ids == ()
    assert resolve("okinawan", LANGS).ids == ("NAHA",)       # Glottolog name, case-insensitive
    assert resolve("naha", LANGS).ids == ("NAHA",)
    assert resolve("", LANGS).ids == ()


def test_tables():
    iso1, macro = load_iso1(), load_macro()
    assert iso1["ja"] == "jpn" and iso1["en"] == "eng" and len(iso1) == 184
    assert {"cmn", "yue", "wuu"} <= set(macro["zho"])
