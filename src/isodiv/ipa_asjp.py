"""Convert IPA segments (CLDF ``Segments``, space-separated, CLTS/BIPA style) to ASJPcode symbols.

ASJPcode has 41 symbols and no diacritics in the PMI model, so the conversion keeps each segment's base articulation
and drops length, aspiration, palatalization, labialization, nasalization, tone, stress and syllabicity. A segment is
NFD-decomposed; combining marks and modifier letters are removed; what remains is looked up whole (affricates such as
``t͡s``/``ts``, ``tɕ``), then by its first letter. Classes follow the ASJP orthography (Brown et al. 2008):

    8 θ ð   4 dental nasal (not distinguished in IPA input; n is used)   c ts dz   C tʃ tɕ tʂ   j dʒ dʑ dʐ
    S ʃ ʂ ɕ   Z ʒ ʐ ʑ   T palatal stops c ɟ   5 ɲ   N ŋ ɴ   L laterals other than l (ɬ ɮ ʎ ʟ ɭ)
    x velar/palatal fricatives x ɣ ç ʝ   X uvular and pharyngeal fricatives χ ʁ ħ ʕ   7 ʔ   ! clicks
    i i ɪ y ʏ   e e ø   E ɛ æ œ ɶ   3 central vowels ə ɘ ɵ ɜ ɞ ɨ ʉ   a a ɑ ɒ ɐ ä   u u ʊ ɯ   o o ɔ ɤ ʌ

``convert`` returns the ASJP string and the segments it could not map (so callers can count and report them).
"""

from __future__ import annotations

import unicodedata

# whole segments (after stripping marks; tie bars removed) checked first
_MULTI = {
    "ts": "c", "dz": "c", "ʦ": "c", "ʣ": "c",
    "tʃ": "C", "ʧ": "C", "tɕ": "C", "ʨ": "C", "tʂ": "C", "cç": "C",
    "dʒ": "j", "ʤ": "j", "dʑ": "j", "ʥ": "j", "dʐ": "j", "ɟʝ": "j",
    "tθ": "8", "dð": "8", "tɬ": "L", "dɮ": "L", "kx": "k", "pf": "f", "qχ": "q",
}

_SINGLE = {
    # labials
    "p": "p", "b": "b", "ɸ": "f", "f": "f", "β": "v", "v": "v", "ʋ": "v", "m": "m", "ɱ": "m", "w": "w", "ʍ": "w",
    "ɥ": "w", "ɰ": "w",
    # dentals / alveolars
    "θ": "8", "ð": "8", "t": "t", "d": "d", "ʈ": "t", "ɖ": "d", "s": "s", "z": "z", "n": "n", "ɳ": "n",
    "r": "r", "ɾ": "r", "ɽ": "r", "ɹ": "r", "ɻ": "r", "ʀ": "r", "ʙ": "r", "ⱱ": "r", "l": "l", "ɫ": "l", "ɺ": "l",
    "ɭ": "L", "ʎ": "L", "ɬ": "L", "ɮ": "L", "ʟ": "L",
    # postalveolar / retroflex / alveolo-palatal fricatives
    "ʃ": "S", "ʂ": "S", "ɕ": "S", "ɧ": "S", "ʒ": "Z", "ʐ": "Z", "ʑ": "Z",
    # palatals
    "c": "T", "ɟ": "T", "ɲ": "5", "j": "y", "ç": "x", "ʝ": "x",
    # velars, uvulars, pharyngeals, glottals
    "k": "k", "g": "g", "ɡ": "g", "x": "x", "ɣ": "x", "ŋ": "N", "ɴ": "N", "q": "q", "ɢ": "G", "χ": "X",
    "ʁ": "X", "ħ": "X", "ʕ": "X", "h": "h", "ɦ": "h", "ʔ": "7", "ʡ": "7", "ʜ": "X", "ʢ": "X",
    # clicks and implosives / ejectives (marks already stripped)
    "ǀ": "!", "ǃ": "!", "ǁ": "!", "ǂ": "!", "ʘ": "!", "ɓ": "b", "ɗ": "d", "ʄ": "T", "ɠ": "g", "ʛ": "G",
    # vowels
    "i": "i", "ɪ": "i", "y": "i", "ʏ": "i", "e": "e", "ø": "e", "ɛ": "E", "æ": "E", "œ": "E", "ɶ": "E",
    "ə": "3", "ɘ": "3", "ɵ": "3", "ɜ": "3", "ɞ": "3", "ɨ": "3", "ʉ": "3", "a": "a", "ɑ": "a", "ɒ": "a",
    "ɐ": "a", "u": "u", "ʊ": "u", "ɯ": "u", "o": "o", "ɔ": "o", "ɤ": "o", "ʌ": "o",
}

_DROP_SEGMENTS = {"+", "_", "#", "-", "∅", "Ø", "|", "ˈ", "ˌ", "."}


def _base(seg: str) -> str:
    """The segment without combining marks, modifier letters (ʰ ʲ ʷ ː ˤ ˀ ...) and tie bars."""
    out = []
    for ch in unicodedata.normalize("NFD", seg):
        cat = unicodedata.category(ch)
        if cat.startswith("M") or cat == "Lm" or ch in "͜͡":
            continue
        if cat in ("Sk", "Po", "Pd", "Zs", "No", "Nd"):       # also tone numbers (², ⁵¹)
            continue
        out.append(ch)
    return "".join(out)


def segment(seg: str) -> str | None:
    """ASJP symbol for one IPA segment, '' for a segment to drop, None if unknown."""
    if seg in _DROP_SEGMENTS:
        return ""
    b = _base(seg)
    if not b:
        return ""
    if b in _MULTI:
        return _MULTI[b]
    if b[0] in _SINGLE:
        return _SINGLE[b[0]]
    low = b[0].lower()
    return _SINGLE.get(low)


def convert(segments: str) -> tuple[str, list[str]]:
    """'s i l m æ' -> ('silmE', []). Unknown segments are skipped and returned."""
    out, unknown = [], []
    for seg in segments.split():
        a = segment(seg)
        if a is None:
            unknown.append(seg)
        else:
            out.append(a)
    return "".join(out), unknown
