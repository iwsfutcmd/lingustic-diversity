"""The browser distance (explorer/distance.js) must agree with the Python pipeline on real ASJP lists."""

import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from isodiv.asjp import load
from isodiv.codes import load_iso1, load_macro, resolve
from isodiv.metric import get_metric

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data" / "wordlists.json"
CLDF = ROOT / "data" / "asjp" / "cldf"
MODEL = ROOT / "results" / "pmi_model_all.tsv"

pytestmark = pytest.mark.skipif(
    not (shutil.which("node") and DATA.exists() and CLDF.exists() and MODEL.exists()),
    reason="needs node, docs/data/wordlists.json, data/asjp and the PMI model")

NODE = r"""
const D = require(process.argv[2]); const data = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const job = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const model = D.makeModel({ symbols: data.model.symbols, scores: data.model.scores, go: data.model.go, ge: data.model.ge });
const by = new Map(data.docs.map(r => [r[0], r]));
const out = { pairs: job.pairs.map(([a, b, kind]) => {
  const A = D.parseList(by.get(a)[6], data.concepts, model), B = D.parseList(by.get(b)[6], data.concepts, model);
  return D.pairDistance(kind, model, A, B).value;
}), resolve: job.queries.map(q => D.resolve(q, data.docs.map(r => ({ id: r[0], name: r[1], glottocode: r[2], iso3: r[3], gname: r[4] })),
                                            data.iso1, data.macro).ids) };
console.log(JSON.stringify(out));
"""


def run_js(job):
    (ROOT / "data" / "tmp").mkdir(parents=True, exist_ok=True)
    script = ROOT / "data" / "tmp" / "run_distance.js"
    script.write_text(NODE, encoding="utf-8")
    res = subprocess.run(["node", str(script), str(ROOT / "explorer" / "distance.js"), str(DATA)], input=json.dumps(job),
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def test_distances_match_python_and_codes_resolve_alike():
    docs = load(CLDF)
    ids = sorted(docs)
    rng = random.Random(3)
    fixed = [("JAPANESE", "ENGLISH"), ("JAPANESE", "NAHA"), ("ENGLISH", "STANDARD_GERMAN"), ("BASQUE", "NAHA")]
    close = [(a, b) for a, b in zip(ids[::900], ids[450::900])]
    pairs = fixed + close + [tuple(rng.sample(ids, 2)) for _ in range(20)]
    langs = list(csv.DictReader((CLDF / "languages.csv").open(encoding="utf-8")))
    queries = ["jpn", "ja", "nucl1643", "zh", "en", "okinawan", "zzz", "xx", "NAHA", "arb"]
    job = {"pairs": [[a, b, kind] for a, b in pairs for kind in ("pmi", "ldnd")], "queries": queries}
    out = run_js(job)
    for kind in ("pmi", "ldnd"):
        metric = get_metric(kind, MODEL)
        want = metric.values([(docs[a], docs[b]) for a, b in pairs])
        got = out["pairs"][0 if kind == "pmi" else 1::2]
        assert len(got) == len(want)
        for w, g in zip(want, got):
            if w != w:
                assert g is None                      # NaN -> null in JSON
            else:
                assert g == pytest.approx(w, abs=2e-6), (kind, w, g)       # model is stored to 4 decimals
    iso1, macro = load_iso1(), load_macro()
    for q, ids_js in zip(queries, out["resolve"]):
        assert ids_js == list(resolve(q, langs, iso1, macro).ids), q
