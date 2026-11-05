"""Run the whole analysis in order, skipping steps whose inputs have not changed, committing after each step.

Each step lists its inputs (files or globs). A step is skipped when the hash of its inputs and its command matches the
one recorded after its last successful run (results/.pipeline_state.json) and its outputs exist. Code and config
are hashed by content; large data files by size and modification time. After a step that ran, results/ and docs/
are committed and pushed (with --commit), so an interrupted run loses at most one step.

Usage:
    python scripts/pipeline.py [--commit] [--force STEP ...] [--only STEP ...] [--message "..."]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "results" / ".pipeline_state.json"
CODE = ["src/isodiv/*.py", "scripts/run_analysis.py", "config/*.tsv"]
ASJP = ["data/asjp/cldf/*.csv"]
PMI = ["results/pmi_model_all.tsv"]

# name, command, inputs, outputs
STEPS = [
    ("regularity_eval", ["scripts/regularity_eval.py"], CODE + ASJP + PMI + ["scripts/regularity_eval.py"],
     ["results/regularity_eval.json"]),
    ("run_analysis", ["scripts/run_analysis.py"], CODE + ASJP + PMI + ["results/regularity_eval.json"],
     ["results/summary.json", "results/samples.json", "results/profiles.json"]),
    ("bootstrap", ["scripts/bootstrap.py"], CODE + ASJP + PMI + ["scripts/bootstrap.py", "results/summary.json"],
     ["results/bootstrap.json"]),
    ("cross_family", ["scripts/cross_family.py"], CODE + ASJP + PMI + ["scripts/cross_family.py", "results/summary.json"],
     ["results/cross_family.json"]),
    ("cross_family_cap10", ["scripts/cross_family.py", "--cap", "10", "--tag", "cap10"],
     CODE + ASJP + PMI + ["scripts/cross_family.py", "results/summary.json"], ["results/cross_family_cap10.json"]),
    ("cross_family_cap50", ["scripts/cross_family.py", "--cap", "50", "--tag", "cap50"],
     CODE + ASJP + PMI + ["scripts/cross_family.py", "results/summary.json"], ["results/cross_family_cap50.json"]),
    ("relatedness", ["scripts/relatedness.py", "--cap", "cap10=results/cross_family_cap10.json",
                     "--cap", "cap50=results/cross_family_cap50.json"],
     ["scripts/relatedness.py", "results/cross_family.json", "results/cross_family_cap10.json",
      "results/cross_family_cap50.json"], ["results/relatedness.json"]),
    ("widest", ["scripts/widest.py"], CODE + ASJP + PMI + ["scripts/widest.py", "results/summary.json", "results/samples.json"],
     ["results/widest.json", "results/contact.json"]),
    ("audit", ["scripts/audit.py"], CODE + ASJP + PMI + ["scripts/audit.py", "results/summary.json", "results/samples.json",
                                                         "results/contact.json"], ["results/audit.json"]),
    ("adversarial", ["scripts/adversarial.py"], CODE + ASJP + PMI + ["scripts/adversarial.py", "results/summary.json"],
     ["results/adversarial.json"]),
    ("single_lists", ["scripts/single_lists.py"], CODE + ASJP + PMI + ["scripts/single_lists.py"], ["results/single_lists.json"]),
    ("sensitivity", ["scripts/sensitivity.py"], CODE + ASJP + PMI + ["scripts/sensitivity.py", "results/summary.json",
                                                                     "results/samples.json", "results/bootstrap.json"],
     ["results/sensitivity.json"]),
    ("rarefaction", ["scripts/rarefaction.py"], CODE + ASJP + PMI + ["scripts/rarefaction.py", "results/summary.json",
                                                                     "results/samples.json"], ["results/rarefaction.json"]),
    ("cognate_check", ["scripts/cognate_check.py"], CODE + ASJP + PMI + ["scripts/cognate_check.py", "results/samples.json",
                                                                         "data/lexibank/*/cldf/*.csv"],
     ["results/cognate_check.json"]),
    ("build_explorer", ["scripts/build_explorer.py"], CODE + ["scripts/build_explorer.py", "explorer/*", "results/*.json"],
     ["results/explorer.html"]),
    ("build_site", ["scripts/build_site.py"], CODE + ASJP + ["scripts/build_site.py", "explainer/*", "results/*.json",
                                                             "results/*.csv", "results/explorer.html", "docs/related_work.md"],
     ["docs/index.html"]),
]
BIG = 2_000_000          # files larger than this are hashed by size and mtime


def fingerprint(patterns: list[str], command: list[str]) -> str:
    h = hashlib.sha256(" ".join(command).encode())
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / pat))):
            p = Path(f)
            if not p.is_file() or p.name.startswith(".pipeline_state"):
                continue
            st = p.stat()
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode() if st.st_size > BIG else p.read_bytes())
    return h.hexdigest()


def git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="commit and push results/ and docs/ after each step that ran")
    ap.add_argument("--force", nargs="*", default=[], help="run these steps even if unchanged ('all' for every step)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--message", default="", help="appended to each commit's subject")
    ap.add_argument("--trailer", default="", help="commit message trailer lines")
    args = ap.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "scripts"), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))

    for name, cmd, inputs, outputs in STEPS:
        if args.only is not None and name not in args.only:
            continue
        fp = fingerprint(inputs, cmd)
        done = all((ROOT / o).exists() for o in outputs)
        if state.get(name) == fp and done and name not in args.force and "all" not in args.force:
            print(f"=== {name}: unchanged, skipped", flush=True)
            continue
        print(f"=== {name} {time.strftime('%H:%M')}", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env)
        if r.returncode != 0:
            print(f"FAILED {name} (exit {r.returncode})", flush=True)
            sys.exit(r.returncode)
        state[name] = fingerprint(inputs, cmd)       # outputs of earlier steps may feed later fingerprints
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(f"done {name} in {(time.time() - t) / 60:.1f} min", flush=True)
        if args.commit:
            git("add", "-A", "results", "docs")
            msg = f"Pipeline step {name}" + (f" ({args.message})" if args.message else "")
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
                git("commit", "-q", "-m", msg + ("\n\n" + args.trailer if args.trailer else ""))
                git("push", "-q", "origin", "main")
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
