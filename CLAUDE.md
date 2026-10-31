# Working in this repo

- Git: commit and push straight to `main`. No pull requests or feature branches unless asked.
- Never leave work only in the container: commit and push every change as soon as it is made, including
  generated results and any page or notes worth keeping (e.g. `explainer/`, `docs/`).
- Published site: GitHub Pages serves `docs/` from `main` (landing page, explorer, explainer, data, Compare varieties
  tab). After changing results or pages, rebuild (`python scripts/build_explorer.py`, then `python scripts/build_site.py`)
  and commit `docs/`. The explorer is built from `explorer/template.html`, `distance.js`, `compare.js` and
  `compare.html`; the explainer is `explainer/word-lists-to-dmax.html`.
- The Claude artifacts (explorer https://claude.ai/artifact/K7Crudh3aj3bGwKTYNotHh, explainer
  https://claude.ai/artifact/SnfTWDbcWUo58pDzsbyzZd) are no longer updated; the site replaces them. Republish them only
  if asked.
