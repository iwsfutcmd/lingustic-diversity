# isodiv — clock-free, dialect-count-robust diversity of language lineages

**Question:** can we say how internally diverse a lineage is (e.g. "is this an isolate?") without
(a) counting languages, and therefore without settling language vs. dialect, and
(b) assuming a rate of change, i.e. without glottochronology?

**Approach**

1. **Distances measured in amount of change, not time:** LDND on the 40-item ASJP lists
   (Wichmann et al. 2010). This is Levenshtein distance normalized by word length, then divided by
   the same measure computed between words with *different* meanings, which corrects for chance
   resemblance. No time depth, no rate assumption.
2. **Diversity that ignores duplicated varieties:** set Z = exp(−k·LDND) and compute the
   **Leinster–Meckes maximum diversity** D_max(Z). This is the effective number of fully distinct
   lineages. It is a property of Z alone (no abundance weights, no choice of order q), and adding
   an identical doculect cannot change it. Computed exactly by subset enumeration when n ≤ 14,
   otherwise as min pᵀZp on the probability simplex (multi-start SLSQP + polishing + KKT check).
3. **Choosing k without a clock:** k_cal is set so that two doculects of the *same* language
   (same Glottocode, entries named X / X_2) are 0.9-similar. The median LDND between such pairs is
   0.30, which gives k_cal ≈ 0.35. The full curve of D_max against k is plotted as well.

## Results (ASJP v21, `asjp` tokenization)

D_max at k_cal. The interval is the 5–95% range over 30 draws, each keeping 30 of the 40 concepts.

| band | lineage | D_max | interval |
|---|---|---|---|
| isolate-like | Basque | 1.038 | 1.029–1.042 |
| | Haida | 1.070 | 1.061–1.073 |
| | Koreanic | 1.109 | 1.096–1.118 |
| | Burushaski | 1.111 | 1.100–1.122 |
| | Nivkh | 1.132 | 1.124–1.141 |
| | Ainu | 1.137 | 1.131–1.148 |
| genus-like | Slavic | 1.220 | 1.210–1.230 |
| | Japonic | 1.224 | 1.215–1.233 |
| | Kartvelian | 1.231 | 1.226–1.236 |
| | Yeniseian | 1.237 | 1.230–1.245 |
| | Germanic | 1.261 | 1.254–1.270 |
| | Romance | 1.264 | 1.255–1.272 |
| | Mongolic | 1.269 | 1.265–1.276 |
| deep | Turkic | 1.300 | 1.295–1.306 |
| | Uralic | 1.351 | 1.348–1.357 |
| | Indo-European | 1.357 | 1.353–1.363 |

**Dialect-counting robustness (Japonic, D_max at k_cal):**

| sample | n | D_max |
|---|---|---|
| all doculects | 33 | 1.223 |
| one per Glottocode | 10 | 1.195 |
| all + Tokyo duplicated 3× | 36 | 1.223 (unchanged) |
| Tokyo + Shuri only | 2 | 1.111 |

The naive equal-weight diversity drops when Tokyo is duplicated; D_max does not move. The last row
falls because two doculects miss real Ryukyuan breadth (e.g. Yonaguni). That is under-sampling of
genuinely different varieties, not a counting artifact.

**Reading:**
- The bands do not overlap. Japonic measures as a genus-sized family, level with Slavic and
  Kartvelian, so including Ryukyuan is not a counting artifact.
- Koreanic (with Jeju) stays in the isolate band.
- Yeniseian scores as genus-sized, so Ket is a "last survivor", not an isolate.

## Caveats

- **Noise floor.** Same-language ASJP doculect pairs have LDND 0.11 / 0.30 / 0.57 at the 10th /
  50th / 90th percentile. At this resolution the isolate band overlaps "one language transcribed
  several times", so lineages *within* that band cannot be ranked against each other.
- **Sampling.** D_max is invariant to *duplicates*, not to documenting *more distinct* varieties.
  That is intended, since more distinct varieties is real diversity, but undersampled lineages
  (Basque n=2, Haida n=2) are lower bounds. `spread` and `spread_rarefied_m3` in
  `results/summary.csv` show this sample-size dependence.
- **Contact** shortens distances (e.g. Ryukyuan under Japanese influence). The 40-item list is
  borrowing-resistant, but not borrowing-proof.
- The intervals reflect concept resampling only, not doculect choice or transcription noise.

## Lineages

48 top-level lineages (the original 16 plus 32 more ASJP families) and 24 subfamilies selected by node in ASJP's
Glottolog classification path (`SUBFAMILIES` in `scripts/run_analysis.py`: Semitic, Arabic, Sinitic, Min, Indo-Aryan,
Iranian, Oceanic, Polynesian, ...). Two lineages count as related, and are left out of the unrelated-pair
statistics, whenever their doculects share a top-level Family.

## Explorer and cross-family check

- `scripts/cross_family.py` compares lineages with each other: mean LDND within and between 48 lineages
  (see Lineages), and D_max of two lineages together vs. apart.
  Japonic ↔ Kartvelian: mean LDND 0.98 vs. 0.55 / 0.69 within. Across 2,483 unrelated pairs the mean is 0.984 (lowest 0.911).
- `scripts/build_explorer.py` bundles `results/*.json` into `results/explorer.html`, a self-contained
  page (diversity ranking, scale profiles, between-lineage heatmap).

## Excluding bad data

`config/exclusions.tsv` (`id<TAB>reason`) lists doculects left out of every script. `scripts/audit.py` ranks
doculects by how much removing them lowers their lineage's D_max (also flags thin coverage) and writes
`results/audit.csv`; it never excludes anything itself. Currently excluded: `AYNU` (Turkic list with heavy
Persian replacement; Turkic D_max 1.304 → 1.281). Review the audit, edit the file, rerun.

`scripts/loans.py` adds a contact check: for each doculect, how much closer is it to a *neighbouring unrelated
language* (within 2000 km) than its own relatives are (`gap`, in LDND units)? Real contact cases stand out
(Aynu 0.23, Moghol 0.24, both with Iranian neighbours); everything else in the lineages tried is below 0.13.
Word-level masking of suspected loans is experimental.

### WOLD loan labels

WOLD (Haspelmath & Tadmor 2009) gives expert loan labels for 41 languages. `git clone --depth 1
https://github.com/lexibank/wold data/wold`, then:

- `scripts/run_analysis.py --wold data/wold/cldf` drops the words WOLD marks as borrowed (80 words over 40
  languages that also have an ASJP list; mapping in `src/isodiv/wold.py`).
- `scripts/wold_effect.py` shows how much that changes D_max per lineage.
- `scripts/wold_validate.py` checks `loans.py` against WOLD: no language reaches a gap above 0.17, so
  gaps above ~0.2 (Aynu 0.23, Moghol 0.24) are unusual. WOLD marks 0–10% of these basic concepts as loans
  for almost every language, so it cannot calibrate the detector on heavy-loan cases.

## PMI distances (learned sound correspondences)

`src/isodiv/pmi.py` replaces Levenshtein's flat substitution cost with segment scores learned as pointwise
mutual information from probable cognates (after Jäger 2013, 2018): affine-gap Needleman–Wunsch, batched in numpy.
Training uses no classification. It starts from geographic-neighbour doculects with LDND < 0.7 and keeps
same-concept word pairs whose score beats the 99th percentile of different-concept pairs, for 7 rounds.
The learned scores encode correspondence, not just phonetic closeness: d:t −0.38 vs d:p −7.85, and h:p −1.27
vs t:p −3.97 (Japonic *p > h).

`scripts/pmi_eval.py` tests whether a distance ranks four classes of doculect pairs in the right order:
same language (X / X_2), same WALS genus, same family but different genus, and unrelated.
That is about 1,000–1,300 pairs per class, and test doculects are excluded from training. The AUC is the
chance that a pair from the closer class scores below one from the farther class:

| separating… | LDND | PMI-LDND | PMI-ERC | LDN-ERC |
|---|---|---|---|---|
| same language vs same genus | 0.899 | **0.908** | 0.901 | 0.901 |
| same genus vs deeper in family | 0.929 | **0.936** | 0.924 | 0.921 |
| deeper in family vs unrelated | 0.852 | **0.892** | 0.848 | 0.811 |

PMI-LDND, which is LDND with the word distance 1 − 2·S(a,b)/(S(a,a)+S(b,b)), wins at every level. Paired bootstrap
95% intervals on the gain exclude zero: +0.009, +0.007 and +0.040, the largest at deep time depth. Per-concept rank
calibration (ERC) adds nothing over LDND's ratio. A feature-weighted Levenshtein (substitution cost = phonological
feature distance between ASJP symbols) was also tried on the same pairs and did slightly *worse* than LDND
(0.882 / 0.904 / 0.823): it rewards sounding alike, which also makes chance resemblances score closer, while
regular correspondences like h:p are far apart in features. The pipeline still uses LDND; switching
`run_analysis.py` to PMI-LDND (and re-calibrating k on its noise floor) is the next step.

## Run

```bash
git clone --depth 1 https://github.com/lexibank/asjp.git data/asjp
pip install -e '.[dev]'
pytest                               # property tests: duplicate invariance, monotonicity, QP == exact
python scripts/run_analysis.py       # → results/summary.{csv,json}, diversity_profiles.png, ldnd_*.tsv
python scripts/bootstrap.py          # → results/bootstrap.json
PYTHONPATH=scripts python scripts/cross_family.py   # → results/cross_family.{csv,json,png}
python scripts/build_explorer.py     # → results/explorer.html
python scripts/pmi_eval.py           # → results/pmi_model.tsv, pmi_eval.json, pmi_eval_pairs.csv
```

`--tokens segments` switches from ASJPcode to NFC-normalized CLDF IPA segments. Words are always
tuples of opaque segment tokens, so any script or segmentation works.

## Next steps

- Switch the pipeline to PMI-LDND (see above), which separates relatedness levels better than LDND.
- Use longer, expert-cognate-coded lists from Lexibank (Japonic, Koreanic, Ainu datasets) to get
  better resolution inside the isolate band.
- Report the whole D_max(k) curve as the diversity measure, rather than a single k.

## References

- Leinster, T. & C. A. Cobbold. 2012. Measuring diversity: the importance of species similarity. *Ecology* 93: 477–489.
- Leinster, T. & M. W. Meckes. 2016. Maximizing diversity in biology and beyond. *Entropy* 18(3): 88.
- Wichmann, S., E. W. Holman, D. Bakker & C. H. Brown. 2010. Evaluating linguistic distance measures. *Physica A* 389: 3632–3639.
- Jäger, G. 2013. Phylogenetic inference from word lists using weighted alignment with empirically determined weights. *Language Dynamics and Change* 3(2): 245–291.
- Jäger, G. 2018. Global-scale phylogenetic linguistic inference from lexical resources. *Scientific Data* 5: 180189.
- Wichmann, S., E. W. Holman, C. H. Brown, M. S. Dryer & Q. Ran (eds.). 2025. *The ASJP Database* v21. CLDF: github.com/lexibank/asjp (CC-BY-4.0).
