# isodiv — clock-free, dialect-count-robust diversity of language lineages

**Question:** can we say how internally diverse a lineage is (e.g. "is this an isolate?") without
(a) counting languages, and therefore without settling language vs. dialect, and
(b) assuming a rate of change, i.e. without glottochronology?

**Approach**

1. **Distances measured in amount of change, not time:** PMI-LDND on the 40-item ASJP lists. LDND
   (Wichmann et al. 2010) is Levenshtein distance normalized by word length, then divided by the same
   measure computed between words with *different* meanings, which corrects for chance resemblance.
   PMI-LDND keeps that construction but aligns words with learned sound-correspondence scores instead of
   a flat cost per changed segment (see [PMI distances](#pmi-distances-learned-sound-correspondences)).
   No time depth, no rate assumption. `--distance ldnd` reproduces the plain-LDND pipeline.
2. **Diversity that ignores duplicated varieties:** set Z = exp(−k·d) and compute the
   **Leinster–Meckes maximum diversity** D_max(Z). This is the effective number of fully distinct
   lineages. It is a property of Z alone (no abundance weights, no choice of order q), and adding
   an identical doculect cannot change it. Computed exactly by subset enumeration when n ≤ 14, as min pᵀZp on the
   probability simplex (multi-start SLSQP + polishing + KKT check) up to n = 120, and above that by Wolfe's
   min-norm-point method on the Gram matrix with an optimality check, falling back to SLSQP with column generation
   when Z is not positive semidefinite on the visited supports (PMI-LDND is not a metric).
   **Every lineage is scored on all its ASJP doculects** (up to 2,042 for Atlantic-Congo), and a lineage that
   contains others contains all of their doculects, so D_max can never fall from a subfamily to its family
   (see [Whole pools, nesting and the size ceiling](#whole-pools-nesting-and-the-size-ceiling)).
3. **Choosing k without a clock:** k_cal is set so that two doculects of the *same* language
   (same Glottocode, entries named X / X_2) are 0.9-similar. The median PMI-LDND between such pairs is
   0.228, which gives k_cal = 0.463 (LDND: 0.30 and 0.353). The full curve of D_max against k is plotted as well.

## Results (ASJP v21, `asjp` tokenization, PMI-LDND)

D_max at k_cal, on every doculect of each lineage. The interval is the 5–95% range over 30 draws, each keeping
30 of the 40 concepts (on at most 40 doculects, shown as a width around the estimate). The LDND column is the
previous pipeline (k_cal 0.353; Indo-European re-run on the whole pool, the rest from the 80-doculect cap, which
changes nothing for lineages with fewer than 80 doculects). The scales differ, so compare orderings and band gaps,
not absolute values.

| band | lineage | D_max | interval | LDND D_max |
|---|---|---|---|---|
| isolate-like | Basque | 1.032 | 1.026–1.038 | 1.037 |
| | Haida | 1.073 | 1.062–1.077 | 1.068 |
| | Koreanic | 1.115 | 1.102–1.128 | 1.110 |
| | Burushaski | 1.116 | 1.102–1.131 | 1.113 |
| | Nivkh | 1.139 | 1.129–1.154 | 1.131 |
| | Ainu | 1.159 | 1.150–1.173 | 1.136 |
| genus-like | Slavic | 1.238 | 1.224–1.254 | 1.220 |
| | Japonic | 1.248 | 1.234–1.264 | 1.223 |
| | Yeniseian | 1.264 | 1.251–1.274 | 1.236 |
| | Kartvelian | 1.271 | 1.263–1.279 | 1.230 |
| | Mongolic | 1.294 | 1.287–1.308 | 1.267 |
| | Germanic | 1.302 | 1.295–1.313 | 1.268 |
| | Romance | 1.307 | 1.294–1.320 | 1.271 |
| | Turkic | 1.324 | 1.313–1.336 | 1.281 |
| deep | Uralic | 1.456 | 1.447–1.466 | 1.358 |
| | Indo-European (517) | 1.507 | 1.499–1.518 | 1.386 |

Across the 72 lineages first compared, the D_max ranking barely changes (Spearman 0.995 against LDND; LDND has not
been re-run on the 128 whole-pool lineages). What changes is the spacing. The isolate band hardly moves, while the
deep families pull away: Uralic and Indo-European were about 0.08 above Turkic under LDND and are now 0.13 and 0.18
above it. PMI finds more shared signal between distantly related lists, so a
family's deep branches count as more distinct from each other relative to its shallow dialect clusters.

**Dialect-counting robustness (Japonic, D_max at k_cal):**

| sample | n | D_max | equal-weight |
|---|---|---|---|
| all doculects | 33 | 1.248 | 1.212 |
| one per Glottocode | 10 | 1.217 | 1.191 |
| all + Tokyo duplicated 3× | 36 | 1.248 (unchanged) | 1.214 |
| Tokyo + Shuri only | 2 | 1.126 | 1.126 |

D_max does not move when Tokyo is duplicated. The last row
falls because two doculects miss real Ryukyuan breadth (e.g. Yonaguni). That is under-sampling of
genuinely different varieties, not a counting artifact.

**Reading:**
- The bands do not overlap. Japonic measures as a genus-sized family, level with Slavic,
  so including Ryukyuan is not a counting artifact.
- Koreanic (with Jeju) stays in the isolate band.
- Yeniseian scores as genus-sized, so Ket is a "last survivor", not an isolate.
- Turkic now sits at the top of the genus band rather than at the bottom of the deep band, closer to its usual
  reading as a shallow family.

**Sinitic groups.** Every major group scores at genus level: Min 1.333, Mandarin 1.322, Pinghua 1.316, Wu 1.304,
Yue 1.294, Gan 1.290, Hakka 1.283, Hui 1.267, Xiang 1.260, Jin 1.242 (Romance 1.307, Germanic 1.302, Slavic 1.238),
and Sinitic as a whole 1.416 (482 doculects). Each group's dialect lists are about as lexically diverse as a whole Western European genus. This is not a
short-word or transcription artifact: Sinitic words average 3.67 segments (Germanic 3.80), and the 33 same-language
pairs inside Sinitic have median PMI-LDND 0.19, below the ASJP-wide 0.228. Wutun, heavily restructured by Tibetan and
Mongolic contact, was Mandarin's top-weighted list in the earlier 79-doculect sample (1.282 → 1.262 without it; not
yet redone on the 186-doculect pool, where Mandarin is 1.322).

**Ancient lists** (measured on the earlier 80-doculect samples; not yet redone on whole pools). Leaving out long-extinct doculects lowers Sinitic 1.389 → 1.375 (Old Chinese −0.009, Middle Chinese
−0.002) and Germanic 1.302 → 1.286 (eight lists incl. Proto-Germanic and Gothic), each about one concept-resampling
interval; every other lineage moves by 0.004 or less. Old Chinese gets Sinitic's largest D_max weight because it is the
most isolated point (nearest neighbour 0.59, Middle Chinese). Only lineages with a written past get this boost.

**Change, not time: Austronesian.** Austronesian (1.549, 1,446 doculects) outscores Indo-European (1.507, 517),
though it is younger. Size is not the whole reason: in the earlier rarefaction, at equal numbers of distinct
languages (3, 5, 10, 20, 40) Austronesian stayed above Indo-European (ad hoc analysis on the 80-doculect samples;
a script and a rerun on whole pools are still to do). Composition and rate matter. Oceanic alone (about 3,500
years; 659 doculects) scores 1.542, and its nest is monotone: Polynesian 1.249 ≤ Oceanic 1.542 ≤ Eastern
Malayo-Polynesian 1.544 ≤ Malayo-Polynesian 1.547 ≤ Austronesian 1.549. Formosan, where Austronesian's deepest splits
are, scores *lowest* (1.452; at 10 languages 1.376 vs Malayo-Polynesian 1.418 and Oceanic 1.438 in the earlier
rarefaction): its languages kept more of the Proto-Austronesian basic vocabulary. D_max measures how much basic vocabulary has changed, so a young, fast lineage
can outscore an old, slow one. It is a measure of lexical diversity, not of age.

**Semitic.** Semitic 1.436; West Semitic 1.433; South Semitic 1.386 (higher than Central Semitic 1.346, with
Ethiosemitic 1.343 the deepest part); Northwest Semitic 1.284, Arabic 1.272, Aramaic 1.229, Modern South Arabian
1.238. East Semitic (three Akkadian lists, 1.095) and Canaanite (4 lists, 1.180) are too thin to compare. Ancient
South Arabian's two lists have 19 concepts each, below the 20 shared concepts LDND needs, so it has no distances to
any other list (D_max 1, NaN in between-lineage tables).

**Ancient and single-list isolates** (`scripts/single_lists.py`). A single list has D_max = 1 by definition, so only
its distance to everything else is informative. For a random list from a known family, the nearest unrelated
lineage sits at mean PMI-LDND 0.908 (5–95%: 0.857–0.935). Sumerian (nearest Etruscan 0.880), Etruscan (Nivkh 0.863;
24 concepts), Hruso, Kusunda and Nihali are within chance. Elamite is closest to Basque (0.851, bottom 5%, as expected
for one of six isolates by chance) and only 14th of 69 to Dravidian (0.925): no Elamo-Dravidian signal.

## Constructed sets (adversarial check)

`scripts/adversarial.py` scores sets that are *not* families, on the same scale. D_max measures diversity, not
relatedness, so unrelated lects should score at or above real families:

| set | D_max at k_cal | at k = 3 |
|---|---|---|
| 10 random unrelated lects (one per family, 30 draws) | 1.484 | 6.7 |
| 25 random unrelated lects | 1.535 | 10.9 |
| 80 random unrelated lects | 1.563 | 15.2 |
| Atlantic-Congo (2,042 doculects) | 1.563 | 16.9 |
| Nuclear Trans New Guinea (307) | 1.557 | 15.9 |
| Austronesian (1,446) | 1.549 | 15.6 |
| Indo-European (517) | 1.507 | 12.3 |
| Uralic (64) | 1.456 | 8.8 |

The random sets stay below the real families only because they are small: a family with hundreds of documented
doculects has more points to spread weight over. At the same size an unrelated assortment scores higher, as it must
(see the relative score below). At k_cal an unrelated pair is still z = e^(−k_cal·0.97) ≈ 0.64 similar, so the scale
saturates near 1/z ≈ 1.57, so the gaps are small there and clearer at k = 3.

Proposed or rejected groupings, as unions of their member lineages (capped at 80 doculects), against random unions of
as many unrelated families, of any depth and of *shallow* (genus-level) families only:

| grouping | D_max | any-depth unions | shallow unions | at k = 3 | shallow at k = 3 |
|---|---|---|---|---|---|
| Altaic (Turkic, Mongolic, Tungusic) | 1.448 | 1.487 (1.435–1.533) | 1.467 (1.451–1.482) | 8.31 | 8.17 (7.66–8.89) |
| "Khoisan" (Khoe-Kwadi, Tuu, Kxa) | 1.452 | same | same | 7.59 | same |
| Macro-Altaic (+ Koreanic, Japonic) | 1.492 | 1.518 (1.487–1.534) | 1.507 (1.500–1.516) | 9.64 | 10.36 (9.89–10.86) |
| Ural-Altaic | 1.499 | 1.510 (1.482–1.534) | 1.498 (1.485–1.505) | 10.58 | 9.67 (8.84–10.17) |
| Dene-Yeniseian | 1.466 | 1.435 (1.287–1.506) | 1.419 (1.405–1.442) | 8.71 | 6.65 (5.73–7.27) |
| "Paleo-Siberian" (Chukotko-Kamchatkan, Yukaghir, Nivkh, Yeniseian, Ainu) | 1.480 | as Macro-Altaic | as Macro-Altaic | 8.22 | as Macro-Altaic |

The constructed sets are unions of their members' samples capped at 80 doculects, as before; the random baselines are
drawn the same way.

Altaic scores below Indo-European and Uralic, but that is what three shallow families give whether or not they are
related: it sits at the bottom edge of the shallow baseline at k_cal and in its middle at k = 3. The shallow baseline
does not fit groupings with deep (Uralic, Athabaskan-Eyak-Tlingit) or isolate members, which score above or below it
for that reason. Basic-vocabulary *distance* is where Altaic stands out: Mongolic ↔ Tungusic (0.868) is the closest
unrelated pair in the data and Turkic ↔ Mongolic (0.925) is in the closest 1%. But the "Khoisan" pairs, an areal
(Kalahari Basin) grouping, are as close (Tuu ↔ Kxa 0.857, Khoe-Kwadi ↔ Tuu 0.876), so contact alone produces this
signal; neither measure separates it from descent. Koreanic ↔ Japonic (1.012) is farther apart than 99% of unrelated
pairs, and Dene-Yeniseian (0.956) is unremarkable, which is expected for a proposal argued from morphology.

**Niger-Congo** (Glottolog splits it into Atlantic-Congo, Mande, Kru, Ijoid, Dogon and the Kordofanian groups): the
nine-branch union scores 1.527, slightly *below* random nine-family unions (1.536, 1.532–1.543), as large unions of
deep families saturate. Distances to Atlantic-Congo: Mande 0.949, Kru 0.950, Ijoid 0.947, Dogon 0.947, each closer
than about 90% of unrelated lineage pairs, a weak signal that West African contact could also produce; the
Kordofanian groups (0.964–0.977) are at chance. Elamo-Dravidian (1.413) is at its baseline (1.419).

The explorer shows these sets as a separate "constructed" category.

## Whole pools, nesting and the size ceiling

**Parent below child.** The first version capped each lineage at 80 doculects separately. Oceanic then scored 1.522 and
Austronesian 1.514, which cannot be right: D_max never falls when points are added, because a larger set can give the
extra points zero weight. The 80-doculect Austronesian sample held only 23 Oceanic lists, while the Oceanic sample
was all Oceanic, and Oceanic is the fast-changing part. Scoring the union of the two samples gave 1.528, above both.
The first fix (a parent's sample contains its children's) made the result depend on which child was largest, so
Oceanic went to 1.490 on a sample that was two thirds Polynesian. The cure was to remove the sampling: every lineage
is now scored on all its doculects, so nesting is exact (33 parent/child pairs, 0 violations at any scale, checked on
every run and recorded in `summary.json`). Lineages defined by Glottolog node and by ASJP Family disagree on a handful
of doculects with a blank Family; near-containment (at most 2% outside) counts as nesting.

**The ceiling.** Related lects are closer than unrelated ones, so a real family scores below an unrelated
assortment of the same size, and at k_cal both are squeezed under 1/z̄ ≈ 1.56. Whole-pool scores of the deepest
families are now near that ceiling (Atlantic-Congo 1.563, Nuclear Trans New Guinea 1.557, Austronesian 1.549). Two
tools undo the compression:

- **Share of the unrelated maximum**, `(D − 1) / (D_unrelated(n) − 1)`, where D_unrelated(n) is the median D_max of n
  lects from n random families (measured up to n = 80 by `scripts/adversarial.py`, extended with the ceiling
  n / (1 + (n − 1)·z̄)). It is in `results/relative.csv` and the explorer. Atlantic-Congo 0.98, Nuclear Trans New
  Guinea 0.97, Austronesian 0.96, Indo-European 0.89, Uralic 0.81, Sinitic 0.73, Turkic 0.58, Germanic 0.54, Japonic
  0.46, Basque 0.18. The deepest families all land between 0.9 and 1, so this separates shallow and mid-depth
  lineages and leaves the deep ones bunched.
- **The whole D_max(k) curve** (explorer, "Scale profiles", also as a share of the unrelated maximum). At k = 3 the
  ceiling is gone: Atlantic-Congo 16.9, Nuclear Trans New Guinea 15.9, Austronesian 15.6, Indo-European 12.3.

**Speed.** The whole-pool run needs about 3M distance pairs and thousands of D_max solves on matrices up to 2,042
points. `pip install -e '.[fast]'` adds numba, which compiles the alignment and LDND aggregation into one parallel loop
(240k Oceanic pairs: 241 s → 12 s, results identical to 4e-15; the numpy path stays as the fallback and the test
reference). D_max uses Wolfe's min-norm-point method with an incrementally updated Cholesky factor (Atlantic-Congo,
2,042 doculects, every scale under 15 s), the 50 scales run in separate processes with one BLAS thread each, and
`data/cache/` holds every distance once so nested lineages and `audit.py` reuse them. Oceanic's full analysis went from
568 s to 24 s. One scale (Atlantic-Congo at k = 40, where D_max is essentially the count of distinct doculects) is not
certified optimal. Wolfe's method assumes Z is positive semidefinite; PMI-LDND is not a metric (two lists can be at
distance 0 from each other and at different distances from a third), so acceptance uses the objective gain a violating
point could still give, which is second order, and anything failing it goes to the slower general solver.

## Caveats

- **Noise floor.** Same-language ASJP doculect pairs have PMI-LDND 0.086 / 0.228 / 0.474 at the 10th /
  50th / 90th percentile (LDND 0.11 / 0.30 / 0.57). At this resolution the isolate band overlaps "one language transcribed
  several times", so lineages *within* that band cannot be ranked against each other.
- **Sampling.** D_max is invariant to *duplicates*, not to documenting *more distinct* varieties.
  That is intended, since more distinct varieties is real diversity, but undersampled lineages
  (Basque n=2, Haida n=2) are lower bounds, and a lineage with many documented doculects scores higher than a
  comparable one with few. Compare lineages of very different size with the share of the unrelated maximum. `spread` and `spread_rarefied_m3` in
  `results/summary.csv` show this sample-size dependence.
- **Contact** shortens distances (e.g. Ryukyuan under Japanese influence). The 40-item list is
  borrowing-resistant, but not borrowing-proof.
- The intervals reflect concept resampling only, not doculect choice or transcription noise.

## Lineages

Each lineage is scored on all its ASJP doculects (minus the exclusions, and minus doculects that share fewer than 20
concepts with too many others), and `--cap N` restores a per-lineage cap. 128 lineages: 70 top-level (the original 16, 32 more ASJP families, six single- or few-list isolates incl. Sumerian,
Elamite and Etruscan, and 16 small families incl. Mande, Kru, Ijoid, Dogon and the Kordofanian groups), 24 subfamilies selected by node in ASJP's
Glottolog classification path (`SUBFAMILIES` in `scripts/run_analysis.py`: Semitic, Arabic, Sinitic, Min, Indo-Aryan,
Iranian, Oceanic, Polynesian, ...), and nine major Sinitic groups (`SINITIC_GROUPS`: Mandarin, Jin, Wu, Hui, Gan,
Xiang, Hakka, Yue, Pinghua). Glottolog treats most of those groups as one language each, so they are picked by
Glottocode; Mandarin and Pinghua by node. `NODE_GROUPS` adds 25 more by node: Austronesian (Malayo-Polynesian,
Formosan = Austronesian outside Malayo-Polynesian, Eastern MP, Greater Central Philippine, Malayic, Celebic) and
Semitic (East/West/Central/Northwest Semitic, Aramaic, NENA, Canaanite, South Semitic = Ethiosemitic + MSA + Sayhadic,
Ethiosemitic, North/South/Outer South Ethiopic, Modern and Ancient South Arabian, five Arabic groups). ASJP has no
Hurrian or Urartian list. Two lineages count as related, and are left out of the unrelated-pair
statistics, whenever their doculects share a top-level Family.

## Explorer and cross-family check

- `scripts/cross_family.py` compares lineages with each other: mean distance within and between the 128 lineages
  (see Lineages), on 25 doculects per lineage, and D_max of two lineages together vs. apart.
  Japonic ↔ Kartvelian: mean PMI-LDND 0.98 vs. 0.43 / 0.60 within. Across 7,544 unrelated lineage pairs the mean is
  0.971 (LDND 0.984 over 2,483 pairs, from the earlier 72-lineage run). The closest are Tuu ↔ Kxa 0.857, Mongolic ↔ Tungusic 0.868, Koreanic ↔ Burushaski 0.872 and
  Khoe-Kwadi ↔ Tuu 0.876 (LDND 0.911–0.929, same pairs). Kalahari Basin and Altaic-area contact fit the others;
  Koreanic ↔ Burushaski (3 vs 10 doculects, no contact) shows PMI can also pull chance resemblances closer.
- `scripts/build_explorer.py` bundles `results/*.json` into `results/explorer.html`, a self-contained
  page (diversity ranking, scale profiles, between-lineage heatmap).

## Excluding bad data

`config/exclusions.tsv` (`id<TAB>reason`) lists doculects left out of every script. `scripts/audit.py` ranks
doculects by how much removing them lowers their lineage's D_max (also flags thin coverage) and writes
`results/audit.csv`; it never excludes anything itself. Currently excluded: `AYNU` (Turkic list with heavy
Persian replacement; under LDND, Turkic D_max 1.304 → 1.281). Review the audit, edit the file, rerun.
The leave-one-out flag (0.015) is in D_max units, so the wider PMI-LDND scale flags more doculects than LDND did, and
whole pools flag more again (81 on the whole-pool run, against 32 on capped samples; 13 are in Dogon, 4 in Nubian).
A doculect with no weight in the D_max optimum cannot lower it, so only weighted candidates get a leave-one-out solve.

`scripts/loans.py` adds a contact check: for each doculect, how much closer is it to a *neighbouring unrelated
language* (within 2000 km) than its own relatives are (`gap`, in LDND units; this check stays on plain LDND,
since its thresholds were set in those units)? Real contact cases stand out
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
Training uses no classification. It screens random doculect pairs, keeps those with LDND < 0.7, then keeps
same-concept word pairs whose score beats the 99th percentile of different-concept pairs, re-estimating until stable.
`scripts/pmi_train.py` trains the pipeline's model on all of ASJP (1M pairs screened, 3,939 kept;
`results/pmi_model_all.tsv`). The evaluation below trains its own model with the test doculects held out.
The learned scores (evaluation model) encode correspondence, not just phonetic closeness: d:t −0.81 vs d:p −7.16, and h:p −0.49
vs t:p −3.49 (Japonic *p > h).

`scripts/pmi_eval.py` tests whether a distance ranks classes of doculect pairs in the right order:
same language (X / X_2), same WALS genus, same family but different genus, and unrelated.
That is about 1,000–1,300 pairs per class, and test doculects are excluded from training. The AUC is the
chance that a pair from the closer class scores below one from the farther class:

| separating… | LDND | PMI-LDND | PMI-ERC | LDN-ERC |
|---|---|---|---|---|
| same language vs same genus | 0.899 | **0.907** | 0.899 | 0.901 |
| same genus vs deeper in family | 0.929 | **0.935** | 0.922 | 0.921 |
| deeper in family vs unrelated | 0.852 | **0.891** | 0.852 | 0.811 |
| deeper in family vs unrelated neighbours (< 500 km) | 0.740 | **0.778** | 0.748 | 0.726 |

PMI-LDND, which is LDND with the word distance 1 − 2·S(a,b)/(S(a,a)+S(b,b)), wins at every level. Paired bootstrap
95% intervals on the gain exclude zero: +0.009, +0.007 and +0.040, the largest at deep time depth. Per-concept rank
calibration (ERC) adds nothing over LDND's ratio. A feature-weighted Levenshtein (substitution cost = phonological
feature distance between ASJP symbols) was also tried on the same pairs and did slightly *worse* than LDND
(0.882 / 0.904 / 0.823): it rewards sounding alike, which also makes chance resemblances score closer, while
regular correspondences like h:p are far apart in features.

**Training pairs and contact.** Choosing training pairs by geographic neighbourhood (`--train geo`, also with
same-language duplicates removed: `geo-no-dup`) gives the same AUCs within ±0.002, so the result does not depend
on how training pairs are found. Any similarity-based selection is dominated by shallow pairs (in `geo`, 52% two
lists of one language, 44% same genus). Unrelated pairs almost never pass LDND < 0.7: 7 of 5,212 neighbour
pairs, all pidgins with their lexifier. Unrelated *neighbours* are closer than distant unrelated pairs under every
distance (AUC 0.64–0.67), so ASJP lists carry some areal signal. PMI-LDND picks up slightly more of it (0.668 vs
0.644), but it separates real deep relatives from unrelated neighbours better than LDND (0.778 vs 0.740).

Every pipeline script takes `--distance pmi|ldnd`. `run_analysis.py` defaults to `pmi` and records the choice in
`summary.json`; the later scripts follow it. Files and fields keep their `ldnd` names (`ldnd_*.tsv`,
`max_ldnd`) and hold whichever distance was used.

## Run

```bash
git clone --depth 1 https://github.com/lexibank/asjp.git data/asjp
pip install -e '.[dev,fast]'           # fast: numba kernel for PMI-LDND (optional; ~20x faster distances)
pytest                               # property tests: duplicate invariance, monotonicity, QP == exact, PMI alignment
python scripts/pmi_train.py          # → results/pmi_model_all.tsv  (~1 min; skip with --distance ldnd)
python scripts/run_analysis.py       # → results/summary.{csv,json}, samples.json, profiles.json, ldnd_*.tsv (n ≤ 200), data/cache/  (~1 h, 4 cores, numba)
python scripts/bootstrap.py          # → results/bootstrap.json  (~5 min)
PYTHONPATH=scripts python scripts/cross_family.py   # → results/cross_family.{csv,json,png}  (~30 min)
PYTHONPATH=scripts python scripts/audit.py          # → results/audit.{csv,json}  (~3 min, reads data/cache/)
PYTHONPATH=scripts python scripts/adversarial.py    # → results/adversarial.{csv,json}  (~4 min)
PYTHONPATH=scripts python scripts/single_lists.py   # → results/single_lists.json
python scripts/build_explorer.py     # → results/explorer.html, results/relative.csv
python scripts/pmi_eval.py           # → results/pmi_model.tsv, pmi_eval.json, pmi_eval_pairs.csv (--train geo|geo-no-dup to compare)
```

`--tokens segments` switches from ASJPcode to NFC-normalized CLDF IPA segments. Words are always
tuples of opaque segment tokens, so any script or segmentation works.

## Next steps

- Recalibrate `loans.py` thresholds in PMI-LDND units, and check whether PMI's pull on chance resemblances
  (Koreanic ↔ Burushaski) needs a correction.
- Use longer, expert-cognate-coded lists from Lexibank (Japonic, Koreanic, Ainu datasets) to get
  better resolution inside the isolate band.
- Redo on whole pools the analyses still quoted from the 80-doculect samples: the Wutun effect on Mandarin, the
  long-extinct doculects (Old Chinese, Proto-Germanic; add an option to exclude them and report both), and the
  Austronesian rarefaction (save it as a script).
- Report the D_max(k) curve and the share of the unrelated maximum as the headline, not one D_max at k_cal. Both are
  in the explorer; the tables above still quote k_cal.
- For a paper: compare with Holman et al. 2011 family ages, add a doculect-level bootstrap and a simulation on
  synthetic trees. LDND has not been re-run on the 128 whole-pool lineages.

## References

- Leinster, T. & C. A. Cobbold. 2012. Measuring diversity: the importance of species similarity. *Ecology* 93: 477–489.
- Leinster, T. & M. W. Meckes. 2016. Maximizing diversity in biology and beyond. *Entropy* 18(3): 88.
- Wichmann, S., E. W. Holman, D. Bakker & C. H. Brown. 2010. Evaluating linguistic distance measures. *Physica A* 389: 3632–3639.
- Jäger, G. 2013. Phylogenetic inference from word lists using weighted alignment with empirically determined weights. *Language Dynamics and Change* 3(2): 245–291.
- Jäger, G. 2018. Global-scale phylogenetic linguistic inference from lexical resources. *Scientific Data* 5: 180189.
- Wichmann, S., E. W. Holman, C. H. Brown, M. S. Dryer & Q. Ran (eds.). 2025. *The ASJP Database* v21. CLDF: github.com/lexibank/asjp (CC-BY-4.0).
