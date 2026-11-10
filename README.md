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
2. **Diversity that ignores duplicated varieties:** turn distances into similarities Z and compute the
   **Leinster–Meckes maximum diversity** D_max(Z). This is the effective number of fully distinct
   lineages. It is a property of Z alone (no abundance weights, no choice of order q), and adding
   an identical doculect cannot change it. Computed exactly by subset enumeration when n ≤ 14, as min pᵀZp on the
   probability simplex (multi-start SLSQP + polishing + KKT check) up to n = 120, and above that by Wolfe's
   min-norm-point method on the Gram matrix with an optimality check, falling back to SLSQP with column generation
   when Z is not positive semidefinite on the visited supports (PMI-LDND is not a metric).
   **Every lineage is scored on all its ASJP doculects** (up to 2,037 for Atlantic-Congo), and a lineage that
   contains others contains all of their doculects, so D_max can never fall from a subfamily to its family
   (see [Whole pools, nesting and size](#whole-pools-nesting-and-size)).
3. **Similarity anchored by evidence, not by a clock or a fitted constant:** Z = 1 at the median distance between
   two doculects of the *same* language (same Glottocode, entries named X / X_2; PMI-LDND 0.228) and falls linearly
   to Z = 0 at **0.940**, the distance at which a permutation test detects a related pair half the time
   (`scripts/regularity_eval.py`). Pairs farther apart than that are indistinguishable from unrelated ones, so they
   count as fully distinct: **n unrelated lects score about n, and D_max reads as the effective number of lineages a
   lineage's word lists cannot tie together** (see [Calibrated similarity](#calibrated-similarity)). The older
   similarity exp(−k·d), with k_cal = 0.463 set so same-language pairs are 0.9 similar, is still reported
   (`d_max_kcal`) and drawn as the D_max(k) curve in the explorer.

## Results (ASJP v21, `asjp` tokenization, PMI-LDND)

D_max on the calibrated similarity (effective number of lineages the 40-word lists cannot tie together), on every
doculect of each lineage. The interval is the 5–95% width of 30 draws, each keeping 30 of the 40 concepts (on at most
40 doculects), placed around the estimate. The last column is the older exp(−k_cal·d) scale, kept for comparison: it
gives the same ordering, squeezed under 1.57.

| band | lineage (doculects) | D_max | interval | at k_cal |
|---|---|---|---|---|
| isolate-like | Basque (2) | 1.00 | 1.00–1.00 | 1.032 |
| | Haida (2) | 1.07 | 1.03–1.08 | 1.073 |
| | Burushaski (10) | 1.15 | 1.09–1.19 | 1.116 |
| | Koreanic (3) | 1.17 | 1.13–1.22 | 1.115 |
| | Nivkh (7) | 1.21 | 1.15–1.26 | 1.139 |
| | Ainu (23) | 1.29 | 1.24–1.37 | 1.159 |
| genus-like | Slavic (34) | 1.61 | 1.52–1.71 | 1.238 |
| | Japonic (33) | 1.69 | 1.59–1.79 | 1.248 |
| | Yeniseian (9) | 1.91 | 1.79–2.01 | 1.264 |
| | Mongolic (29) | 2.04 | 1.96–2.21 | 1.294 |
| | Kartvelian (6) | 2.12 | 2.03–2.23 | 1.271 |
| | Romance (64) | 2.14 | 2.02–2.29 | 1.307 |
| | Germanic (61) | 2.16 | 2.09–2.28 | 1.302 |
| | Turkic (71) | 2.43 | 2.30–2.59 | 1.324 |
| deep | Semitic (108) | 5.24 | 4.94–5.48 | 1.436 |
| | Uralic (64) | 6.54 | 5.95–7.31 | 1.456 |
| | Indo-European (490, with Hittite) | 14.09 | 13.39–15.11 | 1.506 |

The largest lineages: Atlantic-Congo 71.1 (2,037 doculects), Nuclear Trans New Guinea 47.4, Austronesian 43.8,
Afro-Asiatic 33.5, Pama-Nyungan 22.5, Sino-Tibetan 21.4, Chadic 18.2, Otomanguean 18.1, Indo-European 14.1.

The ordering is almost the same as at k_cal; what changes is the spacing. The isolate and genus bands now read
directly: Germanic is about two lineages the 40-word lists cannot tie together (Proto-Germanic carries the largest
weight), Turkic about two and a half (Chuvash carries the largest), and Indo-European about fourteen. A few
small lineages move up within the genus band (Kartvelian 2.12 with 6 lists, Yeniseian 1.91 with 9): on this scale a
lineage with two deeply split branches scores about 2 however few lists it has, where k_cal rewarded many partly
distinct lists.

**Dialect-counting robustness (Japonic):**

| sample | n | D_max | at k_cal |
|---|---|---|---|
| all doculects | 33 | 1.69 | 1.248 |
| one per Glottocode | 10 | 1.54 | 1.217 |
| all + Tokyo duplicated 3× | 36 | 1.69 (unchanged) | 1.248 |
| Tokyo + Shuri only | 2 | 1.29 | 1.126 |

D_max does not move when Tokyo is duplicated. The last row falls because two doculects miss real Ryukyuan breadth
(e.g. Yonaguni). That is under-sampling of genuinely different varieties, not a counting artifact.

**Reading:**
- The bands do not overlap. Japonic measures as a genus-sized family, level with Slavic,
  so including Ryukyuan is not a counting artifact.
- Koreanic (with Jeju) stays in the isolate band.
- Yeniseian scores as genus-sized, so Ket is a "last survivor", not an isolate.
- Turkic sits at the top of the genus band, far below Uralic and Indo-European.

**Sinitic groups.** Every major group scores at genus level: Min 2.51, Pinghua 2.30, Wu 2.12, Gan 2.08,
Mandarin 2.04, Hakka 1.97, Yue 1.88, Xiang 1.81, Hui 1.79, Jin 1.65 (Germanic 2.16, Romance 2.14, Slavic 1.61), and
Sinitic as a whole 4.29 (484 doculects). Each group's dialect lists are about as lexically diverse as a whole Western
European genus. This is not a short-word or transcription artifact: Sinitic words average 3.67 segments (Germanic
3.80), and the 33 same-language pairs inside Sinitic have median PMI-LDND 0.19, below the ASJP-wide 0.228. Wutun,
heavily restructured by Tibetan and Mongolic contact, is excluded as a mixed language (see below).

**Ancient lists and contact** (`scripts/sensitivity.py`, whole pools; the main results keep every doculect). Leaving
out the doculects ASJP flags `long_extinct` (mostly lineages with a written past) lowers Germanic 2.16 → 1.98 (8 lists
incl. Proto-Germanic and Gothic), Sinitic 4.29 → 4.03 (Old and Middle Chinese), Semitic 5.24 → 4.93, Northwest Semitic
2.37 → 2.18, North Ethiopic 1.41 → 1.35 and Canaanite 1.44 → 1.00 (2 of its 4 lists). Indo-European goes 14.09 → 13.68
with 16 long-extinct lists removed (Hittite, Tocharian and others are each nearly a lineage of their own at 40 words);
Afro-Asiatic 33.53 → 33.44. Sumerian, Elamite, Etruscan and East Semitic consist only of long-extinct lists and
have nothing left. Wutun, the contact language, was the other scenario; it is now excluded
as a mixed language (see [Excluding bad data](#excluding-bad-data)), so that scenario removes nothing. Both versions
are in `results/sensitivity.csv` and in the explorer's detail panel (a "Sensitivity" block).

**Change, not time: Austronesian** (`scripts/rarefaction.py`: 100 draws of m distinct languages, one random doculect
each, from each lineage's whole pool). Austronesian (43.8, 1,459 doculects) outscores Indo-European (14.1, 490) on
the whole pools, though it is younger, but most of that is size. At equal numbers of languages the two are close and
separate slowly: P(Austronesian > Indo-European) is 0.48 at m = 3, 0.63 at 10, 0.79 at 20 and 0.94 at 40 (D_max 9.4
vs 7.4 at 40). Composition and rate matter: Oceanic (about 3,500 years) is above Indo-European at every size (P 0.59
at 3, 0.87 at 10, 1.00 at 40; 11.3 vs 7.4 at 40) and above Malayo-Polynesian as a whole (P 0.58 to 0.82). Formosan,
where Austronesian's deepest splits are, is *lowest* of the Austronesian groups and below Indo-European (m = 10: 3.7,
against Malayo-Polynesian 4.6, Oceanic 5.4 and Indo-European 4.3; P(Formosan > Oceanic) 0.01): its languages kept more
of the Proto-Austronesian basic vocabulary. The nest is monotone on the whole pools: Polynesian 1.67 ≤ Oceanic 33.7 ≤
Eastern Malayo-Polynesian 36.2 ≤ Malayo-Polynesian 41.0 ≤ Austronesian 43.8.

Austronesian is not the most diverse family at equal size either. At 40 languages Atlantic-Congo (15.9) and
Afro-Asiatic (15.1) are above Pama-Nyungan (12.0), Oceanic (11.3), Austronesian (9.4), Sino-Tibetan (9.2) and
Indo-European (7.4); Pama-Nyungan is above Indo-European at every size (P 0.71 at 3, 1.00 at 40). D_max measures how
much basic vocabulary has changed, so a young, fast lineage can outscore an old, slow one. It is a measure of lexical
diversity, not of age. The table is in `results/rarefaction.csv`.

**Semitic.** Semitic 5.24; West Semitic 5.07; South Semitic 3.50 (higher than Central Semitic 2.91, with Ethiosemitic
2.67 the deepest part); Northwest Semitic 2.37, Arabic 1.80, Aramaic 1.80, Modern South Arabian 1.62. East Semitic
(three Akkadian lists, 1.11) and Canaanite (4 lists, 1.44) are too thin to compare. Ancient South Arabian's two lists
have 19 concepts each, below the 20 shared concepts LDND needs, so it has no distances to any other list (D_max 1, NaN
in between-lineage tables).

**Ancient and single-list isolates** (`scripts/single_lists.py`). A single list has D_max = 1 by definition, so only
its distance to everything else is informative. For a random list from a known family, the nearest unrelated
lineage sits at mean PMI-LDND 0.907 (5–95%: 0.861–0.934). Sumerian (nearest Etruscan 0.880), Etruscan (Nivkh 0.863;
24 concepts), Hruso, Kusunda and Nihali are within chance. Elamite is closest to Basque (0.851, bottom 5%, as expected
for one of six isolates by chance) and only 14th of 69 to Dravidian (0.925): no Elamo-Dravidian signal.

## Constructed sets (adversarial check)

`scripts/adversarial.py` scores sets that are *not* families, on the same scale. Lects from different families are
almost all beyond the detection limit of each other, so n of them should score about n:

| set | D_max | 5–95% | at k_cal |
|---|---|---|---|
| 5 random unrelated lects (one per family, 30 draws) | 4.95 | 4.69–5.00 | 1.410 |
| 10 random unrelated lects | 9.68 | 9.26–9.96 | 1.486 |
| 25 random unrelated lects | 22.6 | 21.7–23.8 | 1.536 |
| 40 random unrelated lects | 34.7 | 32.8–36.7 | 1.550 |
| 80 random unrelated lects | 63.0 | 60.5–66.6 | 1.563 |
| Atlantic-Congo (2,037 doculects) | 71.1 | | 1.563 |
| Nuclear Trans New Guinea (307) | 47.4 | | 1.557 |
| Austronesian (1,459) | 43.8 | | 1.549 |
| Indo-European (490) | 14.1 | | 1.506 |
| Uralic (64) | 6.54 | | 1.456 |

The random sets fall a little short of n as they grow (a few pairs of families are within the detection limit of
each other, by chance or contact). Every real family scores far below an unrelated set of the same size: 40 random
Indo-European languages score 7.4 (rarefaction above), 40 unrelated lects 34.7. A whole family can outscore a small
random set only by holding more lineages that 40 words cannot tie together, which is what D_max now counts.

Proposed or rejected groupings, as unions of their member lineages (capped at 80 doculects), against random unions of
as many unrelated families, of any depth and of *shallow* (genus-level) families only:

| grouping | D_max | any-depth unions | shallow unions |
|---|---|---|---|
| Altaic (Turkic, Mongolic, Tungusic) | 5.71 | 9.5 (5.3–20.3) | 6.75 (5.92–7.45) |
| "Khoisan" (Khoe-Kwadi, Tuu, Kxa) | 5.52 | same | same |
| Ural-Altaic | 9.92 | 10.2 (4.9–18.1) | 8.78 (7.72–10.54) |
| Macro-Altaic (+ Koreanic, Japonic) | 8.17 | 13.7 (8.2–20.3) | 10.7 (9.6–11.9) |
| "Paleo-Siberian" (Chukotko-Kamchatkan, Yukaghir, Nivkh, Yeniseian, Ainu) | 7.60 | as Macro-Altaic | as Macro-Altaic |
| Dene-Yeniseian | 7.01 | 7.3 (3.2–18.4) | 4.66 (3.93–5.24) |
| Elamo-Dravidian | 4.59 | as Dene-Yeniseian | as Dene-Yeniseian |
| Niger-Congo (9 branches) | 17.8 | 18.6 (16.3–25.1) | 18.2 (17.1–19.8) |

The constructed sets are unions of their members' samples capped at 80 doculects, as before; the random baselines are
drawn the same way. The any-depth baseline is wide because it mixes isolates with families like Atlantic-Congo.

Altaic (5.71) now falls below the 5% edge of random unions of three shallow families (5.92), and "Khoisan" (5.52)
falls further below it. On this scale a union of unrelated lineages scores about the sum of its members, so
falling short means that some cross-family pairs are within the detection limit. Basic-vocabulary *distance* says the
same: Mongolic ↔ Tungusic (0.868) is among the closest unrelated pairs in the data (only Tuu ↔ Kxa, 0.846, and
Basque ↔ Elamite, 0.851, are closer) and Turkic ↔ Mongolic (0.925) is in the closest 2%. But the "Khoisan" pairs, an areal (Kalahari Basin) grouping, are as close (Tuu ↔ Kxa 0.846, Khoe-Kwadi ↔
Tuu 0.869), so contact alone produces this signal; neither measure separates it from descent. Macro-Altaic and
"Paleo-Siberian" fall below the shallow baseline mainly because they contain isolate-like members (Koreanic 1.17,
Nivkh 1.21, Ainu 1.29), which the shallow pool does not; the baseline does not fit groupings with deep (Uralic,
Athabaskan-Eyak-Tlingit) or isolate members. Koreanic ↔ Japonic (1.012) is farther apart than 99% of unrelated pairs,
and Dene-Yeniseian (0.956) is unremarkable, which is expected for a proposal argued from morphology.

**Niger-Congo** (Glottolog splits it into Atlantic-Congo, Mande, Kru, Ijoid, Dogon and the Kordofanian groups): the
nine-branch union scores 17.8, near the median of random unions of nine shallow families (18.2) and near the low end of
any-depth ones (18.6, 16.3–25.1). Distances to Atlantic-Congo: Mande 0.949, Kru 0.950, Ijoid 0.947, Dogon 0.947, each
closer than about 90% of unrelated lineage pairs, a weak signal that West African contact could also produce; the
Kordofanian groups (0.964–0.977) are at chance. Elamo-Dravidian (4.59) is at its shallow baseline (4.66).

The explorer shows these sets as a separate "constructed" category.

## Whole pools, nesting and size

**Parent below child.** The first version capped each lineage at 80 doculects separately. Oceanic then scored 1.522 and
Austronesian 1.514 (at the scale used then), which cannot be right: D_max never falls when points are added, because a
larger set can give the extra points zero weight. The 80-doculect Austronesian sample held only 23 Oceanic lists,
while the Oceanic sample was all Oceanic, and Oceanic is the fast-changing part. Scoring the union of the two samples
gave 1.528, above both. The first fix (a parent's sample contains its children's) made the result depend on which
child was largest, so Oceanic went to 1.490 on a sample that was two thirds Polynesian. The cure was to remove the
sampling: every lineage is now scored on all its doculects, so nesting is exact (58 parent/child pairs, 0 with the
parent below the child at any scale, checked on every run and recorded in `summary.json`). 25 of the 58 children are
not strict subsets of their parent (1–10 lists each): a list that shares fewer than 20 concepts with another list in
the parent can be dropped there and still be kept in the child, where that clash does not occur. `summary.json` lists
them under `nesting.not_subset`. Lineages defined by Glottolog node and by ASJP Family disagree on a handful of
doculects with a blank Family; near-containment (at most 2% outside) counts as nesting.

**Size.** On the calibrated scale there is no ceiling: D_max grows with the number of lineages a family holds that
the data cannot tie together, so the biggest, deepest families lead. Three views put families of different size on
an equal footing:

- **Equal-size rarefaction** (above, `results/rarefaction.csv`): the fairest comparison of how much basic vocabulary
  has changed.
- **Share of the unrelated maximum**, `(D − 1) / (D_unrelated(n) − 1)`, where D_unrelated(n) is the median D_max of n
  lects from n random families (measured up to n = 80 by `scripts/adversarial.py`, extended beyond with the
  equal-similarity curve n / (1 + (n − 1)·z̄)). It is in `results/relative.csv` and the explorer: Atlantic-Congo 0.34,
  Nuclear Trans New Guinea 0.35, Afro-Asiatic 0.22, Austronesian 0.22, Pama-Nyungan 0.20, Uralic 0.11, Indo-European
  0.08, Germanic, Turkic and Japonic 0.02, Basque 0. Its n counts doculects, so a heavily sampled lineage (Sinitic, 484
  doculects, 0.02) is compared against a very large unrelated set; read it next to rarefaction, not instead of it.
- **The whole D_max(k) curve** of the older exp(−k·d) similarity (explorer, "Scale profiles", also as a share of the
  unrelated maximum).

**Speed.** The whole-pool run needs about 3M distance pairs and thousands of D_max solves on matrices up to 2,037
points. `pip install -e '.[fast]'` adds numba, which compiles the alignment and LDND aggregation into one parallel loop
(240k Oceanic pairs: 241 s → 12 s, results identical to 4e-15; the numpy path stays as the fallback and the test
reference). D_max uses Wolfe's min-norm-point method with an incrementally updated Cholesky factor (Atlantic-Congo,
2,037 doculects, every scale under 15 s), the 50 scales run in separate processes with one BLAS thread each, and
`data/cache/` holds every distance once so nested lineages and `audit.py` reuse them. Oceanic's full analysis went from
568 s to 24 s. One scale (Atlantic-Congo at k = 40, where D_max is essentially the count of distinct doculects) is not
certified optimal. Wolfe's method assumes Z is positive semidefinite; PMI-LDND is not a metric (two lists can be at
distance 0 from each other and at different distances from a third), so acceptance uses the objective gain a violating
point could still give, which is second order, and anything failing it goes to the slower general solver.

## Calibrated similarity

**Why the change.** With Z = exp(−k_cal·d), two unrelated languages (PMI-LDND ≈ 0.97) were still 0.64 similar. Every
score was then squeezed under 1/0.64 ≈ 1.57, five unrelated languages scored 1.41, and a 64-list family such as
Uralic (1.46) outscored them simply by having many partly distinct lists. At equal size random sets still won, but
barely (10 unrelated languages 1.49 against at most 1.40 for 10 lists of a real family), and the numbers had no plain
reading.

**The anchors.** `scripts/regularity_eval.py` takes 200 cross-branch pairs from real families with PMI-LDND ≥ 0.85
and 400 random unrelated pairs (half within 1000 km), and tests each with a permutation test: shuffle which concept
is matched with which, 499 times, and ask whether the real same-meaning similarity beats the shuffles. Detection is
almost entirely a function of the distance: 100% of related pairs at 0.85–0.90, 88% at 0.90–0.92, 77% at 0.92–0.94,
37% at 0.94–0.96 and none above. A logistic fit puts the 50% point at **0.940**. Similarity falls linearly from 1 at
the same-language median (0.228) to 0 there (`isodiv.kernel`). Both anchors are measured, nothing is fitted to the
results, and random sets of n unrelated lists score close to n (`results/adversarial.json`).

**Reading it.** D_max is the effective number of lineages that 40-word lists cannot tie together. For shallow
lineages it measures diversity; for deep ones it also measures how much of the family is beyond the resolution of
the data, since branches farther apart than 0.940 count as separate lineages. A large deep family can therefore
outscore a few unrelated languages: it holds more lineages the data cannot tie together. The fair "as diverse as
random?" comparison is at equal size, where every real family is below an unrelated set
(`results/relative.csv`, the explorer's "Share of the unrelated maximum").

## Relatedness beyond 40 words: a negative result

Could recurrent sound correspondences, the comparative method's evidence, tie together what 40-word distances cannot?
`isodiv.regularity` aligns every word of one list with every word of the other and tests, by permuting concepts, (a)
whether consonant correspondences recur across *other* same-meaning pairs (leave-one-out) and (b) whether particular
correspondences are over-represented in same-meaning pairs (a Kessler-style statistic), next to (c) plain similarity.

- **On ASJP-40** (`scripts/regularity_eval.py`) the correspondence statistics are at chance (AUC 0.48 and 0.54 against
  far unrelated pairs) even for English–Latin; plain similarity reaches AUC 0.87. With about 8 cognates per deep pair,
  a correspondence recurs two or three times at most, which shuffled pairings also produce.
- **On long lists** (`scripts/longlist_eval.py`: NorthEuraLex, ~1,000 concepts, IPA converted to ASJPcode by
  `isodiv.ipa_asjp`) the over-representation statistic becomes significant for 62% of Saami–Samoyed pairs, but also
  for 35% of unrelated neighbours, and those hits are all known contact pairs (Russian–Kildin Saami, Arabic–Persian,
  Danish–South Saami, Turkish–Kalmyk, Erzya–Tatar, ...). Loans carry regular correspondences too. Restricted to
  borrowing-resistant concepts (Leipzig–Jakarta, Swadesh-207: 187 concepts) it finds only 4–21% of Saami–Samoyed
  pairs. Permutation p-values also stop being valid on long lists: sound symbolism and loans give even far unrelated
  pairs "significant" similarity (55% at 187 concepts, 100% at 1,000).
- Calibrated against real unrelated pairs instead of shuffles, longer lists do not beat 40: related vs unrelated AUC
  0.94 at ASJP-40, 0.94 at 187 stable concepts, 0.92 at 1,000.

So automatic correspondence tests on word lists do not separate inheritance from borrowing, and the 40-item list with
PMI similarity is about as far as this approach reaches. (Correction: an earlier version of this README said Saami
and Samoyed are beyond 40 words' reach. Most Saami–Samoyed pairs are at 0.87–0.95, around the detection limit and
mostly detected; the 0.996 quoted came from one pair of lists, Ume Saami and Forest Enets.)

## Relatedness from union shortfall

Two lineages the 40-word lists cannot tie together score the sum of their D_max values together. Every cross pair
within the detection limit takes something off, so `scripts/relatedness.py` reads relatedness from the shortfall

    s = (D_a + D_b − D_union) / min(D_a, D_b)

which is 0 for no detectable link and 1 when the smaller lineage is absorbed into the larger
(`results/relatedness.{json,csv,png}`, from the union D_max of `cross_family.py`; a cross pair with too few shared
concepts for a distance counts as no link, and pairs with no cross distance at all are left out as untestable).

**Calibration.** s is judged against real pairs of different families, not shuffles, because chance resemblance,
sound symbolism and loans all shorten distances and real unrelated pairs carry all three. Two confounds matter:

- *Size.* A deep, diverse lineage offers more lists for a chance near-miss (Spearman ρ = 0.37 between s and the
  larger lineage's D_max among unrelated pairs), so each pair is compared with the 250 unrelated pairs nearest in
  (larger D_max, smaller D_max). Against that null, 3.9% of unrelated far-apart pairs reach p < 0.05.
- *Contact.* Among unrelated neighbours (closest doculects within 1,500 km) 10.5% reach p < 0.05, and 15% within
  500 km. The hits are textbook loan cases: Uralic–Iranian and Uralic–Indo-Aryan, Hmong-Mien–Sinitic,
  Indo-Aryan–Arabic, Yeniseian–Turkic, Koreanic–Mandarin, Saharan–Chadic, and Romance–Algic through Michif
  (French–Cree). So the headline p (`p_size_near`) compares a pair with the 100 size-matched unrelated *neighbours*.
  3.4% of unrelated neighbours reach p < 0.05 against it.

Sibling branches of one family (sharing no doculect; nested pairs are left out) are the positive control: AUC 0.98
against unrelated pairs, and 83% reach `p_size_near` < 0.05. Power falls with depth: siblings whose closest lists
are 0.80–0.85 apart are detected 75% of the time, 0.85–0.95 only 33%. A non-significant proposal is therefore not
evidence against it. The cap per lineage barely matters: AUC 0.98 at 10, 25 and 50 doculects, and s moves little
except for pairs with a very large lineage (Atlantic-Congo), where more lists mean more chance near-misses.

**Proposals** (61 pairs; `p_size_near` floors at 0.01 with 100 neighbours, and about 3 of 61 are expected below
0.05 by chance):

| proposal | pair | s | p (size-matched neighbours) |
|---|---|---|---|
| Altaic | Mongolic – Tungusic | 0.26 | 0.01 |
| | Turkic – Mongolic | 0.10 | 0.02 |
| | Turkic – Tungusic | 0.04 | 0.13 |
| Macro-Altaic | Koreanic, Japonic with each other or Turkic/Mongolic/Tungusic | 0.00–0.03 | 0.31–1 |
| Khoisan | Khoe-Kwadi – Tuu / Tuu – Kxa / Khoe-Kwadi – Kxa | 0.26 / 0.19 / 0.10 | 0.01 / 0.01 / 0.02 |
| Great Andamanese – Ongan | | 0.27 | 0.01 |
| Totozoquean | Mixe-Zoque – Totonacan | 0.22 | 0.01 |
| Kordofanian | Narrow Talodi, Rashad, Katla-Tima, Heibanic with each other | 0.07–0.22 | 0.01–0.07 |
| Niger-Congo | Kru with Atlantic-Congo / Mande / Dogon | 0.24 / 0.20 / 0.14 | 0.03 / 0.03 / 0.02 |
| | Mande, Ijoid, Dogon with Atlantic-Congo | 0.16 / 0.18 / 0.14 | 0.05 / 0.05 / 0.05 |
| Hruso in Sino-Tibetan | | 0.30 | 0.03 |
| Dene–Caucasian | Yeniseian – Sino-Tibetan | 0.12 | 0.03 |
| | Basque – Nakh-Daghestanian / Burushaski – Yeniseian / Basque – Abkhaz-Adyge | 0.16 / 0.03 / 0 | 0.08 / 0.27 / 1 |
| Nilo-Saharan | Songhay – Saharan | 0.12 | 0.04 |
| | Nubian – Saharan / Nubian – Songhay / Kuliak – Nubian | 0.07 / 0.02 / 0.01 | 0.10 / 0.49 / 0.73 |
| Elamo-Dravidian | | 0.16 | 0.07 |
| Austric, Austro-Tai | four pairs | 0.05–0.11 | 0.05–0.22 |
| North Caucasian | Abkhaz-Adyge – Nakh-Daghestanian | 0.08 | 0.13 |
| Dene–Yeniseian | Yeniseian – Athabaskan-Eyak-Tlingit | 0.06 | 0.20 |
| Uralic–Yukaghir | | 0.08 | 0.16 |
| Nostratic-type | Uralic – IE, IE – Afro-Asiatic, Uralic – Turkic, IE – Kartvelian, Uralic – Dravidian | 0.03–0.10 | 0.06–0.43 |
| Uralo-Siberian, Chukotko-Kamchatkan–Amuric | four pairs | 0.00–0.05 | 0.26–0.98 |

**Reading.** A low p says some lists of the two lineages are closer than size-matched unrelated neighbours usually
are. It does not say why. Mongolic–Tungusic is driven by Buriat–Nanai-type pairs sharing *bi/bi* 'I', *ši/ši*
'you', *bid/bu* 'we', *mod/mo* 'tree' and *čuluu/jolo* 'stone', the very words argued over as Altaic cognates or
Mongolic loans; the Khoisan pairs by the Kalahari contact zone (Naro and Gǀui-type Khoe varieties beside Taa);
Andamanese by four list pairs, mostly pronouns. Intense, long contact beats the neighbour null too. What the test
does settle is the negative side within its power: Koreanic and Japonic show no link to each other or to the Altaic
families (s ≤ 0.03), and neither do the Nostratic-type, Uralo-Siberian or Dene–Yeniseian pairs. These are deeper
than the 40-item lists reach, which is a statement about the data, not against the proposals. False positives
happen too: Koreanic–Burushaski (no contact, 4,700 km) has s = 0.15, p = 0.02.

## Family-specific sound correspondences (sensitivity check)

PMI-LDND scores alignments with segment weights learned across all of ASJP, so it discounts correspondences common
worldwide but still charges for ones regular in one group only. `scripts/family_pmi.py` learns weights inside each
group (same procedure, `isodiv.pmi.fit`, on the group's own doculect pairs) and recomputes D_max on the same lists
(`results/family_pmi.{json,csv}`). It assumes the group is one family, so it is for internal diversity only, never
for testing relatedness, and it reuses the kernel anchors calibrated on the global distances.

Trained and scored on the same word pairs the group's weights look better than they are (German dialects −17%,
Quechua −5%), so the comparison that counts is held out: weights learned on a random half of the concepts, D_max on
the other half, mean of 4 halves. Scoring each group with another group's weights is also lower than with the global
weights, but less so than with its own: part of the drop is a generic leniency of weights learned from close
relatives.

| group | own weights vs global, held out |
|---|---|
| Turkic, Romance, Germanic | −13% to −16% |
| Arabic, Japonic | −11% to −14% |
| German dialects (12 lists) | −5% to −12% (noisy) |
| Slavic, Min | −7% to −9% |
| Quechua, Yue, Mandarin, Sinitic | −3% to −5% |
| Wu | about 0 (−7% to +5%) |

Family correspondences take 10–15% off the European families, Turkic and Arabic, whose varieties share roots
reshaped by sound change, and almost nothing off the Chinese groups and Quechua. For Chinese the differences are
mostly different roots (and tone, which ASJPcode drops anyway); for Quechua the roots are shared and the regular
changes were already cheap. Rankings within the chart groups barely change; the Chinese groups move up relative to
Romance and Germanic.

## Supplements: filling gaps in key lists

`config/supplements.tsv` fills concepts missing from an ASJP list from an expert-coded source. It never overrides an
ASJP form, every row carries its source form ID and philological reference, and `run_analysis.py --no-supplements` (or
`ISODIV_NO_SUPPLEMENTS=1`) switches it off. The distance cache fingerprints each list's words, so a supplemented list
is recomputed and nothing else is.

Currently: Hittite, 30 → 36 concepts, from IE-CoR (CC-BY-4.0): dog *kuwan-*, nose *titita-*, tooth *kak(k)a-*, liver
*lissi-*, stone *peru(na)-*, full *šu(w)u-* (CHD, HW², Tischler). Sun is left out: *ištanu-* is the Hattic-derived
Sun-god's name, not an inherited word.

**Luwian** (`config/added_languages.tsv`, `config/supplements_luwian.tsv`, CC BY-SA 4.0 from eDiAna; review in
`config/luwian_review.tsv`). ASJP has no Luwian list, so LUWIAN is an added doculect: 23 concepts, Cuneiform Luwian
where eDiAna has a syllabically attested lemma with a secure meaning (16), Hieroglyphic Luwian for the gaps (7). It
is 0.735 from Hittite. With 23 concepts it shares fewer than 20 with 82 of the 490 Indo-European lists, so the drop
rule leaves it out of Indo-European's D_max; it is in the Compare varieties tool.

## Depth: how closely related, in one unit for pairs and families

D_max counts lineages (breadth). **Depth** answers the other question, how far apart: for two varieties, how much
of their basic vocabulary is still shared; for a family, the same across its deepest split. `scripts/depth.py`:

- **Pairwise.** PMI-LDND is mapped to *retention*, the share of basic vocabulary still cognate, relative to two
  lists of the same language. The map is fitted on expert cognate coding (the Lexibank datasets of the cognate
  check): datasets differ in how strictly they draw cognate sets, which scales their shares by a constant, so each
  gets an offset and one common curve is fitted (`results/depth_calibration.json`). After the offsets the curves of
  IE-CoR, ABVD, Bantu, Dravidian, Semitic, Sino-Tibetan and Uto-Aztecan agree within about ±0.2 log units (Bowern's
  Pama-Nyungan differs in shape and is left out of the fit). Depth is τ = −ln retention: additive along a tree if
  basic-vocabulary replacement is a Poisson process, with no rate and no dates assumed (0.69 = half replaced).
- **Horizon.** The curve flattens past PMI-LDND ≈ 0.85: at the detection limit (0.941) retention is
  about 21% (τ ≈ 1.58), and 40 words cannot resolve anything deeper. Pairs and families there are
  reported as *beyond the horizon*, a lower bound.
- **Family.** Average-linkage clustering on pairwise depth (no classification); the root splits the family in two,
  and the family's depth is the median pairwise depth across that split. Dialect counts do not raise it. Lists close
  to another family (contact check, within the same-language band) are left out, and so are a lineage's own
  ancestors (Latin for Romance, Sanskrit for Indo-Aryan, Old and Middle Chinese for Sinitic, Classical Arabic for
  Arabic, Old Aramaic, Proto-Germanic): an ancestor is not a branch. When the root branch is a single list
  (`lone_branch`) it is often a real lone branch (Tlingit, Kwadi, Svan, Cherokee, Defaka, Cochimí) and sometimes a
  fast-changing or doubtful list (Polabian, Bacama, Yaur, Yue Guilin Xingan); `depth_2` is the deepest split with at
  least two lists on each side.

`scripts/lone_branches.py` checks every one-list root branch three ways (`results/lone_branches.{json,csv}`): whether
Glottolog also makes it a primary branch; a relative-rate test against an outgroup in range (a real early branch is no
farther from the outgroup than its sisters, a fast-changing or noisy list is); and the list itself (another list of
its language that disagrees with it, a sibling group or another family nearer than its own). Of 46 cases (43 since the
three bad lists below were excluded), 15 are real lone branches (Tlingit, Cherokee, Catawba, Svan, Kwadi, Cochimí,
Defaka, Jeju, Gong, Ugaritic…). Two are bad lists, now excluded: `BACAMA` (0.67 from the other Bacama list, which sits
normally among its Biu-Mandara relatives) and `NU` (0.70 from the other Nǁng list, which is 0.23 from Kam-ka-ke). One
is misfiled and now removed from Yue only (see `config/lineage_overrides.tsv`): `YUE_GUILIN_XINGAN` (0.60 from every
Yue list, 0.49 from Mandarin). Others are misfiled or transitional (Xiang Guilin
Jieshou nearer Guilin Mandarin, Hui Fenkou nearer Wu, Mandarin Chenzhou Guiyang nearer Xiang), one is a fast changer
(Ume Saami, relative-rate z = 2.9), and the rest are inconclusive: Polabian is nested in West Slavic but no faster
than its sisters against Baltic (z = 1.6), so its depth points at its sparse, philological record more than at change.

Retention across the deepest split (`results/depth.{json,csv}`): Quechua 96% (Central vs Southern Quechua is as
close as two lists of one language on 40 words), Koreanic 83%, Japonic 74%, Arabic 72%, Slavic 72%,
Germanic 70%, Wu 69%, Mandarin 69% (75% without its one-list root branch), Romance 62%, Min 56%,
Sinitic 43% (Southern Min against the rest), Turkic 38% (Chuvash against the rest), Indo-Aryan 34%, Semitic
28%, Uralic 25%, Indo-European 23% (Anatolian, Tocharian, Greek and Armenian against the rest).
Austronesian, Atlantic-Congo, Afro-Asiatic, Pama-Nyungan, Nuclear Trans New Guinea and Otomanguean are beyond the
horizon.

**Constructed sets** (`scripts/depth_random.py`, `results/depth_random.json`; the same draws as the adversarial
check). Random sets of unrelated lects, one per family, are beyond the horizon at their deepest split in every draw,
at a median distance of 0.99 across it for 5 lects and 0.99 for 80: depth does not grow with the number of
languages, where D_max does (about n). For the proposals the deepest split is between member families, and its
median distance is compared with random unions of as many unrelated families: Altaic (0.940, Turkic against
Mongolic and Tungusic), "Khoisan" (0.944), Elamo-Dravidian (0.923, one Elamite list) and Niger-Congo
(0.971, the Kordofanian groups against the rest) are closer across their split than any random union
(about 0.99), while Macro-Altaic (0.995, Japonic against the rest) and Paleo-Siberian (0.995) are not.
The baselines are not matched for geography, so contact counts here too (see the relatedness section, which uses
size-matched neighbours), and all of these are at or beyond the horizon in retention.

Caveats: depth measures change, not time, so a fast-changing branch looks old (Polabian, under German contact, is
Slavic's one-list root branch). The calibration is pooled across families whose cognate curves differ by up to
±0.2 log units, so small depth differences between families are not significant. Depth rests on a family's most
divergent members, which makes it more sensitive to a single doubtful list than D_max is.

## Widest pair (robust, tree-free)

D_max is an effective number of distinct lineages; the widest pair is on the distance scale and answers "how far apart
are the two most different members?". The raw maximum is fragile, since one bad list sets it. `scripts/widest.py`
makes it robust without using any subgrouping: each list's neighbourhood is the list plus up to 4 nearest lists within
the same-language noise band (0.474), and a pair's distance is the median over the two neighbourhoods. A list with no
neighbours (a lone witness, such as Hittite or Akkadian) still counts, unless the contact check flags it: its nearest
list in another family is closer than its nearest list in its own. Flagged lists are reported in
`results/contact.{csv,json}` and the audit, never dropped silently (currently 16 lists, among them MOGOGODO; AYNU and
BALDEMU were flagged before they were excluded, BALDEMU at 0.210 from KUNDU in Atlantic-Congo, which suggests a
mislabelled list). Results: `results/widest.{csv,json}`, and "Show: widest pair" in the explorer. Examples: Japonic
0.553, Turkic 0.809 (Chuvash–Tofa), Mayan 0.857, Uralic 0.996. Old, large families (Indo-European 1.000) reach the
unrelated level of about 0.97, so the widest pair saturates where D_max does not. The widest pair is the most
divergent pair, not the earliest split: ancient lists such as Hittite are not in it, because they had less time to
diverge.

## Cognate check: does PMI-LDND agree with expert cognate coding?

`scripts/cognate_check.py` compares the pipeline's distances with distances from expert cognate judgements in ten
Lexibank datasets, matched to ASJP by Glottocode. Cognate distance is the share of comparable concepts with no shared
cognate set. This check is independent of our method. Only derived statistics are written to `results/cognate_check.*`;
the datasets live in the git-ignored `data/lexibank/` (Grollemund and Kitchen are CC-BY-NC, the others CC-BY-4.0).

**Language pairs** (Spearman ρ with cognate distance; up to 20,000 pairs per dataset):

| dataset | languages | PMI-LDND | LDND | PMI-LDND, ASJP-40 concepts | LDND, ASJP-40 concepts |
|---|---|---|---|---|---|
| IE-CoR (Indo-European) | 98 | 0.556 | 0.550 | 0.693 | 0.654 |
| ABVD (Austronesian) | 826 | 0.531 | 0.495 | 0.650 | 0.598 |
| Bowern (Pama-Nyungan) | 108 | 0.447 | 0.440 | 0.468 | 0.450 |
| Sagart (Sino-Tibetan) | 33 | 0.657 | 0.605 | 0.566 | 0.482 |
| Kitchen (Semitic) | 20 | 0.822 | 0.773 | 0.773 | 0.723 |
| Savelyev (Turkic) | 25 | 0.537 | 0.463 | 0.467 | 0.423 |
| DravLex (Dravidian) | 19 | 0.783 | 0.767 | 0.806 | 0.795 |
| Grollemund (Bantu) | 312 | 0.662 | 0.648 | 0.676 | 0.650 |
| Uto-Aztecan | 32 | 0.700 | 0.665 | 0.788 | 0.745 |
| Lee (Japonic) | 3 | n/a | n/a | n/a | n/a |

PMI-LDND agrees with expert cognates better than plain LDND in every dataset. Agreement is weakest for Pama-Nyungan,
where borrowing is heavy and the cognate signal shallow.

**Lineages.** For 34 lineages with at least 5 matched languages, `spread` is computed from cognate distance and from
PMI-LDND on the same languages. The two rank the lineages almost identically: Spearman 0.975 pooled; within datasets,
IE-CoR 0.929 (7 lineages), ABVD 0.945 (11) and Kitchen 1.000 (6). Under cognates the order is Slavic < Romance < Germanic <
Balto-Slavic < Iranian < Indo-Aryan < Indo-European; PMI-LDND gives the same order except that it swaps two adjacent
pairs (Germanic < Romance, Indo-Aryan < Iranian). PMI-LDND compresses the scale relative to cognates, which
leaves rankings unaffected. At the level that matters for D_max, ASJP-40 with PMI-LDND reproduces the diversity ranking
that expert cognate coding gives.

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

Each lineage is scored on all its ASJP doculects, minus the exclusions and minus lists that leave pairs undefined
(fewer than 20 shared concepts). The drop rule removes one list at a time: the one with the most undefined pairs, then
the shorter list, and only then does a lineage's protected list (the one defining a child) count; a lineage always
keeps its longest list. Ordering by length before protection is what keeps Hittite (36 concepts) in Indo-European:
protecting the child-defining lists first had forced it out. `--cap N` restores a per-lineage cap. 128 lineages: 70 top-level (the original 16, 32 more ASJP families, six single- or few-list isolates incl. Sumerian,
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

## Website and distance tool

`docs/` is a static site (landing page, [explorer](docs/explorer.html), explainer, data downloads) built by
`python scripts/build_site.py` after the pipeline and `build_explorer.py`. Serve it with GitHub Pages: Settings → Pages →
Deploy from a branch → `main`, folder `/docs` (the site lives at https://iwsfutcmd.github.io/lingustic-diversity/). It
replaces republishing the artifacts.

The explorer's **Compare varieties** tab (`explorer.html#compare`) gives the distance between any two ASJP word lists.
Pick each variety by ISO 639-1 or -3 code, Glottocode, doculect ID or part of a name. It is computed in the browser from
`docs/data/wordlists.json` (all 11,536 lists, the PMI model and the code tables), using the pipeline's own distance:
`explorer/distance.js` is a port of `pmi.py` and `distance.py`, and `tests/test_site_js.py` checks it against the Python
values on real pairs (agreement to 2e-6). A code can match several lists (`ja` finds 7 Japanese lists; `zh`, a
macrolanguage, finds 534 Chinese lists), so every pair is scored and the tab reports the median, the range, a gauge against
the same-language noise floor and the unrelated-lineage range, a pair table, and the word-by-word comparison of any pair.
Both PMI-LDND and plain LDND are available, and `?a=ja&b=ryu&d=pmi#compare` links straight to a comparison.

`python scripts/distance.py jpn eng` does the same on the command line (`--pairs` lists every pair, `--distance ldnd`).
Code resolution is `src/isodiv/codes.py`: ISO 639-1 maps to 639-3 through `config/iso639_1.tsv`, and a macrolanguage that
ASJP does not use itself (`zho`, `ara`, `msa`) stands for its member codes (`config/iso639_macro.tsv`, from the ISO 639-3
registry). Examples: Japanese ↔ English 0.974 (unrelated), Japanese ↔ Central Okinawan 0.485 (32 list pairs; at the
edge of the same-language band), Japanese against itself 0.293.


**Chart maker** (explorer tab): grouped bar charts of D_max or depth for any lineages on the page, in up to three
labelled groups, drawn at 1600 px for sharing (Download PNG, Copy image, or save the rendered image). Ad hoc groups that
are not lineages (German dialects, English, Chinese without Min) are defined in `config/custom/custom_groups.tsv` and
scored like lineages by `scripts/custom_groups.py` (`results/custom_groups.json`); they appear in the explorer as
"Custom groups".
## Explorer and cross-family check

- `scripts/cross_family.py` compares lineages with each other: mean distance within and between the 128 lineages
  (see Lineages), on 25 doculects per lineage, and D_max of two lineages together vs. apart. On the calibrated scale
  two unrelated lineages together score the sum of their D_max values (exactly, when every cross pair is beyond the
  detection limit), so a union that falls short of the sum shows cross-lineage pairs within reach of the data
  (calibrated in "Relatedness from union shortfall"; `--cap N --tag capN` writes a variant run).
  Japonic ↔ Kartvelian: mean PMI-LDND 0.98 vs. 0.43 / 0.60 within. Across 7,544 unrelated lineage pairs the mean is
  0.971 (LDND 0.984 over 2,483 pairs, from the earlier 72-lineage run). The closest are Tuu ↔ Kxa 0.846, Mongolic ↔ Tungusic 0.868, Khoe-Kwadi ↔ Tuu 0.869 and
  Koreanic ↔ Burushaski 0.872 (LDND 0.911–0.929, same pairs). Kalahari Basin and Altaic-area contact fit the others;
  Koreanic ↔ Burushaski (3 vs 10 doculects, no contact) shows PMI can also pull chance resemblances closer.
- `scripts/build_explorer.py` bundles `results/*.json` into `results/explorer.html`, a self-contained
  page (diversity ranking, scale profiles, between-lineage heatmap).

## Excluding bad data

`config/exclusions.tsv` (`id<TAB>reason`) lists doculects left out of every script. `scripts/audit.py` ranks
doculects by how much removing them lowers their lineage's D_max (also flags thin coverage) and writes
`results/audit.csv`; it never excludes anything itself. Review the audit, edit the file, rerun. Currently excluded
(reasons and evidence in the file):

- mixed languages: `AYNU` (Turkic–Persian), `MICHIF` (French–Cree), `WUTUN`, `DAO_SICHUAN_YAJIANG` (Daohua) and
  `TANGWANG_GANSU_TANGWANG` (Sinitic with Tibetan or Mongolic), `ANGLOROMANI`, `INNER_MBUGU_BUMBULI` (Ma'a);
- data errors found by `misplaced.py` and the audit's contact check: `ATAYAL_TAIWAN_YILAN` (a Polynesian list),
  `TAMASHEQ_2` (largely Hausa), `BALDEMU` (a Bantu list); and from `lone_branches.py`, `BACAMA` and `NU` (each far
  from the other list of its own language, which sits with its relatives);
- creoles and pidgins (73 lists): English-, French-, Iberian-, Dutch-, Arabic-, Malay- and other-lexified creoles,
  pidgins and creolized vehicular languages (Sango, Kituba, Nagamese). Riverain Sango (the Ngbandi variety), Normal
  Mbugu and Afrikaans are kept;
- a duplicate: `XIANG_YONGZHOU_SHOUYAN` is word-for-word `PINGHUA_YONGZHOU_SHOUYAN`, kept under Pinghua, the closer
  group without it.
The leave-one-out flag is half a lineage (D_max falls by at least 0.5 without the doculect). 46 doculects are flagged
on the current run (by leave-one-out, distance or thin coverage; 13 in Dogon, 5 in Pama-Nyungan). Many are lone
witnesses of a deep branch rather than bad data: Cherokee, for instance, is 0.92 of a lineage on its own, because
its nearest Northern Iroquoian list (Mohawk, 0.895) is almost at the detection limit at 40 words.
A doculect with no weight in the D_max optimum cannot lower it, so only weighted candidates get a leave-one-out solve.

`config/lineage_overrides.tsv` (`id<TAB>remove|add<TAB>lineage<TAB>reason`) changes membership in one lineage
only, for lists Glottolog files in a group their vocabulary does not fit. A list that is surely in a family but cannot be placed
inside it is removed from the subgroup only, never from the family. Currently: `GAN_SHAOWU` (Shaowu, a Min–Gan
transitional variety that many dialectologists classify as Min) out of Gan, and so out of "Chinese without Min";
`YUE_GUILIN_XINGAN` and `YUE_GUILIN_LIANGJIANG` (Guilin-area lists nearer Pinghua or Mandarin than any Yue list) out
of Yue; `MANDARIN_GUILIN_LONGSHENG` (likely Pinghua) out of Mandarin; `MIN_FUZHOU_QINJIANG` (Mandarin-like data under a
Min label) out of Min. All stay in Sinitic and Sino-Tibetan.

`scripts/misplaced.py` looks one level down: a doculect is flagged when a *sibling* group of the same family (Pinghua
for a Mandarin list, Chadic for a Berber one) is closer than its own, by the mean distance to the 3 nearest lists in
each (`results/misplaced.{csv,json}`; its own group is the smallest lineage holding it). 50 of the 3,210 doculects
that had a sibling group were flagged before the exclusions (43 of 3,188 now). At the top were clear data errors:
`ATAYAL_TAIWAN_YILAN` is a Polynesian list (*au* 'I', *ika* 'fish', *mata* 'eye'; 0.135 from Niuafo'ou, 0.75 from the
rest of Formosan), and `TAMASHEQ_2` is largely Hausa (*kifi* 'fish', *kunne* 'ear', *hanci* 'nose'; 0.42 from Hausa).
`MANDARIN_GUILIN_LONGSHENG` (nearest lists all Pinghua) is a likely mislabel. Most of the rest are real transitional
varieties: Min Nanping (a Mandarin speech island), Xuancheng Wu and Jin next to Mandarin, the Hakka–Gan border, the
Yongzhou *tuhua* between Xiang and Pinghua, and Arabic dialects closer to another Arabic group than their own.

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
pip install -e '.[dev,fast]'           # fast: numba kernels (optional; ~20x faster distances)
pytest
python scripts/pmi_train.py          # → results/pmi_model_all.tsv  (~1 min)
python scripts/pipeline.py --commit  # every step below, in order; skips steps whose inputs are unchanged
```

`scripts/pipeline.py` hashes each step's inputs (code, config, word lists, upstream results; large data files by size
and date) and skips a step when nothing it reads has changed since its last successful run
(`results/.pipeline_state.json`; `--force STEP` or `--force all` reruns). With `--commit` it commits and pushes
`results/` and `docs/` after each step. One distance cache over every ASJP list (`data/cache/`, ~530 MB, keyed by each
list's words) is shared by all steps, so a distance is computed once across steps and runs. The steps:

```bash
python scripts/regularity_eval.py    # → results/regularity_eval.*: permutation tests, detection limit (similarity anchor)
python scripts/run_analysis.py       # → results/summary.{csv,json}, samples.json, profiles.json, ldnd_*.tsv (n ≤ 200)
python scripts/bootstrap.py          # → results/bootstrap.json: concept resampling
PYTHONPATH=scripts python scripts/cross_family.py   # → results/cross_family.{csv,json,png}
PYTHONPATH=scripts python scripts/widest.py         # → results/widest.{csv,json}, contact.{csv,json}
PYTHONPATH=scripts python scripts/audit.py          # → results/audit.{csv,json}
PYTHONPATH=scripts python scripts/adversarial.py    # → results/adversarial.{csv,json}
PYTHONPATH=scripts python scripts/single_lists.py   # → results/single_lists.json
PYTHONPATH=scripts python scripts/sensitivity.py    # → results/sensitivity.{csv,json}
PYTHONPATH=scripts python scripts/rarefaction.py    # → results/rarefaction.{csv,json}
PYTHONPATH=scripts python scripts/cognate_check.py  # → results/cognate_check.{csv,json} (needs data/lexibank/)
python scripts/build_explorer.py     # → results/explorer.html, results/relative.csv
python scripts/build_site.py         # → docs/
```

Not in the pipeline: `scripts/longlist_eval.py` (needs data/lexibank/northeuralex and uralex),
`scripts/pmi_eval.py`, `scripts/distance.py jpn eng` (distance between two varieties by code).

`--tokens segments` switches from ASJPcode to NFC-normalized CLDF IPA segments. Words are always
tuples of opaque segment tokens, so any script or segmentation works.

## Next steps

In order:

1. ~~Relatedness from union shortfall~~: done, see "Relatedness from union shortfall". Still open from it:
   recalibrate `loans.py` in PMI-LDND units and use it to separate loan-driven from inherited links in the hits
   (Mongolic–Tungusic, Khoisan), and check whether PMI's pull on chance resemblances (Koreanic ↔ Burushaski, s =
   0.15) needs a correction.
2. **Pipeline dependencies.** Track which `src/isodiv` modules each step imports, so a change to one module reruns
   only the steps that use it.
3. **Doculect-level bootstrap.** The intervals resample concepts only; resample doculects within each lineage too.
4. **Holman et al. 2011.** Rank correlation of D_max with their dated family ages, and the cases where they disagree.
5. **Synthetic trees.** Evolve 40-word lists on known trees with borrowing, uneven sampling and duplicated dialects,
   and check that D_max recovers the number of lineages and ignores dialect counts.
6. **LDND on the 128 whole-pool lineages**, which has not been run.

Any time: decide whether Luwian (23 concepts) enters Indo-European's D_max, and fill under-sampled lineages (Basque,
Haida) from expert sources, as for Hittite. Larger and optional: expert cognate sets (Lexibank) as an alternative
distance where they exist, since they encode correspondences and loan judgements that automatic tests cannot recover
(see the negative result above). Speed: profile `run_analysis` and `cognate_check`, now the two longest steps.

## References

- Leinster, T. & C. A. Cobbold. 2012. Measuring diversity: the importance of species similarity. *Ecology* 93: 477–489.
- Leinster, T. & M. W. Meckes. 2016. Maximizing diversity in biology and beyond. *Entropy* 18(3): 88.
- Wichmann, S., E. W. Holman, D. Bakker & C. H. Brown. 2010. Evaluating linguistic distance measures. *Physica A* 389: 3632–3639.
- Jäger, G. 2013. Phylogenetic inference from word lists using weighted alignment with empirically determined weights. *Language Dynamics and Change* 3(2): 245–291.
- Jäger, G. 2018. Global-scale phylogenetic linguistic inference from lexical resources. *Scientific Data* 5: 180189.
- Wichmann, S., E. W. Holman, C. H. Brown, M. S. Dryer & Q. Ran (eds.). 2025. *The ASJP Database* v21. CLDF: github.com/lexibank/asjp (CC-BY-4.0).
