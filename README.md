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
   **Every lineage is scored on all its ASJP doculects** (up to 2,042 for Atlantic-Congo), and a lineage that
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
| | Yeniseian (9) | 1.91 | 1.79–2.02 | 1.264 |
| | Mongolic (29) | 2.04 | 1.97–2.21 | 1.294 |
| | Kartvelian (6) | 2.12 | 2.03–2.23 | 1.271 |
| | Romance (64) | 2.14 | 2.02–2.29 | 1.307 |
| | Germanic (61) | 2.16 | 2.10–2.28 | 1.302 |
| | Turkic (71) | 2.44 | 2.30–2.60 | 1.324 |
| deep | Semitic (110) | 5.26 | 4.96–5.51 | 1.436 |
| | Uralic (64) | 6.57 | 5.98–7.34 | 1.456 |
| | Indo-European (527, with Hittite) | 14.98 | 14.34–15.92 | 1.510 |

The largest lineages: Atlantic-Congo 72.7 (2,042 doculects), Nuclear Trans New Guinea 47.9, Austronesian 44.4,
Afro-Asiatic 35.1, Pama-Nyungan 22.7, Sino-Tibetan 21.7, Chadic 19.6, Otomanguean 18.2, Indo-European 15.0.

The ordering is almost the same as at k_cal; what changes is the spacing. The isolate and genus bands now read
directly: Germanic is about two lineages the 40-word lists cannot tie together (Proto-Germanic carries the largest
weight), Turkic about two and a half (Chuvash carries the largest), and Indo-European about fifteen. A few
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
(e.g. Yonaguni). That is under-sampling of genuinely different varieties, not a counting artifact. (The kernel column
of this check is added to `summary.json` → `japonic_sampling` from the next full run.)

**Reading:**
- The bands do not overlap. Japonic measures as a genus-sized family, level with Slavic,
  so including Ryukyuan is not a counting artifact.
- Koreanic (with Jeju) stays in the isolate band.
- Yeniseian scores as genus-sized, so Ket is a "last survivor", not an isolate.
- Turkic sits at the top of the genus band, far below Uralic and Indo-European.

**Sinitic groups.** Every major group scores at genus level: Min 2.56, Mandarin 2.40, Pinghua 2.30, Wu 2.12,
Gan 2.08, Yue 2.02, Xiang 2.01, Hakka 1.97, Hui 1.79, Jin 1.66 (Germanic 2.16, Romance 2.14, Slavic 1.61), and
Sinitic as a whole 4.30 (491 doculects). Each group's dialect lists are about as lexically diverse as a whole Western
European genus. This is not a short-word or transcription artifact: Sinitic words average 3.67 segments (Germanic
3.80), and the 33 same-language pairs inside Sinitic have median PMI-LDND 0.19, below the ASJP-wide 0.228. Wutun,
heavily restructured by Tibetan and Mongolic contact, is in Mandarin's pool; without it Mandarin is 2.28 rather than
2.40 (see below).

**Ancient lists and contact** (`scripts/sensitivity.py`, whole pools; the main results keep every doculect). Leaving
out the doculects ASJP flags `long_extinct` (mostly lineages with a written past) lowers Germanic 2.16 → 1.98 (8 lists
incl. Proto-Germanic and Gothic), Sinitic 4.30 → 4.05 (Old and Middle Chinese), Semitic 5.26 → 4.95, Northwest Semitic
2.37 → 2.18, North Ethiopic 1.41 → 1.36 and Canaanite 1.44 → 1.00 (2 of its 4 lists). Indo-European goes 14.97 → 14.54
with 16 long-extinct lists removed (Hittite, Tocharian and others are each nearly a lineage of their own at 40 words);
Afro-Asiatic 35.13 → 35.05. Sumerian, Elamite, Etruscan, East Semitic and Ancient South Arabian consist only of
long-extinct lists and have nothing left. Wutun, the contact language, is the other scenario: Mandarin goes 2.40 → 2.28
without it, Sinitic 4.30 → 4.30 and Sino-Tibetan does not move, so it is kept. Both versions are in
`results/sensitivity.csv` and in the explorer's detail panel (a "Sensitivity" block).

**Change, not time: Austronesian** (`scripts/rarefaction.py`: 100 draws of m distinct languages, one random doculect
each, from each lineage's whole pool). Austronesian (44.4, 1,467 doculects) outscores Indo-European (15.0, 527) on
the whole pools, though it is younger, but most of that is size. At equal numbers of languages the two are close and
separate slowly: P(Austronesian > Indo-European) is 0.61 at m = 3, 0.62 at 10, 0.75 at 20 and 0.92 at 40 (D_max 9.7
vs 7.4 at 40). Composition and rate matter: Oceanic (about 3,500 years) is above Indo-European at every size (P 0.63
at 3, 0.83 at 10, 1.00 at 40; 11.7 vs 7.4 at 40) and above Malayo-Polynesian as a whole (P 0.61 to 0.90). Formosan,
where Austronesian's deepest splits are, is *lowest* of the Austronesian groups and below Indo-European (m = 10: 3.7,
against Malayo-Polynesian 4.7, Oceanic 5.3 and Indo-European 4.4; P(Formosan > Oceanic) 0.02): its languages kept more
of the Proto-Austronesian basic vocabulary. The nest is monotone on the whole pools: Polynesian 1.67 ≤ Oceanic 34.2 ≤
Eastern Malayo-Polynesian 36.7 ≤ Malayo-Polynesian 41.6 ≤ Austronesian 44.4.

Austronesian is not the most diverse family at equal size either. At 40 languages Atlantic-Congo (16.3) and
Afro-Asiatic (15.4) are above Pama-Nyungan (12.3), Oceanic (11.7), Austronesian (9.7), Sino-Tibetan (9.6) and
Indo-European (7.4); Pama-Nyungan is above Indo-European at every size (P 0.74 at 3, 1.00 at 40). D_max measures how
much basic vocabulary has changed, so a young, fast lineage can outscore an old, slow one. It is a measure of lexical
diversity, not of age. The table is in `results/rarefaction.csv`.

**Semitic.** Semitic 5.26; West Semitic 5.09; South Semitic 3.51 (higher than Central Semitic 2.97, with Ethiosemitic
2.68 the deepest part); Northwest Semitic 2.37, Arabic 1.90, Aramaic 1.80, Modern South Arabian 1.62. East Semitic
(three Akkadian lists, 1.11) and Canaanite (4 lists, 1.44) are too thin to compare. Ancient South Arabian's two lists
have 19 concepts each, below the 20 shared concepts LDND needs, so it has no distances to any other list (D_max 1, NaN
in between-lineage tables).

**Ancient and single-list isolates** (`scripts/single_lists.py`). A single list has D_max = 1 by definition, so only
its distance to everything else is informative. For a random list from a known family, the nearest unrelated
lineage sits at mean PMI-LDND 0.908 (5–95%: 0.857–0.935). Sumerian (nearest Etruscan 0.880), Etruscan (Nivkh 0.863;
24 concepts), Hruso, Kusunda and Nihali are within chance. Elamite is closest to Basque (0.851, bottom 5%, as expected
for one of six isolates by chance) and only 14th of 69 to Dravidian (0.925): no Elamo-Dravidian signal.

## Constructed sets (adversarial check)

`scripts/adversarial.py` scores sets that are *not* families, on the same scale. Lects from different families are
almost all beyond the detection limit of each other, so n of them should score about n:

| set | D_max | 5–95% | at k_cal |
|---|---|---|---|
| 5 random unrelated lects (one per family, 30 draws) | 4.92 | 4.63–5.00 | 1.406 |
| 10 random unrelated lects | 9.74 | 9.44–9.91 | 1.484 |
| 25 random unrelated lects | 22.8 | 21.2–23.7 | 1.535 |
| 40 random unrelated lects | 34.8 | 33.3–36.2 | 1.550 |
| 80 random unrelated lects | 63.2 | 60.9–65.9 | 1.563 |
| Atlantic-Congo (2,042 doculects) | 72.7 | | 1.563 |
| Nuclear Trans New Guinea (307) | 47.9 | | 1.557 |
| Austronesian (1,467) | 44.4 | | 1.549 |
| Indo-European (527) | 15.0 | | 1.510 |
| Uralic (64) | 6.57 | | 1.456 |

The random sets fall a little short of n as they grow (a few pairs of families are within the detection limit of
each other, by chance or contact). Every real family scores far below an unrelated set of the same size: 40 random
Indo-European languages score 7.4 (rarefaction above), 40 unrelated lects 34.8. A whole family can outscore a small
random set only by holding more lineages that 40 words cannot tie together, which is what D_max now counts.

Proposed or rejected groupings, as unions of their member lineages (capped at 80 doculects), against random unions of
as many unrelated families, of any depth and of *shallow* (genus-level) families only:

| grouping | D_max | any-depth unions | shallow unions |
|---|---|---|---|
| Altaic (Turkic, Mongolic, Tungusic) | 5.73 | 8.0 (4.9–18.9) | 6.56 (5.79–7.77) |
| "Khoisan" (Khoe-Kwadi, Tuu, Kxa) | 6.09 | same | same |
| Ural-Altaic | 9.96 | 11.9 (8.2–18.2) | 8.99 (8.07–9.94) |
| Macro-Altaic (+ Koreanic, Japonic) | 8.19 | 13.4 (8.5–20.3) | 10.6 (9.7–12.2) |
| "Paleo-Siberian" (Chukotko-Kamchatkan, Yukaghir, Nivkh, Yeniseian, Ainu) | 7.61 | as Macro-Altaic | as Macro-Altaic |
| Dene-Yeniseian | 7.03 | 5.0 (2.2–12.2) | 4.30 (4.01–5.33) |
| Elamo-Dravidian | 4.60 | as Dene-Yeniseian | as Dene-Yeniseian |
| Niger-Congo (9 branches) | 17.9 | 19.3 (17.2–22.6) | 17.9 (16.6–19.4) |

The constructed sets are unions of their members' samples capped at 80 doculects, as before; the random baselines are
drawn the same way. The any-depth baseline is wide because it mixes isolates with families like Atlantic-Congo.

Altaic (5.73) now falls just below the 5% edge of random unions of three shallow families (5.79), and "Khoisan" (6.09)
sits low in the same range. On this scale a union of unrelated lineages scores about the sum of its members, so
falling short means that some cross-family pairs are within the detection limit. Basic-vocabulary *distance* says the
same: Mongolic ↔ Tungusic (0.868) is the closest unrelated pair in the data and Turkic ↔ Mongolic (0.925) is in the
closest 1%. But the "Khoisan" pairs, an areal (Kalahari Basin) grouping, are as close (Tuu ↔ Kxa 0.857, Khoe-Kwadi ↔
Tuu 0.876), so contact alone produces this signal; neither measure separates it from descent. Macro-Altaic and
"Paleo-Siberian" fall below the shallow baseline mainly because they contain isolate-like members (Koreanic 1.17,
Nivkh 1.21, Ainu 1.29), which the shallow pool does not; the baseline does not fit groupings with deep (Uralic,
Athabaskan-Eyak-Tlingit) or isolate members. Koreanic ↔ Japonic (1.012) is farther apart than 99% of unrelated pairs,
and Dene-Yeniseian (0.956) is unremarkable, which is expected for a proposal argued from morphology.

**Niger-Congo** (Glottolog splits it into Atlantic-Congo, Mande, Kru, Ijoid, Dogon and the Kordofanian groups): the
nine-branch union scores 17.9, at the median of random unions of nine shallow families and near the low end of
any-depth ones (19.3, 17.2–22.6). Distances to Atlantic-Congo: Mande 0.949, Kru 0.950, Ijoid 0.947, Dogon 0.947, each
closer than about 90% of unrelated lineage pairs, a weak signal that West African contact could also produce; the
Kordofanian groups (0.964–0.977) are at chance. Elamo-Dravidian (4.60) is at its baseline (5.0).

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
  equal-similarity curve n / (1 + (n − 1)·z̄)). It is in `results/relative.csv` and the explorer: Atlantic-Congo and
  Nuclear Trans New Guinea 0.35, Afro-Asiatic 0.22, Austronesian 0.22, Pama-Nyungan 0.20, Uralic 0.11, Indo-European
  0.09, Germanic, Turkic and Japonic 0.02, Basque 0. Its n counts doculects, so a heavily sampled lineage (Sinitic, 491
  doculects, 0.02) is compared against a very large unrelated set; read it next to rarefaction, not instead of it.
- **The whole D_max(k) curve** of the older exp(−k·d) similarity (explorer, "Scale profiles", also as a share of the
  unrelated maximum).

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

## Calibrated similarity

**Why the change.** With Z = exp(−k_cal·d), two unrelated languages (PMI-LDND ≈ 0.97) were still 0.64 similar. Every
score was then squeezed under 1/0.64 ≈ 1.57, five unrelated languages scored 1.41, and a 64-list family such as
Uralic (1.46) outscored them simply by having many partly distinct lists. At equal size random sets still won, but
barely (10 unrelated languages 1.48 against at most 1.40 for 10 lists of a real family), and the numbers had no plain
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
is 0.735 from Hittite. With 23 concepts it shares fewer than 20 with 87 of the 527 Indo-European lists, so the drop
rule leaves it out of Indo-European's D_max; it is in the Compare varieties tool.

## Widest pair (robust, tree-free)

D_max is an effective number of distinct lineages; the widest pair is on the distance scale and answers "how far apart
are the two most different members?". The raw maximum is fragile, since one bad list sets it. `scripts/widest.py` makes
it robust without using any subgrouping: each list's neighbourhood is the list plus up to 4 nearest lists within the
same-language noise band (0.474), and a pair's distance is the median over the two neighbourhoods. A list with no
neighbours (a lone witness, such as Hittite or Akkadian) still counts, unless the contact check flags it: its nearest
list in another family is closer than its nearest list in its own. Flagged lists are reported in
`results/contact.{csv,json}` and the audit, never dropped silently (currently AYNU, MOGOGODO, BALDEMU; BALDEMU is 0.210
from KUNDU in Atlantic-Congo, which suggests a mislabelled list). Results: `results/widest.{csv,json}`, and "Show:
widest pair" in the explorer. Examples: Japonic 0.553, Turkic 0.809 (Chuvash–Tofa), Mayan 0.857, Uralic 0.996.
Old, large families (Indo-European 1.000) reach the unrelated level of about 0.97, so the widest pair saturates where
D_max does not. The widest pair is the most divergent pair, not the earliest split: ancient lists such as Hittite are
not in it, because they had less time to diverge.

## Cognate check: does PMI-LDND agree with expert cognate coding?

`scripts/cognate_check.py` compares the pipeline's distances with distances from expert cognate judgements in ten
Lexibank datasets, matched to ASJP by Glottocode. Cognate distance is the share of comparable concepts with no shared
cognate set. This check is independent of our method. Only derived statistics are written to `results/cognate_check.*`;
the datasets live in the git-ignored `data/lexibank/` (Grollemund and Kitchen are CC-BY-NC, the others CC-BY-4.0).

**Language pairs** (Spearman ρ with cognate distance; up to 20,000 pairs per dataset):

| dataset | languages | PMI-LDND | LDND | PMI-LDND, ASJP-40 concepts | LDND, ASJP-40 concepts |
|---|---|---|---|---|---|
| IE-CoR (Indo-European) | 97 | 0.555 | 0.548 | 0.693 | 0.654 |
| ABVD (Austronesian) | 827 | 0.528 | 0.492 | 0.650 | 0.601 |
| Bowern (Pama-Nyungan) | 108 | 0.447 | 0.440 | 0.468 | 0.450 |
| Sagart (Sino-Tibetan) | 33 | 0.657 | 0.605 | 0.566 | 0.482 |
| Kitchen (Semitic) | 20 | 0.822 | 0.773 | 0.773 | 0.723 |
| Savelyev (Turkic) | 25 | 0.537 | 0.463 | 0.467 | 0.423 |
| DravLex (Dravidian) | 19 | 0.783 | 0.767 | 0.806 | 0.795 |
| Grollemund (Bantu) | 312 | 0.661 | 0.647 | 0.675 | 0.650 |
| Uto-Aztecan | 32 | 0.700 | 0.665 | 0.788 | 0.745 |
| Lee (Japonic) | 3 | n/a | n/a | n/a | n/a |

PMI-LDND agrees with expert cognates better than plain LDND in every dataset. Agreement is weakest for Pama-Nyungan,
where borrowing is heavy and the cognate signal shallow.

**Lineages.** For 34 lineages with at least 5 matched languages, `spread` is computed from cognate distance and from
PMI-LDND on the same languages. The two rank the lineages almost identically: Spearman 0.973 pooled; within datasets,
IE-CoR 0.929 (7 lineages), ABVD 0.909 (11) and Kitchen 1.000 (6). The order Slavic < Romance < Germanic < Balto-Slavic <
Iranian < Indo-Aryan < Indo-European is the same under both. PMI-LDND compresses the scale relative to cognates, which
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
`docs/data/wordlists.json` (all 11,535 lists, the PMI model and the code tables), using the pipeline's own distance:
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

## Explorer and cross-family check

- `scripts/cross_family.py` compares lineages with each other: mean distance within and between the 128 lineages
  (see Lineages), on 25 doculects per lineage, and D_max of two lineages together vs. apart. On the calibrated scale
  two unrelated lineages together score the sum of their D_max values (exactly, when every cross pair is beyond the
  detection limit), so a union that falls short of the sum shows cross-lineage pairs within reach of the data.
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
The leave-one-out flag is half a lineage (D_max falls by at least 0.5 without the doculect). 51 doculects are flagged
on the current run (by leave-one-out, distance or thin coverage; 13 in Dogon, 5 in Pama-Nyungan). Many are lone
witnesses of a deep branch rather than bad data: Cherokee, for instance, is 0.92 of a lineage on its own, because
its nearest Northern Iroquoian list (Mohawk, 0.895) is almost at the detection limit at 40 words.
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

- Recalibrate `loans.py` thresholds in PMI-LDND units, and check whether PMI's pull on chance resemblances
  (Koreanic ↔ Burushaski) needs a correction.
- Expert cognate sets (Lexibank) as an alternative distance where they exist: they encode correspondences and loan
  judgements that automatic tests cannot recover (see the negative result above).
- For a paper: compare with Holman et al. 2011 family ages, add a doculect-level bootstrap and a simulation on
  synthetic trees. LDND has not been re-run on the 128 whole-pool lineages.

## References

- Leinster, T. & C. A. Cobbold. 2012. Measuring diversity: the importance of species similarity. *Ecology* 93: 477–489.
- Leinster, T. & M. W. Meckes. 2016. Maximizing diversity in biology and beyond. *Entropy* 18(3): 88.
- Wichmann, S., E. W. Holman, D. Bakker & C. H. Brown. 2010. Evaluating linguistic distance measures. *Physica A* 389: 3632–3639.
- Jäger, G. 2013. Phylogenetic inference from word lists using weighted alignment with empirically determined weights. *Language Dynamics and Change* 3(2): 245–291.
- Jäger, G. 2018. Global-scale phylogenetic linguistic inference from lexical resources. *Scientific Data* 5: 180189.
- Wichmann, S., E. W. Holman, C. H. Brown, M. S. Dryer & Q. Ran (eds.). 2025. *The ASJP Database* v21. CLDF: github.com/lexibank/asjp (CC-BY-4.0).
