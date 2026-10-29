# Related work (notes for a paper)

Web search, October 2026. Not yet a full scholarly-database search; see "Still to check".

## Closest prior work

**Essfors, H. 2025. Global Linguistic Diversity – Adapting the Leinster-Cobbold Framework from Ecology for
Humanities Research.** *Computational Humanities Research 2025*, Anthology of Computers and the Humanities 3:
652–668. https://doi.org/10.63744/srhQaCwGo5mj

Applies Leinster–Cobbold similarity-sensitive diversity to countries and continents: abundance = speaker numbers
(Ethnologue 22 + Joshua Project), similarity = 1 − mean LDN on ASJP lists (linear, no chance correction), diversity
profiles over q. Differs from isodiv in every key respect:

| | Essfors 2025 | isodiv |
|---|---|---|
| unit | a country's / continent's speaker population | a lineage's documented material |
| weights | speakers | none: Leinster–Meckes maximum diversity |
| dialect-count invariance | no (counts Ethnologue languages) | yes, by construction |
| chance correction | none | LDND / PMI-LDND |
| similarity | 1 − LDN | exp(−k·d), k calibrated on same-language pairs |

## Mathematical basis

- Leinster, T. & C. A. Cobbold. 2012. Measuring diversity: the importance of species similarity. *Ecology* 93.
- Leinster, T. & M. W. Meckes. 2016. Maximizing diversity in biology and beyond. *Entropy* 18(3): 88.
  https://arxiv.org/abs/1512.06314. Framed for ecology; no linguistic application found.

## Comparators for D_max

- Holman, E. W. et al. 2011. Automated dating of the world's language families based on lexical similarity.
  *Current Anthropology* 52(6): 841–875. ASJP time depths for nearly all families; calibrated on 52 dated groups,
  ~29% mean error. The first thing reviewers will ask D_max to be compared against.
- Wichmann, S., A. Müller & V. Velupillai. 2010. Homelands of the world's language families: a quantitative
  approach. *Diachronica* 27(2): 247–276. LDND-based internal diversity used spatially to locate homelands
  (82 families).

## Diversity indices and language counting

- Greenberg, J. 1956. The measurement of linguistic diversity. *Language* 32. The B-index already has a
  resemblance (similarity) term.
- Civico, M. 2025. Measuring Linguistic Diversity: Limits and Extensions of the Greenberg Index. *Journal of
  Quantitative Linguistics*. https://doi.org/10.1080/09296174.2025.2567699. Bounds; distance-weighted and
  entropy-based extensions. Speaker-based, counts languages.
- Hammarström, H. Counting languages in dialect continua using the criterion of mutual intelligibility.
  https://research.chalmers.se/en/publication/67582. Language count as chromatic number of an intelligibility
  graph: the problem isodiv sidesteps.
- Nonlinear diversification rates of linguistic phylogenies over the Holocene. 2019. *PLOS ONE*.
  https://doi.org/10.1371/journal.pone.0213126. Diversification from Glottolog/Ethnologue language counts.

## Methods used

- Wichmann, S., E. W. Holman, D. Bakker & C. H. Brown. 2010. Evaluating linguistic distance measures.
  *Physica A* 389: 3632–3639 (LDND).
- Jäger, G. 2013. Phylogenetic inference from word lists using weighted alignment with empirically determined
  weights. *Language Dynamics and Change* 3(2).
- Jäger, G. 2018. Global-scale phylogenetic linguistic inference from lexical resources. *Scientific Data* 5.

## Assessment

Apparently new: abundance-free maximum diversity as a measure of a lineage's internal diversity, argued from its
invariance to dialect counting. Not new: similarity-sensitive diversity of languages (Greenberg B; Essfors 2025),
ASJP-based family-level measures (Holman; Wichmann).

## Still to check

1. Dialectometry (Goebl, Nerbonne, Heeringa): aggregate measures of dialect-area diversity.
2. Faith's phylogenetic diversity on Glottolog trees (no hit, absence not confirmed).
3. Leinster 2021, *Entropy and Diversity: The Axiomatic Approach*, for linguistic examples.
4. A proper Google Scholar / Web of Science pass on "maximum diversity" / "magnitude" + language.
