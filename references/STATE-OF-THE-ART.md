# What this tool does, measured against current practice

**Written 2026-07-31.** Two kinds of claim appear below and they are labelled differently on
purpose:

- **Measured** — run on this machine against data whose answer is known. Reproduce with
  `references/kbench.py`.
- **Established** — standard results the tool already cites in code. These come from my own
  knowledge, **not** from reading the library in `~/Desktop/K-Means Clustering Books/`, which is
  in iCloud and would not materialise (see `README.md`). They are the claims to check first when
  those files are readable.

Nothing here is sourced from a paper I could not open. Where I could not verify something, it
says so.

---

## What the engine is

scikit-learn computes, matplotlib draws. `KMeans` with k-means++ initialisation and many
restarts, `GaussianMixture` as a second paradigm, a hand-written EM latent class model for
categorical answers, and Gower k-prototypes for questionnaires that mix the two. Nothing exotic,
which is the point: this is a tool for deciding where to spend a marketing budget, not a place to
try out a new algorithm.

Which one runs is detected, not configured: all ratings go to k-means, all pick-any questions to
latent class, and a mix of both to k-prototypes.

## Mixed questionnaires (added 2026-08-01)

Most real questionnaires have both kinds of question. Until now the pick-any columns were set
aside with an apology and the segmentation ran on the ratings alone — which on a study where the
brand question is the interesting one threw away the finding.

`kprototypes.py` implements Szepannek, Aschenbruck & Wilhelm, *Clustering large mixed-type data
with ordinal variables* (ADAC 2024), matching `kproto(type = "gower")` in `clustMixType`:

| | Distance | Prototype |
|---|---|---|
| Rating (ordinal) | range-normalised gap between **midranks** | level whose rank is nearest the cluster's median rank |
| Measurement (numeric) | range-normalised absolute gap | median |
| Pick-any (nominal) | 1 if different, 0 if the same | mode |

Median and mode rather than means is the 2024 paper's point, not a detail: Gower is an L1-type
distance, so those are the exact per-variable minimisers, and they are what make the algorithm
provably terminate. Porting k-means naively — means everywhere — breaks that.

**Ordinal answers use Podani's metric version (1999, eq. 3), not his tie-corrected non-metric one
(eq. 2b).** The non-metric version is the more commonly cited, and it is wrong for survey data:
it subtracts the within-tie spread from every gap, which with a handful of levels and hundreds of
respondents collapses adjacent answers to nearly zero while leaving the extremes a full 1 apart.
Worked through for three equally-used levels it gives d(1,3) = 1.00 against
d(1,2) + d(2,3) = 0.065 — a triangle-inequality violation large enough that "closest prototype"
stops meaning anything.

### Measured

Nine analyses, 400 respondents, four ratings plus two pick-any questions, true k = 3. Reproduce
with `references/kbench.py`.

| Which questions carry the signal | Found k | Confidence | ARI |
|---|---|---|---|
| Both kinds | **3, 3, 3** | high | 0.88-0.89 |
| Ratings only (pick-any is noise) | 2, 2, 2 | moderate, low | 0.38-0.43 |
| Pick-any only (ratings are noise) | 2, 3, 6 | low | 0.21-0.31 |

**It never reported high confidence while wrong** — the same property the all-ratings path is held
to, and the one that makes the tool usable at all.

The accuracy cost when half the questions are useless is the ordinary cost of clustering on noise
variables, not something particular to this method: measured separately, three useless pick-any
columns beside six real ratings cost 0.25 ARI. The report's variable-selection check names them,
and dropping them lifted the silhouette from 0.25 to 0.52 in that run.

### What this path cannot do, and says so

- **No Gaussian-mixture cross-check, and no GMM BIC vote in the k panel.** A mixture assumes every
  answer is a measurement; fitting one to brand codes returns a number that reads as corroboration
  and is not. Left in by accident during development it did real damage — on a mixed file whose
  true answer was 3, the GMM BIC and ICL both argued for 8. The bottom-up cross-check becomes
  average linkage on Gower's distance instead, so there is one independent paradigm rather than
  two, and the report states this rather than quietly showing fewer numbers.
- **Calinski-Harabasz and Davies-Bouldin are read on the embedding**, not on Gower itself, since
  both are sums of squares. The silhouette, prediction strength, replication, per-segment Jaccard
  and segment-level stability are all exact.
- **The gap statistic falls back to the paper's reference method (a)** — each variable drawn
  independently over its own support — because method (b) needs principal components, and no
  rotated bounding box contains only answer combinations a respondent could have given.

### The identity that makes it maintainable

Divide each rating by its range and replace each pick-any answer with half a one-hot indicator,
and plain Manhattan distance on those coordinates reproduces Gower's distance **exactly**
(verified to 1e-16, and pinned by `test_gower_is_exactly_a_manhattan_distance_in_disguise`). So
the silhouette, the cluster-tendency test, the hierarchical cross-check and the segment map all
run through the same scikit-learn functions as the numeric path with nothing but
`metric="manhattan"` — rather than a second family of hand-written Gower versions that would need
their own tests and would drift.

## Choosing k, and refusing to

The number of groups is not chosen by an elbow. It is a weighted panel, and the weighting is the
opinionated part:

| Signal | Weight | Why |
|---|---|---|
| Prediction strength (Tibshirani & Walther) | Highest | A segmentation nobody can reproduce is not a finding |
| Split-half replication (ARI) | Highest | Same reason, different failure mode |
| Silhouette, Calinski-Harabasz, Davies-Bouldin | Middle | Separation, once reproducibility is established |
| Gap statistic | Middle | Compares against a null of no structure |
| Inertia elbow | Lowest | Shown for completeness; it always bends somewhere |

Per-segment bootstrap Jaccard (Hennig) decides which individual segments are real, which is a
different question from how many there are — a solution can be right about k and still contain
one segment that dissolves on resampling.

### Measured: does it work?

21 analyses, 400 respondents each, three seeds per condition, five-point answer scales.

| Condition | Recovered the true k | Confidence reported |
|---|---|---|
| Well-separated, k=3 | **3/3** | high |
| Sizes 80 / 15 / 5 | **3/3** | high, moderate |
| 3 real questions + 5 that separate nobody | **3/3** | high |
| Two elongated bands (k-means' worst case) | **3/3** | high |
| k=5 | **3/3** | high |
| Pure noise, no structure | **3/3 correctly refused** | low |
| **Overlapping, k=3** | **1/3** | high (when right), moderate (when wrong) |

**19 of 21.** Both misses were overlapping segments, where it merged two into k=2 — and dropped
to "moderate" both times it did.

The single most important line in that table is the last one and the one before it. **It never
reported high confidence while wrong**, and it never invented segments in noise. A tool that is
occasionally conservative is usable; a tool that is confidently wrong is worse than no tool,
because it launders a guess into a decision. Both properties are now pinned by tests
(`test_it_never_claims_high_confidence_for_the_wrong_number_of_groups`).

The weakness is real and worth stating plainly: **overlapping segments get merged.** If two
mind-sets genuinely shade into one another, expect this tool to report one group and tell you the
evidence is directional.

## Verified against the library

Only one file would open (`The Hundred-Page Machine Learning Book`, Burkov). It corroborates the
tool's single most consequential choice — how it picks k — on three specific points:

| Burkov, §9.2.3 "Determining the Number of Clusters" | What the tool does |
|---|---|
| Presents **prediction strength** as the one practical formal method, over eyeballing scatter plots ("subjective, prone to error… an educated guess rather than a scientific method") | Weights prediction strength highest in the panel |
| "a reasonable number of clusters is the largest k such that ps(k) is above **0.8**" | `ps_cutoff = 0.80`, and the k chart draws that line so the reader can see whether anything clears it |
| "For non-deterministic algorithms such as k-means… do multiple runs for the same k and compute the **average** prediction strength" | `ps_splits = 10`, and the report carries `prediction_strength_sd` alongside the mean |

That is three for three, including the exact threshold. It also confirms the tool's decision to
draw the segment map *and* refuse to let it decide anything: Burkov's objection to reading
structure off a scatter plot is why the map is presented next to the stability numbers rather
than instead of them.

## Established practice, not yet verified against the library

Each is cited in the code next to the thing it justifies. These are from my own knowledge and are
the first things to check when the remaining PDFs open.

- **Range standardisation** rather than z-scores (Milligan & Cooper 1988). On a 1-5 answer scale
  z-scoring inflates the questions nobody disagreed about.
- **Many restarts.** k-means finds a local optimum; the reported solution is the best of 50, and
  the report says what share of restarts reached it.
- **Segments are constructed, not discovered** (Dolnicar & Leisch). The report says this in those
  words, because a data-driven segmentation of unstructured data still returns segments.
- **Consensus clustering and PAC** (Monti; Şenbabaoğlu) for robustness to resampling.
- **Silhouette read on Kaufman & Rousseeuw's conventional bands**, with the average — not the
  count of negative scores — deciding the verdict, because noise produces few negatives and
  near-zero averages.
- **Hopkins is caveated rather than trusted, and the caveat is correctly scoped.** Measured
  here: on a five-question survey it reads **0.56 on pure noise** (0.5 being random) and **0.95
  on real structure** — it works. The failure is specific to very short surveys, where duplicate
  answer patterns inflate it to 0.78 on noise, and that is exactly the case the code warns about.
  Checked because it looked like a gap worth closing with a second tendency test; it is not.
  The gap statistic and prediction strength both independently picked k=2 on that noise while the
  confidence came out **low**, so three signals agreed and the tool refused.
- **A typing rule with cross-validated accuracy**, reported as an operational property and
  explicitly not as evidence the segments are real — a partition of noise is still classifiable.

## Gaps I would look at, in order

Ranked by what would change an answer, not by novelty. **None of these is implemented, and I am
not implementing them on unverified recollection** — the point of the library is to decide which
are worth it.

1. ~~**Mixed numeric and categorical answers in one model.**~~ **Done 2026-08-01** — see the
   mixed-questionnaires section above. Gower k-prototypes (Szepannek et al. 2024) rather than
   Huang's original, because Huang needs a `gamma` weight trading numeric distance against
   categorical mismatches that nobody in the room can defend.
2. **A second cluster-tendency test.** Hopkins is the tool's only one and it is unreliable enough
   to need a caveat. Hartigan's dip test on the pairwise-distance distribution is the usual
   companion.
3. **Variable weighting.** The tool reports which questions drive the segmentation (eta-squared)
   and checks whether dropping the noise ones changes the answer, but it weights every question
   equally when clustering. Sparse k-means (Witten & Tibshirani) learns the weights.
4. **Resolving overlapping segments** — the measured weakness above. Worth understanding before
   attempting: merging may be the honest answer, and "improving" it risks trading the confidence
   property, which is the more valuable one.

## What is not worth adopting

The library is heavy on variants — kernel, fuzzy, possibilistic, metaheuristic, deep. For a tool
whose users are marketers acting on the output, each adds a parameter nobody in the room can
defend and a result nobody can reproduce by hand. The bar is not "is this newer" but "would this
change a decision, and could the person making it explain why". k-means with honest validation
clears that bar; a whale-optimisation-tuned kernel fuzzy c-means does not.

## Reproducing the measurements

```bash
cd ~/dev/survey-segmenter
python3 references/kbench.py        # ~1 minute, prints the table above
pytest -k "recovers_the_number or never_claims_high"
```
