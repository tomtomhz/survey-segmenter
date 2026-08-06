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

## Showing the data, not a version of it

The segment map is the one chart that can falsify the whole result, so it is held to the same
standard as the statistics. Three things were wrong with it and are measured here:

| | Was | Now |
|---|---|---|
| Respondents drawn on the map | a random 1,200 | **all of them** |
| Respondents on the per-person fit chart | a random 900 | **all of them** |
| Visible on a 5-question survey, n=3,000 | 422 of 3,000 dots (**14%**) | all 3,000, as 422 dots sized by count |

Rating answers come in whole steps, so respondents land on a finite grid and stack invisibly.
Measured: 3,000 people answering five 1-5 questions occupy 422 distinct positions; with twelve
questions it is 2,962, so this is a short-survey problem and short surveys are the common case.
Each dot is therefore drawn once per distinct answer pattern with **area proportional to how many
people share it**, plus a size key.

**Jitter is rejected on purpose.** It is the usual remedy for discrete data, and Wilke's
*Fundamentals of Data Visualization* states the cost plainly — jittering too much ends up "placing
points in locations that are not representative of the underlying dataset". A chart whose only job
is to let somebody check the answer must not invent coordinates to make itself readable.

The decision-region shading is drawn in two dimensions while the grouping happened in as many
dimensions as there are questions. Rather than assume those agree, the caption reports how often
the picture's own rule reproduces the real assignment: 99.5-100% on well-separated survey data,
and it drops — correctly warned about in the caption — on data where the projection is a poor
shadow, e.g. 89% on pure noise where the two directions carry only 21% of the variation each.

Per-axis variance share is on the axes themselves, not only in the prose.

## Is there anything to segment? Two tests, not one

Hopkins was the only real answer to this, and it has two measured failure modes. Adolfsson,
Ackerman & Brownstein, *To Cluster, or Not to Cluster* (Pattern Recognition, 2019), 35,000
simulated datasets: its power falls to **32%** on partially overlapping clusters, and it reads a
handful of outliers as a cluster. Overlapping segments merging into one is this tool's single
measured failure mode, so the second test earns its place exactly there.

`clusterability.py` adds Hartigan's dip test, and the report states how the two line up — worth
more than either alone, because they fail in opposite directions.

**The form the paper recommends does not work on survey data, and that is worth recording.** Its
headline method is *dip-dist*: the dip on all pairwise distances. Rating answers are whole numbers,
so those distances take very few values — measured, 400 people on five questions give **79,800
distances with 50 distinct values among them** — and the dip reads that comb of spikes as many
modes, returning **p = 0.0000 on data with no groups at all**. The same test on continuous noise of
identical size returns p = 0.9962. The paper's simulations are continuous Gaussians; nothing there
was run against a five-point questionnaire.

What is used is the paper's other evaluated variant, **PCA-dip**: the dip on the first principal
component, which is a weighted sum of every question and therefore continuous even when each answer
is not.

### Measured, with the guard in place

| Condition | Hopkins | Dip p | Correct? |
|---|---|---|---|
| 3 groups, well separated | 0.91 | <0.001 | yes |
| 3 groups, moderate overlap | 0.84 | <0.001 | yes |
| **3 groups, heavy overlap** | **0.66** | **<0.001** | **yes — where Hopkins fades** |
| 5 groups, separated | 0.90 | <0.001 | yes |
| Pure noise, 5 questions | 0.59 | 0.94 | yes |
| Pure noise, 12 questions | 0.51 | 0.995 | yes |
| **Noise + 5 outliers** | **0.55–0.60** | **0.51** | **yes — Hopkins drifts up, dip does not** |
| Pure noise, 2 questions | 1.00 | *refused* | refuses rather than false-alarms |

The floor is four questions: on pure noise the dip false-alarms at two and three and is correct
from four up. The guard counts **questions**, not distinct values, and the first version got that
wrong — guarding on distinct values refused three well-separated groups, because people in one tight
segment genuinely give identical answers. A low count has two causes and only one is a fault.

The statistic itself is the `diptest` package rather than a local implementation. That is a
deliberate dependency: the first hand-rolled attempt scored a plain normal sample at 0.25, the
maximum possible, by measuring distance from *convexity* rather than from *unimodality* — a
unimodal distribution is convex then concave. The reference scores it 0.0091. The tool still runs
without the package and reports the check as not run.

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

### How the two reproducibility signals read the table (corrected 2026-08-06)

The two highest-weighted signals answer their question differently, and conflating them cost a
whole segmentation before anyone noticed.

**Prediction strength** takes *the largest k that clears 0.80*. That is Tibshirani & Walther's
published rule and it is right: prediction strength falls away as k passes the real structure, so
the largest k still above the line is the finest split the data supports.

**Replication stability has no such rule**, and the same "largest above the cutoff" logic had been
carried across to it. It does not hold there. Measured on a file with three planted segments,
stability ran 0.995 at k=3 and 0.778 at k=4; both clear 0.75, so the rule handed a doubled signal
to the visibly worse answer. That tied the vote, parsimony took k=2, and three real segments were
reported as a *constructive* segmentation — the method inventing groups where there are none.
Measured against the planted truth: 0.618 as it stood, 0.992 once corrected.

Each k is now compared against the best **using its own standard error**: it stays in contention
while the gap to the best is no larger than the uncertainty of its own estimate. This needs no
tuning constant, and two alternatives that look equally principled both fail on measured data —

- *A band drawn around the best* collapses whenever the best k scores a standard error of exactly
  zero, which happens readily at k=2 when every resample agrees. On an unequal 80/15/5 split that
  alone excluded k=3 and lost the 5% segment.
- *Taking the smallest k within the band* double-counts parsimony, because the vote already breaks
  ties toward the smaller solution.

The tie-break was the second half of the same fault: it went to the smaller k unconditionally,
including over a k holding **both** doubled criteria. Parsimony now breaks only what those
criteria leave level.

Both are pinned by tests built on the measured diagnostics table that exposed them
(`test_the_stability_signal_backs_the_k_that_is_actually_stable`,
`test_a_tie_is_broken_by_the_criteria_the_method_leans_on`). Re-measured across the battery above,
the correction recovered a planted k=5 that was previously reported as k=2, and changed no case
that was already right.

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

## Built, measured, and not adopted: sparse k-means

Every question counts equally when respondents are grouped, and questions that separate nobody drag
membership accuracy down — a real problem with a standard answer, Witten & Tibshirani's *A Framework
for Feature Selection in Clustering* (JASA, 2010). It was implemented properly, measured against
this tool's own conditions, and rejected. Reproduce with `python3 references/sparse_kmeans.py`.

| What was asked | What was measured |
|---|---|
| Does it group people better? | On ordinary conditions, no: plain k-means already recovers the planted groups at ARI 1.00 and so does this. A gain appears only when noise questions outnumber real ones and the groups are weak — +0.13 at two real against ten noise |
| Is it safe as the clustering method? | **No.** On pure noise it lifts the silhouette from 0.12 to 0.39 at six questions and from 0.06 to 0.43 at twelve. It picks whichever questions best split the noise and weights them up, so structureless data comes out looking separated |
| Can its weights tell real from noise? | **No.** Share of weight on the top three questions: 99% on real structure, 96% on pure noise |
| Does it beat what is already reported? | No. At ranking every real question above every noise question, eta-squared scored 5/5 across four conditions — and so did this |

The stability gates did still catch the inflated case (split-half ARI 0.26-0.45 against 1.00 for
real structure), so nothing would have escaped to a user. That is not a reason to ship it: a
headline separation number that flatters noise should not exist and be contained downstream.

A correctly implemented published method that solves a problem this tool does not have, using a
capability it already owns, at the cost of a statistic that lies about noise. The implementation is
kept under `references/` so the finding can be rechecked, and deliberately not shipped.

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
