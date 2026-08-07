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

## Large studies: what is computed over everybody, and what is not (2026-08-06)

Tested on the UCI bank marketing file — **41,188 real telephone-survey responses**, 21 mixed-type
columns. Previously the largest thing the tool had ever seen was 5,381.

Anything that needs a distance between **every pair** of respondents cannot be computed over
everybody at that size: 41,188 people make 848 million pairs, and the consensus routine holds two
dense n-by-n matrices, which is 27 GB. Three diagnostics are affected, and they are handled
differently on purpose:

| Diagnostic | At large n | Why |
|---|---|---|
| **consensus_PAC** | random 6,000, disclosed | PAC summarises how ambiguously pairs co-cluster; a sample estimates it well |
| **silhouette** | random 6,000, disclosed | An average over respondents; same argument |
| **Consensus ensemble partition** | skipped, stated | Has to place *every* respondent; a partition of a sample cannot |
| **Ward cross-check** | skipped, stated | Same: a partition of a sample cannot be compared with one of everybody |

Everything else — the segmentation itself, prediction strength, replication stability, the gap
statistic, per-segment Jaccard — uses all 41,188.

**What this actually fixed was not a crash.** A guard already dropped consensus clustering
entirely above n=5,000, so the 27 GB allocation was unreachable by the normal path. What it did
instead was quietly delete the `consensus_PAC` column — **one of the three criteria the k panel
weights double** — from every study above 5,000 respondents. The panel ran on two doubled criteria
instead of three and said nothing. Estimating it from a subsample restores the column and
discloses its basis, which is strictly better than a silent absence.

Measured after the change: 41,188 respondents in **1.3 minutes, 1.24 GB peak**, k=3, moderate
confidence, with the report naming the 848,205,078 pairs and the 6,000 respondents used.

The general rule this settles, and the reason the four rows above differ: **a diagnostic that
summarises may be sampled and disclosed; a diagnostic that must place every person may not be
sampled at all.** Sampling one of the latter and reporting it as though it covered the study is
the failure mode this tool exists to avoid.

## Memory at scale: diagnosed and fixed (2026-08-06)

Two large real files. The first pass through them looked like this:

| File | n | Wall | Peak RSS |
|---|---|---|---|
| UCI bank marketing | 41,188 | 1.3 min | 1.24 GB |
| UCI adult / census | 48,842 | 2.2 min | **8.5-11.2 GB** |

The second would swap or die on a 16 GB laptop, and the peak wandered by 3 GB across identical
runs — which is not what a fixed allocation does.

**One line, in `kprototypes.encode`.** It mapped each answer to its nearest known level with

```python
np.abs(col[:, None] - known[None, :]).argmin(1)
```

which allocates one number per respondent **per level**. On a five-point scale that is nothing.
Four of adult's columns are continuous measurements — age, a sampling weight, capital gains, hours
— which the detector types as ordinal, so "levels" meant 48,842 distinct values and the
intermediate was 19 GB per column. Replaced with a binary search: same answer, including the
scan's habit of resolving an exact midpoint downwards, in O(n log levels) time and O(n) space.
Verified identical over 300 random specs, and pinned by
`test_encoding_an_answer_scale_does_not_depend_on_how_many_levels_it_has`.

| File | Wall | Peak RSS |
|---|---|---|
| bank marketing, 41,188 | 0.94 min | 1.66 GB |
| **adult, 48,842** | **0.89 min** (was 2.19) | **1.59 GB** (was 11.04) |

**Worth recording how badly the search went, because the lesson is about method.** RSS sampling
said the memory climbed gradually with no single culprit; instrumenting twenty-six functions
showed no call raising the high-water mark by more than 0.29 GB. Both were true and both were
misleading — `typing_tool`, where it actually lived, was not in the wrapped list. I concluded from
that "no single allocation, therefore allocator churn across a thousand fits", wrote it into this
file as a hypothesis, and it was wrong. What settled it was bisecting the *unwrapped* functions
one at a time until `typing_tool` went 0.24 GB to 11.11 GB in one call.

A negative result from an incomplete instrument is not evidence of absence, and "no single big
allocation" was a conclusion about my wrapper list, not about the program.

## The categorical path was a poor relation (2026-08-07)

A survey made entirely of pick-any questions goes down the latent-class path. Compared against the
numeric report on the same measure — what evidence does the reader get to judge the result by —
it gave **two of eleven**:

| Evidence | k-means | Latent class (was) | Now |
|---|---|---|---|
| Split-half replication | yes | **no** | yes |
| Which of the three kinds of segmentation | yes | **no** | yes |
| Which segments border each other | yes | **no** (computed, discarded) | yes |
| How to read the per-person fit | yes | **no** (column existed, unexplained) | yes |
| Per-segment bootstrap Jaccard | yes | yes | yes |
| Typing-tool accuracy | yes | yes | yes |

The neighbours table is the one worth dwelling on: it was **already being computed**, under a
comment saying it existed "so the gorge plot and the 'which segments sit next to each other' table
are not privileges of the numeric half of the tool" — and then it was never passed to the report.
The intent was written down and the wiring was not finished.

Split-half replication is the one that mattered most. The confidence light is built from it, so a
multiple-choice survey had no answer at all to *would this come back again* — the single question
the whole report exists to answer. `lca_split_half` fits the model on each half independently and
has both label the same held-out half, which is what the numeric path has always done.

**And adding it immediately produced the fault this report has been cleaned of elsewhere**: a
split-half of 0.577 read as "partly reproduces" directly above "the division does not survive being
repeated on half the sample". Two adjacent sentences, one number, opposite readings — because I
wrote a fresh set of bands instead of using the 0.6 threshold `segmentation_kind` already applies.
The self-consistency checker now looks for exactly that pattern, and covers the categorical path.

## The report is now checked against itself (2026-08-06)

Three defects reached `main` while every test passed, and **all three were plainly visible in the
generated report**:

| Defect | What the report said |
|---|---|
| k-selection (v1.5.2) | A table showing 0.995 against 0.658 on the criterion the prose calls decisive, and a recommendation going the other way |
| Wide questionnaire | A green confidence light above evidence that could not support it |
| Persistence | A summary recommending the exact segment the table below told the reader not to fund |

None was caught, because every test in the suite asked whether the pipeline **ran** — never
whether the document it produced held together. That is the blind spot, and it had a perfect
record.

`test_the_report_agrees_with_itself_in_every_regime` now generates reports across the regimes that
produce materially different documents — real structure, none at all, overlapping, very unequal
sizes, and a two-group answer where the persistence table has only one direction available — and
reads each one back against its own numbers:

- the headline count of people and of groups must match the sizes table and the exported
  assignments;
- a share stated as a fraction must be that fraction, within three points;
- a segment may not be labelled `Segment 1` in one table and by name in another unless some table
  carries both on the same row (the exported CSV uses the number, the prose uses the name, so
  both have to exist — what may not exist is the gap between them);
- nothing may be recommended that the report elsewhere calls unsound;
- a green light may not sit above a cluster-tendency score described as essentially random.

The checker is itself checked: a report containing each of the shipped defects is fed to it and
every one must be caught, because a checker that cannot fail proves nothing.

**It found one more on its first run** — segments were labelled `Segment 0/1/2` in the sizes and
Jaccard tables and by name everywhere else, in every regime on the numeric paths, so the decisive
stability table could not be matched to any group the reader had been introduced to. Both tables
now carry the suggested name beside the number.

## Reading a report as its reader (2026-08-06)

Nobody had ever read one end to end as the marketer it is written for. Generated on 420 students
whose three mind-sets the tool recovered at an Adjusted Rand Index of **0.954** — every segment
genuinely real — the report argued with itself in four places.

**A real segment was condemned.** The persistence table scored each segment against the solutions
with one group fewer and one more, and reported the weaker of the two. Asking for one MORE group
forces the solution to split something, so whichever segment gets subdivided scores about 0.5 in
that direction whether or not it is genuine. The largest, cleanest segment — 168 of its 170
planted members recovered — held together perfectly under merging (1.00), scored 0.56 under
splitting, and the report said: *"Do not build a campaign on the ones marked 'dissolves'."*

The two directions are now reported separately, because only one of them is evidence:

| Direction | What a low score means |
|---|---|
| Fewer groups (merge) | The segment's members go to different places — it was never a unit. **Evidence it is not real.** |
| More groups (split) | The segment contains recognisable sub-groups. **An opportunity, not a defect** — some segment must split. |

At k=2 the merge direction does not exist (k-1 is a single group), so the metric now says nothing
about realness there rather than borrowing the split direction. That question is answered above it
by Hopkins, the dip test, split-half replication and per-segment Jaccard.

**Three smaller contradictions, all in the plain-language box:**

- *"Start with the biggest, most distinct group"* — nothing in the analysis establishes that the
  biggest is the most distinct, and here the biggest was the one the table below condemned. It now
  points at the stability tables and says the largest is not always the soundest.
- *"about 1 in 3 (40%)"* — a claim and its own contradiction inside one set of brackets. The
  fraction phrasing is only used when it is accurate to within three points.
- The green light said the groups *"are clear"*, five lines above a Hopkins statistic of 0.59
  described as *"essentially random"*. The light is built from stability numbers and now claims
  only that: the same groups come back when the analysis is repeated.

**And the names.** Every segment appeared throughout the report as something like *"planning things
rather + want meet people outside"* — stopwords stripped and truncated mid-phrase, on a dangling
conjunction. These are placeholders the team is told to replace, so their one job is to be
recognisable as the question behind them, which the original words do and a stub does not.

The pattern across all five: **the summary asserted things the evidence under it did not support.**
Every test in the suite asked whether the pipeline ran, not whether the document agreed with
itself — which is also how the k-selection defect in v1.5.2 and the wide-questionnaire defect
survived. Two tests now read the generated report and check it against its own tables.

## The claim that failed: confidently wrong on a wide questionnaire (2026-08-06)

The property this tool is built around, stated in the section above and pinned by a test, is that
**it never reports high confidence while wrong**. Feeding it a shape it had never seen falsified
that.

150 respondents, 400 questions, a genuine three-group structure carried by 60 of them. It found
**two** groups, at an Adjusted Rand Index of 0.635 against the truth, and called it **High
confidence on two runs out of three**.

The mechanism is the one the panel is least equipped to notice, because it is the panel's own
philosophy turned against it. With many questions and few people, distances concentrate:
respondents end up roughly equidistant, the real structure is diluted across the 340 questions
carrying none, and what remains is **highly reproducible — because noise reproduces**. Prediction
strength, replication stability and per-segment Jaccard all agreed on the wrong answer, and they
are exactly the criteria weighted most heavily.

The light is now capped below green when the sample cannot support the questionnaire (fewer than
two respondents per question), and the report says which numbers it is holding back on and why.
It does not change the segmentation, only the claim made for it. Verified: the same file now
reports Moderate on all three seeds, and normal-shaped studies — 2,800 respondents on 25 Big Five
items, 48,842 on 6 — are untouched.

**The general lesson is about the evidence, not the threshold.** Every criterion here answers
"would this answer come back again?". None answers "is this answer right?". Where a regime makes
noise reproducible, agreement among them is not corroboration, and a panel of them cannot tell.

## How big a study it handles (2026-08-06)

| File | n | Columns | Method | Wall | Peak RSS |
|---|---|---|---|---|---|
| UCI bank marketing | 41,188 | 21 mixed | Gower k-prototypes | 0.94 min | 1.66 GB |
| UCI adult / census | 48,842 | 16 mixed | Gower k-prototypes | 0.89 min | 1.59 GB |
| **UCI covertype** | **581,012** | 55 numeric | k-means | **3.2 min** | **3.71 GB** |

All 581,012 respondents were assigned, the headline says 581,012 people, the segment sizes sum to
581,012, and the report names the 168,787,181,566 pairs the two sampled columns would otherwise
have needed. Nothing is quietly describing a subset.

**An answer list does not grow with the number of respondents.** The cap on how many options a
question may offer was a quarter of the sample, which on the 541,909-row UCI online retail file
permitted 135,477 "options" — so invoice numbers (25,900 distinct), stock codes (4,070) and
free-text product descriptions (4,223) were all clustered on as pick-any answers. Beyond making no
sense (Gower scores two nominal answers as identical or not, so with thousands of levels nearly
every pair simply differs) it did not finish in half an hour. The ceiling is now absolute at 100,
which real option lists — brands, universities, countries — sit well inside, and the columns are
set aside with a note naming the override rather than dropped in silence.

## A column its own outliers have flattened (2026-08-06)

Range scaling divides by max minus min, so two extreme values can flatten a whole column. On the
UCI online retail file a returned order of -80,995 against a median quantity of 3 put **100% of
541,909 respondents inside 2% of the scale**.

That is not only slow — k-means cannot separate points that are all but coincident, so it spends
every restart hitting its iteration limit, which is why three numeric columns took 20 minutes when
55 columns of covertype took three — it is wrong. Whatever comes back describes the two outliers
rather than the half million people.

Detected and stated, not silently corrected. `--scaling robust` divides by the interquartile range
instead and exists for exactly this, but whether those extremes are data-entry errors or the most
interesting rows in the file is a judgement about the study. The note names the remedy and leaves
the choice.

Fires correctly beyond the case that prompted it: `capital_loss` in the adult file (zero for
about 95% of people) and `pdays` in bank marketing (999 for most) are both genuinely degenerate,
and both are now called out rather than quietly contributing nothing.

## Estimates on a working set, answers on everybody (2026-08-06)

Separately from the above, and kept because it is right independently of it: the k-selection panel
is made entirely of resampling estimates — the gap statistic against 20 reference datasets,
replication stability over 30 resamples, prediction strength over 10 splits, consensus over 50,
bootstrap Jaccard over 100 — which at seven candidate k values is well over a thousand clusterings
of whatever it is handed.

Above `MAX_SEARCH_N` (12,000) those estimates run on a random sample of that size, and the answer
still covers everybody: the final fit, every respondent's segment, the profiles, the charts and
the exports use the whole file. The report says so in as many words. Estimating a resampling
statistic from 12,000 people rather than 48,842 is what such statistics are for; a segmentation
that quietly described a sample would not be, and that is not what this does.

Measured: it did not change the k chosen on either large file.

## First contact with real public data (2026-08-06)

Every validation before this one used data I generated, from rounded Gaussians around planted
centres — which is precisely the model k-means with range scaling assumes. The test suite had 85
random-number calls and the repository one example CSV. The engine had never seen a real
respondent.

Five open datasets, fed in raw, exactly as a user would drop them in:

| File | Shape | Routed to | Result | Fair? |
|---|---|---|---|---|
| `psych::bfi` Big Five | 2800 x 25 Likert, 731 missing cells | k-means | Hopkins 0.57, k=2, Moderate, *reproducible* | **Yes.** Personality is continuous trait space; there are no discrete types. "No natural clusters, but a stable working split" is the correct answer |
| `MASS::survey` student measurements | 237 x 13, 131 missing | k-prototypes | k=2, **Low**, *constructive* | **Yes** — a grab-bag with no segment structure, and it refused |
| `carData::WVS` | 5381 x 7 | Latent class | k=2, High | Yes |
| `carData::Mroz` | 753 x 9 | k-prototypes | k=3, High, *reproducible* | Plausible |
| `carData::Chile` plebiscite | 2700 x 9, 295 missing | k-prototypes | k=8, Moderate, *natural* | **No — see below** |

Nothing crashed, nothing hung, and every file routed to the right method unaided. The calibration
result is the reassuring one: on the two datasets that genuinely contain no segments it said so,
in one case with a red light.

**Two real defects surfaced, both in ingest rather than in the statistics.**

*Education was clustered on.* `_DEMO_WORDS` held the Swedish `utbildning` and never the English
`education`, so on the Big Five file a 1-5 education code became a 26th personality "question".
A numeric demographic code cannot be told from a rating scale by its values — 1-5 education looks
exactly like a six-point item — so the name list is load-bearing, and it now carries both
languages for each concept.

*Town size defined half a segmentation.* On the Chilean plebiscite file the eight segments were
each pure on how the person voted and then split in two by `population`, the size of their town
(3,750 to 250,000). Four of the eight "mind-sets" were really "lives somewhere bigger". No answer
scale reaches five figures, so a column on that magnitude is now flagged in the detection notes —
warned about, not excluded, because whether a number is an answer or a circumstance is a
judgement about the study and not a property of the data.

Three tidier rules were tried against this file and died on measurement: that `population` is a
function of `region` (it is not — region C holds nine distinct values), and two variants of the
stability band described above. The pattern is consistent enough to be worth naming: on this
project, rules that look principled in the abstract fail about as often as they succeed, and the
only reliable filter is measuring them against data whose answer is known.

## The export formats the tools actually write (2026-08-06)

Research datasets arrive tidy. The files a survey platform exports do not, and that is what the
app is pointed at. Four formats, same 240 respondents written four ways:

| Format | Before | After |
|---|---|---|
| Qualtrics (3 header rows) | **242 rows**, every column text, routed to latent class — 1-5 scales treated as unordered | 240 rows, k-means, ARI 1.000 against the clean file |
| SurveyMonkey (2 header rows) | **241 rows**, same contamination | 240 rows, k-means, ARI 1.000 |
| Swedish Excel (`;` + decimal comma) | 240 rows, but the 0-10 satisfaction score **silently dropped** | kept as a number, ARI 1.000 |
| Clean CSV | correct | unchanged |

Every one of these failed silently. The Qualtrics case is the worst: two metadata rows became
respondents, the question wording turned every rating column into text, and the survey was then
analysed as if its scales were unordered categories — with no error and an ordinary-looking
report. A professional tool's default export was the input most likely to be wrong.

**Header rows are recognised by being non-numeric where the answers below them are numeric.** A
respondent's answer parses like the answers under it; a question's wording does not. On an
all-categorical survey no column is "otherwise numeric", so nothing matches and nothing is
dropped — the rule is safe by construction rather than by tuning, and
`test_header_stripping_never_eats_a_real_respondent` holds it to that, including for a respondent
who skipped every question.

**Decimal commas** are converted only for whole columns where every value is digits-comma-digits.
Had the file been comma-delimited such a value could not have survived in one cell, so this cannot
misread a comma-joined multi-select answer.

Encoding, BOM, `;`/tab sniffing and `.xlsx` were already handled and needed no change.

## Predicted, measured, and did not happen: response-style segments (2026-08-06)

The textbook failure of survey segmentation is that clustering raw Likert answers recovers **how
people use the scale** rather than what they think — extreme responders in one group, midpoint
huggers in another — and presents scale use as a mind-set. It is the first thing to suspect when a
segmentation meets real respondents, and this engine had never been tested against it.

Tested with 600 respondents each carrying an attitude segment **and**, independently, a response
style (extreme, midpoint, acquiescent), so "which did it recover?" is decidable — which is
precisely what real survey data cannot tell you, since it carries no ground truth for either.

| Run | k | ARI vs attitude | ARI vs response style |
|---|---|---|---|
| Control, no styles | 3 | **1.000** | 0.000 |
| Styles present, default path | 2 | 0.612 | **-0.002** |
| Styles present, `--scaling ipsative` | 2 | 0.612 | -0.002 |

**The failure does not occur.** Association with response style is zero. Pinned by
`test_response_styles_do_not_become_the_segments`.

**Do not add ipsative (row-centred) scaling as a defence against this.** It is the standard
remedy, it is already implemented, and measured here it is *worse*: holding k at the true value,
range scaling recovers attitudes at 0.977 and ipsative at 0.819. Row-centring removes the
between-person level differences that carry real signal along with the response-style artefact.
Adding it as an automatic default would have degraded results on the path the app uses.

What response styles actually cost is **resolution, not correctness**. They blur the three-way
structure until the two-way split is genuinely the more reproducible fact — stability 0.997
against 0.967, prediction strength 0.988 against 0.897 — so the tool merges two segments. It then
reports Moderate rather than High, which is the behaviour that makes the merge survivable. Expect
merged segments and a Moderate rating on style-heavy real data, not nonsense.

### Loose thread: prediction strength is not monotone

On that same file prediction strength ran 0.988, 0.897, 0.848, 0.496, 0.720, 0.872, 0.794 — it
collapses at k=5 and recovers. Tibshirani & Walther's "largest k above the cutoff" assumes decay,
so the signal voted **k=7** on three-segment data. The panel absorbed it (2 of 12 weight units),
but on a closer vote it could decide the answer. This is the same shape as the stability defect
fixed in v1.5.2, still present where the rule is the published one. Not changed yet: how often it
bites has not been measured.

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
