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
restarts, `GaussianMixture` as a second paradigm, and a hand-written EM latent class model for
categorical answers. Nothing exotic, which is the point: this is a tool for deciding where to
spend a marketing budget, not a place to try out a new algorithm.

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

## Established practice the tool already follows

Each is cited in the code next to the thing it justifies. **Unverified against the library** —
these are from my own knowledge and are the first things to check when the PDFs open.

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
- **Hopkins is caveated rather than trusted.** It reads 0.78 on pure noise with two Likert
  questions, because duplicate answer patterns inflate it. Measured on this machine.
- **A typing rule with cross-validated accuracy**, reported as an operational property and
  explicitly not as evidence the segments are real — a partition of noise is still classifiable.

## Gaps I would look at, in order

Ranked by what would change an answer, not by novelty. **None of these is implemented, and I am
not implementing them on unverified recollection** — the point of the library is to decide which
are worth it.

1. **Mixed numeric and categorical answers in one model.** Today numeric goes to k-means and
   categorical to latent class; a survey with both must pick one. k-prototypes (Huang) is the
   standard answer. This is the largest real gap.
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
