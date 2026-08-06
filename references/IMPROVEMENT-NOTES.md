# Reading notes and proposed improvements

**2026-08-01.** A session spent reading rather than coding: the twelve books in
`~/Desktop/K-Means Clustering Books/` (the clustering chapters, not cover to cover), plus
open-access literature online.

> **Status, later the same day: all six are implemented.** What each one actually cost, and where
> the write-up below turned out to be wrong once it met data:
>
> 1. **Shadow values** — as described. Every respondent now gets a `fit`, at any sample size.
> 2. **Segment neighbours** — as described. Thresholds were guessed at 0.90/0.75 and recalibrated
>    empirically to 0.80/0.55.
> 3. **Segment-level stability** — implemented, but **not scored by Jaccard as this note assumed.**
>    Going from k to k-1 must merge two segments, so Jaccard collapses whether or not the
>    structure is real: measured, it could not separate three genuine segments (0.44-0.78) from
>    pure noise (0.55-0.69) at all. Containment — what share of a segment's members stay together
>    — is unharmed by merging and separates them cleanly (0.78-1.00 against 0.47).
> 4. **The three kinds of segmentation** — as described.
> 5. **Gorge plot** — as described, as a seventh chart.
> 6. **Mixed-type k-prototypes** — implemented in `kprototypes.py`; the Podani choice below is the
>    one thing this note got materially wrong, see the correction under item 6.
>
> The full measured picture is in `STATE-OF-THE-ART.md`.

Sources are named inline. Where something is **measured**, it was run on this machine and can be
reproduced. Where it is **quoted**, the source is given so it can be checked.

---

## The single most useful thing I found

**Dolnicar, Grün & Leisch, *Market Segmentation Analysis* (Springer 2018, open access CC BY 4.0).**

This is not a clustering textbook. It is a book about *market segmentation specifically*, by the
authors this tool already cites, and it is free. It covers the whole pipeline as ten numbered
steps — collecting data, extracting segments, profiling, describing, selecting targets — which is
exactly what this tool does end to end. Everything in the "worth doing" list below traces back to
it or to Leisch's methods papers behind it.

Recommendation before any code changes: read Step 5 (Extracting Segments) and Step 6 (Profiling)
properly. It is the closest thing to a specification for this product that exists.

---

## Worth doing, in order

### 1. Shadow values instead of silhouette for the per-respondent `fit` column

**Source:** Leisch, *Neighborhood Graphs, Stripes and Shadow Plots* (2010). The shadow value of a
respondent is

```
s(x) = 2·d(x, closest centroid) / [ d(x, closest) + d(x, second closest) ]
```

near 0 when someone sits squarely in their own segment, near 1 when they are stranded halfway
between two. Leisch says outright it is "similar both in spirit and interpretation to the well
known silhouette plots".

**Why it is better here, concretely:** silhouette needs every pairwise distance — O(n²) — which is
why the `fit` column I added yesterday silently switches to a 6,000-respondent sample above that
size and leaves everyone else blank. Shadow values need only the distance to two centroids —
O(n·k) — so **every respondent gets a real number at any sample size**, and the CRM export stops
having holes in it precisely on the big studies where it matters most.

Same meaning, same reading, no sampling, and faster. This one is close to free.

### 2. Tell the user which segments border each other

**Source:** same paper. Averaging the shadow value over everyone whose closest segment is *i* and
second-closest is *j* gives a similarity between segments *i* and *j*.

The tool currently says how distinct each segment is from the average. It never says **which two
segments are nearly the same**, which is the question a marketer actually asks before signing off
on five campaigns. "Segments 2 and 4 sit next to each other and 30% of their members could
plausibly belong to either — consider one campaign, not two" is a sentence this tool cannot
currently write, and it would change what somebody spends money on.

### 3. Segment-level stability across solutions (SLSA)

**Source:** Dolnicar & Leisch, *Using segment level stability to select target segments in
data-driven market segmentation studies* (2017); implemented as `slsaplot` in the R package
`flexclust`.

The tool measures stability two ways: does the whole solution reproduce on a fresh half (global,
split-half ARI), and does each segment survive resampling (Hennig's bootstrap Jaccard, within one
k). It never asks the third question: **does this segment still exist if I had chosen a different
number of segments?**

That is the question behind a client asking "you said four groups — what if it were five?" A
segment that reappears intact at k=3, 4 and 5 is a real thing. One that only exists at k=4 is an
artefact of the number, and should not get a campaign.

More work than (1) and (2) — it means keeping the fits across the whole k range and tracking
membership overlap between adjacent solutions. But it is the most *decision-relevant* of the three.

### 4. Say which of the three kinds of segmentation this is

**Source:** Dolnicar & Leisch's framing, used throughout the book.

- **Natural** — genuine clusters exist in the data.
- **Reproducible** — no natural clusters, but enough structure that the same solution comes back
  every time, so it is a usable working split.
- **Constructive** — no structure; the segments are entirely an artefact of the method.

The tool's green/amber/red maps onto this almost exactly and already says "constructed, not
discovered" in the report. Adopting the field's own three words would cost a paragraph and make
the output legible to any analyst on the client side who knows the literature. Cheap, and it makes
the tool sound like it belongs in the conversation.

### 5. Gorge plot as a seventh chart

**Source:** same. A density of all the shadow values. Two humps with a dip between them ("a
gorge") means the segments genuinely separate; a single hump piled up near 1 means everybody is
stranded between centroids and there is nothing there.

This is the tool's central question — *are these real?* — answered in one picture, and it needs no
statistical training to read. The existing "Who belongs" chart carries the same information as
sorted bars, which is harder to read at a glance. Low effort once (1) is done, since it is the
same numbers plotted differently.

---

## Worth doing, bigger

### 6. Mixed numeric and categorical answers, with ordinal treated as ordinal

**Source:** Szepannek et al., *Clustering large mixed-type data with ordinal variables*, Advances
in Data Analysis and Classification (2024) — a k-prototypes variant using Gower's distance, with
proven convergence and explicit handling of ordinal variables.

> **Correction, after implementing it.** "Explicit handling of ordinal variables" above hid a
> choice this note did not know it was making. Podani (1999) extended Gower to ordinal data in
> **two** forms, and the widely cited one — the tie-corrected non-metric version, his eq. 2b — is
> the wrong one for survey data. It subtracts the within-tie spread from every gap, so with five
> levels and hundreds of respondents adjacent answers collapse to nearly zero distance while the
> extremes stay a full 1 apart: for three equally-used levels, d(1,3) = 1.00 against
> d(1,2) + d(2,3) = 0.065. That breaks the triangle inequality badly enough that "closest
> prototype" stops meaning anything. The metric version (eq. 3, plain midranks over the rank
> range) is what `clustMixType` uses and what is implemented.
>
> The counterweight below also survived contact with data, in a more reassuring way than expected:
> measured on ratings alone, treating them as ordinal rather than as numbers costs about 0.02 ARI
> — inside run-to-run noise, and on data generated as continuous-then-rounded, which favours the
> numeric reading by construction. So the ordinal treatment is close to free, and the real cost of
> the mixed path is the ordinary one of including questions that separate nobody.

Today a survey with both rating scales and pick-any questions must go down the k-means path or the
latent-class path; it cannot use both kinds of answer in one model. This is still the largest real
capability gap, and 2024 gives a specific, published answer rather than a hand-rolled compromise.

Worth noting the honest counterweight: the same search turned up active debate on whether Likert
answers should be treated as Euclidean at all. The tool's current position — treat them as
numeric, range-standardised — is defensible and widely practised, and Steinley's replication of
Milligan & Cooper backs the standardisation choice specifically for k-means. This would be a real
piece of work, not a switch to flip.

---

## Tried and rejected on the evidence

- **Sparse k-means** (Witten & Tibshirani 2010) for variable weighting. Built, measured, not
  adopted: it inflates the silhouette on pure noise from 0.12 to 0.39, its weights concentrate
  identically on real structure and on noise, and it does not beat the eta-squared already
  reported. See `STATE-OF-THE-ART.md` and `references/sparse_kmeans.py`.

## Deliberately not proposing

- **Kernel, fuzzy, possibilistic, intuitionistic, metaheuristic and deep variants.** Roughly half
  the library. Each adds a parameter nobody in the room can defend and a result nobody can
  reproduce by hand. Nothing found suggests they would change a decision for a 400-respondent
  Likert survey.
- **Replacing k-means++.** Five of the seven initialisation papers in the library never mention
  it; they benchmark against plain random or 1990s methods. No evidence to act on.
- **A second cluster-tendency test.** Measured: Hopkins reads 0.56 on five-question noise and 0.95
  on real structure. It works. The failure is specific to two-question surveys, which the code
  already warns about.

---

## Corroborated while reading — no action needed

| Source | What it says | Tool |
|---|---|---|
| Steinley 2006 §3.4.1 | Milligan & Cooper's range standardisation, **replicated by Steinley 2004a for k-means specifically** | Range is the default |
| Steinley 2006 §3.1 | Local optima "run into the thousands"; few restarts mislead | 50 restarts, and reports the share that reached the best |
| Burkov §9.2.3 | Prediction strength; "largest k such that ps(k) is above 0.8"; average over runs | `ps_cutoff = 0.80`, `ps_splits = 10`, reports the SD |
| ISL §12.4.3 | Vary the parameters, cluster subsets, report as hypothesis not truth | k-means + GMM + Ward agreement; split-half; bootstrap; consensus |
| El Khattabi et al. 2024 | Silhouette and Calinski-Harabasz are the most sensitive to data shape | Both weighted below prediction strength |
| Géron ch. 9 | "does not behave very well when the clusters have varying sizes, different densities, or nonspherical shapes" | Now caps confidence when GMM/Ward disagree |

One line from Géron worth keeping in mind: *"The solution on the right is just terrible, even
though its inertia is lower."* A better objective value is not a better answer.

---

## Sources

- Dolnicar, Grün & Leisch, *Market Segmentation Analysis*, Springer 2018 — [open access](https://library.oapen.org/handle/20.500.12657/23073) · [companion site](https://statistik.boku.ac.at/nachlass_leisch/MSA/)
- Leisch, *Neighborhood Graphs, Stripes and Shadow Plots* — [PDF](https://www.stat.cmu.edu/~rnugent/EDM2014/LeischStripes.pdf)
- Dolnicar & Leisch, *Using segment level stability to select target segments* (2017) — [record](https://forschung.boku.ac.at/en/publications/110644) · `flexclust` [reference](https://rdrr.io/cran/flexclust/man/slswFlexclust.html)
- Szepannek et al., *Clustering large mixed-type data with ordinal variables*, ADAC 2024 — [Springer](https://link.springer.com/article/10.1007/s11634-024-00595-5)
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*; James et al., *An Introduction to Statistical Learning*; MacKay, *Information Theory, Inference, and Learning Algorithms*; Steinley (2006); Burkov; Géron — all in the local library
