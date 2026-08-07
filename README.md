# Survey Segmenter

[![CI](https://github.com/tomtomhz/survey-segmenter/actions/workflows/ci.yml/badge.svg)](https://github.com/tomtomhz/survey-segmenter/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

**Turn a survey export into customer segments you can defend.** It finds the groups, tells you
how much to trust them, draws the data so you can check the answer yourself, and gives you the
files to act on. Everything runs on your own machine — see [PRIVACY.md](docs/PRIVACY.md).

| | |
|---|---|
| **Non-technical users** | Download the app, double-click, drag your file in. [Setup guide](docs/USING-THE-APP.md). |
| **Analysts** | `pip install ".[ai,excel]"` then `segment-kmeans --serve`. |
| **What is new** | [CHANGELOG.md](CHANGELOG.md) |
| **Taking this over** | [ONBOARDING.md](docs/ONBOARDING.md) — scope, the reasoning behind each decision, measured vs asserted, and the mistakes already made here |
| **Current state** | [HANDOVER.md](docs/HANDOVER.md) — engineering state of play, known limits |

---

## Where everything is

| | |
|---|---|
| `segment_kmeans.py` | The engine: preparation, clustering, validation, the report, and the typing rule |
| `webapp.py` | The local web application: uploads, the API, saved projects. Imports the engine, never the other way round |
| `charts.py` | Drawing only, with matplotlib. **scikit-learn computes, matplotlib draws** — nothing here decides anything about the segmentation |
| `maxdiff.py` | Hierarchical Bayes scoring for best-worst (MaxDiff) exports |
| `ai_interpret.py` | The optional Claude layer. Sends the aggregate report and the charts, never a respondent row |
| `frontend/` | The interface: React + TypeScript, built with Vite |
| `webui/` | The **compiled** interface that Python serves. Generated — see `webui/README.md` |
| `tests/` | `pytest` from the repository root |
| `docs/` | [Onboarding](docs/ONBOARDING.md) · [Using the app](docs/USING-THE-APP.md) · [Handover](docs/HANDOVER.md) · [Privacy](docs/PRIVACY.md) |
| `run_app.py` / `build_app.py` | Start the app · build and smoke-test the desktop app |

## The method

A professional, tested k-means segmentation tool in the tradition of Howard Moskowitz's Mind
Genomics: it clusters **respondents on their preferences** (MaxDiff or conjoint utilities,
attitude or importance scores) to find distinct mind-sets, then selects the number of segments
and validates the solution the way the methodological literature says it must be done.

The guiding principle, from Dolnicar & Leisch: **data-driven segments are usually constructed by
the algorithm, not discovered in nature.** So the burden of proof is on stability and
reproducibility, not on a nice-looking elbow — and this tool is built to meet that burden, and
to tell you honestly when the data contain no real segments at all.

The engine is expert-grade, but you do not have to be an expert to use it. See the quick start.

## Quick start (no expertise needed)

One-time setup (ask a colleague if you have never used a terminal): install Python 3, then, from
this folder, run `pip install .` once. That installs everything and gives you a `segment-kmeans`
command you can run from anywhere (add `pip install ".[excel,ai]"` for `.xlsx` files and Claude
interpretation). If you would rather not install it, `pip install -e .` puts the dependencies in
place without the command, and you run it as `python3 segment_kmeans.py`.

**The point-and-click way (no terminal after setup) — a Claude-style chat.** Start the built-in
web app once:

```bash
python3 segment_kmeans.py --serve
```

Your browser opens a chat page that looks and feels like Claude. Drag your survey file anywhere onto
it (or use the paperclip) — a CSV or `.xlsx` from Google Forms, Typeform, Qualtrics, SurveyMonkey, or
a spreadsheet — and it analyses immediately: how many groups, who they are, and a green / amber / red
confidence light, with the full statistical report one click away. Everything runs on your own
computer; your survey file is never uploaded. A colleague can leave this running so the whole team
just opens the page and drops in a file.

**Ask Claude about your segments (optional).** Add your own Anthropic API key in the page's
**Settings** (or set the `ANTHROPIC_API_KEY` environment variable) and the app hands the *aggregate*
result — group sizes, defining preferences, confidence, demographic profile, **never any individual
response** — to Claude (model `claude-opus-5`, via the official `anthropic` SDK). Claude writes a
plain-language readout and go-to-market recommendations, then answers your follow-up questions in the
chat ("which group first?", "draft a headline for group 2", "how much should I trust this?"). That
aggregate summary is the only thing the app ever sends anywhere; it goes only to Claude under your own
account, and nothing is sent without a key. Install the add-on with `pip install ".[ai]"` (it is
bundled in the desktop app). The statistics are complete with or without it.

**The one-command way.** If you are comfortable with a terminal, point it straight at a file:

```bash
python3 segment_kmeans.py my_survey.csv
```

Either way, the tool reads the file the way it actually arrives — comma **or semicolon** separated
(Swedish/European Excel exports), UTF-8 or Latin-1 (so å ä ö survive), and `.xlsx` Excel files
too. It finds the respondent id; ignores timestamps and free-text comments; sets aside background
traits like gender, age, and university so they **describe** the groups but never **form** them
(and profiles the groups by them afterwards); turns agree/disagree answers ("Strongly agree" ...)
into numbers; uses a survey-weight column, if you have one, to project the group sizes to the whole
population; chooses the right method for your data; picks how many groups and checks how much to
trust them; and produces a report you can read in a browser. It explains every choice, and if
something is wrong it tells you in plain language, not a stack trace.

Read the box at the top of the report: a **green / amber / red confidence light**, how many groups
it found, who they are, and what to do next. If you only read one thing, read that box.

**The download-and-double-click way (no Python, no terminal, nothing to install).** There is a
packaged desktop app for exactly this audience. See `docs/USING-THE-APP.md`. Build it with
`python3 build_app.py` (produces `dist/Survey Segmenter.app` on Mac, or a `Survey Segmenter` folder
with an `.exe` on Windows); zip the result and anyone on the team can download it, double-click, and
drop a survey file onto the page that opens. It is fully self-contained — recipients need no Python.

Everything below is for analysts who want to control the method, scaling, validation, and outputs.

## Install and run (for analysts)

```bash
pip install .            # or: pip install ".[excel,ai]" for .xlsx files and Claude

python3 segment_kmeans.py utilities.csv --id-col respondent_id --kmin 2 --kmax 8 \
    --scaling range --method kmeans --demographics demos.csv --outdir results

# model-based (finite-mixture / latent-class) instead of heuristic k-means:
python3 segment_kmeans.py utilities.csv --id-col respondent_id --method gmm --outdir results_gmm
```

**Four modelling paradigms, matched to your data.** `--method kmeans` (default) is the fast
heuristic for continuous utilities; `--method gmm` runs a Gaussian-mixture / latent-class model
(Wedel & Kamakura) that allows elliptical, unequal-size, overlapping segments, gives soft
assignment probabilities, and selects the number of components by BIC and ICL. Both of these treat
the inputs as **continuous**. For **categorical** inputs (agree/disagree, pick-any, multiple
choice) use `--method lca`, a true **Latent Class Analysis** — a finite mixture of categorical
distributions under local independence (Lazarsfeld-Goodman; Wedel & Kamakura), fit by EM, selected
by BIC and ICL, and validated by the same stability-first machinery.

For the common case of a questionnaire holding **both** — some 1-5 scales and some "pick one"
questions — `--method kproto` runs **Gower k-prototypes** (Szepannek, Aschenbruck & Wilhelm,
*ADAC* 2024), which uses every question at once instead of setting half of them aside. Ratings are
read as ordered answers rather than as numbers, pick-any answers by whether two people chose the
same thing, and each question scores 0-1 in its own natural way, so no exchange rate between them
has to be invented. See `kprototypes.py` for why its prototypes are medians and modes rather than
means, and `references/STATE-OF-THE-ART.md` for what it recovers and which checks it cannot run.

The auto-detected path — just point it at a CSV, or use the app — picks between the four for you.

```bash
# Latent Class Analysis for CATEGORICAL survey items:
python3 segment_kmeans.py answers.csv --id-col respondent_id --method lca --outdir results_lca

# Ratings AND pick-any questions in one model:
python3 segment_kmeans.py survey.csv --id-col respondent_id --method kproto --outdir results_mixed
```

For the continuous methods, the entire pipeline — stability, prediction strength, per-segment
Jaccard, typing tool, variable selection — runs under whichever you choose and cross-checks
against the other. The categorical `lca` path shares the stability-first philosophy (BIC/ICL,
bootstrap replication stability, per-class Jaccard) but is reported separately, since the
Euclidean diagnostics (silhouette, gap, Hopkins) do not apply to categorical data.

- `utilities.csv`: one row per respondent, one numeric column per item (see `segment_kmeans_EXAMPLE_input.csv`). An optional id column and any non-numeric columns are ignored for clustering.
- `demos.csv` (optional): respondent-level demographics for profiling. **Never used to form the segments** — only to describe them afterward.

As a library:

```python
from segment_kmeans import Segmenter, SegmentationConfig
seg = Segmenter(SegmentationConfig(scaling="range")).run("utilities.csv", id_col="id", outdir="results")
seg.recommended_k        # chosen number of segments
seg.assignments          # DataFrame: id, segment
seg.centroids            # segment x item mean utilities
seg.jaccard              # per-segment stability
seg.typing               # typing tool: cross-validated accuracy + exportable rule
seg.report_markdown      # the full written report

# classify new respondents from a saved rule:
import json
from segment_kmeans import classify_new
rule = json.load(open("results/typing_rule.json"))
classify_new(rule, pd.read_csv("new_respondents.csv"), id_col="id")   # -> id, segment, confidence
```

Check the version any time with `segment-kmeans --version`. When you pass an output folder the full
report is written there rather than dumped to the terminal, which keeps it clean for scripts.

Run the tests: `pytest` (92 tests, about a minute), which encode the
two headline guarantees (recovers real structure, rejects noise) for both the continuous and the
categorical (Latent Class) paths, all methods and scalings, the typing tool (it recovers held-out
respondents' segments and refuses to leak scaling across folds), the variable-selection check, a
battery of robustness cases, and — for the app — the chat web endpoints and the optional AI
interpretation layer (key handling, message building, and graceful behaviour with no key or SDK).

**Robustness.** The tool has been stress-tested against 28 adversarial inputs — tiny samples,
more items than respondents, constant and near-constant columns, scattered and whole-column
missing data, infinities, extreme outliers, duplicate rows, perfectly correlated columns,
structureless noise, a tiny imbalanced segment, high dimensionality, both methods, and n = 3,000 —
with zero failures: every valid input produces a sensible result and every degenerate input
(one row, a single item, all-constant data) raises a clear error rather than crashing or
returning garbage. It validates and clamps the search range to what the data support, treats
infinities as missing, and guards the O(n²) consensus matrix on very large samples.

## What it does, and why each step is there

| Step | What | Why (and who says so) |
|---|---|---|
| **Cluster tendency** | Hopkins statistic before clustering | Is there anything to segment? ~0.5 = random, >0.75 = real tendency (Lawson & Jurs 1990; Banerjee & Dave 2004) |
| **Scaling** | Range standardization by default | Recovers cluster structure better than z-scores (Milligan & Cooper 1988). z-score / raw / ipsative also offered |
| **Many restarts** | 50 restarts + local-optima diagnostic | k-means gets stuck in local optima; this matters (Steinley 2003, 2006) |
| **Number of segments** | A weighted panel, stability first | No single index is reliable; weight prediction strength and replication stability above fit |
| — prediction strength | largest k above 0.80 | cross-validated co-membership (Tibshirani & Walther 2005) |
| — replication stability | bootstrap Adjusted Rand Index | do the same segments re-emerge? (Dolnicar & Leisch) |
| — consensus PAC | Monti consensus clustering + Proportion of Ambiguous Clustering | how ambiguous is pair co-membership across resamples? lower = cleaner (Monti et al. 2003; Şenbabaoğlu et al. 2014) |
| — GMM BIC and ICL | model-based estimates of k | BIC = the principled criterion heuristic k-means lacks (Wedel & Kamakura); ICL adds an entropy penalty for overlapping components (Biernacki, Celeux & Govaert 2000) |
| — silhouette, Calinski-Harabasz, Davies-Bouldin, gap | separation indices | Calinski-Harabasz was the best single stopping rule (Milligan & Cooper 1985); gap = Tibshirani et al. 2001 |
| — elbow | inertia | shown but treated as the weakest signal |
| **Per-segment validity** | bootstrap Jaccard, per cluster | which segments are *real*: ≥0.85 highly stable, ≥0.75 valid, 0.6–0.75 a pattern, <0.5 dissolved (Hennig 2007) |
| **Model cross-check** | k-means vs Gaussian-mixture **and** Ward hierarchical agreement | three structurally different methods agreeing is strong evidence the structure is real, not one algorithm's artefact |
| **Consensus robustness** | Monti ensemble partition vs the main partition | a resampling-robust partition; `--consensus-final` adopts it |
| **Variable importance** | eta-squared per item | which items drive the segmentation and which are noise that masks it (Dolnicar's variable-selection work) |
| **Variable-selection check** | re-cluster without the near-noise items, compare stability | Dolnicar's point made operational: does dropping the noise items make the solution cleaner? A recommendation with the numbers, not a silent auto-drop |
| **Interpretation** | mind-set per segment | defining items, differentiating items (ANOVA F), auto-name |
| **Typing tool** | nearest-centroid rule + cross-validated accuracy | assign NEW respondents to segments, and measure how consistently that can be done (Mind Genomics builds one as standard) |
| **Profiling** | chi-square vs demographics, FDR-corrected | describe segments; never define them by demographics |

## Outputs (written to `--outdir`)

- `segmentation_report.html` — the same report as a self-contained web page anyone can open and share, opening with the plain-language summary and the green/amber/red confidence light. (Latent Class runs write `latent_class_report.html`.)
- `segmentation_report.md` — the full written report, with a "how to read this" guide and the methodology and citations.
- `segment_assignments.csv` — respondent id and segment.
- `segment_centroids.csv` — segment-by-item mean utilities.
- `k_selection_diagnostics.csv` — the full diagnostic panel for every k.
- `segment_stability_jaccard.csv` — per-segment bootstrap Jaccard stability.
- `variable_importance.csv` — eta-squared and role (driver / contributor / near-noise) per item.
- `typing_rule.json` — the portable **typing tool**: the scaling parameters and per-segment centroids needed to assign a brand-new respondent to a segment, plus the cross-validated typing accuracy. Apply it with `--classify` (see below).
- `run_manifest.json` — config, seed, timestamp, library versions, and headline results, so the run reproduces exactly.
- `diagnostics.png` — elbow, stability-and-prediction-strength, separation-and-BIC, and per-segment silhouette (if matplotlib is installed).

## How to read the result honestly

1. **Start with Hopkins.** If it is near 0.5, the data are essentially random; any segments are constructed, and you should be very cautious.
2. **Trust prediction strength and replication stability over the elbow.** A stable, replicable solution is what survives; a pretty elbow on unstable clusters is a mirage.
3. **Check the per-segment Jaccard.** A segment below 0.6 is not trustworthy as a distinct segment, whatever the overall fit says. If several dissolve, use a smaller k or a model-based method.
4. **Check the model agreement.** If k-means and the Gaussian mixture disagree sharply, the partition is method-dependent — read with caution.
5. **Read the variable-selection check.** The tool re-clusters without the near-noise items and shows whether stability improves. If it does, re-run on the signal items; if it does not, keeping them is defensible. (Turn it off with `--no-varsel`.)
6. **Rename the auto-suggested mind-set names** to something a non-analyst recognises before shipping.

## Typing new respondents (the operational payoff)

A segmentation is only useful if you can act on it, and acting on it means assigning **new**
respondents to the segments you found — long after the study closes. Every run therefore writes a
**typing tool** to `typing_rule.json`: the scaling parameters plus the per-segment centroids in
scaled space, which together classify a fresh respondent by nearest centroid (exactly how k-means
itself assigns a point). Apply it with no re-segmentation:

```bash
# type a new cohort using a previously saved rule
python3 segment_kmeans.py --classify new_respondents.csv --rule results/typing_rule.json \
    --id-col respondent_id --outdir results
# -> results/new_assignments.csv : respondent_id, segment, confidence
```

The `new_respondents.csv` needs the same item columns as the original study (extra columns are
ignored). `confidence` is the inverse-distance share on the winning centroid: near 1 means the
respondent sits on a segment centroid, near 1/k means they are near the boundary between segments.

The report also prints a **cross-validated typing accuracy**: train the rule on part of the
sample, and see how often it reproduces the segment of a held-out respondent (stratified k-fold,
with the scaling refit inside each fold so nothing leaks). Read it honestly:

- It measures how consistently you can **assign** new respondents, which is an operational
  property of the rule you are about to deploy.
- It is **not** a test of whether the segments are real. A k-means partition of pure noise is
  still highly classifiable, because its Voronoi cells are compact convex regions, so a high
  typing accuracy is necessary but not sufficient. Whether the structure is real is decided by
  the Hopkins statistic, prediction strength, and per-segment Jaccard. A **low** typing accuracy
  is informative on its own — the assignment boundary is not even geometrically stable.

For the `gmm` method the exported rule is a nearest-centroid approximation of the mixture's
posterior assignment; it is portable and needs no pickled model, at the cost of ignoring the
covariance shape when typing new points. The `lca` (categorical) path has the same typing tool,
demographics profiling, and weighted population sizes as the k-means path; its typing rule is the
latent-class classifier itself (class weights and class-conditional item probabilities), written
to `latent_class_typing_rule.json`, and `--classify` auto-detects which kind of rule it was given.

## Choosing the scaling (it changes the answer)

- `range` (default): divide each item by its range. Best cluster recovery in Milligan & Cooper's study; the recommended default.
- `standardize`: z-score each item. Familiar, but can under-recover structure.
- `robust`: center by the median and scale by the interquartile range. Use for **skewed or heavy-tailed** items, where a long tail would inflate the standard deviation (z-score) and compress the bulk of the data. **Important:** robust scaling protects the *scale estimate* from the tail, but it does **not** bound an extreme outlier's value, so a single huge outlier can still dominate k-means under `robust`. Counter-intuitively, `range` scaling is more resistant to a lone extreme outlier, because it maps every value (including the outlier) into [0, 1]. If you have a few extreme outliers, prefer `range`, or winsorize first. (Either way, the stability diagnostics will flag an outlier-driven "segment": watch for a tiny segment, a low split-half replication, and one lone item dominating variable importance.)
- `none`: raw utilities. Closest to classic Mind Genomics, where utilities are already comparable within respondent.
- `ipsative`: row-centre each respondent, to segment on the *shape* of preferences rather than their level.

## The honest limitation

k-means finds spherical, roughly equal-size clusters. For genuinely elongated, unequal-variance, or overlapping segments it is the wrong tool no matter how carefully it is validated. That is exactly why two alternatives are built in as first-class methods: run `--method gmm` for a Gaussian-mixture / latent-class model that allows elliptical, unequal-size, overlapping segments (continuous data), or `--method lca` for a true Latent Class Analysis when your inputs are categorical rather than continuous utilities. When the per-segment Jaccard says the k-means segments have dissolved, switch methods and compare.

The honest remaining boundary is *upstream*: this tool does not design the experimental stimuli — the part of a full Moskowitz/Sawtooth pipeline it still leaves to you.

It does now estimate individual-level utilities from raw MaxDiff choices. Drop a tidy best-worst export (`respondent_id | set | item | choice`) in and it is detected, scored by hierarchical Bayes (`maxdiff.py`), and segmented on the resulting utilities. Measured against known utilities on simulated data, HB recovers individual utilities markedly better than counting (0.76 → 0.92 correlation at strong separation); its effect on the segmentation itself is small but consistent, and neither method rescues genuinely weak structure. **It has not yet been run on real MaxDiff responses** — see `docs/HANDOVER.md` for the full sweep, including where HB does not help.

## Versioning, deployment, and licensing

- **Version.** The version is `segment_kmeans.__version__`, mirrored in `pyproject.toml` and listed per release in [CHANGELOG.md](CHANGELOG.md) — read it from there rather than from this line, which cannot be kept current. Every report footer and `run_manifest.json` records the version, the exact config, the random seed, and the library versions, so any run reproduces exactly.
- **Install as a tool.** `pip install .` puts a `segment-kmeans` command on the PATH; `pip install ".[excel]"` adds `.xlsx` support, `".[dev]"` adds the test runner. It runs on Python 3.9 or newer.
- **The web page is local only.** `--serve` binds to `127.0.0.1` (your own machine), never the public internet, caps uploads at 100 MB, and parses uploads with the standard library (no dependency on the removed `cgi` module), so it keeps working on current Python. For a shared internal deployment, put it behind your own authenticated reverse proxy rather than exposing the port.
- **The AI interpretation is opt-in and privacy-preserving.** It uses the official `anthropic` SDK with *your own* API key — read from `ANTHROPIC_API_KEY`, or saved locally to `~/.survey_segmenter/config.json` (owner-readable only) via the page's Settings. It transmits only the aggregate report (segment sizes, profiles, confidence, demographic percentages), never an individual respondent's row, and only when a key is present. It degrades cleanly to statistics-only when the SDK is missing or no key is set. The `anthropic` package is an optional dependency (`pip install ".[ai]"`) and is bundled in the desktop app.
- **License.** MIT, © 2026 Tom Hinderoth Zachrisson — see [LICENSE](LICENSE).

## How it is built

Two halves, deliberately separated.

| | |
|---|---|
| `segment_kmeans.py` | The analysis engine and a small HTTP API. Also the whole command-line tool — it runs standalone with no interface at all. |
| `maxdiff.py` | Hierarchical Bayes estimation of individual utilities from best-worst choices. |
| `ai_interpret.py` | The optional Claude layer. Sends only the aggregate digest, never a respondent row. |
| `frontend/` | The interface: React 19 + TypeScript (strict), built with Vite, tested with Vitest. |
| `webui/` | The compiled interface. Generated by `npm run build`; committed so the app runs without Node. |

Working on the interface:

```bash
cd frontend
npm install
npm run dev          # hot reload on :5173, API calls proxied to the Python app on :8000
npm test             # component and API-layer tests
npm run build        # compile into ../webui — commit the result
```

Run `python3 run_app.py` alongside `npm run dev` so the API is there to proxy to. CI rebuilds
`webui/` and fails if it has drifted from `frontend/`, so a change to the interface that was never
compiled cannot reach a release.

> **If npm is slow here, check whether the folder is cloud-synced.** `node_modules` is ~6,000 small
> files, and iCloud Drive (which syncs `~/Desktop` and `~/Documents` by default) turns every read
> into a sync operation — enough to take a test run from one second to over half an hour, and to
> leave `App 2.tsx` conflict copies behind. Keep the repository somewhere outside the synced
> folders, such as `~/dev/`.

## References

- Banerjee, A. & Dave, R. (2004). Validating clusters using the Hopkins statistic.
- Dolnicar, S. & Leisch, F. (2010, 2017). Stability of market segmentation; *Market Segmentation Analysis*.
- Hennig, C. (2007). Cluster-wise assessment of cluster stability. *Comp. Stat. & Data Analysis* (fpc::clusterboot).
- Lawson, R. & Jurs, P. (1990). The Hopkins statistic.
- Milligan, G. & Cooper, M. (1985) stopping rules; (1988) standardization of variables. *J. Classification*.
- Rousseeuw, P. (1987). Silhouettes. Calinski & Harabasz (1974). Davies & Bouldin (1979).
- Steinley, D. (2003). Local optima in k-means; (2006). K-means: a half-century synthesis. *Br. J. Math. Stat. Psych.*
- Tibshirani, R., Walther, G. & Hastie, T. (2001). Gap statistic. Tibshirani & Walther (2005). Prediction strength.
- Wedel, M. & Kamakura, W. (2000). *Market Segmentation: Conceptual and Methodological Foundations* (finite-mixture / latent-class segmentation; BIC).
- Lazarsfeld, P. & Henry, N. (1968); Goodman, L. (1974). Latent Class Analysis (finite mixtures of categorical variables under local independence).
- Biernacki, C., Celeux, G. & Govaert, G. (2000). Assessing a mixture model for clustering with the Integrated Completed Likelihood (ICL). *IEEE PAMI*.
- Monti, S., Tamayo, P., Mesirov, J. & Golub, T. (2003). Consensus Clustering. *Machine Learning*.
- Şenbabaoğlu, Y., Michailidis, G. & Li, J.Z. (2014). Critical limitations of consensus clustering in class discovery (the PAC score). *Scientific Reports*.
