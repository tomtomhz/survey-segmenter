# Changelog

Notable changes to Survey Segmenter. Versions follow [semantic versioning](https://semver.org/);
the version is set in `pyproject.toml` and stamped into every report footer.

## [1.4.1] — 2026-08-05

### The remaining two charts became interactive

All six now answer questions, rather than four of them — which was an inconsistency a reader
notices immediately.

- **What differs** settles what the static dot plot raises and cannot answer: two marks that nearly
  touch, is that a real difference? Pointing at one reports the group, its value and the spread
  across the whole row, and says outright when a question separates nobody.
- **Do they separate** turns "looks a bit bimodal" into a number — how many respondents sit in a
  band, and what share of the study that is.

### Fixed

- Interactive charts took their segment colours from a stylesheet the *static* renderer injects
  into a different element. It worked, by coincidence of a shared CSS class, and would have broken
  silently the day the static drawing was dropped for charts that have a spec. Both colour steps
  already travel in the spec, so each chart now asks for the one matching the page. The palette
  still has a single home in `charts.py`.

## [1.4.0] — 2026-08-05

### The charts can be asked questions

Four of the six charts are now interactive on screen, while the same picture as before goes into
the printed report, the PDF and the image Claude reads.

- **The segment map** reports which group a mark belongs to and **how many people share that exact
  set of answers** — the count the static chart can only encode as the size of a dot.
- **The full grid** reports the second number behind each cell: how far that group sits from the
  question's own average, which is what the colour encodes and what the printed value cannot say.
- **Who belongs** reports how many people in a group fall in a given range of fit. The low bands
  are the people who could belong to either group, so this is the number to filter a list on.
- **How many groups** reports every quality measure at once for a candidate, instead of asking you
  to trace three lines back to an axis.

Everything reachable by pointer is reachable by keyboard, the segment map offers its numbers as a
table, and the full grid is a real table so a screen reader can simply walk it.

**How this is built, because it matters for what comes next.** A chart is computed once, in Python,
and drawn twice. The computation emits a *spec* — the numbers behind the picture — and both
matplotlib and the browser render from it. The straightforward alternative is a second chart engine
in TypeScript, which then quietly disagrees with the first the day somebody edits one of them. A
test asserts the spec matches what was drawn.

A chart with too many marks to stay responsive ships no spec and shows the static drawing instead;
so does a project saved before this release. There is always a correct picture.

### Fixed

- Past eight groups, two segments could be drawn identically — colour and shape both cycled on
  eight, so group 8 matched group 0 in both.
- A question worded like a colour code came out garbled in the chart.
- The test suite was making real network requests, which failed harmlessly and filled every run
  with connection errors that looked like a broken build.

## [1.3.0] — 2026-08-05

### The charts show every respondent, and colour means one thing at a time

- **Every respondent is on the segment map.** It drew a random 1,200 before. Worse and invisible
  to the reader: rating answers come in whole steps, so people who answered identically land on
  exactly the same spot and stack. Measured — 3,000 people on five 1-5 questions occupy 422
  distinct positions, so a plain scatter was showing **14% of the data**. Each dot is now drawn
  once per distinct answer pattern with its area proportional to how many share it, plus a size
  key. No jitter: it would place points where no respondent sits.
- **The per-person fit chart covers everyone too**, and became one distribution per segment rather
  than every respondent stacked into one column of sub-pixel bars. Which group is weakest is now
  the first thing you read.
- **Colour does one job per chart.** Orange used to mean "Group 1" on four charts, "Separation
  (silhouette)" on the choice-of-k chart, and "above average" on the heatmap. Identity, polarity,
  magnitude and chrome now have separate encodings.
- **The identity palette was replaced because it failed a check.** Three hues sat below the chroma
  floor and read as grey, and the worst adjacent pair — Group 0 against Group 1 — measured CVD
  ΔE 7.9. Segments also carry a **marker shape**, which is required rather than decorative: on a
  scatter every pair of hues is visible at once, and at eight groups the worst pair measured ΔE 3.2
  under protanopia and 7.1 for normal vision.
- **Charts carry their own light and dark steps** as CSS variables, so a downloaded or printed
  chart keeps them.
- **The radar chart is gone.** It encoded value as distance from a centre, so the eye read enclosed
  area — which grows with the square of the values and changes entirely when the questions are
  reordered. Its replacement is a Cleveland dot plot, which also fixed bars starting at zero on a
  1-5 scale.
- The heatmap labels every cell, the choice-of-k chart leads with the criterion that decides, and
  the charts are ordered the way they should be read: are these groups real, then what is in them.

### Fixed

- Two segments could be drawn identically past eight groups — colour and shape both cycled on
  eight, so group 8 matched group 0 in both channels.
- A question worded like a colour ("#2a78d6 is my favourite") was rewritten into a theme token in
  the chart, because the colour swap could not tell data from markup.
- The gorge chart had no entry in the interface's tab labels, so its tab showed a raw id.

## [1.2.0] — 2026-08-01

### Surveys that mix rating scales with pick-any questions

- A questionnaire holding both kinds used to have the multiple-choice columns set aside with an
  apology and the groups built on the ratings alone. On a study where the brand question is the
  interesting one, that threw away the finding.
- `kprototypes.py` implements Gower k-prototypes (Szepannek, Aschenbruck & Wilhelm, *Advances in
  Data Analysis and Classification*, 2024). Ratings are read as **ordered** answers rather than as
  numbers; pick-any answers count by whether two people chose the same thing.
- The report now names the answer each segment actually picks — "Mostly picks Nespresso (69% of
  them, vs 29% overall)" — instead of averaging brand codes, and scores pick-any questions by
  Cramér's V squared rather than eta-squared, which depends on the order the answers were listed.
- Measured over nine analyses: recovers the right number at high confidence when both kinds of
  question carry signal, and drops to moderate or low rather than claiming a result when either
  half is noise.

### Fixed

- Two standard Swedish five-point wordings were unrecognised, and one unmapped phrase fails a whole
  column — so a Swedish survey silently lost the ordering it was measuring.
- The empty-cluster refill could oscillate to `max_iter` on data with fewer distinct answer
  patterns than groups requested.

## [1.1.0] — 2026-07-31

### Charts are drawn by matplotlib

- The six charts are rendered with matplotlib instead of 684 lines of Python f-strings
  concatenating SVG path data by hand. **scikit-learn computes, matplotlib draws** — the
  computation never moved; `KMeans`, `GaussianMixture` and the silhouette statistics were always
  doing the maths. The packaged app grows from about 76 MB to 82 MB.
- The segment map now shades the decision regions behind the dots, so the failure it exists to
  expose — one dense cloud sliced into pie wedges — is unmistakable rather than inferred.
- Charts still emit vector SVG that adapts to the light and dark grounds, and now also render to
  PNG, which is what lets Claude see them.

### Claude reads the charts

- The segment map, the profile bars and the full grid go to Claude as images alongside the text
  digest, so its interpretation is based on the same picture the reader is looking at.
- The privacy test widened with the payload: it now builds the real request and checks the PNG
  bytes as well as the text, since an image could carry a label and a PNG can carry metadata.

### Desktop builds

- Windows and macOS apps are built and smoke-tested by a CI workflow. The Windows gap was never
  a missing machine — GitHub has had the runners all along.
- The first Windows build found a bug that would have broken the app for every Windows user: the
  confidence line prints a coloured circle, the legacy console encoding cannot encode it, and the
  resulting error surfaced as "something went wrong reading that file" on every analysis. The
  same fault would have hit any Swedish column name.
- The first packaged macOS build drew no charts at all: matplotlib resolves output writers lazily,
  so the SVG backend was never bundled.

### The interface is now a real application

- **Rebuilt in React 19 + TypeScript**, built with Vite, in `frontend/`. It replaces 640 lines of
  CSS, markup and hand-written DOM manipulation that lived inside three string constants in
  `segment_kmeans.py`, which had no type checking, no component boundaries and no tests of its own.
- **Every server response is typed** against what the Python handlers actually return, so a change
  on one side that is not reflected on the other fails the build rather than a user's afternoon.
- **38 frontend tests** (Vitest + Testing Library) covering what the Python suite could only grep
  for: that a dropped connection or a non-JSON reply never wedges the page, that unreadable files
  are refused before upload, that a dropped folder is explained, and that the result blocks behave.
- The design is unchanged — it was not the problem.

### Fixed — audit, second pass

- **Save PDF was dropping the full statistical report.** The report, the group names and the
  column picker are collapsed `<details>`, and a browser does not paint the contents of one — so
  the print rule `.card[open] .rep` only ever applied to panels that were already open, and the
  PDF someone circulated was missing the statistics it was supposed to carry. Every panel is now
  opened before the snapshot and put back exactly as the reader had it.
- **Enter sent half-finished words** for anyone using an input method that uses Enter to accept a
  candidate — Japanese, Chinese, Korean, and several European accent composers.
- **A request that never came back left a spinner nobody could cancel.** There is now a fifteen
  minute ceiling — generous, because clustering 17,000 people legitimately takes about a minute —
  after which it becomes an error message like any other.
- **The chart tabs claimed `role="tablist"` without behaving like one.** Arrow keys, Home and End
  now move between the six charts, with a roving tabindex and each panel tied to its tab; before,
  the only way through by keyboard was Tab, which walked out of the group entirely.
- **Naming the groups left the download links stale.** The server has always returned the
  refreshed file list, and the interface ignored it — the name panel compensated by printing two
  hardcoded links of its own. There is one file list now, and it updates.
- **Closing the settings dialog dropped focus to the top of the document**, stranding anyone
  navigating by keyboard.
- **The conversation was not announced.** Replies arrive with no focus change, so a screen reader
  was never told the answer had come back; the thread is a polite live region now.

### Fixed — audit

- **Any path beginning `/quit` shut the app down.** Routes were matched by prefix, so
  `/quitting-time` reached the shutdown handler and `/projects-of-mine` was answered with the
  project list. `/project` only avoided swallowing `/projects` because of the order the branches
  happened to be written in. Routes are matched exactly now.
- **The HTML shell was cached for a year** whenever it was reached by anything other than exactly
  `/` or `/index.html` — including `/?utm_source=x` and every single-page fallback route. A stale
  shell loads asset hashes that no longer exist: a blank page that survives reloading, which is
  what `no-store` is there to prevent.
- **Error text from the server was injected as markup** on the chat failure path. The raw
  exception can carry text out of the uploaded file, so a column heading is attacker-controlled
  the moment someone analyses a spreadsheet a third party sent them.
- **Any render error blanked the whole application.** React unmounts the entire tree when a
  component throws. There is now a boundary around the app and one around each result card.
- **The settings form accepted keys it would then ignore** when `ANTHROPIC_API_KEY` was set.
- **The thread yanked the reader to the bottom** on every new message, mid-report.
- **Deleting a project was one stray click away**, with no undo and the original upload going
  with it. It arms, then asks.
- **The settings dialog ignored Escape** and did not announce itself as a dialog; two controls
  were spans with `role="button"` — focusable but not operable from the keyboard.
- **The busy guard is a ref rather than state**, and the global `unhandledrejection`/`error`
  safety net from the previous interface is back, so a stuck spinner cannot strand anyone.

### Fixed

- **The opening Claude interpretation never ran.** The handler closed over `sessionId` from before
  the analysis that had just set it, so it returned immediately and the automatic read-through of a
  fresh result silently did not happen. Found by the new tests.
- **The composer opened four lines tall, and briefly sat below the fold entirely.** Two separate
  layout faults: the stylesheet made `<body>` the fixed-height flex column while React mounts into
  a `#root` div in between, and measuring a flex item's `scrollHeight` after setting `height:auto`
  returns the container's height rather than the content's.
- **Two stale README claims.** MaxDiff hierarchical Bayes estimation now exists, and the licence is
  MIT rather than absent — both were still described as missing.

### MaxDiff scoring (Hierarchical Bayes)

- Drop a tidy best-worst export (`respondent_id | set | item | choice`) in and it is detected,
  scored, and segmented on individual-level utilities — the input a MaxDiff instrument actually
  calls for, and the gap that previously blocked the Stockholm-Cluster survey.
- Estimation is Hierarchical Bayes in pure numpy: a sequential best-then-worst multinomial logit
  with `b_i ~ MVN(mu, Sigma)`, sampled by Gibbs plus a vectorised Metropolis-Hastings step.
- Measured against known utilities on simulated data, HB recovers *individual* utilities markedly
  better than counting (0.76 → 0.92 correlation at strong separation, 0.67 → 0.84 at moderate).
  Its effect on the segmentation itself is small but consistent. Neither method rescues genuinely
  weak structure — see `docs/HANDOVER.md`, which reports the full sweep including where HB does not
  help. No real MaxDiff responses have been through it yet.

### Charts

- Two more views, six in total. **Compare groups** overlays each segment as a radar outline, so
  the shape of a segment can be read at a glance rather than reconstructed from bars. **Full
  grid** is a heatmap of every question against every segment, diverging around each question's
  own mean.
- The full grid exists because the bar chart stops at nine questions to stay legible — on a
  fifteen-item block that hid a third of the study behind a download link. The bars now say
  where the rest went, and the answer is one tab away.

### Fixed

- **The confidence rating could still read "Moderate" on noise.** The amber band was reachable on
  bootstrap Jaccard alone; split-half replication now gates it too.
- **A broken chart withheld the others.** They were built as one eagerly-evaluated tuple, so a
  single NaN centroid discarded all of them, segment map included. Each is now isolated.
- **The tool could recommend a number of groups it could not support** — either too small to
  target or more groups than there were distinct answer patterns. Unviable k values are now
  filtered before the vote rather than warned about after it.
- **Hopkins read high on structureless data.** Duplicate answer patterns inflate it (0.78 on pure
  noise with two Likert questions); it is now caveated in place rather than silently trusted.
- **A privacy assertion had stopped testing anything** — a substring check became a list-membership
  check when the payload became a list, and passed regardless. Fixed and confirmed to fail
  against a planted identifier.
- **A session evicted from memory 404'd** instead of rehydrating from disk.
- **The Claude layer had no degradation path.** It now steps down through three request shapes,
  because not every account has the fallbacks beta and not every SDK knows the parameter.

## [1.0.0] — 2026-07-30

First release put under version control. Everything below was built and verified before this
point; it is recorded here so the starting state is documented rather than assumed.

### Analysis engine

- Three methods with automatic selection: k-means, Gaussian mixture, and true Latent Class
  Analysis (EM) for categorical data.
- Stability-first validation rather than a single elbow: Hopkins cluster tendency, prediction
  strength (Tibshirani & Walther), bootstrap replication ARI, per-segment Jaccard (Hennig),
  Monti consensus + PAC, the gap statistic, and cross-checks against Gaussian mixture and Ward.
- A typing tool: an exportable rule that assigns new people to existing segments, with
  cross-validated accuracy reported.
- Automatic survey handling — finds the id column, recodes Likert text to numbers, sets aside
  demographics to profile rather than cluster on, reads comma/semicolon files, UTF-8 and
  Latin-1, Swedish characters, `.xlsx`, and Google Forms "select all that apply" columns.

### Charts — "see the data yourself"

- Four inline-SVG charts on every result: the segment map (PCA scatter of respondents coloured
  by group, stating what share of the variation the view carries), per-person fit, quality
  across every number of groups, and what differs between groups.
- Drawn without matplotlib, so the packaged app stays small, the build stays reliable, and the
  charts print, embed in the standalone report, and follow light/dark themes.
- Colourblind-safe palette (Okabe-Ito).

### Interface

- A local web app that looks and works like Claude: drag a file anywhere, get segments, ask
  follow-up questions. Runs on `127.0.0.1`; nothing is uploaded.
- Projects persist across restarts, with a sidebar history.
- Optional Claude interpretation using the user's own Anthropic API key.
- Packaged as a signed, double-clickable macOS `.app` needing no Python.

### Privacy

- All analysis is local. See [PRIVACY.md](docs/PRIVACY.md).
- Only an aggregate digest is ever transmitted, and only when an API key is configured. This is
  enforced by a test that fails the build if any respondent identifier or free-text answer
  reaches the payload.

### Fixed before release

- **Confidence rating could read "Moderate" on pure noise.** The amber band checked only
  bootstrap Jaccard, which sits near 0.7 even on random data, and ignored split-half
  replication (0.06 on the same data). Split-half now gates amber, so structureless data
  correctly reports Low.
- **Row counters were clustered on.** Columns like `City_n` or a second `user_id` were read as
  ratings and injected a fake gradient, measurably degrading results.
- **Swedish column names were shredded.** The tokeniser was `[a-z]+`, so `Kön` became
  `['k','n']` and Swedish demographics were clustered on instead of profiled.
- **`--classify` crashed on text Likert answers**, breaking the scoring path on real exports.
- **Latent class analysis could degenerate** into one category per person and still report high
  confidence.
- **The packaged app could ship broken.** PyInstaller exits 0 even when it omits the entry
  point's own module; the build now wipes its cache and smoke-tests the signed binary,
  discarding the archive if the app cannot analyse a survey.
- **The macOS build's code signature broke on download**, giving recipients "the app is
  damaged". Signing now happens on a clean copy and is verified inside the shipped archive.
