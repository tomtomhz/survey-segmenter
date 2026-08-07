# Changelog

Notable changes to Survey Segmenter. Versions follow [semantic versioning](https://semver.org/);
the version is set in `pyproject.toml` and stamped into every report footer.

## [1.5.7] — 2026-08-07

From a full audit run against the built app rather than the source tree.

### Fixed

- **A survey made entirely of multiple-choice questions was getting almost nothing to judge the
  result by.** Measured against the ratings report: two of eleven pieces of evidence. It now also
  reports split-half replication (would a fresh sample find the same classes), which of the three
  kinds of segmentation this is, which classes sit next to each other, and how to read the
  per-person fit column. The neighbours table was **already being computed and then discarded**.

- **A "no answer" code in a follow-up file was scored silently.** Exports routinely write 99, 999
  or −99, and nothing rejected them: the value is scaled with the study's own parameters, lands far
  outside the space, and drags the respondent to whichever segment is extreme on that item.
  Measured on a 250-person follow-up with 99 in one question, **35 of the 60 affected people were
  put in the wrong segment**, and agreement with the truth fell from 0.967 to 0.593.

  The scored file now carries `answers_off_the_original_scale` per respondent and the command line
  says how many were affected. Counted, not corrected — whether 99 means "no answer" is a fact
  about your questionnaire, not about the data. A skipped answer is not flagged; a blank is a blank.

- **The report could recommend a number of groups below the cutoff it quotes as decisive** without
  saying so. It calls prediction strength "the column to trust most" and quotes 0.80; on one file
  it chose a solution scoring 0.74 while another scored a perfect 1.00. The answer is unchanged —
  both readings were defensible there — but the disagreement is now stated, and the number of
  groups that does clear the line is named.

### Verified, not changed

`.xlsx` reading, nine awkward file shapes, survey weights, eight concurrent analyses, bad-upload
error messages, MaxDiff utility recovery, the downloadable files against the report on screen, and
the AI add-on with no key configured — all checked against the packaged binary and all correct.

## [1.5.6] — 2026-08-07

### Fixed

- **Every packaged release has been producing reports with no tables in them.** Segment sizes, the
  per-segment stability checks, the centroids and the whole k-selection panel arrived as
  run-together text rather than tables — in the app, for as long as there has been an app.

  Every table is rendered by `DataFrame.to_markdown`, which needs `tabulate`. pandas imports it
  lazily, from inside that call, so PyInstaller's static analysis never saw it and never bundled
  it. The same shape as the chart backend that went missing once before: a lazy import is
  invisible until something runs the code path. It was also declared an *optional* extra described
  as "prettier Markdown tables", which is why nothing looked wrong — the description was incorrect,
  since without it there are no tables at all.

  `tabulate` is now a core dependency and is collected explicitly into the build. **Measured on the
  packaged binary: 0 tables before, 8 after.**

- The packaged smoke test now fails the build if the report comes back without tables. It was
  found only because a CI run went red on an unrelated assertion; nothing in the app itself
  complained, and the source tree has `tabulate`, so every test passed on every machine that
  mattered.

## [1.5.5] — 2026-08-06

**Read this one if you have run a segmentation on 1.5.4 or earlier.** The numbers were right; some
of what the report said about them was not, and in one case it pointed a budget at the wrong place.

### Fixed — the report contradicting its own evidence

- **A real segment was being condemned.** Each segment was checked against the solutions with one
  group fewer and one more, and the *weaker* of the two was reported. Asking for one more group
  forces the analysis to split something, so whichever segment gets subdivided scores about 0.5 in
  that direction whether or not it is genuine. On a 420-person study whose three mind-sets were
  recovered almost perfectly, the largest and cleanest segment held together perfectly when groups
  were merged, scored 0.56 when split, and the report said: *"Do not build a campaign on the ones
  marked 'dissolves'."*

  The two directions are now separate, because only one is evidence. Members scattering when
  groups **merge** means the segment was never a unit. Dividing when asked for **more** groups
  means it contains sub-groups — useful, and now reported as an opportunity.

- **"Start with the biggest, most distinct group"** — nothing in the analysis tests whether the
  biggest is the most distinct, and on that same study it was the segment the table below
  condemned. The summary now sends you to the stability tables and says the largest is not always
  the soundest.

- **A 40% share was described as "about 1 in 3 (40%)"** — a claim and its own contradiction inside
  one set of brackets. The fraction wording is only used when it is accurate.

- **The green light claimed the groups "are clear"** five lines above a cluster-tendency score
  described as "essentially random". The light is built from stability numbers and now says only
  what it measured: the same groups come back when the analysis is repeated.

- **Segment names were unreadable.** "planning things rather + want meet people outside" —
  stopwords stripped and cut mid-phrase. They are placeholders your team is meant to replace, so
  they now read like the question they came from.

- **The stability tables could not be matched to the groups.** The sizes table and the decisive
  per-segment stability table used "Segment 0/1/2" while everything else used names. Both now
  carry the name beside the number, which is what the exported CSV uses.

### Fixed — large studies

Driven by the first real files above 5,000 respondents: 41,188 telephone-survey responses, 48,842
census records, and 581,012 rows.

- **A study of 48,842 people needed 11 GB.** One line mapped each answer to its nearest known
  level by comparing it against every level at once — nothing on a five-point scale, and 19 GB per
  column once a continuous measurement had been typed as an ordinal one with 48,842 distinct
  values. Now a binary search: same answer, **11.04 GB → 1.59 GB and 2.2 min → 0.9 min**. A
  581,012-row file runs in 3.2 minutes.
- **The consensus criterion silently vanished above 5,000 respondents** — one of the three the
  panel weights double, absent exactly where a second opinion is worth most. It is now estimated
  from a random sample and the report says so, naming the sample size and the number of pairs.
- **A column flattened by its own outliers is now named.** A returned order of −80,995 against a
  median quantity of 3 put every one of 541,909 respondents inside 2% of the scale, so the answer
  described two outliers rather than half a million people. Stated, with `--scaling robust` named
  as the remedy, rather than silently corrected.
- Anything needing a distance between every pair of people is computed on a bounded sample above
  6,000 respondents, disclosed in the report. **The segmentation itself, and every person's group,
  always uses everybody.**

### Fixed — files the tool read wrongly

- **A title typed above the column names.** In Excel the title became the header; in CSV it broke
  the delimiter sniffer entirely. Both are now found.
- **UTF-16**, which is what Excel's "Unicode Text" export writes — the file arrived as a single
  column of mojibake.
- **Select-all answers packed with `;` or `|`** rather than a comma. A Swedish or German Excel uses
  the semicolon because the comma is the decimal mark, so `Spotify;Netflix` and `Netflix;Spotify`
  were different answers — five options became 74 pseudo-categories.
- **Identifiers treated as questions.** The limit on how many options a question may offer scaled
  with the number of respondents, so a 541,909-row file allowed 135,477 "options" and invoice
  numbers were clustered on. An answer list does not grow because you surveyed more people.

### Fixed — a wrong answer reported confidently

- **Fewer than two respondents per question is now never called high confidence.** With many
  questions and few people everybody ends up roughly equidistant, real structure dilutes, and what
  survives is highly reproducible *because noise reproduces*. On 150 respondents answering 400
  questions with a real three-group structure, the tool found two groups and called it High
  confidence. Being wrong is survivable; being wrong and confident is not.

### Added

- The suite now generates reports across five different kinds of data and **reads each one back
  against its own numbers**. Three defects had reached release with every test passing, all three
  plainly visible in the report, because every test asked whether the analysis ran rather than
  whether the document held together.

## [1.5.4] — 2026-08-06

### Fixed

- **The decimal-comma repair added in 1.5.3 did not run on newer pandas.** It asked whether a
  column's dtype was exactly `object`; pandas 3 gives text columns a `str` dtype, so every string
  column was skipped and the repair quietly stopped happening. The local pandas (2.3) still said
  `object` and the tests passed; CI on Python 3.11 and 3.12 caught it. It now asks whether the
  column is numeric, which is the actual question. The same assumption in the demographic
  profiler is fixed with it.

  The 1.5.3 **download is unaffected** — it bundles pandas 2.3, where the guard happened to hold.
  A build made on a machine with pandas 3, including any built by CI, would have silently dropped
  decimal-comma columns again.

## [1.5.3] — 2026-08-06

The first release driven by real data rather than data the tool generated for itself. Every
validation before this used synthetic answers drawn from the model k-means assumes; this one came
from feeding it five open survey datasets and the export formats real platforms write.

### Fixed — files the tool read wrongly without saying so

- **A Qualtrics export was analysed as if its rating scales were unordered categories.** Qualtrics
  writes three header rows — short name, question wording, and a JSON `{"ImportId": ...}` blob —
  and only the first became the header, so the other two arrived as respondents. A 240-person
  export read as 242 rows, the wording turned every rating column into text, and the survey was
  routed to latent class analysis. No error; the report looked ordinary. **SurveyMonkey** does the
  same with two rows. Both now read correctly, and all formats produce the identical segmentation.
- **A Swedish Excel export silently lost a question.** Swedish and German Excel write `4,5` for
  four-and-a-half; the `;` delimiter was already handled, the decimal comma was not, so a 0-10
  satisfaction score arrived as text and was dropped from the analysis without comment.
- **An education column was used to build the segments.** The demographic vocabulary held the
  Swedish `utbildning` and never the English `education`, so a 1-5 education code became a 26th
  "personality question" on the Big Five inventory. A numeric demographic cannot be told from a
  rating scale by its values — only the name gives it away — and that list now carries both
  languages for each concept.

### Added

- **A warning when a number is too large to be an answer.** On the Chilean plebiscite survey the
  tool reported eight segments; each was pure on how the person voted and then split in two by the
  size of their town (3,750 to 250,000). Four of the eight "mind-sets" were really "lives somewhere
  bigger". Columns on that magnitude are now flagged as possible facts about the person rather than
  answers they gave — flagged, not excluded, since that is a judgement about the study.

### Measured, and left alone

- **Response-style segments do not happen here.** The textbook failure — recovering how people use
  a scale instead of what they think — was tested with both truths planted and came out at ARI
  −0.002 against response style. The standard remedy (ipsative scaling) measures *worse* than what
  the tool already does, 0.819 against 0.977, and was deliberately not adopted.

## [1.5.2] — 2026-08-06

### Fixed

- **The tool could recommend the wrong number of segments and then report the real ones as
  noise.** Found by reading a report end to end on a 400-person file with three planted segments.
  It recommended two, called the result a *constructive* segmentation — the method inventing
  groups where there are none — and rated confidence Moderate. Three segments were sitting in the
  data: at k=3 the solution reproduced at 0.995 and predicted held-out respondents at 0.968,
  against 0.658 and 0.593 for the two it chose. It had recovered the answer and then argued
  itself out of it. Measured against the planted truth, the recovered segmentation scores 0.992
  where the old one scored 0.618.

  Two independent faults, either alone enough to lose it:

  - The **replication-stability signal** used "the largest k that clears the cutoff". That is
    Tibshirani & Walther's rule for prediction strength, where it is correct, and it had been
    carried across to stability, where it is not: 0.778 and 0.995 both clear 0.75, so a signal
    weighted double went to the visibly worse solution. Each k is now compared against the best
    using its own measured standard error, so a k that is genuinely less stable drops out while
    one that cannot be told apart stays in. This needed no tuning constant.
  - The **tie-break** then went to the smaller k on parsimony grounds, over a k holding *both*
    doubled criteria. Parsimony now breaks what those criteria leave level, rather than
    overruling them.

- **`--force-k` did nothing on the default path.** It was wired into each explicit `--method`
  path and omitted from the automatic one, which is the default. The run finished, the report
  never mentioned an override, and the number in it was the tool's own — so the reader believed
  they were looking at the answer they asked for.

- The plain-language line explaining the choice **counted a flat headcount of criteria while the
  decision was weighted**, so it reported "5 of them picked 2" for a k that both of the criteria
  this tool trusts most had argued against. It now reports the tally that actually decided, and
  the weights live in one place instead of two that can disagree.

- CI linted a **hand-written list of Python files** that had gone stale: `clusterability.py` was
  never in it and so was never linted. It now lints what is in the repository. That is the fourth
  time a hand-kept second list has broken something here — after two modules missing from the
  wheel and `diptest` missing from every CI-built app.

## [1.5.1] — 2026-08-06

### Fixed

- **A build made anywhere but this machine shipped without the second cluster-tendency test.**
  `build_app.py` carried its own copy of the dependency list, so a package added to
  `pyproject.toml` and not to that copy was simply absent — the app then started, analysed, drew
  its charts and quietly skipped the check. Both CI runners were in exactly that state. The build
  now installs the project itself, so there is one list rather than two that can disagree.
  (The v1.5.0 archive attached to its release is unaffected: it was built on a machine that already
  had the package.)
- The packaged smoke test now **fails the build** if the dip test did not run, or if any chart
  arrives without the data behind it. Both of those degrade silently — the app starts and does
  less — which is the failure this project keeps meeting and the hardest kind to notice on a
  platform nobody is watching.

### Verified

- **Windows**, properly, for the first time: the actual built binary runs, the compiled dip
  extension works, and all six charts carry their data. Previously the Windows job proved only
  that a build completed.

## [1.5.0] — 2026-08-06

### A second opinion on whether there is anything to segment

The Hopkins statistic was the tool's only real answer to that question, and it has two measured
weaknesses. Adolfsson, Ackerman & Brownstein (*Pattern Recognition*, 2019) put numbers on both
across 35,000 simulated datasets: its power falls to **32%** when groups overlap rather than sit
apart, and it reads a handful of unusual respondents as a group of their own.

Overlapping segments merging into one is this tool's single measured failure mode, so a second test
earns its place exactly there. Hartigan's dip test now runs alongside it, and the report says how
the two line up — worth more than either alone, because they fail in opposite directions.

Measured on this tool's own data: three groups at heavy overlap, where Hopkins sinks to 0.66, are
still detected at p < 0.001; noise with five outliers, where Hopkins drifts over its own threshold
to 0.60, correctly returns p = 0.51.

**The form the literature recommends does not work on survey data**, and that is recorded so nobody
re-adds it. The published method runs the dip on all pairwise distances. Rating answers are whole
numbers, so those distances take very few values — 400 people on five questions produce 79,800
distances with **50 distinct values among them** — and the dip reads that comb as many modes,
returning p = 0.0000 on data with no groups whatsoever. What runs instead is the same paper's other
variant, the dip on the first principal component, which is a weighted sum and therefore continuous.

It needs at least four questions to be trustworthy and says so plainly when a survey is shorter,
rather than guessing.

### Fixed

- `clusterability` was missing from the installed package, the same defect that shipped once
  before with another module. The manifest is now checked by a test rather than maintained by
  memory.

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
