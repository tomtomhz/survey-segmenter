# Handover — state of play

> **New to this project (human or assistant)? Read [ONBOARDING.md](ONBOARDING.md) first.**
> It carries the scope, the decisions that must not be silently reversed and why, what is
> measured versus merely asserted, and the traps this project has already fallen into.
> This file is the narrower engineering state of play.

Working state for whoever picks this up next, human or AI. `README.md` says what the tool is;
this says where it stands, what was decided and why, and what to do next.

**Last updated:** 2026-08-07 (end of day) · repo `github.com/tomtomhz/survey-segmenter` (public) · `main` @ **v1.6.1**, 202 Python + 111 frontend tests green locally on Python 3.9 **and** 3.12 · **GitHub Actions is not running — see below**

---

## Session state — 2026-08-07, end of day (read this first)

**Released today: v1.5.1 → v1.6.1.** The team copy at `~/Desktop/Survey Segmenter (app for the
team)/` is on **1.6.1** — verified by asking the shipped binary itself, over HTTP, which version it
stamps into a report, not by trusting the source tree. Tree clean.

### GitHub Actions has not run since 15:24 today — billing, not code

Every run from `Re-measure the central accuracy claim` (15:33) onward failed **before starting a
single step**: while the repo was private, Actions minutes were metered against a monthly
allowance, running the full matrix on every push exhausted August's allowance in seven days, and
GitHub then halts jobs rather than billing.

So **nothing since v1.5.8 has been tested by CI**, and no Windows build artifact exists for it. This was
recorded as "CI green" for several hours on the strength of a stale reading; the lesson is that a
green memory is not a green run, and `gh run list` costs one command. The workflow has since been
trimmed so an ordinary push runs two jobs rather than five — see the header of `ci.yml`.

What was done instead, locally, and what it does and does not cover:

| | Covered |
|---|---|
| Python 3.9.6 (oldest in the CI matrix) | 202 passed |
| Python 3.12.13, numpy 2.5.1, pandas 3.0.5 (newest) | 202 passed, no skips, all extras installed |
| Python 3.11 | not run — bracketed by the two above, not directly tested |
| Ubuntu | **not covered at all.** Both local runs are macOS/arm64 |
| Windows build | **not covered at all.** No artifact was produced |

Re-running the full matrix on Ubuntu, and producing a Windows artifact somebody actually opens, is
the single highest-value action outstanding on this repo.

**v1.6.0 reports the answer a best-worst study was fielded for.** The tool already scored MaxDiff
exports and grouped people on the utilities — and then described the groups without ever saying
which items the sample preferred. The ranking was computed in the sampler and discarded. It now
appears as a card, a report section and two downloads, each row carrying the probability that it
really beats the row below it. The build's smoke test now puts a best-worst file through the
PACKAGED app as well: without `maxdiff.py` bundled, the engine silently clusters the raw choice
rows instead, and reports two groups on 800 rows where there are 40 people.

**v1.5.9 was the audit release.** Nothing in it came from a user report; all of it came from pointing
the tool at data it had never seen and checking whether what it said was true. The one behaviour
change is the distinct-patterns cap — a short survey can no longer be cut into more groups than its
answers can distinguish. **That changes results on five-question surveys**, which is the most common
shape a short questionnaire takes. See CHANGELOG.md, which also records what was verified and
deliberately left unchanged, so a later session does not repeat the measuring.

### No open findings

The `group_profiles.csv` naming defect recorded here previously is fixed and released. Every finding
raised during this audit is either fixed or written down below as a known limit.

### The three things nobody has checked, and cannot be checked from here

These are not bugs. They are gaps in the evidence, and they are the honest reason the readiness
verdict below is bounded rather than unconditional.

1. **The live Claude API round-trip has never been run.** By deliberate constraint — the real key
   sits in `~/.survey_segmenter/config.json` and is not to be read or used. The no-key path is
   tested and degrades cleanly; the with-key path is tested only against a stub.
2. ~~**HB MaxDiff has never seen a real best-worst export.**~~ **CLOSED 2026-08-07.** The
   `bwsTools` example data from CRAN — 350 real respondents asked which issues facing the country
   matter most and least — now goes through it. Against the classical best-minus-worst score on the
   same data: Spearman 1.0000, Pearson 0.9986. It also independently found three segments, the
   number that package's own authors report. Reading it required two fixes; see CHANGELOG 1.6.1.
   What remains unseen is a **Qualtrics/SurveyMonkey** export specifically, whose column naming
   nobody here has a sample of.
3. **Nobody has hand-clicked the Windows build** — and as of today nobody can, because CI is not
   running and no Windows artifact exists for anything after v1.5.8. Blocked on the billing fix
   above.

### The working rules this session established

- **Audit the built app, not the source tree.** The worst defect of the day lived exactly in that
  gap — every packaged release produced reports with no tables, while every test passed.
- **Verify the instrument before trusting a negative result.** Five false alarms came from broken
  probes: grepping a compressed archive, a "hang" that was my own pipe buffer, HTTP 200s whose
  errors were in the body, a static scan flagging attributes the report shows, and two runs that
  drew from the random stream differently and so were not the same data.
- **A test that passes with and without the fix is not a test.** The memory guard was first written
  at 2.5 GB; the defective code peaks at 2.39 GB and would have sailed through.
- **Measure the fix before adopting it.** Four plausible improvements were rejected on measurement
  this session, including two that made things actively worse.
- **Re-run `python3 references/kbench.py` after touching `recommend_k`.** The documented accuracy
  table goes stale silently otherwise.

### What is genuinely left

Nothing broken that I know of. The remaining gaps are all blocked on something other than code:

| | |
|---|---|
| Live Claude API round-trip | Deliberately unverified — no key is used. The no-key path gives a clean error |
| HB MaxDiff on real responses | Now validated on 350 real respondents: Spearman 1.0000 against the classical score. A Qualtrics-shaped export is still unseen |
| The Windows build | CI-verified only up to v1.5.8; nobody has hand-clicked it |

## Session state — 2026-08-07 (earlier)

**Shipped today: v1.5.1 through v1.5.7.** The team copy at
`~/Desktop/Survey Segmenter (app for the team)/` is on 1.5.7 and verified. CI green on both
workflows. Nothing is outstanding or half-finished.

**The three that change what a user gets, in order of how much they mattered:**

1. **Every packaged release ever built produced reports with no tables** (fixed in 1.5.6). Segment
   sizes, the stability checks, the centroids and the k-selection panel all arrived as
   run-together text. `to_markdown` needs `tabulate`, pandas imports it lazily so the packager
   never bundled it, and it was declared an *optional* extra described as "prettier Markdown
   tables" — a description that was simply wrong. Measured on the binary: 0 tables before, 8 after.

2. **A genuine segment was condemned in print** (fixed in 1.5.5). Segment persistence reported the
   weaker of two directions, and asking for one more group forces a split, so whichever segment
   got subdivided scored about 0.5 whether or not it was real. On a study recovered at ARI 0.954,
   the largest and cleanest segment held together perfectly under merging, scored 0.56 under
   splitting, and the report said *"Do not build a campaign on the ones marked 'dissolves'."*

3. **The number of segments could be wrong on data that plainly contained them** (fixed in 1.5.2).
   Two independent faults in `recommend_k`; against planted truth, 0.618 became 0.992.

**Also shipped:** large studies (48,842 respondents went from 11.04 GB and 2.2 minutes to 1.59 GB
and 0.9; 581,012 rows now run in 3.2 minutes), seven file shapes that were read wrongly and
silently, a cap on calling a wide questionnaire high confidence, the categorical path brought from
two of eleven pieces of evidence up to parity, and "no answer" codes in follow-up files no longer
scored in silence.

### The three rules this session established

- **Audit the built app, not the source tree.** The worst defect of the day lived exactly in that
  gap: the source had `tabulate`, so every test passed on every machine anyone was looking at.
- **Verify the instrument before trusting a negative result.** Four separate false alarms came
  from broken probes — grepping a compressed archive, a "hang" that was my own pipe buffer, HTTP
  200s whose errors were in the body, and a static scan that flagged seven attributes the report
  actually shows.
- **A test that passes with and without the fix is not a test.** The memory guard was first written
  at a 2.5 GB threshold; the defective code peaks at 2.39 GB and would have sailed through.

### What is genuinely left

Nothing broken that I know of. Remaining gaps are blocked on something other than code:

| | |
|---|---|
| The live Claude API round-trip | Deliberately unverified — no key is used. The no-key path is checked and gives a clean error |
| HB MaxDiff on real responses | Recovers planted utilities at rank correlation 1.000 / 0.986, but has never seen a real best-worst export |
| The Windows build | CI-verified only; nobody has hand-clicked it |
| The React front end | The only substantial area not audited this session |

## Session state — 2026-08-06

**Four commits sit on `main` after the v1.5.0 release**, and one of them matters enough to name:
`Build from the declared dependencies` fixed a defect where **any CI-built app silently shipped
without the dip test**. The v1.5.0 archive attached to the GitHub release is sound because it was
built locally on a machine that already had `diptest`; a CI-built one before that commit would not
have been. Cut a 1.5.1 before circulating anything built by CI.

The four, oldest first:

1. `Make the packaged smoke test check capabilities, not just that it runs` — the build now fails
   if the dip test did not run or any chart lacks its spec.
2. `Build from the declared dependencies, not from a second list of them` — the fix above found
   this on its first CI run, on both platforms.
3. `Record the Windows verification and the two-lists lesson` — docs.
4. `Build sparse k-means, measure it, and do not adopt it` — see below.

**Sparse k-means was built, measured and rejected.** It is in `references/sparse_kmeans.py`, not in
the package, with a runnable reproduction. Do not reinstate it without reading
`STATE-OF-THE-ART.md` first: it lifts the silhouette on *pure noise* from 0.12 to 0.39, and adds
nothing over the eta-squared already reported (both ranked real questions above noise ones 5/5).
That closes the last substantial item on the improvement list.

**Nothing substantial is outstanding.** The remaining known gaps are all "blocked on something
other than code" — see *Known limitations* below.

## Where it stands

| | |
|---|---|
| Repo | `github.com/tomtomhz/survey-segmenter` — **public**, MIT, owner `tomtomhz` |
| CI | Python 3.9 / 3.11 / 3.12 + a clean-install job, green |
| Tests | 152 Python (`pytest`) + 99 frontend (`cd frontend && npm test`) |
| Shipped app | **v1.5.0 release**: macOS `.app` (82 MB) and Windows `.exe`, built and smoke-tested by the **Desktop app** workflow. Never in git history. The team's copy lives in `~/Desktop/Survey Segmenter (app for the team)/`. |
| Local path | `~/dev/survey-segmenter` — **moved out of iCloud Drive**, see below |

```bash
cd ~/dev/survey-segmenter
pytest                  # 144 tests, ~125s
python3 run_app.py      # opens the web app
python3 build_app.py    # rebuilds + signs + smoke-tests the .app
```

## The one thing never verified

**No live Claude API call has ever been made from this code.** There is no key on the machine and
none should be added by an assistant. Everything about the request is pinned against a mock HTTP
server that asserts the exact wire format (`test_ai_request_is_well_formed_against_a_mock_anthropic_server`).

The first real call happens when a user pastes a key into **Settings**. If something breaks there,
expect it in *response handling*, not request construction.

## Decisions made, and why (don't silently reverse these)

- **The interface is React + TypeScript in `frontend/`, compiled to `webui/`.** `webui/` is
  committed — unusual for build output, and deliberate: a clone must run without Node, and the
  packaged `.app` must have no build step. CI rebuilds it and fails on drift, so it cannot go
  stale. `npm run dev` proxies to the Python app on 8000 for hot reload.
- **Charts are matplotlib, not hand-built SVG.** This REVERSES the original decision, at the
  owner's instruction. The old engine was 684 lines of Python f-strings concatenating SVG path
  data, tick positions and text anchors by hand; every chart re-derived its own axes and scaling.
  The packaged app grew from ~76 MB to ~82 MB, which is the price. What was kept: charts still
  emit vector SVG, and all chrome is drawn in one sentinel colour swapped for `currentColor` on
  the way out, so a single file is legible on both the light and dark grounds. What was gained:
  a PNG of every chart, which is what lets Claude see them.
  **scikit-learn computes, matplotlib draws** — nothing in `charts.py` decides anything about
  the segmentation.
- **Only an aggregate digest goes to Claude — never a respondent row.** Enforced by
  `test_the_ai_digest_contains_no_individual_respondent_data` against 400 respondents. This is the
  load-bearing privacy guarantee; see `PRIVACY.md`.
- **`fallbacks: "default"`, not a pinned fallback model.** Routes by refusal category and needs no
  maintenance when fallback targets change. It degrades through three rungs because not every
  account has the beta and not every SDK knows the parameter.
- **The binary lives in GitHub Releases, not git.** 190 MB of rebuildable artifact does not belong
  in every clone.
- **Real study material is deliberately NOT on GitHub, and never becomes so as part of a tidy-up.**
  Anything with named individuals in it is personal data under GDPR; publishing it needs a human
  decision taken on purpose. Every data file in this repo is synthetic.

## MaxDiff / Hierarchical Bayes (added 2026-07-31)

`maxdiff.py` estimates individual-level utilities from best-worst data — the input the study's
instrument requires, and the gap that previously blocked the Stockholm-Cluster survey. Drop a tidy
best-worst export (`respondent_id | set | item | choice`) into the tool and it is detected, scored
by HB, and segmented on the utilities, with the report saying so.

Measured on simulated Block D data (15 items, 5 per set, 12 sets), against known utilities:

| Separation | Individual recovery, counting → HB | Segment ARI, counting → HB |
|---|---|---|
| Strong | 0.76 → **0.92** | 1.00 → 1.00 |
| Moderate | 0.67 → **0.84** | 0.978 → 0.992 |
| Weak | 0.57 → **0.67** | 0.646 → 0.677 |
| Very weak | 0.48 → 0.47 | 0.137 → 0.146 |

**Read this honestly:** HB clearly improves *individual* utilities, which is what the instrument
asks for and what you report per respondent. Its effect on the *segmentation* is small but
consistent, and neither method rescues genuinely weak structure. Do not claim HB "fixes" a
segmentation — claim it gives defensible individual scores. The simulation also generates choices
from the same model HB assumes, which flatters it; real data will differ.

**Stress-tested against that flattery** (200 respondents, 3000 draws). Data generated with three
documented departures from the model — respondents differing in choice consistency, respondents
ignoring items outright, and "worst" decided on grounds the model does not represent:

| Data generated with | Counting | HB | Advantage |
|---|---|---|---|
| The model's own assumptions | 0.755 | 0.902 | +0.147 |
| Careless respondents (scale varies) | 0.725 | 0.865 | +0.140 |
| 30% of items ignored per person | 0.582 | 0.771 | **+0.189** |
| Worst chosen on other grounds | 0.722 | 0.880 | +0.157 |
| All three at once | 0.548 | 0.698 | +0.151 |

The advantage never collapses, and is *widest* under the worst violation. That is the useful
result: it is not an artefact of grading the model on its own homework. It still says nothing
about accuracy on real people. Locked in by
`test_hb_still_beats_counting_when_its_assumptions_are_wrong`.

## Validated on real data

Not a survey, but 17,000 real rows through the whole pipeline on 2026-07-31
(California housing, `median_house_value` and four correlates):

- 3 groups, confidence **high**, ~40s per run on an M-series Mac.
- All six charts drawn, no chart errors, every SVG well-formed and theme-aware.
- The standalone HTML report embeds all six charts and fetches nothing from the network — the
  only URLs in it are XML namespace declarations.

## Known limitations (real, not hypothetical)

- **Windows is verified by CI, not by hand.** The **Desktop app** workflow builds and smoke-tests
  on both platforms, and since the capability check was added that smoke test proves the compiled
  dip extension runs and every chart carries its data — not merely that the app starts. Nobody has
  sat in front of the Windows build and clicked through it.

- **A mixed questionnaire is only as good as its questions.** Gower k-prototypes uses every
  question, which means a pick-any question that separates nobody now costs accuracy where it
  used to be set aside. Measured: three useless brand columns beside six real ratings cost 0.25
  ARI. Confidence drops to moderate or low rather than claiming a result, and the
  variable-selection check names the offenders — but read that section before acting.
- **The mixed path runs one cross-paradigm check, not two.** A Gaussian mixture assumes every
  answer is a measurement, so it is disabled there (it argued for k=8 on a file whose answer was
  3 while it was still wired in). Calinski-Harabasz and Davies-Bouldin are read on the Gower
  embedding rather than on Gower itself. The report says both.

- **HB has never seen real MaxDiff responses.** The misspecification sweep below is the
  strongest evidence obtainable without them; it is not a substitute for them.
- **The segment-size floor is a search-time guard, not a hard bound.** It filters the search fit;
  the final fit uses more restarts and can land slightly under. The report's "below 5% of the
  sample" note is the backstop.
- **Hopkins is unreliable on short surveys** and is now caveated in place rather than corrected —
  duplicate answer patterns inflate it (0.78 on pure noise with 2 Likert questions).
- **Sidebar hides below 820px.** Offered a toggle; never requested.
- **Do not put this checkout back inside iCloud Drive.** It lived under `~/Desktop`, which macOS
  syncs by default, and with ~6,000 files in `node_modules` every read went through the sync
  layer. `npm test` took 37 minutes instead of one second, Vitest workers timed out before they
  could start, `npm run build` hung indefinitely, and macOS left `App 2.tsx` conflict copies in
  the source tree. Nothing was ever wrong with the code. Now at `~/dev/survey-segmenter`:
  59 frontend tests in 1.1s, build in 266ms.

## Next candidates, roughly by value

1. **A second cluster-tendency test.** Hopkins is the only one and it needs a caveat on short
   surveys. Hartigan's dip test on the pairwise-distance distribution is the usual companion.
2. **Variable weighting.** The tool reports which questions drive the segmentation and checks
   whether dropping the noise ones changes the answer, but weights every question equally when
   clustering. Sparse k-means (Witten & Tibshirani) learns the weights.
3. **Resolving overlapping segments** — the measured weakness. Worth understanding before
   attempting: merging may be the honest answer, and "improving" it risks trading away the
   never-confidently-wrong property, which is worth more.
4. **Consolidate `segment-kmeans-tool.md`** in the assistant memory directory; it has grown well
   past 22 KB of appended paragraphs and is due a rewrite rather than another append.

## How the modules divide up

`segment_kmeans.py` decides what a segmentation is and what it means. `charts.py` draws it.
`webapp.py` delivers it. `maxdiff.py` scores best-worst data before any of that happens, and
`kprototypes.py` supplies the distance and the prototypes for questionnaires that mix rating
scales with pick-any questions.

The direction of the imports is the point: `webapp` imports the engine, and the engine never
imports `webapp`. `serve()` and `app()` remain on `segment_kmeans` as thin forwarders so the
entry point does not change, and they import lazily because a module-scope import would be a
cycle. `run_app.py` therefore imports `webapp` by name as well — a lazy import is invisible to
PyInstaller, and shipping an app without its own web server is the same class of failure as the
matplotlib SVG backend that was left out of the first packaged build.

## Working conventions

- "Carte blanche" means execute through to completion; only stop if a finding invalidates an
  earlier decision.
- Prefer main-session work over subagent dispatch for pure synthesis (subagents have timed out).
- **Bound every background wait.** An unbounded `until curl ...; do sleep 3; done` left two shells
  parked for hours in a previous session. Use `curl --retry N` or a max-attempts counter.
