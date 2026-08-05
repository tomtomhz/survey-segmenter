# Onboarding — read this before touching anything

You are taking over **Survey Segmenter**, a segmentation tool for survey data. This
document is written to be self-contained: you should be able to work from it without the previous
conversation, and it tells you not only *what* exists but *why*, so you do not spend a day
re-deciding things that were already decided with evidence.

Read it in full before your first edit. It is long because reversing one of the decisions in
§4 by accident costs more than reading it.

---

## 1. What this actually is

A survey goes in — a CSV of people's answers — and out comes a set of customer segments, a report
a non-analyst can act on, charts, a file saying who is in which group, and a rule for classifying
future respondents. It is used by a marketing team, not by statisticians, and it ships as a
double-clickable desktop app so nobody has to touch a terminal.

**The single most important property of this tool is that it refuses.** On data with no structure
it says so. When the evidence is thin it says "low confidence" instead of producing a tidy answer.
Measured across 21 known-answer conditions, it has never reported high confidence while wrong.
That property is worth more than any accuracy improvement, and several plausible "improvements"
would trade it away. If you are ever choosing between *finds more structure* and *never confidently
wrong*, choose the second.

The audience matters for tone too. Reports are written in plain language — "about 1 in 3 of
people", not "n = 137 (34.2%)" — with the statistics underneath for anyone who wants them.

---

## 2. Where everything lives

| Path | What it is |
|---|---|
| `~/dev/survey-segmenter` | **The project.** Git repo, all code, all docs. Work here. |
| `github.com/tomtomhz/survey-segmenter` | Private repo, MIT, owner `tomtomhz`. CI on every push. |
| `~/Desktop/Survey Segmenter (app for the team)/` | What the team downloads: the signed `.zip`, `HOW TO RUN.md`, an example CSV. |
| `~/Desktop/bd-gtm-review-team 3/` | A **separate** workspace: a BD/GTM subagent review team plus research deliverables. Its `CLAUDE.md` governs that folder, not this one. |

Two stale locations that will mislead you:

- `~/Desktop/bd-gtm-review-team 3/tools/survey-segmenter/` — an **old copy**. It has a `MOVED.md`.
  Do not edit it. The repo moved out of `~/Desktop` because macOS syncs it to iCloud, and with
  ~6,000 files in `node_modules` every read went through the sync layer: `npm test` took
  **37 minutes** instead of one second, Vitest workers timed out before starting, and macOS left
  `App 2.tsx` conflict copies in the source tree. Roughly 25 minutes of a session was lost
  diagnosing this as a code problem. **Never put the checkout back inside iCloud Drive.**
- `~/Desktop/Nordic Match Case Prep/` — an unrelated May folder about an Exyte M&A case. Sessions
  sometimes launch there by accident. Nothing current lives in it.

### The modules, and the direction of their dependencies

```
segment_kmeans.py  (3,650 lines)  decides what a segmentation is and what it means
charts.py            (750)        draws it (matplotlib)
webapp.py            (690)        delivers it (stdlib HTTP server + JSON API)
kprototypes.py       (410)        distance + prototypes for mixed-type questionnaires
ai_interpret.py      (340)        the optional "ask Claude about your segments" layer
maxdiff.py           (330)        scores best-worst (MaxDiff) exports into utilities
frontend/                         React 19 + TypeScript + Vite, built into webui/
```

`webapp` imports the engine; **the engine never imports `webapp`**. `serve()` and `app()` stay on
`segment_kmeans` as thin forwarders that import lazily to avoid a cycle. Keep it that way.

---

## 3. Hard constraints — do not violate these

1. **`deliverables/` in the BD/GTM workspace contains roughly 100 named individuals' work email
   addresses.** That is GDPR personal data. It must never go to GitHub, into a prompt, into a
   report, or to any external service. If a task seems to require it, stop and ask.
2. **Respondent-level data never leaves the machine.** The Claude interpretation layer sends only
   the aggregate digest — segment sizes, mean scores, stability numbers — plus rendered charts.
   Never a respondent's row, id, or free-text answer. This is asserted *and* tested: a test checks
   0 of 400 respondent ids and no free-text answers appear in the payload. Do not weaken it.
3. **Do not obtain or use an Anthropic API key.** The user supplies their own through the app's
   Settings. There is a real key in `~/.survey_segmenter/config.json` on this machine — do not
   read it, do not use it. The consequence is that the live API round-trip is deliberately
   unverified (see §7); that is accepted, not an oversight to fix.
4. **GitHub credentials are the user's.** `gh` is already authenticated on this machine; use it
   for reads, CI checks and releases. Never run `gh auth login`, never handle a password.
5. **Never enter credentials, card numbers, or personal data into any form or prompt.**

---

## 4. Decisions already made, with the evidence. Do not silently reverse these.

Each of these was argued or measured. If you want to change one, that is allowed — but say so
explicitly and bring evidence, do not drift.

**How the number of segments is chosen.** Not by an elbow. A weighted panel: prediction strength
(Tibshirani & Walther) and split-half replication weigh most, then the separation indices
(silhouette, Calinski-Harabasz, Davies-Bouldin), then the gap statistic, and the inertia elbow
counts least — it always bends somewhere. Burkov's threshold of 0.80 for prediction strength is
used literally. Rationale: a segmentation nobody can reproduce is not a finding.

**Range standardisation, not z-scores** (Milligan & Cooper 1988, replicated by Steinley 2004a for
k-means specifically). On a 1-5 scale, z-scoring inflates the questions nobody disagreed about.

**Fifty restarts, and the report says what share reached the best solution.** Steinley: local
optima "run into the thousands"; few restarts mislead.

**Four methods, detected automatically, not configured.** All-rating surveys go to k-means; a
Gaussian mixture (`gmm`) is available for elliptical/overlapping segments; all-categorical surveys
go to a hand-written EM latent-class model; and a survey holding **both** kinds goes to Gower
k-prototypes. The user just uploads a file.

**Mixed-type surveys use Gower k-prototypes, not Huang's original** (Szepannek, Aschenbruck &
Wilhelm, *ADAC* 2024). Huang needs a `gamma` weight trading numeric distance against categorical
mismatches, and nobody in a marketing meeting can defend the number they picked. Gower scores every
question 0-1 in its own natural way instead. Three things in `kprototypes.py` are load-bearing and
non-obvious:

- **Prototypes are medians, modes and nearest-rank levels — never means.** Gower is an L1-type
  distance, so those are the exact per-variable minimisers, and they are what make the algorithm
  provably terminate. A naive port of k-means using means everywhere breaks the proof. A test
  drives the objective by hand and asserts it never rises.
- **Ordinal answers use Podani's *metric* version (1999, eq. 3), not the widely cited tie-corrected
  non-metric one.** The famous version subtracts the within-tie spread from every gap, which on
  survey data collapses adjacent answers to nearly zero while leaving the extremes a full 1 apart:
  for three equally-used Likert levels it gives d(1,3) = 1.00 against d(1,2) + d(2,3) = 0.065. That
  violates the triangle inequality badly enough that "closest prototype" stops meaning anything.
- **Gower distance is exactly weighted Manhattan distance** on a transformed matrix — ratings over
  their range, pick-any answers one-hot at half weight. Verified to 1e-16 and pinned by a test.
  This is why the silhouette, cluster tendency, the hierarchical cross-check and the segment map
  all reuse the same scikit-learn calls with `metric="manhattan"` rather than a second family of
  hand-written Gower functions that would need their own tests and would drift. **If you add a
  distance-based diagnostic, go through `_geometry()`** — do not hand-roll.

**Colour does exactly one job per chart, and the palette is validated by a script, not by eye.**
Four jobs — segment identity, above/below average, magnitude, and plain chrome — each get their
own encoding and they do not overlap. They used to share one palette, so orange meant "Group 1" on
four charts, "Separation (silhouette)" on the choice-of-k chart and "above average" on the
heatmap. The identity palette was measured and it FAILED: three hues below the chroma floor
(reading grey) and the worst adjacent pair at CVD ΔE 7.9 — slots 1 and 2, which is Group 0 against
Group 1, the most common comparison in the tool. Its comment claimed Okabe-Ito, which does pass at
15.8; it had been edited into failing. If you touch these hues, re-run the validator. **Slot order
is part of the safety**, because the checks run on adjacent pairs.

**Segments carry a marker shape as well as a colour, and this is required.** The segment map is a
scatter, so every pair of hues is on screen at once. Measured at eight groups: worst all-pairs CVD
ΔE 3.2 and worst normal-vision ΔE 7.1 — two segments a reader with full colour vision cannot tell
apart. Only three slots clear all-pairs on colour alone. The shape also advances an extra step on
each wrap past eight, because colour and shape both cycling on eight made group 8 identical to
group 0 in *both* channels.

**No radar chart. It was removed, not lost.** A radar encodes value as distance from a centre, so
the eye reads enclosed AREA — which grows with the square of the values and changes completely
when the questions are reordered, an order carrying no meaning. Its labels truncated too. The
Cleveland dot plot in `chart_profiles` answers the same question as a distance along a shared
axis. Do not re-add it because a stakeholder asks for a "spider chart".

**Bars start at the scale floor, never at zero, on rating data.** A 1-5 scale drawn from 0 spends a
fifth of every bar on a region no respondent can occupy. That is why profiles is a dot plot.

**Charts carry their own dark mode as CSS variables, and only VECTOR marks follow it.** A
rasterised mark bakes the light palette into a bitmap; measured on the dark card that leaves the
violet slot at 1.77:1. Marks stay vector under a measured size threshold and rasterise above it.
`var()` resolves in a `style="…"` declaration but NOT in an SVG presentation attribute — if
matplotlib ever emits `fill="…"` instead, theming silently stops, which a test guards.

**Respondent text is lifted out of the SVG before colours are swapped.** The swap is string
replacement and cannot tell data from markup: a question worded "#2a78d6 is my favourite" was
being rewritten into a theme token. Whatever holds that text must be per-call, never module or
function state — the server draws for several people at once.

**Per-respondent fit is Leisch's shadow value, not a silhouette.** A silhouette needs every
pairwise distance, O(n²), so it fell back to a 6,000-row sample and left everyone else blank —
holes in the exported file exactly on the studies large enough to matter. Shadow values need only
the two nearest centroids, O(n·k).

**Segment-level stability is scored by containment, not Jaccard.** This one is a trap and the test
pins it deliberately: going from k to k−1 *must* merge two segments, so Jaccard collapses whether
or not the structure is real. Measured, it could not separate three genuine segments (0.44–0.78)
from pure noise (0.55–0.69) at all. Containment — what share of a segment's members stay together —
separates them cleanly: 0.78–1.00 against 0.47.

**Pick-any questions are scored by Cramér's V *squared*, not eta-squared, and the squaring matters.**
Eta-squared on brand codes measures the coding, not the data — renumber the brands and it changes.
V is correlation-like where eta-squared is variance-like, so they sit one square apart: a purely
random pick-any column scores V = 0.06, already over the 0.05 "near-noise" floor, meaning a useless
question could never be flagged as one. V² reads 0.00 on the same data.

**The AI layer uses the official Anthropic SDK with a user-supplied key, not a shell-out to the
Claude CLI.** A CLI dependency cannot be bundled into a double-clickable app for non-technical
users; a key in Settings works.

**Markdown is the source of truth for reports.** HTML and PDF are generated. A fix made only in a
generated file is lost on the next export.

---

## 5. What is measured, and what is only asserted

Keep this distinction — it is the project's habit and its main defence against overclaiming.

**Measured on this machine** (reproduce with `python3 references/kbench.py`, about 4 minutes):

| Condition | Recovered the true k | Confidence |
|---|---|---|
| Well-separated, k=3 | 3/3 | high |
| Sizes 80 / 15 / 5 | 3/3 | high, moderate |
| 3 real questions + 5 that separate nobody | 3/3 | high |
| Two elongated bands (k-means' worst case) | 3/3 | high |
| k=5 | 3/3 | high |
| Pure noise | 3/3 **correctly refused** | low |
| Overlapping, k=3 | **1/3** | high when right, moderate when wrong |

**19 of 21.** Both misses merged two overlapping segments into k=2 and dropped to "moderate" both
times.

Mixed questionnaires (4 ratings + 2 pick-any, true k = 3):

| Which questions carry the signal | Found k | Confidence | ARI |
|---|---|---|---|
| Both kinds | 3, 3, 3 | high | 0.88–0.89 |
| Ratings only (pick-any is noise) | 2, 2, 2 | moderate, low | 0.38–0.43 |
| Pick-any only (ratings are noise) | 2, 3, 6 | low | 0.21–0.31 |

**Asserted from literature, cited in the code next to the thing it justifies** — these are the
first claims to check if you ever have reason to doubt them. `references/STATE-OF-THE-ART.md`
keeps the two categories separate on purpose; maintain that.

One correction worth carrying forward, because the previous session got it wrong and had to fix it
in public: "it never reports high confidence while wrong" was verified **on k recovery only**, not
on membership accuracy. There is a known case (MacKay's broad/narrow) with ARI 0.55 at high
confidence. State the claim with its scope.

---

## 6. How the person you are working for works

- He drafts material with Claude chat, then brings it to Claude Code to audit and correct. **Treat
  Claude-chat output as a draft, not as ground truth.**
- "Carte blanche" means execute through to completion. Do not stop to check in unless a finding
  invalidates a decision he already made.
- He asks direct questions when he suspects you are stuck ("are you stuck in a loop?"). Answer
  honestly and immediately, including "yes, and here is what I have ruled out".
- He values being told what is *not* verified as much as what is.
- Pure synthesis work is better done in the main session than dispatched to a subagent — the
  strategic-advisor subagent timed out twice on synthesis prompts.

---

## 7. Known limitations — real, not hypothetical

- **The live Claude API round-trip has never been exercised.** No key, by design. Request
  construction is verified against a mock server (exact model string, streaming, adaptive thinking,
  the digest actually reaching the model, no `temperature` — which would 400 on Opus 5). The
  response-handling path is untested.
- **Hierarchical Bayes has never seen real MaxDiff responses.** A misspecification sweep is the
  strongest evidence obtainable without them, and it is not a substitute.
- **Overlapping segments get merged.** If two mind-sets genuinely shade into one another, expect
  one group and a "directional" verdict. Before trying to fix this, understand that merging may be
  the honest answer and that "improving" it risks the never-confidently-wrong property.
- **Hopkins is unreliable on very short surveys** — duplicate answer patterns inflate it to 0.78 on
  pure noise with only 2 Likert questions (it reads 0.56 at 4 items and 0.48 at 10, so it works
  where it matters). Caveated in place rather than corrected.
- **A mixed questionnaire pays for its useless questions.** Gower uses every question, so a
  pick-any question that separates nobody now costs accuracy where it used to be set aside:
  measured, three useless brand columns beside six real ratings cost 0.25 ARI. Confidence drops
  rather than the tool claiming a result, and the variable-selection check names the offenders.
- **The mixed path runs one cross-paradigm check, not two.** A Gaussian mixture assumes every
  answer is a measurement, so it is disabled there — while it was still wired in it argued for k=8
  on a file whose answer was 3. Calinski-Harabasz and Davies-Bouldin are read on the Gower
  embedding rather than on Gower itself. The report states both.
- **The segment-size floor is a search-time guard, not a hard bound.**

---

## 8. Getting oriented — do this first

```bash
cd ~/dev/survey-segmenter
pytest                                   # 136 tests, ~140s
cd frontend && npm test && cd ..         # 59 tests, ~2s
python3 references/kbench.py             # reproduces the measurement tables above
python3 run_app.py                       # opens the app in a browser
```

Then read, in this order:

1. `docs/HANDOVER.md` — engineering state of play.
2. `references/STATE-OF-THE-ART.md` — what the tool does versus current practice, with the
   measured/asserted split.
3. `references/IMPROVEMENT-NOTES.md` — the reading notes behind the last round of work, including
   two corrections where the obvious approach turned out to be wrong.
4. `kprototypes.py`'s module docstring — the densest explanation of a single design decision in
   the repo, and a good model for the commenting standard expected here.

**The commenting standard is unusual and deliberate:** comments explain *why*, and where a number
was chosen empirically they say what was measured. Match it. A comment saying "compute the
distance" adds nothing; one saying "0.80 is Burkov's threshold, and the chart draws that line so
the reader can see whether anything clears it" is the house style.

---

## 9. What to do next, in order

Nothing is urgent. The tool is shipped and working at v1.2.0. In rough order of value:

1. **A second cluster-tendency test.** Hopkins is the only one and needs a caveat. Hartigan's dip
   test on the pairwise-distance distribution is the usual companion. Cheap, and closes the one
   diagnostic that currently has to apologise for itself.
2. **Variable weighting.** The tool reports which questions drive the segmentation and checks
   whether dropping the noise ones changes the answer, but it weights every question equally when
   clustering. Sparse k-means (Witten & Tibshirani) learns the weights. This would also reduce the
   mixed-path weakness in §7.
3. **Validate HB MaxDiff on real responses** — blocked on data, not on code.
4. **Consolidate the assistant memory file** `segment-kmeans-tool.md`; it has grown past 22 KB of
   appended paragraphs and is due a rewrite rather than another append.

Explicitly **not** worth doing, and previously argued: kernel, fuzzy, possibilistic, intuitionistic,
metaheuristic and deep k-means variants. Roughly half the reference library is these. Each adds a
parameter nobody in the room can defend and a result nobody can reproduce by hand, and nothing
found suggests any would change a decision for a 400-respondent Likert survey. The bar is not "is
this newer" but "would this change a decision, and could the person making it explain why".

---

## 10. Traps this project has already fallen into

Every one of these cost real time. They are listed so you do not repeat them.

- **A packaged app can exit 0 and still be broken.** PyInstaller caches its analysis in `build/`,
  and a cache written while a file was momentarily unparseable sticks — the next build reports
  success while omitting a module. `build_app.py` now wipes `build/` and smoke-tests the binary.
  It has shipped an app with no chart backend once already (`matplotlib.backends.backend_svg` is a
  lazy import, invisible to static analysis).
- **Verify the packaged app on the feature you just added,** not just on a default survey. The
  default smoke test passes while a lazily-imported new module is missing.
- **Run the exact CI command, not your own approximation.** Linting file by file and skipping
  `tests/` let an f-string with no placeholder reach main. Install the wheel in a clean venv rather
  than trusting an edit to `pyproject.toml`'s hand-kept `py-modules` list — omitting a module there
  makes every installed copy die at import while every test still passes.
- **Copying large files into the iCloud-synced Desktop times out.** Write to a dot-prefixed temp
  name in the destination folder, verify (`unzip -t`), then `mv`. A plain `cp` can leave the team a
  truncated download.
- **A single missing phrase fails a whole column.** `_try_likert` requires *every* answer to map,
  so one unrecognised Swedish wording sent an entire survey down the categorical path and silently
  lost the ordering it was measuring. Swedish surveys are a live case here, not hypothetical.
- **Global mutable state bites under the web server.** A module-level `last_errors` list made 3 of
  9 concurrent runs report failures they never had. Pass the collection the caller owns.
- **A test that passes with and without the fix is not a test.** One was written, noticed, and
  removed rather than shipped as a false green tick. Do the same.
- **Report what you actually verified.** If a claim holds only under a narrower scope than you
  stated, correct it in plain language and move on.

---

*Written 2026-08-01 at v1.2.0. If you change something in §4, update this file in the same commit.*
