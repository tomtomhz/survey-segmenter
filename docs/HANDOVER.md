# Handover — state of play

> **New to this project (human or assistant)? Read [ONBOARDING.md](ONBOARDING.md) first.**
> It carries the scope, the decisions that must not be silently reversed and why, what is
> measured versus merely asserted, and the traps this project has already fallen into.
> This file is the narrower engineering state of play.

Working state for whoever picks this up next, human or AI. `README.md` says what the tool is;
this says where it stands, what was decided and why, and what to do next.

**Last updated:** 2026-08-09 · repo `github.com/tomtomhz/survey-segmenter` (public) · `main` @ **v1.20.0**, 284 Python + 161 frontend tests green locally on Python 3.9 **and** 3.12, and green in CI on Ubuntu and macOS

---

## Session state — 2026-08-08/09 (read this first)

**Released: v1.5.1 → v1.20.0.** Tree clean. The team copy at `~/Desktop/Survey Segmenter (app for
the team)/` is on **1.20.0** — verified by unpacking the zip that is actually sitting there and
reading the version out of the bundle, plus `codesign --verify --deep --strict`, rather than
trusting that the copy succeeded.

### The workflow now covers the whole loop

Up to 1.9.0 the tool analysed a study someone else had designed and fielded. It now covers
**design → plan → field → analyse → rank → decide**, and the two ends added last are the ones that
were previously outsourced to Sawtooth:

| Step | Where | Since |
|---|---|---|
| Build the best-worst questionnaire | `--design`, and a panel in the app | 1.11.0 / 1.12.0 |
| Decide how many respondents to recruit | `--plan`, and a panel in the app | 1.8.0 / 1.9.0 |
| Score best-worst answers by HB | automatic on detection | 1.6.0 |
| Rank the items, with beat probabilities | report card + two downloads | 1.6.0 |
| Which few items to actually launch | TURF, in the report | 1.10.0 |
| Segment people, profile, name, export | the original tool | — |

Nothing in that table is command-line-only any more. `--design` was, for one release, and 1.12.0
closed it the same way 1.9.0 closed the planner.

**The app deliberately caps the design shape** at 40 items, 6 per screen, 15 screens and 300
versions. That is not timidity: it is the largest shape that answers in about twenty seconds, and
the corner beyond it (8 items across 20 screens) takes eighty. The CLI has no ceiling. If someone
raises the app's limits, re-measure first — the cost grows with `screens² × per_screen²` per person,
so it climbs much faster than the numbers look like they should.

### v1.10.0 — TURF, and how much of a reach figure is luck

A ranking says what people like on average; it does not say which three to launch, because the
top three can all appeal to the same people. TURF answers the portfolio question instead, and the
correction it carries is that picking the best-scoring combination out of hundreds and then
reporting that combination's own score is optimistic by construction.

**The numbers in this section were wrong twice, in the same way both times, and 1.13.0 fixed them
against an outside truth. Read that section before touching turf.py.** A docstring first claimed
"several percentage points" and was out by four times; the replacement claimed 9.5-22.3 points and
was measured with the wrong instrument. The current table is checked against a 40,000-person
population where the true reach is a lookup rather than an estimate.

### v1.11.0 — build the questionnaire, not only read one

`segment-kmeans --design my_items.txt` writes which items appear on which screen, for whom. The
design decides what the data can possibly say and is not recoverable afterwards: two items that
never appear together cannot be compared, however many people answer.

Deliberately **not** a textbook balanced incomplete block design — those exist only for particular
combinations of item count, set size and screen count, and a real study fixes those from how long
people will sit still. It searches for the most balanced design of the shape you actually want and
then reports what it achieved, including plainly when the shape *cannot* be even, which is
arithmetic rather than a flaw.

**Verified by closing the loop, not by balance statistics** — which are only a proxy. Build a
design, simulate people answering it from known preferences, run the real estimator, check the
ranking that comes back is the one that went in: Spearman 1.000, sampler converged.

The balance search is scored incrementally. The first version took 24s for sixty respondents and
grew with the sample, so a real study would never have finished; a swap moves items between screens
without changing how often any item appears, so item balance is invariant and only a few dozen pair
counts can change. 24s became 0.3s, four hundred respondents in 1.7s, identical balance.

### v1.11.1 — the fourth time the packager shipped something missing

`turf` and `design` are imported inside functions, so PyInstaller's static analysis never saw
either and neither was bundled. The app launched, segmented and ranked perfectly well, and simply
stopped answering the question a best-worst study is usually commissioned for.

That is now four for four — the dip test, `tabulate`, matplotlib's SVG writer, and these two — all
the same shape: **a lazy import is invisible until something runs the code path.** So the build's
smoke test no longer takes the build's word for it; it puts a best-worst export through the
packaged binary and fails the build if the TURF section is absent. Any new module imported inside
a function needs a `--hidden-import` line *and* a smoke-test assertion, or it will ship missing and
nothing will say so.

**One loose end from that release, harmless but worth knowing.** The version bump moved
`pyproject.toml` and `segment_kmeans.py` and missed `frontend/package.json`, so the `v1.11.1` tag
and the attached artefact carry a frontend version string of 1.11.0. Nothing user-visible depends
on it — the report footer and the app's own version come from `__version__` — and `main` is
consistent again. `test_the_places_the_version_is_written_by_hand_all_agree` caught it, which is
the guard doing its job; it caught it *after* the tag because the bump was pushed on a pyflakes run
with no pytest behind it. **Run the suite before tagging, not just the linter** — and BUILD before tagging too. Tagging first went wrong three times in one session: v1.12.0 and v1.12.1 became tags with no artefact because the build followed a later edit, and v1.13.1 was tagged one commit before the fix that let it build at all. The order is test, build, then tag.

### v1.12.0 — the designer reaches the people it was built for

Same argument as 1.9.0, and worth stating once so it does not have to be re-argued: a capability
that exists only behind a command line, in a tool whose premise is that the user does not use one,
is not delivered. Items are pasted rather than uploaded, because the list lives in an email or a
slide and saving a `.txt` first was a step that existed only for `argparse`.

Two things in it are worth not re-deriving:

* **The CSV comes back inside the JSON reply, not through `/download`.** Everything that route
  serves belongs to an analysed session and a design has none; inventing a session to hold one file
  would put a study nobody ran into the project list. At the capped shapes the file is about a
  megabyte, which is an ordinary reply for a local server.
* **Duplicate lines are collapsed case-insensitively.** A pasted list routinely has the same
  benefit twice, and left alone it produces a screen showing two identical options with preference
  split between them.

**The search also got about twice as fast**, and much more on long lists. The objective — a
standard deviation over the whole pair matrix — was `O(items²)` per candidate swap while the swap
touched a few dozen entries. Because a swap cannot change the total number of pairings, the mean is
fixed, so minimising the variance is exactly minimising the sum of squares, and each changed entry
updates that in constant time. Identical balance at every shape tried; 60 items over 500 people went
from 300s to 157s. `test_a_swap_never_changes_the_total_number_of_pairings` exists because the whole
shortcut rests on that invariance, and a broken version would still report a plausible balance.

**Two stale README claims went with it** — that the tool "does not design the experimental stimuli"
(untrue since 1.11.0) and that HB "has not yet been run on real MaxDiff responses" (untrue since
1.6.1). Both were the project describing an older version of itself, which is the failure mode this
file exists to prevent.

### v1.12.1 — what an hour of adversarial probing found in an hour-old feature

Worth recording as a method rather than as a bug. The design endpoint was written, tested, driven
through the real browser, and green — and then probed with the four shapes a caller most easily
gets wrong. Three of them were accepted:

* **Items sent as a bare string** iterated letter by letter, producing a study comparing `d`, `e`,
  `l`, `i`, `v`, `r`, `y` — and a report calling it a balanced seven-item design. This is the worst
  kind of defect this project produces: not a crash, an answer.
* **Objects and nulls in the list** became items named `{'a': 1}` and `None`.
* **A 50,000-character item** turned a four-item design into a one-megabyte reply.

None of these were reachable from the panel, which is exactly why none of the tests caught them.
**A test suite written against the UI only covers the inputs the UI can produce**, and every
endpoint is reachable directly. The fix refuses all three rather than coercing: `str()` will
cheerfully turn a dict into a string, and every such coercion produces a design that reads correctly
and is nonsense in the field.

**Then the same question was asked of every sibling endpoint, which is the part worth repeating.**
`/name` had it too, and worse: a segment name is free text, so nothing downstream can tell a coerced
name from a real one. `"AB"` named the two segments **A** and **B**, straight into `group_names.csv`
and every export built from it. Fixed in 1.12.2. `/plan` and `/regroup` were probed and are sound —
the planner's bounds catch a coerced number, and re-grouping drops column names that do not exist
and refuses when nothing real is left. Worth re-running that probe whenever an endpoint is added.

### v1.13.0 — auditing TURF against an outside truth, and what that is worth

The method that found these is the one worth keeping: **check a statistic against a truth it did
not produce.** Every earlier check on TURF compared its numbers to each other, which cannot detect
a definition that is measuring the wrong quantity. Drawing samples from a 40,000-person population
and looking up what the chosen items really reach turned three defects up in an afternoon.

1. **Ties decided the recommendation, invisibly.** Reach is a count of people, so it lands on
   multiples of 1/n; hundreds of candidate sets chasing sixty possible values collide constantly.
   The best reach was shared in 14 of 30 studies at sixty people, and **reordering the item list
   changed the recommendation in 8 of 25**. Now reported as a tie rather than resolved silently.
   Do not "improve" the tie-break — every tie-break is arbitrary, which is the point.
2. **The optimism figure measured a study half the size.** `in_sample - holdout` compares two
   half-sample quantities; the headline is a full-sample number. The report printed "Expect about
   93%, not 95%" above "the 3-point difference" — arithmetic a reader can check, and it did not add
   up. Now `reach - holdout`, which also tracks the real error far better against the population.
3. **Below nine items the answer was arithmetic.** "In your top three" of five items is 60% of the
   list, and the best set of three reached 100.0% in every study tried. Refused now.

**The documented optimism table has now been wrong twice** — "several percentage points" (out by
four times), then 9.5-22.3 points (measured with the wrong instrument). Both were plausible numbers
that nobody had checked against anything external. The current table records the real error beside
the reported one so the two can be compared directly; if you change the holdout scheme, re-run that
comparison rather than reasoning about it.

### v1.13.1 — the sampler was crying wolf on the platform it ships on

Two results from pointing the same "check it against an outside truth" method at `maxdiff.py`.

**The alarming one was the false one.** Every estimate run on a Mac printed "divide by zero",
"overflow" and "invalid value" from inside the sampler, and the arithmetic was correct every time.
It is numpy 2 on Apple's Accelerate BLAS: an ordinary matmul of small finite numbers raises all
three flags from SIMD padding lanes. Reproduce with `standard_normal((80,9)) @
standard_normal((9,9))` — warns, returns a finite product. Before believing a fix here, run that
one line; it takes a second and it is the whole diagnosis.

The flags are suppressed with `@np.errstate` on `estimate_hb` and replaced by an explicit
finiteness check on the kept draws, raising `_MAXDIFF_DIVERGED` with a plain-English translation.
**Do not remove that check when tidying the suppression** — it is the only thing now standing
between a genuinely diverged chain and a ranking that still looks like an answer.

**The load-bearing one held up.** `prob_ahead` claims a probability, so it was checked for
calibration against known population utilities: 126 adjacent pairs from fourteen studies, claimed
84.3%, correct 83.3%, tracking bin by bin. That number is quoted in the report as though it were a
probability, and now it has earned it.

### The confidence light, measured against a known truth

The most load-bearing claim in the product — "High · Trust these groups" on the front card — had
never been checked as a whole. Sixty planted studies (known k, known separation) through the real
pipeline:

| light | studies | right k | mean ARI vs truth |
|---|---|---|---|
| high | 16 | 69% | 0.707 |
| moderate | 15 | 27% | 0.267 |
| low | 29 | 3% | 0.088 |

**Two properties held across all sixty**, and are now locked by
`test_the_confidence_light_never_invents_groups_and_never_greenlights_weak_data`:

* **It never over-counts.** Zero studies of sixty reported more groups than were planted. Every
  error was a merge. That is the honest thing to tell a user: a green light can mean fewer groups
  than really exist, never invented ones.
* **It never greenlights weak data.** Zero "high" across the thirty-six studies at the two weakest
  separations.

**What it does not promise:** high confidence accompanied a merged answer in 5 of 16 green runs.
Checking my own instrument mattered here — every one of those five had two planted centres within
about one noise standard deviation per question, i.e. groups that genuinely overlap, and at an
identical separation the tool got it right with 400 people and wrong with 80. So it is a
sample-size effect at real but modest separation, not a broken light. **Do not tighten the light to
catch them** without repeating the sweep: the property worth more is that it does not go green on
weak data, and that is the one a tighter threshold would trade against.

One test was renamed as a result. `test_it_never_claims_high_confidence_for_the_wrong_number_of_
groups` checked a single centre configuration while its name claimed a universal property the
sweep falsifies; it is now `test_this_overlapping_shape_drops_to_moderate_when_it_merges`.

### The performance figures are data-specific, and re-measuring them nearly produced a false alarm

Re-checked 2026-08-13 because a lot of code has gone into the analysis path. A synthetic 48,842 x 5
file took **2.20 minutes** against the documented **0.9**, which looks like a 2.4x regression.

It is not one. Running the **identical fixture** against a worktree at `v1.5.7` — the release the
figure was written for — gave **2.25 minutes and 1.26 GB**, against the current 2.20 minutes and
1.20 GB. Today's code is marginally faster and lighter. The whole gap is the data: the documented
0.9 was the California housing file, whose distribution the k-search settles on faster than three
cleanly planted groups.

Two things to carry forward. **Quote the dataset with any timing**, or the number becomes a trap for
whoever re-measures it. And when a benchmark looks like a regression, A/B the same fixture against
the old tag before believing it — `git worktree add /tmp/old <tag>` costs a minute and settles it.

### The demographics section was silent about every number in it

Found while auditing the opposite claim — that demographics cannot influence the grouping, which
holds. The profiler branched on "non-numeric OR twelve or fewer distinct values" and did nothing
at all with what fell through, so age in years, income and tenure were never tested and never
mentioned. On a study whose segments were 25 and 55 years old, `age` appeared nowhere.

**The failure mode is the one this project treats as most serious: doing less than the reader
believes.** The heading promises profiling against demographics and the list looks complete.

Now Kruskal-Wallis for numerics, inside the same FDR correction as the chi-squares — correcting
only part of a family understates it — and the per-segment median printed beside each p-value,
because "differs" without "which way" is not actionable. If you add another column type here, add
it to the same `pvals` dict rather than a parallel one, or the correction silently stops covering
everything.

### run_analysis discarded the caller's method, and how that surfaced

`auto_prepare` decides what the data supports, and `replace(cfg, **_opts)` then overwrote whatever
was asked for — so `run_analysis(cfg=SegmentationConfig(method="gmm"))` completed happily and
reported `method: kmeans`. The CLI was always fine; only the app's path had it.

**It surfaced as a measurement that was too clean.** Comparing k-means against gmm across five
separations and two seeds returned identical ARI to three decimal places every time. Two methods do
not agree to three decimals ten times running — that is one method running twice. Worth keeping as
a habit: when a comparison comes out perfectly clean, suspect the comparison before believing it.

The fix honours an explicit method where the data allows it and **says so in the report** where it
cannot, rather than substituting silently. "The caller asked" is told from "the caller did not ask"
by comparing against a fresh `SegmentationConfig()`, because the dataclass has no way to express
"unset" and giving it one would change what every other caller means.

### Probe every endpoint you add, the same evening you add it

Three endpoints went in across 1.15.0 and 1.16.0 (`/rename`, `/pin`, `/delete_projects`). Probing
them with the shapes a caller most easily gets wrong — the discipline established when `/design`
was added — found one defect and confirmed one guarantee:

* **A non-string project id** reached the generic catch-all, answering "Something went wrong while
  reading or analysing that file" about a file nobody sent. Fixed in all four store-touching
  handlers via one shared `_project_id` check, so they cannot drift apart.
* **Path traversal is already neutralised.** `_stem()` strips an id to alphanumerics before it
  becomes a filename, so `../../etc/passwd` resolves to nothing on every path including bulk
  delete. That is now a test rather than a property of one regular expression nobody re-reads.

The probe is cheap and has now found something every time it has been run. Run it on the next
endpoint too.

### Bulk delete, and the rules it follows

1.16.0 added multi-select delete. It removes the analysis and the original upload with no undo, so
the constraints are the feature and should not be relaxed for convenience:

* **Two clicks, and the confirming button names the count** — "Delete 12", never a bare "Delete".
* **"Select all" covers the rows the search left, and nothing else.** Not rows scrolled away, not
  rows behind the sixty-project cap. A bulk action must never reach what the user cannot see.
* **Leaving select mode drops the selection**, so a set ticked earlier cannot act later.
* **The server reports what it actually removed**, not what was asked for, and skips ids that are
  already gone rather than failing the batch.

`ids` sent as a bare string is refused. That is the third appearance of the same coercion — after
`/design` and `/name` — and on a path that deletes files it is the worst of the three: iterating
`"keep2"` would have produced five single-character ids. A test covers it explicitly.

### The sidebar was hiding a hundred projects

`ProjectStore.list()` caps at sixty and always has. Nothing said so, which is indistinguishable
from projects having been deleted — and on the real workspace here it was showing **60 of 162**.
Found while adding pinning, not by looking for it, which is the usual way: the count only became
visible because something else needed it.

Two rules came out of it and are in the code:

* **A pinned project is never cut off by the cap.** `list()` sorts pinned first *before* slicing,
  because a pin that only reorders the visible sixty fails at exactly the point someone starts
  pinning — when the list is long.
* **Every reply that hands out the list also carries `total`.** Four endpoints return the project
  list; if one of them forgot, the "showing 60 of 162" line would go stale the moment you renamed
  or deleted something.

### The projects list, and a trap in verifying UI

1.15.0 gave the sidebar rename, search and date grouping. The rename endpoint writes the title to
**both** places it lives — the full record and the small `.meta.json` the sidebar reads so it does
not parse a megabyte of report per row. Writing one and not the other leaves a project called two
different things depending where you look, which is what the test checks.

**The trap is in how UI gets verified, and it cost half an hour.** Driving the app through the
browser tool, rename appeared to be completely broken: typing into the field worked, but Enter and
Escape did nothing, and no `/rename` request was ever made. The endpoint was fine when called
directly, the bundle was current, and the component tests passed.

The cause was the instrument. **The browser tool's synthetic key events do not reach React's root
event listener**, so `onKeyDown` never fired — while `type` still updated the field, which is what
made it look like a real bug. A `KeyboardEvent` dispatched from `javascript_tool` does work, and
with that the whole flow completed and `POST /rename → 200` appeared in the network log.

So: **for React key handling, verify with `javascript_tool`-dispatched events or with the component
tests; the `computer` key action proves nothing.** Clicks are fine — those work. Third time this
session an instrument was the fault rather than the code, which is the pattern to expect.

### The typing tool, checked by holdout

The last major claim with no external check. Build the rule on 70% of a planted study, score the
30% it has never seen, thirty-six times:

* **Agreement with what clustering everyone together would say: 98.7%** where groups genuinely
  separate, 93.5% overall. That is the promise the rule actually makes.
* **No overfitting at all** — accuracy against the planted truth was 0.9 points *higher* on withheld
  people than on the people the rule was built from. Expected of a nearest-centroid rule, but worth
  holding as a fact.
* At weak separation agreement falls to 83%, and that is correct behaviour: the segmentation itself
  matched the truth only 51% of the time on those studies, so the rule is faithfully reproducing an
  unreliable answer rather than adding error of its own.

**The finding was in the presentation, not the statistics.** The app printed "Average confidence
0.58" with no scale. That number runs from 1/k, not from 0, so its floor moves with the group
count: two-group studies averaged 0.586 and three-group studies 0.443, which reads as a large
difference and is not one — both sit about 17% of the way up their own range. The floor is now sent
with the figure and shown beside it. **Do not "simplify" that back to a bare number.**

### Contrast: the palette was never measured, on either ground

Changing the light ground to white was the prompt, but the finding was older than the change.
Three pairings sat below the 4.5:1 WCAG AA line for body text, and `--muted` — which carries every
hint and caption in the interface, at small sizes — had been failing on the beige ground too. That
is how a palette drifts: nobody measures the colour that was inherited.

Fixed to 5.44 / 4.88 for muted, and two smaller lifts for `--ok` and `--accent-soft`. Dark mode
already cleared AA everywhere and was left alone. `test_every_text_colour_in_the_interface_clears_
wcag_aa` parses the stylesheet and checks every pairing the interface actually paints, so this
cannot regress quietly. **If you change a colour, run that test rather than eyeballing it.**

### PRIVACY.md did not mention that images are sent

Found by reading the document against the code rather than on its own. `ai_interpret.py` attaches
three charts as PNGs when the Claude layer is used, and the code is scrupulous about it — its
docstring notes that the segment map plots one mark per respondent and argues why that is not
re-identifiable. **None of that was in PRIVACY.md**, whose table listed only the digest as leaving
the machine, in the one document that exists so this question has a written answer rather than a
recollection. Now disclosed, including the per-respondent point and the advice to leave the layer
off if a data protection assessment treats any per-individual representation as personal data.

### GitHub Actions: RESOLVED 2026-08-08 — the repository is public and CI is green

For about a day every run failed **before starting a single step**: while the repository was
private, Actions minutes were metered against a monthly allowance, running the full matrix on every
push exhausted August's in seven days, and GitHub halts jobs rather than billing. Making the
repository public made minutes free and the block disappeared.

**First green run: 2026-08-08.** The full matrix passed — Ubuntu on 3.9, 3.11 and 3.12, macOS on
3.11 and 3.12, plus the frontend and packaging checks. That is the first CI evidence since v1.5.8,
and the first time this code has ever been tested on the platform it actually ships on.

While it was down, everything from v1.5.9 to v1.7.2 shipped on local runs alone — and for several
hours of that this file said "CI green" on the strength of a stale reading. **A green memory is not
a green run, and `gh run list` costs one command.** The workflow was also trimmed so an ordinary
push runs two jobs rather than five, which is what keeps it affordable if the repository is ever
private again; see the header of `ci.yml`.

The matrix now covers both platforms deliberately, and both have earned it. Linux found a locale
bug that made the tool unrunnable with no `LANG` set — something macOS can never reveal, because it
always reports UTF-8. And the failures only macOS can show, around case-insensitive paths and the
signing and quarantine behaviour of the app bundle, had no coverage at all until 2026-08-08.

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

### The study planner (added 1.8.0, in the app since 1.9.0)

`segment-kmeans --plan`, and a panel on the app's start screen. Answers "how many respondents do we
need" by planting a known number of segments in simulated surveys and running the REAL pipeline
over them at a range of sample sizes — not by a power formula, which would have to assume the
clusters are spherical and well separated when this tool's measured weakness is precisely
overlapping ones.

Three defects were found auditing it hours after it shipped, and they are worth knowing about
because two of them made it look authoritative while measuring the wrong thing: the separation
figure was documented as Cohen's d but delivered 2.83 when asked for 2.0; the default sweep ran
100-800 respondents when recovery actually turns over between 40 and 150, so most columns were
identical; and a design that cannot fit the answer scale was silently clipped rather than refused.
A fourth showed up when it reached the app — requiring every larger sample to clear the bar as
well was too strict for a noisy sweep and suppressed the recommendation entirely.

**Measured and worth not re-deriving:** at 100 people with subtle differences, roughly one
simulated study in ten reports the wrong number of segments while still calling it high confidence.
That is not a contradiction — the tool judges whether a grouping REPRODUCES, and on heavily
overlapping data a merged pair reproduces perfectly well — but it is disclosed in the planner's
output rather than buried.

### No open findings

The `group_profiles.csv` naming defect recorded here previously is fixed and released. Every finding
raised during this audit is either fixed or written down below as a known limit.

### The three evidence gaps — all now resolved

These were never bugs; they were gaps in the evidence, and they were the honest reason the earlier
readiness verdict was bounded rather than unconditional. As of 2026-08-08 all three are closed or
retired, which removes that qualification.

1. ~~**The live Claude API round-trip has never been run.**~~ **CLOSED 2026-08-08.** Tom funded an
   API key and ran it. On a 270-person file with three deliberately obvious mind-sets planted in
   it, the naming came back **Premium Brand Devotees / Spur-of-the-Moment Explorers / Bargain
   Researchers** — all three correct, and the third picked up both planted traits (price
   sensitivity *and* researching before buying) rather than the obvious one only. Key, request,
   response and the write into `group_names.csv` all work.

   The assistant still never reads or uses the key: this was verified by checking the stored
   value's SHAPE (prefix and length) and by Tom running the button himself.
2. ~~**HB MaxDiff has never seen a real best-worst export.**~~ **CLOSED 2026-08-07.** The
   `bwsTools` example data from CRAN — 350 real respondents asked which issues facing the country
   matter most and least — now goes through it. Against the classical best-minus-worst score on the
   same data: Spearman 1.0000, Pearson 0.9986. It also independently found three segments, the
   number that package's own authors report. Reading it required two fixes; see CHANGELOG 1.6.1.
   A **Qualtrics/Sawtooth wide export** — one row per person — is still unseen as a real file,
   but its shape is now recognised and refused with instructions rather than silently clustered as
   ratings, which is what it did before (see CHANGELOG 1.6.2). Reading one properly needs a real
   sample, because the code polarity cannot be inferred from the data.
3. ~~**Nobody has hand-clicked the Windows build.**~~ **NOT A GAP — the tool is for macOS.**
   Confirmed with Tom on 2026-08-08: this is a Mac tool and Windows is not a target. CI still
   builds a Windows artefact when it runs, and the smoke test still drives it, but nobody needs to
   hand-click it and its absence blocks nothing.

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
- **Re-run `python3 references/kbench.py` after touching `recommend_k` OR the confidence light.** Re-run 2026-08-13: the 19-of-21 headline held, but two confidence columns in `ONBOARDING.md` had drifted toward more caution than documented — the guards added to the light since are why. Correct the table from the run rather than reasoning about it. The documented accuracy
  table goes stale silently otherwise.

### What is genuinely left

Nothing broken that I know of. The remaining gaps are all blocked on something other than code:

| | |
|---|---|
| Live Claude API round-trip | **Verified 2026-08-08** end to end, with correct segment names returned. The no-key path still degrades cleanly |
| HB MaxDiff on real responses | Validated on 350 real respondents: Spearman 1.0000 against the classical score. A real wide (Qualtrics/Sawtooth) export is still unseen; its shape is refused with guidance rather than misread |
| The Windows build | Not a target — macOS is the platform. CI still builds it; nothing depends on it |

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
and 0.9 **on the California housing file** — that figure is specific to that data and is not a
general benchmark, see the note below; 581,012 rows now run in 3.2 minutes), seven file shapes that were read wrongly and
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
| CI | Ubuntu 3.9 / 3.11 / 3.12 and macOS 3.11 / 3.12, plus frontend and packaging checks — green |
| Tests | 234 Python (`pytest`) + 115 frontend (`cd frontend && npm test`) |
| Shipped app | **v1.11.1**: macOS `.app` (82 MB), built, signed and smoke-tested locally. Never in git history — it lives in GitHub Releases. The team's copy lives in `~/Desktop/Survey Segmenter (app for the team)/`. |
| Local path | `~/dev/survey-segmenter` — **moved out of iCloud Drive**, see below |

```bash
cd ~/dev/survey-segmenter
pytest                  # 234 tests
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
- ~~**Sidebar hides below 820px.** Offered a toggle; never requested.~~ **Fixed in 1.17.1** — and it was worse than a preference by then: four releases of project features (rename, search, pinning, bulk delete, show-all) all live in that panel, so a narrow window meant none of them existed. It is a drawer now, opened from the header, closed by choosing a project, tapping the page, or Escape.
- **Do not put this checkout back inside iCloud Drive.** It lived under `~/Desktop`, which macOS
  syncs by default, and with ~6,000 files in `node_modules` every read went through the sync
  layer. `npm test` took 37 minutes instead of one second, Vitest workers timed out before they
  could start, `npm run build` hung indefinitely, and macOS left `App 2.tsx` conflict copies in
  the source tree. Nothing was ever wrong with the code. Now at `~/dev/survey-segmenter`:
  59 frontend tests in 1.1s, build in 266ms.

## Next candidates, roughly by value

1. **Read a real Qualtrics/Sawtooth wide export.** Its shape is recognised and refused with
   instructions rather than silently misread, which was the dangerous behaviour. Reading one
   properly is blocked on having a real file: the code polarity cannot be inferred from the data.
2. ~~**Resolving overlapping segments**~~ — **answered in 1.19.0, and the answer is to leave it
   alone.** Merging IS the honest answer: on planted overlapping spherical groups the Gaussian
   mixture, which the README offered as the remedy, never beat k-means at any separation tried, and
   was notably worse at the separation where k-means starts to recover the third group. The
   mixture's real strength is elliptical clusters (0.954 against 0.872), and it degrades badly on
   unequal spreads by over-splitting. Do not reopen this without new evidence of a *different*
   kind — the property being protected is that the tool is never confidently wrong, and every
   attempt to sharpen overlapping segments trades against it.
3. **Duplicate a project** to re-run the same file with a different set of questions. Partly
   covered already by the column picker's re-group, which is why it is third rather than first.

Three former entries are settled and should not be reopened without reading why: a **second
cluster-tendency test** shipped (Hartigan's dip, alongside Hopkins); **sparse k-means** was built,
measured and deliberately rejected (see `references/sparse_kmeans.py` and `STATE-OF-THE-ART.md`);
and the assistant memory file **`segment-kmeans-tool.md` has been consolidated** — it had reached
58 KB of appended paragraphs and had gone stale enough to point at the pre-move directory and claim
49 tests when there were 279. It is now 3 KB that points here instead. **Do not let it grow back:**
a memory that restates the repo will always lose to the repo, and the only things worth keeping
there are the ones this file cannot say about itself.

## How the modules divide up

`segment_kmeans.py` decides what a segmentation is and what it means. `charts.py` draws it.
`webapp.py` delivers it. `maxdiff.py` scores best-worst data before any of that happens, and
`kprototypes.py` supplies the distance and the prototypes for questionnaires that mix rating
scales with pick-any questions.

Three modules sit outside that chain because they run before or after a segmentation rather than
inside one: `design.py` builds the questionnaire, `planner.py` decides how many people to recruit,
and `turf.py` turns a ranking into a launch shortlist. None of them import the engine and the
engine does not import them — which is also why all three are imported lazily, and why each needs
a `--hidden-import` line in `build_app.py`. See v1.11.1 above for what that costs when forgotten.

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
