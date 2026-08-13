# Changelog

Notable changes to Survey Segmenter. Versions follow [semantic versioning](https://semver.org/);
the version is set in `pyproject.toml` and stamped into every report footer.

## [1.19.1] — 2026-08-13

### Verified

Three claims the tool makes about itself, checked against a known truth rather than re-read. All
three hold; the point of the exercise is that they are now tests instead of sentences.

- **The k-selection benchmark still scores 19 of 21**, re-run because the handover warns this table
  goes stale silently. Two of its *confidence* columns had drifted, both toward more caution than
  documented: overlapping k=3 was written up as "high when right" and is now moderate on all three
  runs including the correct one, and the two elongated bands as "high" and are now high once,
  moderate twice. `ONBOARDING.md` is the first document a new reader is pointed at and it was
  overstating how confident the tool is, so it has been corrected and dated.

- **Demographics genuinely cannot influence the grouping they are used to profile.** If they could,
  every demographic difference in the report would be circular — grouping people by age and then
  announcing the groups differ by age. Checked with a decoy demographic that separates people
  perfectly along a *different* split from the one in the answers: agreement with the real
  structure 1.000, with the decoy −0.003.

- **A survey weight projects the segment sizes without shaping the segments.** Getting this
  backwards is subtle and expensive: weighting the distances would let a few heavily-weighted
  people drag a segment centre around, while leaving the weight out of the sizes would report
  "19% of our customers" for a group that is 79% of them. On a sample that is 81% one group with
  weights saying the population is 79% the other, both figures appear and both are right.

## [1.19.0] — 2026-08-13

### Fixed

- **A requested method was thrown away without a word.** `run_analysis` — the path the app uses —
  read the questionnaire, decided what the data supported, and then overwrote whatever the caller
  had asked for. A request for `gmm` came back as a completed run whose report said
  *"method: kmeans"*. The command line never had this problem; only this path did.

  The project's own comment on `--force-k` states the principle it was breaking: an ignored flag
  that reports success is worse than an unimplemented one, because the reader believes the answer
  is the one they asked for.

  An explicitly requested method is now used where the data allows it. Where it does not — a
  Gaussian mixture over categorical answers is not defensible whoever asks — the detection stays in
  charge and the substitution is **stated in the report** instead of made silently.

  Found while trying to measure something else, which is how the last several have turned up: the
  two methods returned byte-identical results at five separations and two seeds, and that is not
  what agreement looks like.

### Changed

- **The README's advice about `--method gmm` is now measured, and narrower than it was.** It was
  offered as the remedy for k-means's honest limitation — "elliptical, unequal-size, overlapping".
  Against planted truth, three seeds at 400 respondents:

  | planted group shape | k-means | gmm |
  |---|---|---|
  | elongated, similar spread | 0.872 | **0.954** |
  | elongated, one group twice as wide | 0.695 | 0.668 |
  | elongated, one group three times as wide | 0.340 | **0.147** |

  It earns its place on elliptical clusters and gets **worse** as spreads become unequal, spending
  the extra flexibility on splitting the broad group rather than modelling it — 3.7 groups found
  where there were 2. And on overlapping *spherical* groups it never beat k-means at any separation
  tried, so it is not the answer to two segments that have merged.

  That last point settles a question the handover has carried for months as "the measured
  weakness": when the tool merges two overlapping groups, **merging is the honest answer**, and the
  documented alternative does not recover them.

## [1.18.0] — 2026-08-13

### Added

- **⌘K (Ctrl-K on Windows) jumps to the project search.** With a hundred and sixty projects the
  search is the way in, and reaching for it with the mouse is the slow half of the interaction. On
  a narrow window it opens the projects drawer first, since otherwise there is nothing to focus.

- **Escape clears the search** while the cursor is in it. Emptying the filter is what Escape is for
  when you are standing in one — and it stops there rather than also closing the drawer behind it.

## [1.17.1] — 2026-08-13

### Fixed

- **On a narrow window the projects panel was unreachable, not just hidden.** Below 820px the
  sidebar was `display:none` with nothing anywhere to bring it back — so every saved study, and
  everything added to that panel across the last four releases (rename, search, pinning, bulk
  delete, "show all"), simply did not exist on a smaller window.

  It is now a drawer: a **Projects** button appears in the header at exactly the widths where the
  column collapses, and slides the panel over the page. Choosing a project closes it, since on a
  narrow window the drawer covers the result it was opened to reach. Tapping the page behind it
  closes it, and so does Escape — anything that covers the page needs a way out that does not
  depend on hitting a particular target.

  The button reports its state through `aria-expanded` and names the panel it controls, so this is
  navigable without sight of the animation.

## [1.17.0] — 2026-08-13

### Added

- **"Show all" reaches the projects behind the cap.** 1.15.1 made the sixty-project limit visible
  ("Showing the 60 most recent of 162"), which was the honest half of the fix. This is the useful
  half: the count is now a button. Until now the older hundred were reachable only by remembering a
  name and searching for it, which is no way to find something you have forgotten.

  Once the whole archive has been asked for it stays that way for the session — refetching after a
  rename or a delete would otherwise snap the list back to sixty, which looks exactly like the
  older projects having just been deleted. That is the confusion the count was added to remove, so
  reintroducing it through the back door would be worse than not having the button.

  The order is identical in both views — pinned first, then newest — so "show all" reads as more of
  the same list rather than a different one.

## [1.16.1] — 2026-08-13

### Fixed

- **A malformed project id was explained as a problem with a file.** Sending a list or a number
  where the project id belongs fell through to the generic handler, so the reply read *"Something
  went wrong while reading or analysing that file"* — about a file the caller never sent. All four
  handlers that reach into the project store now check the id first and answer about projects.

  Found by probing the three endpoints added in 1.15.0 and 1.16.0 rather than by using the app,
  which is the only way it could be found: the interface cannot produce those shapes. The same
  probe confirmed the good news — **a project id cannot reach outside the project folder**, on
  rename, pin, delete and bulk delete alike, and that is now a test rather than a property of one
  regular expression nobody re-reads.

### Verified

- **The whole bulk-delete flow works with the keyboard alone** — enter select mode, tick rows with
  space, confirm — and every control that is only a glyph on screen (`×`, `⋯`, `☆`) carries its own
  spoken label, with the star reporting its state through `aria-pressed` rather than only by being
  filled in. A destructive action reachable only by mouse would be a worse regression than not
  having the feature.

- **The contrast check now covers the sidebar's own ground**, not just cards. The search hint, the
  date headings and the red "Delete N selected" link all sit on `--surface`, which the earlier
  check never looked at. All pass: 5.44, 6.61 and above against a 4.5 requirement.

## [1.16.0] — 2026-08-09

### Added

- **Clear out several projects at once.** "Select" beside the search turns the list into
  checkboxes; tick what you want gone and delete it in one go. Clearing a hundred throwaway runs
  one arm-and-confirm at a time is the kind of chore that means they never get cleared.

  Three deliberate limits, because this removes the analysis *and* the original upload with no
  undo:

  * **Two clicks, like the single-row delete**, and the confirming button names the number:
    "Delete 12", not "Delete".
  * **"Select all" means the rows the search left**, never rows scrolled out of view or hidden
    behind the list cap. A bulk action must not reach anything the user cannot see.
  * **Leaving select mode drops the selection**, so a set ticked five minutes ago cannot act later.

  While selecting, clicking a row picks it rather than opening it — opening a project mid-selection
  would swap the whole page out from under a half-made choice.

  Server-side it is one request rather than one per project, and an id that is already gone is
  skipped rather than failing the batch: half a delete is worse than either outcome. The reply says
  how many were **actually** removed rather than how many were asked for. A malformed request —
  ids sent as a bare string rather than a list — is refused, because iterating a string into
  single-character ids on a path that deletes files is the worst version of a bug this project has
  already found twice elsewhere.

## [1.15.1] — 2026-08-09

### Fixed

- **The projects list silently hid everything past the sixtieth.** The sidebar shows the sixty most
  recent projects and said nothing about the rest, which looks exactly like projects having been
  deleted. Found while adding pinning, on a real workspace: **the list was showing 60 of 162.** It
  now says "Showing the 60 most recent of 162", so the cap is a fact rather than a disappearance.

### Added

- **Pin a project to the top.** The star on a row keeps it in a Pinned section above the date
  groups, and — the part that matters — **a pinned project is never cut off by the cap**. A pin
  that only reordered the visible sixty would fail at exactly the point someone starts pinning
  things, which is when the list gets long.

  The pin state is sent rather than toggled, so two windows open on the same app cannot disagree
  about which way the switch was pointing.

## [1.15.0] — 2026-08-09

### Added

The projects list was a flat column of filenames. That is fine for five projects and unusable for
sixty, which is what a real workspace becomes — `export (3).csv`, and four copies of `s.csv` that
cannot be told apart. Three things, in the order they were missed:

- **Rename a project.** The name it arrives with is the file's name, not what the study was.
  Click the `⋯` on a row, or double-click the row. Enter saves, Escape abandons the edit. Names are
  collapsed to one line and capped at 80 characters so they fit the row they have to live in.

- **Search.** Filters by name as you type. It only appears past a handful of projects — a filter
  box above three rows is furniture.

- **Grouped by when you last touched them** — Today, Yesterday, Previous 7 days, Previous 30 days,
  Older. Empty buckets are not rendered, so a workspace where everything happened today shows one
  heading rather than five saying nothing. A project with a missing or unreadable timestamp goes
  into "Older" rather than disappearing.

One behaviour worth naming because it was found by using the thing rather than by reading it:
renaming a project while a search is active used to make the row vanish, since the name you
searched for is the one you just replaced — the list emptied to "Nothing matches". The search now
clears itself when the new name would hide the row you just acted on.

## [1.14.1] — 2026-08-09

### Fixed

- **The scoring confidence was printed without the scale it sits on.** "Average confidence 0.58"
  looks like a figure out of 1. It is not: scoring confidence is an inverse-distance share on the
  winning group's centre, so it runs from **1/k** — equidistant from every group, a coin toss — up
  to 1, and the bottom of the scale moves with the number of groups.

  Measured over thirty-six holdout studies, two-group runs averaged **0.586** and three-group runs
  **0.443**. A reader would take the first for the better-typed study. They are the same: 17.1% and
  16.4% of the way up their respective ranges. The app now prints the floor beside the figure.

### Verified

- **The typing tool works as well on strangers as on the people it was built from**, checked by
  holdout rather than asserted: build the rule on 70% of a planted study, score the 30% it has
  never seen. Agreement with what clustering everyone together would have said was **98.7%** where
  the groups genuinely separate and **93.5%** overall, and accuracy against the planted truth was
  0.9 points *higher* on withheld people than on training people — no overfitting to find.

  Where groups barely separate, agreement falls to 83%. That is the rule faithfully reproducing an
  unreliable segmentation, not a fault in the rule: on those same studies the segmentation itself
  matched the truth only 51% of the time.

## [1.14.0] — 2026-08-09

### Added

- **The confidence light now says what it is worth.** "Trust these groups" is the sentence people
  act on, and on its own it invites a reader to treat the number of groups as exact — the one thing
  a sixty-study measurement does not support. A quiet "What does *High* actually mean?" now sits
  under the result cards and answers it from evidence: a green light found the right number about
  seven times in ten, never invented groups that were not there, and when it is wrong it has merged
  two real groups — so the number is a floor, not a headcount. Amber and red get their own equally
  concrete accounts.

  Placed below the card row rather than inside the confidence tile: the tiles are equal-height, so
  three lines of caveat in one of them stretched all three, left "Groups found" as a tall empty
  box, and squeezed the text into a third of the width.

### Fixed

- **Three text colours were below the WCAG AA line for body copy.** Found by measuring the palette
  after the ground changed from beige to white — a palette tuned against one ground is not
  automatically legible on another. `--muted`, which carries every hint and caption in the
  interface at small sizes, was at 4.49:1 on white and 4.03:1 on the sunk tone against a 4.5:1
  requirement, and it had been failing on the old beige ground too. It is now 5.44 / 4.88. `--ok`
  on its own tint went 4.25 → 4.74, and accent-coloured text on a hovered button 4.45 → 4.66.

  Dark mode already cleared AA on every pair and is untouched. A test now parses the stylesheet and
  checks every pairing the interface actually paints, so the next colour change cannot fail this
  silently.

## [1.13.3] — 2026-08-09

### Changed

- **The light theme is white, not beige.** The paper-toned ground read as decoration the tool had
  not earned. Everything that used to separate by tint now separates by border — cards, notes and
  stat tiles already carried a 1px line, and the header and sidebar their own edge — so flattening
  the grounds to white cost no hierarchy.

  Changed in all four places it lived, not just the obvious one: the interface tokens, the
  no-interface fallback page, and **the charts**, whose background tone is baked into every
  exported PNG and is also the colour used to ring overlapping marks so they read as separated. A
  beige chart on a white page would have looked like a bug.

  The remaining neutrals — the line colours, and the tone behind table stripes and inline code —
  keep a slight green bias toward the accent, so they read as chosen rather than as default grey.
  Dark mode is untouched; it was never beige.

## [1.13.2] — 2026-08-09

### Fixed

- **The v1.13.1 tag could not build itself.** Raising TURF's minimum item count made the build's own
  smoke test fail on a five-item fixture, and the fixture fix landed one commit *after* the tag, so
  CI rebuilding that tag hit the failure that `main` had already fixed. The macOS artefact attached
  to v1.13.1 was built locally from the corrected tree and is sound; the tag simply could not
  reproduce it.

  This is the third time this session that tagging before building has caused a mess. The order is
  **test, build, then tag** — a tag is a claim that the commit under it works, and it is not worth
  making before that has been checked.

  The smoke test's failure message now also names the other possible cause, because it blamed
  turf.py for not being bundled when turf.py was bundled perfectly well.

## [1.13.1] — 2026-08-09

### Fixed

- **Every best-worst estimate ever run on a Mac told the user it had overflowed.** The sampler
  printed "divide by zero", "overflow" and "invalid value" — all three, on every study — and the
  arithmetic was correct every single time. On macOS with numpy 2 on Apple's Accelerate BLAS, an
  ordinary matrix multiply of small finite numbers raises all three while returning a finite
  product: the vectorised kernel raises the exception flags from padding lanes holding no data.
  Reproduced with nothing more than `standard_normal((80, 9)) @ standard_normal((9, 9))`.

  macOS is the platform this ships on, so this was a warning shown to every command-line user about
  a problem that did not exist. The flags are now ignored inside the sampler — and, because
  suppressing a warning blindly is how a real divergence becomes silent, the draws are explicitly
  checked for finiteness afterwards and a genuine failure is refused in plain words. That check is
  the assurance the warnings were never actually providing.

### Verified

- **The ranking's "probability this beats the next item" is honest**, checked against a known truth
  rather than against itself. Simulating studies from chosen population utilities, running the real
  design and the real estimator, and asking how often the item placed above another really was
  above it: over 126 adjacent pairs from fourteen studies the table claimed **84.3%** and was right
  **83.3%** of the time, tracking bin by bin. A miscalibrated probability would be worse than none,
  because it gets quoted as though it were one. Now locked in by a test.

## [1.13.0] — 2026-08-09

Three defects in the "which few to launch" answer, all found by auditing it against an outside
truth rather than against itself. Each changes what the report says, so this is a minor version
rather than a patch.

### Fixed

- **The launch recommendation could be decided by the order of your spreadsheet.** Reach counts
  people, so it can only land on multiples of 1/n — sixty respondents give sixty-one possible
  values for hundreds of candidate combinations, and exact ties at the top are ordinary rather than
  freakish. The best reach was shared by more than one set in **14 of 30** simulated studies at
  sixty people and ten items. Whichever tied set came first in the item list was printed as the
  finding: **reordering the item list changed the recommendation in 8 of 25 studies**, with
  identical reach both times.

  No tie-break fixes this, because every tie-break is arbitrary. The report now says how many sets
  tie, names some of them, and tells the reader to choose on grounds the survey does not contain —
  cost, margin, brand fit. The two search paths also disagreed with each other about ties: greedy
  resolved to the highest item index and exhaustive to the lowest, so which of two equal items you
  got depended on how long your item list was. They now agree.

- **The "how much of this is luck" figure measured the wrong thing.** It was `in_sample - holdout`,
  and both of those are half-sample quantities: it measured the optimism of a study half the size
  and attributed it to the full-sample headline. The report printed *"Expect about 93%, not 95%"*
  directly above *"the 3-point difference"* — two sentences that do not agree, with the 96% those
  three points came from appearing nowhere in the report.

  It is now `reach - holdout`: the gap between the two numbers the reader can actually see.
  Checked against a 40,000-person population where the true reach is known, the corrected gap
  tracks the headline's real error (pure noise, 100 people, 20 items: reports 14.8 points against a
  real 14.0). The module's documented table has been re-measured against that truth and replaced —
  the old one claimed 9.5 to 22.3 points, computed with the wrong instrument.

- **TURF ran on item lists too short for it to mean anything.** Someone counts as reached when the
  item is in their own top three, so each person accepts three of however many items exist. At five
  items that is 60% of the list, and the best set of three reached **100.0% in every study tried**,
  with ten sets tied for it. The section is now refused below nine items, where the answer stops
  being arithmetic. Measured best reach for a set of three: 5 items 100.0%, 8 items 98.4%,
  10 items 95.1%, 20 items 83.9%.

### Added

- **"2 of these 3 would do."** When the items after the first already reach everybody the earlier
  ones do, the report now says so in the headline rather than leaving it implied by a column of
  `+0 points`. Recommending three items when one does the job is an expensive way to be right.

## [1.12.2] — 2026-08-09

### Fixed

- **Segment names were coerced the same way, in the place it matters most.** Having found the
  defect in the new design endpoint, the obvious question was where else it lived — and `/name` had
  it. Sending `"AB"` instead of `["A", "B"]` named the two segments **A** and **B**; a list holding
  an object named one of them `{'x': 1}`.

  This one is worse than the design case, because a segment name is free text: nothing downstream
  can tell a coerced name from a real one, so it lands in `group_names.csv`, in
  `segment_assignments.csv`, and in whatever a colleague opens next. Names must now be a list of
  text, and internal whitespace is tidied.

  `/plan` and `/regroup` were checked for the same thing and are sound — the planner's bounds catch
  a coerced number, and re-grouping drops column names that do not exist and refuses when nothing
  real is left.

## [1.12.1] — 2026-08-09

### Fixed

- **The design endpoint coerced malformed input into a plausible-looking questionnaire.** Found by
  probing it adversarially straight after building it, not by a user. Sending the items as a bare
  string rather than a list made the server iterate it letter by letter and build a study asking
  people to choose between `d`, `e`, `l`, `i`, `v`, `r` and `y` — reported, with a straight face, as
  a well-balanced seven-item design. A list containing objects or nulls produced items literally
  named `{'a': 1}` and `None`. A single 50,000-character item turned a four-item design into a
  one-megabyte reply.

  All three are now refused with an explanation rather than coerced, and whitespace inside an item
  is collapsed, because an item carrying a newline splits a cell in most survey platforms and
  corrupts the import rather than merely looking wrong. Coercion is the wrong instinct here: it
  costs a message to refuse and a whole study to accept.

## [1.12.0] — 2026-08-09

### Added

- **The questionnaire designer is in the app.** It was command-line-only for one release, which in
  a tool built for people who do not use a command line means it may as well not have existed —
  the same half-delivered state the planner was in before 1.9.0. It now sits on the start screen
  next to the planner, above it, because that is the order the work happens in: you write the
  questionnaire, then decide how many people to put in front of it.

  Items are **pasted, not uploaded**. Someone deciding what to ask has the list in an email or a
  slide; making them save a `.txt` first was a step that existed only because the command line
  needed one. Lines listed twice are collapsed, case-insensitively, so an item cannot end up
  competing against itself on the same screen.

  The app caps the shape at 40 items, 6 per screen, 15 screens and 300 versions. That ceiling is
  measured rather than guessed: it is the largest shape that still answers in about twenty seconds,
  while eight items across twenty screens takes eighty. `segment-kmeans --design` has no ceiling
  and remains the right tool for anything bigger.

### Changed

- **The design search got about twice as fast**, and much more than that on long item lists.
  Rebuilding a standard deviation over the whole pair matrix cost `O(items²)` for every candidate
  swap while the swap itself touched a few dozen entries — sixty items over five hundred people
  spent five minutes almost entirely re-reading numbers that had not changed.

  A swap moves items between screens without changing how many pairs a screen contains, so the
  total of the pair counts is invariant; with the mean fixed, minimising the variance is exactly
  minimising the sum of squares, which each changed entry can update in constant time. Measured, at
  identical balance in every shape tried: 12 items/200 people 2.4s → 1.3s, 30 items/500 people
  35s → 20s, 60 items/500 people 300s → 157s. The invariance the shortcut rests on is now a test
  rather than an assumption.

### Fixed

- **Two stale claims in the README.** It said the tool "does not design the experimental stimuli",
  which stopped being true in 1.11.0, and that hierarchical Bayes "has not yet been run on real
  MaxDiff responses", which stopped being true in 1.6.1 — 350 real respondents, Spearman 1.0000
  against the classical score. Both were the project describing an older version of itself.

## [1.11.1] — 2026-08-08

### Fixed

- **The packaged app shipped without TURF or the design generator.** Both modules are imported
  inside functions rather than at the top of a file — TURF from the report builder, the design
  generator from the command line — so PyInstaller's static analysis never saw either one and
  neither was bundled. The app still launched, still segmented, still ranked; a best-worst study
  simply came back with no "which few to launch" section, and `--design` failed on a module that
  was not there.

  This is the fourth time this exact blindness has shipped something missing — the dip test,
  tabulate, matplotlib's SVG writer, and now these two — so the build's smoke test no longer takes
  the build's word for it: it runs a best-worst export through the packaged binary and fails the
  build if the TURF section is absent. A check that costs nothing and would have caught all four.

## [1.11.0] — 2026-08-08

### Added

- **Build the best-worst questionnaire itself.** `segment-kmeans --design my_items.txt` takes one
  item per line and writes the design a survey platform can field: which items appear on which
  screen, for whom. The tool could previously read a MaxDiff study but not help you run one, which
  left the front of the workflow to Sawtooth — and the design decides what the data can possibly
  say. Two items that never appear together cannot be compared afterwards, however many people
  answer.

  Every respondent gets a different arrangement, so fatigue on screen three does not become an
  opinion about whatever is always on screen three.

  Not a textbook balanced incomplete block design: those exist only for particular combinations of
  item count, set size and screen count, and a real study fixes those from how long people will sit
  still. This searches for the most balanced design of the shape you actually want and then
  **reports what it achieved** — exposures per item, appearances per pair, and plainly when the
  shape cannot be even, which is arithmetic rather than a flaw.

  **Verified by closing the loop rather than by balance statistics**, which are only a proxy: build
  a design, simulate people answering it from known preferences, run the real estimator, and check
  the ranking that comes back is the one that went in. Spearman 1.000.

  The balance search was rewritten to score incrementally after the first version took
  twenty-four seconds for sixty respondents and grew with the sample. Swaps move items between
  screens without changing how often any item appears, so item balance is invariant and only a few
  dozen pair counts can change: 24s became 0.3s, four hundred respondents in 1.7s, with identical
  balance (worst pair spread 37-71 down to 51-59).

## [1.10.0] — 2026-08-08

### Added

- **TURF: which few items should you actually launch?** A best-worst study says what people like on
  average. That is a different question from which SET to offer, and the difference matters
  whenever tastes divide — the three best-liked items can all appeal to the same crowd while
  everyone else is left with nothing. Reach counts people rather than preference.

  Measured on a study with a 60/40 split in taste: the ranking's top three all served the majority,
  while the reaching set swapped one out for the only item the other 40% wanted.

- **And how much of that reach is luck.** TURF reports the best of every combination it tried, and
  a maximum chosen on one sample flatters whichever combination happened to suit it. Every result
  therefore carries a holdout figure: the set is chosen on half the respondents and scored on the
  other half, forty times over.

  The gap is not a rounding error, which is the assumption worth killing. Measured (percentage
  points of overstatement, choosing three items):

  | | real taste groups | pure noise |
  |---|---|---|
  | 100 people, 10 items | 9.5 | 15.6 |
  | 100 people, 20 items | 22.3 | 20.7 |
  | 1,000 people, 10 items | 2.4 | 4.9 |

  It grows with the item list — exactly when TURF is most tempting — and shrinks with sample size,
  because there is less luck left to find. At a hundred respondents over twenty items the headline
  overstates reach by more than twenty points, which is the difference between a launch decision
  and a mistake. The report quotes the holdout figure as the one to take to a budget meeting.

  The search is exhaustive wherever the binomial allows and greedy beyond, and says which it used —
  a reader is entitled to know whether "best" means best or merely good. Downloadable as
  `what_to_launch.csv`.

## [1.9.0] — 2026-08-08

### Added

- **The study planner is now in the app**, on the start screen, where the people who plan surveys
  can reach it. It was command-line only, which for a tool whose whole argument is that
  non-specialists can use it made the feature half-delivered.

  It is the one thing here that runs with no file, so it lives in the conversation before any
  result and disappears once there is one — offering to plan a study already fielded would be
  clutter. About ninety seconds, and the button says so: seventy-five complete segmentations,
  five sample sizes by three levels of distinctness by five simulated studies each.

### Fixed

- **The recommendation could be suppressed by a single unlucky simulation.** Requiring every larger
  sample to clear the bar as well looked right and was too strict: with a handful of repeats per
  cell, one bad draw anywhere above the answer made the tool announce that no sample size was
  reliable for a perfectly good design. Caught by running the app's cheaper sweep, where it
  happened immediately — the moderate column read `0/3, 2/3, 2/3, 3/3, 2/3` and the verdict was
  "nothing worked".

  The bar must now hold on average from the recommended size upward, which still refuses a size
  that only got lucky (one that passes while everything above it collapses) without being defeated
  by noise. The app's sweep also went from three repeats to five, because three cannot distinguish
  "nearly always" from "usually" and that is the distinction the whole recommendation rests on.

## [1.8.1] — 2026-08-08

An audit of the study planner shipped hours earlier. Three defects, all of which made it look more
authoritative than it was.

### Fixed

- **The separation figure was not the effect size it claimed to be.** It was used directly as a
  step between segment centres and documented as Cohen's d. Because the centre pattern rotates, an
  adjacent pair differs by one step on some questions and two on others, so the realised effect
  overshot: asking for 2.0 delivered **2.83** at three segments and **3.11** at four. The centres
  are now calibrated so the effect actually present in the answers is the one requested, checked
  across two to four segments.

- **The default sweep measured almost nothing.** It ran 100 to 800 respondents. Recovery actually
  turns over between roughly **40 and 150** — above that every regime is flat — so six of the eight
  columns reported the same number and the table appeared to show that sample size did not matter.
  Now 50 to 400, which is where the answer changes.

- **A design that cannot fit the answer scale was silently clipped.** Five segments two standard
  deviations apart do not fit on a 1-5 scale; there is nowhere to put the outer ones. The centres
  were clipped to the ends, producing a study far less separable than requested while the table
  still called it "obvious". It is now reported as not applicable, with the reason — a limit of the
  answer scale that no sample size changes.

- **The recommendation could name a size that only got lucky.** Recovery is not monotonic in sample
  size; one measured regime went 3/4, 4/4, 3/4. Taking the first size to clear the bar could
  therefore recommend a sample that a larger one then failed to match. The bar must now hold from
  that size upward. Repeats also went from four to five, because four cannot separate "nearly
  always" from "usually" and that is the distinction the recommendation turns on.

## [1.8.0] — 2026-08-08

### Added

- **A study planner: how many respondents do you need?** `segment-kmeans --plan` needs no data,
  because you have not fielded yet. It plants a known number of segments in surveys of the shape
  you describe, runs **the real pipeline** over them at a range of sample sizes, and reports how
  often the right answer came back.

  It simulates rather than using a power formula on purpose. A closed-form calculation has to
  assume the clusters are spherical, equal-sized and well separated; this tool's own measured
  weakness is precisely overlapping segments, so a formula would confidently under-quote the sample
  needed. Running the pipeline inherits the method's real behaviour instead.

  Three distinctness regimes, chosen by measurement rather than by picking round numbers — a sweep
  of effect size against sample size showed that only the middle one is a question about sample
  size at all. Segments two standard deviations apart were recovered at every size tried; segments
  0.6 apart were recovered at none, **including the largest**. That last finding is the most useful
  thing the planner says: where more respondents cannot help, it says so and points the budget at
  the questionnaire instead.

  It also discloses a limit found while building it: at 100 people with subtle differences, about
  one simulated study in ten reported the wrong number of segments while still calling the result
  high confidence. The tool judges whether a grouping *reproduces*, and on heavily overlapping data
  a merged pair reproduces perfectly well — so small-sample segment counts are flagged as
  provisional rather than the contradiction being hidden.

  Deliberately not covered: fielding cost, incidence and screen-out, and best-worst designs.

## [1.7.3] — 2026-08-08

Found while auditing the repository the way a stranger meets it, before making it public.

### Fixed

- **The tool did not run at all on a machine with no locale set.** Python takes its encoding from
  the locale, and a Linux box without `LANG` — a container, a cron job, a minimal CI image —
  reports ASCII. Writing the report then failed with `'ascii' codec can't encode character
  '\U0001f534'`: the red confidence light. The analysis had already finished; the results were
  destroyed on the way to disk.

  Every file this tool writes now names `utf-8` explicitly. Console output is separately made
  tolerant, so a terminal that genuinely cannot show a character prints a question mark instead of
  ending the run — that part fixes a neighbouring path, not this crash, and the code says so.

  Nothing on macOS reveals this, because macOS always reports UTF-8. It is the exact shape of
  defect that passes every local test and fails on somebody else's machine, so the regression test
  runs the real command with the locale stripped out and then reads the report back as UTF-8.

- **`.gitignore` would not have stopped the obvious accidents.** `config.json` — the exact filename
  the Anthropic key lives in — was not ignored, nor was `api_key.txt`, a stray `.zip` of the 78 MB
  build, or a root-level `node_modules/`. Copying a config file into the repo to reproduce
  something is an ordinary thing to do; it should not be able to commit a key.

### Changed

- The README's examples now run. They named `utilities.csv`, `demos.csv` and `my_survey.csv`, none
  of which exist here, so the first command a visitor copied would fail; they now use the example
  survey that ships in the repository, verified from a fresh clone. "Download the app" linked to
  nothing and the newest published release was seven versions old — both fixed. And the two charts
  the tool's whole argument rests on are now shown rather than described.

## [1.7.2] — 2026-08-08

### Fixed

- **A value that could not possibly be an API key was accepted, stored, and reported as
  configured.** `save_api_key` checked only that the string was non-empty. Found in the field: a
  nine-character string sat in the config file, `status()` called the app configured, and the only
  sign of trouble came after uploading a survey, waiting for it to run and clicking *Suggest names*
  — at which point the error blamed the key without saying it had never been one.

  The shape is now checked as it is typed: it must start with `sk-ant-` and be at least 20
  characters, with a message that says what a real key looks like and where to copy it from. Only
  the prefix and a length floor are checked — anything cleverer would guess at a format Anthropic
  is free to change, and wrongly refusing a real key is worse than passing a bad one to the API,
  which is the real judge either way.

## [1.7.1] — 2026-08-08

Found when the app would not open at all.

### Fixed

- **The app would not launch when unzipped onto an iCloud-synced Desktop.** macOS syncs Desktop and
  Documents by default, and iCloud writes its own metadata (`com.apple.FinderInfo`,
  `com.apple.fileprovider.fpfs#P`) onto everything inside them. On an app bundle that metadata
  **invalidates the code signature**, and macOS then refuses to launch it — silently, with nothing
  said about why. The archive itself was fine: the same .zip extracted outside iCloud verifies and
  runs.

  The build already guarded against this when *creating* the archive; nothing warned about it on
  *extraction*. The instructions now say to unzip in Applications and explain what goes wrong
  otherwise, because the symptom — an app that just does not open — gives a recipient nothing to
  go on.

- **The bundle reported its version as 0.0.0.** PyInstaller's default was never replaced, so every
  release looked identical in Finder's Get Info, and "which version are you running?" could only be
  answered by launching the app and reading a report footer. The real version is now stamped into
  `Info.plist` before signing (after, it would break the very signature the signing step exists to
  protect).

## [1.7.0] — 2026-08-08

From an audit of the Hierarchical Bayes sampler itself. Its *outputs* had been checked hard —
recovery of planted utilities, interval coverage, agreement with the classical score on real data.
The sampler's own behaviour had not.

### Changed

- **How long the sampler runs is now decided by measurement, not by a constant.** The fixed 6,000
  draws were ample for a small study and **not enough for a larger one**, and the shortfall was
  silent: an under-mixed chain still produces a tidy ranking and a tight-looking interval, with
  nothing in the output to betray it.

  Measured on 300 respondents and 12 items, four independent chains disagreed at R-hat 1.13 while
  the tool reported utilities and 95% intervals with no hint that anything was unsettled. Extending
  the chain fixes it (R-hat 1.008 at 60,000 draws), so the chain now grows — 6,000, then 20,000,
  then 60,000 — until split-R-hat says it has settled.

  Studies that do not need it pay nothing: 40 people over 6 items settles first time in 0.6s, and
  150 over 8 items in 1.6s. The real 350-person dataset settles at the first step in 4.1s. A caller
  that names its own chain length is never overridden.

### Added

- **The report says when an estimate has not settled.** If the chain is still wandering at the cap,
  the ranking carries a note giving the convergence score and saying to treat the exact numbers and
  ranges as approximate. A caveat that fired on good data would teach the reader to ignore it, so
  it appears only when it is true.

### Verified, not changed

- **The answer does not depend on the order of the file, or on which item gets pinned.** The
  sampler holds one item at zero for identification, and that item is whichever sorts last — so the
  concern is real. Shuffling every row moves the utilities by at most 0.016 on a scale spanning
  4.6; renaming the items so a different one is pinned, by at most 0.032. The ranking is identical
  in all three.
- **Agreement with the classical best-minus-worst score on the real 350-person dataset is
  unchanged at Spearman 1.0000** after the sampling change.

## [1.6.2] — 2026-08-07

### Fixed

- **A best-worst survey saved one row per person was silently analysed as a rating grid.** This is
  how Qualtrics and Sawtooth write MaxDiff: a column for every (task, item) pair holding a small
  code, commonly 3 for the item picked best, 1 for the one picked worst and 2 for the others shown.
  Nothing in such a file announces that it is a preference exercise, so the tool read the CODES as
  scores and clustered them. Measured on an 80-person export: **two confident segments, no warning
  anywhere** — the worst kind of answer this tool can give.

  It is now recognised and refused, with a worked example of the four-column layout to reshape to.

  **Why refused rather than read.** The layout can be recovered; the polarity cannot. Whether 3
  means best or 1 means best is a fact about how the survey was built, not about the data, and
  choosing wrong would turn every ranking upside down with nothing to reveal it. `choicetools`,
  the R package that does read these files, makes the analyst state it for the same reason.

  Recognition is deliberately narrow — it needs at least two question blocks in which nearly every
  respondent has exactly one lowest code, exactly one highest code, and no more than three distinct
  values. Ordinary matrix surveys with the same column naming were checked against it and are read
  normally: 5-point matrices, **3-point matrices** (where one-lowest-one-highest happens by
  chance), binary pick-any grids, 0-10 sliders, and every real dataset in the local project store.

## [1.6.1] — 2026-08-07

Found by feeding the tool a **real published best-worst dataset** for the first time: the
`bwsTools` example data from CRAN, 350 respondents asked which issues facing the country matter
most and least.

### Fixed

- **A genuine best-worst export was refused outright.** Real data names its columns for the subject
  rather than the method — the item column was `issue`, the set column `block`, the choice column
  `value` — and codes the pick as **1 / -1 / 0** instead of the words "best" and "worst". Neither
  the aliases nor the numeric coding were handled, so the file failed with an error naming no
  cause. Both are now read.

  Numeric coding is accepted only when unambiguous: values within {-1, 0, 1} with both 1 and -1
  present. A column of 1s and 2s is left alone, because 2 could mean worst, or second choice, or a
  two-point rating, and guessing would invent preferences out of an ordinary number.

- **An internal sentinel was being shown to users.** The refusal above arrived as *"Technical
  detail: `_MAXDIFF_MISSING:item`"* beside generic advice to check the file has one row per person
  — the opposite of what a best-worst export looks like. It now names the missing column, says
  which words it accepts, and states that these files have one row per item *shown*.

- **Detection now reads the choice column, not just its name.** Widening the aliases to include
  words as generic as `value` and `code` made a false positive possible, and a false positive is
  expensive: an ordinary survey would go through a preference sampler and come back as confident
  nonsense. A column of 1-5 ratings is refused; an empty one is still recognised as a broken
  best-worst file, so the reader can explain itself rather than falling back to advice about
  columns.

### Verified, not changed

- **The estimator agrees with the standard method on real human data.** Against the classical
  best-minus-worst score on the same 350 respondents: **Spearman rank correlation 1.0000**, Pearson
  0.9986. Healthcare and the economy came top, media bias last. The tool independently found three
  segments, which is the number the package's own authors report for this data.
- **The MaxDiff path scales.** 2,000 respondents x 40 items x 15 tasks — 150,000 rows — in 110
  seconds at 1.04 GB, recovering the planted order at correlation 0.983. A deliberately sparse
  design (60 items, 8 tasks each) still recovers at 0.978.

## [1.6.0] — 2026-08-07

### Added

- **A best-worst (MaxDiff) study now reports the answer it was fielded to produce.** The tool
  already detected these exports, estimated individual utilities by Hierarchical Bayes, and grouped
  people on them — and then described the groups while never saying which items the sample actually
  preferred. The ranking was computed inside the sampler and discarded on the way out. For a
  best-worst study that ranking *is* the finding; the segments are what you do about it.

  It now appears as a card above the charts, as a section at the top of the report, and as
  `item_utilities.csv` and `respondent_utilities.csv` — so a follow-up analysis never has to re-run
  the sampler, which is the expensive part.

- **The ranking says how sure it is, item by item.** Each row carries the probability that it
  really does beat the row below it, taken from the posterior draws the sampler was already
  producing. Anything under 95% is marked on the row itself.

  This replaced a yes/no flag, and the reason is worth recording: on a thin study the flag returned
  "too close to call" for a pair at 58% and for a pair at 93% — a coin flip and a finding most
  people would act on, reported in identical words. The rule behind it was also the wrong
  comparison, reading two *marginal* intervals as if they were independent when these utilities are
  centred and therefore correlated by construction.

### Fixed

- **The ranking's opening sentence no longer claims more than its own table.** Sorting items with
  no real differences still produces a first row, and the section announced "*X* comes out
  strongest" before explaining that nothing had been separated. It now reads the same evidence the
  table does.
- **A row without a credible interval no longer crashes the card.** `low !== null` is true when the
  key is simply absent, which reached `row.low.toFixed(2)`.
- **`frontend/package.json` had been a full release behind since 1.5.9.** The check that the
  version agrees everywhere covered two of the three files that carry it; it now covers all three.

### Verified, not changed

The privacy guarantee — only aggregates are sent to Claude — was an executable check for rating
grids only. A best-worst export takes a different route entirely, including respondent ids
travelling through a table index rather than a column. It holds there too: zero identifiers and
zero free-text answers out of 90 respondents, now tested rather than assumed.

## [1.5.9] — 2026-08-07

The audit release. Nothing here came from a user report — all of it came from pointing the tool at
data it had never seen and checking whether what it said was true.

### Changed

- **A short survey can no longer be cut into more groups than its answers can distinguish.** Five
  questions on a 1–5 scale asked of 400 people is a very common shape, and the tool would happily
  look for eight segments in it. It now requires roughly four distinct answer patterns per group
  before that group can be a type, and the report says in words why the search was narrowed rather
  than silently searching a shorter range. **This changes results on short surveys**, which is why
  it is called out first.

### Fixed

- **A best-worst export is now read in the words real exports use.** The MaxDiff reader recognised
  only `best`/`worst`; Qualtrics, SurveyMonkey and their Nordic equivalents write `most`/`least`,
  `bäst`/`sämst`, `beste`/`verste`, `bedste`/`værst`, `paras`/`huonoin`. Six of seven malformed
  exports in the test set now load; the seventh is refused with a reason.
- **The names you choose now reach the file that describes the groups.** Naming a segment renamed
  it in the report and in the scored export, but not in the profile download.
- **The charts are held to the same standard as the report: they must show everybody.**

### Verified, not changed

Recorded because the checking is the work, and a later session should not redo it: the gap statistic
(picks the planted k on 3- and 5-group data, stays at the floor on noise and on a single blob),
question importance reported as Cramér's V **squared** (perfect 1.0000, independent 0.0000,
invariant to renumbering), the Benjamini-Hochberg correction against its definition, the Hopkins
statistic across five geometries, the two cluster-tendency tests as a pair, the dip test's question
and respondent floors at their exact boundaries, and that the HTML report loses nothing from the
markdown — same eight tables, same sections, same figures.

## [1.5.8] — 2026-08-07

### Fixed

- **The app now tells you about "no answer" codes in a follow-up file.** 1.5.7 counted them per
  respondent and warned from the command line, and the scored CSV carried the column — but the app
  itself said nothing, so the people most likely to be scoring a follow-up were the least likely to
  hear about it. It showed *"250 people scored. Average confidence 0.58."* with no hint that sixty
  of them had a 99 in one question and had been pulled towards whichever group is extreme on it.

  The summary now says how many, why it matters, and which column in the download identifies them.
  It stays quiet when there are none.

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
