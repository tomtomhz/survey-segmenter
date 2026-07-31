# Changelog

Notable changes to Survey Segmenter. Versions follow [semantic versioning](https://semver.org/);
the version is set in `pyproject.toml` and stamped into every report footer.

## [Unreleased]

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
  weak structure — see `HANDOVER.md`, which reports the full sweep including where HB does not
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

- All analysis is local. See [PRIVACY.md](PRIVACY.md).
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
