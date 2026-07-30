# Changelog

Notable changes to Survey Segmenter. Versions follow [semantic versioning](https://semver.org/);
the version is set in `pyproject.toml` and stamped into every report footer.

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
