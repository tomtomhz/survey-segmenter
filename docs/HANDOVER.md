# Handover — state of play

Working state for whoever picks this up next, human or AI. `README.md` says what the tool is;
this says where it stands, what was decided and why, and what to do next.

**Last updated:** 2026-07-31 · repo `github.com/tomtomhz/survey-segmenter` (private) · `main` @ 112 Python + 59 frontend tests, CI green

---

## Where it stands

| | |
|---|---|
| Repo | `github.com/tomtomhz/survey-segmenter` — **private**, MIT, owner `tomtomhz` |
| CI | Python 3.9 / 3.11 / 3.12 + a clean-install job, green |
| Tests | 109 Python (`pytest`) + 59 frontend (`cd frontend && npm test`) |
| Shipped app | **v1.1.0 release**: macOS `.app` (72 MB) and Windows `.exe` (99 MB), both built and smoke-tested by the **Desktop app** workflow. Never in git history. |
| Local path | `~/dev/survey-segmenter` — **moved out of iCloud Drive**, see below |

```bash
cd ~/dev/survey-segmenter
pytest                  # 112 tests, ~95s
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
- **`deliverables/` (in the parent workspace) is deliberately NOT on GitHub.** It contains ~100
  named individuals' work email addresses — personal data under GDPR. Publishing needs a human
  decision, not a tidy-up.

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

3. **Let Claude see the charts** — it currently reads only the text digest.
4. **Consolidate `segment-kmeans-tool.md`** in the assistant memory directory; it has grown to
   22 KB of appended paragraphs and is due a rewrite rather than another append.

## How the modules divide up

`segment_kmeans.py` decides what a segmentation is and what it means. `charts.py` draws it.
`webapp.py` delivers it. `maxdiff.py` scores best-worst data before any of that happens.

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
