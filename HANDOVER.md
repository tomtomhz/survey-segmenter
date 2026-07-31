# Handover — state of play

Working state for whoever picks this up next, human or AI. `README.md` says what the tool is;
this says where it stands, what was decided and why, and what to do next.

**Last updated:** 2026-07-31 · repo `github.com/tomtomhz/survey-segmenter` (private) · `main` @ 109 Python + 50 frontend tests, CI green

---

## Where it stands

| | |
|---|---|
| Repo | `github.com/tomtomhz/survey-segmenter` — **private**, MIT, owner `tomtomhz` |
| CI | Python 3.9 / 3.11 / 3.12 + a clean-install job, green |
| Tests | 109 Python (`pytest`) + 50 frontend (`cd frontend && npm test`) |
| Shipped app | macOS `.app`, ~76 MB, attached to the **v1.0.0 Release** — never in git history |
| Local path | `~/Desktop/bd-gtm-review-team 3/tools/survey-segmenter/` |

```bash
cd ~/Desktop/"bd-gtm-review-team 3"/tools/survey-segmenter
pytest                  # 108 tests, ~80s
python3 run_app.py      # opens the web app
python3 build_app.py    # rebuilds + signs + smoke-tests the .app
./finish-setup.sh       # re-runs the GitHub publish flow
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
- **Charts are hand-built SVG, not matplotlib.** Keeps the packaged app ~76 MB, keeps the
  PyInstaller build reliable, and the charts survive print/PDF and the standalone HTML report.
  Six views: segment map, per-person fit, quality across k, what differs, compare groups (radar),
  full grid (heatmap). The bar chart trims to nine questions for legibility — the full grid is
  the uncensored version, and the bars link to it.
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

## Known limitations (real, not hypothetical)

- **HB is validated on simulated data only.** No real MaxDiff responses have gone through it.
- **Windows build doesn't exist.** `build_app.py` is cross-platform but must run *on* Windows to
  produce the `.exe`. Needs a Windows machine or a CI runner.
- **The segment-size floor is a search-time guard, not a hard bound.** It filters the search fit;
  the final fit uses more restarts and can land slightly under. The report's "below 5% of the
  sample" note is the backstop.
- **Hopkins is unreliable on short surveys** and is now caveated in place rather than corrected —
  duplicate answer patterns inflate it (0.78 on pure noise with 2 Likert questions).
- **Sidebar hides below 820px.** Offered a toggle; never requested.
- **This machine's checkout is inside iCloud Drive** (`~/Desktop` is synced by default). With
  ~6,000 files in `node_modules`, every read goes through the sync layer: `npm test` took over
  half an hour instead of one second, Vitest workers timed out before starting, and macOS left
  `App 2.tsx` conflict copies behind. Nothing is wrong with the code — move the repository
  somewhere unsynced (`~/dev/`) before doing frontend work.

## Next candidates, roughly by value

1. **Move the repository out of iCloud Drive.** See the limitation above; it makes local
   frontend work impractical, and CI is currently the only place the frontend suite runs quickly.
2. **Windows build** — the only platform gap. Blocked on a Windows runner.
3. **Let Claude see the charts** — it currently reads only the text digest.
4. **Consolidate `segment-kmeans-tool.md`** in the assistant memory directory; it has grown to
   22 KB of appended paragraphs and is due a rewrite rather than another append.

## Working conventions

- "Carte blanche" means execute through to completion; only stop if a finding invalidates an
  earlier decision.
- Prefer main-session work over subagent dispatch for pure synthesis (subagents have timed out).
- **Bound every background wait.** An unbounded `until curl ...; do sleep 3; done` left two shells
  parked for hours in a previous session. Use `curl --retry N` or a max-attempts counter.
