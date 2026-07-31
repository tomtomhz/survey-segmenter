# Handover — state of play

Working state for whoever picks this up next, human or AI. `README.md` says what the tool is;
this says where it stands, what was decided and why, and what to do next.

**Last updated:** 2026-07-31 · repo `github.com/tomtomhz/survey-segmenter` (private) · `main` @ 100 tests, CI green

---

## Where it stands

| | |
|---|---|
| Repo | `github.com/tomtomhz/survey-segmenter` — **private**, MIT, owner `tomtomhz` |
| CI | Python 3.9 / 3.11 / 3.12 + a clean-install job, green |
| Tests | 100, `pytest` from `tools/survey-segmenter/` |
| Shipped app | macOS `.app`, ~76 MB, attached to the **v1.0.0 Release** — never in git history |
| Local path | `~/Desktop/bd-gtm-review-team 3/tools/survey-segmenter/` |

```bash
cd ~/Desktop/"bd-gtm-review-team 3"/tools/survey-segmenter
pytest                  # 100 tests, ~75s
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

- **Charts are hand-built SVG, not matplotlib.** Keeps the packaged app ~76 MB, keeps the
  PyInstaller build reliable, and the charts survive print/PDF and the standalone HTML report.
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

## Known limitations (real, not hypothetical)

- **Windows build doesn't exist.** `build_app.py` is cross-platform but must run *on* Windows to
  produce the `.exe`. Needs a Windows machine or a CI runner.
- **The segment-size floor is a search-time guard, not a hard bound.** It filters the search fit;
  the final fit uses more restarts and can land slightly under. The report's "below 5% of the
  sample" note is the backstop.
- **Hopkins is unreliable on short surveys** and is now caveated in place rather than corrected —
  duplicate answer patterns inflate it (0.78 on pure noise with 2 Likert questions).
- **Sidebar hides below 820px.** Offered a toggle; never requested.

## Next candidates, roughly by value

1. **Windows build** — the only platform gap. Blocked on a Windows runner.
2. **Let Claude see the charts** — it currently reads only the text digest.
3. **Consolidate `segment-kmeans-tool.md`** in the assistant memory directory; it has grown to
   22 KB of appended paragraphs and is due a rewrite rather than another append.

## Working conventions

- "Carte blanche" means execute through to completion; only stop if a finding invalidates an
  earlier decision.
- Prefer main-session work over subagent dispatch for pure synthesis (subagents have timed out).
- **Bound every background wait.** An unbounded `until curl ...; do sleep 3; done` left two shells
  parked for hours in a previous session. Use `curl --retry N` or a max-attempts counter.
