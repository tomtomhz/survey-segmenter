#!/usr/bin/env bash
# One-time GitHub setup for Survey Segmenter.
#
#   ./finish-setup.sh <your-github-username>
#
# Creates a PRIVATE repo, pushes main, fixes the README badge URLs, and uploads the packaged
# macOS app as a v1.0.0 release asset (the app is deliberately not in git history — 190 MB of
# rebuildable binary does not belong in every clone).
set -euo pipefail
OWNER="${1:?usage: ./finish-setup.sh <your-github-username>}"
REPO="survey-segmenter"

command -v gh >/dev/null || { echo "GitHub CLI not found. Install it, then re-run."; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Not logged in. Run:  gh auth login"; exit 1; }

# Point the badges and packaging metadata at the real repository before the first push.
sed -i '' "s|github.com/OWNER/survey-segmenter|github.com/$OWNER/$REPO|g" README.md
python3 - "$OWNER" "$REPO" <<'PY'
import sys, pathlib
owner, repo = sys.argv[1], sys.argv[2]
p = pathlib.Path("pyproject.toml"); s = p.read_text()
if "[project.urls]" not in s:
    s = s.replace("[project.scripts]",
                  f'[project.urls]\nHomepage = "https://github.com/{owner}/{repo}"\n'
                  f'Repository = "https://github.com/{owner}/{repo}"\n'
                  f'Changelog = "https://github.com/{owner}/{repo}/blob/main/CHANGELOG.md"\n\n'
                  "[project.scripts]")
    p.write_text(s)
PY
git add README.md pyproject.toml
git commit -q -m "Point badges and packaging metadata at the repository" || true

gh repo create "$REPO" --private --source=. --remote=origin --push \
  --description "Turn a survey export into customer segments you can defend. Local-first, GDPR-conscious."

if [ -f "dist/Survey Segmenter.zip" ]; then
  gh release create v1.0.0 "dist/Survey Segmenter.zip#Survey Segmenter (macOS)" \
    --title "v1.0.0" --notes-file CHANGELOG.md
fi
echo
echo "Done: https://github.com/$OWNER/$REPO"
