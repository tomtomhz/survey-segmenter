# webui/ — generated, do not edit

This directory is the **compiled** interface. The source is in [`../frontend`](../frontend)
(React + TypeScript), and everything here is produced by:

```bash
cd frontend && npm run build
```

It is committed to git on purpose, which is unusual for build output and worth justifying:

- `python3 run_app.py` works from a fresh clone with no Node installed.
- PyInstaller bundles this directory into the packaged `.app`, which therefore has no build step.
- Non-technical users are the audience; "install Node first" is not an acceptable first screen.

CI rebuilds the interface and fails if the result differs from what is committed, so this cannot
silently fall behind `frontend/`. Editing a file here directly will be overwritten by the next
build and flagged by that check.

(This file itself lives in `frontend/public/` and is copied here by the build — `npm run build`
empties this directory first, so anything that is meant to survive has to come from the source
side.)
