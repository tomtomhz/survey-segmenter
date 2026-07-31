"""Build the standalone Survey Segmenter desktop app.

Run from this folder:

    python3 build_app.py

Output:
    macOS   -> dist/Survey Segmenter.app  AND  dist/Survey Segmenter.zip  (share the .zip)
    Windows -> dist/Survey Segmenter/     (a folder with Survey Segmenter.exe; zip and share)

The result is fully self-contained: recipients do NOT need Python or any setup. Build on the same
kind of machine you want to distribute to (build on a Mac for the Mac app, on Windows for the .exe).
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Make sure the tool's own dependencies and PyInstaller are available in this environment.
# `anthropic` (+pydantic) powers the optional "ask Claude about your segments" chat; it is bundled
# so the app has it out of the box (the user still supplies their own API key in Settings).
subprocess.run([sys.executable, "-m", "pip", "install", "--user",
                "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "openpyxl",
                "anthropic", "pyinstaller"],
               check=True)

# Always analyse from scratch. PyInstaller caches its analysis in build/, and a cache written while
# a source file was momentarily unparseable sticks: the next build then reports success while
# quietly omitting segment_kmeans, and the shipped app dies on launch with ModuleNotFoundError.
# Re-analysing costs about a minute and removes a whole class of "it worked on my machine".
shutil.rmtree("build", ignore_errors=True)


def _build_frontend():
    """Compile the React interface into webui/ before packaging it.

    webui/ is committed, so a build can proceed without Node — but shipping whatever happened to
    be committed while frontend/ has moved on is exactly how a released app ends up one version
    behind its own source. If npm is here, rebuild; if it is not, say which it is rather than
    letting the difference pass silently.
    """
    npm = shutil.which("npm")
    if not npm:
        if not Path("webui/index.html").exists():
            sys.exit("No npm and no built interface in webui/. Install Node, then re-run:\n"
                     "    cd frontend && npm install && npm run build")
        print("NOTE: npm not found — packaging the committed build in webui/ as-is.\n"
              "      If frontend/ has changed since it was built, the app will not show it.")
        return
    print("Building the interface (npm run build)…")
    if not Path("frontend/node_modules").is_dir():
        subprocess.run([npm, "ci"], cwd="frontend", check=True)
    subprocess.run([npm, "run", "build"], cwd="frontend", check=True)


_build_frontend()

cmd = [sys.executable, "-m", "PyInstaller", "--name", "Survey Segmenter", "--windowed",
       "--noconfirm", "--collect-submodules", "sklearn",
       "--collect-all", "anthropic", "--collect-all", "pydantic", "--collect-all", "pydantic_core",
       # matplotlib draws every chart. Its data directory carries the font it measures and
       # renders with, so without this the packaged app raises on the first chart it tries.
       "--collect-data", "matplotlib",
       # Belt and braces alongside the explicit import in charts.py: matplotlib resolves output
       # writers lazily, so a bundler sees only the backend selected at import time.
       "--collect-submodules", "matplotlib.backends",
       # The interface itself. Without this the packaged app starts, serves its API, and shows
       # nothing at all — the failure looks like a broken server rather than a missing folder.
       "--add-data", f"webui{os.pathsep}webui",
       "run_app.py"]
if sys.platform == "darwin":
    cmd += ["--osx-bundle-identifier", "io.github.tomtomhz.surveysegmenter"]

subprocess.run(cmd, check=True)


def _smoke_test():
    """Actually start the thing we just built and put a survey through it.

    PyInstaller exits 0 for builds that cannot even import their own entry point, so "build
    complete" is not evidence of anything. This is the cheapest check that catches it: launch the
    binary, analyse a tiny survey, confirm groups and charts come back."""
    import csv
    import io
    import json
    import time
    import urllib.request

    # Test the SIGNED bundle, which is why this runs after _sign_and_zip_macos(). Straight out of
    # PyInstaller the macOS bundle carries a half-written signature, and the OS refuses to run it —
    # silently, with exit status 0 and no output — so testing it before signing measures nothing
    # about the build and fails every time.
    if sys.platform == "darwin":
        exe = Path("dist/Survey Segmenter.app/Contents/MacOS/Survey Segmenter")
    else:
        # PyInstaller appends .exe on Windows. Without the suffix the path never exists, the
        # smoke test prints a warning and returns, and a broken build ships looking tested.
        exe = Path("dist/Survey Segmenter") / (
            "Survey Segmenter.exe" if os.name == "nt" else "Survey Segmenter")
    if not exe.exists():
        sys.exit(f"BUILD IS BROKEN — no binary at {exe}. PyInstaller reported success, so this is "
                 "a packaging problem rather than a compile error.")
    # Two obvious mind-sets, jittered. Perfectly identical answers within a group leave zero
    # variance, which makes the validation maths degenerate and the run crawl — the check would
    # then fail on its own test data rather than on anything wrong with the build.
    rnd = random.Random(0)
    rows = [["respondent_id", "q1", "q2", "q3", "q4"]]
    for i in range(80):
        base = [5, 1, 5, 2] if i % 2 else [1, 5, 2, 4]
        rows.append([f"P{i}", *[max(1, min(5, b + rnd.randint(-1, 1))) for b in base]])
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    body, boundary = _multipart(buf.getvalue())

    # SEG_LOG gives a windowed build somewhere to print. On Windows there is no console at all,
    # so without it the app's own account of what went wrong is discarded — which is exactly
    # where a packaging failure is most likely and hardest to guess at.
    log_path = str(Path(tempfile.mkdtemp()) / "app.log")
    env = {**os.environ, "SEG_PORT": "8765", "SEG_NO_BROWSER": "1", "SEG_LOG": log_path,
           "SURVEY_SEGMENTER_PROJECTS": tempfile.mkdtemp()}
    # Output goes to a FILE, not a pipe. A pipe nobody reads has a fixed buffer — a few tens of
    # kilobytes on Windows — and the moment it fills, the app blocks forever on its next print.
    # That is a deadlock in the test harness masquerading as a hung application, and it is what
    # the Windows smoke test was actually dying of: enough chatter (matplotlib's font-cache
    # notice, the per-chart trace) to fill the buffer part-way through drawing.
    log_file = open(log_path, "w+", buffering=1)
    proc = subprocess.Popen([str(exe)], env=env, stdout=log_file, stderr=subprocess.STDOUT,
                            text=True)

    def app_output():
        try:
            return Path(log_path).read_text(errors="replace")
        except Exception:
            return ""
    try:
        for _ in range(60):
            if proc.poll() is not None:
                print("\nBUILD IS BROKEN — the app exited on launch:\n"
                      + (app_output().strip()[-1500:] or
                         "(it produced no output — a windowed macOS build discards stdout, so "
                         "check the signature with: codesign --verify --deep --strict "
                         "'dist/Survey Segmenter.app')"))
                _discard_archive()
            try:
                urllib.request.urlopen("http://127.0.0.1:8765/", timeout=1).read()
                break
            except Exception:
                time.sleep(1)
        else:
            proc.terminate()
            sys.exit("BUILD IS BROKEN — the app started but never answered on its port. It is "
                     "running without serving, which is worse than crashing: a user would see a "
                     "window that does nothing.")
        req = urllib.request.Request(
            "http://127.0.0.1:8765/analyze", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            result = json.loads(urllib.request.urlopen(req, timeout=900).read())
        except Exception as e:
            # An inconclusive smoke test used to be a warning. That let a Windows build go green
            # while its own verification had timed out, and CI then uploaded the unverified
            # artefact — which is the whole failure mode this check exists to prevent. If the app
            # cannot analyse a survey within ten minutes, it is broken from a user's point of view.
            print(f"\nBUILD IS BROKEN — the smoke test could not complete "
                  f"({type(e).__name__}: {e}). The app was built but never proved it works.")
            try:
                tail = app_output().strip().splitlines()[-25:]
                if tail:
                    print("What the app itself reported:")
                    for line in tail:
                        print(f"  {line}")
            except Exception:
                print("  (the app produced no log at all)")
            _discard_archive()
        if not result.get("ok"):
            print(f"\nBUILD IS BROKEN — the app ran but could not analyse a survey: "
                  f"{result.get('error') or result}")
            _discard_archive()
        if not result.get("charts"):
            # Separated from the failure above because it means something quite different: the
            # statistics worked and only the drawing did not, which in a packaged build almost
            # always means a charting dependency did not survive bundling.
            why = result.get("chart_errors") or ["no reason reported"]
            print("\nBUILD IS BROKEN — the analysis worked but no charts were drawn:")
            for reason in why:
                print(f"  - {reason}")
            _discard_archive()
        print(f"Smoke test passed: the built app found {result['k']} groups in "
              f"{result['n_people']} people and drew {len(result['charts'])} charts.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _discard_archive():
    """A broken build must not leave something that looks shippable sitting in dist/."""
    Path("dist/Survey Segmenter.zip").unlink(missing_ok=True)
    print("Removed dist/Survey Segmenter.zip — do not share this build.")
    sys.exit(1)


def _multipart(text, field="file", filename="smoke.csv"):
    boundary = "----surveysegmenterbuildcheck"
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: text/csv\r\n\r\n{text}\r\n--{boundary}--\r\n")
    return body.encode(), boundary


def _sign_and_zip_macos():
    """Produce a shareable .zip whose code signature actually survives the download.

    Two macOS facts make this necessary, and both bite silently:

    1. Apple Silicon refuses to run a binary with a BROKEN signature. PyInstaller's own signing
       step fails whenever the build folder carries extended attributes (`com.apple.FinderInfo`
       and friends) — which it always does inside an iCloud-synced folder such as Desktop. The
       result looks fine locally but a downloaded copy dies with "the app is damaged", which a
       non-technical recipient cannot recover from.
    2. A plain `zip` of a .app can mangle the bundle; `ditto` is Apple's supported way to archive
       one (and compresses far better besides).

    So: copy the bundle somewhere clean and attribute-free, sign THAT, and archive it with ditto.
    """
    app = Path("dist/Survey Segmenter.app")
    if not app.is_dir():
        return
    out_zip = Path("dist/Survey Segmenter.zip")
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / app.name
        # --noextattr/--norsrc/--noacl strip exactly the detritus that breaks codesign
        subprocess.run(["ditto", "--norsrc", "--noextattr", "--noacl", str(app), str(clean)],
                       check=True)
        subprocess.run(["xattr", "-cr", str(clean)], check=False)
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(clean)], check=True)
        out_zip.unlink(missing_ok=True)
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent",
                        str(clean), str(out_zip)], check=True)
        shutil.rmtree(app, ignore_errors=True)      # replace the build with the signed copy
        subprocess.run(["ditto", str(clean), str(app)], check=True)
        # Verify what actually ships — unpack the zip and check THAT. Checking the folder we just
        # signed would be misleading: in an iCloud-synced folder (Desktop) the sync service re-adds
        # extended attributes within seconds and breaks the on-disk copy's signature, while the
        # archived one stays intact. The zip is the artefact recipients download.
        check = Path(tmp) / "verify"
        subprocess.run(["ditto", "-x", "-k", str(out_zip), str(check)], check=True)
        verify = subprocess.run(["codesign", "--verify", "--deep", "--strict",
                                 str(check / app.name)], capture_output=True, text=True)
        if verify.returncode != 0:
            print("\nWARNING: the archive's signature did not verify — recipients may see "
                  "'app is damaged':\n" + (verify.stderr or "").strip())
        else:
            print("\nCode signature verified inside the archive — it survives being downloaded.")
    print(f"Shareable archive: {out_zip}  ({out_zip.stat().st_size / 1e6:.0f} MB)")


if sys.platform == "darwin":
    _sign_and_zip_macos()

_smoke_test()

if sys.platform == "darwin":
    print("\nBuilt. Share 'dist/Survey Segmenter.zip'. On first launch each recipient must approve "
          "it once:\n  System Settings > Privacy & Security > 'Open Anyway'  (see APP_README.md).")
else:
    print("\nBuilt. The app is in the dist/ folder. Zip it and share it with the team.")
