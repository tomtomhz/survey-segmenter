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

# Install what the app needs FROM THE PROJECT'S OWN DECLARATION, not from a list repeated here.
#
# This used to name the dependencies again — numpy, pandas, scipy and the rest — and a second list
# maintained by memory goes stale the same way the first one did. It had: `diptest` was added to
# pyproject.toml and not here, so every build made on a machine that did not already happen to have
# it produced an app whose second cluster-tendency test silently never ran. Both CI runners were in
# exactly that state, which is how the check below caught it.
#
# `.[excel,ai]` pulls the extras that are bundled deliberately: spreadsheet reading, and the
# optional "ask Claude about your segments" chat, which ships present so the app has it out of the
# box (the user still supplies their own key in Settings). PyInstaller is a build tool rather than
# a dependency of the app, so it stays separate.
subprocess.run([sys.executable, "-m", "pip", "install", "--user", ".[excel,ai]", "pyinstaller"],
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
       # The dip test is a compiled extension, so PyInstaller has to be told to carry its binary
       # rather than only the Python that wraps it. Without this the packaged app runs and simply
       # reports the second cluster-tendency check as unavailable — a silent downgrade, which is
       # the kind this project has shipped before.
       "--collect-all", "diptest",
       # pandas imports tabulate lazily, from inside to_markdown, so static analysis never sees it
       # and every packaged release shipped without it — which means every report shipped with no
       # tables at all, the segment sizes and stability checks arriving as run-together text. The
       # same shape as matplotlib.backends.backend_svg above: a lazy import is invisible until
       # something runs the code path, and the smoke test now does.
       "--collect-all", "tabulate",
       # The interface itself. Without this the packaged app starts, serves its API, and shows
       # nothing at all — the failure looks like a broken server rather than a missing folder.
       "--add-data", f"webui{os.pathsep}webui",
       "run_app.py"]
if sys.platform == "darwin":
    # Reverse-DNS under the GitHub org that hosts the project, which is the convention for a
    # project without its own domain. Changing this changes the app's IDENTITY to macOS: a build
    # with a new identifier is a different application, so the first launch after this change asks
    # for the usual 'Open Anyway' approval again even for someone who already granted it.
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
    import math
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
    # Five questions and 120 people, not four and 80: the second cluster-tendency check needs at
    # least four questions and forty respondents before it will run, and a smoke test that skips
    # the capability it is meant to be checking proves nothing about it.
    rnd = random.Random(0)
    rows = [["respondent_id", "q1", "q2", "q3", "q4", "q5"]]
    for i in range(120):
        base = [5, 1, 5, 2, 4] if i % 2 else [1, 5, 2, 4, 1]
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
        # Beyond "it ran": the two capabilities that fail SILENTLY when a dependency does not
        # survive bundling. Neither crashes the app — it starts, analyses, and quietly does less —
        # which is precisely the shape of failure this project has shipped before, and the shape
        # hardest to notice on a platform nobody is sitting in front of.
        missing = []
        report = result.get("report_html", "")
        if "Dip test (second opinion)" not in report:
            # The dip test is a compiled extension. Absent, the app reports the check as not run.
            why = "not run"
            marker = report.find("Second opinion on cluster tendency")
            if marker >= 0:
                why = report[marker:marker + 160].split("<")[0]
            missing.append(f"the second cluster-tendency test did not run ({why})")
        # The report's own wording. It is generated by the bundled code, so if the binary is stale
        # or a module did not survive bundling, these are absent while everything still "works" —
        # and a check for them in the source directory proves nothing about the artefact, which is
        # the mistake that prompted this: grepping the .app for report strings finds none of them,
        # old or new, because the code is compiled into a compressed archive.
        # Anchor on wording that is ALWAYS present. "stays together with fewer groups" is not: the
        # smoke survey has two mind-sets by construction, and at k=2 there is no k-1 to merge into,
        # so that column is legitimately absent. Asserting it failed a sound build — caught here
        # rather than by someone wondering why a good release would not ship.
        for phrase, what in [
                ("ask for a different number of groups",
                 "the segment-persistence section is the old single-number form, which condemned "
                 "real segments for being subdivided"),
                ("suggested name",
                 "the stability tables carry no segment names, so a reader cannot match them to "
                 "the groups")]:
            if phrase not in report:
                missing.append(f"{what} (expected {phrase!r} in the report)")
        # Tables. Every one goes through DataFrame.to_markdown, which needs `tabulate`; without it
        # the report has NO tables — segment sizes, the stability checks, the centroids and the
        # whole k-selection panel arrive as run-together text. It was declared an optional extra
        # that neither CI nor this script installed, so every packaged release shipped that way,
        # and nothing noticed because the source tree has tabulate and the tests run there.
        if result.get("report_html", "").count("<table") < 4:
            missing.append("the report has no tables in it (tabulate did not survive bundling) — "
                           "the segment sizes and stability checks will be run-together text")
        specs = sum(1 for c in result["charts"] if c.get("spec"))
        if specs < len(result["charts"]):
            missing.append(f"only {specs} of {len(result['charts'])} charts carry the data behind "
                           "them, so the interactive versions will fall back to pictures")
        # A second survey, of a different KIND. `maxdiff.py` is a separate top-level module and the
        # engine imports it inside a try/except, because scoring best-worst data is optional. If it
        # does not survive bundling, `_maxdiff` is None, the detector never fires, and a best-worst
        # export is clustered on its raw choice codes — a confident-looking segmentation of
        # nonsense, with no error anywhere. That is the same silent shape as the missing tables,
        # and the source tree cannot reveal it because the module is always importable there.
        # Two seconds of sampling on forty people is the whole cost.
        bw = [["respondent_id", "task", "item", "choice"]]
        items = ["a", "b", "c", "d", "e"]
        strength = [1.5, 0.8, 0.0, -0.8, -1.5]
        rnd2 = random.Random(11)
        for person in range(40):
            for task in range(5):
                shown = rnd2.sample(range(len(items)), 4)
                draw = [strength[i] - math.log(-math.log(max(rnd2.random(), 1e-12)))
                        for i in shown]
                best = shown[draw.index(max(draw))]
                worst = shown[draw.index(min(draw))]
                for i in shown:
                    bw.append([f"R{person}", task, items[i],
                               "Best" if i == best else ("Worst" if i == worst else "")])
        buf2 = io.StringIO()
        csv.writer(buf2).writerows(bw)
        body2, boundary2 = _multipart(buf2.getvalue(), filename="best_worst.csv")
        req2 = urllib.request.Request(
            "http://127.0.0.1:8765/analyze", data=body2,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary2}"})
        try:
            bw_result = json.loads(urllib.request.urlopen(req2, timeout=600).read())
        except Exception as e:
            print(f"\nBUILD IS BROKEN — the app could not analyse a best-worst export "
                  f"({type(e).__name__}: {e}).")
            _discard_archive()
        if not bw_result.get("ok"):
            print(f"\nBUILD IS BROKEN — a best-worst export was refused: "
                  f"{bw_result.get('error') or bw_result}")
            _discard_archive()
        if "Hierarchical Bayes" not in bw_result.get("report_html", ""):
            print("\nBUILD IS BROKEN — a best-worst export was NOT scored as one. maxdiff.py did "
                  "not survive bundling, so the choice codes were clustered as if they were "
                  "ratings. The app reports groups and gives no indication anything is wrong.")
            _discard_archive()
        if not bw_result.get("ranking"):
            print("\nBUILD IS BROKEN — a best-worst export was scored but its ranking is missing, "
                  "so the app shows segments without the answer the study was fielded for.")
            _discard_archive()

        if missing:
            print("\nBUILD IS BROKEN — it runs, but quietly does less than it should:")
            for reason in missing:
                print(f"  - {reason}")
            _discard_archive()

        print(f"Smoke test passed: the built app found {result['k']} groups in "
              f"{result['n_people']} people, drew {len(result['charts'])} charts with the data "
              f"behind all of them, ran both cluster-tendency tests, and ranked "
              f"{len(bw_result['ranking'])} items from a best-worst export.")
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


def _stamp_macos_version():
    """Write the real version into the bundle, before it is signed.

    PyInstaller leaves CFBundleShortVersionString at 0.0.0, so every release looked identical in
    Finder's Get Info and in any inventory tool. Someone holding a copy had no way to tell which
    build it was without launching it and reading a report footer — and "which version are you
    running?" is the first question of every support conversation.

    This has to run BEFORE signing, because the signature covers Info.plist: stamping afterwards
    would break exactly the signature `_sign_and_zip_macos` exists to protect.
    """
    plist = Path("dist/Survey Segmenter.app/Contents/Info.plist")
    if not plist.is_file():
        return
    version = None
    for line in Path("segment_kmeans.py").read_text().splitlines():
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break
    if not version:
        return
    for key in ("CFBundleShortVersionString", "CFBundleVersion"):
        subprocess.run(["/usr/libexec/PlistBuddy", "-c", f"Set :{key} {version}", str(plist)],
                       capture_output=True)
    print(f"Stamped the bundle as version {version}.")


if sys.platform == "darwin":
    _stamp_macos_version()
    _sign_and_zip_macos()

_smoke_test()

if sys.platform == "darwin":
    print("\nBuilt. Share 'dist/Survey Segmenter.zip'. On first launch each recipient must approve "
          "it once:\n  System Settings > Privacy & Security > 'Open Anyway'  (see docs/USING-THE-APP.md).")
else:
    print("\nBuilt. The app is in the dist/ folder. Zip it and share it with the team.")
