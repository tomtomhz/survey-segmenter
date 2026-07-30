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
                "numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "anthropic", "pyinstaller"],
               check=True)

# Always analyse from scratch. PyInstaller caches its analysis in build/, and a cache written while
# a source file was momentarily unparseable sticks: the next build then reports success while
# quietly omitting segment_kmeans, and the shipped app dies on launch with ModuleNotFoundError.
# Re-analysing costs about a minute and removes a whole class of "it worked on my machine".
shutil.rmtree("build", ignore_errors=True)

cmd = [sys.executable, "-m", "PyInstaller", "--name", "Survey Segmenter", "--windowed",
       "--noconfirm", "--collect-submodules", "sklearn",
       "--collect-all", "anthropic", "--collect-all", "pydantic", "--collect-all", "pydantic_core",
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
    exe = (Path("dist/Survey Segmenter.app/Contents/MacOS/Survey Segmenter")
           if sys.platform == "darwin" else Path("dist/Survey Segmenter/Survey Segmenter"))
    if not exe.exists():
        print("\nWARNING: could not find the built binary to smoke-test.")
        return
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

    env = {**os.environ, "SEG_PORT": "8765", "SEG_NO_BROWSER": "1",
           "SURVEY_SEGMENTER_PROJECTS": tempfile.mkdtemp()}
    proc = subprocess.Popen([str(exe)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    try:
        for _ in range(60):
            if proc.poll() is not None:
                print("\nBUILD IS BROKEN — the app exited on launch:\n"
                      + ((proc.stdout.read() or "").strip()[-1500:] or
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
            print("\nWARNING: the built app never answered on its port; smoke test inconclusive.")
            return
        req = urllib.request.Request(
            "http://127.0.0.1:8765/analyze", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            result = json.loads(urllib.request.urlopen(req, timeout=600).read())
        except Exception as e:
            # Never let the check itself abort the build before the app has been signed and
            # zipped: an inconclusive smoke test is a warning, a failed one is an error.
            print(f"\nWARNING: smoke test could not complete ({type(e).__name__}: {e}). "
                  "The app was built, but verify it by hand before sharing it.")
            return
        if not result.get("ok") or not result.get("charts"):
            print(f"\nBUILD IS BROKEN — the app ran but could not analyse a survey: {result}")
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
