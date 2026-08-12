"""The local web application: the page, the API behind it, and saved projects.

This is the delivery layer, kept apart from the statistics on purpose. `segment_kmeans` decides
what a segmentation is and what it means; nothing in this file makes an analytical judgement. It
parses uploads, serves the compiled interface out of `webui/`, keeps projects on disk, and hands
questions to the optional Claude layer.

Everything is local. The server binds to 127.0.0.1, the survey file never leaves the machine, and
the only thing that can be transmitted anywhere is the aggregate digest and the rendered charts —
and only when the user has configured their own Anthropic key. See docs/PRIVACY.md.
"""
import io
import json
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import segment_kmeans as sk

def _parse_multipart_file(content_type, body, with_name=False):
    """Extract the first uploaded file's bytes from a multipart/form-data POST body, using only the
    standard library (Python's `cgi` module is deprecated and removed in 3.13, so we do not rely on
    it). Returns the raw bytes, or None if no file part is present."""
    if "boundary=" not in content_type:
        return (None, None) if with_name else None
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    for part in body.split(b"--" + boundary.encode()):
        header_blob, sep, payload = part.partition(b"\r\n\r\n")
        if sep and b"filename=" in header_blob:
            data = payload.rsplit(b"\r\n", 1)[0]   # drop the trailing CRLF before the next boundary
            if not with_name:
                return data
            m = re.search(r'filename="([^"]*)"', header_blob.decode("utf-8", "replace"))
            return data, (m.group(1) if m else None)
    return (None, None) if with_name else None


_WEBUI_DIRNAME = "webui"

# What the built interface is allowed to be made of. Anything else in the directory is not served
# — a stray file sitting next to the bundle should not become a route.
_WEBUI_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                ".map": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon",
                ".png": "image/png", ".woff2": "font/woff2", ".json": "application/json"}


def _webui_dir():
    """Where the built interface lives.

    The interface's source is `frontend/` (React + TypeScript); `npm run build` compiles it into
    `webui/`, which is what actually ships. That directory is committed so a clone runs without
    Node installed, and PyInstaller bundles it into the packaged app — where it lands beside the
    unpacked modules in sys._MEIPASS rather than next to this file.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(getattr(sys, "_MEIPASS", here), _WEBUI_DIRNAME),
                 os.path.join(here, _WEBUI_DIRNAME)):
        if os.path.isdir(path):
            return path
    return None


def _webui_asset(url_path):
    """Resolve a URL to a file inside the built interface, returning (bytes, content-type).

    Everything is resolved through realpath and checked to be inside the bundle. The server only
    listens on localhost, but "only listens locally" has never been a good reason to let a path
    escape its directory.
    """
    root = _webui_dir()
    if not root:
        return None
    from urllib.parse import unquote, urlparse
    rel = unquote(urlparse(url_path).path).lstrip("/")
    if not rel or rel.endswith("/"):
        rel = "index.html"
    if os.path.splitext(rel)[1].lower() not in _WEBUI_TYPES:
        return None
    real_root = os.path.realpath(root)
    target = os.path.realpath(os.path.join(real_root, rel))
    if os.path.commonpath([real_root, target]) != real_root or not os.path.isfile(target):
        return None
    with open(target, "rb") as fh:
        return fh.read(), _WEBUI_TYPES[os.path.splitext(rel)[1].lower()] + "; charset=utf-8"


def _missing_ui_page():
    """Shown when the interface has not been built.

    Only reachable from a source checkout where `webui/` was deleted — the packaged app always
    carries it. Names the one command that fixes it rather than showing a blank page.
    """
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>Survey Segmenter</title><style>body{margin:0;min-height:100vh;display:flex;"
            "align-items:center;justify-content:center;background:#F4F2E8;color:#1B2420;"
            "font:16px/1.6 ui-sans-serif,-apple-system,system-ui,sans-serif;padding:24px}"
            "div{max-width:34rem}code{background:#EFEDE1;padding:.15em .4em;border-radius:5px;"
            "font-size:.9em}h1{font-size:1.3rem;margin:0 0 .6em}"
            "@media(prefers-color-scheme:dark){body{background:#161A17;color:#E9EBE4}"
            "code{background:#2F3531}}</style></head><body><div>"
            "<h1>The interface has not been built yet.</h1>"
            "<p>The statistics engine is fine \u2014 only the web interface is missing. Build it "
            "once with:</p><p><code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build"
            "</code></p><p>Then reload this page. If you do not have Node installed, "
            "<code>python3 segment_kmeans.py your_survey.csv</code> runs the whole analysis from "
            "the command line and needs nothing extra.</p></div></body></html>")


def _charts_for_browser(charts):
    """The charts, minus the raster copies.

    Each chart carries both an SVG (what the page draws) and a PNG (what Claude is shown). The
    PNG is roughly 40 kB of base64 apiece, so sending all six to a browser that renders only the
    vector version would add a quarter of a megabyte to every analysis response and every project
    reopen, for bytes nothing on the page reads. They stay in the session, where the Claude layer
    picks them up.
    """
    return [{k: v for k, v in chart.items() if k != "png_b64"} for chart in (charts or [])]


class ProjectStore:
    """Saved projects — one per survey you analyse, like a chat history.

    Kept as plain files under ~/.survey_segmenter/projects so a project survives closing the app,
    and so a half-finished analysis is never lost to a stray refresh. Everything stays on this
    computer — including the original upload, which is kept alongside the results so the user can
    re-pick which questions to group on without uploading the file again. Nothing here is ever sent
    anywhere; the AI layer only ever sees the aggregate digest."""

    def __init__(self, root=None):
        # SURVEY_SEGMENTER_PROJECTS lets a test (or a locked-down machine) point the store
        # somewhere else without touching the user's home directory.
        self.root = Path(root or os.environ.get("SURVEY_SEGMENTER_PROJECTS")
                         or (Path.home() / ".survey_segmenter" / "projects"))
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.root = Path(tempfile.gettempdir()) / "survey_segmenter_projects"
            self.root.mkdir(parents=True, exist_ok=True)

    # A project is three files, deliberately: the big record, a tiny summary the sidebar can read
    # without parsing megabytes, and the original upload so questions can be re-picked later.
    MAX_RAW = 8 * 1024 * 1024

    def _stem(self, pid):
        return re.sub(r"[^A-Za-z0-9_-]", "", str(pid))[:64]       # never leave the store directory

    def _path(self, pid, suffix=".json"):
        stem = self._stem(pid)
        return (self.root / f"{stem}{suffix}") if stem else None

    @staticmethod
    def _write_atomic(path, text_or_bytes):
        """Write, then rename into place, so a reader never sees a half-written file.

        The scratch name carries a random suffix because the server is threaded and one project
        is saved repeatedly — after the analysis, after every chat reply, after the groups are
        named. With a fixed `.tmp` name, two overlapping saves of the same project used the same
        scratch file: the first rename moved it away and the second raised FileNotFoundError out
        of the request handler, so the user saw an error and the save was lost. Reproduced with
        60 concurrent saves; it failed on the first few.
        """
        tmp = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        try:
            if isinstance(text_or_bytes, bytes):
                tmp.write_bytes(text_or_bytes)
            else:
                tmp.write_text(text_or_bytes, encoding="utf-8")
            tmp.replace(path)                                     # atomic on POSIX and Windows
        except Exception:
            # Never leave scratch files behind for a save that failed; the store is a directory
            # the user can open, and a drift of dead .tmp files is its own small bug report.
            tmp.unlink(missing_ok=True)
            raise

    def save(self, pid, data, raw=None):
        p = self._path(pid)
        if not p:
            return
        body = dict(data)
        body["id"] = pid
        body["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._write_atomic(p, json.dumps(body))
        # The sidebar only ever needs these few fields; keeping them separate stops it from
        # parsing every full report on every refresh.
        self._write_atomic(self._path(pid, ".meta.json"), json.dumps(
            {"id": pid, "title": body.get("title") or "Untitled survey",
             "updated": body["updated"], "k": body.get("k"),
             "n_people": body.get("n_people"), "confidence": body.get("confidence")}))
        # The upload never changes for a given project, so write it once. Without this check every
        # chat message would rewrite the whole file — up to 8 MB per keystroke-sized interaction.
        rawp = self._path(pid, ".data")
        if raw is not None and len(raw) <= self.MAX_RAW and not (rawp and rawp.exists()):
            self._write_atomic(rawp, raw)

    def load(self, pid):
        p = self._path(pid)
        if not p or not p.exists():
            return None
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        data = self._path(pid, ".data")
        if data and data.exists():
            try:
                saved["raw"] = data.read_bytes()   # so the questions can still be re-picked
            except Exception:
                pass
        return saved

    def delete(self, pid):
        for suffix in (".json", ".meta.json", ".data"):
            p = self._path(pid, suffix)
            if p and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    def list(self, limit=60):
        out = []
        for f in self.root.glob("*.meta.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({"id": d.get("id", f.name.split(".")[0]),
                        "title": d.get("title") or "Untitled survey",
                        "updated": d.get("updated", ""), "k": d.get("k"),
                        "n_people": d.get("n_people"), "confidence": d.get("confidence")})
        out.sort(key=lambda d: d["updated"], reverse=True)
        return out[:limit]


def _shutdown_page():
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Closed</title>"
            "<style>body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;"
            "text-align:center;background:#f4f2ec;color:#28261f;font:16px/1.6 -apple-system,"
            "BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}"
            "@media(prefers-color-scheme:dark){body{background:#1f1d1a;color:#ece7dd}}</style></head>"
            "<body><div><h1>The app has closed.</h1><p>You can close this browser tab now.</p></div>"
            "</body></html>")


def serve(port=8000):
    """Start the local web app: a Claude-style chat page where anyone can drop a survey file, get the
    statistical report, and (with their own Anthropic API key) have Claude interpret the results and
    answer follow-up questions. Localhost only; the survey never leaves the computer. The optional AI
    layer sends ONLY the aggregate report to Anthropic, under the user's own account. Backs `--serve`
    and the packaged desktop app."""
    import http.server
    import threading
    import uuid
    import webbrowser

    sessions = {}    # session_id -> {"digest": report_markdown, "messages": [...]}  (in-memory, local)
    store = ProjectStore()

    _REHYDRATE = ("digest", "messages", "files", "title", "report_html", "columns", "k",
                  "n_people", "confidence", "transcript", "raw", "names", "charts",
                  "ranking")

    def session(sid):
        """Look up a session, falling back to disk before giving up.

        Only the last few sessions are held in memory. A user who analyses several files and then
        clicks Re-group on a card still on screen would otherwise be told to 'analyse a survey
        first' — the work is safely on disk, so the button was lying rather than the data being
        lost. Rehydrating here makes every action behave the same whether the session happens to
        be in memory or not."""
        if not sid:
            return None
        live = sessions.get(sid)
        if live:
            return live
        saved = store.load(sid)
        if not saved:
            return None
        sessions.setdefault(sid, {}).update({k: saved.get(k) for k in _REHYDRATE})
        sessions[sid].setdefault("messages", [])
        return sessions[sid]

    def remember(sid):
        """Persist a project so it survives a refresh or a restart."""
        s = session(sid)
        if not s:
            return
        store.save(sid, {"title": s.get("title"), "digest": s.get("digest"),
                         "messages": s.get("messages", []), "files": s.get("files", {}),
                         "report_html": s.get("report_html"), "columns": s.get("columns", {}),
                         "k": s.get("k"), "n_people": s.get("n_people"),
                         "confidence": s.get("confidence"), "transcript": s.get("transcript", []),
                         "charts": s.get("charts", []), "names": s.get("names", []),
                         "ranking": s.get("ranking")},
                   raw=s.get("raw"))

    class Handler(http.server.BaseHTTPRequestHandler):
        def _bytes(self, body, ctype="text/html; charset=utf-8", code=200):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def _json(self, obj, code=200):
            self._bytes(json.dumps(obj), "application/json; charset=utf-8", code)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                return {}

        def _route(self):
            """The path this request is for, with any query string removed.

            Routes are matched exactly. Prefix matching (`path.startswith("/project")`) was wrong
            in two ways that both reached users: `/projects-of-mine` was answered with the project
            list, and `/project` only avoided swallowing `/projects` because of the order the
            branches happened to be written in — one reordering away from a silent break.
            """
            from urllib.parse import urlparse
            return urlparse(self.path).path

        def do_GET(self):
            route = self._route()
            if route == "/quit":
                self._bytes(_shutdown_page())
                threading.Thread(target=httpd.shutdown, daemon=True).start()
                return
            if route == "/settings":
                self._json(sk._ai.status() if sk._ai else {"sdk_installed": False, "configured": False,
                                                     "source": None, "env_key": False, "model": None})
                return
            if route == "/download":
                self._do_download()
                return
            if route == "/projects":
                self._json({"ok": True, "projects": store.list()})
                return
            if route == "/project":
                from urllib.parse import parse_qs, urlparse
                pid = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
                saved = store.load(pid)
                if not saved:
                    self._json({"ok": False, "error": "That project could not be found."}, 404)
                    return
                # Re-open it in memory so chatting and downloading carry on where they left off.
                session(pid)
                st = sk._ai.status() if sk._ai else {}
                self._json({"ok": True, "session_id": pid, "title": saved.get("title"),
                            "report_html": saved.get("report_html") or "",
                            "downloads": sorted(saved.get("files") or {}),
                            "k": saved.get("k"), "n_people": saved.get("n_people"),
                            "columns": saved.get("columns") or {},
                            "confidence": saved.get("confidence") or "unknown",
                            "charts": _charts_for_browser(saved.get("charts")),
                            "names": saved.get("names") or [],
                            "ranking": saved.get("ranking"),
                            "transcript": saved.get("transcript") or [],
                            "ai_available": bool(st.get("configured") and st.get("sdk_installed")),
                            "reopened": True})
                return
            self._serve_ui()

        def _serve_ui(self):
            """Serve the built interface.

            Anything that is not an API route is either a file in the bundle or the app itself:
            the single page is returned for unknown paths so a reload of any URL comes back to a
            working app rather than a 404 from a server that has one page.
            """
            asset = _webui_asset(self.path)
            if asset is None:
                asset = _webui_asset("/index.html")
            if asset is None:
                self._bytes(_missing_ui_page(), code=503)
                return
            body, ctype = asset
            # Hashed filenames are content-addressed, so they can be cached hard; the HTML shell
            # names which hashes are current and must never be, or a rebuilt app keeps loading the
            # old bundle and the user sees a blank page they cannot fix by reloading.
            #
            # Decided on the served content type, not the requested path. Keying it on the path
            # meant `/?utm_source=x` and every single-page fallback route missed the exact-match
            # test and cached the shell for a year — the precise failure the no-store is for.
            cache = ("no-store" if ctype.startswith("text/html")
                     else "public, max-age=31536000, immutable")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _do_download(self):
            """Hand back one of this run's result files (who is in which group, what defines each
            group, the typing rule). Everything stays local — this is a read from memory, not a
            network fetch."""
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            sess = session((q.get("session_id") or [""])[0])
            name = (q.get("file") or [""])[0]
            if not sess or name not in sess.get("files", {}):
                self._bytes("Not found — analyse a survey first.", "text/plain; charset=utf-8", 404)
                return
            body = sess["files"][name].encode("utf-8")
            ctype = "application/json" if name.endswith(".json") else "text/csv"
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            route = self._route()
            try:
                if route == "/analyze":
                    self._do_analyze()
                elif route == "/score":
                    self._do_score()
                elif route == "/regroup":
                    self._do_regroup()
                elif route == "/name":
                    self._do_name()
                elif route == "/plan":
                    self._do_plan()
                elif route == "/design":
                    self._do_design()
                elif route == "/delete_project":
                    body = self._read_json()
                    store.delete(body.get("session_id", ""))
                    sessions.pop(body.get("session_id", ""), None)
                    self._json({"ok": True, "projects": store.list()})
                elif route == "/chat":
                    self._do_chat()
                elif route == "/settings":
                    self._do_settings()
                else:
                    self._json({"ok": False, "error": "Unknown request."}, 404)
            except Exception as e:                    # never leak a traceback to the browser
                self._json({"ok": False, "error": sk._explain_run_error(str(e))})

        def _do_analyze(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length > 100 * 1024 * 1024:
                self._json({"ok": False, "error": "That file is very large. Please upload a survey "
                            "export under 100 MB (one row per respondent)."})
                return
            data, filename = _parse_multipart_file(self.headers.get("Content-Type", ""),
                                                   self.rfile.read(length), with_name=True)
            if not data:
                self._json({"ok": False, "error": "Please choose a .csv or .xlsx file first."})
                return
            try:
                r = sk.run_analysis(data)
            except Exception as e:
                self._json({"ok": False, "error": sk._explain_run_error(str(e))})
                return
            sid = uuid.uuid4().hex
            # Keep the raw upload so the user can re-group on different questions without
            # re-uploading; bounded below so memory cannot creep.
            sessions[sid] = {"digest": r["digest"], "messages": [], "files": r["files"],
                             "raw": data, "title": filename or r["title"],
                             "report_html": r["report_html"], "columns": r.get("columns", {}),
                             "k": r["k"], "n_people": r["n_people"],
                             "confidence": r.get("confidence"), "charts": r.get("charts", []),
                             "ranking": r.get("ranking"),
                             "transcript": [{"role": "you", "text": f"Analyse: {filename}"}]}
            for old in list(sessions)[:-5]:    # bound memory: sessions hold the file + its results
                sessions.pop(old, None)
            remember(sid)
            self._json(self._analysis_payload(sid, r))

        def _analysis_payload(self, sid, r):
            st = sk._ai.status() if sk._ai else {}
            return {"ok": True, "session_id": sid, "title": r["title"],
                    "report_html": r["report_html"],
                    "ai_available": bool(st.get("configured") and st.get("sdk_installed")),
                    "downloads": sorted(r["files"]), "k": r["k"], "n_people": r["n_people"],
                    "columns": r.get("columns", {}),
                    "charts": _charts_for_browser(r.get("charts")),
                    "chart_errors": r.get("chart_errors") or [],
                    "ranking": r.get("ranking"),
                    "confidence": r.get("confidence", "unknown")}

        def _do_regroup(self):
            """Re-run on the SAME uploaded file, grouping people on the questions the user picked.
            The detector's guess is a starting point, not a verdict."""
            body = self._read_json()
            sid = body.get("session_id")
            sess = session(sid)
            if not sess:
                self._json({"ok": False, "error": "Please analyse a survey file first."})
                return
            if sess.get("raw") is None:
                # Saved projects keep the original upload, but a very large one is not stored.
                # Say what to do rather than pretending nothing was analysed.
                self._json({"ok": False, "error": "I no longer have the original file for this "
                            "project, so I cannot re-group it. Upload the survey again to pick "
                            "different questions."})
                return
            items = [c for c in (body.get("items") or []) if isinstance(c, str)]
            if len(items) < 2:
                self._json({"ok": False, "error": "Pick at least two questions to group people on."})
                return
            try:
                r = sk.run_analysis(sess["raw"], force_items=items)
            except Exception as e:
                self._json({"ok": False, "error": sk._explain_run_error(str(e))})
                return
            # Replace the whole stored result, not just part of it: leaving the old report_html and
            # counts behind would mean reopening the project showed the PREVIOUS grouping.
            # Names describe the OLD groups and there may now be a different number of them, so they
            # are dropped rather than silently re-applied to groups that mean something else.
            sess.update({"digest": r["digest"], "files": r["files"], "messages": [], "names": [],
                         "report_html": r["report_html"], "columns": r.get("columns", {}),
                         "k": r["k"], "n_people": r["n_people"],
                         "confidence": r.get("confidence"), "charts": r.get("charts", []),
                         "ranking": r.get("ranking"),
                         "transcript": [{"role": "you",
                                         "text": "Group people on: " + ", ".join(items)}]})
            remember(sid)
            self._json(self._analysis_payload(sid, r))

        def _do_score(self):
            """Assign BRAND-NEW people to the groups already found, using this run's typing rule.
            This is what turns a one-off study into something reusable: field the survey once, then
            score every later signup without re-segmenting."""
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            sid = (q.get("session_id") or [""])[0]
            sess = session(sid)
            if not sess or "typing_rule.json" not in sess.get("files", {}):
                self._json({"ok": False, "error": "Analyse a survey first, then score new people "
                                                  "against those groups."})
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length > 100 * 1024 * 1024:
                self._json({"ok": False, "error": "That file is very large. Please keep it under "
                                                  "100 MB (one row per person)."})
                return
            data = _parse_multipart_file(self.headers.get("Content-Type", ""), self.rfile.read(length))
            if not data:
                self._json({"ok": False, "error": "Please choose a .csv or .xlsx file of new people."})
                return
            try:
                rule = json.loads(sess["files"]["typing_rule.json"])
                classifier = sk.classify_new_lca if rule.get("method") == "lca" else sk.classify_new
                out = classifier(rule, sk._read_table(data))
            except Exception as e:
                self._json({"ok": False, "error": sk._explain_run_error(str(e))})
                return
            # Label the scored people with the SAME column name the original study used ("class" on
            # the categorical path), so the two files can be joined without a surprise.
            header = sess["files"].get("segment_assignments.csv", "").split("\n", 1)[0].split(",")
            # Match on the exact column name, not a substring: a student survey can legitimately
            # have its own column called "class" (a school class), and renaming then would collide.
            if "class" in header and "segment" not in header and "segment" in out.columns:
                out = out.rename(columns={"segment": "class"})
            group_col = "class" if "class" in out.columns else "segment"
            sess["files"]["scored_new_people.csv"] = out.to_csv(index=False)
            remember(sid)                                # keep the scored list with the project
            counts = out[group_col].value_counts().sort_index()
            # How many of these people answered something the original study never saw — a "no
            # answer" code such as 99 or -99. The scored CSV has carried this per respondent since
            # it was added, and the command line prints it, but the app said nothing, so the
            # people most likely to be scoring a follow-up were the least likely to hear about it.
            off_col = "answers_off_the_original_scale"
            off_scale = int((out[off_col] > 0).sum()) if off_col in out.columns else 0
            self._json({"ok": True, "n": int(len(out)),
                        "breakdown": {str(k): int(v) for k, v in counts.items()},
                        "mean_confidence": round(float(out["confidence"].mean()), 2),
                        "off_scale": off_scale,
                        "file": "scored_new_people.csv"})

        def _do_name(self):
            """Give the groups names a human recognises, and push them into the downloads. The
            auto-generated labels ("Q9 consideration BrandA + ...") are unusable in a campaign
            brief; this is what makes the exports shareable."""
            body = self._read_json()
            sess = session(body.get("session_id"))
            if not sess or "segment_assignments.csv" not in sess.get("files", {}):
                self._json({"ok": False, "error": "Please analyse a survey file first."})
                return
            assign = pd.read_csv(io.StringIO(sess["files"]["segment_assignments.csv"]))
            seg_col = "segment" if "segment" in assign.columns else "class"
            groups = sorted(assign[seg_col].unique())
            if body.get("suggest"):
                if sk._ai is None:
                    self._json({"ok": False, "kind": "nosdk",
                                "error": "The AI add-on is not installed, so I cannot suggest "
                                         "names. You can still type your own."})
                    return
                try:
                    names = sk._ai.suggest_names(sess["digest"], len(groups))
                except sk._ai.AIError as e:
                    self._json({"ok": False, "kind": e.kind, "error": str(e)})
                    return
                if len(names) != len(groups):
                    # A short list would raise IndexError below and surface as a cryptic error.
                    self._json({"ok": False, "error": "Claude suggested "
                                f"{len(names)} names for {len(groups)} groups. Try again, or type "
                                "the names yourself."})
                    return
            else:
                names = [str(n).strip() for n in (body.get("names") or [])]
                if len([n for n in names if n]) != len(groups):
                    self._json({"ok": False,
                                "error": f"Please give a name for each of the {len(groups)} groups."})
                    return
            mapping = {g: names[i] for i, g in enumerate(groups)}
            assign["group_name"] = assign[seg_col].map(mapping)
            sess["files"]["segment_assignments.csv"] = assign.to_csv(index=False)
            sess["files"]["group_names.csv"] = pd.DataFrame(
                {"segment": groups, "name": [mapping[g] for g in groups],
                 "people": [int((assign[seg_col] == g).sum()) for g in groups]}).to_csv(index=False)
            # The profiles file describes the same groups and was left labelled "Segment 0/1/2"
            # while the assignments beside it carried the chosen names — one thing under two names,
            # which is the fault the report itself was cleaned of. Give it the name as well; the
            # number stays, because that is what the assignments join on.
            profiles = sess["files"].get("group_profiles.csv")
            if profiles:
                try:
                    frame = pd.read_csv(io.StringIO(profiles))
                    label = frame.columns[0]
                    named = frame[label].astype(str).str.extract(r"(\d+)")[0].astype("Int64")
                    if named.notna().all():
                        frame.insert(1, "suggested name",
                                     [mapping.get(int(v), "") for v in named])
                        sess["files"]["group_profiles.csv"] = frame.to_csv(index=False)
                except Exception:
                    pass                     # a profiles file we cannot parse is not worth failing over
            sess["names"] = names
            remember(body.get("session_id"))             # names must survive reopening the project
            self._json({"ok": True, "names": names, "downloads": sorted(sess["files"])})

        def _do_chat(self):
            body = self._read_json()
            sess = session(body.get("session_id"))
            if not sess:
                self._json({"ok": False, "error": "Please analyse a survey file first."})
                return
            if sk._ai is None:
                self._json({"ok": False, "kind": "nosdk", "error": "The AI add-on is not installed. "
                            "Install it with  pip install anthropic  (or rebuild the app), then "
                            "reopen this page."})
                return
            question = None if body.get("initial") else (body.get("message") or "").strip()
            try:
                # The charts go with the digest so Claude reads the same picture the user is
                # looking at — the segment map in particular shows whether the groups separate,
                # which no amount of summary statistics conveys as directly.
                reply, sess["messages"] = sk._ai.chat_once(sess["messages"], sess["digest"], question,
                                                        charts=sess.get("charts"))
            except sk._ai.AIError as e:
                self._json({"ok": False, "kind": e.kind, "error": str(e)})
                return
            html = sk._markdown_to_html(reply)
            tr = sess.setdefault("transcript", [])
            if question:
                tr.append({"role": "you", "text": question})
            tr.append({"role": "ai", "html": html})
            remember(body.get("session_id"))
            self._json({"ok": True, "reply_html": html})

        def _do_plan(self):
            """How many respondents will a planned study need? Runs before there is any data.

            Deliberately a smaller sweep than the command line uses. The CLI answers into a
            terminal where a four-minute wait is normal; a browser spinner is not, and the sizes
            below still bracket the range where the answer actually changes — recovery turns over
            between about forty and a hundred and fifty respondents, so these five points carry the
            whole decision. Someone who wants the finer sweep has `segment-kmeans --plan`.
            """
            # Resolved BEFORE the parameters are checked, on purpose. `planner` is imported here
            # rather than at module load, and a lazy import is invisible to PyInstaller — the same
            # blindness that shipped every release without tabulate and without the dip test. Doing
            # it first means a request with deliberately bad parameters still proves the module is
            # present, so the build's smoke test can verify bundling in milliseconds rather than by
            # running a ninety-second sweep.
            try:
                import planner
            except Exception:
                self._json({"ok": False, "error": "The study planner is not installed."})
                return
            body = self._read_json()
            try:
                questions = int(body.get("questions", 6))
                segments = int(body.get("segments", 3))
            except (TypeError, ValueError):
                self._json({"ok": False, "error": "Give a whole number of questions and segments."})
                return
            if not 2 <= questions <= 60:
                self._json({"ok": False, "error": "Plan for between 2 and 60 questions."})
                return
            if not 2 <= segments <= 8:
                self._json({"ok": False, "error": "Plan for between 2 and 8 segments."})
                return
            try:
                plan = planner.plan_study(n_questions=questions, n_segments=segments,
                                          sizes=(50, 75, 100, 200, 400), seeds=5)
            except Exception as e:                       # a planner fault must not kill the app
                self._json({"ok": False, "error": sk._explain_run_error(str(e))})
                return
            advice = planner.recommend(plan)
            self._json({"ok": True, "cells": plan["cells"], "sizes": plan["sizes"],
                        "seeds": plan["seeds"], "questions": questions, "segments": segments,
                        "regimes": [{"name": n, "separation": d} for n, d in planner.REGIMES],
                        "recommended_n": advice["recommended_n"],
                        "subtle_reachable": advice["subtle_reachable"],
                        "prose": planner.render(plan)})

        def _do_design(self):
            """Build the best-worst questionnaire. Runs before there is anything to analyse.

            The command line takes a file of items; this takes them pasted into a box, because
            someone deciding what to ask has the list in an email or a slide, not saved as a .txt.

            The CSV goes back INSIDE the reply rather than being written to disk and downloaded
            through /download. Everything that route serves belongs to an analysed session, and a
            design has none — inventing a session to hold one file would mean it turned up in the
            project list as a study nobody ran. At the shapes allowed below the file is at most
            about a megabyte, which is a normal JSON reply for a local server.
            """
            # Imported before the parameters are checked, exactly as in _do_plan and for the same
            # reason: a lazy import is invisible to PyInstaller, so this line is what lets the
            # build's smoke test prove design.py was bundled without generating a whole design.
            try:
                import design as design_mod
            except Exception:
                self._json({"ok": False, "error": "The questionnaire designer is not installed."})
                return
            body = self._read_json()
            raw = body.get("items")
            # A bare string is the shape a caller most easily gets wrong, and it is the one that
            # fails silently: iterating "delivery" yields letters, so the app cheerfully built a
            # questionnaire comparing d, e, l, i, v, r and y. Anything that is not a list of text
            # is refused rather than coerced — str() of a dict is a perfectly good Python string
            # and a nonsensical thing to ask someone to choose between.
            if raw is not None and not isinstance(raw, list):
                self._json({"ok": False, "error": "Send the items as a list, one entry per item."})
                return
            if any(not isinstance(line, str) for line in (raw or [])):
                self._json({"ok": False, "error": "Every item has to be text."})
                return
            # Collapsed rather than merely stripped: an item carrying a newline splits a cell in
            # half in most survey platforms, which corrupts the import rather than looking wrong.
            items = [" ".join(str(line).split()) for line in (raw or [])]
            items = [line for line in items if line]
            # Long enough for a real benefit statement, short enough that forty of them cannot turn
            # a small design into a multi-megabyte reply. Truncating instead would silently change
            # the wording of the questionnaire, which is worse than saying no.
            too_long = [line for line in items if len(line) > 200]
            if too_long:
                self._json({"ok": False, "error": f"One item is {len(too_long[0])} characters long. "
                                                  f"Keep each under 200 — people have to read them "
                                                  f"on a screen and choose between them."})
                return
            seen, unique = set(), []
            for name in items:                      # a duplicated item would compete with itself
                if name.lower() not in seen:
                    seen.add(name.lower())
                    unique.append(name)
            try:
                per_screen = int(body.get("per_screen", 4))
                screens = int(body.get("screens", 10))
                people = int(body.get("people", 200))
            except (TypeError, ValueError):
                self._json({"ok": False, "error": "Give whole numbers for the questionnaire shape."})
                return
            if len(unique) < 3:
                self._json({"ok": False, "error": "List at least three items, one per line. A "
                                                  "best-worst exercise compares things, so there "
                                                  "have to be things to compare."})
                return
            # The ceilings are measured rather than guessed, and they are what the browser can wait
            # for: the largest shape allowed here takes about twenty seconds, while eight items on
            # a screen over twenty screens takes eighty. Anyone who needs that shape has
            # `segment-kmeans --design`, which is the same code with no clock on it.
            if len(unique) > 40:
                self._json({"ok": False, "error": f"That is {len(unique)} items. Forty is the most "
                                                  f"this screen will build; beyond that use "
                                                  f"segment-kmeans --design."})
                return
            if not 2 <= per_screen <= 6:
                self._json({"ok": False, "error": "Show between 2 and 6 items on each screen."})
                return
            if per_screen >= len(unique):
                self._json({"ok": False, "error": f"You cannot show {per_screen} of {len(unique)} "
                                                  f"items at once — there would be nothing left to "
                                                  f"compare them against."})
                return
            if not 2 <= screens <= 15:
                self._json({"ok": False, "error": "Ask between 2 and 15 screens per person."})
                return
            if not 20 <= people <= 300:
                self._json({"ok": False, "error": "Build between 20 and 300 versions."})
                return
            try:
                built, report = design_mod.make_design(len(unique), per_screen, screens, people)
            except ValueError as e:
                self._json({"ok": False, "error": sk._explain_run_error(str(e))})
                return
            self._json({"ok": True, "items": unique, "report": report,
                        "prose": design_mod.render(report),
                        "csv": design_mod.to_frame(built, unique).to_csv(index=False)})

        def _do_settings(self):
            if sk._ai is None:
                self._json({"ok": False, "error": "The AI add-on is not installed."})
                return
            body = self._read_json()
            try:
                if body.get("clear"):
                    sk._ai.clear_api_key()
                else:
                    sk._ai.save_api_key(body.get("api_key", ""))
            except sk._ai.AIError as e:
                self._json({"ok": False, "error": str(e)})
                return
            resp = {"ok": True}
            resp.update(sk._ai.status())
            self._json(resp)

        def log_message(self, *a):
            pass

    server_cls = getattr(http.server, "ThreadingHTTPServer", http.server.HTTPServer)
    httpd = server_cls(("127.0.0.1", port), Handler)   # threaded: the team can use it at once
    url = f"http://localhost:{httpd.server_address[1]}"
    print(f"\nThe Survey Segmenter is running. If your browser did not open, go to:  {url}\n"
          "It runs on your computer. Close it from the Quit button in the page.\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def app():
    """Entry point for the packaged desktop app: find a free port, start the local web app, and open
    the browser. Used by the PyInstaller build and by `segment-kmeans --app`."""
    import os
    import socket
    import sys
    # A windowed (double-clickable) app has no console, so stdout/stderr are None; guard the prints.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    s = socket.socket(); s.bind(("127.0.0.1", 0)); free = s.getsockname()[1]; s.close()
    serve(free)


