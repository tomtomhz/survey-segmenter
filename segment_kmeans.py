"""
segment_kmeans.py — Preference / utility segmentation via k-means, done to the standard of
the market-segmentation and cluster-validation literature (Mind-Genomics style)
================================================================================================

Clusters RESPONDENTS on their PREFERENCES (MaxDiff or conjoint utilities, attitude or
importance scores) to find distinct mind-sets, then selects the number of segments and
validates the solution the way the methodological authorities say it must be done — because,
as Dolnicar & Leisch show, data-driven segments are usually *constructed* by the algorithm
rather than *discovered* in nature, so the burden of proof is on stability and reproducibility,
not on a pretty elbow.

WHAT MAKES THIS DIFFERENT FROM A NAIVE k-means SCRIPT
----------------------------------------------------
1. PREPROCESSING that recovers structure. Defaults to RANGE standardization (divide each item
   by its range), which Milligan & Cooper (1988) found gives consistently better cluster
   recovery than the usual z-score, across error conditions, separations, and methods. z-score,
   raw, and ipsative (row-centred) scalings are also offered — the choice changes the answer.

2. MANY RANDOM STARTS. k-means is prone to local optima; Steinley (2003, "what you don't know
   may hurt you"; 2006 synthesis) shows this materially changes results, so the final fit uses
   many restarts and the tool reports how often the best solution was actually reached.

3. NUMBER-OF-SEGMENTS chosen by a panel, weighting STABILITY above fit. It computes:
   - Global replication stability (bootstrap Adjusted Rand Index) — the Dolnicar & Leisch
     criterion: repeat the segmentation and see if the same segments re-emerge.
   - Prediction strength (Tibshirani & Walther 2005) — largest k with prediction strength
     above 0.8, a cross-validated co-membership measure.
   - The Bayesian Information Criterion from a Gaussian mixture (the model-based / latent-class
     view, Wedel & Kamakura) — the principled criterion heuristic k-means otherwise lacks.
   - Silhouette (Rousseeuw), Calinski-Harabasz (the best single stopping rule in Milligan &
     Cooper's 1985 comparison), Davies-Bouldin, and the Tibshirani gap statistic.
   - The inertia elbow, shown but treated as the weakest signal.

4. VALIDATION that tells you WHICH segments are real, not just whether the solution "fits":
   - Per-cluster bootstrap Jaccard stability (Hennig 2007, the fpc `clusterboot` method), with
     the standard reading: mean Jaccard >= 0.85 highly stable, >= 0.75 stable/valid,
     0.6-0.75 a pattern with doubtful membership, < 0.5 dissolved (not a real cluster).
   - Split-half replication of the whole solution.

5. INTERPRETATION as mind-sets: mean utility per item, the items that most define each segment
   (above / below the grand mean), the items that most DIFFERENTIATE the segments (one-way
   ANOVA F), sizes, and an editable auto-name. Demographics are used only to PROFILE segments
   after the fact (chi-square), never to form them.

6. A TYPING TOOL to operationalize the result: an exportable nearest-centroid rule that assigns
   BRAND-NEW respondents to the discovered segments (as Mind Genomics does as standard), plus a
   leakage-free cross-validated estimate of how consistently that assignment can be made. High
   accuracy means the rule is deployable; it is NOT on its own proof the segments are real.

CATEGORICAL DATA: for multiple-choice / agree-disagree items (not continuous utilities) use
    --method lca, a true Latent Class Analysis (finite mixture of categorical distributions under
    local independence; Lazarsfeld-Goodman, Wedel & Kamakura), fit by EM, chosen by BIC and ICL,
    and validated with the same stability-first machinery. k-means/gmm are for continuous inputs.

INPUT : a CSV, one row per respondent, one numeric column per item (or categorical columns for
        --method lca); optional id column; optional separate demographics CSV for profiling.
OUTPUT: cluster assignments, segment-by-item centroids, the full k-selection diagnostics, a
        readable Markdown report, a portable typing_rule.json for classifying new respondents,
        and (if matplotlib is present) a diagnostics figure.
        Apply a saved rule:  python segment_kmeans.py --classify new.csv --rule typing_rule.json

COMMAND LINE (after `pip install .` the command is `segment-kmeans`; or run `python segment_kmeans.py`):
    segment-kmeans my_survey.csv                        # auto: detects everything, writes a report
    segment-kmeans --serve                              # local web page: upload a CSV in a browser
    segment-kmeans data.csv --method lca --outdir out   # explicit control (kmeans | gmm | lca)
    segment-kmeans --classify new.csv --rule out/typing_rule.json    # type NEW respondents
    segment-kmeans --version

LIBRARY:
    from segment_kmeans import Segmenter, SegmentationConfig
    seg = Segmenter(SegmentationConfig(scaling="range")).run("utilities.csv", id_col="id")

Dependencies: numpy, pandas, scikit-learn, scipy (required); matplotlib, tabulate, openpyxl
(figures, prettier tables, .xlsx reading) optional. Install as a tool with `pip install .`.

KEY REFERENCES
- Dolnicar, S. & Leisch, F. (2010, 2017). Market segmentation stability / "Market Segmentation
  Analysis" — stability of repeated segmentation as the selection criterion; natural vs.
  reproducible vs. constructed segments.
- Hennig, C. (2007). Cluster-wise assessment of cluster stability (bootstrap Jaccard; fpc::clusterboot).
- Milligan, G.W. & Cooper, M.C. (1985) stopping rules; (1988) standardization of variables.
- Tibshirani, R., Walther, G. & Hastie, T. (2001) gap statistic; Tibshirani & Walther (2005)
  prediction strength.
- Steinley, D. (2003) local optima; (2006) K-means: a half-century synthesis.
- Rousseeuw, P. (1987) silhouettes. Calinski & Harabasz (1974). Davies & Bouldin (1979).
- Wedel, M. & Kamakura, W. (2000). Market Segmentation: Conceptual and Methodological Foundations
  (finite-mixture / latent-class segmentation; BIC for the number of segments).
"""
from __future__ import annotations

import argparse
import csv
import html as _html
import io
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score)
from sklearn.neighbors import NearestNeighbors

import kprototypes
import clusterability


def _use_utf8_for_output():
    """Make console output UTF-8 everywhere, which on Windows it is not by default.

    Found by the first Windows build: the confidence line prints a coloured circle, and Windows'
    legacy console encoding (cp1252) cannot encode it — so the analysis raised UnicodeEncodeError
    and every run failed with "something went wrong reading that file". The emoji is the symptom;
    the bug is the encoding, and it would have hit any Swedish column name just as hard, which
    for a Nordic survey tool is the more serious half.

    `errors="replace"` as well as UTF-8: a console genuinely stuck on a legacy code page should
    print a question mark, never abort the analysis.
    """
    # A windowed Windows build has no console at all: PyInstaller leaves sys.stdout and
    # sys.stderr as None, so every print() in this module would raise AttributeError. Give them
    # somewhere to go — a file when SEG_LOG is set, which is how the packaged app is diagnosed,
    # and the bit bucket otherwise.
    target = os.environ.get("SEG_LOG")
    if target or sys.stdout is None or sys.stderr is None:
        try:
            sink = (open(target, "a", buffering=1, encoding="utf-8", errors="replace")
                if target else open(os.devnull, "w"))
        except Exception:
            sink = io.StringIO()
        # SEG_LOG redirects unconditionally, not only when the streams are missing: the whole
        # point is to get the app's own account of itself out of a build machine, and on Windows
        # the streams may well exist while going somewhere nobody can read.
        if target or sys.stdout is None:
            sys.stdout = sink
        if target or sys.stderr is None:
            sys.stderr = sink

    for stream in (sys.stdout, sys.stderr):
        try:
            # line_buffering is not optional here. reconfigure() rebuilds the text wrapper and
            # resets buffering to the default, which for a redirected stream is block buffering —
            # so it silently undoes PYTHONUNBUFFERED. That cost an hour: the packaged app was
            # printing exactly why it could not draw a chart, into a buffer that was never
            # flushed because the process was killed rather than exiting.
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass                     # older Python, or a stream that is not reconfigurable


_use_utf8_for_output()

# numpy 2.0 on macOS/Accelerate emits spurious "... encountered in matmul" RuntimeWarnings from
# ordinary matrix products inside scikit-learn; they do not affect results. Filter only these.
for _m in ("divide by zero encountered in matmul", "overflow encountered in matmul",
           "invalid value encountered in matmul"):
    warnings.filterwarnings("ignore", message=_m, category=RuntimeWarning)

__version__ = "1.14.1"    # keep in sync with pyproject.toml

# Optional "ask Claude about your segments" add-on. Imported here (not lazily) so the packaged app
# bundles it; wrapped so a missing file/SDK never stops the core segmentation tool from loading.
try:
    import maxdiff as _maxdiff
except Exception:            # MaxDiff scoring is optional; rating-grid surveys never touch it
    _maxdiff = None

try:
    import ai_interpret as _ai
except Exception:       # pragma: no cover - defensive: the tool must run without the AI layer
    _ai = None


# =====================================================================================
# Configuration
# =====================================================================================
@dataclass
class SegmentationConfig:
    k_min: int = 2
    k_max: int = 8
    # "kmeans" (heuristic), "gmm" (model-based / finite-mixture / latent-class, Wedel & Kamakura),
    # or "kproto" (Gower k-prototypes for surveys mixing rating and pick-any questions,
    # Szepannek et al. 2024 — see kprototypes.py)
    method: str = "kmeans"
    gmm_covariance: str = "full"    # "full" | "tied" | "diag" | "spherical" (gmm method only)
    # Set only on the kproto path: the per-variable Gower constants, fitted once on the whole
    # sample. Carried here rather than passed around because every resampling routine already
    # threads cfg through, and the spec must be identical on every resample for the stability
    # numbers to mean anything.
    gower_spec: object = None
    # {column name: "numeric" | "ordinal" | "nominal"} on the kproto path. Keyed by name rather
    # than position because load_and_prepare drops columns nobody varied on, which would shift
    # every kind after the dropped one if these were a plain list.
    var_kinds: dict | None = None
    # {column name: {code: original answer}} so the report can print "Nespresso" where the model
    # holds a 2. Pick-any answers have to be coded to travel through a float matrix; without this
    # the profiles would show the codes.
    level_labels: dict | None = None
    # "range" (Milligan-Cooper, recommended), "standardize" (z-score), "robust" (median/IQR,
    # outlier-resistant), "none", "ipsative" (row-centred, segments on preference shape)
    scaling: str = "range"
    n_init_final: int = 50          # restarts for the final fit (Steinley: use many)
    n_init_search: int = 20         # restarts during the k-search (speed/quality trade-off)
    gap_B: int = 20                 # reference datasets for the gap statistic
    stability_B: int = 30           # resamples for global replication stability (bootstrap ARI)
    stability_frac: float = 0.80
    ps_splits: int = 10             # splits for prediction strength (each gives 2 clusterings)
    jaccard_B: int = 100            # bootstraps for per-cluster Jaccard stability (Hennig: ~100)
    ps_cutoff: float = 0.80         # prediction-strength threshold (Tibshirani-Walther)
    stability_cutoff: float = 0.75  # "stable" replication ARI (Dolnicar-style reading)
    fit_gmm_bic: bool = True        # model-based (latent-class) cross-check on k
    run_consensus: bool = True      # Monti consensus clustering + PAC (adds runtime; --no-consensus to skip)
    consensus_H: int = 50           # resamples per k for the consensus matrix
    consensus_frac: float = 0.80    # subsample fraction for consensus resampling
    use_consensus_final: bool = False   # adopt the consensus ensemble partition as the final labels
    random_state: int = 42
    top_items: int = 4
    impute: str = "mean"            # "mean" | "drop"
    min_segment_frac: float = 0.05
    check_variable_selection: bool = True   # Dolnicar: re-cluster without near-noise items and compare


# =====================================================================================
# Scaling (fit once, re-apply to new respondents)
# =====================================================================================
# Kept as a fit/apply pair rather than inline so a saved segmentation can scale NEW respondents
# with EXACTLY the parameters learned on the original sample. That matters twice: for the typing
# tool (classify_new), and for honest cross-validation (scaling is refit inside each fold, so a
# held-out respondent is never scaled with parameters that saw them).
def _scale_fit(arr, scaling):
    """Fit a scaling on `arr`; return (scaled array, params dict). params carries everything
    _scale_apply needs to transform new rows the same way."""
    arr = np.asarray(arr, float)
    if scaling == "range":                          # Milligan-Cooper: divide by the range
        lo = arr.min(0); rng_ = arr.max(0) - lo
        return (arr - lo) / np.where(rng_ > 0, rng_, 1.0), \
            {"scaling": "range", "lo": lo.tolist(), "range": rng_.tolist()}
    if scaling == "standardize":                    # z-score (population std, like StandardScaler)
        mu = arr.mean(0); sd = arr.std(0)
        return (arr - mu) / np.where(sd > 0, sd, 1.0), \
            {"scaling": "standardize", "mean": mu.tolist(), "std": sd.tolist()}
    if scaling == "robust":                         # median/IQR: resistant scale estimate
        med = np.median(arr, 0); q25, q75 = np.percentile(arr, [25, 75], axis=0); iqr = q75 - q25
        return (arr - med) / np.where(iqr > 0, iqr, 1.0), \
            {"scaling": "robust", "median": med.tolist(), "iqr": iqr.tolist()}
    if scaling == "ipsative":                       # row-centre: segment on preference SHAPE
        return arr - arr.mean(1, keepdims=True), {"scaling": "ipsative"}
    if scaling == "none":
        return arr.copy(), {"scaling": "none"}
    raise ValueError(f"Unknown scaling '{scaling}'.")


def _scale_apply(arr, params):
    """Apply a scaling fitted by _scale_fit to NEW rows (ipsative/none are row-local, so they
    need no fitted parameters)."""
    arr = np.asarray(arr, float); s = params["scaling"]
    if s == "range":
        lo = np.asarray(params["lo"]); rng_ = np.asarray(params["range"])
        return (arr - lo) / np.where(rng_ > 0, rng_, 1.0)
    if s == "standardize":
        mu = np.asarray(params["mean"]); sd = np.asarray(params["std"])
        return (arr - mu) / np.where(sd > 0, sd, 1.0)
    if s == "robust":
        med = np.asarray(params["median"]); iqr = np.asarray(params["iqr"])
        return (arr - med) / np.where(iqr > 0, iqr, 1.0)
    if s == "ipsative":
        return arr - arr.mean(1, keepdims=True)
    if s == "none":
        return arr.copy()
    raise ValueError(f"Unknown scaling '{s}'.")


# =====================================================================================
# Data loading and preparation
# =====================================================================================
_DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")


def _skip_preamble_lines(text):
    """Drop title and spacer lines above the column names of a CSV.

    A spreadsheet exported with "Customer Survey — Q1 2026 Results" on the first line breaks the
    delimiter sniffer before it breaks anything else: the sniffer looks at that line, decides the
    file is space-separated, and returns seven columns called Customer, Survey, Q1, 2026...
    (`_promote_header_below_a_title` catches the Excel form of this, where the columns come back
    as "Unnamed: 1"; a CSV never gets that far.)

    Real rows agree with each other about how many separators they contain; a sentence does not.
    So the separator is whichever candidate has the most consistent count across the file, and any
    leading line that disagrees with that count is preamble.
    """
    lines = text.splitlines()
    if len(lines) < 4:
        return text
    body = [ln for ln in lines if ln.strip()][:200]
    best_sep, best_count, best_agree = None, 0, 0.0
    for sep in (",", ";", "\t", "|"):
        counts = [ln.count(sep) for ln in body]
        common = max(set(counts), key=counts.count)
        if common == 0:
            continue
        agree = counts.count(common) / len(counts)
        if agree > best_agree or (agree == best_agree and common > best_count):
            best_sep, best_count, best_agree = sep, common, agree
    if best_sep is None or best_agree < 0.8:
        return text                       # no consistent shape to compare against; leave it alone
    for i, ln in enumerate(lines[:5]):
        if ln.count(best_sep) == best_count and ln.strip():
            return "\n".join(lines[i:]) if i else text
    return text


def _promote_header_below_a_title(df):
    """Find the real header when somebody has typed a title above it.

    A spreadsheet that opens "Customer Survey — Q1 2026 Results" in A1 and puts the column names
    on row 3 is completely ordinary, and pandas takes that title as the header: one named column
    and the rest called "Unnamed: 1", "Unnamed: 2"... Measured on such a file, every real column
    name became data, the whole sheet read as text, and the survey was routed down the categorical
    path — 302 rows where there were 300 respondents, and no complaint anywhere.

    A row of column names is recognisable: it fills nearly every column, and its entries are
    distinct — titles and blank spacer rows are neither. Only the first few rows are considered,
    and only when the header is mostly "Unnamed", so an ordinary file is never touched.
    """
    cols = [str(c) for c in df.columns]
    unnamed = sum(c.startswith("Unnamed:") or c.strip() == "" for c in cols)
    if not len(cols) or unnamed <= len(cols) / 2 or len(df) < 3:
        return df
    for i in range(min(4, len(df) - 1)):
        row = df.iloc[i]
        filled = row.dropna()
        if len(filled) < 0.8 * len(cols):
            continue                                   # a title or a spacer, not the header
        names = [str(v).strip() for v in row]
        if len(set(names)) < len(names):
            continue                                   # column names do not repeat
        out = df.iloc[i + 1:].reset_index(drop=True)
        out.columns = names
        for c in out.columns:                          # re-type: it was text only because of this
            converted = pd.to_numeric(out[c], errors="coerce")
            if converted.notna().sum() == out[c].notna().sum():
                out[c] = converted
        return out
    return df


def _strip_metadata_rows(df):
    """Drop the extra header rows that professional survey tools put above the data.

    Qualtrics exports three header rows — short name, full question text, and a JSON blob like
    {"ImportId": "QID1"} — and pandas keeps only the first as the header, so the other two arrive
    as respondents. SurveyMonkey does the same with two. Measured on a 240-person export: the file
    read as 242 rows, every column came out as text because the question wording contaminated it,
    and the survey was then routed to latent class analysis and its rating scales treated as
    unordered categories. No error at any point.

    A row is a header remnant if it is non-numeric in columns that are otherwise numeric — a
    respondent's answer parses like the answers below it, and a question's wording does not. On a
    genuinely categorical survey no column is "otherwise numeric", nothing qualifies, and nothing
    is dropped; that is what makes this safe rather than clever.
    """
    if len(df) < 4:
        return df, 0
    head = df.head(3)
    body = df.iloc[3:]
    # Columns the body agrees are numeric. Blank cells are not evidence either way.
    numeric_cols = []
    for c in df.columns:
        vals = body[c].dropna()
        if len(vals) and pd.to_numeric(vals, errors="coerce").notna().mean() >= 0.8:
            numeric_cols.append(c)
    if not numeric_cols:
        return df, 0
    drop = 0
    for i in range(min(3, len(head))):
        row = head.iloc[i]
        offending = [c for c in numeric_cols
                     if pd.notna(row[c]) and pd.isna(pd.to_numeric(pd.Series([row[c]]),
                                                                   errors="coerce").iloc[0])]
        if len(offending) < max(1, len(numeric_cols) // 2):
            break                       # this row answers like a respondent; stop here
        drop = i + 1
    if not drop:
        return df, 0
    out = df.iloc[drop:].reset_index(drop=True)
    # Re-type each column now the wording is gone: it was read as text only because of the rows
    # just removed. Written out explicitly rather than with to_numeric(errors="ignore"), which
    # pandas deprecates and will RAISE on in a later version — a read path that starts throwing on
    # a routine dependency bump is the silent-breakage pattern this project keeps meeting.
    for c in out.columns:
        converted = pd.to_numeric(out[c], errors="coerce")
        if converted.notna().sum() == out[c].notna().sum():
            out[c] = converted
    return out, drop


def _fix_decimal_commas(df):
    """Swedish/German Excel writes 4,5 for four-and-a-half. Such a column arrives as text and is
    then dropped from the rating grid — measured on a Swedish export whose 0-10 satisfaction score
    vanished from the analysis without comment.

    Only whole columns are converted, and only when every value is digits-comma-digits. If the file
    had been comma-delimited such a value could not have survived as one cell, so this cannot
    misread a comma-separated multi-select answer.
    """
    for c in df.columns:
        s = df[c]
        # Ask whether the column is numeric, not whether its dtype is exactly `object`. pandas 3
        # gives text columns a `str` dtype, so an `!= object` guard skipped every one of them and
        # this repair silently stopped happening -- caught by CI on 3.11/3.12 while the local
        # pandas 2.3 still said `object`.
        if pd.api.types.is_numeric_dtype(s):
            continue
        vals = s.dropna().astype(str)
        if len(vals) and vals.str.match(_DECIMAL_COMMA).all():
            df[c] = pd.to_numeric(vals.str.replace(",", ".", regex=False)
                                  .reindex(s.index), errors="coerce")
    return df


def _read_table(source):
    """Read a survey export ROBUSTLY, the way real exports actually arrive. Handles comma OR
    semicolon delimiters (European / Excel exports, e.g. Swedish locale), UTF-8, UTF-8-with-BOM,
    and Latin-1 (so aao survive), and .xlsx/.xls if openpyxl is available. Accepts a path, raw
    bytes, a file-like object, or an existing DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()      # the caller has already decided what this contains

    def _clean(df):
        """Find the real header, drop the tools' extra header rows, and repair decimal commas.
        Each of the three otherwise degrades the analysis without a word: see the helpers above."""
        df = _promote_header_below_a_title(df)
        df, _ = _strip_metadata_rows(df)
        return _fix_decimal_commas(df)

    is_excel = ((isinstance(source, str) and source.lower().endswith((".xlsx", ".xls")))
                or (isinstance(source, (bytes, bytearray)) and bytes(source[:2]) == b"PK"))
    if is_excel:
        try:
            return _clean(pd.read_excel(io.BytesIO(source)
                                        if isinstance(source, (bytes, bytearray)) else source))
        except ImportError:
            raise ValueError("_NEED_OPENPYXL")
    def _buf():
        if isinstance(source, (bytes, bytearray)):
            return io.BytesIO(source)
        if hasattr(source, "seek"):
            source.seek(0)
        return source
    read_errors = (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error)
    # utf-16 sits between them deliberately. It is what Excel's "Unicode Text (*.txt)" export
    # writes, and latin-1 decodes ANY byte without complaining — so with utf-16 missing, such a
    # file came back as a single column named 'ÿþr' full of mojibake, and the run failed later
    # with "no questions found" rather than anything about encodings.
    for enc in ("utf-8-sig", "utf-16", "latin-1"):     # utf-8-sig also decodes plain UTF-8
        try:
            raw = _buf()
            data = raw.read() if hasattr(raw, "read") else Path(str(source)).read_bytes()
            if isinstance(data, bytes):
                data = data.decode(enc)
            # Strip any title or spacer lines first: they mislead the delimiter sniffer below.
            df = pd.read_csv(io.StringIO(_skip_preamble_lines(data)), sep=None, engine="python")
            if df.shape[1] >= 1 and len(df):
                return _clean(df)
        except read_errors:
            continue
    try:                                            # last resort: plain comma, skip unparsable lines
        df = pd.read_csv(_buf(), encoding="latin-1", on_bad_lines="skip")
        if df.shape[1] >= 1 and len(df):
            return _clean(df)
    except read_errors:
        pass
    raise ValueError("_BAD_FILE")                   # empty, binary, or not a table at all


def load_and_prepare(path, cfg: SegmentationConfig, id_col: str | None,
                     item_cols: list[str] | None):
    df = _read_table(path)
    ids = df[id_col].astype(str).to_numpy() if id_col and id_col in df else \
        np.array([f"r{i}" for i in range(len(df))])
    if item_cols is None:
        item_cols = [c for c in df.columns if c != id_col and
                     pd.api.types.is_numeric_dtype(df[c])]
    if len(item_cols) < 2:
        raise ValueError("Need at least two numeric item columns to segment on.")
    X = df[item_cols].copy()

    # Non-finite guard: infinities corrupt scaling and clustering silently, so treat them as
    # missing and let the imputation path handle them (with a visible warning).
    n_inf = int(np.isinf(X.to_numpy(float)).sum())
    if n_inf:
        print(f"WARNING: {n_inf} non-finite (inf) value(s) found; treating them as missing.")
        X = X.replace([np.inf, -np.inf], np.nan)

    kinds = dict(cfg.var_kinds or {})
    if X.isna().any().any():
        if cfg.impute == "drop":
            keep = ~X.isna().any(axis=1)
            X, ids = X.loc[keep], ids[keep.to_numpy()]
            print(f"Dropped {(~keep).sum()} rows with missing values.")
        else:
            # The mean of a set of brand codes is not a brand. Pick-any answers get the most
            # common answer instead, which is at least one somebody gave.
            fill = {c: (X[c].mode().iloc[0] if kinds.get(c) == kprototypes.NOMINAL
                                               and not X[c].mode().empty else X[c].mean())
                    for c in X.columns}
            X = X.fillna(fill)
            print("Imputed missing cells with the item mean (most common answer for "
                  "pick-any questions)." if kinds else "Imputed missing cells with the item mean.")

    nonconst = X.std(axis=0) > 1e-12
    if not nonconst.all():
        print(f"Dropping constant item(s): {list(X.columns[~nonconst])}")
        X = X.loc[:, nonconst]
    X_raw = X.reset_index(drop=True)
    if getattr(cfg, "method", "kmeans") == "kproto":
        # No scaling step: Gower normalises each variable by its own range as part of the
        # distance, so scaling here would be applied twice.
        col_kinds = [kinds.get(c, kprototypes.ORDINAL) for c in X_raw.columns]
        spec = kprototypes.fit_spec(X_raw.to_numpy(float), col_kinds)
        cfg.gower_spec = spec
        scale_params = {"scaling": "gower", "items": list(X_raw.columns), "kinds": col_kinds}
        return kprototypes.encode(X_raw.to_numpy(float), spec), X_raw, ids, scale_params
    Xs, scale_params = _scale_fit(X_raw.to_numpy(float), cfg.scaling)
    scale_params["items"] = list(X_raw.columns)
    return Xs, X_raw, ids, scale_params


# =====================================================================================
# Helpers
# =====================================================================================
def _km(X, k, n_init, seed):
    return KMeans(n_clusters=k, n_init=n_init, random_state=int(seed)).fit(X)


class _GMMWrap:
    """Wraps a fitted Gaussian mixture so it exposes the same .labels_/.predict()/.cluster_centers_
    interface the rest of the pipeline expects from k-means."""
    def __init__(self, gm, X):
        self.gm = gm
        self.labels_ = gm.predict(X)
        self.cluster_centers_ = gm.means_
        self.inertia_ = float(((X - gm.means_[self.labels_]) ** 2).sum())

    def predict(self, Y):
        return self.gm.predict(Y)


def _fit(X, k, cfg, n_init, seed):
    """Method-aware base learner. Everything downstream (stability, prediction strength,
    split-half, per-cluster Jaccard, the final fit) goes through this, so choosing the model
    (k-means vs. Gaussian mixture) changes the whole pipeline consistently, not just a label."""
    if getattr(cfg, "method", "kmeans") == "kproto":
        # Fewer restarts than k-means gets: each one costs a full Gower pass, and the seeding is
        # k-means++ so the restarts agree with each other far more often than random starts would.
        return kprototypes.KPrototypes(k, cfg.gower_spec, n_init=max(1, n_init // 4),
                                       random_state=int(seed)).fit(X)
    if getattr(cfg, "method", "kmeans") == "gmm":
        # Try the requested covariance; on a degenerate/singular subsample fall back to simpler
        # covariances, and only then to k-means, so resampling never crashes the run.
        for cov in (cfg.gmm_covariance, "diag", "spherical"):
            try:
                gm = GaussianMixture(n_components=k, covariance_type=cov,
                                     n_init=max(1, n_init // 5), random_state=int(seed),
                                     reg_covar=1e-4).fit(X)
                return _GMMWrap(gm, X)
            except Exception:
                continue
    return KMeans(n_clusters=k, n_init=n_init, random_state=int(seed)).fit(X)


def _is_gower(cfg):
    return getattr(cfg, "method", "kmeans") == "kproto" and getattr(cfg, "gower_spec", None)


def _config_for_manifest(cfg):
    """The config as plain JSON. The Gower spec is measured from the data rather than chosen by
    anyone, and it holds numpy arrays, so it goes in as its own serialisable form — the manifest
    stays a complete record of the run instead of becoming un-writable."""
    blob = asdict(cfg)
    spec = blob.get("gower_spec")
    blob["gower_spec"] = spec.to_json() if hasattr(spec, "to_json") else None
    return blob


def _geometry(X, cfg):
    """Coordinates and metric for anything that measures distance, so one call site serves both
    paradigms. On the numeric path this is the scaled data under Euclidean distance; on the mixed
    path it is the Manhattan embedding of Gower's distance (see kprototypes.gower_embedding),
    which lets the silhouette, the cluster-tendency test and the segment map keep using the same
    library routines instead of growing hand-written Gower twins."""
    if _is_gower(cfg):
        return kprototypes.gower_embedding(np.asarray(X, float), cfg.gower_spec), "manhattan"
    return np.asarray(X, float), "euclidean"


# Above this many respondents, a diagnostic that needs every PAIR of people is computed on a
# random subsample of this size, and the report says so. The pair count grows as the square: a
# dense pairwise matrix is 288 MB at 6,000 people, 3.2 GB at 20,000 and 20 GB at 50,000 — and the
# consensus routine holds two of them, per k. Measured: a 41,188-respondent file asked for 27 GB
# and would have taken the machine down before producing anything, with no guard anywhere.
#
# Subsample rather than skip, where the quantity allows it. The consensus matrix is itself a
# resampling method and PAC is a summary of it; the silhouette is an average over respondents.
# Both mean the same thing on a random sample of this size. The Ward cross-check below skips
# instead, correctly — a partition of a sample cannot be compared with one of everybody. What none
# of them may do is report a sampled number as though it covered the whole study.
MAX_PAIRWISE_N = 6000

# The working set for the k-selection panel and the validation estimates. Above this many
# respondents those are computed from a random sample of this size; the final fit, every
# respondent's segment, the profiles, the charts and the exports always use the whole file.
#
# 12,000 rather than less because the quantities estimated are proportions and rand indices whose
# standard error is already small at a few thousand, and rather than more because the panel
# performs well over a thousand clusterings of whatever it is handed — which is what turned 48,842
# respondents into 11 GB and two minutes, through no single allocation but a thousand small ones.
MAX_SEARCH_N = 12000


def _pairwise_sample(n, cfg):
    """Row indices for a diagnostic that needs every pair, or None meaning 'use everybody'."""
    if n <= MAX_PAIRWISE_N:
        return None
    return np.random.default_rng(cfg.random_state + 77).choice(n, MAX_PAIRWISE_N, replace=False)


def _silhouette(X, labels, cfg):
    coords, metric = _geometry(X, cfg)
    take = _pairwise_sample(len(coords), cfg)
    if take is not None:
        return float(silhouette_score(coords[take], np.asarray(labels)[take], metric=metric))
    return float(silhouette_score(coords, labels, metric=metric))


def _internal_indices(X, labels, cfg):
    """Silhouette, Calinski-Harabasz and Davies-Bouldin, the three separation indices.

    The silhouette is a ratio of distances and so is exact under Gower. The other two are built
    from sums of squares and exist only in a Euclidean space; on the mixed-type path they are
    therefore read on the Gower embedding under Euclidean distance rather than Gower's own. That
    is a real, well-defined reading of a real coordinate space, but it is not Gower's, and the
    report says so — these two sit in the middle tier of the panel, below prediction strength and
    replication, so the panel does not turn on them."""
    coords, metric = _geometry(X, cfg)
    # Only the silhouette needs every pair; Calinski-Harabasz and Davies-Bouldin are built from
    # cluster centres and stay exact on everybody however large the study.
    return {"silhouette": _silhouette(X, labels, cfg),
            "calinski_harabasz": float(calinski_harabasz_score(coords, labels)),
            "davies_bouldin": float(davies_bouldin_score(coords, labels))}


def _pooled_within_ss(X, labels, cfg=None):
    """Within-cluster dispersion, in whichever units the method's own objective uses.

    Squared distance to the cluster mean for k-means. On the mixed-type path, Gower distance to
    the cluster prototype, because that is what k-prototypes minimises — mixing an L1 objective
    with an L2 dispersion would make the gap statistic score a quantity nothing is optimising."""
    if _is_gower(cfg):
        spec = cfg.gower_spec
        total = 0.0
        for c in np.unique(labels):
            pts = X[labels == c]
            if len(pts) > 1:
                proto = kprototypes._update(pts, spec)
                total += float(kprototypes.gower_distances(pts, proto[None, :], spec).sum())
        return total
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) > 1:
            total += ((pts - pts.mean(0)) ** 2).sum()
    return total


# =====================================================================================
# k-selection diagnostics
# =====================================================================================
def gap_statistic(X, k_range, B, rng, n_init, cfg=None):
    """Tibshirani-Walther-Hastie (2001), reference method (b): uniform over the bounding box
    of the data rotated to its principal components, then rotated back — the recommended,
    shape-aware reference distribution. Recommended k: smallest with gap(k) >= gap(k+1)-se(k+1).

    Method (b) needs principal components, so it is only available where the data are all
    measurements. The mixed-type path falls back to the paper's method (a), each variable drawn
    independently over its own observed support — less powerful against elongated structure, but
    the only one of the two that produces answer combinations a respondent could actually give."""
    rows = []
    if _is_gower(cfg):
        spec = cfg.gower_spec
        for k in k_range:
            logWk = np.log(_pooled_within_ss(
                X, _fit(X, k, cfg, n_init, rng.integers(1e9)).labels_, cfg) + 1e-12)
            refs = np.empty(B)
            for b in range(B):
                ref = kprototypes.reference_sample(X, spec, len(X), rng)
                refs[b] = np.log(_pooled_within_ss(
                    ref, _fit(ref, k, cfg, 1, rng.integers(1e9)).labels_, cfg) + 1e-12)
            rows.append({"k": k, "gap": refs.mean() - logWk,
                         "gap_se": refs.std() * np.sqrt(1.0 + 1.0 / B)})
        return pd.DataFrame(rows)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ Vt.T
    lo, hi = Xp.min(0), Xp.max(0)
    for k in k_range:
        logWk = np.log(_pooled_within_ss(X, _km(X, k, n_init, rng.integers(1e9)).labels_) + 1e-12)
        refs = np.empty(B)
        for b in range(B):
            Zp = rng.uniform(lo, hi, size=Xp.shape)   # Xp may have fewer columns than X when n <= d
            ref = Zp @ Vt + X.mean(0)
            refs[b] = np.log(_pooled_within_ss(ref, _km(ref, k, 1, rng.integers(1e9)).labels_) + 1e-12)
        rows.append({"k": k, "gap": refs.mean() - logWk,
                     "gap_se": refs.std() * np.sqrt(1.0 + 1.0 / B)})
    return pd.DataFrame(rows)


def supports_single_cluster(X, B, rng, n_init, cfg=None):
    """Does the gap statistic say this data is ONE group — that is, no segmentation at all?

    The gap statistic is the only criterion in the panel that can answer this. Hastie, Tibshirani
    and Friedman put it plainly in *The Elements of Statistical Learning* (§14.3.11): the gap
    "works reasonably well when the data fall into a single cluster, and in that case will tend to
    estimate the optimal number of clusters to be one. **This is the scenario where most other
    competing methods fail.**" Silhouette, Calinski-Harabasz and Davies-Bouldin are undefined at
    k=1; the elbow has no kink to find; prediction strength compares two partitions and needs at
    least two groups to compare.

    The search starts at k=2 because everything downstream — profiles, charts, the typing rule —
    needs at least two groups to describe. That is a reasonable engineering choice and it quietly
    discarded the one test that detects "there is nothing here". This puts it back: k=1 is scored
    against k=2 on its own, and the answer is reported rather than used to change k.

    Measured on this machine: pure noise gives gap(1)=0.862 against gap(2)=0.850 and selects k=1;
    data with three real segments gives gap(1)=-0.357 and selects k=3.

    Returns (single_cluster, gap_one, gap_two) — or (False, nan, nan) if it could not be computed,
    because a diagnostic that fails should never take the analysis down with it.
    """
    try:
        rows = gap_statistic(X, range(1, 3), B, rng, n_init, cfg)
        g = {int(r["k"]): (float(r["gap"]), float(r["gap_se"])) for _, r in rows.iterrows()}
        # Tibshirani's own rule, applied at the first step: choose k=1 unless k=2 is better by
        # more than the standard error of the reference distribution.
        return (g[1][0] >= g[2][0] - g[2][1], g[1][0], g[2][0])
    except Exception as e:
        print(f"NOTE: could not test for a single cluster ({type(e).__name__}: {e}).")
        return False, float("nan"), float("nan")


def replication_stability(X, k_range, B, frac, rng, cfg):
    """Global stability (Dolnicar & Leisch): fit a reference clustering, then cluster B
    subsamples and measure the Adjusted Rand Index between reference and subsample labels on
    the shared points. Mean ARI near 1 => the same segments re-emerge; low => constructed.
    Uses the chosen model (k-means or Gaussian mixture) throughout."""
    rows = []
    for k in k_range:
        ref = _fit(X, k, cfg, cfg.n_init_search, rng.integers(1e9)).labels_
        aris = np.empty(B)
        n_sub = max(k + 1, int(frac * len(X)))
        for b in range(B):
            idx = rng.choice(len(X), n_sub, replace=False)
            aris[b] = adjusted_rand_score(ref[idx], _fit(X[idx], k, cfg, 1, rng.integers(1e9)).labels_)
        rows.append({"k": k, "stability_ARI": aris.mean(), "stability_ARI_sd": aris.std()})
    return pd.DataFrame(rows)


def prediction_strength(X, k_range, n_splits, rng, cfg, min_k_for_singleton=1.0):
    """Tibshirani & Walther (2005). Split the data in two halves, cluster each with the chosen
    model; classify each half's points by the OTHER half's fitted model. For each test cluster,
    the fraction of its within-cluster point pairs that are also co-assigned by the training
    model; prediction strength = the MINIMUM of that fraction over test clusters. Average over
    both directions and all splits. Pick the largest k with prediction strength >= cutoff."""
    def ps_one(test_labels, train_pred):
        strengths = []
        for c in np.unique(test_labels):
            members = np.where(test_labels == c)[0]
            m = len(members)
            if m <= 1:
                strengths.append(float(min_k_for_singleton)); continue
            counts = np.bincount(train_pred[members], minlength=int(train_pred.max()) + 1)
            same = np.sum(counts * (counts - 1))      # co-assigned ordered pairs
            strengths.append(same / (m * (m - 1)))
        return min(strengths) if strengths else 0.0

    rows = []
    for k in k_range:
        vals = []
        for _ in range(n_splits):
            idx = rng.permutation(len(X)); h = len(X) // 2
            A, Bh = X[idx[:h]], X[idx[h:]]
            mA = _fit(A, k, cfg, cfg.n_init_search, rng.integers(1e9))
            mB = _fit(Bh, k, cfg, cfg.n_init_search, rng.integers(1e9))
            vals.append(ps_one(mA.labels_, mB.predict(A)))
            vals.append(ps_one(mB.labels_, mA.predict(Bh)))
        rows.append({"k": k, "prediction_strength": float(np.mean(vals)),
                     "prediction_strength_sd": float(np.std(vals))})
    return pd.DataFrame(rows)


def gmm_bic_icl(X, k_range, rng, covariance="full"):
    """Model-based number-of-segments criteria. Fit a Gaussian mixture at each k and record two
    criteria (lower is better for both): the Bayesian Information Criterion (Wedel & Kamakura —
    the principled criterion heuristic k-means lacks) and the Integrated Completed Likelihood
    (Biernacki, Celeux & Govaert 2000), which adds an entropy penalty for overlapping components
    and so favours cleaner, better-separated segments. ICL = BIC + 2 x classification entropy."""
    rows = []
    for k in k_range:
        try:
            g = GaussianMixture(n_components=k, covariance_type=covariance, n_init=3,
                                random_state=int(rng.integers(1e9)), reg_covar=1e-4).fit(X)
            bic = g.bic(X)
            post = g.predict_proba(X)
            entropy = -(post * np.log(post + 1e-12)).sum()      # total classification entropy
            rows.append({"k": k, "gmm_BIC": bic, "gmm_ICL": bic + 2 * entropy})
        except Exception:
            rows.append({"k": k, "gmm_BIC": np.nan, "gmm_ICL": np.nan})
    return pd.DataFrame(rows)


def consensus_matrix(X, k, cfg, rng):
    """Monti et al. (2003) consensus matrix. Resample the rows H times, cluster each subsample
    with the chosen model, and record for every pair the fraction of resamples (in which both
    were sampled) that placed them in the same cluster. A perfectly stable k gives a matrix of
    only 0s and 1s; instability shows up as intermediate values."""
    # Two dense n-by-n matrices live here, so this is the routine that decides how large a study
    # the tool survives. Unguarded it asked for 27 GB on a 41,188-respondent file. Above
    # MAX_PAIRWISE_N the consensus is built for a random subsample of that many respondents: PAC
    # is a summary of how ambiguously pairs co-cluster, and a random sample of 6,000 people
    # estimates it perfectly well. The caller records that it was sampled.
    take = _pairwise_sample(len(X), cfg)
    if take is not None:
        X = X[take]
    n = len(X)
    M = np.zeros((n, n)); I = np.zeros((n, n))
    n_sub = max(k + 1, int(cfg.consensus_frac * n))
    for _ in range(cfg.consensus_H):
        idx = rng.choice(n, n_sub, replace=False)
        lab = _fit(X[idx], k, cfg, cfg.n_init_search, rng.integers(1e9)).labels_
        I[np.ix_(idx, idx)] += 1
        for c in np.unique(lab):
            mem = idx[lab == c]
            M[np.ix_(mem, mem)] += 1
    C = np.divide(M, I, out=np.zeros_like(M), where=I > 0)
    np.fill_diagonal(C, 1.0)
    return C


def pac_score(C, lower=0.1, upper=0.9):
    """Proportion of Ambiguous Clustering (Senbabaoglu et al. 2014): the share of off-diagonal
    consensus entries strictly between `lower` and `upper` (default 0.1-0.9). Lower is better;
    the k with the smallest PAC is the cleanest, least ambiguous solution. PAC is more reliable
    than Monti's original delta-area-under-CDF heuristic."""
    vals = C[np.triu_indices_from(C, k=1)]
    return float(np.mean((vals > lower) & (vals < upper))) if len(vals) else 1.0


def consensus_pac(X, k_range, cfg, rng):
    rows = []
    for k in k_range:
        rows.append({"k": k, "consensus_PAC": pac_score(consensus_matrix(X, k, cfg, rng))})
    return pd.DataFrame(rows)


def consensus_partition(X, k, cfg, rng):
    """Build the consensus matrix at k and derive a robust ENSEMBLE partition from it by average-
    linkage hierarchical clustering on the consensus distance (1 - consensus). This partition is
    far less sensitive to initialization than a single k-means run (Monti et al. 2003).

    Unlike the PAC score, this one cannot be taken from a subsample: it has to place every
    respondent, and the linkage that derives it is O(n^2) besides. Above MAX_PAIRWISE_N it returns
    None, and the caller keeps the ordinary partition and says why — adopting an ensemble that had
    only seen a fraction of the study would be exactly the silent substitution this tool exists to
    avoid."""
    if len(X) > MAX_PAIRWISE_N:
        return None, None
    C = consensus_matrix(X, k, cfg, rng)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0                                  # enforce exact symmetry for squareform
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, k, criterion="maxclust") - 1
    return labels, C


def hopkins_statistic(X, rng, m_frac=0.10, cfg=None):
    """Cluster-TENDENCY pre-check (Lawson & Jurs 1990; Banerjee & Dave 2004): should you even
    cluster these data? Compare nearest-neighbour distances of real points to those of uniform
    random points over the data's bounding box. H = sum(u) / (sum(u) + sum(w)), where u are
    random-point-to-data distances and w are data-point-to-data distances. Reading: H ~ 0.5 =
    random (no cluster tendency); H > 0.75 = strong tendency to cluster; H < 0.5 = regularly
    spaced. Sampling a small fraction (default 10%) keeps the test valid.

    On the mixed-type path the same test runs under Gower's distance, and the uniform bounding box
    is replaced by a column-by-column reference: a bounding box would otherwise place the null
    points on brand codes nobody could have chosen (see kprototypes.reference_sample)."""
    n, d = X.shape
    m = max(5, int(m_frac * n))
    if _is_gower(cfg):
        spec = cfg.gower_spec
        U = kprototypes.gower_embedding(
            kprototypes.reference_sample(X, spec, m, rng), spec)
        X = kprototypes.gower_embedding(np.asarray(X, float), spec)
        metric = "manhattan"
    else:
        U = rng.uniform(X.min(0), X.max(0), size=(m, d))
        metric = "euclidean"
    nbrs = NearestNeighbors(n_neighbors=2, metric=metric).fit(X)
    w = nbrs.kneighbors(X[rng.choice(n, m, replace=False)], n_neighbors=2)[0][:, 1]  # skip self
    u = nbrs.kneighbors(U, n_neighbors=1)[0][:, 0]
    su, sw = u.sum(), w.sum()
    return float(su / (su + sw)) if (su + sw) > 0 else 0.5


def hopkins_reading(h):
    if h >= 0.75: return "strong tendency to cluster"
    if h >= 0.60: return "some tendency to cluster"
    if h >= 0.45: return "essentially random — segments will be constructed, not natural"
    return "regularly spaced — clustering is not appropriate"


def model_based_agreement(X, k, base_labels, cfg, rng):
    """Cross-paradigm agreement check (Wedel & Kamakura). Compares the chosen partition against
    the OTHER method's partition via the Adjusted Rand Index: when the final model is k-means, the
    comparison is against a Gaussian mixture; when it is a Gaussian mixture, against k-means. High
    agreement is extra evidence the structure is real (two different paradigms see the same
    segments); low agreement warns the partition is method-dependent. Always also returns the
    mixture's assignment confidence (mean top posterior) and normalized entropy (0 = crisp, 1 = fuzzy).

    Not available on the mixed-type path: a Gaussian mixture assumes every variable is a
    continuous measurement, and fitting one to brand codes would produce a number that looks like
    corroboration while meaning nothing. That path keeps the hierarchical cross-check, which is
    well defined under Gower, and the report says which one is missing rather than quietly
    reporting fewer checks."""
    if _is_gower(cfg):
        return {"agreement_ARI": np.nan, "other_method": "n/a", "covariance": "n/a",
                "mean_max_posterior": np.nan, "normalized_entropy": np.nan}
    best = None
    for cov in ("full", "tied", "diag", "spherical"):
        try:
            g = GaussianMixture(n_components=k, covariance_type=cov, n_init=3,
                                random_state=int(rng.integers(1e9)), reg_covar=1e-4).fit(X)
            b = g.bic(X)
            if best is None or b < best[0]:
                best = (b, g, cov)
        except Exception:
            continue
    if best is None:
        return {"agreement_ARI": np.nan, "other_method": "n/a", "covariance": "n/a",
                "mean_max_posterior": np.nan, "normalized_entropy": np.nan}
    _, g, cov = best
    post = g.predict_proba(X)
    ent = max(0.0, -(post * np.log(post + 1e-12)).sum(1).mean() / np.log(k))  # 0 = certain, 1 = uniform
    if getattr(cfg, "method", "kmeans") == "gmm":
        other = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e9))).fit(X).labels_
        other_name = "k-means"
    else:
        other = g.predict(X)
        other_name = "a Gaussian mixture"
    return {"agreement_ARI": float(adjusted_rand_score(base_labels, other)),
            "other_method": other_name, "covariance": cov,
            "mean_max_posterior": float(post.max(1).mean()), "normalized_entropy": float(ent)}


def ward_agreement(X, labels, k, cfg=None):
    """Third cross-check: does Ward agglomerative clustering (structurally different from k-means
    and the mixture: it merges bottom-up, not around centroids) recover the same partition? Three
    different methods agreeing is strong evidence the structure is real, not an artefact of one
    algorithm. Skipped for very large n (the linkage is O(n^2)).

    Ward's criterion is a sum of squares, so it is only defined in a Euclidean space. On the
    mixed-type path the bottom-up cross-check therefore uses average linkage on Gower's distance
    instead — still hierarchical, still a genuinely different paradigm from prototypes, and the
    standard partner to Gower in the literature."""
    if len(X) > 3000:
        return float("nan")
    try:
        if _is_gower(cfg):
            Z = kprototypes.gower_embedding(np.asarray(X, float), cfg.gower_spec)
            other = fcluster(linkage(Z, method="average", metric="cityblock"),
                             t=k, criterion="maxclust")
        else:
            other = fcluster(linkage(X, method="ward"), t=k, criterion="maxclust")
        return float(adjusted_rand_score(labels, other))
    except Exception:
        return float("nan")


def _cramers_v(codes, labels):
    """Association between a pick-any answer and segment membership, on a 0-1 scale.

    The counterpart to eta-squared for a variable that has no arithmetic. Eta-squared on brand
    codes would be a real number computed from nothing: renumbering the brands changes it, so it
    measures the coding rather than the data. Cramér's V is invariant to how the levels are
    numbered, which is the only honest property to want here. Bias-corrected (Bergsma 2013), since
    a plain V drifts upward with the number of levels and a brand question can have many.
    """
    table = pd.crosstab(codes, labels).to_numpy(float)
    n = table.sum()
    if n <= 0 or min(table.shape) < 2:
        return 0.0
    expected = np.outer(table.sum(1), table.sum(0)) / n
    chi2 = float((((table - expected) ** 2) / np.where(expected > 0, expected, 1)).sum())
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    k_c = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_c - 1, k_c - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def variable_importance(X_raw, labels, cfg=None):
    """Which items actually drive the segmentation, and which are noise? Reports eta-squared per
    item (between-segment sum of squares / total sum of squares) — the share of an item's
    variance explained by segment membership. Items with near-zero eta-squared add noise and,
    per Dolnicar's variable-selection work, can mask real structure; consider dropping them and
    re-running.

    A pick-any question is scored by Cramér's V instead, because eta-squared on its codes would
    depend on the arbitrary order they were assigned. Both land on 0-1 with the same reading, so
    they share a column and the bands below apply to both."""
    items = list(X_raw.columns)
    kinds = dict(getattr(cfg, "var_kinds", None) or {})
    rows = []
    for it in items:
        y = X_raw[it].to_numpy(float)
        if kinds.get(it) == kprototypes.NOMINAL:
            # SQUARED, and the squaring is the whole point. Eta-squared is a share of variance;
            # Cramer's V is correlation-like, so the two are one square apart and the bands below
            # do not transfer between them. Measured on matched pure noise: a random pick-any
            # column scores V = 0.06, which already clears the 0.05 "near-noise" floor, so a
            # useless question could never be flagged as one. V-squared reads 0.00 on the same
            # data, against eta-squared's 0.00 for a random rating — the same scale at last.
            rows.append({"item": it, "eta_squared": round(_cramers_v(y, labels) ** 2, 3),
                         "measure": "Cramer's V^2"})
            continue
        grand = y.mean()
        ss_tot = ((y - grand) ** 2).sum()
        ss_between = sum(len(y[labels == c]) * (y[labels == c].mean() - grand) ** 2
                         for c in np.unique(labels))
        eta2 = ss_between / ss_tot if ss_tot > 0 else 0.0
        rows.append({"item": it, "eta_squared": round(eta2, 3), "measure": "eta-squared"})
    df = pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)
    df["role"] = np.where(df["eta_squared"] >= 0.15, "drives segmentation",
                 np.where(df["eta_squared"] >= 0.05, "contributes", "near-noise (consider dropping)"))
    return df


def variable_selection_check(X_raw, labels, cfg, k, full_metrics):
    """Dolnicar's variable-selection point, made operational. Near-noise items (low eta-squared)
    can MASK real structure. Re-cluster on the SIGNAL items only, at the same k, and report whether
    the solution gets cleaner (more stable, better separated). This is a diagnostic COMPARISON, not
    a silent drop: dropping variables can also manufacture spurious structure, so the analyst
    decides. The 'all items' side reuses the shipped solution's own metrics, so only the reduced
    solution is recomputed."""
    vi = variable_importance(X_raw, labels, cfg)
    noise = vi.loc[vi["role"].str.startswith("near-noise"), "item"].tolist()
    signal = [c for c in X_raw.columns if c not in noise]
    if not noise or len(signal) < 2:
        return {"applicable": False, "dropped": noise, "n_signal": len(signal)}
    if _is_gower(cfg):
        # The spec describes the columns it was fitted on, so a subset needs its own — and the
        # config handed to every downstream call has to carry that one, or the distance would be
        # computed against variables that are no longer there. This check matters most on this
        # path: measured, three noise pick-any columns alongside six real ratings cost 0.25 ARI,
        # which is exactly the harm it exists to surface.
        arr = X_raw[signal].to_numpy(float)
        spec = kprototypes.fit_spec(arr, [cfg.var_kinds.get(c, kprototypes.ORDINAL)
                                          for c in signal])
        cfg = replace(cfg, gower_spec=spec,
                      var_kinds={c: v for c, v in (cfg.var_kinds or {}).items() if c in signal})
        Xs = kprototypes.encode(arr, spec)
    else:
        Xs, _ = _scale_fit(X_raw[signal].to_numpy(float), cfg.scaling)
    lab = fit_final(Xs, k, cfg)[0].labels_
    jac = list(clusterboot_jaccard(Xs, lab, k, cfg).values())
    reduced = {"split_half": split_half_replication(Xs, k, cfg),
               "mean_jaccard": float(np.mean(jac)), "min_jaccard": float(np.min(jac)),
               "silhouette": _silhouette(Xs, lab, cfg)}
    cleaner = (reduced["min_jaccard"] >= full_metrics["min_jaccard"] - 1e-9 and
               reduced["split_half"] >= full_metrics["split_half"] - 1e-9)
    return {"applicable": True, "dropped": noise, "n_signal": len(signal),
            "full": full_metrics, "reduced": reduced, "reduced_is_cleaner": bool(cleaner)}


def _fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up: returns a dict name -> significant(bool) at FDR alpha."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    max_i = 0
    for i, (_, p) in enumerate(items, 1):
        if p <= (i / m) * alpha:
            max_i = i
    return {name: (i <= max_i) for i, (name, _) in enumerate(items, 1)}


def auto_elbow(ks, inertia):
    if len(ks) < 2:                                 # a single candidate k: it is the elbow trivially
        return int(ks[0])
    x = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min() + 1e-12)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0) + 1e-12
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return int(ks[int(np.argmax(dist))])


def selection_diagnostics(X, cfg):
    k_range = range(cfg.k_min, cfg.k_max + 1)
    base = []
    for k in k_range:
        lab = _fit(X, k, cfg, cfg.n_init_search, cfg.random_state).labels_
        base.append({"k": k, "inertia": _pooled_within_ss(X, lab, cfg),
                     **_internal_indices(X, lab, cfg),
                     # Share of respondents in the SMALLEST segment. A statistically tidy
                     # solution made of two-person segments cannot be marketed to, so this is
                     # what makes cfg.min_segment_frac enforceable rather than decorative.
                     "min_segment_share": float(np.bincount(lab, minlength=k).min() / len(lab))})
    diag = pd.DataFrame(base)
    diag = diag.merge(gap_statistic(X, k_range, cfg.gap_B,
                                    np.random.default_rng(cfg.random_state + 1),
                                    cfg.n_init_search, cfg), on="k")
    diag = diag.merge(replication_stability(X, k_range, cfg.stability_B, cfg.stability_frac,
                                            np.random.default_rng(cfg.random_state + 2), cfg), on="k")
    diag = diag.merge(prediction_strength(X, k_range, cfg.ps_splits,
                                          np.random.default_rng(cfg.random_state + 3), cfg), on="k")
    # Not on the mixed-type path. A Gaussian mixture assumes every column is a measurement, so
    # its BIC on codes standing for brands is a real number computed from a false premise — and
    # left in, it votes: on a mixed file whose true answer was 3 it argued for 8. The report says
    # this check is unavailable here, and it has to actually be unavailable.
    if (cfg.fit_gmm_bic or cfg.method == "gmm") and not _is_gower(cfg):
        diag = diag.merge(gmm_bic_icl(X, k_range, np.random.default_rng(cfg.random_state + 4),
                                      cfg.gmm_covariance), on="k")
    if cfg.run_consensus:
        diag = diag.merge(consensus_pac(X, k_range, cfg,
                                        np.random.default_rng(cfg.random_state + 9)), on="k")
    return diag


def _gap_choice(diag):
    d = diag.sort_values("k").reset_index(drop=True)
    for i in range(len(d) - 1):
        if d.loc[i, "gap"] >= d.loc[i + 1, "gap"] - d.loc[i + 1, "gap_se"]:
            return int(d.loc[i, "k"])
    return int(d.loc[d["gap"].idxmax(), "k"])


def signal_weights(cfg):
    """How much each criterion counts. Kept in one place because the report explains this vote in
    prose, and a second hand-kept copy of the numbers is how the explanation and the decision
    drift apart — the report once said "5 of them picked 2" about a tally that had not decided
    anything. When the chosen model IS the Gaussian mixture, the model-based criteria (BIC, ICL)
    are the primary basis for the number of components, so they count double too."""
    mb_w = 2 if getattr(cfg, "method", "kmeans") == "gmm" else 1
    return {"prediction strength": 2, "global stability": 2, "consensus PAC": 2,
            "silhouette": 1, "Calinski-Harabasz": 1, "Davies-Bouldin": 1, "gap": 1,
            "GMM BIC (model-based)": mb_w, "GMM ICL (model-based)": mb_w, "elbow (weak)": 0}


def recommend_k(diag, cfg):
    """Consensus that weights STABILITY and prediction strength above internal fit indices,
    because a stable, replicable solution is what survives — the market-segmentation view."""
    # Rule out solutions whose smallest segment is too small to act on BEFORE the vote, rather
    # than noting it afterwards. A segmentation exists to be targeted: on 120 respondents the
    # criteria will happily crown k=55 (segments of two people) on separation grounds alone, and
    # that is a confidently-presented answer nobody can use. cfg.min_segment_frac already
    # expressed the floor; until now it only printed a footnote under the finished report.
    # Keep at least two candidates so the search still has something to choose between, and if
    # the floor would rule out everything, defer to the criteria rather than inventing an answer.
    # This filters the SEARCH fit; the final fit uses more restarts and may settle on a slightly
    # different optimum, so treat it as removing unusable answers rather than as a hard bound.
    # A segment that still lands under the floor is caught by the note under the sizes table.
    excluded = []
    if "min_segment_share" in diag:
        viable = diag[diag["min_segment_share"] >= cfg.min_segment_frac]
        if len(viable) >= 2:
            excluded = sorted(set(diag["k"]) - set(viable["k"]))
            diag = viable
    ks = diag["k"].to_numpy()
    signals = {}
    signals["elbow (weak)"] = auto_elbow(ks, diag["inertia"].to_numpy())
    signals["gap"] = _gap_choice(diag)
    signals["silhouette"] = int(diag.loc[diag["silhouette"].idxmax(), "k"])
    signals["Calinski-Harabasz"] = int(diag.loc[diag["calinski_harabasz"].idxmax(), "k"])
    signals["Davies-Bouldin"] = int(diag.loc[diag["davies_bouldin"].idxmin(), "k"])
    # Prediction strength: largest k with PS >= cutoff (Tibshirani-Walther); else best PS
    ps_ok = diag[diag["prediction_strength"] >= cfg.ps_cutoff]
    signals["prediction strength"] = int(ps_ok["k"].max()) if len(ps_ok) \
        else int(diag.loc[diag["prediction_strength"].idxmax(), "k"])
    # Global stability. This used to read "the largest k that still clears the cutoff", borrowed
    # from the prediction-strength rule below — where it is Tibshirani & Walther's published
    # advice and correct. Applied to stability it is not: measured on a file with three planted
    # segments, stability ran 0.995 at k=3 and 0.778 at k=4, and "largest above 0.75" handed this
    # signal — one of the two the whole method leans on — to the visibly worse answer, which was
    # enough to tie the vote and lose the segmentation.
    #
    # A one-standard-error rule instead (after Hastie, Tibshirani & Friedman, ESL 7.10): keep the
    # k values whose stability cannot be told apart from the best — within one standard error of
    # it — and take the largest of those. That keeps the original intent, the largest k the
    # evidence still supports, while refusing the jump to a k that is measurably worse. The
    # standard error comes from the resampling that produced the estimate, so a noisy measurement
    # widens the band by itself rather than by a constant somebody guessed.
    #
    # Taking the SMALLEST of that band instead looks equally principled and is not: the vote
    # already breaks ties toward the smaller solution, so preferring it here too counts parsimony
    # twice, and measured on an unequal 80/15/5 split that lost the 5% segment entirely.
    # Each k is compared against the best using ITS OWN standard error, rather than a single band
    # drawn around the best. Two reasons, both measured. A band around the best collapses to
    # nothing whenever the best k scores a standard error of exactly zero — every resample agreed,
    # which happens readily at k=2 — and then no other k can ever qualify. And the width that
    # matters is the uncertainty of the k being judged: a solution whose stability is itself
    # measured loosely has not been shown to be worse, while one measured tightly has.
    #
    # On the three files this was calibrated against it needs no tuning constant. k=4 at
    # 0.778 +/- 0.075 against a best of 0.995 is 0.217 away, far outside its own error, and drops
    # out. k=3 at 0.940 +/- 0.135 against 1.000, and k=3 at 0.897 +/- 0.200 against 1.000, are
    # both well inside theirs and stay in — which is right, since each is the same structure split
    # one level finer.
    _stab = diag["stability_ARI"]
    _best_i = _stab.idxmax()
    _best = float(_stab.max())
    _sd = diag["stability_ARI_sd"] if "stability_ARI_sd" in diag else _stab * 0.0
    _sd = _sd.fillna(0.0)
    _within = diag[(_stab >= cfg.stability_cutoff) & ((_best - _stab) <= _sd)]
    signals["global stability"] = int(_within["k"].max()) if len(_within) \
        else int(diag.loc[_best_i, "k"])
    if "gmm_BIC" in diag and diag["gmm_BIC"].notna().any():
        signals["GMM BIC (model-based)"] = int(diag.loc[diag["gmm_BIC"].idxmin(), "k"])
    if "gmm_ICL" in diag and diag["gmm_ICL"].notna().any():
        signals["GMM ICL (model-based)"] = int(diag.loc[diag["gmm_ICL"].idxmin(), "k"])
    if "consensus_PAC" in diag and diag["consensus_PAC"].notna().any():
        signals["consensus PAC"] = int(diag.loc[diag["consensus_PAC"].idxmin(), "k"])

    # Weighted vote: stability signals and prediction strength count double. When the chosen
    # model IS the Gaussian mixture, the model-based criteria (BIC, ICL) are the primary basis
    # for the number of components, so they count double too.
    weights = signal_weights(cfg)
    tally = {}
    for name, k in signals.items():
        tally[k] = tally.get(k, 0) + weights.get(name, 1)
    best_score = max(tally.values())
    winners = sorted([k for k, s in tally.items() if s == best_score])
    # Ties used to go to the smaller k unconditionally, on parsimony grounds. That silently
    # outranked the priority this function is built around: measured on three planted segments,
    # k=2 and k=3 tied at 5, and k=3 held BOTH doubled criteria (prediction strength 0.97 against
    # 0.59, consensus PAC 0.003 against 0.353) while k=2 held only the separation indices. The
    # parsimony rule took k=2 — a solution whose prediction strength does not even clear the 0.80
    # cutoff this report quotes — and the segmentation was then written up as constructed noise.
    # So on a tie, the heavily-weighted criteria break it, and parsimony breaks what remains.
    if len(winners) > 1:
        heavy = {n for n, w in weights.items() if w >= 2}
        backing = {k: sum(1 for n, s in signals.items() if s == k and n in heavy) for k in winners}
        most = max(backing.values())
        winners = [k for k in winners if backing[k] == most]
    pick = winners[0]   # still smaller, more interpretable k (Dolnicar: parsimony)

    ruled_out = ""
    if excluded:
        ruled_out = ("\n\nRuled out before the vote: k = "
                     + ", ".join(str(k) for k in excluded)
                     + f" — each of those splits the sample into at least one segment holding "
                       f"under {cfg.min_segment_frac:.0%} of respondents, which is too small to "
                       "target even if the statistics look clean.")
    # When the winner fails the one cutoff this report quotes as decisive, say so. Measured on a
    # file whose answers round onto a few tight patterns: k=6 won the weighted vote on the
    # separation indices with prediction strength 0.740 — under the 0.80 the text below calls the
    # column to trust most — while k=2 scored a perfect 1.000. Both readings are defensible there,
    # which is exactly why the reader should be told they disagree rather than shown a single
    # number. This changes nothing about the answer, only about what is claimed for it.
    _ps_note = ""
    try:
        _ps_pick = float(diag.loc[diag["k"] == pick, "prediction_strength"].iloc[0])
        if _ps_pick < cfg.ps_cutoff:
            _clears = diag[diag["prediction_strength"] >= cfg.ps_cutoff]
            _best = (f"k = {int(_clears.loc[_clears['prediction_strength'].idxmax(), 'k'])} reaches "
                     f"{_clears['prediction_strength'].max():.2f}" if len(_clears)
                     else "no number of segments reaches it")
            _ps_note = (f"\n\n> **Note.** Prediction strength at k = {pick} is "
                        f"{_ps_pick:.2f}, below the {cfg.ps_cutoff:.2f} that Tibshirani & Walther "
                        f"suggest, and it is the column this report calls the most important — "
                        f"{_best}. The separation and stability criteria outvoted it here. Read "
                        "the table before committing: this is a case where the evidence genuinely "
                        "points two ways.")
    except Exception:
        _ps_note = ""
    rationale = ("Recommended number of segments: **{}**.\n\nWhat each criterion points to: "
                 .format(pick)
                 + "; ".join(f"{n} -> {k}" for n, k in signals.items())
                 + "." + ruled_out
                 + "\n\nThe recommendation weights prediction strength and replication "
                 "stability most heavily (a segmentation is only useful if it reproduces), "
                 "then the separation indices, then the gap statistic, and treats the inertia "
                 "elbow as the weakest signal. On a tie it prefers the smaller, more "
                 "interpretable solution. Read the whole table before committing; if the "
                 "signals disagree sharply, that itself is evidence the data may not contain "
                 "natural segments, and the right move may be a smaller k or a different method."
                 + _ps_note)
    return pick, rationale, signals


# =====================================================================================
# Final fit, validation
# =====================================================================================
def fit_final(X, k, cfg):
    """Fit with many restarts and report how often the best solution was reached (a local-optima
    diagnostic in the spirit of Steinley 2003)."""
    model = _fit(X, k, cfg, cfg.n_init_final, cfg.random_state)
    inertias = []
    for s in range(20):
        inertias.append(_fit(X, k, cfg, 1, cfg.random_state + 100 + s).inertia_)
    inertias = np.array(inertias)
    reached = float(np.mean(np.isclose(inertias, inertias.min(), rtol=1e-3)))
    return model, reached


def split_half_replication(X, k, cfg):
    rng = np.random.default_rng(cfg.random_state + 5)
    idx = rng.permutation(len(X)); h = len(X) // 2
    a, b = idx[:h], idx[h:]
    ma = _fit(X[a], k, cfg, cfg.n_init_search, 1)
    mb = _fit(X[b], k, cfg, cfg.n_init_search, 2)
    return adjusted_rand_score(ma.predict(X[b]), mb.predict(X[b]))


def clusterboot_jaccard(X, base_labels, k, cfg):
    """Hennig (2007) fpc::clusterboot. For each of B bootstrap resamples (sample n with
    replacement), re-cluster and, for each ORIGINAL cluster, record the Jaccard similarity to
    the most similar bootstrap cluster (computed on the original points present in the sample).
    Per-cluster stability = mean Jaccard over resamples. Reading: >=0.85 highly stable,
    >=0.75 valid/stable, 0.6-0.75 a pattern with doubtful membership, <0.5 dissolved."""
    rng = np.random.default_rng(cfg.random_state + 6)
    n = len(X)
    per_cluster = {int(c): [] for c in np.unique(base_labels)}
    base_sets = {int(c): set(np.where(base_labels == c)[0]) for c in np.unique(base_labels)}
    for _ in range(cfg.jaccard_B):
        idx = rng.choice(n, n, replace=True)
        present = np.unique(idx)
        m = _fit(X[idx], k, cfg, 5, rng.integers(1e9))
        boot_of_present = m.predict(X[present])           # bootstrap cluster of each present original point
        boot_sets = [set(present[boot_of_present == d]) for d in range(k)]
        for c, C in base_sets.items():
            Cp = C & set(present.tolist())
            best = 0.0
            for D in boot_sets:
                u = len(Cp | D)
                if u:
                    best = max(best, len(Cp & D) / u)
            per_cluster[c].append(best)
    return {c: float(np.mean(v)) for c, v in per_cluster.items()}


def jaccard_reading(j):
    if j >= 0.85: return "highly stable"
    if j >= 0.75: return "stable / valid"
    if j >= 0.60: return "a pattern (membership doubtful)"
    if j >= 0.50: return "weak"
    return "DISSOLVED (not a real cluster)"


# =====================================================================================
# Interpretation
# =====================================================================================
def _picks_line(codes, labels, c, level_labels):
    """What this segment answers to a pick-any question, against what everybody answers.

    "Values 0.6 below average" is not a statement about a brand — it is arithmetic on the order
    the brands happened to be listed in. The honest version of the same sentence names the answer
    the segment actually gives and how much more often than the sample as a whole it gives it."""
    mine = codes[labels == c]
    if not len(mine):
        return None
    values, counts = np.unique(mine, return_counts=True)
    top = values[counts.argmax()]
    share = counts.max() / len(mine)
    overall = float((codes == top).mean())
    name = (level_labels or {}).get(float(top), str(top))
    return f"{name} ({share:.0%} of them, vs {overall:.0%} overall)"


def interpret(X_raw, labels, cfg):
    items = list(X_raw.columns)
    kinds = dict(getattr(cfg, "var_kinds", None) or {})
    all_labels = getattr(cfg, "level_labels", None) or {}
    nominal = [it for it in items if kinds.get(it) == kprototypes.NOMINAL]
    rated = [it for it in items if it not in nominal]
    seg = pd.Series(labels, name="segment")
    centroids = X_raw.groupby(seg).mean()
    centroids.index = [f"Segment {c}" for c in centroids.index]
    # Name the index, or the exported profiles open in Excel with a blank first heading that
    # pandas reads back as "Unnamed: 0" — a column of segment labels with nothing saying so.
    centroids.index.name = "segment"
    # A pick-any column has no meaningful mean, so the profile table carries the segment's most
    # common answer by name instead of the average of its codes.
    for it in nominal:
        codes = X_raw[it].to_numpy(float)
        centroids[it] = [(_picks_line(codes, labels, c, all_labels.get(it)) or "").split(" (")[0]
                         for c in np.unique(labels)]
    grand = X_raw[rated].mean()
    defining = {}
    for c in np.unique(labels):
        diff = (X_raw[labels == c][rated].mean() - grand).sort_values(ascending=False)
        # With few items, head(top_items) and tail(top_items) would overlap and print the same
        # items as both "values most" and "values least"; cap each side to a non-overlapping half.
        half = max(1, min(cfg.top_items, len(diff) // 2))
        picks = [f"{it}: {line}" for it in nominal
                 if (line := _picks_line(X_raw[it].to_numpy(float), labels, c,
                                         all_labels.get(it)))]
        defining[f"Segment {c}"] = {
            "most_above_average": [f"{it} ({diff[it]:+.1f})" for it in diff.head(half).index],
            "most_below_average": [f"{it} ({diff[it]:+.1f})" for it in diff.tail(half).index],
            "picks": picks,
            "auto_name": " + ".join(_short_label(t) for t in diff.head(2).index)}
    groups = [X_raw[labels == c] for c in np.unique(labels)]
    fvals = {}
    for it in items:
        try:
            if it in nominal:
                # An F test compares means, which a pick-any question does not have. Chi-square on
                # the answer-by-segment table asks the same question — does this split people? —
                # without pretending the codes are quantities.
                chi2, p, _, _ = stats.chi2_contingency(
                    pd.crosstab(X_raw[it], pd.Series(labels)).to_numpy())
                fvals[it] = (float(chi2), float(p))
                continue
            f, p = stats.f_oneway(*[g[it].to_numpy() for g in groups]); fvals[it] = (f, p)
        except Exception:
            fvals[it] = (np.nan, np.nan)
    differentiating = (pd.DataFrame({"item": list(fvals), "F": [v[0] for v in fvals.values()],
                                     "p": [v[1] for v in fvals.values()]})
                       .sort_values("F", ascending=False).reset_index(drop=True))
    sizes = (pd.Series(labels).value_counts().sort_index()
             .rename(lambda c: f"Segment {c}").to_frame("n"))
    sizes["share"] = (sizes["n"] / sizes["n"].sum()).round(3)
    return centroids, defining, differentiating, sizes


# =====================================================================================
# Report
# =====================================================================================
def _md(df, index=False):
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


_LABEL_STOP = {"i", "to", "the", "a", "an", "my", "in", "of", "you", "for", "it", "and", "is",
               "are", "on", "me", "we", "do", "does", "how", "what", "this", "that", "with",
               "your", "am", "be", "or", "at", "as", "would", "like"}


_TRAILING_FILLER = {"to", "for", "of", "in", "on", "at", "a", "an", "the", "and", "or",
                    "rather", "than", "with", "by", "from", "that", "as", "is", "are", "my",
                    "own", "it", "about", "would", "will", "can"}


def _short_label(text, max_chars=30):
    """Shorten a question into a label a person can recognise, keeping it English.

    This used to drop every stopword first, which reads as a telegram once two of them are joined
    with a "+": "I like planning things rather than deciding last minute" came out as "planning
    things rather" — a fragment ending on a dangling conjunction — and the segment it named
    appeared throughout a report as "planning things rather + want meet people outside". These
    names are placeholders the team is told to replace, so their one job is to be recognisable as
    the question they came from, which the original words do and a stopword-stripped stub does not.

    A leading "I " is dropped because every question in a battery has one, and the label is cut at
    a word boundary with an ellipsis so a shortened phrase looks shortened rather than finished.
    """
    t = " ".join(str(text).replace("_", " ").split()).strip().rstrip("?").strip()
    if t[:2].lower() == "i " and len(t) > 2:
        t = t[2:].lstrip()
    if len(t) <= max_chars:
        return t
    out = ""
    for w in t.split():
        if out and len(out) + len(w) + 1 > max_chars:
            break
        out = (out + " " + w).strip()
    # Do not end a shortened label on a word that was leading somewhere: "...feel fake to" and
    # "...planning things rather" both read as a sentence someone gave up on.
    words = out.split()
    while len(words) > 1 and words[-1].lower() in _TRAILING_FILLER:
        words.pop()
    out = " ".join(words)
    return (out + "…") if out else t[:max_chars]


def _plural(unit):
    """English plural for the unit words we use ('group'->'groups', 'class'->'classes')."""
    return unit + ("es" if unit.endswith(("s", "x", "z", "ch", "sh")) else "s")


def _fraction_phrase(share):
    """A non-analyst reads 'about 1 in 3' faster than '0.34'."""
    pct = round(share * 100)
    if share >= 0.45:
        return f"about half ({pct}%)"
    d = min(range(3, 11), key=lambda d: abs(share - 1 / d))
    # Only say "1 in d" when it is actually that. 40% was being read out as "about 1 in 3 (40%)",
    # which puts a claim and its own contradiction in the same set of brackets.
    if abs(share - 1 / d) > 0.03:
        return f"{pct}%"
    return f"about 1 in {d} ({pct}%)"


def how_k_was_chosen(signals, chosen, k_min, k_max, unit="group", cfg=None):
    """One sentence saying the number was searched for, not assumed.

    The tool tries every number in the range and scores each on nine criteria, but the report only
    said so several sections down, under a heading a marketer has no reason to open. The person
    who commissioned this asked whether the tool could "figure out the best k" — it already did,
    and had been doing it silently. A finding nobody knows about is not a feature.

    Naming the RUNNER-UP is the part that carries real information: "3 groups, and 4 was the next
    best" is a different situation from "3 groups, and nothing else came close", and it is the
    honest answer to the question every client asks — you said three, what if it were four?
    """
    if not signals:
        return ""
    # Count the vote the way the decision counted it. A flat headcount reads as an explanation of
    # a weighted decision and is not one: it once reported "5 of them picked 2" for a k that the
    # two criteria this tool trusts most had both argued against.
    w = signal_weights(cfg) if cfg is not None else {}
    votes, criteria = {}, 0
    for name, k in signals.items():
        if k:
            votes[int(k)] = votes.get(int(k), 0) + w.get(name, 1)
            criteria += 1
    if not sum(votes.values()):
        return ""
    mine = votes.get(int(chosen), 0)
    rivals = sorted(((v, k) for k, v in votes.items() if k != int(chosen)), reverse=True)
    line = (f"**I tried every number of {_plural(unit)} from {k_min} to {k_max}** and scored each "
            f"one on {criteria} independent criteria, counting the ones that measure whether the "
            f"answer reproduces twice over; {chosen} scored {mine}")
    if not rivals:
        return line + ", and nothing else was chosen by any of them.\n"
    count, runner = rivals[0]
    if count >= mine:
        return (line + f" and {runner} scored {count}. The two are close, so treat the number "
                "itself as a judgement call and read the stability checks below before "
                "committing to it.\n")
    # "the number these criteria agree on" rather than "the clear answer": this sentence reports a
    # vote, not a verdict. On structureless data a number still wins the vote, and calling that
    # clear would contradict the red light two lines above it.
    return (line + f" and the next nearest was {runner} with {count}, so {chosen} is the number "
            "these criteria agree on most.\n")


def executive_summary(n_resp, names, shares, wants, min_jaccard, repro, unit="group",
                      k_agreement=None, cross_method=None, k_choice="", n_items=None):
    """The plain-language box at the top of every report: how many groups, who they are, how much
    to trust it (a green/amber/red confidence light built from the stability numbers), and what to
    do next. Written for someone who will never read the word 'eta-squared'.

    The light is deliberately conservative: a solution can be individually reproducible yet still
    have an uncertain NUMBER of groups (the selection criteria disagreed). When k_agreement is low
    we refuse to show green, because 'high confidence in 6 groups' misleads when the count could as
    easily have been 3.

    **Measured end to end over sixty planted studies** — a known number of groups at a known
    separation, through the real pipeline — the light is informative and, more importantly, wrong
    in only one direction:

        light      studies   right k   mean ARI vs truth
        high          16       69%          0.707
        moderate      15       27%          0.267
        low           29        3%          0.088

    Two properties held across all sixty and are now protected by a test: it **never reported more
    groups than were planted** (every error was a merge), and it **never showed green on the two
    weakest separations** (zero "high" in thirty-six studies). What it does not promise is that a
    green light means the count is exactly right — high confidence accompanied a merged answer in
    5 of 16 green runs, always where two planted centres sat within about one noise standard
    deviation per question. Tightening the light to catch those would trade against the property
    worth more, which is that it does not go green on weak data; do not do it without repeating
    the sweep.

    Both stability measures have to agree before the light goes above red. They fail in different
    ways and the amber band used to consult only the first: bootstrap Jaccard sits around 0.7 even
    on structureless data (Hennig's own reading of that band is "a pattern, membership doubtful"),
    so on random answers it alone would report Moderate and the wording would claim the groups
    reproduce while split-half replication said 0.06. Split-half is the measure that actually
    answers "would a fresh sample find these same groups", so it now gates the amber band too."""
    k_contested = k_agreement is not None and k_agreement < 0.6
    # Reproducibility asks "would a fresh sample give the same answer" — and a WRONG answer can be
    # perfectly reproducible. MacKay gives the mechanism in *Information Theory, Inference, and
    # Learning Algorithms* §20.1: k-means "has no way of representing the size or shape of a
    # cluster", so where one group is broad and another narrow it misassigns the broad group's
    # members to the narrow one, and does it the same way every time.
    #
    # Measured on exactly that data (240 broad, 60 narrow): split-half replication 1.000 while the
    # partition agreed with a Gaussian mixture at only 0.43 and with Ward at 0.47 — about half the
    # memberships in dispute — and the report said High.
    methods_disagree = cross_method is not None and cross_method < 0.6
    # More questions than the sample can support. With many questions and few people, distances
    # between respondents concentrate: everybody ends up roughly equidistant, real structure is
    # diluted across the questions that carry none, and — the dangerous part — what survives is
    # extremely REPRODUCIBLE, because noise reproduces. Every criterion this light is built from
    # then agrees on an answer that is wrong.
    #
    # Measured on 150 respondents answering 400 questions where only 60 carried a three-group
    # signal: the tool found two groups at an Adjusted Rand Index of 0.635 against the truth, and
    # called it High confidence on two runs out of three. That is the one thing this report is not
    # allowed to do, so where the sample cannot support the questionnaire the light is capped and
    # the reason is stated. It does not change the segmentation, only the claim made for it.
    too_wide = n_items is not None and n_resp < 2 * n_items
    if min_jaccard >= 0.75 and repro >= 0.6 and not k_contested and not methods_disagree \
            and not too_wide:
        # "the groups are clear" was a claim about SEPARATION, which this light does not measure —
        # it is built from stability numbers. On a real run it sat five lines above a Hopkins
        # statistic of 0.59 described as "essentially random", and the reader had to reconcile
        # them. The light now claims what it actually tested.
        light, label, meaning = "🟢", "High", ("the same groups come back reliably when the "
                                               "analysis is repeated, and each one holds together "
                                               "under resampling.")
    elif min_jaccard >= 0.6 and repro >= 0.4:
        light, label, meaning = "🟡", "Moderate", ("the groups mostly hold up, but " + (
            f"there are {n_items} questions and only {n_resp:,} people, which is too few to pin "
            "down that many answers at once: respondents end up roughly equidistant from each "
            "other, and a wrong answer can still reproduce perfectly. Treat the number of groups "
            "as unsettled, and consider re-running on the questions that matter most."
            if too_wide else
            "the methods disagree on how many groups there really are, so the exact number is "
            "uncertain." if k_contested else
            "other clustering methods put a sizeable share of people in different groups, so who "
            "belongs where is less settled than the group definitions are — the signature of one "
            "broad group and one narrow one, which k-means divides badly." if methods_disagree else
            "their edges are fuzzy, so treat them as a strong hypothesis."))
    elif min_jaccard >= 0.6:
        # Individually stable-looking segments that do not survive a split of the sample. This is
        # what a segmentation of noise looks like from the inside, so say so plainly.
        light, label, meaning = "🔴", "Low", ("re-running the analysis on half the sample produces "
                                              "a different answer, which is what happens when the "
                                              "data has no real groups in it and the method invents "
                                              "them. Check the segment map before going further.")
    else:
        light, label, meaning = "🔴", "Low", ("the groups do not reproduce reliably, so treat them "
                                              "as tentative and gather more data before betting on them.")
    L = ["## In plain language (read this first)\n",
         f"**Confidence: {light} {label}.** In plain terms, {meaning}\n",
         k_choice or None,
         f"We looked at **{n_resp:,} people** and found **{len(names)} {_plural(unit)}** "
         "(a working number, not the only possibility):\n" if k_contested else
         f"We looked at **{n_resp:,} people** and found **{len(names)} {_plural(unit)}**:\n"]
    for i in sorted(range(len(names)), key=lambda i: -shares[i]):
        L.append(f"- **{names[i]}** ({_fraction_phrase(shares[i])} of people) stand out for: {wants[i]}.")
    # This used to say "start with the biggest, most distinct group". Two things wrong with that:
    # nothing here establishes the biggest group is the most distinct, and on a real run the
    # largest segment was the one the stability table below told the reader not to spend money on.
    # The summary must not hand out an instruction the rest of the report contradicts.
    L.append(f"\n**What to do next.** Give the {_plural(unit)} names your team recognises (the ones "
             "above are generated automatically). Then read the two stability tables below before "
             f"committing a budget: they say which {_plural(unit)} hold together and which have "
             "fuzzy edges, and the largest is not always the soundest.")
    if label != "High":
        L.append(f"\n> Because confidence is {label}, treat these {_plural(unit)} as a direction to test "
                 "with a few interviews, not a settled fact.")
    # Filtered, because an optional line is carried as None rather than as an empty string that
    # would leave a blank paragraph in the middle of the box.
    return "\n".join(x for x in L if x is not None) + "\n"


def _dip_section(dip, tendency):
    """The second cluster-tendency test, and what it means read against the first.

    Written to be useful when the two disagree, which is the interesting case: they fail in
    opposite directions, so a disagreement narrows down what kind of data this is rather than
    leaving the reader stuck between two numbers.
    """
    if not dip:
        return None
    if "p" not in dip:
        return (f"\n*Second opinion on cluster tendency: not run — {dip['skipped']}.*\n")
    # "p < 0.001" rather than "p 0": the test reports a probability, and printing zero claims a
    # certainty no statistical test has.
    shown = "< 0.001" if dip["p"] < 0.001 else f"= {dip['p']:.3g}"
    line = (f"\n**Dip test (second opinion): p {shown}** — "
            f"{clusterability.dip_reading(dip['p'])}. This is a different question asked a "
            "different way: whether everyone's answers form one single spread or more than one. "
            "It is worth having beside the Hopkins statistic because the two fail in opposite "
            "directions — Hopkins can read a handful of unusual respondents as a group, and it "
            "loses power when groups overlap instead of sitting apart, which is exactly where "
            "this tool is weakest.\n")
    if tendency:
        line += f"\n> {tendency[1]}\n"
    return line


def _varsel_section(varsel, rec_k):
    """Report section for the Dolnicar variable-selection comparison (None if it was not run)."""
    if varsel is None:
        return None
    if not varsel.get("applicable"):
        return ("\n## Variable-selection check (Dolnicar)\n"
                "No near-noise items to drop — every item contributes signal, so no reduced "
                "comparison is needed.")
    f, r = varsel["full"], varsel["reduced"]
    tbl = pd.DataFrame({"metric": ["split-half ARI", "mean Jaccard", "min Jaccard", "silhouette"],
                        "all items": [round(f["split_half"], 3), round(f["mean_jaccard"], 3),
                                      round(f["min_jaccard"], 3), round(f["silhouette"], 3)],
                        "signal only": [round(r["split_half"], 3), round(r["mean_jaccard"], 3),
                                        round(r["min_jaccard"], 3), round(r["silhouette"], 3)]})
    verdict = ("> Dropping them makes the solution **cleaner or equal** on stability (min Jaccard "
               "and split-half both hold up). Consider re-running on the signal items only."
               if varsel["reduced_is_cleaner"] else
               "> Dropping them does **not** improve stability here, so keeping all items is "
               "defensible — judge by whether the near-noise items are conceptually meaningful.")
    return ("\n## Variable-selection check (Dolnicar): do the near-noise items hurt?\n"
            f"Near-noise items (eta-squared below 0.05): {', '.join(varsel['dropped'])}. "
            f"Re-clustering on the {varsel['n_signal']} signal items only, at k = {rec_k}:\n\n"
            + _md(tbl) + "\n\n" + verdict +
            "\n\nThis is a recommendation; the shipped segmentation above still uses all items.")


def _typing_line(typing, rule_file="typing_rule.json", unit="segment"):
    """One validation bullet describing how reliably the typing tool reproduces the segments
    out-of-sample (read against the majority-class baseline)."""
    acc = typing["cv_accuracy"]
    if np.isnan(acc):
        return (f"- {unit.capitalize()} predictability (typing tool): not estimated — a {unit} was "
                f"too small for cross-validation. The exportable rule ({rule_file}) is still written.")
    base = typing["baseline_majority"]
    verdict = ("the rule assigns new respondents consistently and is safe to deploy for typing"
               if acc >= 0.85 and (acc - base) >= 0.15 else
               "typing is workable but with meaningful misclassification — carry the confidence "
               "score into how you use it" if acc >= 0.60 else
               "even the assignment boundary is unstable — do not rely on typing individuals")
    return (f"- {unit.capitalize()} predictability (typing tool): a rule trained on part of the "
            f"sample reproduces the {unit} of a held-out respondent **{acc:.0%}** of the time "
            f"(stratified cross-validation), versus **{base:.0%}** for always guessing the largest "
            f"{unit} — {verdict}. This measures how consistently the rule *assigns*, not whether the "
            f"{_plural(unit)} are real (a partition of noise is still classifiable) — for that, read the "
            f"stability and Jaccard columns above. The rule is exported to {rule_file} (apply it "
            f"with `--classify new.csv --rule {rule_file}`).")



def _hopkins_caveat(distinct_share, n_items):
    """Warn when the Hopkins statistic cannot be trusted for THIS data.

    Hopkins compares distances between real points to distances from uniformly sampled ones, so
    it is inflated wherever real points sit exactly on top of each other. On a short Likert
    survey that happens constantly: two 1-to-5 questions admit only 25 answer patterns, so 120
    respondents pile onto duplicates and Hopkins reads 0.78 — "strong tendency to cluster" — on
    data with no structure whatsoever. It is also biased upward in very low dimensions. Both
    cases would tell a reader the opposite of the truth, so say so next to the number rather
    than leaving it to be discovered."""
    reasons = []
    if distinct_share is not None and distinct_share < 0.5:
        reasons.append(f"only {distinct_share:.0%} of respondents have a distinct combination of "
                       "answers, so many sit exactly on top of each other")
    if n_items and n_items <= 3:
        reasons.append(f"there are only {n_items} questions, and this statistic reads high in "
                       "very few dimensions")
    if not reasons:
        return ""
    return ("\n> **Do not lean on the Hopkins number here:** " + "; and ".join(reasons) +
            ". Both inflate it, so a high value is not evidence of real segments in this "
            "dataset. Judge this run on the replication and per-segment stability figures "
            "below, which are not affected.\n")

def stability_across_solutions(X, k_range, chosen_k, labels, n_init, rng, cfg=None):
    """Would each segment still be there if a different number had been chosen?

    Dolnicar & Leisch, *Using segment level stability to select target segments in data-driven
    market segmentation studies* (2017), and Step 5 of *Market Segmentation Analysis*. The tool
    already asks whether the whole solution reproduces on a fresh half (global stability) and
    whether each segment survives resampling at a fixed k (Hennig's bootstrap Jaccard). Neither
    asks the question a client actually puts: *you said four groups — what if it were five?*

    A segment that reappears intact at k-1, k and k+1 is a real feature of the data. One that
    exists only at the chosen k is an artefact of the number, and should not be handed a budget.

    Scored by CONTAINMENT — what share of a segment's members stay together in the neighbouring
    solution — not by Jaccard. Jaccard was the first attempt and it does not work here: going from
    k to k-1 must merge two segments, so Jaccard collapses mechanically whether or not the
    structure is real. Measured, it could not tell them apart at all (three genuine segments
    scored 0.44-0.78, pure noise 0.55-0.69). Containment is unharmed by merging — everyone still
    lands together — and falls only when a segment genuinely fragments. Measured again with it:
    three real segments 0.78, five real 0.79, two adjacent 0.82, pure noise 0.47.

    **The two directions are reported separately, and only one of them is evidence about whether
    a segment is real.** Taking the weaker of the two — which this did — produces a false alarm on
    exactly the segments most worth having. Asking for one MORE group forces the solution to split
    something, so whichever segment gets subdivided scores about 0.5 in that direction whether or
    not it is genuine. Measured on 420 students whose three mind-sets the tool recovered at an
    Adjusted Rand Index of 0.954 — every segment real — the largest and cleanest of them held
    together perfectly under merging (1.00) and scored 0.56 under splitting, and the report told
    the reader not to build a campaign on it.

    Merging is the informative direction. When the solution is asked for one FEWER group, a real
    segment moves into a bigger group intact; one that was never a unit scatters across several.
    Splitting says something useful too, but a different thing: that the segment contains
    recognisable sub-groups if the team wants finer detail. That is an opportunity, not a defect.

    Returns {segment index: {"merges": float, "splits": float}} — either key absent when that
    neighbouring solution does not exist — or {} when there are no neighbours at all.
    """
    labels = np.asarray(labels)
    chosen = {c: set(np.flatnonzero(labels == c)) for c in range(int(chosen_k))}
    neighbours = [k for k in (chosen_k - 1, chosen_k + 1) if k in set(k_range) and k >= 2]
    if not neighbours or not chosen:
        return {}

    scores = {c: {} for c in chosen}
    for k in neighbours:
        try:
            other = _fit(X, k, cfg, n_init, rng.integers(1e9)).labels_
        except Exception:
            continue
        where = "merges" if k < chosen_k else "splits"
        for c, members in chosen.items():
            if not members:
                continue
            counts = np.bincount(other[sorted(members)], minlength=k)
            scores[c][where] = float(counts.max() / len(members))
    return {c: v for c, v in scores.items() if v}


def persistence_paragraph(persistence, names):
    """Report what happens to each segment when a different number of groups is asked for.

    Two questions, kept apart because they have different answers and different consequences:
    does the segment stay together when the solution is coarser, and does it subdivide when the
    solution is finer. Only the first says whether the segment is a real unit. The second is a
    finding a marketer can use — "this group has two flavours in it if you want them".
    """
    if not persistence:
        return ""
    if not isinstance(next(iter(persistence.values())), dict):   # older single-number form
        persistence = {c: {"merges": v} for c, v in persistence.items()}

    rows = []
    for c, v in sorted(persistence.items()):
        row = {"segment": names[c] if c < len(names) else f"Segment {c}"}
        if "merges" in v:
            row["stays together with fewer groups"] = round(v["merges"], 2)
            row["reading"] = ("holds together" if v["merges"] >= 0.7 else
                              "partly holds" if v["merges"] >= 0.55 else "scatters")
        if "splits" in v:
            row["survives asking for more groups"] = round(v["splits"], 2)
            row["if you asked for more"] = ("stays one group" if v["splits"] >= 0.7 else
                                            "divides into sub-groups")
        rows.append(row)

    holds = [v["merges"] for v in persistence.values() if "merges" in v]
    weakest = min(holds) if holds else None
    if weakest is None:
        verdict = ""
    elif weakest >= 0.7:
        verdict = ("Every segment stays together when the analysis is asked for fewer groups, "
                   "which is the strongest sign that they are features of your customers rather "
                   "than of the number you happened to choose.")
    elif weakest >= 0.55:
        verdict = ("At least one segment partly breaks up when fewer groups are requested. Its "
                   "centre is real, but its boundaries are a choice — be careful quoting its "
                   "exact size.")
    else:
        verdict = ("At least one segment scatters when fewer groups are requested: its members go "
                   "to different places, which is what happens when it was never a single group. "
                   "Treat the ones marked 'scatters' as the weakest part of this segmentation.")

    # Quoted, because these names are themselves sentences: unquoted, the line read "would divide
    # would use an app to find... into smaller ones".
    splitters = ['"%s"' % (names[c] if c < len(names) else f"Segment {c}")
                 for c, v in sorted(persistence.items())
                 if v.get("splits") is not None and v["splits"] < 0.7]
    if splitters:
        verdict += ("\n\nAsking for one more group would divide "
                    + (", ".join(splitters[:2]) if len(splitters) <= 2 else
                       f"{len(splitters)} of the segments")
                    + " into smaller ones. That is not a fault: with any number of groups, asking "
                      "for one more has to split something. It means finer detail is available "
                      "there if the team wants it.")
    return ("\n**What happens if you ask for a different number of groups?** Each segment is "
            "re-checked against the solutions with one group fewer and one more, scored by the "
            "share of its members that stay together (1 = the whole segment lands in one place). "
            "The two directions answer different questions: staying together with FEWER groups is "
            "what says a segment is a real unit, while dividing when asked for MORE groups is "
            "normal and simply means it has sub-groups in it. Dolnicar & Leisch call this segment "
            "level stability across solutions.\n"
            + _md(pd.DataFrame(rows)) + "\n" + verdict + "\n")


def segmentation_kind(single_cluster, repro, median_shadow):
    """Which of the three kinds of segmentation this is, in the field's own words.

    Dolnicar, Grün & Leisch use a three-way distinction throughout *Market Segmentation Analysis*
    (Springer 2018), and it is more informative than a single confidence word because it separates
    "there is nothing here" from "there is nothing natural here but the split is stable enough to
    work with" — two very different situations that both feel like a weak result:

      natural       — genuine groups exist in the data and the analysis found them.
      reproducible  — no natural groups, but the data has enough structure that the same split
                      comes back every time, so it is a usable working division.
      constructive  — no structure at all; the segments are an artefact of the method.

    Deciding between them from what the tool already measures: the gap statistic's verdict on a
    single cluster, split-half replication, and the typical shadow value (how stranded the average
    respondent is between two segments — near 0 means people sit firmly somewhere).
    """
    if single_cluster and repro is not None and repro < 0.5:
        kind = "constructive"
        gloss = ("there is no group structure in these answers. The split below is one the method "
                 "constructed, not one it found, and it will not correspond to anything real "
                 "about your customers.")
    elif median_shadow is not None and median_shadow < 0.55 and (repro or 0) >= 0.6:
        kind = "natural"
        gloss = ("the groups are really there in the answers — most respondents sit firmly inside "
                 "one of them, and the same groups come back when the analysis is repeated.")
    elif (repro or 0) >= 0.6:
        kind = "reproducible"
        gloss = ("there are no sharply separated natural groups here, but the division is stable: "
                 "repeat the analysis and you get the same split. That makes it a usable working "
                 "segmentation as long as nobody claims these are naturally occurring types.")
    else:
        kind = "constructive"
        gloss = ("the division does not survive being repeated on half the sample, which is what "
                 "happens when the method is imposing groups rather than finding them.")
    return kind, gloss


def neighbours_paragraph(pairs, names, k):
    """Which two segments are nearly the same — the question that decides how many campaigns.

    The report says how each segment differs from the average respondent. It never said which
    segments sit next to each other, and a marketer signing off five campaigns needs to know that
    two of them are aimed at much the same people. Leisch's s_ij (average shadow value over the
    respondents caught between a given pair) is exactly that number.

    The bands were calibrated on this machine rather than guessed, by planting segments at known
    separations and reading off the worst pair: two segments far apart score 0.27, three far
    apart 0.40, two just touching 0.65, three with two adjacent 0.87, three with two nearly
    identical 0.97, and pure noise cut into three 0.99. Hence 0.55 and 0.80 as the boundaries.

    Only pairs with respondents actually stranded between them are shown, and each unordered
    pair once.
    """
    if not pairs or k < 2:
        return ""

    # The automatic names come from question codes and two segments can easily end up with the
    # same one, which produced a row reading "q2 + q4 | q2 + q4". Number the duplicates.
    def label(i):
        base = names[i] if i < len(names) else f"Segment {i}"
        return f"{base} (segment {i})" if names.count(base) > 1 else base

    seen, rows = set(), []
    for i, j, mean_shadow, n in pairs:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        rows.append((label(i), label(j), mean_shadow, n))
    rows = rows[:4]
    worst = rows[0][2]
    if worst >= 0.80:
        verdict = ("**These two are barely distinguishable at their boundary.** Before running "
                   "separate campaigns, check that the difference is one your team can actually "
                   "act on; one message may serve both.")
    elif worst >= 0.55:
        verdict = ("The closest pair shades into one another at the edges. The centres are "
                   "distinct, but expect people near the boundary to respond to either message.")
    else:
        verdict = "No two segments crowd each other; each has its own territory."
    lines = ["\n**Which segments sit next to each other.** For every respondent, how close they "
             "are to their own segment compared with the next nearest one. A pair scoring near 1 "
             "means the people on that boundary are almost equally at home in either.\n",
             _md(pd.DataFrame([{"segment": a, "next nearest": b,
                                "how alike at the boundary": round(m, 2),
                                "people between them": n} for a, b, m, n in rows])),
             "\n" + verdict + "\n"]
    return "".join(lines)


def make_report(diag, rec_k, rationale, reached, split_half, sil_overall, jaccard,
                sizes, defining, differentiating, centroids, hopkins, mb_agreement,
                var_importance, consensus_agreement, cfg, typing=None, varsel=None,
                k_agreement=None, ward_ari=None, distinct_share=None,
                single_cluster=False, gap_one=None, gap_two=None, neighbours=None,
                median_shadow=None, persistence=None, signals=None, dip=None, tendency=None,
                k_max_note=""):
    _method = getattr(cfg, "method", "kmeans")
    # How many respondents this report covers, read off the sizes table. Used only to disclose
    # which columns were estimated on a subsample.
    try:
        _total_n = int(sizes["n"].sum())
    except Exception:
        _total_n = None
    # Name only the columns actually present. consensus_PAC is absent when consensus was turned
    # off, and a note claiming it had been sampled would describe a column the reader cannot see.
    _sampled_cols = [c for c in ("silhouette", "consensus_PAC") if c in diag.columns]
    method_name = {"gmm": "a Gaussian mixture / latent-class model ("
                          + cfg.gmm_covariance + " covariance)",
                   "kproto": "Gower k-prototypes (Szepannek et al. 2024), which uses the rating "
                             "and the pick-any questions together",
                   }.get(_method, "k-means")
    # Said first and said plainly. The gap statistic is the only criterion here that can return
    # "one group" — silhouette and the rest are undefined at k=1 — and Hastie, Tibshirani and
    # Friedman note it is the scenario where most competing methods fail. If it says there is
    # nothing to divide, the reader should know before they read a division.
    single_cluster_note = ("" if not single_cluster else
        "> **The gap statistic says these respondents are one group, not several.** Compared "
        f"against data with no structure at all, one group scores {gap_one:.2f} and two score "
        f"{gap_two:.2f} — splitting does not earn its keep. Everything below still describes the "
        "best split that could be made, because that is what was asked for, but it is a division "
        "imposed on the data rather than one found in it. Treat the groups as a working "
        "convenience and do not build a campaign on them.\n")
    _keys = list(defining.keys())
    _names = [defining[k]["auto_name"] for k in _keys]
    _wants = [", ".join(_short_label(w.split(" (")[0]) for w in defining[k]["most_above_average"][:2])
              or "a distinct mix of answers" for k in _keys]
    _shares = [float(sizes.loc[k, "share"]) for k in _keys]
    L = ["# Segmentation report\n",
         executive_summary(int(sizes["n"].sum()), _names, _shares, _wants,
                           min(jaccard.values()), split_half, unit="group", k_agreement=k_agreement,
                           k_choice=how_k_was_chosen(signals, rec_k, cfg.k_min, cfg.k_max,
                                                     cfg=cfg),
                           # How many questions the sample has to support at once.
                           n_items=(len(centroids.columns) if centroids is not None else None),
                           # The weaker of the two independent paradigms: a mixture model, which
                           # unlike k-means can represent a cluster's breadth, and Ward, which
                           # builds bottom-up rather than around centroids. If both put people
                           # somewhere else, the memberships are not settled.
                           # NaN has to be filtered, not just type-checked: it is a float, and
                           # min() returns whichever NaN it meets first, so a single unavailable
                           # check would silently throw away the one that did run.
                           cross_method=min(
                               [v for v in ((mb_agreement or {}).get("agreement_ARI"), ward_ari)
                                if isinstance(v, (int, float)) and not np.isnan(v)],
                               default=None)),
         (f"Respondents clustered with **{method_name}**; final fit used "
          f"{cfg.n_init_final} restarts. Search range: k = {cfg.k_min} to {cfg.k_max}."
          f"{k_max_note}\n"
          if _method == "kproto" else
          f"Respondents clustered with **{method_name}** on **{cfg.scaling}**-scaled utilities; "
          f"final fit used {cfg.n_init_final} restarts. Search range: k = "
          f"{cfg.k_min} to {cfg.k_max}.{k_max_note}\n"),
         # Say plainly which checks this path cannot run. Reporting four corroborating numbers
         # where the numeric path reports five, without mentioning the fifth, would quietly
         # overstate how much agreement there is behind the answer.
         ("\n> **What is different about a mixed-question segmentation.** Every question is scored "
          "on Gower's distance: a rating by how far apart the two answers are on the scale, a "
          "pick-any question by whether the two people chose the same thing. Ratings are read as "
          "ordered answers rather than as numbers, so the distance between \"agree\" and "
          "\"strongly agree\" reflects how many people actually sit between them.\n>\n"
          "> Two things in the panel below work differently as a result. The Gaussian-mixture "
          "cross-check is **not run** — a mixture assumes every answer is a measurement, and "
          "fitting one to brand codes would produce a number that looks like corroboration and "
          "is not — so the bottom-up cross-check is average linkage on Gower's distance instead. "
          "And Calinski-Harabasz and Davies-Bouldin are sums of squares, so they are read on the "
          "coordinates behind Gower's distance rather than on Gower itself; the silhouette, "
          "prediction strength, replication and per-segment stability are all exact.\n"
          if _method == "kproto" else None),
         "## Is there anything to segment? (cluster tendency)\n",
         (lambda kg: f"**This is a {kg[0]} segmentation** — {kg[1]}\n\n"
                     "The three kinds, as market segmentation research distinguishes them "
                     "(Dolnicar, Grün & Leisch): a **natural** segmentation finds groups that are "
                     "genuinely there; a **reproducible** one finds no natural groups but a stable "
                     "division you can still work with; a **constructive** one is the method "
                     "inventing groups in data that has none.\n"
          )(segmentation_kind(single_cluster, split_half, median_shadow)),
         single_cluster_note,
         f"Hopkins statistic = **{hopkins:.2f}** — {hopkins_reading(hopkins)}. "
         "A value near 0.5 means the data are essentially random and any segments will be "
         "constructed by the method rather than discovered; above ~0.75 signals a real tendency "
         "to cluster. Read the rest of this report in that light.\n"
         + _hopkins_caveat(distinct_share, centroids.shape[1] if centroids is not None else 0),
         _dip_section(dip, tendency),
         "**Who fits, person by person.** `segment_assignments.csv` carries a `fit` column: how "
         "well each respondent sits in the segment they were given (about 1 = squarely inside, "
         "about 0 = on the boundary between two, below 0 = closer to a different segment). "
         "k-means gives everybody a segment because it has no way to answer *none of these* — so "
         "before spending money on a list, filter out the low scores rather than mailing people "
         "who matched nothing in particular.\n",
         neighbours_paragraph(neighbours, _names, len(_names)),
         persistence_paragraph(persistence, _names),
         "## Choosing the number of segments\n", rationale, "\n",
         _md(diag.round(3)),
         # Two columns here are estimated on a subsample once the study is large enough that
         # every-pair arithmetic will not fit in memory. Saying so is the whole point: a sampled
         # number presented as if it covered everybody is the failure this tool keeps meeting.
         (None if _total_n is None or _total_n <= MAX_PAIRWISE_N else
          "\n> **{} above {} estimated on a sample.** With {:,} respondents, {} {} a distance "
          "between every pair of people — {:,} pairs, which does not fit in memory — so {} "
          "computed on a random {:,} of them. Every other column, and the segmentation itself, "
          "uses everybody.".format(
              "Two columns" if len(_sampled_cols) > 1 else "One column",
              "are" if len(_sampled_cols) > 1 else "is",
              _total_n,
              " and ".join(f"**{c}**" for c in _sampled_cols),
              "need" if len(_sampled_cols) > 1 else "needs",
              _total_n * (_total_n - 1) // 2,
              "both are" if len(_sampled_cols) > 1 else "it is",
              MAX_PAIRWISE_N)),
         "\n**How to read this table.** The two columns to trust most are **prediction_strength** "
         "(Tibshirani & Walther: pick the largest k above 0.80) and **stability_ARI** (Dolnicar & "
         "Leisch: near 1.0 means the same segments re-emerge when you repeat the analysis; below "
         "~0.6 the segments are constructed noise). **consensus_PAC** (Monti consensus clustering; "
         "Senbabaoglu's Proportion of Ambiguous Clustering) is the share of point pairs whose "
         "co-membership is ambiguous across resamples — LOWER is better, and the k with the "
         "smallest PAC is the cleanest solution. **gmm_BIC** and **gmm_ICL** are the model-based "
         "estimates — the k with the lowest value; ICL (Biernacki-Celeux-Govaert) adds an entropy "
         "penalty for overlapping components and so favours cleaner, better-separated segments. "
         "**silhouette** and **Calinski-Harabasz** (higher = better) and "
         "**davies_bouldin** (lower = better) measure separation; the Calinski-Harabasz index was "
         "the best single stopping rule in Milligan & Cooper's classic comparison. **gap** peaks at "
         "a supported k. The inertia **elbow** is the weakest signal and is shown for completeness.\n",
         f"\n## Validation of the chosen {rec_k}-segment solution\n",
         f"- Local-optima check: the best solution was reached in **{reached:.0%}** of random "
         "restarts (low means the landscape is rough — keep the many-restarts setting).",
         f"- Split-half replication (Adjusted Rand Index): **{split_half:.3f}** "
         f"({'reproduces well' if split_half >= 0.5 else 'does NOT reproduce — treat as unstable'}).",
         f"- Overall average silhouette: **{sil_overall:.3f}**.",
         (f"- Cross-paradigm check: the chosen partition agrees with {mb_agreement['other_method']} "
          f"({mb_agreement['covariance']} covariance) at Adjusted Rand Index "
          f"**{mb_agreement['agreement_ARI']:.3f}** "
          f"({'strong agreement — two different methods see the same segments' if mb_agreement['agreement_ARI'] >= 0.7 else 'only partial agreement — the partition is somewhat method-dependent, read with caution'}). "
          f"Mixture assignment confidence (mean top posterior) = {mb_agreement['mean_max_posterior']:.2f}, "
          f"normalized entropy = {mb_agreement['normalized_entropy']:.2f} (0 = crisp, 1 = fuzzy)."
          if not np.isnan(mb_agreement['agreement_ARI']) else
          "- Model-based cross-check: the Gaussian mixture did not converge; skip."),
         (None if ward_ari is None else
          ("- Third-method (Ward) cross-check: skipped (sample too large for the O(n^2) linkage)."
           if np.isnan(ward_ari) else
           f"- Third-method cross-check: Ward hierarchical clustering (merges bottom-up, not around "
           f"centroids) agrees with the chosen partition at Adjusted Rand Index **{ward_ari:.3f}** ("
           + ("a third, structurally different method sees the same segments)."
              if ward_ari >= 0.7 else
              "only partial agreement with a third method, so read with some caution)."))),
         (f"- Consensus (ensemble) robustness: a Monti consensus partition, aggregated over many "
          f"resampled clusterings, agrees with the main partition at Adjusted Rand Index "
          f"**{consensus_agreement:.3f}** "
          f"({'the segmentation is robust to resampling' if consensus_agreement >= 0.8 else 'the segmentation shifts under resampling — treat the boundary cases as uncertain'}). "
          f"Run with --consensus-final to adopt the more robust ensemble partition."
          if consensus_agreement is not None else
          "- Consensus robustness: not run (use --no-consensus was set, or run_consensus=False)."),
         (_typing_line(typing) if typing is not None else None),
         "\n**Per-segment bootstrap stability (Hennig's Jaccard).** This is the decisive test of "
         "which segments are real:\n"]
    # Carry the suggested name alongside the number. The number is what the exported
    # segment_assignments.csv uses, so it has to stay; the name is how every other table in this
    # report refers to the same group. With only the number here, the decisive stability table
    # could not be matched to any group the reader had been introduced to.
    _auto = {k: defining[k].get("auto_name", "") for k in defining}
    jac = pd.DataFrame({"segment": [f"Segment {c}" for c in jaccard],
                        "suggested name": [_auto.get(f"Segment {c}", "") for c in jaccard],
                        "mean_Jaccard": [round(v, 3) for v in jaccard.values()],
                        "reading": [jaccard_reading(v) for v in jaccard.values()]})
    L.append(_md(jac))
    if (np.array(list(jaccard.values())) < 0.6).any():
        L.append("\n> WARNING: one or more segments fall below 0.60 Jaccard — they are not "
                 "trustworthy as distinct segments. Consider a smaller k, a different scaling, "
                 "or a model-based method (Gaussian mixture / latent class).")
    small = sizes[sizes["share"] < cfg.min_segment_frac]
    if len(small):
        L.append(f"\n> NOTE: {len(small)} segment(s) below {cfg.min_segment_frac:.0%} of the "
                 "sample; check they are real rather than fragments.")
    L += ["\n## Segment sizes\n",
          ("The **population_share** column reweights each respondent by your survey weight to "
           "estimate each segment's size in the whole population (the segments were still formed "
           "on unweighted data). Compare it with **share** (the raw sample) to see over- or "
           "under-representation.\n" if "population_share" in sizes.columns else None),
          _md(sizes.assign(**{"suggested name": [_auto.get(i, "") for i in sizes.index]})
              [["suggested name"] + list(sizes.columns)], index=True),
          "\n## The mind-sets (what defines each segment)\n"]
    for seg, d in defining.items():
        L.append(f"**{seg}** — suggested name: *{d['auto_name']}*")
        L.append(f"  - Values most: {', '.join(d['most_above_average'])}")
        L.append(f"  - Values least: {', '.join(d['most_below_average'])}")
        for pick in d.get("picks", []):
            L.append(f"  - Mostly picks {pick}")
        L.append("")
    _mixed = any(d.get("picks") for d in defining.values())
    L += [("## What differentiates the segments (one-way ANOVA F, or chi-square for the "
           "pick-any questions; high = splits them most)\n" if _mixed else
           "## What differentiates the segments (one-way ANOVA F; high = splits them most)\n"),
          _md(differentiating.head(10).round(2)),
          "\n## Which items drive the segmentation (variable importance)\n",
          ("Eta-squared is the share of each item's variance explained by segment membership. For "
           "the pick-any questions it is Cramer's V squared, which puts the same idea on the same "
           "scale without depending on the order the answers happened to be listed in. "
           if _mixed else
           "Eta-squared is the share of each item's variance explained by segment membership. ")
          + "Near-zero items add noise and can mask real structure (Dolnicar's variable-selection "
          "point) — consider dropping the near-noise items and re-running.\n",
          _md(var_importance),
          _varsel_section(varsel, rec_k),
          "\n## Segment-by-item mean utilities (centroids, on the raw scale)\n",
          _md(centroids.round(1), index=True),
          "\n---\n**Methodology.** Number of segments chosen by a weighted panel (prediction "
          "strength and replication stability first, then separation indices, then the gap "
          "statistic) rather than a single elbow; per-segment validity judged by bootstrap "
          "Jaccard stability (Hennig 2007). "
          + ("Distance and prototypes follow Szepannek, Aschenbruck & Wilhelm (2024); ordinal "
             "answers use Podani's metric rank transformation (1999). "
             if _method == "kproto" else
             "Range standardization follows Milligan & Cooper (1988). ")
          + "A reminder from Dolnicar & Leisch: data-driven segments are usually "
          "*constructed* by the method, not discovered — so trust the stability columns, and "
          "rename the auto-suggested mind-set names to something a non-analyst would recognise "
          "before shipping. Demographics were not used to form the segments; profile them "
          "separately (below, if a demographics file was supplied).",
          f"\n*Generated by segment_kmeans version {__version__}.*"]
    return "\n".join(x for x in L if x is not None)


def maybe_plot(diag, X, labels, rec_k, outdir, cfg=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(diag["k"], diag["inertia"], "o-"); ax[0, 0].axvline(rec_k, ls="--", c="grey")
    ax[0, 0].set(title="Elbow (inertia) — weakest signal", xlabel="k", ylabel="within-cluster SS")
    ax[0, 1].plot(diag["k"], diag["prediction_strength"], "o-", label="prediction strength")
    ax[0, 1].plot(diag["k"], diag["stability_ARI"], "s-", label="replication stability (ARI)")
    ax[0, 1].axhline(0.8, ls=":", c="green"); ax[0, 1].axvline(rec_k, ls="--", c="grey")
    ax[0, 1].legend(); ax[0, 1].set(title="Stability & prediction strength (trust these)", xlabel="k", ylim=(0, 1.05))
    ax[1, 0].plot(diag["k"], diag["silhouette"], "o-", label="silhouette")
    if "gmm_BIC" in diag:
        ax2 = ax[1, 0].twinx(); ax2.plot(diag["k"], diag["gmm_BIC"], "d-", c="firebrick", label="GMM BIC")
        ax2.set_ylabel("GMM BIC (lower=better)", color="firebrick")
    ax[1, 0].axvline(rec_k, ls="--", c="grey"); ax[1, 0].set(title="Separation & model-based BIC", xlabel="k")
    # Every-pair again, and the last unguarded instance: measured at 48,842 respondents this one
    # call cost 1.07 GB on its own, for a panel in a PNG. Same subsample as the diagnostics table,
    # so the picture and the number it illustrates are computed on the same people, and the title
    # says when it is a sample.
    _take = _pairwise_sample(len(X), cfg) if cfg is not None else None
    if _take is not None:
        sv = silhouette_samples(np.asarray(X)[_take], np.asarray(labels)[_take])
        labels = np.asarray(labels)[_take]
    else:
        sv = silhouette_samples(X, labels)
    y = 10
    for c in np.unique(labels):
        vals = np.sort(sv[labels == c]); ax[1, 1].fill_betweenx(np.arange(y, y + len(vals)), 0, vals); y += len(vals) + 10
    ax[1, 1].axvline(sv.mean(), ls="--", c="red"); ax[1, 1].set(title=f"Silhouette by segment (k={rec_k})", xlabel="silhouette")
    fig.tight_layout(); path = outdir / "diagnostics.png"; fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)


# =====================================================================================
# Charts — so the reader can judge the segments with their own eyes
# =====================================================================================
# Every number in the report is a summary, and a summary can flatter a bad segmentation: k-means
# always returns k groups, so "we found 3 segments" is true even when the data is one shapeless
# cloud. The charts exist so nobody has to take the write-up (or Claude's reading of it) on trust
# — a point cloud that is obviously one blob cut into wedges is visible in a second and arguable
# in a meeting, which no confidence word ever is.
#
# They are drawn as hand-built SVG rather than with matplotlib, deliberately:
#   * no extra dependency, so the packaged desktop app stays ~80 MB and its build stays reliable
#     (matplotlib is one of the more fragile things to freeze with PyInstaller),
#   * vector output stays crisp at any zoom and prints properly,
#   * it is inline text, so it survives the "Save as PDF" path and the standalone HTML report
#     with no image files to lose.
# Chrome (axes, labels, grid) is drawn in `currentColor` so the charts follow the surrounding
# light/dark theme for free; only the categorical segment colours are fixed, which is correct —
# a segment's colour should not change meaning between themes.

# Okabe-Ito, the standard colourblind-safe categorical set, led by the app's own green and with
# its yellow dropped (it vanishes against the beige ground).
# Charts live in charts.py, drawn with matplotlib. scikit-learn above does the computation —
# KMeans, GaussianMixture, the silhouette and neighbour statistics — and nothing in charts.py
# decides anything about the segmentation; it only renders what was already found.
#
# This used to be 684 lines here that built SVG by concatenating f-strings, working out tick
# positions, path data and text anchors by hand. Every chart re-derived axes and scaling from
# scratch, which is why adding one cost dozens of lines of coordinate arithmetic.
_REPORT_CSS = """
:root{color-scheme:light dark}
body{margin:0;background:#f6f7f9;color:#1a1a1a;font:16px/1.6 -apple-system,BlinkMacSystemFont,
 "Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:820px;margin:0 auto;padding:32px 22px 80px;background:#fff;box-shadow:0 0 0 1px #e6e8eb}
h1{font-size:1.7rem;margin:.2em 0 .6em;border-bottom:2px solid #eaecef;padding-bottom:.3em}
h2{font-size:1.28rem;margin:1.5em 0 .5em}h3{font-size:1.08rem;margin:1.2em 0 .4em}
p,li{margin:.4em 0}code{background:#f0f2f4;padding:.1em .35em;border-radius:4px;font-size:.9em}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92rem;overflow-x:auto;display:block}
th,td{border:1px solid #e2e5e9;padding:6px 10px;text-align:left}
thead th{background:#f2f4f7}tbody tr:nth-child(even){background:#fafbfc}
blockquote{margin:1em 0;padding:.7em 1em;background:#fff8e6;border-left:4px solid #f0c000;border-radius:4px}
hr{border:none;border-top:1px solid #e2e5e9;margin:1.6em 0}
strong{color:#111}
.cwrap{overflow-x:auto;border:1px solid #e2e5e9;border-radius:8px;padding:10px;margin:.6em 0}
.cwrap>.chart{display:block;min-width:430px;max-width:100%;height:auto}
.ccap{font-size:.9rem;color:#5b6470;line-height:1.55}
@media print{.cwrap{border:none;padding:0;overflow:visible;break-inside:avoid}}
@media(prefers-color-scheme:dark){
 body{background:#0e1116;color:#d8dce1}main{background:#161a20;box-shadow:0 0 0 1px #2a2f37}
 h1{border-color:#2a2f37}code{background:#22272e}th,td{border-color:#2a2f37}thead th{background:#1c2128}
 tbody tr:nth-child(even){background:#1a1f26}blockquote{background:#2a2410;border-color:#b58900}
 strong{color:#fff}hr{border-color:#2a2f37}.cwrap{border-color:#2a2f37}.ccap{color:#9aa4b1}}
"""


# Drawing lives in charts.py (matplotlib); the computation above stays here (scikit-learn).
#
# Re-exported by assignment rather than a bare `from charts import ...` so that
# `segment_kmeans` remains the one public surface — the tests, the report writer and the web
# layer all reach for `sk.chart_segment_map` and friends, and moving the drawing into its own file
# should not change what any caller imports.
import charts as _charts
from charts import build_charts

SEG_COLOURS = _charts.SEG_COLOURS
SEG_LIGHT = _charts.SEG_LIGHT
SEG_DARK = _charts.SEG_DARK
SEG_MARKERS = _charts.SEG_MARKERS
seg_colour = _charts.seg_colour
seg_marker = _charts.seg_marker
pca_2d = _charts.pca_2d
chart_segment_map = _charts.chart_segment_map
chart_silhouette = _charts.chart_silhouette
chart_k_choice = _charts.chart_k_choice
chart_profiles = _charts.chart_profiles
chart_heatmap = _charts.chart_heatmap
chart_gorge = _charts.chart_gorge
onehot_matrix = _charts.onehot_matrix


def _inline_md(t):
    t = _html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", t)
    return t


def _markdown_to_html(md):
    """A small, dependency-free converter for the exact Markdown the reports emit: headings, pipe
    tables, blockquotes, bullet lists, horizontal rules, bold/italic/code. Not a general parser."""
    lines = md.split("\n"); out = []; i = 0; n = len(lines); in_list = False; list_tag = "ul"

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(f"</{list_tag}>"); in_list = False

    while i < n:
        s = lines[i].rstrip()
        st = s.strip()
        if st.startswith("|") and i + 1 < n and set(lines[i + 1].strip()) <= set("|:- "):
            close_list()
            header = [c.strip() for c in st.strip("|").split("|")]
            i += 2; rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            th = "".join(f"<th>{_inline_md(c)}</th>" for c in header)
            body = "".join("<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in r) + "</tr>"
                           for r in rows)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if not st:
            close_list()
        elif st.startswith("### "):
            close_list(); out.append(f"<h3>{_inline_md(st[4:])}</h3>")
        elif st.startswith("## "):
            close_list(); out.append(f"<h2>{_inline_md(st[3:])}</h2>")
        elif st.startswith("# "):
            close_list(); out.append(f"<h1>{_inline_md(st[2:])}</h1>")
        elif st == "---":
            close_list(); out.append("<hr>")
        elif st.startswith("> "):
            close_list(); out.append(f"<blockquote>{_inline_md(st[2:])}</blockquote>")
        elif st.startswith("- ") or re.match(r"\d+\. ", st):
            # Bullets and numbered lists. Claude's interpretation sometimes numbers its points, so
            # render those as a real ordered list rather than leaving "1." adrift in a paragraph.
            ordered = not st.startswith("- ")
            tag = "ol" if ordered else "ul"
            if in_list and tag != list_tag:
                close_list()
            if not in_list:
                list_tag = tag; out.append(f"<{tag}>"); in_list = True
            out.append(f"<li>{_inline_md(st.split('. ', 1)[1] if ordered else st[2:])}</li>")
        else:
            close_list(); out.append(f"<p>{_inline_md(st)}</p>")
        i += 1
    close_list()
    return "\n".join(out)


def _html_document(markdown_text, title="Segmentation report"):
    """The self-contained, styled HTML page for a Markdown report (used by the file writer and the
    web UI alike)."""
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_html.escape(title)}</title><style>{_REPORT_CSS}</style></head>"
            f"<body><main>{_markdown_to_html(markdown_text)}</main></body></html>")


def write_html_report(markdown_text, path, title="Segmentation report"):
    """Render the Markdown report as a self-contained, styled HTML page anyone can open."""
    Path(path).write_text(_html_document(markdown_text, title), encoding="utf-8")
    return str(path)


# =====================================================================================
# Typing tool: assign NEW respondents to segments, and measure how reliably that can be done
# =====================================================================================
def segment_centres(X, labels, cfg=None):
    """The centre of each segment, in whatever space the clustering happened in.

    The mean on the numeric path, because that is what k-means puts at the centre. On the
    mixed-type path it is the k-prototypes prototype instead — median, mode and nearest-rank level
    by variable type — because a mean of brand codes is not an answer anybody gave.

    An empty segment keeps a row of NaN rather than being dropped, so row i is always segment i
    and nothing downstream has to renumber.
    """
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    k = int(labels.max()) + 1 if len(labels) else 0
    gower = _is_gower(cfg)
    out = []
    for c in range(k):
        pts = X[labels == c]
        if not len(pts):
            out.append(np.full(X.shape[1], np.nan))
        else:
            out.append(kprototypes._update(pts, cfg.gower_spec) if gower else pts.mean(0))
    return np.array(out)


def shadow_values(X, centroids, cfg=None):
    """Leisch's shadow value for every respondent, plus who their two nearest segments are.

        s(x) = 2 d(x, closest) / [ d(x, closest) + d(x, second closest) ]

    Near 0 the respondent sits squarely inside their own segment; near 1 they are stranded
    halfway between two and could plausibly belong to either. Leisch introduces it in
    *Neighborhood Graphs, Stripes and Shadow Plots* (2010) and says outright that it is "similar
    both in spirit and interpretation to the well known silhouette plots".

    Used here in preference to the silhouette for one practical reason. A silhouette needs the
    distance from every respondent to every other — O(n^2) — so it had to be computed on a sample
    of 6,000 and everybody else left blank, which put holes in the exported file exactly on the
    large studies where per-person fit matters most. A shadow value needs only the distance to
    two centroids, O(n·k), so every respondent gets a real number at any sample size.

    Returns (shadow, closest, second_closest). A degenerate case — one segment, or a respondent
    sitting exactly on top of two centroids — yields 0.0 rather than a division by zero: someone
    at zero distance from their centroid fits perfectly, which is what 0 means.
    """
    X = np.asarray(X, float)
    C = np.asarray(centroids, float)
    if C.ndim != 2 or len(C) < 2:
        return np.zeros(len(X)), np.zeros(len(X), int), np.zeros(len(X), int)
    # Whichever distance the segmentation itself minimises: a shadow value compares a
    # respondent's two nearest centres, so it is only meaningful under the metric that decided
    # which centre was nearest in the first place.
    if _is_gower(cfg):
        d = kprototypes.gower_distances(X, C, cfg.gower_spec)
    else:
        d = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(2))     # (n, k)
    order = np.argsort(d, axis=1)
    closest, second = order[:, 0], order[:, 1]
    rows = np.arange(len(X))
    d1, d2 = d[rows, closest], d[rows, second]
    total = d1 + d2
    return np.where(total > 0, 2 * d1 / np.where(total > 0, total, 1.0), 0.0), closest, second


def segment_neighbours(shadow, closest, second, k):
    """How close each pair of segments sits, from the respondents caught between them.

    Leisch's s_ij: average the shadow value over everyone whose closest segment is i and whose
    second-closest is j. A high value means the people on i's edge are barely nearer to i than to
    j — the two segments are neighbours and their boundary is soft.

    This answers a question the tool could not previously ask. It reports how distinct each
    segment is from the average respondent; it never said WHICH TWO SEGMENTS ARE NEARLY THE SAME,
    which is what somebody needs before committing to five campaigns rather than three.

    Returns a list of (i, j, mean_shadow, n_between) sorted worst-first, covering only pairs that
    actually have respondents stranded between them.
    """
    out = []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            between = (closest == i) & (second == j)
            n = int(between.sum())
            if n:
                out.append((i, j, float(shadow[between].mean()), n))
    return sorted(out, key=lambda r: -r[2])


def _per_respondent_fit(X, labels, centroids=None, cfg=None):
    """How well each respondent sits in the segment they were given: 1 = squarely inside, 0 =
    stranded between two. This is 1 - the shadow value, flipped so that higher reads as better
    in the exported file, which is what a non-analyst expects of a column called `fit`.

    Falls back to the silhouette when no centroids are available (the latent-class path has
    probabilities rather than centres), and to NaN if neither can be computed. A blank is honest;
    a fabricated 0.5 is not.
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return np.full(len(labels), np.nan)
    if centroids is not None:
        shadow, _, _ = shadow_values(X, centroids, cfg)
        return np.round(1.0 - shadow, 3)
    try:
        from sklearn.metrics import silhouette_samples
        if len(X) <= 6000:
            return np.round(silhouette_samples(X, labels), 3)
        rng = np.random.default_rng(0)
        take = np.sort(rng.choice(len(X), 6000, replace=False))
        out = np.full(len(labels), np.nan)
        out[take] = np.round(silhouette_samples(X[take], labels[take]), 3)
        return out
    except Exception as e:
        print(f"NOTE: could not score how well each respondent fits ({type(e).__name__}: {e}).")
        return np.full(len(labels), np.nan)


def typing_tool(arr_raw, labels, cfg):
    """Build and cross-validate a 'typing tool' — the rule that assigns a NEW respondent to a
    segment from their item scores. This is the operational payoff of a segmentation (Mind
    Genomics builds one as a matter of course): you discover segments once, then type every
    future respondent. It returns two things:

      1. the exportable rule: per-segment centroids in scaled space plus the scaling parameters,
         which classify_new() applies to fresh respondents; and
      2. a leakage-free estimate of how reliably that rule reproduces the segmentation on
         held-out respondents (stratified k-fold, with the scaling refit inside each fold).

    Nearest-centroid in scaled space is used deliberately: it is exactly how k-means itself
    assigns a point (to the closest centroid), it serialises to a few numbers with no pickled
    model, and it is a reasonable portable approximation of a mixture's posterior assignment for
    the gmm method.

    Read the cross-validated accuracy as an OPERATIONAL property — how consistently a new
    respondent can be typed into the same segment — NOT as proof the segments are real. A k-means
    partition of pure noise is still highly classifiable, because its Voronoi cells are compact
    convex regions; so a high typing accuracy is necessary but not sufficient for real structure.
    Whether the structure is real is decided upstream by the Hopkins statistic, prediction
    strength, and per-segment Jaccard. A very low typing accuracy is still informative on its own
    (the assignment boundary is not even geometrically stable).
    """
    from sklearn.model_selection import StratifiedKFold
    labels = np.asarray(labels)
    classes = np.unique(labels)
    k = len(classes); n = len(labels)
    counts = np.array([(labels == c).sum() for c in classes])
    min_class = int(counts.min())

    # On the mixed-type path the rule is the same idea in a different metric: prototypes instead
    # of centroids, Gower instead of Euclidean, and the Gower spec standing in for the scaling
    # parameters. It has to be the distance the segmentation itself used, or the exported rule
    # would put new people somewhere the analysis never would.
    gower = _is_gower(cfg)

    def _fit_space(arr):
        if gower:
            spec = kprototypes.fit_spec(arr, list(cfg.gower_spec.kinds))
            return kprototypes.encode(arr, spec), spec
        return _scale_fit(arr, cfg.scaling)

    def _apply_space(arr, params):
        return kprototypes.encode(arr, params) if gower else _scale_apply(arr, params)

    def _centroids(Xs, y, params):
        if gower:
            return np.vstack([kprototypes._update(Xs[y == c], params) for c in classes])
        return np.vstack([Xs[y == c].mean(0) for c in classes])

    def _assign(Xte, cents, params):
        if gower:
            return classes[kprototypes.gower_distances(Xte, cents, params).argmin(1)]
        d = ((Xte[:, None, :] - cents[None, :, :]) ** 2).sum(2)
        return classes[d.argmin(1)]

    if min_class < 2 or n < 10:                      # too little data for honest cross-validation
        cv_acc = float("nan"); recalls = {int(c): float("nan") for c in classes}
    else:
        n_splits = int(min(5, min_class))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        correct = np.zeros(k); total = np.zeros(k); n_ok = 0
        for tr, te in skf.split(arr_raw, labels):
            Xtr, p = _fit_space(arr_raw[tr])       # refit the space on the train fold only
            pred = _assign(_apply_space(arr_raw[te], p), _centroids(Xtr, labels[tr], p), p)
            n_ok += int((pred == labels[te]).sum())
            for i, c in enumerate(classes):
                m = labels[te] == c
                total[i] += int(m.sum()); correct[i] += int((pred[m] == c).sum())
        cv_acc = float(n_ok / n)
        recalls = {int(c): (float(correct[i] / total[i]) if total[i] else float("nan"))
                   for i, c in enumerate(classes)}

    Xs_full, params = _fit_space(arr_raw)                      # the exported rule, fit on all data
    return {"cv_accuracy": cv_acc,
            "baseline_majority": float(counts.max() / n),      # "always guess the biggest segment"
            "per_segment_recall": recalls,
            "scaled_centroids": _centroids(Xs_full, labels, params),
            "scale_params": ({"scaling": "gower", "gower_spec": params.to_json()}
                             if gower else params),
            "classes": [int(c) for c in classes]}


def _pick_id_column(df, items, id_col=None):
    """Which column identifies each person in a file of NEW respondents. Without one the scored list
    is unusable — you cannot target people you cannot name — so fall back to the same id heuristics
    the auto-detector uses rather than silently returning unlabelled rows."""
    if id_col and id_col in df.columns:
        return id_col
    for c in df.columns:
        if c in items:
            continue
        if _looks_like_id(df[c], str(c).lower().strip()) or _looks_like_index(df[c]):
            return c
    return "id" if "id" in df.columns else None


def _training_centre(params):
    """Each item's centre in the ORIGINAL study, on the raw answer scale.

    This is what a skipped answer should fall back to when scoring new people. Imputing from the
    new batch's own mean instead makes scoring depend on who else happens to be in the upload —
    the same person comes out with a different segment or confidence in a batch of 1 than in a
    batch of 500 — which defeats the point of a fixed typing rule. Row-local scalings (ipsative,
    none) fit no centre, so they have nothing to offer here and return None."""
    s = params.get("scaling")
    if s == "gower":
        return kprototypes.GowerSpec.from_json(params["gower_spec"]).centre
    if s == "standardize":
        return np.asarray(params["mean"], float)
    if s == "robust":
        return np.asarray(params["median"], float)
    if s == "range":
        return np.asarray(params["lo"], float) + np.asarray(params["range"], float) / 2.0
    return None


def _warn_off_scale(out):
    """Say it out loud when a follow-up file contains answers the original study never saw.

    The count is in the exported CSV either way, but somebody typing a few hundred people from the
    command line will not open it, and a "no answer" code of 99 quietly moves people into the
    wrong segment — measured, 35 of 60 affected respondents, with the agreement against the truth
    falling from 0.967 to 0.593.
    """
    col = "answers_off_the_original_scale"
    if col not in out or not int((out[col] > 0).sum()):
        return
    n = int((out[col] > 0).sum())
    print(f"\nNOTE: {n} of {len(out)} respondent(s) gave at least one answer outside the scale the "
          "original study used — often a 'no answer' code such as 99 or -99. They have been scored "
          f"anyway, and their '{col}' count and low confidence say so. Check that column before "
          "using this list for anything.")


def _answers_off_scale(sub, params, items):
    """How many of each respondent's answers fall outside the range the study itself covered.

    Only meaningful for the scalings that record the observed range — range scaling, which is the
    default and what the app uses. The others fit a centre and a spread rather than bounds, so
    there is nothing here to compare against and this returns None rather than a guess.
    """
    if params.get("scaling") != "range":
        return None
    lo = np.asarray(params.get("lo", []), float)
    rng = np.asarray(params.get("range", []), float)
    if lo.shape != (len(items),) or rng.shape != (len(items),):
        return None
    vals = sub.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    tol = 1e-9 + 0.01 * np.where(rng > 0, rng, 1.0)      # a hair's grace for float rounding
    outside = (vals < lo - tol) | (vals > lo + rng + tol)
    return pd.Series(np.where(np.isnan(vals), False, outside).sum(1).astype(int), index=sub.index)


def classify_new(rule, df, id_col=None):
    """Assign new respondents to segments using a saved typing rule (parsed typing_rule.json).
    `df` must contain the rule's item columns. Returns the respondent id (from id_col, or a column
    named "id" if present), the assigned segment, and a confidence in [1/k, 1] (inverse-distance
    share on the winning centroid: 1 = sits on a centroid, ~1/k = equidistant from all)."""
    items = rule["items"]
    missing = [c for c in items if c not in df.columns]
    if missing:
        raise ValueError(f"New data is missing required item column(s): {missing}")
    # New responses arrive exactly as the survey exported them, so any item the original run
    # recoded from words ("Strongly agree") to 1-5 is still text here. Recode it the same way
    # before scoring, or the rule cannot be applied to a real fresh export at all.
    kinds = rule.get("var_kinds") or {}
    level_labels = rule.get("level_labels") or {}
    sub = df[items].copy()
    for c in items:
        if kinds.get(c) == kprototypes.NOMINAL:
            # A pick-any answer arrives as the word the respondent chose. Map it back to the code
            # the study used; anything the study never saw becomes UNSEEN, which sits a full
            # mismatch from every known answer rather than being quietly treated as one of them.
            back = {name: float(code) for code, name in level_labels.get(c, {}).items()}
            sub[c] = sub[c].astype(str).map(back).fillna(kprototypes.UNSEEN).astype(float)
            continue
        if not pd.api.types.is_numeric_dtype(sub[c]):
            # The rule already established this item is a rating scale; applying it must not
            # depend on how many distinct answers this particular batch happens to contain.
            rec = _apply_likert(sub[c])
            if rec is None:
                raise ValueError(f"_UNSCORABLE_ITEM:{c}")
            sub[c] = rec
    arr = sub.to_numpy(float)
    # UNSEEN is -inf by construction and must survive; only genuine infinities are missing data.
    nominal_at = [i for i, c in enumerate(items) if kinds.get(c) == kprototypes.NOMINAL]
    bad = np.isinf(arr)
    bad[:, nominal_at] = False
    arr = np.where(bad, np.nan, arr)
    if np.isnan(arr).any():
        centre = _training_centre(rule["scale_params"])
        if centre is None:                   # row-local scaling — no fitted centre exists to use
            seen = (~np.isnan(arr)).sum(0)   # column means without nanmean's empty-slice warning
            centre = np.where(seen > 0, np.nansum(arr, 0) / np.maximum(seen, 1), 0.0)
        arr = np.where(np.isnan(arr), centre, arr)
    classes = np.asarray(rule["classes"])
    cents = np.asarray(rule["scaled_centroids"], float)
    if rule["scale_params"].get("scaling") == "gower":
        spec = kprototypes.GowerSpec.from_json(rule["scale_params"]["gower_spec"])
        d = kprototypes.gower_distances(kprototypes.encode(arr, spec), cents, spec)
    else:
        d = np.sqrt(((_scale_apply(arr, rule["scale_params"])[:, None, :]
                      - cents[None, :, :]) ** 2).sum(2))
    inv = 1.0 / (d + 1e-9)
    out = pd.DataFrame({"segment": classes[d.argmin(1)],
                        "confidence": (inv.max(1) / inv.sum(1)).round(3)})
    # Answers outside the scale the study was built on. Survey exports routinely code "no answer"
    # as 99, 999 or -99, and such a value is not rejected by anything above: it is scaled with the
    # study's own parameters, lands far outside the space, and drags the respondent to whichever
    # segment is extreme on that item. Measured on a 250-person follow-up with 99 in one question,
    # 35 of the 60 affected people were put in the wrong segment and the agreement with the truth
    # fell from 0.967 to 0.593.
    #
    # Their confidence does drop sharply (0.34 against 0.72), so the signal was already there —
    # what was missing is any reason for the reader to go looking. Counted per respondent so a
    # list can be filtered on it, rather than corrected silently: whether 99 means "no answer" or
    # is a real value is a fact about the questionnaire, not about the data.
    off = _answers_off_scale(sub[items], rule["scale_params"], items)
    if off is not None:
        out["answers_off_the_original_scale"] = off
    idc = _pick_id_column(df, items, id_col)
    if idc:
        out.insert(0, idc, df[idc].to_numpy())
    return out


# =====================================================================================
# Latent Class Analysis (categorical / multiple-choice data)
# =====================================================================================
# For CATEGORICAL inputs (agree/disagree, pick-any, multiple choice) the right model is not
# k-means on numeric codes but a latent-class model: the classic Lazarsfeld-Goodman / Wedel-
# Kamakura finite mixture under LOCAL INDEPENDENCE — within a class, the items are independent,
# and each item j in class c has its own category distribution theta[c][j]. Fit by EM with many
# restarts (Steinley's local-optima warning applies here too), number of classes chosen by BIC
# and ICL, validated with the same label-based stability the k-means path uses. This is a
# different modelling FAMILY from the Gaussian mixture, which assumes continuous data.
def _onehot(labels, K):
    Z = np.zeros((len(labels), K)); Z[np.arange(len(labels)), labels] = 1.0
    return Z


def _lca_logjoint(Xcat, log_weights, log_theta):
    """log P(class=c, x_i) for every i, c -> (n, K), under local independence."""
    lp = np.tile(log_weights, (Xcat.shape[0], 1))       # (n, K) prior
    for j, lth in enumerate(log_theta):                 # lth: (K, L_j)
        lp = lp + lth[:, Xcat[:, j]].T                  # add log theta[c][j][x_ij]
    return lp


def _lca_mstep(Xcat, level_counts, resp, alpha):
    """Weighted category frequencies with additive (Laplace) smoothing alpha to keep the model off
    the 0/1 boundary — a well-known degeneracy of latent-class EM."""
    n, K = resp.shape
    weights = resp.sum(0) / n
    theta = []
    for j, L in enumerate(level_counts):
        cnt = np.empty((K, L))
        for lvl in range(L):
            cnt[:, lvl] = resp[Xcat[:, j] == lvl].sum(0)
        cnt += alpha
        theta.append(cnt / cnt.sum(1, keepdims=True))
    return weights, theta


def _lca_fit_once(Xcat, level_counts, K, rng, max_iter=200, tol=1e-6, alpha=0.1):
    resp = _onehot(rng.integers(0, K, len(Xcat)), K)    # random hard start
    weights, theta = _lca_mstep(Xcat, level_counts, resp, alpha)
    prev = -np.inf
    for _ in range(max_iter):
        lj = _lca_logjoint(Xcat, np.log(weights), [np.log(t) for t in theta])
        ll_i = logsumexp(lj, axis=1)
        loglik = float(ll_i.sum())
        resp = np.exp(lj - ll_i[:, None])
        weights, theta = _lca_mstep(Xcat, level_counts, resp, alpha)
        if abs(loglik - prev) < tol * (abs(prev) + 1):
            break
        prev = loglik
    n_params = (K - 1) + sum(K * (L - 1) for L in level_counts)
    return {"weights": weights, "theta": theta, "loglik": loglik, "resp": resp,
            "labels": resp.argmax(1), "n_params": n_params, "level_counts": level_counts}


def _lca_fit(Xcat, level_counts, K, n_init, seed):
    """Best of n_init EM restarts (highest log-likelihood)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(max(1, n_init)):
        m = _lca_fit_once(Xcat, level_counts, K, rng)
        if best is None or m["loglik"] > best["loglik"]:
            best = m
    return best


def _lca_predict(model, Xcat):
    return _lca_logjoint(Xcat, np.log(model["weights"]), [np.log(t) for t in model["theta"]]).argmax(1)


def _lca_entropy(resp):
    r = np.clip(resp, 1e-12, 1.0)
    return float(-(resp * np.log(r)).sum())


def latent_class_select(Xcat, level_counts, cfg):
    """Fit k_min..k_max and score each by log-likelihood, BIC, ICL, normalized entropy, and a
    bootstrap replication stability (Adjusted Rand Index). Lower BIC/ICL is better; ICL adds an
    entropy penalty and is the Biernacki-Celeux-Govaert criterion for how many latent classes."""
    n = len(Xcat)
    rows = []
    for K in range(cfg.k_min, cfg.k_max + 1):
        m = _lca_fit(Xcat, level_counts, K, cfg.n_init_search, cfg.random_state)
        bic = -2 * m["loglik"] + m["n_params"] * np.log(n)
        icl = bic + 2 * _lca_entropy(m["resp"])
        norm_ent = _lca_entropy(m["resp"]) / (n * np.log(K)) if K > 1 else 0.0
        # bootstrap replication stability: does the same partition re-emerge on resamples?
        rng = np.random.default_rng(cfg.random_state + 100 + K)
        ref = m["labels"]
        aris = []
        for _ in range(max(5, cfg.stability_B // 2)):
            idx = rng.choice(n, int(cfg.stability_frac * n), replace=False)
            mb = _lca_fit(Xcat[idx], level_counts, K, max(3, cfg.n_init_search // 2),
                          int(rng.integers(1_000_000_000)))
            aris.append(adjusted_rand_score(ref[idx], mb["labels"]))
        rows.append({"k": K, "loglik": round(m["loglik"], 1), "n_params": m["n_params"],
                     "BIC": round(bic, 1), "ICL": round(icl, 1),
                     "normalized_entropy": round(norm_ent, 3),
                     "stability_ARI": round(float(np.mean(aris)), 3)})
    return pd.DataFrame(rows)


def recommend_k_lca(diag, cfg):
    """Weighted vote: ICL (designed for choosing the number of latent classes) weighted most, then
    BIC, then replication stability above the cutoff; ties broken toward the smaller, more
    interpretable solution."""
    tally = {int(k): 0.0 for k in diag["k"]}
    tally[int(diag.loc[diag["ICL"].idxmin(), "k"])] += 2.0
    tally[int(diag.loc[diag["BIC"].idxmin(), "k"])] += 1.0
    stable = diag[diag["stability_ARI"] >= cfg.stability_cutoff]
    if len(stable):
        tally[int(stable.loc[stable["k"].idxmin(), "k"])] += 1.0
    best = max(tally.values())
    pick = min(k for k, s in tally.items() if s == best)
    rationale = (f"Recommended number of latent classes: **{pick}**.\n\nChosen by a weighted vote "
                 "of the ICL (weighted most, since it is designed for choosing the number of "
                 "latent classes and penalises fuzzy overlapping classes), the BIC, and bootstrap "
                 "replication stability, preferring the smaller solution on a tie. BIC and ICL are "
                 "the model-based criteria a heuristic method lacks; read them alongside the "
                 "stability column, which says whether the same classes re-emerge on resamples.")
    return pick, rationale


def lca_bootstrap_jaccard(Xcat, level_counts, labels, k, cfg):
    """Per-class bootstrap Jaccard stability (Hennig's clusterboot idea), for the latent-class
    partition: how reliably does each class re-appear under resampling?"""
    rng = np.random.default_rng(cfg.random_state + 3)
    n = len(Xcat)
    base = {c: set(np.where(labels == c)[0]) for c in range(k)}
    best = {c: [] for c in range(k)}
    for _ in range(cfg.jaccard_B):
        idx = rng.integers(0, n, n)                     # bootstrap resample (with replacement)
        m = _lca_fit(Xcat[idx], level_counts, k, max(3, cfg.n_init_search // 2),
                     int(rng.integers(1_000_000_000)))
        present = np.unique(idx)
        boot_lab = _lca_predict(m, Xcat[present])
        boot_sets = [set(present[boot_lab == b]) for b in range(k)]
        for c, cset in base.items():
            cp = cset & set(present)
            if not cp:
                best[c].append(0.0); continue
            best[c].append(max((len(cp & bs) / len(cp | bs)) if (cp | bs) else 0.0
                               for bs in boot_sets))
    return {c: float(np.mean(v)) for c, v in best.items()}


def load_categorical(path, id_col, item_cols):
    """Read a CSV of CATEGORICAL responses. Each item column is factorised to integer levels
    0..L-1 (missing becomes its own 'MISSING' level, so nothing is silently dropped). Returns the
    integer matrix, per-item level counts, item names, the original level labels, and ids."""
    df = _read_table(path)
    ids = df[id_col].astype(str).to_numpy() if id_col and id_col in df else \
        np.array([f"r{i}" for i in range(len(df))])
    if item_cols is None:
        item_cols = [c for c in df.columns if c != id_col]
    codes, level_labels, names = [], [], []
    for c in item_cols:
        s = df[c].astype("object").where(df[c].notna(), "MISSING")
        cat = pd.Categorical(s)
        if len(cat.categories) >= 2:                    # drop constant items (no information)
            codes.append(np.asarray(cat.codes)); level_labels.append(list(cat.categories))
            names.append(c)
        else:
            print(f"Dropping constant item (one level): {c}")
    if len(names) < 2:
        raise ValueError("Need at least two categorical item columns (each with 2+ levels).")
    Xcat = np.column_stack(codes).astype(int)
    return Xcat, [len(l) for l in level_labels], names, level_labels, ids


def profile_demographics(labels, ids, demo_source, id_col, unit="segment"):
    """Chi-square + Benjamini-Hochberg FDR profiling of the groups against background traits. Shared
    by the k-means and latent-class paths. Demographics DESCRIBE the groups, never form them."""
    demo = _read_table(demo_source)
    if id_col and id_col in demo:
        seg = pd.Series(labels, index=np.asarray(ids), name=unit)
        demo = demo.set_index(demo[id_col].astype(str)).join(seg, how="inner")
    else:
        demo = demo.copy(); demo[unit] = labels
    pvals = {}
    for col in [c for c in demo.columns if c not in (unit, id_col)]:
        # Non-numeric or few-valued columns are profiled as categories. Tested by dtype
        # rather than against `object` for the pandas 3 reason above.
        if not pd.api.types.is_numeric_dtype(demo[col]) or demo[col].nunique() <= 12:
            ct = pd.crosstab(demo[unit], demo[col])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                _, p, _, _ = stats.chi2_contingency(ct)
                pvals[col] = p
    sig = _fdr_bh(pvals) if pvals else {}
    L = [f"## Profiling the {_plural(unit)} against demographics "
         "(chi-square, Benjamini-Hochberg FDR-corrected; profiling only)\n"]
    for col, p in sorted(pvals.items(), key=lambda kv: kv[1]):
        L.append(f"- **{col}**: chi-square p = {p:.4f}"
                 + (f"  <- differs by {unit} (survives FDR correction)" if sig.get(col) else ""))
    L.append(f"\n(Demographics describe {_plural(unit)}; they do not define them.)")
    return "\n".join(L)


def _lca_supervised(Xcat, level_counts, labels, classes, alpha=0.5):
    """Class weights and class-conditional item probabilities estimated GIVEN fixed labels (a
    latent-class / naive-Bayes typing model aligned to the discovered classes, so there is no label
    switching to confuse the cross-validation)."""
    pi = np.array([(labels == c).mean() for c in classes])
    theta = []
    for j, L in enumerate(level_counts):
        cnt = np.zeros((len(classes), L))
        for ci, c in enumerate(classes):
            xc = Xcat[labels == c, j]
            for lvl in range(L):
                cnt[ci, lvl] = (xc == lvl).sum()
        cnt += alpha
        theta.append(cnt / cnt.sum(1, keepdims=True))
    return pi, theta


def _lca_typing_predict(Xcat, pi, theta, classes):
    lp = np.tile(np.log(pi), (len(Xcat), 1))
    for j, th in enumerate(theta):
        lp = lp + np.log(th)[:, Xcat[:, j]].T
    return np.asarray(classes)[lp.argmax(1)], lp


def lca_typing_tool(Xcat, level_counts, labels, cfg):
    """Typing tool for the categorical path: a latent-class classifier that assigns a NEW respondent
    to a class from their answers, plus a leakage-free cross-validated estimate of how reliably that
    can be done. Parallels the k-means typing tool; read the accuracy as operational, not proof the
    classes are real."""
    from sklearn.model_selection import StratifiedKFold
    classes = np.unique(labels); k = len(classes); n = len(labels)
    counts = np.array([(labels == c).sum() for c in classes]); min_class = int(counts.min())
    if min_class < 2 or n < 10:
        cv_acc = float("nan"); recalls = {int(c): float("nan") for c in classes}
    else:
        skf = StratifiedKFold(int(min(5, min_class)), shuffle=True, random_state=cfg.random_state)
        correct = np.zeros(k); total = np.zeros(k); n_ok = 0
        for tr, te in skf.split(Xcat, labels):
            pi, theta = _lca_supervised(Xcat[tr], level_counts, labels[tr], classes)
            pred, _ = _lca_typing_predict(Xcat[te], pi, theta, classes)
            n_ok += int((pred == labels[te]).sum())
            for i, c in enumerate(classes):
                m = labels[te] == c
                total[i] += int(m.sum()); correct[i] += int((pred[m] == c).sum())
        cv_acc = float(n_ok / n)
        recalls = {int(c): (float(correct[i] / total[i]) if total[i] else float("nan"))
                   for i, c in enumerate(classes)}
    pi, theta = _lca_supervised(Xcat, level_counts, labels, classes)
    return {"cv_accuracy": cv_acc, "baseline_majority": float(counts.max() / n),
            "per_segment_recall": recalls, "weights": pi.tolist(),
            "theta": [t.tolist() for t in theta], "classes": [int(c) for c in classes],
            "level_counts": list(level_counts)}


def classify_new_lca(rule, df, id_col=None):
    """Assign new respondents to latent classes using a saved categorical typing rule. New answers
    are mapped to the training categories; an unseen answer simply does not vote for that item."""
    items = rule["items"]
    missing = [c for c in items if c not in df.columns]
    if missing:
        raise ValueError(f"New data is missing required item column(s): {missing}")
    classes = np.asarray(rule["classes"]); pi = np.asarray(rule["weights"], float)
    theta = [np.asarray(t, float) for t in rule["theta"]]
    codemaps = [{str(lbl): i for i, lbl in enumerate(labels)} for labels in rule["level_labels"]]
    lp = np.tile(np.log(pi), (len(df), 1))
    for j, item in enumerate(items):
        col = df[item].astype("object")
        logth = np.log(theta[j])
        for r, v in enumerate(col):
            code = codemaps[j].get(str(v))
            if code is not None:                        # unseen categories abstain (no contribution)
                lp[r] += logth[:, code]
    post = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))   # proper posterior probability
    out = pd.DataFrame({"segment": classes[lp.argmax(1)], "confidence": post.max(1).round(3)})
    idc = _pick_id_column(df, items, id_col)
    if idc:
        out.insert(0, idc, df[idc].to_numpy())
    return out


def lca_split_half(Xcat, level_counts, k, cfg):
    """Would a fresh sample find these same classes? The categorical answer to the question the
    numeric path answers with `split_half_replication`.

    Fit the model on each half of the sample independently, then have both label the SAME held-out
    half and compare. The k-means path has reported this since the beginning; the categorical path
    reported nothing of the kind, so a survey made entirely of multiple-choice questions arrived
    with no answer at all to "does this reproduce", which is the single question the confidence
    light is built from.
    """
    rng = np.random.default_rng(cfg.random_state + 5)
    idx = rng.permutation(len(Xcat)); h = len(Xcat) // 2
    a, b = idx[:h], idx[h:]
    if min(len(a), len(b)) < max(2 * k, 10):
        return None                       # too little data either side to mean anything
    try:
        ma = _lca_fit(Xcat[a], level_counts, k, max(3, cfg.n_init_search // 2), 1)
        mb = _lca_fit(Xcat[b], level_counts, k, max(3, cfg.n_init_search // 2), 2)
        return float(adjusted_rand_score(_lca_predict(ma, Xcat[b]), _lca_predict(mb, Xcat[b])))
    except Exception:
        return None


def latent_class_report(diag, rec_k, rationale, model, jaccard, names, level_labels, labels, cfg,
                        typing=None, weighted_share=None, neighbours=None, split_half=None,
                        median_shadow=None):
    method = "Latent Class Analysis (categorical, local-independence finite mixture)"
    n = len(labels)
    weights = model["weights"]; theta = model["theta"]
    norm_ent = _lca_entropy(model["resp"]) / (n * np.log(rec_k)) if rec_k > 1 else 0.0
    mean_top = float(model["resp"].max(1).mean())
    _cn, _cw = [], []
    for c in range(rec_k):
        d = sorted(((float(theta[j][c].max()), names[j], str(level_labels[j][int(theta[j][c].argmax())]))
                    for j in range(len(names))), reverse=True)
        _cn.append(" + ".join(f"{_short_label(nm)}={lv}" for _, nm, lv in d[:2]))
        _cw.append(", ".join(f"{_short_label(nm)}={lv}" for _, nm, lv in d[:2]))
    _shares = [float((labels == c).mean()) for c in range(rec_k)]
    _repro = float(diag.loc[diag["k"] == rec_k, "stability_ARI"].iloc[0])
    _picks = [int(diag.loc[diag["BIC"].idxmin(), "k"]), int(diag.loc[diag["ICL"].idxmin(), "k"]),
              int(diag.loc[diag["stability_ARI"].idxmax(), "k"])]
    _kagree = float(np.mean([abs(p - rec_k) <= 1 for p in _picks]))
    L = ["# Latent class segmentation report\n",
         executive_summary(n, _cn, _shares, _cw, min(jaccard.values()), _repro, unit="class",
                           k_agreement=_kagree),
         f"Respondents clustered with **{method}** on {len(names)} categorical items; the final "
         f"fit used {cfg.n_init_final} EM restarts. Search range: k = {cfg.k_min} to {cfg.k_max}.\n",
         "## Choosing the number of classes\n", rationale, "\n",
         "**How to read this table.** **BIC** and **ICL** are the model-based criteria (lower is "
         "better); ICL adds an entropy penalty for fuzzy, overlapping classes and is the criterion "
         "designed for choosing the number of latent classes. **stability_ARI** says whether the "
         "same classes re-emerge on resamples (near 1 is good). **normalized_entropy** is 0 when "
         "class membership is crisp and 1 when it is maximally fuzzy.\n",
         _md(diag), "\n",
         f"Classification certainty: mean top posterior = **{mean_top:.2f}**, normalized entropy = "
         f"**{norm_ent:.2f}** (0 = crisp, 1 = fuzzy).\n",
         # Everything below here the categorical path either already computed and discarded, or
         # could answer and did not. A survey made entirely of multiple-choice questions used to
         # arrive with two of the eleven pieces of evidence the numeric path gives, and nothing
         # saying so.
         (None if split_half is None else
          f"\n**Would a fresh sample find the same classes?** Split-half replication (Adjusted "
          f"Rand Index): **{split_half:.3f}** "
          # Read against the SAME 0.6 threshold segmentation_kind uses in the sentence below.
          # With a lower band here, 0.577 was reported as "partly reproduces" immediately above
          # "the division does not survive being repeated" — two adjacent sentences disagreeing
          # about one number, which is the fault this report has been cleaned of elsewhere.
          + ("(reproduces well)." if split_half >= 0.7 else
             "(reproduces, though not strongly)." if split_half >= 0.6 else
             "(does NOT reproduce — re-running on half the sample gives a different answer, which "
             "is what happens when the data has no real classes in it).")),
         # segmentation_kind returns (kind, gloss) — use the gloss it already writes rather than
         # keeping a second copy of the same three sentences, which would drift.
         (None if split_half is None else
          "\n**This is a {} segmentation** — {}".format(
              *segmentation_kind(False, split_half, median_shadow))),
         neighbours_paragraph(neighbours, _cn, len(_cn)),
         "\n**Who fits, person by person.** `segment_assignments.csv` carries a `fit` column: how "
         "clearly each respondent belongs to the class they were given (about 1 = squarely inside, "
         "about 0 = poised between two). Before spending money on a list, drop the low scores "
         "rather than contacting people who matched nothing in particular.\n"]
    rec_stab = float(diag.loc[diag["k"] == rec_k, "stability_ARI"].iloc[0])
    if rec_stab < cfg.stability_cutoff:
        L.append(f"\n> WARNING: replication stability at k = {rec_k} is only {rec_stab:.2f} (below "
                 f"{cfg.stability_cutoff:.2f}). The classes do not re-emerge reliably on resamples "
                 "— the data may contain little real latent structure. Treat the classes as "
                 "constructed, not discovered, and be cautious.")
    L.append("\n**Per-class bootstrap stability (Hennig's Jaccard).** Which classes are real:\n")
    jac = pd.DataFrame({"class": [f"Class {c}" for c in jaccard],
                        "mean_Jaccard": [round(v, 3) for v in jaccard.values()],
                        "reading": [jaccard_reading(v) for v in jaccard.values()]})
    L.append(_md(jac))
    if (np.array(list(jaccard.values())) < 0.6).any():
        L.append("\n> WARNING: one or more classes fall below 0.60 Jaccard — not trustworthy as "
                 "distinct classes. Consider fewer classes.")
    if typing is not None:
        L.append("\n" + _typing_line(typing, "latent_class_typing_rule.json", unit="class"))
    sizes = pd.DataFrame({"class": [f"Class {c}" for c in range(rec_k)],
                          "n": [int((labels == c).sum()) for c in range(rec_k)],
                          "share": [round((labels == c).mean(), 3) for c in range(rec_k)],
                          "model_weight": [round(float(w), 3) for w in weights]})
    if weighted_share is not None:
        sizes["population_share"] = [round(weighted_share.get(c, 0.0), 3) for c in range(rec_k)]
    L += ["\n## Class sizes\n",
          ("The **population_share** column reweights each respondent by your survey weight to "
           "estimate each class's size in the whole population.\n" if weighted_share is not None else None),
          _md(sizes), "\n## The mind-sets (what defines each class)\n"]
    for c in range(rec_k):
        defs = []
        for j in range(len(names)):
            probs = theta[j][c]                         # (L_j,)
            m = int(probs.argmax())
            defs.append((float(probs[m]), names[j], str(level_labels[j][m]), j, m))
        defs.sort(reverse=True)
        top = defs[:cfg.top_items]
        name = " + ".join(f"{_short_label(nm)}={lv}" for _, nm, lv, _, _ in top[:2])
        L.append(f"**Class {c}** — suggested name: *{name}*")
        L.append("  - Most likely answers: " +
                 ", ".join(f"{nm}={lv} (p={p:.2f})" for p, nm, lv, _, _ in top) + "\n")
    L += ["\n---\n**Methodology.** Latent Class Analysis: a finite mixture of categorical "
          "distributions under local independence (Lazarsfeld & Goodman; Wedel & Kamakura), fit by "
          "EM with many restarts (Steinley on local optima). Number of classes chosen by BIC and "
          "the entropy-penalised ICL (Biernacki, Celeux & Govaert 2000), cross-checked with "
          "bootstrap replication stability (Dolnicar & Leisch) and per-class Jaccard (Hennig 2007). "
          "Rename the auto-suggested names before shipping.",
          f"\n*Generated by segment_kmeans version {__version__}.*"]
    return "\n".join(x for x in L if x is not None)


class LatentClassSegmenter:
    """Categorical counterpart to Segmenter: Latent Class Analysis for multiple-choice / agree-
    disagree survey data. Same validation philosophy (stability first), different model family."""
    def __init__(self, cfg: SegmentationConfig | None = None):
        self.cfg = cfg or SegmentationConfig()

    def run(self, path, id_col=None, item_cols=None, force_k=None, outdir=None,
            demographics=None, weights=None):
        cfg = self.cfg
        Xcat, level_counts, names, level_labels, ids = load_categorical(path, id_col, item_cols)
        n = len(Xcat)
        if n < 4:
            raise ValueError(f"Need at least 4 respondents to segment; got {n}.")
        max_valid_k = n // 2
        if cfg.k_min > max_valid_k:
            raise ValueError(f"k_min={cfg.k_min} too large for n={n} (max supportable {max_valid_k}).")
        if cfg.k_max > max_valid_k:
            print(f"NOTE: clamping k_max from {cfg.k_max} to {max_valid_k} for n={n}.")
            cfg = replace(cfg, k_max=max_valid_k)
        self.cfg = cfg
        print(f"Latent Class Analysis of {n} respondents on {len(names)} categorical items "
              f"(levels: {level_counts}). Fitting by EM...\n")

        self.diagnostics = latent_class_select(Xcat, level_counts, cfg)
        rec_k, rationale = recommend_k_lca(self.diagnostics, cfg)
        self.recommended_k = force_k or rec_k
        if force_k:
            rationale += f"\n\n(Overridden by force_k = {force_k}.)"
        print(rationale.replace("**", ""), "\n")

        self.model = _lca_fit(Xcat, level_counts, self.recommended_k, cfg.n_init_final,
                              cfg.random_state)
        self.labels = self.model["labels"]
        self.jaccard = lca_bootstrap_jaccard(Xcat, level_counts, self.labels,
                                             self.recommended_k, cfg)
        self.typing = lca_typing_tool(Xcat, level_counts, self.labels, cfg)
        self.level_labels = level_labels; self.names = names
        weighted_share = None                       # cluster unweighted; project sizes weighted
        if weights is not None:
            w = np.asarray(weights, float)
            if len(w) == len(self.labels):
                w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
                if w.sum() > 0:
                    weighted_share = {c: float(w[self.labels == c].sum() / w.sum())
                                      for c in range(self.recommended_k)}
        self.assignments = pd.DataFrame({"id": ids, "class": self.labels})
        # Categorical answers have no distances of their own, so the charts work on the indicator
        # (one-hot) coding — the standard way to put pick-any data in a Euclidean space.
        self.Xcat, self.level_counts = Xcat, level_counts
        # Shadow values for the categorical path too, so the gorge plot and the "which segments
        # sit next to each other" table are not privileges of the numeric half of the tool.
        # Categorical answers have no distances of their own, so this uses the same indicator
        # coding the charts already work in.
        try:
            _Xoh = onehot_matrix(Xcat, level_counts)
            _oh_cents = segment_centres(_Xoh, self.labels)
            self.shadow, _c1, _c2 = shadow_values(_Xoh, _oh_cents)
            self.neighbours = segment_neighbours(self.shadow, _c1, _c2, int(self.recommended_k))
            self.assignments["fit"] = np.round(1.0 - self.shadow, 3)
        except Exception as e:
            print(f"NOTE: could not score class fit ({type(e).__name__}: {e}).")
            self.shadow, self.neighbours = None, None
        # The neighbours table was computed above and then dropped on the floor — the comment
        # beside that computation says it exists so the categorical half is not a poor relation,
        # and then it never reached the report. Split-half replication is new here: the numeric
        # path has always reported it, and without it a multiple-choice survey arrived with no
        # answer to "does this reproduce", which is what the confidence light is built from.
        self.split_half = lca_split_half(Xcat, level_counts, self.recommended_k, cfg)
        self.report_markdown = latent_class_report(self.diagnostics, self.recommended_k, rationale,
                                                   self.model, self.jaccard, names, level_labels,
                                                   self.labels, cfg, typing=self.typing,
                                                   weighted_share=weighted_share,
                                                   neighbours=self.neighbours,
                                                   split_half=self.split_half,
                                                   median_shadow=(float(np.median(self.shadow))
                                                                  if self.shadow is not None else None))
        if demographics is not None and (not isinstance(demographics, pd.DataFrame)
                                         or not demographics.empty):
            self.report_markdown += "\n\n" + profile_demographics(self.labels, ids, demographics,
                                                                  id_col, unit="class")
        if outdir:
            self._save(Path(outdir), names, level_labels)
        else:
            print(self.report_markdown)      # only dump to stdout when not saved to a file
        return self

    def profiles_frame(self):
        """Class-conditional item-level probabilities — the profiles that define each class. This is
        the latent-class equivalent of the k-means centroid table."""
        prof = []
        for j, nm in enumerate(self.names):
            for c in range(self.recommended_k):
                for lvl in range(self.model["level_counts"][j]):
                    prof.append({"class": c, "item": nm, "level": str(self.level_labels[j][lvl]),
                                 "probability": round(float(self.model["theta"][j][c][lvl]), 4)})
        return pd.DataFrame(prof)

    def typing_rule_dict(self):
        """The portable latent-class classifier for BRAND-NEW respondents (class weights plus the
        class-conditional item probabilities and the category maps). Shared by the saved
        latent_class_typing_rule.json and the copy the web app hands out."""
        return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tool_version": __version__,
                "method": "lca", "items": self.names, "classes": self.typing["classes"],
                "weights": self.typing["weights"], "theta": self.typing["theta"],
                "level_labels": [[str(x) for x in labs] for labs in self.level_labels],
                "cross_validated_accuracy": (None if np.isnan(self.typing["cv_accuracy"])
                                             else round(self.typing["cv_accuracy"], 3))}

    def _save(self, outdir, names, level_labels):
        outdir.mkdir(parents=True, exist_ok=True)
        self.assignments.to_csv(outdir / "segment_assignments.csv", index=False)
        self.diagnostics.to_csv(outdir / "lca_selection_diagnostics.csv", index=False)
        pd.DataFrame({"class": [f"Class {c}" for c in self.jaccard],
                      "mean_jaccard": list(self.jaccard.values())}).to_csv(
            outdir / "class_stability_jaccard.csv", index=False)
        self.profiles_frame().to_csv(outdir / "latent_class_profiles.csv", index=False)
        (outdir / "latent_class_report.md").write_text(self.report_markdown, encoding="utf-8")
        write_html_report(self.report_markdown, outdir / "latent_class_report.html",
                          "Latent class segmentation report")
        # Typing rule: the portable classifier for NEW respondents.
        # Apply with --classify ... --rule <this file>.
        (outdir / "latent_class_typing_rule.json").write_text(
            json.dumps(self.typing_rule_dict(), indent=2), encoding="utf-8")
        import sklearn
        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_version": __version__,
            "method": "latent_class", "config": asdict(self.cfg),
            "n_respondents": int(len(self.labels)), "n_items": len(names),
            "recommended_k": int(self.recommended_k),
            "log_likelihood": round(self.model["loglik"], 1),
            "typing_cv_accuracy": (None if np.isnan(self.typing["cv_accuracy"])
                                   else round(self.typing["cv_accuracy"], 3)),
            "per_class_jaccard": {f"Class {c}": round(v, 3) for c, v in self.jaccard.items()},
            "library_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                 "scikit-learn": sklearn.__version__},
        }
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")
        print(f"\nSaved to {outdir}/: assignments, class profiles, selection diagnostics, "
              f"per-class Jaccard, report (.md + .html), latent_class_typing_rule.json, "
              f"run_manifest.json")


# =====================================================================================
# Automatic data preparation and a friendly front door (so anyone can use it, not just analysts)
# =====================================================================================
# The engine above is expert-grade; this layer makes it usable by someone who just has a survey
# export and no idea what "range standardization" or "ICL" means. It reads any CSV, recognises the
# respondent id, converts agree/disagree answers to numbers, sets aside timestamps and free text,
# picks k-means or Latent Class Analysis to match the data, and explains every choice in plain
# language. Errors come back as guidance, not tracebacks.
_MISSING_TOKENS = {"", "n/a", "na", "none given", "-", "--", "prefer not to say", "no answer",
                   "nan", "null", "."}
_LIKERT_SCALES = [
    {"strongly disagree": 1, "disagree": 2, "somewhat disagree": 2, "neither agree nor disagree": 3,
     "neutral": 3, "neither": 3, "somewhat agree": 4, "agree": 4, "strongly agree": 5},
    {"strongly dissatisfied": 1, "dissatisfied": 2, "neutral": 3, "satisfied": 4,
     "very satisfied": 5, "strongly satisfied": 5},
    {"never": 1, "rarely": 2, "sometimes": 3, "often": 4, "very often": 5, "always": 5},
    {"very unlikely": 1, "unlikely": 2, "neutral": 3, "likely": 4, "very likely": 5},
    {"poor": 1, "fair": 2, "good": 3, "very good": 4, "excellent": 5},
    {"not important at all": 1, "not important": 2, "neutral": 3, "important": 4,
     "very important": 5, "extremely important": 5},
    # Swedish scales. Without these a Nordic survey falls through to the categorical path and loses
    # the ordering (that "Instämmer helt" is more than "Instämmer"), which is real information.
    # "delvis inte" ("partly not") is the standard wording for the second step on a five-point
    # Swedish agreement scale, and it has to be listed: an unrecognised answer fails the whole
    # column, not just that cell, so a single missing phrase sends the entire survey down the
    # categorical path and throws away the ordering it was measuring.
    {"instämmer inte alls": 1, "instämmer inte": 2, "instämmer delvis inte": 2,
     "instämmer knappt": 2, "varken eller": 3, "varken instämmer eller inte": 3,
     "varken eller/vet ej": 3, "neutral": 3, "vet ej": 3, "instämmer delvis": 4,
     "instämmer": 4, "instämmer till stor del": 4, "instämmer helt": 5,
     "instämmer helt och hållet": 5},
    {"håller inte alls med": 1, "håller inte med": 2, "håller delvis inte med": 2,
     "varken eller": 3, "neutral": 3, "håller delvis med": 4, "håller med": 4,
     "håller helt med": 5, "håller helt och hållet med": 5},
    {"mycket missnöjd": 1, "missnöjd": 2, "ganska missnöjd": 2, "varken eller": 3,
     "varken nöjd eller missnöjd": 3, "neutral": 3, "ganska nöjd": 4, "nöjd": 4,
     "mycket nöjd": 5},
    {"aldrig": 1, "sällan": 2, "ibland": 3, "ofta": 4, "mycket ofta": 5, "alltid": 5},
    {"mycket osannolikt": 1, "osannolikt": 2, "neutral": 3, "sannolikt": 4, "mycket sannolikt": 5},
    {"inte alls viktigt": 1, "inte viktigt": 2, "neutral": 3, "viktigt": 4, "mycket viktigt": 5},
]


def _norm(v):
    return str(v).strip().lower()


def _apply_likert(series):
    """Recode a column already KNOWN to be Likert, however few distinct answers it contains.

    Detection and application are different problems, and sharing one function conflated them.
    `_try_likert` refuses a column with fewer than two distinct answers, which is right when
    deciding whether something IS a rating scale — one repeated word is no evidence. It is wrong
    when applying a scale that was already established: scoring a single new person means every
    column has exactly one answer, so the typing tool refused every batch of one, and refused a
    batch of twenty if any one question happened to get the same answer from everybody.

    Safe because the scales do not disagree: no token appears in two of them with different
    numbers, so a lone answer still resolves to exactly one value.
    """
    vals = {_norm(v) for v in series.dropna().unique()} - _MISSING_TOKENS
    if not vals:
        return None
    for scale in _LIKERT_SCALES:
        if vals <= set(scale):
            return series.map(lambda v: np.nan if (pd.isna(v) or _norm(v) in _MISSING_TOKENS)
                              else scale.get(_norm(v), np.nan))
    return None


def _try_likert(series):
    """If every non-missing answer maps under one known agree/disagree-style scale, return the
    recoded 1-5 series; otherwise None. Missing tokens are tolerated and become NaN.

    Used for DETECTION, where two distinct answers is the minimum evidence that a column is a
    rating scale at all. To apply a scale already known, use `_apply_likert`.
    """
    vals = {_norm(v) for v in series.dropna().unique()} - _MISSING_TOKENS
    if len(vals) < 2:
        return None
    for scale in _LIKERT_SCALES:
        if vals <= set(scale):
            return series.map(lambda v: np.nan if (pd.isna(v) or _norm(v) in _MISSING_TOKENS)
                              else scale.get(_norm(v), np.nan))
    return None


# Split a column name into whole words. Deliberately NOT [a-z]+: that shreds Nordic names — 'kön'
# would become ['k','n'] and 'ålder' ['lder'] — so a Swedish survey's gender and age columns could
# never be recognised as background traits, and would end up FORMING the segments.
_WORD_RE = r"[^\W\d_]+"

def _option_column_name(question, option):
    """Name for one yes/no column split out of a select-all question. Keeps the wording intact —
    the report's label shortener drops small words to save space, which turns a question into
    broken English once it becomes a real column name."""
    q = re.sub(r"\s*\([^)]*\)\s*$", "", str(question)).strip()      # drop a trailing "(select ...)"
    q = re.sub(r"\s+", " ", q).rstrip("?:").strip()
    if len(q) > 34:
        q = q[:33].rsplit(" ", 1)[0].rstrip(",;") + "…"
    return f"{q} — {option}"


def _multiselect_options(series, max_options=15):
    """A 'select all that apply' question. Google Forms (and Typeform, and Microsoft Forms) pack
    every ticked option into ONE cell, comma-separated: "Brand A, Brand C". Treating that as a
    plain multiple-choice column makes every COMBINATION its own category — four options become
    fourteen pseudo-categories with a handful of people each, which is meaningless. Returns the
    option list when the column looks like one, so it can be split into yes/no columns instead.

    Deliberately strict: the options must be short, repeated labels, or an ordinary free-text
    answer containing a comma would be shredded into nonsense."""
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if len(vals) < 8:
        return None
    # Which character packs the ticked options together. Google Forms uses a comma; a Swedish or
    # German Excel writes a semicolon, because the comma is already the decimal mark; some exports
    # use a pipe. Whichever appears in most cells wins, and if none does this is not that shape.
    # Missing the semicolon left "Spotify;Netflix" and "Netflix;Spotify" as different answers —
    # 74 pseudo-categories from five options on a 300-person file.
    sep = max((",", ";", "|"), key=lambda ch: vals.str.contains(re.escape(ch)).mean())
    if vals.str.contains(re.escape(sep)).mean() < 0.15:
        return None                                   # nothing is multi-answer: not this shape
    items = []
    for v in vals:
        items.extend([p.strip() for p in v.split(sep) if p.strip()])
    counts = pd.Series(items).value_counts()
    if not 2 <= len(counts) <= max_options:
        return None
    if counts.index.str.len().max() > 60:             # sentences, not option labels
        return None
    if (counts >= 2).mean() < 0.9:                    # real options recur; text fragments do not
        return None
    # The decisive test: options RECOMBINE. A real option turns up next to different other options
    # ("Brand A" alone, and "Brand A, Brand C"), whereas a clause of free text only ever appears
    # inside its own sentence. Without this, repeated boilerplate comments ("Fine, I guess") get
    # shredded into fake options.
    parents = {}
    for whole in vals.unique():
        for part in {p.strip() for p in whole.split(sep) if p.strip()}:
            parents.setdefault(part, set()).add(whole)
    if float(np.median([len(v) for v in parents.values()])) < 2:
        return None
    return list(counts.index), sep


_ID_NAME_HINTS = ("id", "respondent", "email", "e-mail", "user", "name", "uuid", "participant",
                  "svarsnummer", "deltagare")
_SKIP_NAME_HINTS = ("timestamp", "date", "time", "start", "end", "duration", "ip address",
                    "comment", "feedback", "anything else", "free text", "notes", "explain")
# Background traits that DESCRIBE people but must never FORM the segments (Dolnicar, Kotler): set
# aside for profiling, not clustering. Matched as whole words to avoid 'age' hitting 'usage' etc.
_DEMO_WORDS = {"gender", "sex", "age", "school", "university", "college", "country", "nationality",
               "citizenship", "programme", "program", "major", "cohort", "campus", "faculty",
               "department", "ethnicity", "income", "region", "hometown", "domestic",
               "international", "nationality",
               # Everyday English background traits. 'education' was missing while the Swedish
               # 'utbildning' below was present, so a Swedish survey's education column was set
               # aside correctly and an English one was clustered on as though it were a rating —
               # found on the Big Five inventory, where a 1-5 education code sat among 25 six-point
               # items and became the 26th "question". A numeric demographic code is
               # indistinguishable from a rating scale by its values alone; only the name betrays
               # it, so the name list has to carry both languages for every entry.
               "education", "educational", "employment", "employed", "occupation", "job",
               "profession", "industry", "sector", "employer", "seniority", "tenure",
               "marital", "household", "dependents", "salary", "wage", "earnings",
               "postcode", "postal", "zip", "city", "town", "state", "province", "county",
               "municipality", "language", "race", "degree", "qualification", "diploma",
               "birthyear", "dob",
               # Nordic equivalents, because a questionnaire fielded in Swedish or Norwegian labels
               # its demographics in that language. A mis-detected 'Kön' or 'Universitet' would
               # define the segments instead of describing them.
               "kön", "kjønn", "ålder", "alder", "universitet", "högskola", "hogskola",
               "lärosäte", "larosate", "land", "hemland", "nationalitet", "medborgarskap",
               "stad", "ort", "studieort", "kommun", "utbildning", "fakultet", "institution",
               "årskurs", "arskurs", "termin", "inkomst", "examen", "studieprogram",
               "utbildningsnivå", "utbildningsniva", "sysselsättning", "sysselsattning", "yrke",
               "civilstånd", "civilstand", "hushåll", "hushall", "lön", "lon", "språk", "sprak",
               "modersmål", "modersmal", "postnummer", "födelseår", "fodelsear", "bransch"}
_DEMO_PHRASES = ("study year", "year of study", "class year", "study programme", "study program")
# The most options a real question offers. Beyond this a text column is an identifier, a date or
# free text, whatever the study's size — 'which brand', 'which university', 'which country' all sit
# well inside it, while an invoice number does not.
_NOMINAL_LEVELS_MAX = 100
# The largest value any response scale plausibly reaches. Sliders run to 100, NPS to 10, Likert to
# 7; ages and years sit below this too, so nothing a respondent actually answers is excluded. Above
# it the number is a measurement or an amount attached to the person, and worth a second look.
_RESPONSE_SCALE_MAX = 1000
# A survey weight (e.g. post-stratification / design weight). Cluster UNWEIGHTED, but project the
# segment SIZES to the population with these weights — the usual case being a study that pools
# several sampling strata (schools, regions, panels) that were not sampled in their true proportions.
_WEIGHT_WORDS = {"weight", "weights", "weighting", "vikt", "designweight", "poststrat"}


def _name_matches(name, hints):
    """Match single-word hints as WHOLE WORDS (so 'end' does not match 'gender') and multi-word
    hints as substrings ('ip address', 'anything else')."""
    words = set(re.findall(_WORD_RE, name))
    return any((h in name) if " " in h else (h in words) for h in hints)


def _looks_demographic(name):
    words = set(re.findall(_WORD_RE, name))
    return bool(_DEMO_WORDS & words) or any(p in name for p in _DEMO_PHRASES)


def _looks_like_id(series, name):
    return (any(h in name for h in _ID_NAME_HINTS)
            and series.nunique(dropna=True) >= 0.8 * len(series))


def _looks_like_index(series):
    """A row number or record counter masquerading as an answer (e.g. a 'City_n' or 'Response
    number' column running 1, 2, 3, ...). Clustering on one injects a straight-line gradient that is
    pure bookkeeping, so it must be set aside no matter what the column is called.

    Deliberately strict — every value distinct AND a perfectly consecutive run — because a genuine
    rating or utility must never be mistaken for a counter."""
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any() or len(s) < 8:              # any gap/missing -> not a clean counter
        return False
    v = s.to_numpy(float)
    if not np.all(np.equal(np.mod(v, 1), 0)):     # integers only
        return False
    u = np.unique(v)
    return len(u) == len(v) and bool(np.all(np.diff(u) == 1))


def classify_columns(df, id_col=None):
    """Decide, per column, whether it is the respondent id, a usable answer (continuous rating or
    categorical choice), or something to skip (timestamp, free text, constant). Returns a plan with
    a plain-language note for every column."""
    plan = {"id": None, "weight": None, "continuous": [], "categorical": [], "demographics": [],
            "skipped": [], "recoded": {}, "multiselect": {}, "multiselect_sep": {},
            "notes": []}
    index_like = []          # row counters / stray record ids: never answers, but usable as the id
    for c in df.columns:
        name = str(c).lower().strip()
        s = df[c]
        nun = s.nunique(dropna=True)
        if (id_col and c == id_col) or (plan["id"] is None and id_col is None and _looks_like_id(s, name)):
            plan["id"] = c; plan["notes"].append(f"'{c}': used as the person's id"); continue
        if plan["weight"] is None and set(re.findall(_WORD_RE, name)) & _WEIGHT_WORDS \
                and pd.api.types.is_numeric_dtype(s):
            plan["weight"] = c
            plan["notes"].append(f"'{c}': used as a survey weight (to project group sizes to the "
                                 "whole population, not to form the groups)"); continue
        if _name_matches(name, _SKIP_NAME_HINTS):
            plan["skipped"].append(c); plan["notes"].append(f"'{c}': skipped (date, note, or free-text column)"); continue
        if nun <= 1:
            plan["skipped"].append(c); plan["notes"].append(f"'{c}': skipped (everyone gave the same answer)"); continue
        # A row counter or a second identifier column is bookkeeping, not an answer. Clustering on
        # one silently injects a fake gradient, so set it aside before the numeric branch below.
        #
        # This runs BEFORE the demographic name check because it is structural evidence and that
        # one is only a name: a column running 1, 2, 3, ... n is bookkeeping whatever it is called,
        # and the real file that prompted this rule had its counter named 'City_n'. Once 'city'
        # joined the demographic vocabulary, a name match would otherwise have claimed it first and
        # the counter would have been profiled as though it were a background trait.
        if _looks_like_index(s) or _looks_like_id(s, name):
            index_like.append(c)
            plan["skipped"].append(c)
            plan["notes"].append(f"'{c}': skipped (a row number or record id, not an answer)")
            continue
        # A demographic column is a SHORT label ('Gender', 'Home country'), not a full sentence.
        # Requiring few substantive words stops an attitude question like "Campus politics puts me
        # off an app" from being mistaken for a demographic just because it mentions "campus".
        if _looks_demographic(name) and \
                len([w for w in re.findall(_WORD_RE, name) if w not in _LABEL_STOP]) <= 3:
            plan["demographics"].append(c)
            # Name the escape hatch in the note itself. The words that trigger this rule — age,
            # income, region — are background traits in a SURVEY, where profiling segments by
            # them is right and clustering on them is a beginner's mistake. On other data they
            # can be the most informative column in the file: on a housing dataset, setting
            # median_income aside cost the analysis its richest signal. The default stays, but a
            # reader should not have to infer that it is overridable.
            plan["notes"].append(f"'{c}': set aside as a background trait (used to describe the "
                                 "groups afterwards, not to form them). If it should shape the "
                                 "groups instead, tick it under 'Group people on different "
                                 "questions'"); continue
        if pd.api.types.is_numeric_dtype(s):
            plan["continuous"].append(c)
            # No answer scale reaches five figures. A column on that magnitude is a measurement or
            # a fact attached to the respondent — a headcount, a salary, a distance — not something
            # anybody typed on a 1-5 scale, and it will help define the segments unless someone
            # notices. Found on the Chilean plebiscite survey, where 'population' (the size of the
            # respondent's town, 3,750 to 250,000) split every vote bloc in two: the tool reported
            # eight mind-sets, and four of them were really "lives somewhere bigger".
            #
            # Warn rather than exclude. Whether a number is an answer or a circumstance is a
            # judgement about the study, not a property of the data — on a housing dataset the
            # large-magnitude column is the most informative one in the file — and the guesses this
            # file has tempted me into have not survived measurement. So the reader decides, with
            # the escape hatch named in the sentence.
            if _RESPONSE_SCALE_MAX < float(np.nanmax(np.abs(pd.to_numeric(s, errors="coerce")))):
                plan["notes"].append(
                    f"'{c}': used to form the groups, but its values are far larger than any "
                    "answer scale, so it may be a fact about the person (a headcount, an amount, "
                    "a distance) rather than an answer they gave. If so, tick it under 'Group "
                    "people on different questions' to profile the groups by it instead.")
            else:
                plan["notes"].append(f"'{c}': number ratings, used as-is")
            # A couple of extreme values can flatten a whole column. Range scaling divides by
            # max - min, so a return of -80,995 against a median of 3 puts every ordinary
            # respondent inside 2% of the scale — measured on the UCI online retail file, where
            # 100% of 541,909 people landed in that band. The segmentation then has almost no
            # geometry to work with: k-means cannot separate points that are all but coincident,
            # spends every restart hitting its iteration limit (20 minutes on three columns), and
            # what it does return describes the outliers rather than the people.
            #
            # Said, not silently corrected. --scaling robust exists for this and divides by the
            # interquartile range instead, but which is right depends on whether those extremes
            # are errors or the most interesting rows in the file, and that is the reader's call.
            _v = pd.to_numeric(s, errors="coerce").to_numpy(float)
            _v = _v[np.isfinite(_v)]
            if len(_v) > 20:
                _span = float(_v.max() - _v.min())
                if _span > 0:
                    _crowded = float(np.mean(np.abs((_v - _v.min()) / _span
                                                    - (np.median(_v) - _v.min()) / _span) < 0.01))
                    if _crowded > 0.95:
                        plan["notes"].append(
                            f"'{c}': {_crowded:.0%} of answers sit within 1% of the same value "
                            "once the column is scaled, because a few extreme values set its "
                            "range. Groups built on it will describe those extremes rather than "
                            "the people. Consider removing the outliers, or --scaling robust.")
            continue
        rec = _try_likert(s)
        if rec is not None:
            plan["continuous"].append(c); plan["recoded"][c] = rec
            plan["notes"].append(f"'{c}': agree/disagree scale, converted to 1-5"); continue
        found = _multiselect_options(s)
        if found is not None:
            opts, sep = found
            plan["multiselect"][c] = opts
            plan["multiselect_sep"][c] = sep
            plan["notes"].append(f"'{c}': a select-all question — split into {len(opts)} yes/no "
                                 f"columns ({', '.join(opts[:4])}"
                                 + (", ..." if len(opts) > 4 else "") + ")")
            continue
        # The ceiling on how many options a question may offer is ABSOLUTE, not a share of the
        # sample. It used to be a quarter of the respondents, which grows with the study: on a
        # 541,909-row file that permitted 135,477 "answer options", so invoice numbers with 25,900
        # distinct values and free-text product descriptions with 4,223 were all clustered on as
        # though they were pick-any answers. That run had not finished after half an hour, and
        # could not have said anything if it had — Gower scores two nominal answers as identical
        # or not, and with thousands of levels almost every pair simply differs.
        #
        # An answer list does not get longer because more people were surveyed.
        if 2 <= nun <= min(max(12, int(0.25 * len(s))), _NOMINAL_LEVELS_MAX):
            plan["categorical"].append(c); plan["notes"].append(f"'{c}': multiple-choice answers")
        else:
            plan["skipped"].append(c)
            plan["notes"].append(
                f"'{c}': skipped — {nun:,} different answers is more than a question offers, so "
                "this looks like an identifier, a date or free text. Grouping people on it would "
                "put almost everybody in a group of their own. If it really is a question, tick "
                "it under 'Group people on different questions'.")
    # If the file had no named id, a row counter is a perfect one — keep it to label the results,
    # while still never grouping people on it.
    if plan["id"] is None and index_like:
        c = index_like[0]
        plan["id"] = c
        plan["skipped"].remove(c)
        old = f"'{c}': skipped (a row number or record id, not an answer)"
        new = f"'{c}': used as the person's id (a row number, so never grouped on)"
        plan["notes"] = [new if note == old else note for note in plan["notes"]]
    return plan


#: A pick-any question with a level per respondent is not a set of choices, it is an identifier
#: or a free-text box. Coding it would hand k-prototypes a variable that separates everybody from
#: everybody, which fits beautifully and means nothing. Same ceiling the latent-class path uses.
def _max_levels(n_rows):
    return max(12, int(0.25 * n_rows))


def _code_nominals(clean, cat_cols, items, plan, df):
    """Turn each pick-any column into integer codes, remembering what the codes mean.

    Gower travels through a float matrix like everything else in the pipeline, so the answers have
    to become numbers. The mapping is kept so the report can say "Nespresso" rather than "2" —
    a profile table full of codes would be worse than not having the column at all.
    """
    kinds, labels = plan.setdefault("kinds", {}), plan.setdefault("level_labels", {})
    for c in items:
        if c not in cat_cols:
            kinds[c] = kprototypes.ORDINAL
            continue
        values = df[c].astype(str).fillna("")
        nun = int(values.nunique())
        if nun > _max_levels(len(df)):
            raise ValueError(f"_TOO_MANY_LEVELS:{c}:{nun}")
        codes, uniques = pd.factorize(values, sort=True)
        clean[c] = codes.astype(float)
        kinds[c] = kprototypes.NOMINAL
        labels[c] = {float(i): str(v) for i, v in enumerate(uniques)}


def auto_prepare(df, id_col=None, force_items=None):
    """Turn an arbitrary survey export into (clean_df, method, id_col, item_cols, plan). Clusters on
    the rating questions with k-means when there are at least two of them (the richer signal), and
    otherwise on the multiple-choice questions with Latent Class Analysis, so a non-expert never has
    to choose a method.

    `force_items` overrides the automatic choice with exactly the columns the user picked. That is
    what lets someone group people on something the detector set aside — the ethnicity and income
    columns of a city dataset, say — instead of being stuck with the tool's guess."""
    plan = classify_columns(df, id_col)
    # Turn each select-all question into one yes/no column per option — the standard way to analyse
    # multi-response data, and the only way it can contribute to grouping at all.
    expanded = {}
    for col, options in plan.get("multiselect", {}).items():
        sep = plan.get("multiselect_sep", {}).get(col, ",")
        picked = df[col].fillna("").astype(str).apply(
            lambda v, _s=sep: {p.strip() for p in v.split(_s) if p.strip()})
        for opt in options:
            name = _option_column_name(col, opt)
            # Two long questions can shorten to the same stem; a duplicate column name would
            # silently overwrite the first one's answers.
            if name in expanded or name in df.columns:
                name = f"{name} ({len(expanded) + 1})"
            expanded[name] = picked.apply(lambda chosen, o=opt: int(o in chosen))
    if expanded:
        df = df.copy()
        for name, values in expanded.items():
            df[name] = values
    cont, cat = plan["continuous"], plan["categorical"]
    ms_items = list(expanded)
    # Which brands someone ticks is a different basis from what they think, and a long select-all
    # question would otherwise outvote the rating questions purely by weight of columns. So follow
    # the same rule already used for multiple-choice: prefer the ratings, and fall back to these
    # only when there aren't enough. They stay in the file either way, so the picker can add them.
    if ms_items and len(cont) < 2:
        cont = plan["continuous"] = cont + ms_items
    elif ms_items:
        plan["notes"].append(f"Set aside {len(ms_items)} yes/no column(s) from the select-all "
                             "question and grouped people by the rating questions (the richer "
                             "signal). Tick them in 'Group people on different questions' to "
                             "include them.")
    if force_items:
        chosen = [c for c in force_items if c in df.columns]
        if len(chosen) < 2:
            raise ValueError("_AUTO_NO_ITEMS")
        clean = pd.DataFrame(index=df.index)
        if plan["id"] is not None and plan["id"] not in chosen:
            clean[plan["id"]] = df[plan["id"]]
        numeric = 0
        for c in chosen:
            col = plan["recoded"].get(c)
            if col is None:
                col = df[c]
                rec = None if pd.api.types.is_numeric_dtype(col) else _try_likert(col)
                if rec is not None:
                    col = rec
            clean[c] = col
            numeric += int(pd.api.types.is_numeric_dtype(clean[c]))
        # All-numeric picks are continuous utilities (k-means). A pick with ratings AND pick-any
        # answers goes to Gower k-prototypes, which can use both at once; only an all-categorical
        # pick falls through to latent class.
        text_cols = [c for c in chosen if not pd.api.types.is_numeric_dtype(clean[c])]
        if numeric == len(chosen):
            method = "kmeans"
        elif numeric >= 2 and text_cols:
            method = "kproto"
            _code_nominals(clean, text_cols, chosen, plan, df)
        else:
            method = "lca"
        if method == "lca":
            for c in chosen:
                # Guard against a nonsense model: a column with hundreds of distinct values is a
                # measurement, not a set of choices. Treating it as categorical invents one "level"
                # per person, which fits perfectly and reports high confidence while meaning
                # nothing. Refuse rather than hand back a confidently wrong answer.
                nun = int(df[c].nunique(dropna=True))
                if nun > max(12, int(0.25 * len(df))):
                    raise ValueError(f"_TOO_MANY_LEVELS:{c}:{nun}")
                clean[c] = df[c]
        plan["notes"] = [f"You chose to group people on {len(chosen)} question(s): "
                         + ", ".join(map(str, chosen))]
        return clean, method, plan["id"], chosen, plan
    clean = pd.DataFrame(index=df.index)
    if plan["id"] is not None:
        clean[plan["id"]] = df[plan["id"]]
    if len(cont) >= 2 and cat:
        # Both kinds of question, so use both. Until this existed the multiple-choice columns were
        # set aside with an apology, which threw away answers the respondent took the trouble to
        # give — and on a study where the brand question is the interesting one, threw away the
        # finding. Gower k-prototypes (Szepannek et al. 2024) puts them on one footing.
        method, items = "kproto", cont + cat
        for c in cont:
            clean[c] = plan["recoded"].get(c, df[c])
        _code_nominals(clean, cat, items, plan, df)
        plan["notes"].append(
            f"Used both kinds of question together: {len(cont)} rating question(s) and "
            f"{len(cat)} multiple-choice question(s), grouped with Gower k-prototypes. Ratings "
            "are read as ordered answers rather than as numbers.")
    elif len(cont) >= 2:
        method, items = "kmeans", cont
        for c in items:
            clean[c] = plan["recoded"].get(c, df[c])
    elif len(cat) >= 2:
        method, items = "lca", cat
        for c in items:
            clean[c] = df[c]
    else:
        raise ValueError("_AUTO_NO_ITEMS")
    return clean, method, plan["id"], items, plan


def _detection_summary(plan, method, n_items, n_resp):
    how = {"kmeans": "grouping people by their rating answers (k-means)",
           "lca": "grouping people by their multiple-choice answers (Latent Class Analysis)",
           "kproto": "grouping people by their rating AND multiple-choice answers together "
                     "(Gower k-prototypes)"}[method]
    lines = ["Here is what I found in your file (override anything with flags such as --id-col or --method):"]
    lines += ["  - " + note for note in plan["notes"]]
    lines.append(f"\nGrouping {n_resp} people on {n_items} question(s) by {how}.\n")
    return "\n".join(lines)


_FRIENDLY = {
    "_AUTO_NO_ITEMS":
        ("I could not find at least two answer columns to group people by.\n"
         "I look for rating questions (numbers, or agree/disagree scales) and multiple-choice\n"
         "questions. Check that your file has at least two of those, separate from any id,\n"
         "timestamp, or free-text columns."),
    "_NEED_OPENPYXL":
        ("That looks like an Excel file. Either save it as CSV from Excel (File > Save As, then\n"
         "choose CSV), or install the Excel reader once by running:  pip install openpyxl"),
    "_BAD_FILE":
        ("I could not read that as a survey table. It looks empty, or it is not a spreadsheet\n"
         "(CSV or Excel) file. Export your survey with one row per person and try again."),
    # This used to reach the reader as the bare sentinel, naming no cause. Almost always the cause
    # is the choice column: every set where the best and the worst cannot both be identified is
    # dropped, so an unrecognised word empties the file in silence.
    "_MAXDIFF_TOO_FEW":
        ("That looks like a best-worst (MaxDiff) export, but I could not read enough complete\n"
         "sets from it to score anybody.\n"
         "Each set needs exactly one row marked as the best answer and one as the worst. I read\n"
         "best/worst, most/least, and their Nordic equivalents (bäst/sämst, beste/verste,\n"
         "paras/huonoin); anything else in that column is treated as 'just shown'.\n"
         "Check the choice column, and that each person answered at least a few sets."),
}


def _explain_run_error(msg):
    if msg in _FRIENDLY:                             # sentinel errors (_BAD_FILE, _AUTO_NO_ITEMS, ...)
        return _FRIENDLY[msg]
    if msg.startswith("_TOO_MANY_LEVELS:"):
        _, col, nun = msg.split(":", 2)
        return (f"'{col}' has {nun} different answers, so it is a measurement rather than a set of "
                "choices. Mixing it with word answers would invent one category per person and "
                "produce a confident-looking but meaningless result.\n\nEither pick questions that "
                "are ALL numbers, or pick ones that are all multiple-choice.")
    if msg.startswith("_DESIGN_TOO_FEW_ITEMS:"):
        _, n_items, per_set = msg.split(":", 2)
        return (f"You asked for {per_set} items on each screen but gave only {n_items} items in "
                f"total. Each screen has to show fewer items than you have, or every screen would "
                f"be the same and there would be nothing to compare.\n\nUse --per-screen with a "
                f"smaller number, or add more items.")
    if msg == "_DESIGN_SET_TOO_SMALL":
        return ("A best-worst screen needs at least two items on it — people pick a best and a "
                "worst, which takes two things to choose between. Try --per-screen 4, which is "
                "the usual choice.")
    if msg.startswith("_MAXDIFF_WIDE:"):
        n = msg.split(":", 1)[1]
        return (f"That looks like a best-worst (MaxDiff) survey saved with one row per PERSON — I "
                f"found {n} question blocks where each person has exactly one top pick and one "
                f"bottom pick.\n\n"
                "I stopped instead of analysing it, because a file like this does not say which "
                "code means 'best'. Some tools write 3 for the best item and 1 for the worst, "
                "others the other way round, and choosing wrong would turn the whole ranking "
                "upside down without either of us noticing.\n\n"
                "Reshape it to one row per item SHOWN, with four columns:\n\n"
                "    respondent_id, task, item, choice\n"
                "    P001, 1, Fast delivery, best\n"
                "    P001, 1, Low price, worst\n"
                "    P001, 1, Good support, \n"
                "    P001, 2, Wide range, best\n\n"
                "The last column can be the words best and worst (or most and least), or numbers: "
                "1 for best, -1 for worst, 0 for the rest. Any spreadsheet can do this with a "
                "pivot, and it is the point where YOU tell me which pick was which.")
    if msg.startswith("_MAXDIFF_DIVERGED"):
        # Never seen in practice — the sampler is checked for this rather than trusted not to do
        # it. Written out anyway, because a sentinel that reaches a reader is the failure mode this
        # translation table exists to prevent, and the day it fires is the worst day to discover it
        # says "_MAXDIFF_DIVERGED".
        return ("The best-worst estimate did not settle — the numbers ran away rather than "
                "converging, so there is no ranking I would stand behind.\n\n"
                "This usually means the answers carry almost no information: a study where nearly "
                "everyone picked the same item every time, or where most screens were left blank. "
                "Check that the best and worst picks are recorded the right way round, and that "
                "people actually answered more than a screen or two each.")
    if msg.startswith("_MAXDIFF_MISSING:"):
        # Reached a user verbatim, as "Technical detail: _MAXDIFF_MISSING:item", alongside generic
        # advice to check the file has one row per person — which is the opposite of what a
        # best-worst export looks like. Found by feeding a real published dataset whose item column
        # was called 'issue'.
        role = msg.split(":", 1)[1]
        wanted = {"respondent": "which person answered (respondent_id, id, person)",
                  "set": "which screen or set the row belongs to (set, task, block, question)",
                  "item": "what was being compared (item, statement, issue, option, brand)",
                  "choice": "what they picked (choice, answer, value, best_worst)"}
        return (f"That file looks like a best-worst (MaxDiff) export, but I could not find the "
                f"column saying {wanted.get(role, role)}.\n\n"
                "A best-worst file needs one row per item SHOWN — not one row per person — with "
                "four columns: who answered, which screen they saw it on, the item itself, and "
                "whether they picked it best or worst. The pick can be words (best/worst, "
                "most/least) or numbers (1 for best, -1 for worst, 0 for the rest).\n\n"
                "Rename that column and try again. If this is an ordinary rating survey rather "
                "than a best-worst exercise, it needs none of these columns.")
    if msg.startswith("_UNSCORABLE_ITEM:"):
        return (f"The answers in '{msg.split(':', 1)[1]}' are not in a form I can score. That "
                "question needs the same answer options as the original survey (either numbers, or "
                "the same agree/disagree wording).")
    if msg.startswith("New data is missing required item column"):
        return ("The new file is missing some of the questions the groups were built from, so I "
                "cannot place these people. Export it with the same question columns as the "
                "original survey.\n\n" + msg)
    if "at least 4 respondents" in msg or "too few" in msg.lower():
        return ("Your file has too few people in it to find reliable groups (I need at least a\n"
                "few dozen, and ideally 100+). Collect more responses and try again.")
    if "k_min" in msg:
        return ("You asked for more groups than the data can support. Try fewer groups, or leave\n"
                "the number to the tool by not setting --force-k.")
    if "two numeric" in msg or "two categorical" in msg or "2+ levels" in msg:
        return _FRIENDLY["_AUTO_NO_ITEMS"]
    # Anything unrecognised is a technical message ("Connection reset by peer", a library error).
    # Non-experts must still get a plain-language sentence, with the detail kept for whoever helps.
    return ("Something went wrong while reading or analysing that file, so I stopped rather than\n"
            "show you a result I am not sure about. Check that it is a survey export with one row\n"
            "per person, then try again.\n\nTechnical detail (for whoever supports you): " + msg)


def _friendly_fail(parser, msg):
    parser.exit(2, "\n" + "=" * 72 + "\nI could not finish. Here is the problem in plain language:\n\n"
                + msg + "\n" + "=" * 72 + "\n")


def run_auto(path, args, parser):
    """The no-expertise-required path: point it at a CSV, get a segmentation and a readable report,
    with every automatic choice explained and every failure returned as guidance."""
    try:
        df = _read_table(path)
    except FileNotFoundError:
        _friendly_fail(parser, f"I could not find a file called '{path}'.\nCheck the spelling, and "
                       "that your terminal is in the same folder as the file.")
    except ValueError as e:
        _friendly_fail(parser, _FRIENDLY.get(str(e), str(e)))
    except Exception as e:
        _friendly_fail(parser, f"I could not read '{path}' as a spreadsheet.\nMake sure it is a .csv "
                       f"file exported from your survey tool. (Technical detail: {e})")
    try:
        clean, method, id_col, items, plan = auto_prepare(df, args.id_col)
    except ValueError as e:
        _friendly_fail(parser, _FRIENDLY.get(str(e), str(e)))
    print(_detection_summary(plan, method, len(items), len(clean)))
    outdir = args.outdir or (Path(str(path)).stem + "_results" if not isinstance(path, pd.DataFrame)
                             else "results")
    demo_df = df[[id_col] + plan["demographics"]] if (plan["demographics"] and id_col) else None
    weights = df[plan["weight"]].to_numpy() if plan["weight"] else None
    cfg = SegmentationConfig(method=method, random_state=args.seed,
                             var_kinds=plan.get("kinds"), level_labels=plan.get("level_labels"))
    # --force-k has to be forwarded here, not only on the explicit-method paths below. This is the
    # DEFAULT path, so for as long as it was missing the flag did nothing at all for almost every
    # user: the run completed, the report never mentioned the override, and the number in it was
    # the one the tool had picked by itself. An ignored flag that reports success is worse than an
    # unimplemented one, because the user believes the answer they are reading is the one they
    # asked for.
    force_k = getattr(args, "force_k", None)
    try:
        if method == "lca":
            LatentClassSegmenter(cfg).run(clean, id_col=id_col, item_cols=items, outdir=outdir,
                                          demographics=demo_df, weights=weights, force_k=force_k)
        else:
            Segmenter(cfg).run(clean, id_col=id_col, item_cols=items, outdir=outdir,
                               demographics=demo_df, weights=weights, force_k=force_k)
    except ValueError as e:
        _friendly_fail(parser, _explain_run_error(str(e)))
    report = "latent_class_report" if method == "lca" else "segmentation_report"
    print(f"\nAll done. Open '{outdir}/{report}.html' in your browser to read the results "
          "(or the .md version in any text editor). The plain-language summary is at the top.")


def _maxdiff_ranking_section(est):
    """The headline answer to a best-worst study: what the sample wants, strongest first.

    This exists because the tool was computing it and throwing it away. A MaxDiff export was
    detected, converted to individual utilities, and handed to the segmenter — and the report then
    described the groups while never once saying which items the sample actually preferred. That is
    the question the study was fielded to answer; the segments are what you do about the answer.

    Two things are deliberate in the table:

    * **The interval, not just the score.** Hierarchical Bayes knows how sure it is, and the whole
      argument for paying its cost is that it produces a distribution rather than a point.
    * **Whether each item is genuinely ahead of the one below it.** A ranking prints an order
      whether or not the data supports one. Adjacent items whose intervals overlap are printed in
      an order this study did not establish, and saying so is the difference between a ranking and
      a league table nobody should act on.
    """
    rank = est.ranking()
    # Two independent questions, kept independent. An estimate saved before the credible interval
    # was retained has neither; one saved before the posterior draws were retained has the interval
    # but not the probability. Conflating them would silently drop the range column from a result
    # that has a perfectly good range.
    has_range = "low" in rank.columns and rank["low"].notna().any()
    has_prob = "prob_ahead" in rank.columns
    cols = ["#", "item", "score"]
    show = rank.rename(columns={"rank": "#", "utility": "score"})
    if has_range:
        show["95% range"] = [f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(rank["low"], rank["high"])]
        cols.append("95% range")
    if has_prob:
        # The probability itself, not a verdict derived from it. A column that said only "yes" or
        # "too close to call" gave the same three words to a coin flip and to a 93% finding.
        show["chance it beats the next"] = [
            "" if pd.isna(p) else (">99%" if p > 0.995 else f"{p * 100:.0f}%")
            for p in rank["prob_ahead"]]
        cols.append("chance it beats the next")
    show = show[cols]
    show["score"] = show["score"].map(lambda v: f"{v:+.2f}")

    top, bottom = rank.iloc[0]["item"], rank.iloc[-1]["item"]
    # The opening sentence has to know what the table knows. Asserting "X comes out strongest" and
    # then spending the next paragraph explaining that nothing is separated is the report arguing
    # with itself, which is a fault this tool has had to fix twice elsewhere. On data with no real
    # differences every position is unsettled, and the lead must say so rather than name a winner
    # picked out of noise.
    settled = ([bool(rank.iloc[i]["separated_from_next"]) for i in range(len(rank) - 1)]
               if has_prob else [])
    if has_prob and not any(settled):
        lead = (f"Before any grouping: this is what your respondents want, taken together — except "
                f"that **this study did not separate these items at all**. **{top}** scores highest "
                f"and **{bottom}** lowest, but no item is clearly ahead of any other, so the order "
                f"below should not be read as a ranking.")
    elif has_prob and not settled[0]:
        lead = (f"Before any grouping: this is what your respondents want, taken together. "
                f"**{top}** scores highest and **{bottom}** lowest — though **{top}** is not "
                f"clearly ahead of **{rank.iloc[1]['item']}**, so treat the top of this list as a "
                f"pair rather than a winner.")
    else:
        lead = (f"Before any grouping: this is what your respondents want, taken together. "
                f"**{top}** comes out strongest and **{bottom}** weakest.")
    # disable_numparse, or tabulate re-parses the formatted strings as numbers and re-formats them
    # its own way: "+1.50" comes back as "1.5" and "+2.00" as "2", so a column that was deliberately
    # aligned to two decimals arrives ragged. The signs matter here too — this scale is centred on
    # zero, so whether a number is positive is the first thing being read.
    lines = ["## What matters most, overall", "", lead, "",
             show.to_markdown(index=False, disable_numparse=True), ""]

    if has_prob:
        close = [i for i in range(len(rank) - 1)
                 if not bool(rank.iloc[i]["separated_from_next"])]
        if close:
            # Named WITH their probabilities, because those differ enormously between pairs and a
            # single collective warning would flatten them back into "we cannot tell".
            pairs = [f"**{rank.iloc[i]['item']}** over **{rank.iloc[i + 1]['item']}** "
                     f"({rank.iloc[i]['prob_ahead'] * 100:.0f}%)" for i in close]
            joined = pairs[0] if len(pairs) == 1 else ", ".join(pairs[:-1]) + " and " + pairs[-1]
            lines += [f"> **{len(close)} position{'s' if len(close) > 1 else ''} in this order "
                      f"{'are' if len(close) > 1 else 'is'} not settled** at the usual 95% mark: "
                      f"{joined}. The percentage is the chance that pair really is the right way "
                      f"round — read it rather than the position. More respondents, or more "
                      f"questions each, is what would settle them.", ""]
        else:
            lines += ["> Every item beats the one below it with at least 95% certainty, so this "
                      "order is one you can act on rather than an artefact of rounding.", ""]

    # An MCMC estimate that has not settled still produces a tidy table and a confident-looking
    # interval — there is nothing in the output itself to betray it. Saying so is the only
    # protection a reader has, and it belongs beside the numbers rather than in a log nobody keeps.
    if getattr(est, "converged", None) is False:
        lines += [f"> **These numbers have not fully settled.** The estimator works by drawing "
                  f"repeatedly and averaging, and its draws were still wandering when it stopped "
                  f"(a convergence score of {est.rhat:.2f}, where under 1.05 counts as settled). "
                  f"The order above is still the best available read of your data, but treat the "
                  f"exact scores and the ranges around them as approximate. Fewer items to "
                  f"compare, or more people, is what would settle it.", ""]
    lines += ["**How to read the score.** It is a relative preference, centred so the average item "
              "sits at zero: positive means wanted more than the average item, negative less. Only "
              "*differences* carry meaning — the units themselves are arbitrary, so a score of "
              "+2.0 is not 'twice' +1.0, it is further ahead of the average.", ""]
    return "\n".join(lines)


def _turf_module_top_n():
    """The reach rule TURF will use, asked of the module rather than repeated here.

    A second copy of this number is exactly how a threshold and the guard that depends on it drift
    apart — the guard below is derived from it, so it has to be the same one.
    """
    try:
        import turf as _turf
        return int(_turf.DEFAULT_TOP_N)
    except Exception:
        return 3


def _turf_section(est, size=3):
    """Which few items to launch — the decision a best-worst study is usually commissioned for.

    The ranking answers "what do people like"; this answers "what do we put on the menu", and they
    are different questions whenever tastes divide. The three best-liked items can all appeal to
    the same crowd while a fourth is the only thing anyone else will take.

    Returns None rather than a section when there is nothing worth saying: too few items to choose
    between, or too few respondents for the holdout to mean anything.
    """
    try:
        import turf as _turf
    except Exception:
        return None
    # The item list has to be long enough for "reached" to discriminate. Someone counts as reached
    # when the item is in their top `top_n`, so each person accepts top_n/n_items of the list: at
    # five items that is 60% of everything, and the best set of three then reaches 100.0% of people
    # in every study tried, with ten combinations tied for the honour. That is not a finding, it is
    # arithmetic wearing a finding's clothes.
    #
    # Measured, best reach for a set of three over 200 people: 5 items 100.0%, 6 items 99.8%,
    # 8 items 98.4%, 10 items 95.1%, 20 items 83.9%. The rule below asks that a person accept no
    # more than a third of the list, which cuts in at nine items for the default top three, and is
    # where the tied-combination count settles to roughly one.
    top_n = _turf_module_top_n()
    if est.utilities is None or len(est.respondent_ids) < 40:
        return None
    if len(est.item_names) < max(size + 2, 3 * top_n):
        return None
    result = _turf.turf(est.utilities, est.item_names, size=size)

    rows = pd.DataFrame({
        "#": range(1, len(result["items"]) + 1),
        "item": result["items"],
        "reaches on its own": [f"{a * 100:.0f}%" for a in result["alone"]],
        "adds over the ones above": [f"+{g * 100:.0f} points" for g in result["incremental"]],
    })
    lines = [f"## Which {size} to launch", "",
             "The ranking above says what people like. This says which **set** reaches the most "
             "people, which is a different question whenever tastes divide — the best-liked items "
             "can all appeal to the same crowd.", "",
             f"**{', '.join(result['items'])}** — together reaching "
             f"**{result['reach'] * 100:.0f}%** of your respondents.", "",
             rows.to_markdown(index=False, disable_numparse=True), ""]

    # A tie is the difference between an answer and a coin toss, and it is common enough to matter:
    # reach is a count of people, so it lands on multiples of 1/n, and a few hundred candidate
    # combinations chasing sixty possible values collide often. Left unsaid, the reader takes a set
    # chosen by the order of their own spreadsheet as the finding.
    if (result.get("n_tied") or 1) > 1:
        others = result["tied_items"][1:]
        shown = "; ".join(", ".join(c) for c in others[:3])
        more = len(others) > 3 or result.get("tie_capped")
        lines += [f"> **{result['n_tied']}{'+' if result.get('tie_capped') else ''} different sets "
                  f"reach exactly these same people.** The one named above is not better than the "
                  f"others — it came first in the item list. Equally good: {shown}"
                  f"{', and others' if more else ''}. **Choose between them on grounds this survey "
                  f"does not contain** — cost, margin, how hard each is to build, whether one fits "
                  f"the brand. Ties like this are ordinary at this sample size; more respondents "
                  f"would separate them.", ""]

    # How many of the chosen items are actually pulling their weight. The incremental column shows
    # this row by row, but a reader looking for the headline needs it said once: if the first item
    # already reaches everyone the second and third are decoration, and recommending three when one
    # would do is an expensive way to be right.
    enough = len(result["incremental"])
    running = 0.0
    for position, gain in enumerate(result["incremental"], start=1):
        running += gain
        if running >= result["reach"] - 0.01:
            enough = position
            break
    if enough < len(result["items"]):
        kept = ", ".join(result["items"][:enough])
        lines += [f"> **{enough} of these {len(result['items'])} would do.** {kept} already "
                  f"reach{'es' if enough == 1 else ''} "
                  f"{running * 100:.0f}% — everything after that is adding people who are already "
                  f"counted. Launch the rest if you want them for other reasons, but not for reach.",
                  ""]

    weakest = result["incremental"][-1]
    if enough == len(result["items"]) and weakest < 0.03:
        lines += [f"> **{result['items'][-1]} is carrying almost nobody the others do not already "
                  f"carry** — it adds {weakest * 100:.0f} points. If a place on the list is "
                  f"expensive, that is the one to question.", ""]

    # Only when the gap is material. A warning that fires at zero optimism reads as broken —
    # "Expect about 100%, not 100%", "the 0-point difference" — and a caveat that appears when
    # there is nothing to caveat teaches the reader to skip the one that matters.
    if result["optimism"] == result["optimism"] and result["optimism"] >= 0.01:
        lines += [f"> **Expect about {result['holdout_reach'] * 100:.0f}%, not "
                  f"{result['reach'] * 100:.0f}%.** The headline is the best of "
                  f"{'every' if result['search'] == 'exhaustive' else 'many'} combination tried, "
                  f"and a maximum chosen on one sample flatters itself. Choosing the set on half "
                  f"these people and measuring it on the other half — forty times over — gives "
                  f"{result['holdout_reach'] * 100:.0f}%. The "
                  f"{result['optimism'] * 100:.0f}-point difference is what the search borrowed "
                  f"from luck, and it is the number to take to a budget meeting.", ""]
    elif result["optimism"] == result["optimism"]:
        lines += [f"> **This one holds up.** Choosing the set on half these people and measuring it "
                  f"on the other half gives {result['holdout_reach'] * 100:.0f}%, which is what it "
                  f"scored here — so the figure is not an artefact of searching for the best "
                  f"combination. That is unusual enough to be worth saying; on smaller studies with "
                  f"longer item lists the two can differ by twenty points.", ""]

    lines += [f"*Someone counts as reached by an item when it is among their own top "
              f"{result['top_n']}. A different rule would give a different winner, so it is stated "
              f"rather than assumed.*", ""]
    return "\n".join(lines), result


def run_analysis(data, cfg=None, force_items=None):
    """Raw survey (bytes or a path) -> a dict with everything the web app and the AI layer need:
    the title, the report as an HTML fragment (auto-detection notes on top), and a plain-text
    `digest` (the same content as Markdown) that is safe to hand to Claude for interpretation.

    The digest is AGGREGATE only — segment sizes, mean scores, stability numbers, and demographic
    percentages — never an individual respondent's row. Pass cfg only to speed up tests; the app
    uses full-quality defaults."""
    df = _read_table(data)
    # A MaxDiff export is not a rating grid and cannot be clustered as one: its rows are
    # best/worst choices, not scores. Detect that shape and convert it to individual-level
    # utilities first — that conversion IS the analysis for a MaxDiff study, and doing it
    # silently wrong (by clustering the raw choice codes) would look like a working result.
    maxdiff_note = None
    maxdiff_est = None
    # Computed once, next to the estimate it derives from: the downloads are assembled before the
    # report body, and running the search twice would double the cost for no reason.
    _turf_made = None
    # A best-worst export written one row per PERSON carries no sign that it is a preference
    # exercise, so it was read as an ordinary rating grid and its response CODES were clustered as
    # though they were scores — two confident segments out of an 80-person file, no warning
    # anywhere. Stop rather than answer. The layout can be recovered but the polarity cannot:
    # whether 3 means best or 1 means best is a fact about how the survey was built, not about the
    # data, and guessing it would invert every ranking silently.
    if _maxdiff is not None and not _maxdiff.looks_like_maxdiff(df):
        _wide = _maxdiff.looks_like_wide_best_worst(df)
        if _wide:
            raise ValueError(f"_MAXDIFF_WIDE:{_wide}")
    if _maxdiff is not None and _maxdiff.looks_like_maxdiff(df):
        est = maxdiff_est = _maxdiff.utilities_from_export(df)
        df = est.as_frame().reset_index().rename(columns={"index": "respondent_id"})
        maxdiff_note = (
            f"This file is a MaxDiff (best-worst) export, not a rating grid, so it was scored "
            f"first: individual-level utilities for {len(est.item_names)} items were estimated "
            f"for {len(est.respondent_ids)} respondents by Hierarchical Bayes, and the groups "
            f"below are built on those utilities. Counting how often each item was picked best "
            f"minus worst would have been too coarse to describe a single person.")
    if maxdiff_est is not None:
        _turf_made = _turf_section(maxdiff_est)
    clean, method, id_col, items, plan = auto_prepare(df, force_items=force_items)
    _opts = {"method": method, "var_kinds": plan.get("kinds"),
             "level_labels": plan.get("level_labels")}
    base = replace(cfg, **_opts) if cfg is not None else SegmentationConfig(**_opts)
    demo_df = df[[id_col] + plan["demographics"]] if (plan["demographics"] and id_col) else None
    weights = df[plan["weight"]].to_numpy() if plan["weight"] else None
    if method == "lca":
        seg = LatentClassSegmenter(base).run(clean, id_col=id_col, item_cols=items,
                                             demographics=demo_df, weights=weights)
    else:
        seg = Segmenter(base).run(clean, id_col=id_col, item_cols=items, demographics=demo_df,
                                  weights=weights)
    title = "Latent class segmentation report" if method == "lca" else "Segmentation report"
    if maxdiff_note:
        plan["notes"].insert(0, maxdiff_note)
    notes_html = ("<blockquote><strong>What I found in your file:</strong><ul>"
                  + "".join(f"<li>{_html.escape(n)}</li>" for n in plan["notes"])
                  + "</ul></blockquote>")
    notes_md = "**What the tool found in your file:**\n" + "".join(f"- {n}\n" for n in plan["notes"])
    # The actionable outputs: who is in which group, what defines each group, and the portable rule
    # for typing NEW people later. Handed to the app so a non-technical user can download them
    # without ever touching the command line.
    files = {"segment_assignments.csv": seg.assignments.to_csv(index=False),
             "typing_rule.json": json.dumps(seg.typing_rule_dict(), indent=2)}
    if maxdiff_est is not None:
        # The scored study itself, not just the grouping of it. `item_utilities.csv` is the
        # deliverable most people came for; `respondent_utilities.csv` is what makes any further
        # analysis possible without re-running the sampler, which is the expensive part.
        files["item_utilities.csv"] = maxdiff_est.ranking().to_csv(index=False)
        _per_person = maxdiff_est.as_frame()
        _per_person.index.name = "respondent_id"
        files["respondent_utilities.csv"] = _per_person.to_csv()
        if _turf_made is not None:
            _tp = _turf_made[1]
            files["what_to_launch.csv"] = pd.DataFrame({
                "rank": range(1, len(_tp["items"]) + 1),
                "item": _tp["items"],
                "reach_alone": _tp["alone"],
                "incremental_reach": _tp["incremental"],
                "combined_reach": _tp["reach"],
                "expected_reach_holdout": _tp["holdout_reach"],
            }).to_csv(index=False)
    if method == "lca":
        files["group_profiles.csv"] = seg.profiles_frame().to_csv(index=False)
    else:
        files["group_profiles.csv"] = seg.centroids.to_csv()
    # What the user could group people on instead — every column that is not the id, so they can
    # override a detector guess (e.g. group cities on income, which auto-detection sets aside).
    roles = {}
    for c in df.columns:
        if c == plan["id"]:
            continue
        roles[str(c)] = ("used" if c in items else
                         "background" if c in plan["demographics"] else
                         "choice" if c in plan["categorical"] else
                         "rating" if c in plan["continuous"] else "skipped")
    # Lift the confidence light out of the prose so the UI can show it as a state, not a sentence:
    # how much to trust the answer is the first thing a reader needs, not a line buried mid-report.
    m = re.search(r"\*\*Confidence: .{0,3} (High|Moderate|Low)\.\*\*", seg.report_markdown)
    _chart_errors = []          # this analysis's own; the server runs several at once
    # For a best-worst study the ranking comes BEFORE the segmentation, because it is the answer
    # the study was fielded for — the groups are what you do about it. Putting it after would bury
    # the headline under three pages of validation the reader did not ask for yet.
    _body = seg.report_markdown
    _ranking = None
    if maxdiff_est is not None:
        if _turf_made is not None:
            _body = _turf_made[0] + "\n" + _body
        _body = _maxdiff_ranking_section(maxdiff_est) + "\n" + _body
        # Also as structured data, not only as prose inside the report. The report is rendered
        # into a panel that is COLLAPSED by default and sits below the segment map, so a study
        # whose entire purpose was to rank items was hiding that ranking two clicks down. Handing
        # the interface the numbers lets it show the answer where the answer belongs.
        _rk = maxdiff_est.ranking()
        _ranking = [{"rank": int(r["rank"]), "item": str(r["item"]),
                     "score": round(float(r["utility"]), 3),
                     "low": None if pd.isna(r.get("low")) else round(float(r["low"]), 3),
                     "high": None if pd.isna(r.get("high")) else round(float(r["high"]), 3),
                     "prob_ahead": (None if pd.isna(r.get("prob_ahead"))
                                    else round(float(r["prob_ahead"]), 4)),
                     "clear_of_next": (None if pd.isna(r.get("separated_from_next"))
                                       else bool(r["separated_from_next"]))}
                    for _, r in _rk.iterrows()]
    return {"title": title, "method": method, "ranking": _ranking,
            "report_html": notes_html + _markdown_to_html(_body),
            "digest": notes_md + "\n" + _body,
            "files": files, "n_people": int(len(seg.assignments)),
            "k": int(seg.recommended_k), "columns": roles,
            "charts": build_charts(seg, method, errors=_chart_errors),
            # Why any chart is missing, so a packaged build that cannot draw says so instead of
            # quietly showing nothing. Empty on a healthy run, and only ever carries exception
            # text from the drawing layer — never anything from the respondent data. The list is
            # created per analysis, so two people running at once cannot see each other's.
            "chart_errors": _chart_errors,
            # Per segment, how much of it stays together when a different number of segments is
            # asked for. Separate from the confidence light because it answers a different
            # question: confidence covers the solution, this covers each segment on its own.
            "persistence": dict(getattr(seg, "persistence", {}) or {}),
            "confidence": (m.group(1).lower() if m else "unknown")}


def analyze_csv_to_html(data, cfg=None):
    """Raw CSV (bytes or a path) -> a self-contained, styled HTML report page (auto-detection notes
    on top). Kept for the library API and tests; the web app uses run_analysis() directly."""
    r = run_analysis(data, cfg)
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>{_html.escape(r["title"])}</title><style>{_REPORT_CSS}</style></head>'
           f'<body><main>{r["report_html"]}{charts_html(r.get("charts"))}</main></body></html>')
    return r["title"], doc


def charts_html(charts):
    """The charts as a plain HTML section, for the standalone report and anything that prints it.
    Stacked rather than tabbed: a document has no JavaScript, and a reader scrolling a printout
    should meet all the evidence rather than only whichever tab happened to be open."""
    if not charts:
        return ""
    out = ['<hr><h2>See the data yourself</h2>'
           '<p>The charts below are the same analysis, drawn. Read them before accepting the '
           'summary above &mdash; a segmentation that looks weak here <em>is</em> weak, whatever '
           'any write-up says about it.</p>']
    for c in charts:
        out.append(f'<h3>{_html.escape(c["title"])}</h3>'
                   f'<div class="cwrap">{c["svg"]}</div>'
                   f'<p class="ccap">{c["caption"]}</p>')
    return "".join(out)


# The web application lives in webapp.py — the page, its API, and saved projects. It is the
# delivery layer and it imports from here, not the other way round, so the import happens inside
# these functions: at module scope it would be a cycle.
#
# `run_app.py` imports webapp directly as well. That is not redundant: a lazy import is invisible
# to a bundler doing static analysis, and an app that packages without its own web server is
# exactly the class of failure that shipped once already with matplotlib's SVG backend.


def serve(port=8000):
    """Start the local web app. See webapp.serve."""
    from webapp import serve as _serve
    return _serve(port)


def app():
    """Entry point for the packaged desktop app. See webapp.app."""
    from webapp import app as _app
    return _app()


# =====================================================================================
# Orchestrator
# =====================================================================================
class Segmenter:
    def __init__(self, cfg: SegmentationConfig | None = None):
        self.cfg = cfg or SegmentationConfig()

    def run(self, path, id_col=None, item_cols=None, force_k=None, outdir=None,
            demographics=None, weights=None):
        cfg = self.cfg
        X, X_raw, ids, self.scale_params = load_and_prepare(path, cfg, id_col, item_cols)
        n = len(X)

        # Validate and clamp the search range to what the data can support. The binding limit is
        # the half-split used by prediction strength (each half is clustered into k), so k <= n//2;
        # this is also within the silhouette limit of k <= n-1.
        if n < 4:
            raise ValueError(f"Need at least 4 respondents to segment; got {n}.")
        # A cluster needs a distinct point to sit on, so the number of DISTINCT answer patterns
        # is a hard ceiling independent of n. Two 1-to-5 questions admit only 25 patterns, so a
        # 120-person file cannot yield 55 groups no matter what the criteria vote for: k-means
        # silently returns duplicate/empty clusters and warns.
        #
        # The binding limit is the half-split, not the whole file: prediction strength clusters
        # each half into k, and a half necessarily holds fewer distinct patterns than the whole.
        # Measure that directly over a few splits and take the worst, so the search only ever
        # scores solutions the validation can actually fit.
        _Xa = np.asarray(X, float)
        _rng = np.random.default_rng(cfg.random_state)
        n_distinct = min(
            [len(np.unique(_Xa, axis=0))]
            + [len(np.unique(_Xa[_rng.choice(n, n // 2, replace=False)], axis=0))
               for _ in range(5)])
        # A segment needs enough DISTINCT answer patterns to be a type rather than a coordinate.
        # The cap used to be the number of patterns itself, which allows a group per pattern. On a
        # short survey that is how the separation indices win an argument they should not: three
        # questions on a 1-5 scale gave 16 distinct patterns among 400 people, the silhouette
        # preferred eight groups of about two patterns each, and the answer came out at 0.39
        # against a planted truth it recovers perfectly at k=2.
        #
        # Requiring roughly four patterns per group fixes that case (0.39 -> 1.00) and changes
        # nothing else: measured across well-separated, unequal 80/15/5, overlapping, five-group
        # and noise data, and against every real file in the corpus — bfi, Chile, Mroz, the MASS
        # student survey, adult — where the smallest count of distinct patterns is 236 and the cap
        # never comes near binding.
        # Two different bounds, kept apart because they answer different questions.
        self.k_max_note = ""
        feasible = max(2, min(n // 2, n_distinct))          # more groups than this cannot exist
        sensible = max(2, min(feasible, n_distinct // 4))   # ~4 patterns per group to be a type
        # The ratio is a heuristic for choosing among many k, not a feasibility bound, so it never
        # overrides a k the caller asked for outright — on six respondents with six answer patterns
        # it would otherwise refuse an explicit k_min=3, which is a legitimate thing to ask.
        max_valid_k = max(sensible, min(cfg.k_min, feasible))
        # Kept for the report: Hopkins is inflated when many respondents share an identical
        # answer pattern, so the reader needs to know how common that is here.
        self.distinct_share = float(len(np.unique(_Xa, axis=0)) / n)
        if cfg.k_min > max_valid_k:
            raise ValueError(f"k_min={cfg.k_min} is too large for n={n} (the most segments the "
                             f"validation can support is {max_valid_k}).")
        if cfg.k_max > max_valid_k:
            why = ("the resampling-based validation cannot reliably support more segments"
                   if max_valid_k == n // 2 else
                   f"there are only {n_distinct} distinct answer patterns in the data, and a "
                   "group built from fewer than a handful of them is a coordinate rather than a "
                   "mind-set")
            print(f"NOTE: clamping k_max from {cfg.k_max} to {max_valid_k} — with n={n} "
                  f"respondents {why}.")
            # The reader has to be told too. The report stated "Search range: k = 2 to 5" with no
            # hint that 55 had been asked for and cut, and a note printed to a terminal is invisible
            # to everyone using the app — which is a search quietly narrowed, the same shape of
            # silence this report has been cleaned of elsewhere.
            self.k_max_note = (f" The search stopped at {max_valid_k} rather than the "
                               f"{cfg.k_max} requested: with {n:,} respondents {why}.")
            cfg = replace(cfg, k_max=max_valid_k)

        # The consensus matrix is n x n, so a large study cannot have one built over everybody:
        # at 41,188 respondents the two matrices would want 27 GB. This used to drop consensus
        # clustering altogether above 5,000, which cost the PAC column entirely — one of the three
        # criteria the k panel weights double, silently absent on exactly the large studies where
        # a second opinion is most useful. It is now estimated from a random MAX_PAIRWISE_N
        # respondents instead, and the report says which columns that applies to. PAC is a summary
        # of how ambiguously pairs co-cluster and a sample of 6,000 estimates it well; what cannot
        # be sampled is the ensemble PARTITION, which has to place everybody, and that still skips.
        if cfg.run_consensus and n > MAX_PAIRWISE_N:
            print(f"NOTE: n={n:,} respondents; the consensus matrix needs a value for every pair, "
                  f"so consensus_PAC is estimated on a random {MAX_PAIRWISE_N:,} of them. The "
                  "segmentation itself uses everybody.")
        self.cfg = cfg   # use the validated/clamped config for the rest of the run

        # Gower normalises each question inside the distance itself, so there is no scaling step
        # to name on that path — printing one would describe something that did not happen.
        _how = ("method: kproto, Gower distance" if _is_gower(cfg)
                else f"method: {cfg.method}, scaling: {cfg.scaling}")
        print(f"Segmenting {X.shape[0]} respondents on {X.shape[1]} items "
              f"({_how}). Running the diagnostic panel...\n")

        # THE WORKING SET FOR ESTIMATES. The panel that chooses k is made entirely of resampling
        # estimates -- the gap statistic against 20 reference datasets, replication stability over
        # 30 resamples, prediction strength over 10 splits, consensus over 50, bootstrap Jaccard
        # over 100 -- which at seven candidate values of k is well over a thousand clusterings of
        # the whole file. Measured on 48,842 respondents that climbed to 11 GB, and not through
        # any single allocation: instrumenting every stage showed no call raising the high-water
        # mark by more than 0.29 GB. It is a thousand fits each leaving a little behind.
        #
        # So the estimates run on a bounded random sample and the ANSWER covers everybody: the
        # final fit, every respondent's segment, the profiles, the exports and the charts all use
        # the full file. Estimating a resampling statistic from 12,000 people rather than 48,842
        # is what those statistics are for; what would not be acceptable is a segmentation that
        # quietly described a sample, and that is not what this does.
        self.search_sample = None
        if len(X) > MAX_SEARCH_N:
            self.search_sample = np.random.default_rng(cfg.random_state + 91).choice(
                len(X), MAX_SEARCH_N, replace=False)
            print(f"NOTE: {len(X):,} respondents. The checks that choose the number of segments "
                  f"are estimated on a random {MAX_SEARCH_N:,} of them; the segmentation itself, "
                  "and every person's group, uses everybody.\n")
        Xs = X[self.search_sample] if self.search_sample is not None else X

        self.hopkins = hopkins_statistic(Xs, np.random.default_rng(cfg.random_state + 7), cfg=cfg)
        # A second, independent read on the same question. It uses the SAME projection the segment
        # map draws, so a reader comparing the number with the picture is looking at one thing.
        # See clusterability.py for why this is the dip on a principal component rather than on
        # pairwise distances, which is the form the literature recommends and which whole-number
        # survey answers break.
        try:
            _coords, _kept, _ = _charts.pca_2d(_geometry(Xs, cfg)[0])
            self.dip = clusterability.pca_dip(Xs, _coords[:, 0])
        except Exception as e:
            self.dip = {"skipped": f"could not be computed ({type(e).__name__})"}
        self.tendency = clusterability.agreement(self.hopkins, self.dip)
        # The one test in the panel that can return "this is a single group". See
        # supports_single_cluster for why the k>=2 search would otherwise never ask.
        self.single_cluster, self.gap_one, self.gap_two = supports_single_cluster(
            Xs, cfg.gap_B, np.random.default_rng(cfg.random_state + 8), cfg.n_init_search, cfg)
        print(f"Cluster-tendency (Hopkins) = {self.hopkins:.2f} — {hopkins_reading(self.hopkins)}.\n")

        self.diagnostics = selection_diagnostics(Xs, cfg)
        rec_k, rationale, self.signals = recommend_k(self.diagnostics, cfg)
        self.recommended_k = force_k or rec_k
        if force_k:
            rationale += f"\n\n(Overridden by force_k = {force_k}.)"
        print(rationale.replace("**", ""), "\n")

        model, reached = fit_final(X, self.recommended_k, cfg)
        self.labels = model.labels_
        # Consensus ensemble partition (Monti): a resampling-robust alternative partition; report
        # its agreement with the main partition, and optionally adopt it as the final labels.
        self.consensus_agreement = None
        if cfg.run_consensus:
            cons_labels, _ = consensus_partition(X, self.recommended_k, cfg,
                                                 np.random.default_rng(cfg.random_state + 10))
            if cons_labels is None:
                # Too large to place every respondent by consensus. Keep the ordinary partition
                # and say so rather than adopting an ensemble built from part of the study.
                print(f"Consensus ensemble partition: skipped, {len(X):,} respondents exceeds the "
                      f"{MAX_PAIRWISE_N:,} this needs every pair for. The PAC column in the table "
                      "above is estimated on a random subsample; the segmentation itself uses "
                      "everybody.\n")
            else:
                self.consensus_agreement = float(adjusted_rand_score(self.labels, cons_labels))
                if cfg.use_consensus_final:
                    self.labels = cons_labels
                    print("Adopted the consensus ensemble partition as the final segmentation.\n")
        self.split_half = split_half_replication(Xs, self.recommended_k, cfg)
        sil_overall = _silhouette(X, self.labels, cfg)
        # The remaining resampling checks run on the same working set, with the FINAL labels
        # carried across so they judge the segmentation that was actually produced rather than one
        # refitted to the sample. Bootstrap Jaccard alone is 100 clusterings, and it was being
        # given all 48,842 respondents to resample.
        _lab_s = self.labels[self.search_sample] if self.search_sample is not None else self.labels
        self.jaccard = clusterboot_jaccard(Xs, _lab_s, self.recommended_k, cfg)
        self.mb_agreement = model_based_agreement(Xs, self.recommended_k, _lab_s, cfg,
                                                  np.random.default_rng(cfg.random_state + 8))
        self.ward_ari = ward_agreement(Xs, _lab_s, self.recommended_k, cfg)
        self.var_importance = variable_importance(X_raw, self.labels, cfg)
        # Typing tool: the exportable rule for classifying new respondents, plus a leakage-free
        # cross-validated estimate of how reliably the segments can be reproduced out-of-sample.
        self.typing = typing_tool(X_raw.to_numpy(float), self.labels, cfg)
        # Variable-selection check (Dolnicar): does dropping the near-noise items make it cleaner?
        self.varsel = None
        if cfg.check_variable_selection:
            _jv = list(self.jaccard.values())
            self.varsel = variable_selection_check(
                X_raw, self.labels, cfg, self.recommended_k,
                {"split_half": self.split_half, "mean_jaccard": float(np.mean(_jv)),
                 "min_jaccard": float(np.min(_jv)), "silhouette": sil_overall})

        centroids, defining, differentiating, sizes = interpret(X_raw, self.labels, cfg)
        # Weighted population projection: cluster UNWEIGHTED, but report sizes reweighted so an
        # over-sampled subgroup does not distort how big each segment looks in the real population.
        if weights is not None:
            w = np.asarray(weights, float)
            if len(w) != len(self.labels):
                print("NOTE: weights did not align to the respondents; reporting unweighted sizes only.")
            else:
                neg = int((w < 0).sum())
                w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)   # negative/NaN/inf weights -> 0
                if neg:
                    print(f"NOTE: {neg} negative weight(s) treated as zero.")
                if w.sum() > 0:
                    pop = {f"Segment {c}": float(w[self.labels == c].sum() / w.sum())
                           for c in np.unique(self.labels)}
                    sizes["population_share"] = [round(pop.get(idx, 0.0), 3) for idx in sizes.index]
                else:
                    print("NOTE: weights summed to zero; reporting unweighted sizes only.")
        self.centroids, self.sizes = centroids, sizes
        # Every respondent also carries how well they actually fit the segment they were put in.
        #
        # k-means assigns everybody to something — there is no "none of these". James, Witten,
        # Hastie & Tibshirani make the consequence explicit in *An Introduction to Statistical
        # Learning* §12.4.3: "K-means and hierarchical clustering force every observation into a
        # cluster… the clusters found may be heavily distorted due to the presence of outliers
        # that do not belong to any cluster." Without a per-person number this file hands a
        # marketer a tidy segment for the respondent who matched nothing, and it goes to the CRM
        # looking exactly like everyone else.
        #
        # The number is 1 minus Leisch's shadow value: 1 sits squarely inside their own segment,
        # 0 means they are stranded halfway between two and could plausibly belong to either.
        # Filter on it before spending money on a list.
        #
        # Computed from the two nearest centroids rather than as a silhouette, which needs every
        # pairwise distance: O(n·k) against O(n^2), so every respondent gets a real number instead
        # of the largest studies falling back to a 6,000-row sample with the rest left blank.
        # Does each segment survive being asked for a different number of groups?
        self.persistence = stability_across_solutions(
            X, range(cfg.k_min, cfg.k_max + 1), int(self.recommended_k), self.labels,
            cfg.n_init_search, np.random.default_rng(cfg.random_state + 9), cfg)
        _cents = segment_centres(X, self.labels, cfg)
        self.shadow, _closest, _second = shadow_values(X, _cents, cfg)
        self.neighbours = segment_neighbours(self.shadow, _closest, _second,
                                             int(self.recommended_k))
        self.assignments = pd.DataFrame({"id": ids, "segment": self.labels,
                                         "fit": _per_respondent_fit(X, self.labels, _cents, cfg)})
        # Keep the clustered matrix so the charts can show the reader the actual point cloud.
        # A confidence word is a claim; a picture of the data is evidence they can check.
        self.X, self.item_names = X, list(X_raw.columns)
        _votes = [int(v) for v in self.signals.values() if isinstance(v, (int, np.integer))]
        k_agree = (float(np.mean([abs(v - self.recommended_k) <= 1 for v in _votes]))
                   if _votes else None)
        self.report_markdown = make_report(self.diagnostics, self.recommended_k, rationale,
                                            reached, self.split_half, sil_overall, self.jaccard,
                                            sizes, defining, differentiating, centroids,
                                            self.hopkins, self.mb_agreement, self.var_importance,
                                            self.consensus_agreement, cfg, self.typing, self.varsel,
                                            k_agree, self.ward_ari,
                                            getattr(self, 'distinct_share', None),
                                            self.single_cluster, self.gap_one, self.gap_two,
                                            getattr(self, 'neighbours', None),
                                            float(np.median(self.shadow))
                                            if getattr(self, 'shadow', None) is not None
                                            and len(self.shadow) else None,
                                            getattr(self, 'persistence', None),
                                            getattr(self, 'signals', None),
                                            getattr(self, 'dip', None),
                                            getattr(self, 'tendency', None),
                                            k_max_note=getattr(self, 'k_max_note', ''))
        if demographics is not None and (not isinstance(demographics, pd.DataFrame) or not demographics.empty):
            self.report_markdown += "\n\n" + self._profile_external(demographics, id_col)
        if outdir:
            self._save(Path(outdir), X)
        else:
            print(self.report_markdown)      # only dump to stdout when not saved to a file
        return self

    def _profile_external(self, demo_path, id_col):
        return profile_demographics(self.labels, self.assignments["id"].to_numpy(), demo_path,
                                    id_col, unit="segment")

    def typing_rule_dict(self):
        """The portable rule that assigns BRAND-NEW respondents to these segments: the scaling
        parameters plus each segment's centre in scaled space. This is the single source of truth
        for both the saved typing_rule.json and the copy the web app hands out, so the two can
        never drift apart."""
        return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tool_version": __version__,
                "method": self.cfg.method, "scaling": self.cfg.scaling,
                "items": list(self.centroids.columns),
                # What each pick-any code stands for, so a fresh export that still says
                # "Nespresso" can be scored at all. Without it the rule would hold prototypes in
                # codes it could not translate back to answers.
                "level_labels": {c: {str(code): name for code, name in levels.items()}
                                 for c, levels in (self.cfg.level_labels or {}).items()},
                "var_kinds": dict(self.cfg.var_kinds or {}),
                "classes": self.typing["classes"],
                "scale_params": self.typing["scale_params"],
                "scaled_centroids": self.typing["scaled_centroids"].tolist(),
                "cross_validated_accuracy": (None if np.isnan(self.typing["cv_accuracy"])
                                             else round(self.typing["cv_accuracy"], 3))}

    def _save(self, outdir, X):
        outdir.mkdir(parents=True, exist_ok=True)
        self.assignments.to_csv(outdir / "segment_assignments.csv", index=False)
        self.centroids.to_csv(outdir / "segment_centroids.csv")
        self.diagnostics.to_csv(outdir / "k_selection_diagnostics.csv", index=False)
        pd.DataFrame({"segment": [f"Segment {c}" for c in self.jaccard],
                      "mean_jaccard": list(self.jaccard.values())}).to_csv(
            outdir / "segment_stability_jaccard.csv", index=False)
        self.var_importance.to_csv(outdir / "variable_importance.csv", index=False)
        (outdir / "segmentation_report.md").write_text(self.report_markdown, encoding="utf-8")
        write_html_report(self.report_markdown, outdir / "segmentation_report.html",
                          "Segmentation report")
        # Typing rule: the portable classifier for NEW respondents.
        # Apply with: --classify new.csv --rule typing_rule.json
        (outdir / "typing_rule.json").write_text(json.dumps(self.typing_rule_dict(), indent=2),
                                                 encoding="utf-8")
        fig = maybe_plot(self.diagnostics, X, self.labels, self.recommended_k, outdir,
                         self.cfg)
        # Reproducibility manifest: everything needed to reproduce this exact run.
        import sklearn
        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_version": __version__,
            "config": _config_for_manifest(self.cfg),
            "n_respondents": int(len(self.labels)),
            "n_items": int(X.shape[1]),
            "recommended_k": int(self.recommended_k),
            "hopkins_statistic": round(self.hopkins, 3),
            "split_half_ARI": round(self.split_half, 3),
            "model_agreement_ARI": (round(self.mb_agreement["agreement_ARI"], 3)
                                    if not np.isnan(self.mb_agreement["agreement_ARI"]) else None),
            "ward_agreement_ARI": (round(self.ward_ari, 3) if not np.isnan(self.ward_ari) else None),
            "typing_cv_accuracy": (round(self.typing["cv_accuracy"], 3)
                                   if not np.isnan(self.typing["cv_accuracy"]) else None),
            "variable_selection": ({"dropped": self.varsel["dropped"],
                                    "reduced_is_cleaner": self.varsel.get("reduced_is_cleaner")}
                                   if self.varsel and self.varsel.get("applicable") else None),
            "per_segment_jaccard": {f"Segment {c}": round(v, 3) for c, v in self.jaccard.items()},
            "library_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                 "scikit-learn": sklearn.__version__},
        }
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")
        print(f"\nSaved to {outdir}/: assignments, centroids, diagnostics, per-segment Jaccard "
              f"stability, variable importance, report, typing_rule.json, run_manifest.json"
              + (", diagnostics.png" if fig else " (no figure — matplotlib missing)"))


def _make_output_unicode_safe():
    """Let a console that cannot show a character print a question mark instead of dying.

    Python takes its stdout encoding from the locale, and a machine with no LANG — a container, a
    cron job, a minimal CI image — reports ASCII. Everything this tool prints is full of em-dashes,
    Nordic characters and a red/amber/green confidence circle.

    **This is not what fixed the crash that prompted it, and the distinction matters.** Under a
    bare locale the run died with "'ascii' codec can't encode character '\U0001f534'", and that
    came from WRITING THE REPORT FILE: `write_text` with no encoding also falls back to the locale.
    The fix was to name utf-8 at every call site that writes a file, and removing this function
    changes nothing about that failure — verified by reverting each separately.

    What this covers is the neighbouring path: printing to a console that genuinely cannot
    represent the character. `errors="replace"` turns that into a cosmetic loss rather than the end
    of a run whose results were already computed.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").lower().replace("-", "") not in ("utf8", "utf8mb4"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass          # a stream that cannot be reconfigured is not worth failing the run over


def _cli():
    _make_output_unicode_safe()
    p = argparse.ArgumentParser(description="Market-segmentation-grade k-means segmentation of "
                                            "respondents on their preference utilities.")
    p.add_argument("--version", action="version", version=f"segment_kmeans {__version__}")
    p.add_argument("csv", nargs="?", help="respondent x item utilities CSV (omit only with --classify)")
    p.add_argument("--id-col", default=None)
    p.add_argument("--items", nargs="*", default=None)
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=8)
    p.add_argument("--scaling", choices=["range", "standardize", "robust", "none", "ipsative"],
                   default="range")
    p.add_argument("--method", choices=["auto", "kmeans", "gmm", "lca", "kproto"], default="auto",
                   help="auto (default: inspect the file and choose for you), kmeans (heuristic), "
                        "gmm (Gaussian mixture, continuous), lca (Latent Class Analysis, for "
                        "CATEGORICAL / multiple-choice items), or kproto (Gower k-prototypes, for "
                        "a questionnaire holding BOTH rating scales and pick-any questions)")
    p.add_argument("--gmm-covariance", choices=["full", "tied", "diag", "spherical"], default="full")
    p.add_argument("--force-k", type=int, default=None)
    p.add_argument("--no-gmm", action="store_true", help="skip the Gaussian-mixture BIC/ICL cross-check")
    p.add_argument("--no-consensus", action="store_true", help="skip Monti consensus clustering / PAC (faster)")
    p.add_argument("--no-varsel", action="store_true",
                   help="skip the Dolnicar variable-selection check (re-cluster without near-noise items)")
    p.add_argument("--consensus-final", action="store_true",
                   help="adopt the resampling-robust consensus ensemble partition as the final labels")
    p.add_argument("--demographics", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--classify", metavar="NEW_CSV", default=None,
                   help="type NEW respondents into segments using a saved --rule, without "
                        "re-segmenting")
    p.add_argument("--rule", default=None,
                   help="path to a saved typing_rule.json (required with --classify)")
    p.add_argument("--serve", action="store_true",
                   help="start a local web page where anyone can upload a CSV and get the report in "
                        "a browser (no terminal or expertise needed after this)")
    p.add_argument("--app", action="store_true",
                   help="run as the desktop app (like --serve, but picks a free port automatically)")
    p.add_argument("--port", type=int, default=8000, help="port for --serve (default 8000)")
    p.add_argument("--design", metavar="ITEMS_FILE", default=None,
                   help="build a best-worst (MaxDiff) questionnaire: give a text file with one "
                        "item per line and it writes the design a survey platform can field. "
                        "Needs no data — this is what you run BEFORE collecting any")
    p.add_argument("--per-screen", type=int, default=4,
                   help="with --design: how many items on each screen (default 4)")
    p.add_argument("--screens", type=int, default=10,
                   help="with --design: how many screens each person answers (default 10)")
    p.add_argument("--people", type=int, default=200,
                   help="with --design: how many respondents to build versions for (default 200)")
    p.add_argument("--plan", action="store_true",
                   help="how many respondents do you need? Simulates the study you are about to "
                        "run and reports what each sample size can and cannot find. Needs no data")
    p.add_argument("--questions", type=int, default=6,
                   help="with --plan: how many questions the planned survey will have")
    p.add_argument("--segments", type=int, default=3,
                   help="with --plan: how many segments you expect to find")
    p.add_argument("--sizes", nargs="*", type=int, default=None,
                   help="with --plan: sample sizes to try (default 100 200 300 400 600 800)")
    a = p.parse_args()

    if a.design:                     # questionnaire-building mode: run before collecting anything
        import design as _design
        names = [line.strip() for line in
                 Path(a.design).read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) < 3:
            _friendly_fail(p, "That file needs one item per line, and at least three items — "
                              "a best-worst exercise compares things, so there have to be things "
                              "to compare.")
        try:
            built, report = _design.make_design(len(names), a.per_screen, a.screens, a.people)
        except ValueError as e:
            _friendly_fail(p, _explain_run_error(str(e)))
        out = Path(a.outdir) if a.outdir else Path(".")
        out.mkdir(parents=True, exist_ok=True)
        frame = _design.to_frame(built, names)
        frame.to_csv(out / "maxdiff_design.csv", index=False, encoding="utf-8")
        print(_design.render(report))
        print(f"Written to {out / 'maxdiff_design.csv'} — one row per item shown, which is the "
              f"shape most survey platforms import.\n")
        print("Field it, then bring the answers back to this tool: add a 'choice' column saying "
              "best or worst (or 1 and -1) against the item each person picked, and run the "
              "analysis on that file.")
        return

    if a.plan:                       # planning mode: no data, because there is none yet
        import planner
        sizes = tuple(a.sizes) if a.sizes else planner.DEFAULT_SIZES
        print(f"Simulating {len(planner.REGIMES) * len(sizes) * planner.DEFAULT_SEEDS} studies "
              f"through the full analysis. This takes a few minutes.\n")

        def _tick(done, total, cell):
            print(f"  [{done:>2}/{total}] {cell['regime']:<9} {cell['n_people']:>5} people: "
                  f"found the right number {cell['right_k']} of {cell['runs']} times")

        plan = planner.plan_study(n_questions=a.questions, n_segments=a.segments,
                                  sizes=sizes, progress=_tick)
        print("\n" + planner.render(plan))
        return

    if a.app:                        # packaged desktop app: auto free port + open browser
        app()
        return
    if a.serve:                      # zero-terminal front door: a local upload page in the browser
        serve(a.port)
        return

    # Typing mode: apply a saved rule to new respondents and exit (no segmentation).
    if a.classify:
        if not a.rule:
            p.error("--classify requires --rule pointing to a typing_rule.json")
        rule = json.loads(Path(a.rule).read_text(encoding="utf-8"))
        classifier = classify_new_lca if rule.get("method") == "lca" else classify_new
        out = classifier(rule, _read_table(a.classify), id_col=a.id_col)
        if a.outdir:
            Path(a.outdir).mkdir(parents=True, exist_ok=True)
            out.to_csv(Path(a.outdir) / "new_assignments.csv", index=False)
            print(f"Typed {len(out)} new respondent(s) -> {a.outdir}/new_assignments.csv")
        else:
            print(out.to_string(index=False))
        _warn_off_scale(out)
        return
    if not a.csv:
        p.error("a respondent x item utilities CSV is required (or use --classify with --rule)")

    if a.method == "auto":           # the no-expertise front door: detect, choose, explain
        run_auto(a.csv, a, p)
        return

    cfg = SegmentationConfig(k_min=a.kmin, k_max=a.kmax, scaling=a.scaling, method=a.method,
                             gmm_covariance=a.gmm_covariance, fit_gmm_bic=not a.no_gmm,
                             run_consensus=not a.no_consensus, use_consensus_final=a.consensus_final,
                             check_variable_selection=not a.no_varsel, random_state=a.seed)
    if a.method == "lca":            # categorical path: Latent Class Analysis, not k-means
        LatentClassSegmenter(cfg).run(a.csv, id_col=a.id_col, item_cols=a.items,
                                      force_k=a.force_k, outdir=a.outdir)
        return
    if a.method == "kproto":
        # Asked for explicitly rather than detected, so read the column types off the file the
        # same way the detector would: text answers are choices, numbers are ordered ratings.
        frame = _read_table(a.csv)
        chosen = a.items or [c for c in frame.columns if c != a.id_col]
        text = [c for c in chosen if not pd.api.types.is_numeric_dtype(frame[c])]
        if len(chosen) - len(text) < 2:
            p.error("--method kproto needs at least two rating columns alongside the pick-any "
                    "ones; for an all-categorical file use --method lca.")
        prepared = frame[([a.id_col] if a.id_col in frame.columns else []) + chosen].copy()
        plan = {}
        _code_nominals(prepared, text, chosen, plan, frame)
        cfg = replace(cfg, var_kinds=plan.get("kinds"), level_labels=plan.get("level_labels"))
        Segmenter(cfg).run(prepared, id_col=a.id_col, item_cols=chosen, force_k=a.force_k,
                           outdir=a.outdir, demographics=a.demographics)
        return
    Segmenter(cfg).run(a.csv, id_col=a.id_col, item_cols=a.items, force_k=a.force_k,
                       outdir=a.outdir, demographics=a.demographics)


if __name__ == "__main__":
    _cli()
