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
import tempfile
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
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass                     # older Python, or a stream that is not reconfigurable


_use_utf8_for_output()

# numpy 2.0 on macOS/Accelerate emits spurious "... encountered in matmul" RuntimeWarnings from
# ordinary matrix products inside scikit-learn; they do not affect results. Filter only these.
for _m in ("divide by zero encountered in matmul", "overflow encountered in matmul",
           "invalid value encountered in matmul"):
    warnings.filterwarnings("ignore", message=_m, category=RuntimeWarning)

__version__ = "1.0.0"    # keep in sync with pyproject.toml

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
    # "kmeans" (heuristic) or "gmm" (model-based / finite-mixture / latent-class, Wedel & Kamakura)
    method: str = "kmeans"
    gmm_covariance: str = "full"    # "full" | "tied" | "diag" | "spherical" (gmm method only)
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
def _read_table(source):
    """Read a survey export ROBUSTLY, the way real exports actually arrive. Handles comma OR
    semicolon delimiters (European / Excel exports, e.g. Swedish locale), UTF-8, UTF-8-with-BOM,
    and Latin-1 (so aao survive), and .xlsx/.xls if openpyxl is available. Accepts a path, raw
    bytes, a file-like object, or an existing DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    is_excel = ((isinstance(source, str) and source.lower().endswith((".xlsx", ".xls")))
                or (isinstance(source, (bytes, bytearray)) and bytes(source[:2]) == b"PK"))
    if is_excel:
        try:
            return pd.read_excel(io.BytesIO(source) if isinstance(source, (bytes, bytearray))
                                 else source)
        except ImportError:
            raise ValueError("_NEED_OPENPYXL")
    def _buf():
        if isinstance(source, (bytes, bytearray)):
            return io.BytesIO(source)
        if hasattr(source, "seek"):
            source.seek(0)
        return source
    read_errors = (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error)
    for enc in ("utf-8-sig", "latin-1"):            # utf-8-sig also decodes plain UTF-8
        try:
            df = pd.read_csv(_buf(), sep=None, engine="python", encoding=enc)   # sep=None sniffs , ; \t
            if df.shape[1] >= 1 and len(df):
                return df
        except read_errors:
            continue
    try:                                            # last resort: plain comma, skip unparsable lines
        df = pd.read_csv(_buf(), encoding="latin-1", on_bad_lines="skip")
        if df.shape[1] >= 1 and len(df):
            return df
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

    if X.isna().any().any():
        if cfg.impute == "drop":
            keep = ~X.isna().any(axis=1)
            X, ids = X.loc[keep], ids[keep.to_numpy()]
            print(f"Dropped {(~keep).sum()} rows with missing values.")
        else:
            X = X.fillna(X.mean())
            print("Imputed missing cells with the item mean.")

    nonconst = X.std(axis=0) > 1e-12
    if not nonconst.all():
        print(f"Dropping constant item(s): {list(X.columns[~nonconst])}")
        X = X.loc[:, nonconst]
    X_raw = X.reset_index(drop=True)
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


def _pooled_within_ss(X, labels):
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) > 1:
            total += ((pts - pts.mean(0)) ** 2).sum()
    return total


# =====================================================================================
# k-selection diagnostics
# =====================================================================================
def gap_statistic(X, k_range, B, rng, n_init):
    """Tibshirani-Walther-Hastie (2001), reference method (b): uniform over the bounding box
    of the data rotated to its principal components, then rotated back — the recommended,
    shape-aware reference distribution. Recommended k: smallest with gap(k) >= gap(k+1)-se(k+1)."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ Vt.T
    lo, hi = Xp.min(0), Xp.max(0)
    rows = []
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
    far less sensitive to initialization than a single k-means run (Monti et al. 2003)."""
    C = consensus_matrix(X, k, cfg, rng)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0                                  # enforce exact symmetry for squareform
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, k, criterion="maxclust") - 1
    return labels, C


def hopkins_statistic(X, rng, m_frac=0.10):
    """Cluster-TENDENCY pre-check (Lawson & Jurs 1990; Banerjee & Dave 2004): should you even
    cluster these data? Compare nearest-neighbour distances of real points to those of uniform
    random points over the data's bounding box. H = sum(u) / (sum(u) + sum(w)), where u are
    random-point-to-data distances and w are data-point-to-data distances. Reading: H ~ 0.5 =
    random (no cluster tendency); H > 0.75 = strong tendency to cluster; H < 0.5 = regularly
    spaced. Sampling a small fraction (default 10%) keeps the test valid."""
    n, d = X.shape
    m = max(5, int(m_frac * n))
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    w = nbrs.kneighbors(X[rng.choice(n, m, replace=False)], n_neighbors=2)[0][:, 1]  # skip self
    U = rng.uniform(X.min(0), X.max(0), size=(m, d))
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
    mixture's assignment confidence (mean top posterior) and normalized entropy (0 = crisp, 1 = fuzzy)."""
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


def ward_agreement(X, labels, k):
    """Third cross-check: does Ward agglomerative clustering (structurally different from k-means
    and the mixture: it merges bottom-up, not around centroids) recover the same partition? Three
    different methods agreeing is strong evidence the structure is real, not an artefact of one
    algorithm. Skipped for very large n (the linkage is O(n^2))."""
    if len(X) > 3000:
        return float("nan")
    try:
        wl = fcluster(linkage(X, method="ward"), t=k, criterion="maxclust")
        return float(adjusted_rand_score(labels, wl))
    except Exception:
        return float("nan")


def variable_importance(X_raw, labels):
    """Which items actually drive the segmentation, and which are noise? Reports eta-squared per
    item (between-segment sum of squares / total sum of squares) — the share of an item's
    variance explained by segment membership. Items with near-zero eta-squared add noise and,
    per Dolnicar's variable-selection work, can mask real structure; consider dropping them and
    re-running."""
    items = list(X_raw.columns)
    rows = []
    for it in items:
        y = X_raw[it].to_numpy(float)
        grand = y.mean()
        ss_tot = ((y - grand) ** 2).sum()
        ss_between = sum(len(y[labels == c]) * (y[labels == c].mean() - grand) ** 2
                         for c in np.unique(labels))
        eta2 = ss_between / ss_tot if ss_tot > 0 else 0.0
        rows.append({"item": it, "eta_squared": round(eta2, 3)})
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
    vi = variable_importance(X_raw, labels)
    noise = vi.loc[vi["role"].str.startswith("near-noise"), "item"].tolist()
    signal = [c for c in X_raw.columns if c not in noise]
    if not noise or len(signal) < 2:
        return {"applicable": False, "dropped": noise, "n_signal": len(signal)}
    Xs, _ = _scale_fit(X_raw[signal].to_numpy(float), cfg.scaling)
    lab = fit_final(Xs, k, cfg)[0].labels_
    jac = list(clusterboot_jaccard(Xs, lab, k, cfg).values())
    reduced = {"split_half": split_half_replication(Xs, k, cfg),
               "mean_jaccard": float(np.mean(jac)), "min_jaccard": float(np.min(jac)),
               "silhouette": float(silhouette_score(Xs, lab))}
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
        base.append({"k": k, "inertia": _pooled_within_ss(X, lab),
                     "silhouette": silhouette_score(X, lab),
                     "calinski_harabasz": calinski_harabasz_score(X, lab),
                     "davies_bouldin": davies_bouldin_score(X, lab),
                     # Share of respondents in the SMALLEST segment. A statistically tidy
                     # solution made of two-person segments cannot be marketed to, so this is
                     # what makes cfg.min_segment_frac enforceable rather than decorative.
                     "min_segment_share": float(np.bincount(lab, minlength=k).min() / len(lab))})
    diag = pd.DataFrame(base)
    diag = diag.merge(gap_statistic(X, k_range, cfg.gap_B,
                                    np.random.default_rng(cfg.random_state + 1), cfg.n_init_search), on="k")
    diag = diag.merge(replication_stability(X, k_range, cfg.stability_B, cfg.stability_frac,
                                            np.random.default_rng(cfg.random_state + 2), cfg), on="k")
    diag = diag.merge(prediction_strength(X, k_range, cfg.ps_splits,
                                          np.random.default_rng(cfg.random_state + 3), cfg), on="k")
    if cfg.fit_gmm_bic or cfg.method == "gmm":
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
    # Global stability: largest k that is still "stable"; else the most stable k
    stab_ok = diag[diag["stability_ARI"] >= cfg.stability_cutoff]
    signals["global stability"] = int(stab_ok["k"].max()) if len(stab_ok) \
        else int(diag.loc[diag["stability_ARI"].idxmax(), "k"])
    if "gmm_BIC" in diag and diag["gmm_BIC"].notna().any():
        signals["GMM BIC (model-based)"] = int(diag.loc[diag["gmm_BIC"].idxmin(), "k"])
    if "gmm_ICL" in diag and diag["gmm_ICL"].notna().any():
        signals["GMM ICL (model-based)"] = int(diag.loc[diag["gmm_ICL"].idxmin(), "k"])
    if "consensus_PAC" in diag and diag["consensus_PAC"].notna().any():
        signals["consensus PAC"] = int(diag.loc[diag["consensus_PAC"].idxmin(), "k"])

    # Weighted vote: stability signals and prediction strength count double. When the chosen
    # model IS the Gaussian mixture, the model-based criteria (BIC, ICL) are the primary basis
    # for the number of components, so they count double too.
    mb_w = 2 if getattr(cfg, "method", "kmeans") == "gmm" else 1
    weights = {"prediction strength": 2, "global stability": 2, "consensus PAC": 2,
               "silhouette": 1, "Calinski-Harabasz": 1, "Davies-Bouldin": 1, "gap": 1,
               "GMM BIC (model-based)": mb_w, "GMM ICL (model-based)": mb_w, "elbow (weak)": 0}
    tally = {}
    for name, k in signals.items():
        tally[k] = tally.get(k, 0) + weights.get(name, 1)
    best_score = max(tally.values())
    winners = sorted([k for k, s in tally.items() if s == best_score])
    pick = winners[0]   # ties -> smaller, more interpretable k (Dolnicar: parsimony)

    ruled_out = ""
    if excluded:
        ruled_out = ("\n\nRuled out before the vote: k = "
                     + ", ".join(str(k) for k in excluded)
                     + f" — each of those splits the sample into at least one segment holding "
                       f"under {cfg.min_segment_frac:.0%} of respondents, which is too small to "
                       "target even if the statistics look clean.")
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
                 "natural segments, and the right move may be a smaller k or a different method.")
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
def interpret(X_raw, labels, cfg):
    items = list(X_raw.columns)
    seg = pd.Series(labels, name="segment")
    centroids = X_raw.groupby(seg).mean()
    centroids.index = [f"Segment {c}" for c in centroids.index]
    grand = X_raw.mean()
    defining = {}
    for c in np.unique(labels):
        diff = (X_raw[labels == c].mean() - grand).sort_values(ascending=False)
        # With few items, head(top_items) and tail(top_items) would overlap and print the same
        # items as both "values most" and "values least"; cap each side to a non-overlapping half.
        half = max(1, min(cfg.top_items, len(diff) // 2))
        defining[f"Segment {c}"] = {
            "most_above_average": [f"{it} ({diff[it]:+.1f})" for it in diff.head(half).index],
            "most_below_average": [f"{it} ({diff[it]:+.1f})" for it in diff.tail(half).index],
            "auto_name": " + ".join(_short_label(t) for t in diff.head(2).index)}
    groups = [X_raw[labels == c] for c in np.unique(labels)]
    fvals = {}
    for it in items:
        try:
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


def _short_label(text, max_chars=26):
    """Turn a raw column header ('I want to meet people in real life') into a short, human label
    ('want meet people real') without cutting a word in half — the auto-names a non-expert sees."""
    t = str(text).replace("_", " ").strip().rstrip("?").strip()
    words = [w for w in t.split() if w.lower() not in _LABEL_STOP] or t.split()
    out = ""
    for w in words:
        if out and len(out) + len(w) + 1 > max_chars:
            break
        out = (out + " " + w).strip()
    return out or t[:max_chars]


def _plural(unit):
    """English plural for the unit words we use ('group'->'groups', 'class'->'classes')."""
    return unit + ("es" if unit.endswith(("s", "x", "z", "ch", "sh")) else "s")


def _fraction_phrase(share):
    """A non-analyst reads 'about 1 in 3' faster than '0.34'."""
    pct = round(share * 100)
    if share >= 0.45:
        return f"about half ({pct}%)"
    d = min(range(3, 11), key=lambda d: abs(share - 1 / d))
    return f"about 1 in {d} ({pct}%)"


def executive_summary(n_resp, names, shares, wants, min_jaccard, repro, unit="group",
                      k_agreement=None):
    """The plain-language box at the top of every report: how many groups, who they are, how much
    to trust it (a green/amber/red confidence light built from the stability numbers), and what to
    do next. Written for someone who will never read the word 'eta-squared'.

    The light is deliberately conservative: a solution can be individually reproducible yet still
    have an uncertain NUMBER of groups (the selection criteria disagreed). When k_agreement is low
    we refuse to show green, because 'high confidence in 6 groups' misleads when the count could as
    easily have been 3.

    Both stability measures have to agree before the light goes above red. They fail in different
    ways and the amber band used to consult only the first: bootstrap Jaccard sits around 0.7 even
    on structureless data (Hennig's own reading of that band is "a pattern, membership doubtful"),
    so on random answers it alone would report Moderate and the wording would claim the groups
    reproduce while split-half replication said 0.06. Split-half is the measure that actually
    answers "would a fresh sample find these same groups", so it now gates the amber band too."""
    k_contested = k_agreement is not None and k_agreement < 0.6
    if min_jaccard >= 0.75 and repro >= 0.6 and not k_contested:
        light, label, meaning = "🟢", "High", ("the groups are clear and they reproduce reliably "
                                               "when the analysis is repeated.")
    elif min_jaccard >= 0.6 and repro >= 0.4:
        light, label, meaning = "🟡", "Moderate", ("the groups mostly hold up, but " + (
            "the methods disagree on how many groups there really are, so the exact number is "
            "uncertain." if k_contested else "their edges are fuzzy, so treat them as a strong "
            "hypothesis."))
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
         f"We looked at **{n_resp} people** and found **{len(names)} {_plural(unit)}** "
         "(a working number, not the only possibility):\n" if k_contested else
         f"We looked at **{n_resp} people** and found **{len(names)} {_plural(unit)}**:\n"]
    for i in sorted(range(len(names)), key=lambda i: -shares[i]):
        L.append(f"- **{names[i]}** ({_fraction_phrase(shares[i])} of people) stand out for: {wants[i]}.")
    L.append(f"\n**What to do next.** Start with the biggest, most distinct {unit}. Give the "
             f"{_plural(unit)} names your team recognises (the ones above are generated automatically). "
             "Everything below is the supporting detail and the confidence checks.")
    if label != "High":
        L.append(f"\n> Because confidence is {label}, treat these {_plural(unit)} as a direction to test "
                 "with a few interviews, not a settled fact.")
    return "\n".join(L) + "\n"


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

def make_report(diag, rec_k, rationale, reached, split_half, sil_overall, jaccard,
                sizes, defining, differentiating, centroids, hopkins, mb_agreement,
                var_importance, consensus_agreement, cfg, typing=None, varsel=None,
                k_agreement=None, ward_ari=None, distinct_share=None):
    method_name = ("a Gaussian mixture / latent-class model (" + cfg.gmm_covariance +
                   " covariance)" if getattr(cfg, "method", "kmeans") == "gmm" else "k-means")
    _keys = list(defining.keys())
    _names = [defining[k]["auto_name"] for k in _keys]
    _wants = [", ".join(_short_label(w.split(" (")[0]) for w in defining[k]["most_above_average"][:2])
              or "a distinct mix of answers" for k in _keys]
    _shares = [float(sizes.loc[k, "share"]) for k in _keys]
    L = ["# Segmentation report\n",
         executive_summary(int(sizes["n"].sum()), _names, _shares, _wants,
                           min(jaccard.values()), split_half, unit="group", k_agreement=k_agreement),
         f"Respondents clustered with **{method_name}** on **{cfg.scaling}**-scaled utilities; "
         f"final fit used {cfg.n_init_final} restarts. Search range: k = "
         f"{cfg.k_min} to {cfg.k_max}.\n",
         "## Is there anything to segment? (cluster tendency)\n",
         f"Hopkins statistic = **{hopkins:.2f}** — {hopkins_reading(hopkins)}. "
         "A value near 0.5 means the data are essentially random and any segments will be "
         "constructed by the method rather than discovered; above ~0.75 signals a real tendency "
         "to cluster. Read the rest of this report in that light.\n"
         + _hopkins_caveat(distinct_share, centroids.shape[1] if centroids is not None else 0),
         "## Choosing the number of segments\n", rationale, "\n",
         _md(diag.round(3)),
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
    jac = pd.DataFrame({"segment": [f"Segment {c}" for c in jaccard],
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
          _md(sizes, index=True),
          "\n## The mind-sets (what defines each segment)\n"]
    for seg, d in defining.items():
        L.append(f"**{seg}** — suggested name: *{d['auto_name']}*")
        L.append(f"  - Values most: {', '.join(d['most_above_average'])}")
        L.append(f"  - Values least: {', '.join(d['most_below_average'])}\n")
    L += ["## What differentiates the segments (one-way ANOVA F; high = splits them most)\n",
          _md(differentiating.head(10).round(2)),
          "\n## Which items drive the segmentation (variable importance)\n",
          "Eta-squared is the share of each item's variance explained by segment membership. "
          "Near-zero items add noise and can mask real structure (Dolnicar's variable-selection "
          "point) — consider dropping the near-noise items and re-running.\n",
          _md(var_importance),
          _varsel_section(varsel, rec_k),
          "\n## Segment-by-item mean utilities (centroids, on the raw scale)\n",
          _md(centroids.round(1), index=True),
          "\n---\n**Methodology.** Number of segments chosen by a weighted panel (prediction "
          "strength and replication stability first, then separation indices, then the gap "
          "statistic) rather than a single elbow; per-segment validity judged by bootstrap "
          "Jaccard stability (Hennig 2007). Range standardization follows Milligan & Cooper "
          "(1988). A reminder from Dolnicar & Leisch: data-driven segments are usually "
          "*constructed* by the method, not discovered — so trust the stability columns, and "
          "rename the auto-suggested mind-set names to something a non-analyst would recognise "
          "before shipping. Demographics were not used to form the segments; profile them "
          "separately (below, if a demographics file was supplied).",
          f"\n*Generated by segment_kmeans version {__version__}.*"]
    return "\n".join(x for x in L if x is not None)


def maybe_plot(diag, X, labels, rec_k, outdir):
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
    sv = silhouette_samples(X, labels); y = 10
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
# layer all reach for `sk.chart_radar` and friends, and moving the drawing into its own file
# should not change what any caller imports.
import charts as _charts
from charts import build_charts

SEG_COLOURS = _charts.SEG_COLOURS
seg_colour = _charts.seg_colour
pca_2d = _charts.pca_2d
chart_segment_map = _charts.chart_segment_map
chart_silhouette = _charts.chart_silhouette
chart_k_choice = _charts.chart_k_choice
chart_profiles = _charts.chart_profiles
chart_radar = _charts.chart_radar
chart_heatmap = _charts.chart_heatmap
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
    Path(path).write_text(_html_document(markdown_text, title))
    return str(path)


# =====================================================================================
# Typing tool: assign NEW respondents to segments, and measure how reliably that can be done
# =====================================================================================
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

    def _centroids(Xs, y):
        return np.vstack([Xs[y == c].mean(0) for c in classes])

    def _assign(Xte, cents):
        d = ((Xte[:, None, :] - cents[None, :, :]) ** 2).sum(2)
        return classes[d.argmin(1)]

    if min_class < 2 or n < 10:                      # too little data for honest cross-validation
        cv_acc = float("nan"); recalls = {int(c): float("nan") for c in classes}
    else:
        n_splits = int(min(5, min_class))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        correct = np.zeros(k); total = np.zeros(k); n_ok = 0
        for tr, te in skf.split(arr_raw, labels):
            Xtr, p = _scale_fit(arr_raw[tr], cfg.scaling)      # refit scaling on the train fold only
            pred = _assign(_scale_apply(arr_raw[te], p), _centroids(Xtr, labels[tr]))
            n_ok += int((pred == labels[te]).sum())
            for i, c in enumerate(classes):
                m = labels[te] == c
                total[i] += int(m.sum()); correct[i] += int((pred[m] == c).sum())
        cv_acc = float(n_ok / n)
        recalls = {int(c): (float(correct[i] / total[i]) if total[i] else float("nan"))
                   for i, c in enumerate(classes)}

    Xs_full, params = _scale_fit(arr_raw, cfg.scaling)         # the exported rule, fit on all data
    return {"cv_accuracy": cv_acc,
            "baseline_majority": float(counts.max() / n),      # "always guess the biggest segment"
            "per_segment_recall": recalls,
            "scaled_centroids": _centroids(Xs_full, labels),
            "scale_params": params,
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
    if s == "standardize":
        return np.asarray(params["mean"], float)
    if s == "robust":
        return np.asarray(params["median"], float)
    if s == "range":
        return np.asarray(params["lo"], float) + np.asarray(params["range"], float) / 2.0
    return None


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
    sub = df[items].copy()
    for c in items:
        if not pd.api.types.is_numeric_dtype(sub[c]):
            rec = _try_likert(sub[c])
            if rec is None:
                raise ValueError(f"_UNSCORABLE_ITEM:{c}")
            sub[c] = rec
    arr = sub.to_numpy(float)
    arr = np.where(np.isinf(arr), np.nan, arr)
    if np.isnan(arr).any():
        centre = _training_centre(rule["scale_params"])
        if centre is None:                   # row-local scaling — no fitted centre exists to use
            seen = (~np.isnan(arr)).sum(0)   # column means without nanmean's empty-slice warning
            centre = np.where(seen > 0, np.nansum(arr, 0) / np.maximum(seen, 1), 0.0)
        arr = np.where(np.isnan(arr), centre, arr)
    Xs = _scale_apply(arr, rule["scale_params"])
    cents = np.asarray(rule["scaled_centroids"], float); classes = np.asarray(rule["classes"])
    d = np.sqrt(((Xs[:, None, :] - cents[None, :, :]) ** 2).sum(2))
    inv = 1.0 / (d + 1e-9)
    out = pd.DataFrame({"segment": classes[d.argmin(1)],
                        "confidence": (inv.max(1) / inv.sum(1)).round(3)})
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
        if demo[col].dtype == object or demo[col].nunique() <= 12:
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


def latent_class_report(diag, rec_k, rationale, model, jaccard, names, level_labels, labels, cfg,
                        typing=None, weighted_share=None):
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
         f"**{norm_ent:.2f}** (0 = crisp, 1 = fuzzy).\n"]
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
        self.report_markdown = latent_class_report(self.diagnostics, self.recommended_k, rationale,
                                                   self.model, self.jaccard, names, level_labels,
                                                   self.labels, cfg, typing=self.typing,
                                                   weighted_share=weighted_share)
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
        (outdir / "latent_class_report.md").write_text(self.report_markdown)
        write_html_report(self.report_markdown, outdir / "latent_class_report.html",
                          "Latent class segmentation report")
        # Typing rule: the portable classifier for NEW respondents.
        # Apply with --classify ... --rule <this file>.
        (outdir / "latent_class_typing_rule.json").write_text(
            json.dumps(self.typing_rule_dict(), indent=2))
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
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
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
    {"instämmer inte alls": 1, "instämmer inte": 2, "varken eller": 3,
     "varken instämmer eller inte": 3, "neutral": 3, "instämmer delvis": 4, "instämmer": 4,
     "instämmer helt": 5, "instämmer helt och hållet": 5},
    {"håller inte alls med": 1, "håller inte med": 2, "varken eller": 3, "neutral": 3,
     "håller med": 4, "håller helt med": 5},
    {"mycket missnöjd": 1, "missnöjd": 2, "varken eller": 3, "neutral": 3, "nöjd": 4,
     "mycket nöjd": 5},
    {"aldrig": 1, "sällan": 2, "ibland": 3, "ofta": 4, "mycket ofta": 5, "alltid": 5},
    {"mycket osannolikt": 1, "osannolikt": 2, "neutral": 3, "sannolikt": 4, "mycket sannolikt": 5},
    {"inte alls viktigt": 1, "inte viktigt": 2, "neutral": 3, "viktigt": 4, "mycket viktigt": 5},
]


def _norm(v):
    return str(v).strip().lower()


def _try_likert(series):
    """If every non-missing answer maps under one known agree/disagree-style scale, return the
    recoded 1-5 series; otherwise None. Missing tokens are tolerated and become NaN."""
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
    if len(vals) < 8 or vals.str.contains(",").mean() < 0.15:
        return None                                   # nothing is multi-answer: not this shape
    items = []
    for v in vals:
        items.extend([p.strip() for p in v.split(",") if p.strip()])
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
        for part in {p.strip() for p in whole.split(",") if p.strip()}:
            parents.setdefault(part, set()).add(whole)
    if float(np.median([len(v) for v in parents.values()])) < 2:
        return None
    return list(counts.index)


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
               # Nordic equivalents — Swedish-language surveys are common, and a mis-detected 'Kön' or
               # 'Universitet' would define the segments instead of describing them.
               "kön", "kjønn", "ålder", "alder", "universitet", "högskola", "hogskola",
               "lärosäte", "larosate", "land", "hemland", "nationalitet", "medborgarskap",
               "stad", "ort", "studieort", "kommun", "utbildning", "fakultet", "institution",
               "årskurs", "arskurs", "termin", "inkomst", "examen", "studieprogram"}
_DEMO_PHRASES = ("study year", "year of study", "class year", "study programme", "study program")
# A survey weight (e.g. post-stratification / design weight). Cluster UNWEIGHTED, but project the
# segment SIZES to the population with these weights (studies often pool strata with weights).
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
            "skipped": [], "recoded": {}, "multiselect": {}, "notes": []}
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
        # A row counter or a second identifier column is bookkeeping, not an answer. Clustering on
        # one silently injects a fake gradient, so set it aside before the numeric branch below.
        if _looks_like_index(s) or _looks_like_id(s, name):
            index_like.append(c)
            plan["skipped"].append(c)
            plan["notes"].append(f"'{c}': skipped (a row number or record id, not an answer)")
            continue
        if pd.api.types.is_numeric_dtype(s):
            plan["continuous"].append(c); plan["notes"].append(f"'{c}': number ratings, used as-is"); continue
        rec = _try_likert(s)
        if rec is not None:
            plan["continuous"].append(c); plan["recoded"][c] = rec
            plan["notes"].append(f"'{c}': agree/disagree scale, converted to 1-5"); continue
        opts = _multiselect_options(s)
        if opts is not None:
            plan["multiselect"][c] = opts
            plan["notes"].append(f"'{c}': a select-all question — split into {len(opts)} yes/no "
                                 f"columns ({', '.join(opts[:4])}"
                                 + (", ..." if len(opts) > 4 else "") + ")")
            continue
        if 2 <= nun <= max(12, int(0.25 * len(s))):
            plan["categorical"].append(c); plan["notes"].append(f"'{c}': multiple-choice answers")
        else:
            plan["skipped"].append(c); plan["notes"].append(f"'{c}': skipped ({nun} different answers, looks like free text)")
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
        picked = df[col].fillna("").astype(str).apply(
            lambda v: {p.strip() for p in v.split(",") if p.strip()})
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
        # All-numeric picks are continuous utilities (k-means); a mixed or text pick is categorical.
        method = "kmeans" if numeric == len(chosen) else "lca"
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
    if len(cont) >= 2:
        method, items = "kmeans", cont
        for c in items:
            clean[c] = plan["recoded"].get(c, df[c])
        if cat:
            plan["notes"].append(f"Set aside {len(cat)} multiple-choice column(s) and grouped people "
                                 "by the rating questions (the richer signal).")
    elif len(cat) >= 2:
        method, items = "lca", cat
        for c in items:
            clean[c] = df[c]
    else:
        raise ValueError("_AUTO_NO_ITEMS")
    return clean, method, plan["id"], items, plan


def _detection_summary(plan, method, n_items, n_resp):
    how = {"kmeans": "grouping people by their rating answers (k-means)",
           "lca": "grouping people by their multiple-choice answers (Latent Class Analysis)"}[method]
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
    cfg = SegmentationConfig(method=method, random_state=args.seed)
    try:
        if method == "lca":
            LatentClassSegmenter(cfg).run(clean, id_col=id_col, item_cols=items, outdir=outdir,
                                          demographics=demo_df, weights=weights)
        else:
            Segmenter(cfg).run(clean, id_col=id_col, item_cols=items, outdir=outdir,
                               demographics=demo_df, weights=weights)
    except ValueError as e:
        _friendly_fail(parser, _explain_run_error(str(e)))
    report = "latent_class_report" if method == "lca" else "segmentation_report"
    print(f"\nAll done. Open '{outdir}/{report}.html' in your browser to read the results "
          "(or the .md version in any text editor). The plain-language summary is at the top.")


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
    if _maxdiff is not None and _maxdiff.looks_like_maxdiff(df):
        est = _maxdiff.utilities_from_export(df)
        df = est.as_frame().reset_index().rename(columns={"index": "respondent_id"})
        maxdiff_note = (
            f"This file is a MaxDiff (best-worst) export, not a rating grid, so it was scored "
            f"first: individual-level utilities for {len(est.item_names)} items were estimated "
            f"for {len(est.respondent_ids)} respondents by Hierarchical Bayes, and the groups "
            f"below are built on those utilities. Counting how often each item was picked best "
            f"minus worst would have been too coarse to describe a single person.")
    clean, method, id_col, items, plan = auto_prepare(df, force_items=force_items)
    base = replace(cfg, method=method) if cfg is not None else SegmentationConfig(method=method)
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
    return {"title": title, "method": method,
            "report_html": notes_html + _markdown_to_html(seg.report_markdown),
            "digest": notes_md + "\n" + seg.report_markdown,
            "files": files, "n_people": int(len(seg.assignments)),
            "k": int(seg.recommended_k), "columns": roles,
            "charts": build_charts(seg, method),
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
        tmp = path.with_suffix(path.suffix + ".tmp")
        if isinstance(text_or_bytes, bytes):
            tmp.write_bytes(text_or_bytes)
        else:
            tmp.write_text(text_or_bytes)
        tmp.replace(path)                                         # never leave a half-written file

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
            saved = json.loads(p.read_text())
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
                d = json.loads(f.read_text())
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
                  "n_people", "confidence", "transcript", "raw", "names", "charts")

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
                         "charts": s.get("charts", []), "names": s.get("names", [])},
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
                self._json(_ai.status() if _ai else {"sdk_installed": False, "configured": False,
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
                st = _ai.status() if _ai else {}
                self._json({"ok": True, "session_id": pid, "title": saved.get("title"),
                            "report_html": saved.get("report_html") or "",
                            "downloads": sorted(saved.get("files") or {}),
                            "k": saved.get("k"), "n_people": saved.get("n_people"),
                            "columns": saved.get("columns") or {},
                            "confidence": saved.get("confidence") or "unknown",
                            "charts": _charts_for_browser(saved.get("charts")),
                            "names": saved.get("names") or [],
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
                self._json({"ok": False, "error": _explain_run_error(str(e))})

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
                r = run_analysis(data)
            except Exception as e:
                self._json({"ok": False, "error": _explain_run_error(str(e))})
                return
            sid = uuid.uuid4().hex
            # Keep the raw upload so the user can re-group on different questions without
            # re-uploading; bounded below so memory cannot creep.
            sessions[sid] = {"digest": r["digest"], "messages": [], "files": r["files"],
                             "raw": data, "title": filename or r["title"],
                             "report_html": r["report_html"], "columns": r.get("columns", {}),
                             "k": r["k"], "n_people": r["n_people"],
                             "confidence": r.get("confidence"), "charts": r.get("charts", []),
                             "transcript": [{"role": "you", "text": f"Analyse: {filename}"}]}
            for old in list(sessions)[:-5]:    # bound memory: sessions hold the file + its results
                sessions.pop(old, None)
            remember(sid)
            self._json(self._analysis_payload(sid, r))

        def _analysis_payload(self, sid, r):
            st = _ai.status() if _ai else {}
            return {"ok": True, "session_id": sid, "title": r["title"],
                    "report_html": r["report_html"],
                    "ai_available": bool(st.get("configured") and st.get("sdk_installed")),
                    "downloads": sorted(r["files"]), "k": r["k"], "n_people": r["n_people"],
                    "columns": r.get("columns", {}),
                    "charts": _charts_for_browser(r.get("charts")),
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
                r = run_analysis(sess["raw"], force_items=items)
            except Exception as e:
                self._json({"ok": False, "error": _explain_run_error(str(e))})
                return
            # Replace the whole stored result, not just part of it: leaving the old report_html and
            # counts behind would mean reopening the project showed the PREVIOUS grouping.
            # Names describe the OLD groups and there may now be a different number of them, so they
            # are dropped rather than silently re-applied to groups that mean something else.
            sess.update({"digest": r["digest"], "files": r["files"], "messages": [], "names": [],
                         "report_html": r["report_html"], "columns": r.get("columns", {}),
                         "k": r["k"], "n_people": r["n_people"],
                         "confidence": r.get("confidence"), "charts": r.get("charts", []),
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
                classifier = classify_new_lca if rule.get("method") == "lca" else classify_new
                out = classifier(rule, _read_table(data))
            except Exception as e:
                self._json({"ok": False, "error": _explain_run_error(str(e))})
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
            self._json({"ok": True, "n": int(len(out)),
                        "breakdown": {str(k): int(v) for k, v in counts.items()},
                        "mean_confidence": round(float(out["confidence"].mean()), 2),
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
                if _ai is None:
                    self._json({"ok": False, "kind": "nosdk",
                                "error": "The AI add-on is not installed, so I cannot suggest "
                                         "names. You can still type your own."})
                    return
                try:
                    names = _ai.suggest_names(sess["digest"], len(groups))
                except _ai.AIError as e:
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
            sess["names"] = names
            remember(body.get("session_id"))             # names must survive reopening the project
            self._json({"ok": True, "names": names, "downloads": sorted(sess["files"])})

        def _do_chat(self):
            body = self._read_json()
            sess = session(body.get("session_id"))
            if not sess:
                self._json({"ok": False, "error": "Please analyse a survey file first."})
                return
            if _ai is None:
                self._json({"ok": False, "kind": "nosdk", "error": "The AI add-on is not installed. "
                            "Install it with  pip install anthropic  (or rebuild the app), then "
                            "reopen this page."})
                return
            question = None if body.get("initial") else (body.get("message") or "").strip()
            try:
                # The charts go with the digest so Claude reads the same picture the user is
                # looking at — the segment map in particular shows whether the groups separate,
                # which no amount of summary statistics conveys as directly.
                reply, sess["messages"] = _ai.chat_once(sess["messages"], sess["digest"], question,
                                                        charts=sess.get("charts"))
            except _ai.AIError as e:
                self._json({"ok": False, "kind": e.kind, "error": str(e)})
                return
            html = _markdown_to_html(reply)
            tr = sess.setdefault("transcript", [])
            if question:
                tr.append({"role": "you", "text": question})
            tr.append({"role": "ai", "html": html})
            remember(body.get("session_id"))
            self._json({"ok": True, "reply_html": html})

        def _do_settings(self):
            if _ai is None:
                self._json({"ok": False, "error": "The AI add-on is not installed."})
                return
            body = self._read_json()
            try:
                if body.get("clear"):
                    _ai.clear_api_key()
                else:
                    _ai.save_api_key(body.get("api_key", ""))
            except _ai.AIError as e:
                self._json({"ok": False, "error": str(e)})
                return
            resp = {"ok": True}
            resp.update(_ai.status())
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
        max_valid_k = max(2, min(n // 2, n_distinct))
        # Kept for the report: Hopkins is inflated when many respondents share an identical
        # answer pattern, so the reader needs to know how common that is here.
        self.distinct_share = float(len(np.unique(_Xa, axis=0)) / n)
        if cfg.k_min > max_valid_k:
            raise ValueError(f"k_min={cfg.k_min} is too large for n={n} (the most segments the "
                             f"validation can support is {max_valid_k}).")
        if cfg.k_max > max_valid_k:
            why = ("the resampling-based validation cannot reliably support more segments"
                   if max_valid_k == n // 2 else
                   f"there are only {n_distinct} distinct answer patterns in the data, so more "
                   "groups than that cannot exist")
            print(f"NOTE: clamping k_max from {cfg.k_max} to {max_valid_k} — with n={n} "
                  f"respondents {why}.")
            cfg = replace(cfg, k_max=max_valid_k)

        # Memory guard: the consensus matrix is n x n. Skip it (with a note) for very large n.
        if cfg.run_consensus and n > 5000:
            print(f"NOTE: n={n} is large; skipping consensus clustering (its n x n matrix would use "
                  f"~{2 * n * n * 8 / 1e6:.0f} MB). Re-enable deliberately if you have the memory.")
            cfg = replace(cfg, run_consensus=False)
        self.cfg = cfg   # use the validated/clamped config for the rest of the run

        print(f"Segmenting {X.shape[0]} respondents on {X.shape[1]} items "
              f"(method: {cfg.method}, scaling: {cfg.scaling}). Running the diagnostic panel...\n")

        self.hopkins = hopkins_statistic(X, np.random.default_rng(cfg.random_state + 7))
        print(f"Cluster-tendency (Hopkins) = {self.hopkins:.2f} — {hopkins_reading(self.hopkins)}.\n")

        self.diagnostics = selection_diagnostics(X, cfg)
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
            self.consensus_agreement = float(adjusted_rand_score(self.labels, cons_labels))
            if cfg.use_consensus_final:
                self.labels = cons_labels
                print("Adopted the consensus ensemble partition as the final segmentation.\n")
        self.split_half = split_half_replication(X, self.recommended_k, cfg)
        sil_overall = silhouette_score(X, self.labels)
        self.jaccard = clusterboot_jaccard(X, self.labels, self.recommended_k, cfg)
        self.mb_agreement = model_based_agreement(X, self.recommended_k, self.labels, cfg,
                                                  np.random.default_rng(cfg.random_state + 8))
        self.ward_ari = ward_agreement(X, self.labels, self.recommended_k)
        self.var_importance = variable_importance(X_raw, self.labels)
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
        self.assignments = pd.DataFrame({"id": ids, "segment": self.labels})
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
                                            getattr(self, 'distinct_share', None))
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
        (outdir / "segmentation_report.md").write_text(self.report_markdown)
        write_html_report(self.report_markdown, outdir / "segmentation_report.html",
                          "Segmentation report")
        # Typing rule: the portable classifier for NEW respondents.
        # Apply with: --classify new.csv --rule typing_rule.json
        (outdir / "typing_rule.json").write_text(json.dumps(self.typing_rule_dict(), indent=2))
        fig = maybe_plot(self.diagnostics, X, self.labels, self.recommended_k, outdir)
        # Reproducibility manifest: everything needed to reproduce this exact run.
        import sklearn
        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_version": __version__,
            "config": asdict(self.cfg),
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
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nSaved to {outdir}/: assignments, centroids, diagnostics, per-segment Jaccard "
              f"stability, variable importance, report, typing_rule.json, run_manifest.json"
              + (", diagnostics.png" if fig else " (no figure — matplotlib missing)"))


def _cli():
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
    p.add_argument("--method", choices=["auto", "kmeans", "gmm", "lca"], default="auto",
                   help="auto (default: inspect the file and choose for you), kmeans (heuristic), "
                        "gmm (Gaussian mixture, continuous), or lca (Latent Class Analysis, for "
                        "CATEGORICAL / multiple-choice items)")
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
    a = p.parse_args()

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
        rule = json.loads(Path(a.rule).read_text())
        classifier = classify_new_lca if rule.get("method") == "lca" else classify_new
        out = classifier(rule, _read_table(a.classify), id_col=a.id_col)
        if a.outdir:
            Path(a.outdir).mkdir(parents=True, exist_ok=True)
            out.to_csv(Path(a.outdir) / "new_assignments.csv", index=False)
            print(f"Typed {len(out)} new respondent(s) -> {a.outdir}/new_assignments.csv")
        else:
            print(out.to_string(index=False))
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
    Segmenter(cfg).run(a.csv, id_col=a.id_col, item_cols=a.items, force_k=a.force_k,
                       outdir=a.outdir, demographics=a.demographics)


if __name__ == "__main__":
    _cli()
