"""
Test suite for segment_kmeans.py — encodes the professional guarantees as executable tests.

Run:  pytest              (from tools/survey-segmenter/)

The two tests that matter most:
  - test_recovers_planted_structure : on data with real segments, it finds them.
  - test_rejects_structureless_noise : on random noise, it does NOT manufacture stable segments.
A segmentation tool that fails either of those is worse than useless, because it will mislead.
"""
import io
import os
import subprocess
import time
import json
import re
import contextlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The module under test lives one level up (this file is in tests/), so put that on the path
# rather than relying on the working directory pytest happened to be invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import segment_kmeans as sk

ROOT = Path(__file__).resolve().parent.parent
import charts
import clusterability
import kprototypes as kp
import maxdiff as _maxdiff
import webapp
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

# Silence the known-spurious numpy-2.0/macOS-Accelerate "... encountered in matmul" warnings
# (harmless, suppressed in normal runs but re-enabled by pytest) and third-party deprecations.
pytestmark = [
    pytest.mark.filterwarnings("ignore:.*encountered in matmul"),
    pytest.mark.filterwarnings("ignore:.*deprecated - use"),
]

# Small, fast config so the suite runs in a few seconds while still exercising every path.
# Consensus clustering is off here for speed and exercised by its own dedicated test below.
FAST = dict(gap_B=8, stability_B=12, ps_splits=6, jaccard_B=25, n_init_final=10, n_init_search=8,
            run_consensus=False, check_variable_selection=False)


# ----------------------------------------------------------------------------- fixtures
def structured(n_per=(120, 100, 100), n_items=10, sep=3.0, seed=0):
    """Utilities with `len(n_per)` planted mind-sets + two pure-noise items (8 signal, 2 noise)."""
    rng = np.random.default_rng(seed)
    items = [f"item_{i}" for i in range(n_items)]
    centers = rng.normal(0, sep, size=(len(n_per), n_items))
    centers[:, -2:] = 0.0                                   # last two items carry no segment signal
    rows, truth = [], []
    for seg, cnt in enumerate(n_per):
        for _ in range(cnt):
            rows.append(centers[seg] + rng.normal(0, 1, n_items)); truth.append(seg)
    df = pd.DataFrame(rows, columns=items); df.insert(0, "id", [f"r{i}" for i in range(len(df))])
    return df, np.array(truth), items


def null_data(n=320, n_items=10, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(0, 1, (n, n_items)), columns=[f"item_{i}" for i in range(n_items)])
    df.insert(0, "id", [f"r{i}" for i in range(n)])
    return df


def _write(tmp_path, df, name="u.csv"):
    p = tmp_path / name; df.to_csv(p, index=False); return str(p)


# ----------------------------------------------------------------------------- headline tests
def test_recovers_planted_structure(tmp_path):
    df, truth, _ = structured()
    cfg = sk.SegmentationConfig(k_min=2, k_max=6, **FAST)
    seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id")
    assert seg.recommended_k == 3, "should recover the 3 planted segments"
    assert adjusted_rand_score(truth, seg.labels) > 0.95, "should recover the planted membership"
    assert min(seg.jaccard.values()) >= 0.75, "every real segment should be Jaccard-stable"
    assert seg.hopkins > 0.6, "clustered data should show cluster tendency"
    assert seg.split_half >= 0.8, "a real solution should replicate across halves"


def test_rejects_structureless_noise(tmp_path):
    cfg = sk.SegmentationConfig(k_min=2, k_max=6, **FAST)
    seg = sk.Segmenter(cfg).run(_write(tmp_path, null_data()), id_col="id")
    # On noise: no cluster tendency, and the forced segments must NOT all look stable.
    assert seg.hopkins < 0.6, "uniform noise should not show a strong cluster tendency"
    assert min(seg.jaccard.values()) < 0.75, "noise 'segments' must not all pass the stability bar"
    # prediction strength should never strongly cross the 0.8 bar for k>=2 on noise
    ps = seg.diagnostics.set_index("k")["prediction_strength"]
    assert (ps[ps.index >= 3] < 0.8).all(), "noise should fail prediction strength for k>=3"


# ----------------------------------------------------------------------------- component tests
def test_gmm_method_recovers_structure(tmp_path):
    """The first-class model-based (Gaussian-mixture / latent-class) path recovers structure too,
    and the whole pipeline (stability, prediction strength, Jaccard) runs under it."""
    df, truth, _ = structured(sep=4.5, seed=0)   # well-separated so the true k is unambiguous
    cfg = sk.SegmentationConfig(k_min=2, k_max=5, method="gmm", **FAST)
    seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id")
    assert seg.recommended_k == 3
    assert adjusted_rand_score(truth, seg.labels) > 0.95
    assert "gmm_ICL" in seg.diagnostics.columns, "ICL should be computed"
    # cross-paradigm check should compare against k-means when the method is gmm
    assert seg.mb_agreement["other_method"] == "k-means"


def test_consensus_pac_and_partition():
    """Monti consensus clustering: PAC is near zero on real structure and high on noise, and the
    consensus ensemble partition recovers the planted structure."""
    df, truth, items = structured(sep=4.0, seed=0)
    X = df[items].to_numpy(float)
    Xn = null_data()[[f"item_{i}" for i in range(10)]].to_numpy(float)
    cfg = sk.SegmentationConfig(k_min=2, k_max=5, consensus_H=15, n_init_search=8)
    pac_s = sk.consensus_pac(X, range(2, 6), cfg, np.random.default_rng(1)).set_index("k")["consensus_PAC"]
    pac_n = sk.consensus_pac(Xn, range(2, 6), cfg, np.random.default_rng(1)).set_index("k")["consensus_PAC"]
    assert pac_s.loc[3] < 0.10, "real 3-cluster structure should be nearly unambiguous at k=3"
    assert pac_n.min() > 0.30, "noise should be ambiguous at every k"
    lab, C = sk.consensus_partition(X, 3, cfg, np.random.default_rng(2))
    assert adjusted_rand_score(truth, lab) > 0.90, "consensus ensemble partition recovers structure"
    assert C.shape == (len(X), len(X)) and 0.0 <= C.min() and C.max() <= 1.0


def test_consensus_end_to_end(tmp_path):
    """The full pipeline runs with consensus on, adds the PAC column, and reports agreement."""
    df, _, _ = structured(sep=4.0, seed=0)
    cfg = sk.SegmentationConfig(k_min=2, k_max=4, gap_B=6, stability_B=8, ps_splits=4,
                                jaccard_B=15, n_init_final=8, n_init_search=6,
                                consensus_H=12)
    seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id")
    assert "consensus_PAC" in seg.diagnostics.columns
    assert seg.consensus_agreement is not None and 0.0 <= seg.consensus_agreement <= 1.0


def test_variable_importance_flags_noise():
    df, truth, items = structured()
    X = df[items].to_numpy(float)
    lab = KMeans(3, n_init=10, random_state=0).fit(X).labels_
    vi = sk.variable_importance(df[items], lab).set_index("item")
    assert vi.loc["item_8", "eta_squared"] < 0.05 and vi.loc["item_9", "eta_squared"] < 0.05
    assert "near-noise" in vi.loc["item_9", "role"]
    assert vi["eta_squared"].iloc[0] > 0.3, "the top signal item should have high eta-squared"


def test_prediction_strength_drops_past_true_k():
    df, _, items = structured()
    Xr = df[items].to_numpy(float)
    cfg = sk.SegmentationConfig(**FAST)
    ps = sk.prediction_strength(Xr, range(2, 6), 6, np.random.default_rng(0), cfg).set_index("k")
    assert ps.loc[3, "prediction_strength"] > 0.85, "prediction strength high at the true k"
    assert ps.loc[5, "prediction_strength"] < ps.loc[3, "prediction_strength"], "and lower past it"


def test_gmm_bic_icl_find_true_k():
    df, _, items = structured()
    tbl = sk.gmm_bic_icl(df[items].to_numpy(float), range(2, 6), np.random.default_rng(0))
    assert int(tbl.loc[tbl["gmm_BIC"].idxmin(), "k"]) == 3
    assert int(tbl.loc[tbl["gmm_ICL"].idxmin(), "k"]) == 3, "ICL should also find the true k"


def test_clusterboot_jaccard_high_for_real_low_for_noise():
    df, _, items = structured()
    Xr = df[items].to_numpy(float)
    lab = KMeans(3, n_init=10, random_state=0).fit(Xr).labels_
    cfg = sk.SegmentationConfig(**FAST)
    jr = sk.clusterboot_jaccard(Xr, lab, 3, cfg)
    assert min(jr.values()) >= 0.75

    Xn = null_data()[[f"item_{i}" for i in range(10)]].to_numpy(float)
    labn = KMeans(3, n_init=10, random_state=0).fit(Xn).labels_
    jn = sk.clusterboot_jaccard(Xn, labn, 3, cfg)
    assert min(jn.values()) < 0.75


def test_hopkins_bounds_and_ordering():
    df, _, items = structured()
    hs = sk.hopkins_statistic(df[items].to_numpy(float), np.random.default_rng(0))
    hn = sk.hopkins_statistic(null_data()[[f"item_{i}" for i in range(10)]].to_numpy(float),
                              np.random.default_rng(0))
    assert 0.0 <= hs <= 1.0 and 0.0 <= hn <= 1.0
    assert hs > hn, "structured data should be more clusterable than noise"


def test_fdr_bh_helper():
    sig = sk._fdr_bh({"a": 0.001, "b": 0.02, "c": 0.30, "d": 0.80})
    assert sig["a"] is True and sig["d"] is False


@pytest.mark.parametrize("scaling", ["range", "standardize", "robust", "none", "ipsative"])
def test_all_scalings_run(tmp_path, scaling):
    df, _, _ = structured(seed=2)
    cfg = sk.SegmentationConfig(k_min=2, k_max=4, scaling=scaling, **FAST)
    seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id")
    assert len(np.unique(seg.labels)) == seg.recommended_k


def test_missing_data_imputed(tmp_path):
    df, _, items = structured(seed=3)
    df.loc[0, items[0]] = np.nan; df.loc[5, items[1]] = np.nan
    cfg = sk.SegmentationConfig(k_min=2, k_max=4, **FAST)
    seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id")   # must not crash
    assert len(seg.labels) == len(df)


def test_constant_column_dropped(tmp_path):
    df, _, items = structured(seed=4)
    df[items[0]] = 42.0                                              # a constant item
    cfg = sk.SegmentationConfig(k_min=2, k_max=4, **FAST)
    seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id")   # must not crash on zero variance
    assert seg.recommended_k >= 2


# ----------------------------------------------------------------------------- robustness / stress
def test_clamps_k_when_n_small(tmp_path):
    """k_max is clamped to what the resampling validation can support, rather than crashing."""
    df, _, _ = structured(n_per=(4, 3, 3), seed=1)   # n=10
    cfg = sk.SegmentationConfig(k_min=2, k_max=8, **FAST)
    seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id")
    assert seg.recommended_k <= 10 // 2


def test_single_k_collapse_is_warning_free(tmp_path):
    """When n forces the search range down to a single k, normalisation must not divide by zero
    (the elbow x-axis range is 0). Record warnings and assert none is a divide-by-zero from our
    code (the known-spurious '... in matmul' warnings from numpy/Accelerate are excluded)."""
    import warnings
    df, _, _ = structured(n_per=(3, 3), n_items=6, sep=4, seed=2)   # n=6 -> k_max clamps to 3
    cfg = sk.SegmentationConfig(k_min=3, k_max=3, **FAST)           # range is the single value {3}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id")
    divide = [str(w.message) for w in caught
              if "divide" in str(w.message).lower() and "matmul" not in str(w.message).lower()]
    assert not divide, f"divide-by-zero warning(s) from our code: {divide}"
    assert seg.recommended_k == 3


def test_errors_on_too_few_rows(tmp_path):
    df = pd.DataFrame({"id": [0, 1, 2], "v0": [1.0, 2, 3], "v1": [3.0, 2, 1]})
    with pytest.raises(ValueError):
        sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=4, **FAST)).run(_write(tmp_path, df), id_col="id")


def test_errors_on_one_item(tmp_path):
    df = pd.DataFrame({"id": range(40), "v0": np.random.default_rng(0).normal(0, 1, 40)})
    with pytest.raises(ValueError):
        sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=4, **FAST)).run(_write(tmp_path, df), id_col="id")


def test_handles_inf_and_nan(tmp_path):
    """Infinities are treated as missing and imputed, not silently clustered; NaNs imputed."""
    df, _, items = structured(seed=2)
    df.loc[0, items[0]] = np.inf
    df.loc[1, items[1]] = -np.inf
    df.loc[2, items[2]] = np.nan
    seg = sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=4, **FAST)).run(_write(tmp_path, df), id_col="id")
    assert np.isfinite(seg.centroids.to_numpy(float)).all(), "no infinities should reach the output"


def test_handles_more_items_than_respondents(tmp_path):
    """n <= number of items: the gap statistic's PCA reference must not break."""
    df, _, _ = structured(n_per=(4, 4, 4), n_items=20, sep=5, seed=3)   # n=12, d=20
    seg = sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=5, **FAST)).run(_write(tmp_path, df), id_col="id")
    assert seg.recommended_k >= 2


def test_ignores_nonnumeric_columns(tmp_path):
    df, _, _ = structured(seed=4)
    df["city"] = np.random.default_rng(0).choice(list("ABC"), len(df))   # a text column
    seg = sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=4, **FAST)).run(_write(tmp_path, df), id_col="id")
    assert seg.centroids.shape[1] == sum(c.startswith("item_") for c in df.columns)


def test_deterministic(tmp_path):
    df, _, _ = structured(seed=5)
    p = _write(tmp_path, df)
    a = sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=5, **FAST)).run(p, id_col="id")
    b = sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=5, **FAST)).run(p, id_col="id")
    assert np.array_equal(a.labels, b.labels) and a.recommended_k == b.recommended_k


def test_force_k_and_outputs_saved(tmp_path):
    df, _, _ = structured(seed=5)
    cfg = sk.SegmentationConfig(k_min=2, k_max=5, **FAST)
    out = tmp_path / "out"
    seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id", force_k=4, outdir=str(out))
    assert seg.recommended_k == 4
    for f in ("segment_assignments.csv", "segment_centroids.csv", "k_selection_diagnostics.csv",
              "segment_stability_jaccard.csv", "variable_importance.csv",
              "segmentation_report.md", "run_manifest.json", "typing_rule.json"):
        assert (out / f).exists(), f"missing output: {f}"


# ----------------------------------------------------------------------------- typing tool
@pytest.mark.parametrize("scaling", ["range", "standardize", "robust", "ipsative", "none"])
def test_scale_fit_apply_roundtrip(scaling):
    """The fitted scaling, re-applied to the same rows, must reproduce the fit exactly — this is
    what lets a saved rule scale NEW respondents identically (and CV refit without leakage)."""
    arr = np.random.default_rng(0).normal(3, 2, (60, 5))
    Xf, params = sk._scale_fit(arr, scaling)
    assert np.allclose(Xf, sk._scale_apply(arr, params))


def test_typing_tool_reports_and_beats_baseline(tmp_path):
    """On real structure the typing tool cross-validates well above the majority-class baseline,
    and writes an exportable rule."""
    df, _, _ = structured(n_per=(120, 110, 110), sep=3.5, seed=11)
    out = tmp_path / "seg"
    seg = sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=5, **FAST)).run(
        _write(tmp_path, df), id_col="id", force_k=3, outdir=str(out))
    assert not np.isnan(seg.typing["cv_accuracy"])
    assert seg.typing["cv_accuracy"] > seg.typing["baseline_majority"] + 0.15
    assert seg.typing["scaled_centroids"].shape == (3, seg.centroids.shape[1])
    assert "Segment predictability (typing tool)" in seg.report_markdown


def test_classify_new_recovers_holdout_segments(tmp_path):
    """The operational payoff: train on 80%, export the rule, and correctly type the held-out 20%
    into their true mind-sets — with the respondent id preserved."""
    import json
    df, truth, _ = structured(n_per=(130, 120, 120), sep=3.5, seed=12)
    idx = np.random.default_rng(0).permutation(len(df)); cut = int(0.8 * len(df))
    train_df = df.iloc[idx[:cut]].reset_index(drop=True)
    test_df = df.iloc[idx[cut:]].reset_index(drop=True); truth_te = truth[idx[cut:]]
    out = tmp_path / "seg"
    sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=5, **FAST)).run(
        _write(tmp_path, train_df, "train.csv"), id_col="id", force_k=3, outdir=str(out))
    rule = json.loads((out / "typing_rule.json").read_text())
    assigned = sk.classify_new(rule, test_df, id_col="id")
    assert adjusted_rand_score(truth_te, assigned["segment"]) > 0.80   # permutation-invariant
    assert "id" in assigned.columns and len(assigned) == len(test_df)
    assert assigned["confidence"].between(0, 1).all()


def test_classify_new_missing_column_errors():
    rule = {"items": ["a", "b"], "classes": [0, 1], "scale_params": {"scaling": "none"},
            "scaled_centroids": [[0.0, 0.0], [1.0, 1.0]]}
    with pytest.raises(ValueError):
        sk.classify_new(rule, pd.DataFrame({"a": [1.0, 2.0]}))          # "b" is missing


# ------------------------------------------------------------------- variable-selection check
def test_variable_selection_flags_planted_noise(tmp_path):
    """The structured fixture plants two pure-noise items (item_8, item_9). The Dolnicar check must
    flag them, and the signal-only solution must not be materially less stable."""
    df, _, _ = structured(n_per=(120, 110, 110), n_items=10, sep=3.0, seed=7)
    cfg = sk.SegmentationConfig(k_min=2, k_max=5, **{**FAST, "check_variable_selection": True})
    seg = sk.Segmenter(cfg).run(_write(tmp_path, df), id_col="id", force_k=3)
    assert seg.varsel is not None and seg.varsel["applicable"]
    assert {"item_8", "item_9"}.issubset(set(seg.varsel["dropped"]))
    assert seg.varsel["reduced"]["min_jaccard"] >= seg.varsel["full"]["min_jaccard"] - 0.15
    assert "Variable-selection check" in seg.report_markdown


# --------------------------------------------------- latent class analysis (categorical data)
def categorical_structured(n_per=(200, 200, 200), n_items=9, seed=0, sep=0.85):
    """Binary agree/disagree items with planted latent classes: each class 'agrees' on its own
    distinct block of items and 'disagrees' on the rest."""
    rng = np.random.default_rng(seed); K = len(n_per); block = n_items // K
    proto = np.full((K, n_items), 1 - sep)
    for c in range(K):
        proto[c, c * block:(c + 1) * block] = sep
    ad = np.array(["agree", "disagree"]); rows, truth = [], []
    for c, cnt in enumerate(n_per):
        for r in (rng.random((cnt, n_items)) < proto[c]).astype(int):
            rows.append(ad[r]); truth.append(c)
    df = pd.DataFrame(rows, columns=[f"q{i}" for i in range(n_items)])
    df.insert(0, "id", [f"r{i}" for i in range(len(df))])
    return df, np.array(truth)


def test_lca_recovers_categorical_structure(tmp_path):
    """On categorical data with planted latent classes, BIC/ICL/stability find the true count and
    the classes match the truth."""
    df, truth = categorical_structured(seed=0)
    seg = sk.LatentClassSegmenter(sk.SegmentationConfig(k_min=2, k_max=5, **FAST)).run(
        _write(tmp_path, df), id_col="id")
    assert seg.recommended_k == 3
    assert adjusted_rand_score(truth, seg.labels) > 0.85
    assert min(seg.jaccard.values()) > 0.70


def test_lca_rejects_categorical_noise(tmp_path):
    """On structureless categorical noise, replication stability at the chosen k is low and the
    report says so — it does not manufacture confident classes."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"q{i}": rng.choice(list("abc"), 360) for i in range(6)})
    df.insert(0, "id", [f"r{i}" for i in range(360)])
    seg = sk.LatentClassSegmenter(sk.SegmentationConfig(k_min=2, k_max=5, **FAST)).run(
        _write(tmp_path, df), id_col="id")
    stab = seg.diagnostics.set_index("k").loc[seg.recommended_k, "stability_ARI"]
    assert stab < 0.75 and "WARNING" in seg.report_markdown


def test_lca_handles_polytomous_and_is_deterministic(tmp_path):
    """A 3-level item is modelled with three category probabilities (not collapsed), and the same
    seed gives the same result."""
    rng = np.random.default_rng(2); n = 450; truth = rng.integers(0, 3, n)
    ch = np.array([rng.choice(3, p=np.eye(3)[t] * 0.7 + 0.1) for t in truth])
    b1 = np.array([rng.choice(2, p=[0.85, 0.15] if t == 0 else [0.15, 0.85]) for t in truth])
    df = pd.DataFrame({"id": range(n), "channel": np.array(["app", "campus", "email"])[ch],
                       "q1": np.array(["y", "n"])[b1],
                       "q2": np.array(["y", "n"])[rng.integers(0, 2, n)]})
    p = _write(tmp_path, df)
    a = sk.LatentClassSegmenter(sk.SegmentationConfig(k_min=2, k_max=4, **FAST)).run(p, id_col="id")
    b = sk.LatentClassSegmenter(sk.SegmentationConfig(k_min=2, k_max=4, **FAST)).run(p, id_col="id")
    assert 3 in a.model["level_counts"]                       # the 3-level item kept its 3 levels
    assert np.array_equal(a.labels, b.labels) and a.recommended_k == b.recommended_k


def test_lca_outputs_and_loader(tmp_path):
    """End-to-end: all files written; a constant item is dropped by the loader; too-few-items errors."""
    df, _ = categorical_structured(seed=4)
    df["constant_item"] = "same"
    out = tmp_path / "lca"
    sk.LatentClassSegmenter(sk.SegmentationConfig(k_min=2, k_max=4, **FAST)).run(
        _write(tmp_path, df), id_col="id", force_k=3, outdir=str(out))
    for f in ("segment_assignments.csv", "latent_class_profiles.csv", "lca_selection_diagnostics.csv",
              "class_stability_jaccard.csv", "latent_class_report.md", "run_manifest.json"):
        assert (out / f).exists(), f"missing {f}"
    assert "constant_item" not in set(pd.read_csv(out / "latent_class_profiles.csv")["item"])
    with pytest.raises(ValueError):                            # needs >= 2 informative items
        one = pd.DataFrame({"id": range(20), "only": ["a", "b"] * 10})
        sk.LatentClassSegmenter(sk.SegmentationConfig(**FAST)).run(_write(tmp_path, one, "one.csv"), id_col="id")


# ------------------------------------- non-expert front door: auto mode, plain output, HTML report
def test_try_likert_recodes_and_rejects():
    s = pd.Series(["Strongly agree", "Agree", "Neutral", "Disagree", "Strongly disagree"])
    assert list(sk._try_likert(s)) == [5, 4, 3, 2, 1]
    assert sk._try_likert(pd.Series(["red", "blue", "green", "red"])) is None   # not a scale


def test_short_label_reads_like_the_question_it_came_from():
    """These names appear all over a report as placeholders the team is told to replace, so their
    one job is to be recognisable as the question behind them.

    They used to have every stopword stripped, which reads as a telegram once two are joined with
    a "+": "I like planning things rather than deciding last minute" became "planning things
    rather", and a real report named a segment "planning things rather + want meet people outside".
    """
    lbl = sk._short_label("I want to meet people in real life")
    assert "want" in lbl and "meet" in lbl
    assert not lbl.endswith("rea"), "cut a word in half"
    assert "  " not in lbl

    # Ordinary English is kept, not filleted.
    assert sk._short_label("Most social apps feel fake to me").startswith("Most social apps feel")

    # A shortened label must look shortened, and must not trail off on a word that was leading
    # somewhere — "...feel fake to" and "...planning things rather" read as abandoned sentences.
    long = sk._short_label("I like planning things rather than deciding last minute")
    assert long.endswith("…"), long
    assert long.rstrip("…").split()[-1].lower() not in sk._TRAILING_FILLER, long

    # Short questions are left exactly as they are, with no ellipsis and no leading "I".
    assert sk._short_label("I go out often") == "go out often"


def test_auto_prepare_picks_kmeans_for_ratings_and_finds_id():
    rng = np.random.default_rng(0); n = 120; t = rng.integers(0, 3, n)
    ag = np.array(["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"])
    df = pd.DataFrame({
        "Timestamp": ["2026-01-01 10:00"] * n,
        "Email address": [f"u{i}@x.se" for i in range(n)],
        "privacy": ag[np.clip(2 + t + rng.integers(-1, 2, n), 0, 4)],
        "politics": ag[np.clip(2 - t + rng.integers(-1, 2, n), 0, 4)],
        "rating": rng.integers(1, 6, n),
        "Anything else?": rng.choice(["", "good", "bad", "meh", "x"], n)})
    clean, method, id_col, items, plan = sk.auto_prepare(df)
    assert id_col == "Email address" and method == "kmeans"
    assert "Timestamp" in plan["skipped"] and "Anything else?" in plan["skipped"]
    assert {"privacy", "politics", "rating"}.issubset(set(items))
    assert pd.api.types.is_numeric_dtype(clean["privacy"])                     # Likert -> numbers


def test_auto_prepare_picks_lca_for_categorical():
    rng = np.random.default_rng(1); n = 120
    df = pd.DataFrame({"id": range(n), "channel": rng.choice(["app", "email", "campus"], n),
                       "goal": rng.choice(["friends", "dating", "events"], n)})
    _, method, id_col, items, _ = sk.auto_prepare(df)
    assert method == "lca" and set(items) == {"channel", "goal"}


def test_auto_prepare_no_usable_items_raises():
    df = pd.DataFrame({"id": range(10), "note": [f"free text number {i} here" for i in range(10)]})
    with pytest.raises(ValueError):
        sk.auto_prepare(df)


def test_executive_summary_confidence_light():
    base = dict(n_resp=100, names=["A", "B"], shares=[0.6, 0.4], wants=["x", "y"])
    assert "High" in sk.executive_summary(**base, min_jaccard=0.9, repro=0.9, k_agreement=1.0)
    assert "Moderate" in sk.executive_summary(**base, min_jaccard=0.65, repro=0.4, k_agreement=1.0)
    kc = sk.executive_summary(**base, min_jaccard=0.9, repro=0.9, k_agreement=0.2)   # k contested
    assert "Moderate" in kc and "disagree on how many" in kc
    assert "Low" in sk.executive_summary(**base, min_jaccard=0.3, repro=0.2, k_agreement=1.0)


def test_html_report_renders(tmp_path):
    md = ("# Title\n\n## In plain language\n\n**Bold** and `code`.\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n\n> a warning\n\n- one\n- two\n")
    out = tmp_path / "r.html"; sk.write_html_report(md, out, "T")
    html = out.read_text()
    for frag in ("<h1>Title</h1>", "<table>", "<th>a</th>", "<blockquote>",
                 "<strong>Bold</strong>", "<ul>", "<!doctype html>"):
        assert frag in html, f"missing {frag}"


def test_auto_end_to_end_messy_survey(tmp_path):
    """A messy real-world export (Likert text, email id, timestamp) runs to a report with a
    plain-language summary and an HTML file — the whole point of 'anyone can use it'."""
    rng = np.random.default_rng(3); n = 200; t = rng.integers(0, 3, n)
    ag = np.array(["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"])
    df = pd.DataFrame({
        "Timestamp": ["2026-01-01 10:00"] * n,
        "Email address": [f"s{i}@kth.se" for i in range(n)],
        "privacy": ag[np.clip(2 + t + rng.integers(-1, 2, n), 0, 4)],
        "politics": ag[np.clip(2 - t + rng.integers(-1, 2, n), 0, 4)],
        "meet": ag[np.clip(2 + t + rng.integers(-1, 2, n), 0, 4)],
        "events": rng.integers(0, 8, n)})
    clean, method, id_col, items, _ = sk.auto_prepare(df)
    out = tmp_path / "res"
    seg = sk.Segmenter(sk.SegmentationConfig(**FAST)).run(clean, id_col=id_col, item_cols=items,
                                                          outdir=str(out))
    assert "In plain language" in seg.report_markdown and "Confidence:" in seg.report_markdown
    assert (out / "segmentation_report.html").exists()


def test_analyze_csv_to_html_is_the_web_core():
    """The web UI's core: raw CSV bytes -> a self-contained HTML report with the plain-language box
    and the auto-detection notes (the whole no-terminal path)."""
    rng = np.random.default_rng(4); n = 180; t = rng.integers(0, 3, n)
    ag = np.array(["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"])
    df = pd.DataFrame({"Email address": [f"s{i}@x.se" for i in range(n)],
                       "privacy": ag[np.clip(2 + t + rng.integers(-1, 2, n), 0, 4)],
                       "politics": ag[np.clip(2 - t + rng.integers(-1, 2, n), 0, 4)],
                       "meet": ag[np.clip(2 + t + rng.integers(-1, 2, n), 0, 4)]})
    _, doc = sk.analyze_csv_to_html(df.to_csv(index=False).encode(),
                                    cfg=sk.SegmentationConfig(**FAST))
    for frag in ("<!doctype html>", "In plain language", "What I found in your file", "Confidence:"):
        assert frag in doc, f"missing {frag}"


# ------------------------------------------- real-world robustness: messy files, demographics, Ward
def test_read_table_handles_semicolon_latin1_bom_tab():
    """Real exports are not always comma + UTF-8: Swedish/Excel use ';' and Latin-1 (aao)."""
    df = sk._read_table(b"id;q1;q2\n1;4;5\n2;3;2\n")                         # semicolon
    assert list(df.columns) == ["id", "q1", "q2"] and len(df) == 2
    df = sk._read_table("id;fr\xe5ga\n1;Ja\n".encode("latin-1"))            # latin-1 + semicolon
    assert "fr\xe5ga" in df.columns
    df = sk._read_table("id,a,b\n1,2,3\n".encode("utf-8-sig"))              # UTF-8 BOM
    assert list(df.columns) == ["id", "a", "b"]
    df = sk._read_table(b"id\tx\ty\n1\t2\t3\n")                             # tab
    assert list(df.columns) == ["id", "x", "y"]


def test_read_table_reads_excel_if_openpyxl_present(tmp_path):
    pytest.importorskip("openpyxl")
    p = tmp_path / "s.xlsx"
    pd.DataFrame({"id": [1, 2], "q1": [4, 5], "q2": [3, 2]}).to_excel(p, index=False)
    df = sk._read_table(str(p))
    assert list(df.columns) == ["id", "q1", "q2"] and len(df) == 2


def test_skip_matching_is_whole_word():
    assert not sk._name_matches("gender", sk._SKIP_NAME_HINTS)               # 'end' must NOT match
    assert not sk._name_matches("calendar", sk._SKIP_NAME_HINTS)
    assert sk._name_matches("timestamp", sk._SKIP_NAME_HINTS)
    assert sk._name_matches("submission date", sk._SKIP_NAME_HINTS)


def test_auto_sets_aside_demographics_not_clusters_on_them():
    rng = np.random.default_rng(0); n = 200; t = rng.integers(0, 3, n)
    ag = np.array(["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"])
    df = pd.DataFrame({"Email": [f"u{i}@x.se" for i in range(n)],
                       "Gender": rng.choice(["Male", "Female"], n),
                       "University": rng.choice(["KTH", "Lund"], n),
                       "Age": rng.integers(18, 30, n),
                       "privacy": ag[np.clip(2 + t + rng.integers(-1, 2, n), 0, 4)],
                       "politics": ag[np.clip(2 - t + rng.integers(-1, 2, n), 0, 4)]})
    clean, _, _, items, plan = sk.auto_prepare(df)
    assert set(items) == {"privacy", "politics"}
    assert set(plan["demographics"]) == {"Gender", "University", "Age"}
    assert not ({"Gender", "University", "Age"} & set(clean.columns))        # never clustered on


def test_interpret_values_most_and_least_are_distinct_with_few_items():
    """With few items, 'values most' and 'values least' must not list the same items."""
    rng = np.random.default_rng(1)
    Xr = pd.DataFrame(rng.normal(0, 1, (90, 4)), columns=list("abcd"))
    labels = np.array([0, 1, 2] * 30)
    _, defining, _, _ = sk.interpret(Xr, labels, sk.SegmentationConfig())
    for d in defining.values():
        above = {x.split(" (")[0] for x in d["most_above_average"]}
        below = {x.split(" (")[0] for x in d["most_below_average"]}
        assert not (above & below)


def test_ward_agreement_recovers_and_guards_large_n():
    rng = np.random.default_rng(0); C = rng.normal(0, 6, (3, 8))
    a = rng.integers(0, 3, 300); X = sk._scale_fit((C[a] + rng.normal(0, 0.7, (300, 8))), "range")[0]
    lab = KMeans(3, n_init=10, random_state=0).fit(X).labels_
    assert sk.ward_agreement(X, lab, 3) > 0.90
    assert np.isnan(sk.ward_agreement(np.zeros((3001, 3)), np.zeros(3001, int), 2))   # O(n^2) guard


def test_weighted_population_projection(tmp_path):
    """Cluster unweighted, but project segment sizes to the population using a weight column: an
    over-sampled group must shrink to its true population share."""
    rng = np.random.default_rng(0)
    C = np.array([[5., 5, 5, 5], [-5., -5, -5, -5]]); lab = np.r_[np.zeros(300, int), np.ones(100, int)]
    X = C[lab] + rng.normal(0, 0.6, (400, 4))
    df = pd.DataFrame(X.round(2), columns=[f"q{i}" for i in range(4)])
    df.insert(0, "id", range(400)); df["design_weight"] = np.where(lab == 0, 0.5, 2.0)
    clean, _, id_col, items, plan = sk.auto_prepare(df)
    assert plan["weight"] == "design_weight" and "design_weight" not in items      # not clustered on
    seg = sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=3, **FAST)).run(
        clean, id_col=id_col, item_cols=items, force_k=2, weights=df["design_weight"].to_numpy())
    assert "population_share" in seg.sizes.columns
    big = seg.sizes["share"].idxmax()
    assert seg.sizes.loc[big, "population_share"] < seg.sizes.loc[big, "share"] - 0.1   # shrinks
    assert abs(seg.sizes["population_share"].sum() - 1.0) < 0.01


# ----------------------------------------------------- professional hardening: bad files, packaging
def test_read_table_rejects_empty_and_binary_cleanly():
    for bad in (b"", b"\x00\x01\xff not a csv"):           # unreadable -> clean _BAD_FILE, not a crash
        with pytest.raises(ValueError):
            sk._read_table(bad)
    assert "could not read that as a survey" in sk._explain_run_error("_BAD_FILE").lower()


def test_hostile_uploads_never_crash_the_web_core():
    """The non-expert path must return a clean error, never a traceback (professional requirement)."""
    fast = sk.SegmentationConfig(gap_B=5, stability_B=6, ps_splits=4, jaccard_B=10, n_init_final=5,
                                 n_init_search=4, run_consensus=False, check_variable_selection=False)
    for data in (b"", b"\x00\x01\xff", b"id\n1\n2\n", b"only,two\nx,y\n"):
        with pytest.raises(ValueError):                 # a clean, explainable error (not a crash)
            sk.analyze_csv_to_html(data, cfg=fast)


def test_multipart_parser_extracts_uploaded_file():
    b = "BND123"; csv = b"id,q1\n1,4\n"
    body = (f"--{b}\r\n".encode() + b'Content-Disposition: form-data; name="file"; filename="s.csv"'
            b"\r\n\r\n" + csv + f"\r\n--{b}--\r\n".encode())
    assert webapp._parse_multipart_file(f"multipart/form-data; boundary={b}", body) == csv
    assert webapp._parse_multipart_file("text/plain", body) is None


def test_demographics_do_not_swallow_long_attitude_questions():
    df = pd.DataFrame({"id": range(60), "Gender": ["M", "F"] * 30,
                       "Campus politics puts me off an app": ["Agree", "Disagree"] * 30,
                       "I value privacy online": ["Disagree", "Agree"] * 30})
    _, _, _, items, plan = sk.auto_prepare(df)
    assert "Campus politics puts me off an app" in items         # attitude, not a demographic
    assert plan["demographics"] == ["Gender"]


def test_version_stamped_in_manifest_and_report(tmp_path):
    import json
    df, _, _ = structured(seed=6)
    out = tmp_path / "o"
    seg = sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=4, **FAST)).run(
        _write(tmp_path, df), id_col="id", force_k=3, outdir=str(out))
    assert json.loads((out / "run_manifest.json").read_text())["tool_version"] == sk.__version__
    assert sk.__version__ in seg.report_markdown


def test_the_built_interface_is_served_and_cannot_be_escaped():
    """The interface is a compiled React bundle in webui/, not a string in this file.

    Python's job is now only to hand out those files, so that is what is checked: the page comes
    back, hashed assets come back with the right type, and nothing outside the bundle is
    reachable. The server listens on localhost only, which is a reason to be careful about path
    traversal rather than a reason not to bother.
    """
    assert webapp._webui_dir(), "webui/ is missing — run: cd frontend && npm run build"

    index = webapp._webui_asset("/")
    assert index and b"<!doctype html>" in index[0].lower()
    assert index[1].startswith("text/html")
    assert webapp._webui_asset("/index.html")[0] == index[0]

    # The real bundle, whatever this build's content hash happens to be.
    import os
    asset = next(f for f in os.listdir(os.path.join(webapp._webui_dir(), "assets"))
                 if f.endswith(".js"))
    served = webapp._webui_asset(f"/assets/{asset}")
    assert served and served[1].startswith("text/javascript")

    for escape in ("/../segment_kmeans.py", "/assets/../../ai_interpret.py",
                   "/%2e%2e/%2e%2e/maxdiff.py", "/../../../../etc/hosts"):
        assert webapp._webui_asset(escape) is None, escape
    # Not an asset type we serve, even if the file exists next to the bundle.
    assert webapp._webui_asset("/build_app.py") is None

    assert "The app has closed" in webapp._shutdown_page()
    assert callable(sk.app) and callable(sk.serve)


def _likert(x):
    """Push a latent score onto a 1-5 answer scale, which is what a survey actually contains."""
    return np.clip(np.round(x), 1, 5).astype(int)


def test_shadow_values_replace_the_silhouette_and_leave_nobody_blank():
    """Leisch's shadow value, s(x) = 2·d(closest) / [d(closest) + d(second closest)].

    The per-respondent fit column used a silhouette, which needs every pairwise distance — O(n^2)
    — so above 6,000 respondents it fell back to a sample and left everyone else blank, putting
    holes in the CRM export exactly on the studies large enough to matter. A shadow value needs
    only the two nearest centroids, O(n·k).
    """
    rng = np.random.default_rng(0)
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    who = rng.integers(0, 3, 400)
    X = np.clip(np.round(centres[who] + rng.normal(0, 0.5, (400, 5))), 1, 5)

    # The formula itself, against a hand-checkable case: a point sitting on a centroid scores 0,
    # a point exactly between two scores 1.
    cents = np.array([[0.0, 0.0], [10.0, 0.0]])
    shadow, closest, second = sk.shadow_values(np.array([[0.0, 0.0], [5.0, 0.0]]), cents)
    assert abs(shadow[0] - 0.0) < 1e-9 and abs(shadow[1] - 1.0) < 1e-9
    assert closest[0] == 0 and second[0] == 1

    df = pd.DataFrame(X.astype(int), columns=[f"q{i+1}" for i in range(5)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
    assigned = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
    assert assigned["fit"].isna().sum() == 0, "some respondents were left without a fit"
    assert assigned["fit"].between(0, 1).all()

    # The gorge plot is drawn from those same values, and its shape separates real structure
    # from noise: measured here, a typical respondent scores 0.33 on three genuine segments and
    # 0.86 when the same machinery is pointed at random answers.
    assert any(c["id"] == "gorge" for c in r["charts"])
    assert float(np.median(1.0 - assigned["fit"])) < 0.55

    noise = np.clip(np.round(rng.normal(3, 1.1, (400, 5))), 1, 5).astype(int)
    df = pd.DataFrame(noise, columns=[f"q{i+1}" for i in range(5)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    rn = sk.run_analysis(df.to_csv(index=False).encode(),
                         cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
    noisy = pd.read_csv(io.StringIO(rn["files"]["segment_assignments.csv"]))
    assert float(np.median(1.0 - noisy["fit"])) > 0.7, (
        "respondents in structureless data should be stranded between segments")


def test_the_report_names_which_segments_are_nearly_the_same():
    """The question that decides how many campaigns get funded.

    The report said how each segment differs from the average respondent. It never said which two
    segments sit next to each other — so somebody could sign off five campaigns without being
    told that two of them target much the same people. Leisch's s_ij, the average shadow value
    over the respondents caught between a given pair, is exactly that number.

    Bands calibrated on this machine by planting segments at known separations: two far apart
    0.27, three far apart 0.40, two just touching 0.65, three with two adjacent 0.87, three with
    two nearly identical 0.97, pure noise 0.99.
    """
    def analyse(centres, spread=0.5):
        rng = np.random.default_rng(0)
        who = rng.integers(0, len(centres), 400)
        X = np.clip(np.round(centres[who] + rng.normal(0, spread, (400, centres.shape[1]))), 1, 5)
        df = pd.DataFrame(X.astype(int), columns=[f"q{i+1}" for i in range(centres.shape[1])])
        df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
        return sk.run_analysis(df.to_csv(index=False).encode(),
                               cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))

    apart = analyse(np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float))
    assert "Which segments sit next to each other" in apart["digest"]
    assert "each has its own territory" in apart["digest"]

    # Two segments planted almost on top of one another must be called out, not glossed over.
    crowded = analyse(np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [1.2, 4.8, 1.2, 4.8, 3]], float))
    assert "each has its own territory" not in crowded["digest"], (
        "two nearly identical segments were reported as having their own territory")


def test_segments_are_checked_against_a_different_number_of_segments():
    """Dolnicar & Leisch's segment level stability across solutions (2017), `slsaplot` in flexclust.

    The tool already asks whether the whole solution reproduces on a fresh half, and whether each
    segment survives resampling at a fixed k. Neither answers the question a client actually puts:
    *you said four groups — what if it were five?* A segment that only exists at the chosen k is an
    artefact of the number and should not be handed a budget.

    Scored by containment, not Jaccard. Jaccard was tried first and pinned here because the failure
    is not obvious: going from k to k-1 must merge two segments, so Jaccard collapses whether or
    not the structure is real, and measured it could not separate three genuine segments (0.44-0.78)
    from pure noise (0.55-0.69) at all. Containment survives merging and falls only on genuine
    fragmentation.
    """
    # The mechanics, on a case whose answer is known by construction. Six points, two obvious
    # pairs-of-pairs; the chosen split is the true one.
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                  [9.0, 9.0], [9.1, 9.0], [9.0, 9.1]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    got = sk.stability_across_solutions(X, range(2, 4), 2, labels, 5, np.random.default_rng(0))
    assert set(got) == {0, 1}
    # k-1 is 1, which is not a segmentation, so only k=3 is compared against. One of the two
    # triples must split to make a third group; the other survives whole. Both are reported under
    # "splits", because that is the direction being measured here.
    assert all("splits" in v for v in got.values())
    splits = [v["splits"] for v in got.values()]
    assert max(splits) == 1.0 and min(splits) >= 2 / 3

    # A segment index with no members must not produce a score or a division by zero.
    lopsided = sk.stability_across_solutions(X, range(2, 4), 3, labels, 5, np.random.default_rng(0))
    assert 2 not in lopsided

    # No neighbouring solution to compare against means no claim, not a fabricated one.
    assert sk.stability_across_solutions(X, [2], 2, labels, 5, np.random.default_rng(0)) == {}

    assert sk.persistence_paragraph({}, []) == ""
    holds = sk.persistence_paragraph({0: {"merges": 0.95}, 1: {"merges": 0.88}},
                                     ["Loyal", "Curious"])
    assert "holds together" in holds and "Loyal" in holds
    assert "scatters" in sk.persistence_paragraph({0: {"merges": 0.95}, 1: {"merges": 0.2}},
                                                  ["A", "B"])

    # THE FALSE ALARM THIS METRIC USED TO RAISE. Asking for one more group forces the solution to
    # split something, so whichever segment is subdivided scores about 0.5 in that direction
    # whether or not it is genuine. Taking the weaker of the two directions then condemned it.
    # Measured on 420 students whose three mind-sets were recovered at ARI 0.954 — every segment
    # real — the largest held together perfectly when merged (1.00), scored 0.56 when split, and
    # the report told the reader not to build a campaign on it.
    both = sk.persistence_paragraph({0: {"merges": 1.0, "splits": 0.56},
                                     1: {"merges": 1.0, "splits": 1.0}}, ["Big", "Small"])
    assert "holds together" in both, both
    assert "scatters" not in both, "a segment that survives merging must not be called unreal"
    assert "Every segment stays together" in both
    assert "finer detail is available" in both, "splitting should be offered as an opportunity"

    # End to end, and the part that matters: it discriminates. Measured on this machine, three
    # planted segments score 0.78 and above while pure noise reaches only 0.69.
    def analyse(X):
        df = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(X.shape[1])])
        df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
        return sk.run_analysis(df.to_csv(index=False).encode(),
                               cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))

    rng = np.random.default_rng(0)
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    who = rng.integers(0, 3, 400)
    real = analyse(_likert(centres[who] + rng.normal(0, 0.5, (400, 5))))
    assert "ask for a different number" in real["digest"]
    held = [v["merges"] for v in real["persistence"].values() if "merges" in v]
    assert held and min(held) >= 0.7, real["persistence"]
    assert "Every segment stays together" in real["digest"]

    # Structureless data lands on k=2, where there IS no merge direction: k-1 is a single group,
    # which is not a segmentation. The metric therefore has nothing to say about whether those two
    # segments are real, and must say nothing rather than borrowing the split direction — which is
    # what it used to do, and how a genuine segment came to be reported as one that dissolves.
    # Realness at k=2 is decided by the Hopkins statistic, the dip test, split-half replication and
    # per-segment Jaccard, all of which appear above this table in the report.
    noise = analyse(_likert(rng.normal(3, 1.1, (400, 5))))
    noise_held = [v["merges"] for v in noise["persistence"].values() if "merges" in v]
    if noise_held:
        assert min(noise_held) < 0.7, "noise segments should not survive being merged"
    else:
        assert "Every segment stays together" not in noise["digest"], (
            "claimed the segments hold together with no evidence that they do")


def test_the_report_says_which_of_the_three_kinds_of_segmentation_this_is():
    """Dolnicar, Grün & Leisch's distinction, used throughout *Market Segmentation Analysis*.

    A single confidence word cannot separate "there is nothing here" from "there is nothing
    NATURAL here, but the split is stable enough to work with" — two very different situations
    that both feel like a weak result. The field's three words do.
    """
    def kind_of(centres, spread=0.5):
        rng = np.random.default_rng(0)
        who = rng.integers(0, len(centres), 400)
        X = np.clip(np.round(centres[who] + rng.normal(0, spread, (400, centres.shape[1]))), 1, 5)
        df = pd.DataFrame(X.astype(int), columns=[f"q{i+1}" for i in range(centres.shape[1])])
        df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
        r = sk.run_analysis(df.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
        found = re.search(r"\*\*This is a (\w+) segmentation\*\*", r["digest"])
        assert found, "the report did not classify the segmentation at all"
        return found.group(1)

    assert kind_of(np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)) == "natural"
    assert kind_of(np.array([[3, 3, 3, 3, 3]], float), spread=1.2) == "constructive"


def test_a_stable_but_wrong_partition_does_not_get_high_confidence():
    """A wrong answer can be perfectly reproducible, and reproducibility was the whole verdict.

    MacKay, *Information Theory, Inference, and Learning Algorithms* §20.1: k-means "has no way of
    representing the size or shape of a cluster". Where one group is broad and another narrow, it
    assigns members of the broad group to the narrow one — and does it the same way every time,
    so every stability measure looks perfect.

    Measured on his exact case (240 broad, 60 narrow): split-half replication 1.000, agreement
    with a Gaussian mixture 0.43 and with Ward 0.47 — about half the memberships in dispute — and
    the report said **High**. The tool already computed both cross-method numbers and printed
    them in a table; they simply did not reach the verdict.

    Note for whoever reads the earlier benchmark: the "never confidently wrong" property was
    verified on recovering the NUMBER of groups, not on who ended up in them. This is the case
    that distinguishes those, and it was failing.
    """
    rng = np.random.default_rng(0)
    broad = rng.normal([2.2, 2.2, 3, 3, 3], 1.05, (240, 5))
    narrow = rng.normal([4.4, 4.4, 3, 3, 3], 0.28, (60, 5))
    truth = np.r_[np.zeros(240), np.ones(60)]
    X = np.clip(np.round(np.vstack([broad, narrow])), 1, 5).astype(int)
    df = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(5)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))

    from sklearn.metrics import adjusted_rand_score
    assigned = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
    recovered = adjusted_rand_score(truth, assigned["segment"])
    assert recovered < 0.8, (
        f"the planted failure did not occur (ARI {recovered:.2f}); this test is not testing "
        "anything and the data needs to be made harder")
    assert r["confidence"] != "high", (
        f"memberships recovered at only ARI {recovered:.2f} and the report still said High")

    # And it must not cry wolf: genuine, well-separated segments keep their high confidence.
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    who = rng.integers(0, 3, 400)
    real = np.clip(np.round(centres[who] + rng.normal(0, 0.5, (400, 5))), 1, 5).astype(int)
    df = pd.DataFrame(real, columns=[f"q{i+1}" for i in range(5)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
    assert r["confidence"] == "high" and r["k"] == 3


def test_every_respondent_carries_how_well_they_fit():
    """k-means has no way to answer "none of these" — everybody gets a segment.

    James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* §12.4.3:
    "K-means and hierarchical clustering force every observation into a cluster… the clusters
    found may be heavily distorted due to the presence of outliers that do not belong to any
    cluster." The assignments file is what goes into a CRM, and with only id and segment the
    respondent who matched nothing looks exactly like everyone else.

    Each row now carries its silhouette, so a poor fit can be filtered before money is spent.
    """
    rng = np.random.default_rng(0)
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3]], float)
    X = np.clip(np.round(centres[rng.integers(0, 2, 200)] + rng.normal(0, .5, (200, 5))), 1, 5)
    # Four people who belong to neither mind-set.
    X = np.vstack([X, np.array([[1, 1, 5, 5, 1], [5, 5, 1, 1, 5],
                                [3, 1, 1, 5, 5], [1, 5, 5, 1, 1]], float)])
    df = pd.DataFrame(X.astype(int), columns=[f"q{i+1}" for i in range(5)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=5, **FAST))

    assigned = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
    assert list(assigned.columns) == ["id", "segment", "fit"]

    ordinary = assigned.head(200)["fit"]
    misfits = assigned.tail(4)["fit"]
    assert ordinary.median() > 0.5, f"ordinary respondents scored {ordinary.median():.2f}"
    assert misfits.max() < ordinary.median(), (
        "the people who belong to no segment do not look any different from those who do")

    # And the report tells the reader the column is there and what it is for.
    assert "fit" in r["digest"] and "none of these" in r["digest"]


def test_the_report_says_outright_when_there_is_only_one_group():
    """The gap statistic is the only criterion here that can answer "is this one group?".

    Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* §14.3.11: the gap
    "works reasonably well when the data fall into a single cluster, and in that case will tend
    to estimate the optimal number of clusters to be one. This is the scenario where most other
    competing methods fail." Silhouette, Calinski-Harabasz and Davies-Bouldin are undefined at
    k=1, the elbow has no kink, and prediction strength needs two partitions to compare.

    The search starts at k=2 because profiles, charts and the typing rule all need at least two
    groups — a reasonable choice that quietly threw away the one test for "there is nothing here".
    Measured: pure noise gives gap(1)=0.86 against gap(2)=0.85 and selects one group; three real
    segments give gap(1)=-0.36 and select three.
    """
    rng = np.random.default_rng(0)
    noise = np.clip(np.round(rng.normal(3, 1.1, (400, 5))), 1, 5).astype(int)
    df = pd.DataFrame(noise, columns=[f"q{i+1}" for i in range(5)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
    assert "one group, not several" in r["digest"], (
        "structureless data was divided up without saying the division is imposed")
    assert r["confidence"] == "low"

    # And it must not cry wolf on data that genuinely has segments.
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    who = rng.integers(0, 3, 400)
    real = np.clip(np.round(centres[who] + rng.normal(0, 0.5, (400, 5))), 1, 5).astype(int)
    df = pd.DataFrame(real, columns=[f"q{i+1}" for i in range(5)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
    assert "one group, not several" not in r["digest"], "warned about real structure"
    assert r["k"] == 3


def test_it_recovers_the_number_of_groups_across_realistic_conditions():
    """The tool's central claim, measured against data whose answer is known.

    Swept at three seeds per condition (21 analyses) while writing this: well-separated k=3,
    overlapping k=3, sizes of 80/15/5, three real questions plus five that separate nobody, two
    elongated bands, k=5, and pure noise. It got 19 of 21 right. Both misses were the overlapping
    case, where it merged two segments into k=2 — and downgraded the confidence when it did.

    This keeps one seed per condition so the suite stays quick. The stochastic part of the claim
    is in the test below.
    """
    conditions = {
        "separated": (np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float),
                      None, 0.5, 3),
        "unequal 80/15/5": (np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float),
                            [0.80, 0.15, 0.05], 0.5, 3),
        "five groups": (np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1],
                                  [5, 5, 1, 1, 5], [1, 1, 3, 3, 1]], float), None, 0.5, 5),
    }
    for name, (centres, weights, spread, true_k) in conditions.items():
        rng = np.random.default_rng(0)
        who = rng.choice(len(centres), 400, p=weights)
        X = _likert(centres[who] + rng.normal(0, spread, (400, centres.shape[1])))
        df = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(X.shape[1])])
        df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
        r = sk.run_analysis(df.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=8, **FAST))
        assert r["k"] == true_k, f"{name}: found {r['k']} groups, not {true_k}"
        assert r["confidence"] == "high", f"{name}: recovered k but reported {r['confidence']}"

    # Questions that separate nobody must not create groups of their own (Dolnicar's point).
    rng = np.random.default_rng(1)
    centres = np.array([[5, 1, 5], [1, 5, 1], [3, 3, 5]], float)
    who = rng.integers(0, 3, 400)
    X = np.hstack([_likert(centres[who] + rng.normal(0, 0.5, (400, 3))),
                   _likert(rng.normal(3, 1.2, (400, 5)))])
    df = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(8)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=8, **FAST))
    assert r["k"] == 3, f"five noise questions changed the answer to {r['k']}"


def test_it_never_claims_high_confidence_for_the_wrong_number_of_groups():
    """Being wrong is survivable. Being wrong and confident is not.

    Overlapping segments are the realistic case and the hard one: measured over three seeds, the
    tool found the true k once and merged two segments into k=2 the other two times. What makes
    that acceptable is that it dropped to "moderate" both times it was wrong, so the report told
    the reader to treat the groups as directional rather than settled.

    A change that improved accuracy by reporting high confidence more often would be a
    regression, and this is what would catch it.
    """
    centres = np.array([[4, 2, 4, 2, 3], [2, 4, 2, 4, 3], [3, 3, 4, 4, 2]], float)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        who = rng.integers(0, 3, 400)
        X = _likert(centres[who] + rng.normal(0, 1.3, (400, 5)))
        df = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(5)])
        df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
        r = sk.run_analysis(df.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=8, **FAST))
        if r["k"] != 3:
            assert r["confidence"] != "high", (
                f"seed {seed}: reported HIGH confidence while finding {r['k']} groups instead "
                "of 3 — a wrong answer presented as a settled one")


def test_scoring_a_single_new_person_works():
    """"Score new people" exists to type a handful of new leads, and it refused every batch of one.

    The rule already knows an item is a rating scale, but applying it went through the DETECTION
    helper, which requires two distinct answers before it will call a column Likert. Scoring one
    person means every column has exactly one answer, so it refused — and a batch of twenty
    failed too if any single question happened to get the same answer from everybody, which on a
    consensus question is normal. Both raised _UNSCORABLE_ITEM at the user.

    Safe to relax when applying rather than detecting: no token appears in two of the built-in
    scales with different numbers, so a lone answer resolves to exactly one value.
    """
    words = {1: "Strongly disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly agree"}
    rng = np.random.default_rng(0)
    rows = [[f"P{i}", *[words[int(np.clip(round(b + rng.normal(0, .6)), 1, 5))]
                        for b in ([5, 1, 5, 2] if i % 2 else [1, 5, 2, 4])]] for i in range(160)]
    df = pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4"])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))
    rule = json.loads(r["files"]["typing_rule.json"])

    one_of_each = {
        "A": {"q1": "Strongly agree", "q2": "Strongly disagree",
              "q3": "Strongly agree", "q4": "Disagree"},
        "B": {"q1": "Strongly disagree", "q2": "Strongly agree",
              "q3": "Disagree", "q4": "Agree"},
    }
    # A single person, and the two opposed mind-sets must not land in the same segment.
    scored = {name: sk.classify_new(rule, pd.DataFrame([{"id": name, **answers}]))
              for name, answers in one_of_each.items()}
    for name, out in scored.items():
        assert len(out) == 1
        assert out["confidence"].iloc[0] > 0.7, f"{name} scored with no confidence: {out}"
    assert scored["A"]["segment"].iloc[0] != scored["B"]["segment"].iloc[0], (
        "opposite answers were typed into the same segment")

    # Two identical people, and a batch where one question got a single answer from everyone.
    pair = pd.DataFrame([{"id": "a", **one_of_each["A"]}, {"id": "b", **one_of_each["A"]}])
    assert len(sk.classify_new(rule, pair)) == 2
    varied = pd.DataFrame([{"id": f"N{i}", "q1": words[1 + i % 5], "q2": words[5 - i % 5],
                            "q3": words[1 + (i * 2) % 5], "q4": words[1 + (i * 3) % 5]}
                           for i in range(20)])
    assert len(sk.classify_new(rule, varied.assign(q2="Agree"))) == 20

    # Numbers still work, and text that is genuinely not a rating scale is still refused rather
    # than guessed at — the point was never to accept anything.
    assert len(sk.classify_new(rule, pd.DataFrame(
        [{"id": "X", "q1": 5, "q2": 1, "q3": 5, "q4": 2}]))) == 1
    with pytest.raises(ValueError, match="_UNSCORABLE_ITEM"):
        sk.classify_new(rule, pd.DataFrame(
            [{"id": "X", "q1": "banana", "q2": "kiwi", "q3": "fig", "q4": "plum"}]))


def test_typing_a_person_who_is_nothing_like_the_survey_says_so():
    """Confidence is what stops a nonsense row being labelled as if it were a real customer.

    The rule assigns a segment to whatever it is given — nearest centroid always has a winner.
    What protects the CRM is that the number next to it collapses toward chance (1/k) when the
    respondent is nowhere near any segment, so a filter on confidence catches them.
    """
    rng = np.random.default_rng(0)
    rows = [[f"P{i}", *[int(np.clip(round(b + rng.normal(0, .6)), 1, 5))
                        for b in ([5, 1, 5, 2] if i % 2 else [1, 5, 2, 4])]] for i in range(160)]
    df = pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4"])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))
    rule = json.loads(r["files"]["typing_rule.json"])
    chance = 1.0 / len(rule["classes"])

    real = sk.classify_new(rule, pd.DataFrame([{"id": "real", "q1": 5, "q2": 1, "q3": 5, "q4": 2}]))
    assert real["confidence"].iloc[0] > 0.8

    for label, row in [("answered nothing", {"q1": np.nan, "q2": np.nan,
                                             "q3": np.nan, "q4": np.nan}),
                       ("straightlined", {"q1": 3, "q2": 3, "q3": 3, "q4": 3}),
                       ("off the scale entirely", {"q1": 900, "q2": -900,
                                                   "q3": 900, "q4": -900})]:
        out = sk.classify_new(rule, pd.DataFrame([{"id": "x", **row}]))
        assert out["confidence"].iloc[0] < chance + 0.1, (
            f"{label} was typed with {out['confidence'].iloc[0]:.2f} confidence, which reads as "
            "a real customer")

    # Order of columns and unrelated extra columns must not change who somebody is.
    base = pd.DataFrame([{"id": "n", "q1": 5, "q2": 1, "q3": 5, "q4": 2}])
    shuffled = base[["id", "q4", "q3", "q2", "q1"]].assign(favourite_colour="blue")
    assert (sk.classify_new(rule, base)["segment"].iloc[0]
            == sk.classify_new(rule, shuffled)["segment"].iloc[0])
    with pytest.raises(ValueError, match="missing required item"):
        sk.classify_new(rule, base.drop(columns=["q3"]))


def test_saving_one_project_from_several_threads_at_once(tmp_path):
    """The store writes to disk from a threaded server, and one project is saved repeatedly —
    after the analysis, after every chat reply, after the groups are named. Two of those
    overlapping is ordinary, not exotic.

    The scratch file used for the atomic rename had a fixed name, so overlapping saves of the
    same project shared it: the first rename moved it away and the second raised
    FileNotFoundError straight out of the request handler. The user saw an error and lost the
    save. It failed within the first few of 60 concurrent attempts.
    """
    import concurrent.futures

    store = webapp.ProjectStore(tmp_path)

    def save(i):
        store.save("sess-abc", {"title": f"survey {i}", "k": 3, "n_people": 100 + i,
                                "report_html": "x" * 40_000}, raw=b"id,q1\n1,2\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(save, range(60)))          # raises if any save failed

    for f in tmp_path.glob("*.json"):
        json.loads(f.read_text())                # every file is complete, not half-written
    assert list(tmp_path.glob("*.tmp")) == [], "scratch files were left behind"
    assert store.load("sess-abc") is not None
    assert len(store.list()) == 1


def test_the_project_store_survives_hostile_and_broken_input(tmp_path):
    """It holds the user's surveys, including the original uploads, and its directory is one the
    user can open. It has to stay inside itself and keep working when something on disk is not
    what it expects."""
    store = webapp.ProjectStore(tmp_path)

    # A project id reaches this from a request body, so it is not trusted.
    for pid in ["../../ESCAPED", "..", "/etc/passwd", "....//....//x", "", "  ", "a" * 300]:
        store.save(pid, {"title": "x", "k": 2})
        store.delete(pid)
    assert not [p for p in tmp_path.parent.iterdir() if "ESCAPED" in p.name], "wrote outside"

    # One unreadable project must not take the sidebar down with it.
    store.save("good", {"title": "fine", "k": 2, "n_people": 10})
    (tmp_path / "broken.meta.json").write_text("{not json at all")
    (tmp_path / "broken.json").write_text("{also broken")
    assert [d["title"] for d in store.list()] == ["fine"]
    assert store.load("broken") is None

    # An upload too big to keep is declined without losing the analysis it belongs to.
    store.save("big", {"title": "big", "k": 2}, raw=b"x" * (store.MAX_RAW + 1))
    assert not (tmp_path / "big.data").exists()
    assert store.load("big") is not None


def test_a_messy_maxdiff_export_is_read_without_losing_people_quietly():
    """Real exports are not tidy, and the ways they are untidy must not change the answer
    silently. The one that matters most is respondent loss: someone who skipped the exercise
    disappears from the utilities, and a study that reports 400 respondents when 380 were
    analysed has the wrong sample size in its write-up. Sets are counted; people are now too.
    """
    md = pytest.importorskip("maxdiff")
    rng = np.random.default_rng(0)
    items = [f"item_{i}" for i in range(8)]

    def export(n_resp=10, n_sets=8):
        rows = []
        for r in range(n_resp):
            for s in range(n_sets):
                shown = list(rng.choice(items, 4, replace=False))
                for i, item in enumerate(shown):
                    rows.append([f"R{r}", f"R{r}-s{s}", item,
                                 "best" if i == 0 else "worst" if i == 1 else ""])
        return pd.DataFrame(rows, columns=["respondent_id", "set", "item", "choice"])

    # Shapes a survey tool really does emit, none of which should raise or drop anybody.
    for label, frame in [
        ("upper-cased labels", export().assign(choice=lambda d: d["choice"].str.upper())),
        ("padded labels", export().replace({"choice": {"best": "  Best "}})),
        ("numeric item codes",
         export().assign(item=lambda d: d["item"].str.replace("item_", "").astype(int))),
    ]:
        _, _, _, names, resp = md.read_maxdiff(frame)
        assert len(resp) == 10, f"{label}: lost respondents"
        assert len(names) == 8, f"{label}: lost items"

    # A respondent who answered nothing is dropped — and said so, with the count.
    silent = export()
    silent.loc[silent["respondent_id"].isin(["R0", "R3"]), "choice"] = ""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        _, _, _, _, resp = md.read_maxdiff(silent)
    said = printed.getvalue()
    assert len(resp) == 8
    assert "2 of 10 respondents" in said, f"respondent loss was not reported: {said!r}"

    # Sets abandoned partway contribute nothing at all — they are padding, not data.
    ragged = pd.concat([export(n_resp=9, n_sets=8),
                        export(n_resp=1, n_sets=2).assign(respondent_id="RQ",
                                                          set=lambda d: "RQ-" + d["set"])])
    design, best, worst, names, resp = md.read_maxdiff(ragged)
    quitter = resp.index("RQ")
    padded = (design[quitter] < 0).all(axis=1)
    assert padded.sum() > 0, "expected the short respondent to be padded"

    beta = rng.normal(0, 1, (1, len(names)))
    real = ~padded
    only_real = md._loglik(beta, design[quitter][real][None], best[quitter][real][None],
                           worst[quitter][real][None], design[quitter][real][None] >= 0)
    with_padding = md._loglik(beta, design[quitter][None], best[quitter][None],
                              worst[quitter][None], design[quitter][None] >= 0)
    # Compared as arrays: _loglik returns one value per respondent, and NumPy 2 refuses to
    # convert even a one-element array with float(). That passed here on an older NumPy and
    # failed on CI, which is exactly what the version matrix is for.
    assert np.allclose(only_real, with_padding), (
        "abandoned sets are being scored as if they were answers")


def test_a_hostile_column_name_cannot_inject_markup_through_a_chart():
    """Charts are injected into the page with dangerouslySetInnerHTML, so their content is
    trusted — and every axis label comes from the uploaded file, which is trusted by nobody. A
    spreadsheet a third party emails you is attacker-controlled the moment you analyse it.

    matplotlib escapes text into SVG correctly. This pins that, because the day it stops being
    true nothing else in the chain would catch it.
    """
    import charts

    hostile = "</text><script>alert(1)</script><text>"
    centroids = pd.DataFrame(
        np.random.default_rng(2).normal(0, 1, (3, 4)),
        columns=[hostile, "q2 <script>x</script>", 'a&b "quoted"', "Kön"],
        index=["Seg <b>0</b>", "Segment 1", "Segment 2"])

    for draw in (charts.chart_profiles, charts.chart_heatmap):
        chart = draw(centroids)
        assert chart, draw.__name__
        assert "<script>" not in chart["svg"], f"{draw.__name__} wrote a live script tag"
        assert "</text><script>" not in chart["svg"], f"{draw.__name__} let a label break out"

    # Group names are user-typed too, and they go into the legend.
    named = charts.chart_profiles(centroids, names=["<img src=x onerror=alert(1)>", "B", "C"])
    assert "<img src=x" not in named["svg"]


def test_two_analyses_at_once_do_not_swap_their_chart_failures():
    """The server is threaded — the team is meant to be able to use it at once.

    The account of why a chart could not be drawn used to live in a module-level list, which
    every request shared. Measured on 18 concurrent runs with a third of them failing: three
    healthy runs were told charts had failed that never did, four failed runs were given no
    reason at all, and five were handed other people's failures alongside their own. Somebody
    would have been told their survey was broken because a colleague's was.
    """
    import concurrent.futures
    import threading

    import charts

    class _Seg:
        labels = np.array([0] * 30 + [1] * 30)
        recommended_k = 2
        X = np.random.default_rng(0).normal(size=(60, 4))
        centroids = pd.DataFrame({"q1": [1.0, 4.0], "q2": [4.0, 1.0], "q3": [2.0, 3.0]})
        diagnostics = pd.DataFrame({"k": [2, 3], "silhouette": [0.5, 0.3],
                                    "stability_ARI": [0.9, 0.6],
                                    "prediction_strength": [0.85, 0.5]})

    real_chart = charts.chart_heatmap
    # Half the callers hit a chart that raises; the other half are perfectly healthy.
    should_fail = threading.local()

    def flaky(*a, **kw):
        if getattr(should_fail, "yes", False):
            raise RuntimeError("this file is corrupt")
        return real_chart(*a, **kw)

    def run(i):
        should_fail.yes = bool(i % 2)
        mine = []
        drawn = charts.build_charts(_Seg(), "kmeans", errors=mine)
        return bool(i % 2), len(drawn), list(mine)

    charts.chart_heatmap = flaky
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(run, range(18)))
    finally:
        charts.chart_heatmap = real_chart

    for failed, drawn, errors in results:
        if failed:
            assert drawn == 4, f"expected the failing chart to be missing, got {drawn}"
            assert len(errors) == 1, f"a failed run must get its own one reason, got {errors}"
            assert "corrupt" in errors[0]
        else:
            assert drawn == 5, f"a healthy run lost a chart: {drawn}"
            assert errors == [], f"a healthy run was handed someone else's failure: {errors}"


def test_the_svg_backend_is_an_explicit_dependency():
    """The packaged app analysed a survey perfectly and drew nothing at all.

    matplotlib resolves the writer for an output format lazily, inside savefig, so a bundler
    doing static analysis saw only the Agg backend selected at import time and left backend_svg
    out. Every chart then failed with ModuleNotFoundError, six times, in a build that otherwise
    looked healthy. charts.py imports both backends by name so any packager keeps them.
    """
    import charts

    assert charts._REQUIRED_BACKENDS, "the backends are no longer pinned"
    names = {module.__name__ for module in charts._REQUIRED_BACKENDS}
    assert "matplotlib.backends.backend_svg" in names
    assert "matplotlib.backends.backend_agg" in names

    # And the thing those backends exist for actually works end to end.
    chart = charts.chart_segment_map(
        np.random.default_rng(0).normal(size=(60, 4)), np.repeat([0, 1], 30))
    assert chart["svg"].startswith("<svg") and chart["png_b64"]


def test_output_is_utf8_so_windows_consoles_do_not_break_the_analysis():
    """The first Windows build failed every analysis with "something went wrong reading that
    file". The real cause was the confidence line printing a coloured circle to a console using
    Windows' legacy cp1252 encoding, which cannot encode it — UnicodeEncodeError, thrown from
    inside the run and reported as a file problem.

    The emoji was only the symptom. Any Swedish column name would have done the same, which for
    a tool built for Nordic surveys is the more serious half of the bug."""
    assert (sys.stdout.encoding or "").lower().replace("-", "") == "utf8"

    # The characters that actually broke it: the confidence lights, and Swedish.
    for char in ("\U0001f7e2", "\U0001f7e1", "\U0001f534", "Kön", "Ålder", "Göteborg"):
        char.encode(sys.stdout.encoding)          # raises if the stream could not carry it


def test_routes_match_exactly_so_the_app_owns_every_other_path():
    """Routes were matched by prefix, which handed API responses to paths that merely started the
    same way. `/projects-of-mine` was answered with the project list, and — worse — anything
    beginning `/quit` shut the server down, so a single-page route named `/quit-guide` would have
    closed the app. Exact matching, with everything else falling through to the interface."""
    import threading
    import time
    import urllib.request

    port = 8793
    threading.Thread(target=lambda: sk.serve(port), daemon=True).start()
    time.sleep(3)

    def fetch(path):
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}")
        return r.status, r.headers.get("Content-Type", ""), r.headers.get("Cache-Control")

    # Real routes still answer with JSON.
    for route in ("/projects", "/settings"):
        status, ctype, _ = fetch(route)
        assert status == 200 and ctype.startswith("application/json"), route

    # Anything else is the app, not an API call that happens to share a prefix.
    for impostor in ("/projects-of-mine", "/project-notes", "/settings-help", "/quitting-time",
                     "/downloads", "/some/deep/route"):
        status, ctype, _ = fetch(impostor)
        assert status == 200 and ctype.startswith("text/html"), impostor

    # The server is still running: /quitting-time must not have shut it down.
    assert fetch("/projects")[0] == 200

    # And the HTML shell is never cached, however it was reached. It names which asset hashes are
    # current, so a stale copy loads a bundle that no longer exists — a blank page that survives
    # a reload. Keying this on the path missed query strings and every fallback route.
    for path in ("/", "/index.html", "/?utm_source=news", "/some/deep/route"):
        assert fetch(path)[2] == "no-store", path


def test_a_missing_interface_says_how_to_build_it():
    """webui/ is committed, so this is only reachable in a checkout where it was deleted. It must
    name the command that fixes it — a blank page would look like the whole tool is broken, when
    in fact the statistics are fine and only the interface is absent."""
    page = webapp._missing_ui_page()
    assert "npm run build" in page and "not been built" in page
    # The command line still works without any of this, and should say so.
    assert "segment_kmeans.py your_survey.csv" in page


def test_report_not_dumped_to_stdout_when_saved(tmp_path, capsys):
    """When results are written to a folder, the full report should not also flood stdout — it is
    redundant noise for scripted/professional use. Without an outdir it still prints."""
    df, _, _ = structured(seed=8)
    sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=4, **FAST)).run(
        _write(tmp_path, df), id_col="id", force_k=3, outdir=str(tmp_path / "o"))
    out = capsys.readouterr().out
    assert "Generated by segment_kmeans version" not in out and "Saved to" in out


def test_pluralization_of_unit_words():
    assert sk._plural("group") == "groups" and sk._plural("class") == "classes"
    assert sk._plural("segment") == "segments"


def test_lca_parity_typing_demographics_weights(tmp_path):
    """The categorical path has parity with k-means: a typing tool (with an exportable rule that
    types new respondents), demographics profiling, weighted population sizes, and correct grammar."""
    import json
    rng = np.random.default_rng(0); n = 600; truth = rng.integers(0, 3, n)
    proto = np.full((3, 6), 0.15); proto[0, 0:2] = 0.9; proto[1, 2:4] = 0.9; proto[2, 4:6] = 0.9
    ans = np.array(["No", "Yes"])
    df = pd.DataFrame({f"q{j}": ans[(rng.random(n) < proto[truth, j]).astype(int)] for j in range(6)})
    df.insert(0, "id", range(n)); df["Gender"] = rng.choice(["M", "F"], n)
    df["weight"] = np.where(truth == 0, 0.5, 1.5)
    clean, method, idc, items, plan = sk.auto_prepare(df)
    assert method == "lca" and plan["weight"] == "weight" and plan["demographics"] == ["Gender"]
    out = tmp_path / "lca"
    seg = sk.LatentClassSegmenter(sk.SegmentationConfig(k_min=2, k_max=5, **FAST)).run(
        clean, id_col=idc, item_cols=items, force_k=3, outdir=str(out),
        demographics=df[[idc, "Gender"]], weights=df["weight"].to_numpy())
    r = seg.report_markdown
    assert "predictability (typing tool)" in r
    assert "Profiling the classes against demographics" in r
    assert "population_share" in r and "classs" not in r          # weighted sizes + correct grammar
    assert (out / "latent_class_typing_rule.json").exists()
    tn = rng.integers(0, 3, 60)
    newdf = pd.DataFrame({f"q{j}": ans[(rng.random(60) < proto[tn, j]).astype(int)] for j in range(6)})
    newdf.insert(0, "id", range(1000, 1060))
    rule = json.loads((out / "latent_class_typing_rule.json").read_text())
    assigned = sk.classify_new_lca(rule, newdf, id_col="id")
    assert adjusted_rand_score(tn, assigned["segment"]) > 0.6      # types a fresh cohort
    assert "id" in assigned.columns and assigned["confidence"].between(0, 1).all()


def test_negative_weights_treated_as_zero(tmp_path):
    df, _, _ = structured(n_per=(60, 60), n_items=6, sep=4, seed=2)
    w = np.ones(len(df)); w[:5] = -3.0                          # nonsensical negative weights
    seg = sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=3, **FAST)).run(
        _write(tmp_path, df), id_col="id", force_k=2, weights=w)
    assert "population_share" in seg.sizes.columns
    assert seg.sizes["population_share"].between(0, 1).all()
    assert abs(seg.sizes["population_share"].sum() - 1.0) < 0.01


# ======================= AI interpretation layer (ai_interpret.py) + web endpoints =======================
import ai_interpret as ai


def test_ai_config_roundtrip(tmp_path, monkeypatch):
    """The API key is saved to a local file, read back trimmed, and cleared; the ANTHROPIC_API_KEY
    environment variable takes precedence when set."""
    monkeypatch.setattr(ai, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(ai, "_CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ai.load_api_key() is None
    # Realistic length: `save_api_key` now checks the shape, and a ten-character stand-in is
    # exactly the kind of value it exists to refuse.
    fake = "sk-ant-api03-" + "x" * 90
    ai.save_api_key(f"  {fake}  ")
    assert ai.load_api_key() == fake                            # trimmed
    assert ai.key_source() == "this app's Settings"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    assert ai.load_api_key() == "sk-ant-env"                    # env wins over the file
    assert "environment" in ai.key_source()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ai.clear_api_key()
    assert ai.load_api_key() is None


@pytest.mark.parametrize("junk, why", [
    ("hello", "no prefix, far too short"),
    ("sk-ant-", "the prefix and nothing else"),
    ("sk-ant-abc", "a plausible-looking stub"),
    ("sk-proj-" + "x" * 90, "the right length but another vendor's prefix"),
])
def test_a_value_that_cannot_be_a_key_is_refused_when_it_is_typed(tmp_path, monkeypatch, junk, why):
    """Found in the field. A nine-character string was sitting in the config file; `status()`
    reported the app as configured, and the only sign of trouble came after uploading a survey,
    waiting for it to run and clicking Suggest names — at which point the error blamed the key
    without saying it had never been one.

    Checking at the moment of typing turns a confusing failure five minutes later into an obvious
    one straight away.
    """
    monkeypatch.setattr(ai, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(ai, "_CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ai.AIError) as caught:
        ai.save_api_key(junk)
    assert "does not look like" in str(caught.value), why
    assert ai.load_api_key() is None, "a value that was refused still got written to the file"


def test_ai_build_messages_embeds_report_then_appends():
    """Turn 1 embeds the full report in the user message; later turns carry the running history and
    append only the new question (the report is not re-sent every turn)."""
    first = ai.build_messages("REPORT_TEXT", None, None)
    assert len(first) == 1 and first[0]["role"] == "user"
    # The report rides in a content block rather than a bare string so it can carry a cache
    # breakpoint — it is the large, unchanging part resent on every follow-up.
    block = first[0]["content"][0]
    assert "REPORT_TEXT" in block["text"]
    assert block["cache_control"] == {"type": "ephemeral"}
    hist = [{"role": "user", "content": "...REPORT..."},
            {"role": "assistant", "content": "an interpretation"}]
    nxt = ai.build_messages("REPORT_TEXT", "which segment first?", hist)
    assert len(nxt) == 3 and nxt[-1] == {"role": "user", "content": "which segment first?"}
    assert "REPORT_TEXT" not in nxt[-1]["content"]


def test_ai_chat_requires_sdk_and_key(monkeypatch):
    """No SDK -> a friendly 'nosdk' AIError; SDK present but no key -> 'nokey'. Never a traceback."""
    monkeypatch.setattr(ai, "have_sdk", lambda: False)
    with pytest.raises(ai.AIError) as e1:
        ai.chat_once([], "digest", None)
    assert e1.value.kind == "nosdk"
    monkeypatch.setattr(ai, "have_sdk", lambda: True)
    monkeypatch.setattr(ai, "load_api_key", lambda: None)
    with pytest.raises(ai.AIError) as e2:
        ai.chat_once([], "digest", None)
    assert e2.value.kind == "nokey"


def test_ai_status_shape(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    st = ai.status()
    assert set(st) == {"sdk_installed", "configured", "source", "env_key", "model"}
    assert st["model"] == ai.MODEL


def test_row_counters_and_stray_ids_are_never_clustered_on():
    """Regression: a row counter ('City_n' = 1..49) or a second numeric id column used to be treated
    as a rating and clustered on, injecting a straight-line gradient that is pure bookkeeping. It
    must be set aside — whatever the column is called — while genuine answers survive untouched."""
    rng = np.random.default_rng(0)
    def cont(df):
        return sk.classify_columns(df)["continuous"]

    # the exact cities-file failure, plus a shuffled counter and a second numeric id
    counter = pd.DataFrame({"City_n": range(1, 50), "q1": rng.integers(1, 6, 49),
                            "q2": rng.integers(1, 6, 49)})
    assert cont(counter) == ["q1", "q2"]
    shuffled = pd.DataFrame({"row": rng.permutation(np.arange(1, 41)),
                             "q1": rng.integers(1, 6, 40), "q2": rng.integers(1, 6, 40)})
    assert cont(shuffled) == ["q1", "q2"]
    second_id = pd.DataFrame({"respondent_id": [f"R{i}" for i in range(40)], "user_id": range(1000, 1040),
                              "q1": rng.integers(1, 6, 40), "q2": rng.integers(1, 6, 40)})
    plan = sk.classify_columns(second_id)
    assert plan["id"] == "respondent_id" and plan["continuous"] == ["q1", "q2"]

    # ...and genuine data must NOT be mistaken for a counter (this is the dangerous direction)
    assert cont(pd.DataFrame({"u1": rng.normal(0, 1, 60), "u2": rng.normal(0, 1, 60)})) == ["u1", "u2"]
    assert cont(pd.DataFrame({"score": np.linspace(.5, 30.2, 60),          # all-distinct floats
                              "q2": rng.integers(1, 6, 60)})) == ["score", "q2"]
    assert cont(pd.DataFrame({"spend_gbp": rng.integers(10, 900, 60),
                              "q2": rng.integers(1, 6, 60)})) == ["spend_gbp", "q2"]

    # a bare counter with no named id becomes the id, so results stay labelled by the file's own key
    assert sk.classify_columns(counter)["id"] == "City_n"


def test_typing_rule_scores_a_raw_export_with_text_likert_answers(tmp_path):
    """Regression: the typing tool was unusable on a REAL follow-up export. Auto-detection recodes
    'Strongly agree' to 1-5 when building the segments, but new files still arrive as words, and
    scoring crashed on them. It must recode the same way, and must label each person with their id
    (a scored list you cannot tie back to people is worthless for targeting)."""
    import json
    rng = np.random.default_rng(3)
    n = 260
    seg = rng.integers(0, 2, n)
    agree = np.array(["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"])
    def make(idx):
        s = seg[idx]
        text_code = np.clip(np.round(rng.normal(np.where(s == 0, 3.4, 0.6), .6)), 0, 4).astype(int)
        return pd.DataFrame({
            "respondent_id": [f"P{i}" for i in idx],
            "q_rating_a": np.clip(np.round(rng.normal(np.where(s == 0, 4.5, 1.5), .5)), 1, 5).astype(int),
            "q_rating_b": np.clip(np.round(rng.normal(np.where(s == 0, 1.5, 4.5), .5)), 1, 5).astype(int),
            "q_text_scale": agree[text_code],
        })
    train_idx, new_idx = np.arange(0, 200), np.arange(200, n)
    r = sk.run_analysis(make(train_idx).to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))
    rule = json.loads(r["files"]["typing_rule.json"])
    assert "q_text_scale" in rule["items"]                 # the recoded item is part of the rule

    scored = sk.classify_new(rule, make(new_idx))          # raw words, exactly as exported
    # The id, the segment and the confidence, in that order and first: a scored list you cannot
    # tie back to people is worthless for targeting. Later columns may be added after them —
    # answers_off_the_original_scale is one — so this pins the contract rather than the exact set.
    assert list(scored.columns)[:3] == ["respondent_id", "segment", "confidence"]
    assert len(scored) == len(new_idx)
    truth = seg[new_idx]
    assert adjusted_rand_score(truth, scored["segment"]) > 0.7   # it really recovers the mind-sets

    with pytest.raises(ValueError):                        # a genuinely unscorable column still errors
        bad = make(new_idx); bad["q_text_scale"] = "banana"
        sk.classify_new(rule, bad)


def test_user_can_override_which_questions_group_people():
    """The detector's guess is a starting point, not a verdict. A user must be able to group people
    on columns it set aside — e.g. clustering cities on ethnicity/income, which auto-detection
    treats as background traits and therefore refuses to use."""
    rng = np.random.default_rng(5)
    n = 60
    df = pd.DataFrame({
        "City_n": range(1, n + 1),                      # a row counter: never an answer
        "gender_mix": rng.integers(20, 80, n),          # "demographic" by name -> set aside
        "income": rng.integers(10, 40, n),              # ditto
        "q_rating": rng.integers(1, 6, n),
    })
    # auto: the two demographic-looking columns are set aside, leaving too little to group on
    plan = sk.classify_columns(df)
    assert set(plan["demographics"]) == {"gender_mix", "income"} and plan["continuous"] == ["q_rating"]

    # the override: group on exactly what the user ticked
    clean, method, idc, items, _ = sk.auto_prepare(df, force_items=["gender_mix", "income"])
    assert method == "kmeans" and items == ["gender_mix", "income"]
    assert idc == "City_n"                              # the counter still labels the results
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST),
                        force_items=["gender_mix", "income"])
    assert r["n_people"] == n
    assert r["columns"]["gender_mix"] == "used" and r["columns"]["income"] == "used"
    assert "City_n" not in r["columns"]        # the id is not offered as something to group on
    with pytest.raises(ValueError):                     # fewer than two questions is not a grouping
        sk.auto_prepare(df, force_items=["income"])


def test_refuses_to_treat_a_measurement_as_categories():
    """Regression: picking a text column together with a high-cardinality number (salary) pushed the
    run down the categorical path, inventing one 'category' per person. That fits perfectly and
    reported 'Confidence: High' while meaning nothing — the exact failure this tool exists to
    prevent. It must refuse, in plain language."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"id": range(n), "salary": rng.integers(20000, 90000, n),
                       "pref": rng.choice(["Yes", "No"], n), "q": rng.integers(1, 6, n)})
    with pytest.raises(ValueError) as e:
        sk.auto_prepare(df, force_items=["salary", "pref"])
    msg = sk._explain_run_error(str(e.value))
    assert "different answers" in msg and "salary" in msg and "meaningless" in msg
    # a genuine all-categorical pick, and a genuine all-numeric pick, both still work
    assert sk.auto_prepare(df, force_items=["pref", "q"])[1] == "lca"
    assert sk.auto_prepare(df, force_items=["salary", "q"])[1] == "kmeans"


def test_swedish_survey_profiles_demographics_instead_of_grouping_on_them():
    """Regression, and the one that matters most for a Nordic team: the word-splitter used [a-z]+,
    which shredded 'Kön' into ['k','n'] and 'Ålder' into ['lder']. Swedish gender/age/university
    columns were therefore never recognised as background traits and ended up FORMING the segments —
    the circular result the whole tool is built to prevent. Swedish answer scales were also unread,
    losing the ordering of 'Instämmer helt' over 'Instämmer'."""
    rng = np.random.default_rng(4)
    n = 180
    t = rng.integers(0, 2, n)
    sv = np.array(["Instämmer inte alls", "Instämmer inte", "Neutral", "Instämmer", "Instämmer helt"])
    def col(hi, lo):
        return sv[np.clip(np.round(rng.normal(np.where(t == 0, hi, lo), .7)).astype(int), 0, 4)]
    df = pd.DataFrame({
        "Svarsnummer": range(1, n + 1),
        "Jag vill träffa nya människor på campus": col(3.6, 1.2),
        "Jag oroar mig för integritet i appar": col(1.2, 3.4),
        "Jag använder gärna en app för studentevent": col(3.5, 1.4),
        "Kön": rng.choice(["Man", "Kvinna"], n),
        "Universitet": rng.choice(["Lund", "KTH", "Uppsala"], n),
        "Ålder": rng.choice(["18-24", "25-34"], n),
    })
    clean, method, idc, items, plan = sk.auto_prepare(df)
    assert set(plan["demographics"]) == {"Kön", "Universitet", "Ålder"}
    assert "Kön" not in items and "Universitet" not in items and "Ålder" not in items
    assert method == "kmeans"                      # the Swedish scale was read as ordered 1-5
    assert len(items) == 3
    r = sk.run_analysis(df.to_csv(index=False).encode(), cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    got = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))["segment"]
    assert adjusted_rand_score(t, got) > 0.8       # and it still finds the real split

    # the English path must be untouched by the unicode change
    en = pd.DataFrame({"Gender": ["M", "F"] * 40, "Age": ["18-24", "25-34"] * 40,
                       "q1": np.resize([1, 5], 80), "q2": np.resize([5, 1], 80)})
    p_en = sk.classify_columns(en)
    assert set(p_en["demographics"]) == {"Gender", "Age"} and p_en["continuous"] == ["q1", "q2"]


def test_google_forms_export_including_select_all_questions():
    """Google Forms is how this team actually collects data, and its 'select all that apply'
    questions pack every ticked option into one comma-separated cell. Treated naively, each
    COMBINATION becomes its own category — four options turn into fourteen pseudo-categories.
    They must be split into yes/no columns, kept out of the way when real rating questions exist,
    and never confused with free text that happens to contain a comma."""
    rng = np.random.default_rng(7)
    n = 140
    t = rng.integers(0, 2, n)
    brands = np.array(["Brand A", "Brand B", "Brand C", "Brand D"])

    def checkbox(seg):
        pool = brands[[0, 1]] if seg == 0 else brands[[2, 3]]
        return ", ".join(sorted(rng.choice(pool, size=rng.integers(1, 3), replace=False)))

    gf = pd.DataFrame({
        "Timestamp": [f"2026/05/0{i % 9 + 1} 10:2{i % 9}:11" for i in range(n)],
        "Email Address": [f"s{i}@lu.se" for i in range(n)],
        "Which of these apps have you used? (select all that apply)": [checkbox(x) for x in t],
        "Rate the following [Ease of use]":
            np.clip(np.round(rng.normal(np.where(t == 0, 4.4, 2), .7)), 1, 5).astype(int),
        "Rate the following [Privacy]":
            np.clip(np.round(rng.normal(np.where(t == 0, 2, 4.3), .7)), 1, 5).astype(int),
        "How likely are you to recommend us?":
            np.clip(np.round(rng.normal(np.where(t == 0, 4.5, 2), .8)), 1, 5).astype(int),
        "What is your gender?": rng.choice(["Male", "Female"], n),
        "Anything else you want to tell us?": ["" if rng.random() < .85 else f"c{i}" for i in range(n)],
    })
    plan = sk.classify_columns(gf)
    ms = plan["multiselect"]
    assert len(ms) == 1 and sorted(next(iter(ms.values()))) == sorted(brands.tolist())
    assert plan["id"] == "Email Address"                       # Google Forms email column
    assert "Timestamp" in plan["skipped"]
    assert plan["demographics"] == ["What is your gender?"]

    # ratings present -> the select-all is set aside, so it cannot outvote them by column count
    clean, method, idc, items, plan2 = sk.auto_prepare(gf)
    assert method == "kmeans" and len(items) == 3 and all("—" not in i for i in items)
    r = sk.run_analysis(gf.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    got = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))["segment"]
    assert adjusted_rand_score(t, got) > 0.9

    # only a select-all -> the yes/no columns become the grouping, with readable names
    only = gf[["Email Address", "Which of these apps have you used? (select all that apply)"]]
    _, m2, _, items2, _ = sk.auto_prepare(only)
    assert m2 == "kmeans" and len(items2) == 4
    assert all(i.startswith("Which of these apps have you used — ") for i in items2)

    # free text containing commas must NOT be shredded into fake options
    ft = pd.Series(["I like it, a lot", "Too expensive, sadly", "Great, would recommend",
                    "Not sure, maybe", "Fine, I guess", "Could be better, honestly"] * 4)
    assert sk._multiselect_options(ft) is None


def test_projects_persist_to_disk_and_reopen(tmp_path):
    """Analysed surveys are saved as projects, like a chat history, so closing the app does not
    throw away an analysis. A saved project must come back with its report, its downloads and its
    conversation — and the store must never write outside its own folder."""
    store = webapp.ProjectStore(tmp_path)
    store.save("abc123", {"title": "campus_wave1.csv", "digest": "# Report",
                          "report_html": "<h2>In plain language</h2>" + "x" * 300,
                          "files": {"segment_assignments.csv": "id,segment\n1,0\n"},
                          "messages": [{"role": "user", "content": "hi"}],
                          "transcript": [{"role": "you", "text": "Analyse: campus_wave1.csv"}],
                          "k": 3, "n_people": 150, "confidence": "high", "columns": {"q1": "used"}})
    listed = store.list()
    assert len(listed) == 1 and listed[0]["title"] == "campus_wave1.csv"
    assert listed[0]["k"] == 3 and listed[0]["confidence"] == "high" and listed[0]["updated"]

    back = store.load("abc123")                       # survives as if the app had restarted
    assert back["files"]["segment_assignments.csv"].startswith("id,segment")
    assert back["transcript"][0]["text"].startswith("Analyse:")
    assert back["k"] == 3

    # a hostile id must not escape the store directory
    store.save("../../../etc/evil", {"title": "nope"})
    assert not (tmp_path / ".." / ".." / ".." / "etc" / "evil.json").exists()
    assert all(p.parent == tmp_path for p in tmp_path.glob("*.json"))

    # Regression: the original upload must be kept, or re-grouping a REOPENED project fails while
    # the UI still offers the question picker.
    store.save("withraw", {"title": "wave2.csv", "k": 2}, raw=b"id,q1,q2\n1,5,1\n2,1,5\n")
    assert store.load("withraw")["raw"].startswith(b"id,q1,q2")

    # Regression: the sidebar must not parse every full report just to list titles.
    big = "<p>" + ("x" * 40000) + "</p>"
    store.save("bigone", {"title": "big.csv", "report_html": big, "digest": big, "k": 4})
    meta = sorted(tmp_path.glob("*.meta.json"))
    assert meta, "expected a small summary file per project"
    assert max(p.stat().st_size for p in meta) < 1000        # summaries stay tiny
    assert any(p.stat().st_size > 40000 for p in tmp_path.glob("*.json")
               if not p.name.endswith(".meta.json"))         # the full record is separate
    assert {d["id"] for d in store.list()} >= {"abc123", "withraw", "bigone"}

    store.delete("abc123")
    assert store.load("abc123") is None
    assert store.load("never-existed") is None        # missing project is None, not a crash
    store.delete("withraw")                           # delete removes record, summary AND upload
    assert not list(tmp_path.glob("withraw*"))


# The front end's own hardening — one un-throwable request helper, pre-upload validation,
# dropped-folder handling, no stuck spinner, no injected markup — is now tested where it lives,
# against the real components: frontend/src/**/*.test.tsx, run by `npm test`. Grepping the
# compiled bundle for those strings from here would assert less than those tests do, and would
# break on a minifier rather than on a regression.


def test_markdown_renders_numbered_lists_and_errors_stay_plain_language():
    """Claude may number its points; they must render as a real list, not stray text. And an
    unrecognised technical error must reach a non-expert as a plain sentence, not a raw exception."""
    html = sk._markdown_to_html("## Plan\n1. First step\n2. Second step\n\nAfter")
    assert "<ol>" in html and "<li>First step</li>" in html and "</ol>" in html
    assert "<ul>" in sk._markdown_to_html("- a\n- b")            # bullets unaffected
    assert "<p>Hopkins was 0.76 overall.</p>" in sk._markdown_to_html("Hopkins was 0.76 overall.")
    friendly = sk._explain_run_error("Connection reset by peer")
    assert "Something went wrong" in friendly and "Connection reset by peer" in friendly


def test_ai_request_is_well_formed_against_a_mock_anthropic_server(monkeypatch):
    """Prove the REAL request Claude would receive is correct, without spending a real API key:
    point the Anthropic SDK at a local mock that speaks the streaming wire format, then inspect
    exactly what was sent. Guards the model id, the system prompt, that the report digest actually
    reaches the model, and that a streamed reply is parsed back out."""
    import http.server
    import json as _json
    import threading

    captured = {}
    SSE = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",'
        '"model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta",'
        '"text":"## Your segments\\n- **Champions** lead on consideration."}}\n\n'
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
        '"usage":{"output_tokens":12}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    class Mock(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            captured["path"] = self.path
            captured["body"] = _json.loads(self.rfile.read(n).decode())
            b = SSE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    try:
        reply, history = ai.chat_once([], "# Report\n- Segment 1 is 41% of students", None)
    finally:
        srv.shutdown()

    assert "Champions" in reply                      # the streamed answer was parsed back out
    assert history[-1]["role"] == "assistant"
    body = captured["body"]
    assert captured["path"].split("?")[0].endswith("/v1/messages")
    assert body["model"] == "claude-opus-5"
    assert body["stream"] is True                    # streaming, so long answers don't time out
    assert body["max_tokens"] >= 2000
    assert "segmentation strategist" in body["system"]
    # Opus 5 rejects these outright (400) — a regression here breaks every chat, silently in
    # dev if the reviewer has no key, so pin their absence rather than trusting review.
    for banned in ("temperature", "top_p", "top_k", "budget_tokens"):
        assert banned not in body, f"{banned} is rejected by Opus 5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "medium"
    # Safety classifiers can decline a request; fallbacks turn that into an answer rather than
    # an apology. "default" lets Anthropic route by refusal category instead of us pinning a
    # model we would have to maintain.
    assert body["fallbacks"] == "default"
    assert "beta=true" in captured["path"]            # fallbacks require the beta endpoint
    sent = body["messages"][0]["content"][0]["text"]
    assert "Segment 1 is 41% of students" in sent    # the aggregate report really reaches Claude
    # Cached so follow-up questions are not billed for the whole report again.
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][0]["role"] == "user"


def test_ai_flags_a_truncated_answer(monkeypatch):
    """A reply cut off at the token ceiling must be labelled, never shown as if it were complete."""
    class _Block:
        type, text = "text", "Segment 1 is the biggest and"

    class _Msg:
        stop_reason, content = "max_tokens", [_Block()]

    class _Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self): return _Msg()

    monkeypatch.setattr(ai, "have_sdk", lambda: True)
    monkeypatch.setattr(ai, "load_api_key", lambda: "sk-ant-x")
    import anthropic
    monkeypatch.setattr(anthropic.Anthropic, "__init__", lambda self, **k: None)
    monkeypatch.setattr(anthropic.Anthropic, "messages",
                        property(lambda self: type("M", (), {"stream": lambda *a, **k: _Stream()})()))
    reply, _ = ai.chat_once([], "digest", "which first?")
    assert "cut short" in reply and "Segment 1 is the biggest" in reply


def test_web_server_endpoints_end_to_end(monkeypatch):
    """Start the real local server and exercise the JSON endpoints end to end: the chat page loads, a
    survey 'analyses' into a session, a key saves, and /chat returns Claude's reply — with the no-key
    path returning a friendly 'nokey', never a crash. The (slow) statistics and the (network) Claude
    call are stubbed, so this tests the wiring fast and offline."""
    import json as _json
    import socket
    import threading
    import time
    import urllib.request

    monkeypatch.setattr(sk, "run_analysis", lambda data, cfg=None, force_items=None: {
        "title": "Segmentation report",
        "report_html": "<h2>In plain language</h2><p>Three groups.</p>",
        "digest": "# Report\n- three groups", "k": 3, "n_people": 2,
        "columns": {"q1": "used", "q2": "used", "age": "background"},
        "files": {"segment_assignments.csv": "id,segment\n1,0\n2,1\n",
                  "group_profiles.csv": ",q1\nSegment 0,4.2\n",
                  "typing_rule.json": '{"method": "kmeans", "items": ["q1"]}'}})

    class FakeAI:
        class AIError(Exception):
            def __init__(self, msg, kind="error"):
                super().__init__(msg)
                self.kind = kind

        def __init__(self):
            self.key = None

        def status(self):
            return {"sdk_installed": True, "configured": self.key is not None,
                    "source": "test" if self.key else None, "env_key": False, "model": "claude-opus-5"}

        def chat_once(self, history, digest, question=None, charts=None):
            if self.key is None:
                raise self.AIError("Add your Anthropic API key in Settings.", kind="nokey")
            # Recorded, not asserted on content: this fixture is two respondents and two
            # questions, which honestly supports no charts at all. What matters here is that the
            # server passes the argument through — a double that silently swallowed it would let
            # the wiring rot without a test noticing. The content is covered where charts exist,
            # in test_the_ai_digest_contains_no_individual_respondent_data.
            self.charts_seen = charts
            reply = ("## Your segments\n- **Privacy-First Students** value real-life meetups."
                     if question is None else "Target the largest group first.")
            return reply, list(history) + [{"role": "assistant", "content": reply}]

        def suggest_names(self, digest, k):
            if self.key is None:
                raise self.AIError("Add your key.", kind="nokey")
            return ["Loyal Fans", "Price Hunters"][:k]

        def save_api_key(self, k):
            if not (k or "").strip():
                raise self.AIError("empty", kind="nokey")
            self.key = k.strip()

        def clear_api_key(self):
            self.key = None

    monkeypatch.setattr(sk, "_ai", FakeAI())
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)

    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=sk.serve, kwargs={"port": port}, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def get(path):
        return urllib.request.urlopen(base + path, timeout=5).read().decode()

    def post_json(path, obj):
        req = urllib.request.Request(base + path, data=_json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return _json.loads(urllib.request.urlopen(req, timeout=10).read().decode())

    def post_file(path, name, blob):
        b = "----t"
        body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                f'Content-Type: text/csv\r\n\r\n').encode() + blob + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(base + path, data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        return _json.loads(urllib.request.urlopen(req, timeout=10).read().decode())

    up = False
    for _ in range(60):
        try:
            if "Survey Segmenter" in get("/"):
                up = True
                break
        except Exception:
            time.sleep(0.1)
    assert up, "server did not start"

    try:
        assert _json.loads(get("/settings"))["sdk_installed"] is True
        a = post_file("/analyze", "u.csv", b"id,q1,q2\n1,4,5\n2,1,2\n")
        assert a["ok"] and a["session_id"] and "In plain language" in a["report_html"]
        assert a["ai_available"] is False                       # no key yet
        c0 = post_json("/chat", {"session_id": a["session_id"], "initial": True})
        assert c0["ok"] is False and c0["kind"] == "nokey"      # friendly, not a crash
        st = post_json("/settings", {"api_key": "sk-ant-xyz"})
        assert st["ok"] and st["configured"] is True
        a2 = post_file("/analyze", "u.csv", b"id,q1,q2\n1,4,5\n2,1,2\n")
        assert a2["ai_available"] is True                       # key now present
        c1 = post_json("/chat", {"session_id": a2["session_id"], "initial": True})
        assert c1["ok"] and "Your segments" in c1["reply_html"]
        c2 = post_json("/chat", {"session_id": a2["session_id"], "message": "which first?"})
        assert c2["ok"] and "Target the largest" in c2["reply_html"]
        c3 = post_json("/chat", {"session_id": "nope", "message": "hi"})
        assert c3["ok"] is False and "analyse" in c3["error"].lower()

        # the actionable exports: the growth team must be able to pull the results out
        assert sorted(a2["downloads"]) == ["group_profiles.csv", "segment_assignments.csv",
                                           "typing_rule.json"]
        got = get(f"/download?session_id={a2['session_id']}&file=segment_assignments.csv")
        assert got.startswith("id,segment") and "1,0" in got
        with pytest.raises(urllib.error.HTTPError) as e404:      # unknown file -> clean 404
            urllib.request.urlopen(base + f"/download?session_id={a2['session_id']}&file=nope.csv")
        assert e404.value.code == 404

        # re-grouping on chosen questions, and the guard against a meaningless pick
        rg = post_json("/regroup", {"session_id": a2["session_id"], "items": ["q1", "q2"]})
        assert rg["ok"] and rg["session_id"] == a2["session_id"]
        bad = post_json("/regroup", {"session_id": a2["session_id"], "items": ["q1"]})
        assert bad["ok"] is False and "two questions" in bad["error"]
        # scoring needs a real session
        nos = post_json("/score?session_id=nope", {})
        assert nos["ok"] is False

        # naming: human names must reach the downloads, or the exports are unusable in a brief
        nm = post_json("/name", {"session_id": a2["session_id"],
                                 "names": ["Champions", "Sceptics"]})
        assert nm["ok"] and nm["names"] == ["Champions", "Sceptics"]
        assert "group_names.csv" in nm["downloads"]
        rows = get(f"/download?session_id={a2['session_id']}&file=segment_assignments.csv")
        assert "group_name" in rows and "Champions" in rows
        names_csv = get(f"/download?session_id={a2['session_id']}&file=group_names.csv")
        assert "segment,name,people" in names_csv and "Sceptics" in names_csv
        wrong = post_json("/name", {"session_id": a2["session_id"], "names": ["only one"]})
        assert wrong["ok"] is False and "each of the 2 groups" in wrong["error"]
        sug = post_json("/name", {"session_id": a2["session_id"], "suggest": True})
        assert sug["ok"] and sug["names"] == ["Loyal Fans", "Price Hunters"]   # from the fake AI
    finally:
        try:
            urllib.request.urlopen(base + "/quit", timeout=5).read()
        except Exception:
            pass


def test_scoring_a_person_does_not_depend_on_who_else_is_in_the_file():
    """A typing rule has to be a fixed rule. Skipped answers used to be filled in from the mean of
    whatever file they arrived in, so the SAME person got a different confidence — and could get a
    different segment — depending on which other people happened to be uploaded alongside them.
    The study's own centre is the only defensible fallback, and it makes each row self-contained."""
    rng = np.random.default_rng(11)
    n = 130
    hi = pd.DataFrame({"q1": rng.integers(6, 8, n), "q2": rng.integers(6, 8, n),
                       "q3": rng.integers(1, 3, n)})
    lo = pd.DataFrame({"q1": rng.integers(1, 3, n), "q2": rng.integers(1, 3, n),
                       "q3": rng.integers(6, 8, n)})
    df = pd.concat([hi, lo], ignore_index=True)
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))
    rule = json.loads(r["files"]["typing_rule.json"])

    person = {"respondent_id": "NEW1", "q1": 7, "q2": 7, "q3": np.nan}   # skipped one question
    alone = sk.classify_new(rule, pd.DataFrame([person]))
    company = [{"respondent_id": f"F{i}", "q1": 1, "q2": 1, "q3": 7} for i in range(40)]
    crowded = sk.classify_new(rule, pd.DataFrame([person] + company))

    assert alone.loc[0, "segment"] == crowded.loc[0, "segment"]
    assert alone.loc[0, "confidence"] == crowded.loc[0, "confidence"]

    # A column nobody answered must not blow up or silently become an extreme score.
    blank = pd.DataFrame([person, {"respondent_id": "NEW2", "q1": 6, "q2": 7, "q3": np.nan}])
    scored = sk.classify_new(rule, blank)
    assert len(scored) == 2 and scored["confidence"].between(0, 1).all()


def test_naming_regrouping_and_scoring_all_survive_a_restart(tmp_path, monkeypatch):
    """Everything the user does after the first analysis has to be saved, not just the analysis and
    the chat. Naming the groups, scoring new people and re-grouping all mutate the project; if they
    are not persisted, the work quietly disappears the next time the app opens. Re-grouping must
    also replace the WHOLE stored result — a reopened project showing the previous grouping's report
    beside the new group count is worse than not saving at all."""
    monkeypatch.setenv("SURVEY_SEGMENTER_PROJECTS", str(tmp_path / "projects"))
    store = webapp.ProjectStore()

    # Jittered rather than perfectly constant: identical answers within a group make the ANOVA
    # undefined and flood the run with warnings that have nothing to do with what is being tested.
    rng = np.random.default_rng(7)
    base = np.tile([5, 1], 40)
    df = pd.DataFrame({"respondent_id": [f"P{i}" for i in range(80)],
                       "q1": np.clip(base + rng.integers(-1, 2, 80), 1, 5),
                       "q2": np.clip(6 - base + rng.integers(-1, 2, 80), 1, 5),
                       "q3": np.clip(base + rng.integers(-1, 2, 80), 1, 5)})
    raw = df.to_csv(index=False).encode()
    r = sk.run_analysis(raw, cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))

    store.save("proj1", {"title": "wave 1", "digest": r["digest"], "files": r["files"],
                         "report_html": r["report_html"], "k": r["k"],
                         "n_people": r["n_people"], "names": ["Champions", "Sceptics"]}, raw=raw)
    back = store.load("proj1")
    assert back["names"] == ["Champions", "Sceptics"]     # names come back
    assert back["raw"] == raw                             # and so does the file, so re-grouping works

    # Re-grouping replaces the result wholesale rather than leaving a stale report behind.
    r2 = sk.run_analysis(raw, force_items=["q1", "q2"],
                         cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))
    store.save("proj1", {"title": "wave 1", "digest": r2["digest"], "files": r2["files"],
                         "report_html": r2["report_html"], "k": r2["k"],
                         "n_people": r2["n_people"], "names": []}, raw=raw)
    again = store.load("proj1")
    assert again["names"] == []                           # stale names dropped, not re-applied
    assert again["report_html"] == r2["report_html"]      # the report matches the new grouping
    assert again["raw"] == raw


# ------------------------------------------------------------------- charts: seeing the data
def _three_group_survey(n=240, seed=4):
    """A survey with three genuinely distinct mind-sets planted in it.

    Sized so k=3 wins outright. At n=150 the search legitimately prefers 4 and splits one real
    group in half, which is a correct Low-confidence result but a useless fixture for asserting
    what a GOOD run looks like."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        g = i % 3
        base = {0: [5, 1, 5, 2], 1: [1, 5, 2, 4], 2: [3, 3, 4, 5]}[g]
        rows.append([f"R{i:03d}"] + [int(np.clip(round(b + rng.normal(0, 0.7)), 1, 5))
                                     for b in base])
    return pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4"])


def _noise_survey(n=150, seed=11):
    """Answers drawn at random. There are no groups here; any the tool reports it invented."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"respondent_id": [f"R{i}" for i in range(n)],
                         **{f"q{j}": rng.integers(1, 6, n) for j in range(1, 6)}})


def test_every_run_produces_the_full_chart_set_as_valid_standalone_svg():
    """The charts are the user's own check on the result, so they are not optional decoration:
    a run that produces a report must produce the pictures of that report too."""
    r = sk.run_analysis(_three_group_survey().to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    assert [c["id"] for c in r["charts"]] == ["map", "gorge", "fit", "k",
                                              "profiles", "heatmap"]
    for c in r["charts"]:
        assert c["svg"].startswith("<svg") and c["svg"].endswith("</svg>")
        assert c["svg"].count("<svg") == c["svg"].count("</svg>") == 1
        assert c["title"] and c["caption"]
        # Chrome must be themeable: hard-coding a text colour breaks the chart in dark mode.
        assert "currentColor" in c["svg"]
    # And they reach the page the user actually looks at.
    doc = sk.charts_html(r["charts"])
    assert doc.count("<svg") == len(r["charts"])


def test_the_charts_say_out_loud_when_the_segments_are_not_real():
    """The whole point of showing the data: on structureless answers the tool still returns k
    groups, and the charts have to be the thing that contradicts the tidy-looking result rather
    than illustrating it."""
    cfg = sk.SegmentationConfig(k_min=2, k_max=4, **FAST)
    noise = sk.run_analysis(_noise_survey().to_csv(index=False).encode(), cfg=cfg)
    real = sk.run_analysis(_three_group_survey().to_csv(index=False).encode(), cfg=cfg)

    cap = {c["id"]: c["caption"] for c in noise["charts"]}
    assert "no number of groups reproduces strongly" in cap["k"]
    assert "not really there" in cap["fit"]          # low average silhouette, stated as such
    # The headline confidence has to agree with the charts. It used to read "Moderate — the groups
    # reproduce" on exactly this data, which is the wrong conclusion the charts exist to catch.
    assert noise["confidence"] == "low"

    good = {c["id"]: c["caption"] for c in real["charts"]}
    assert "real answer from the data" in good["k"]
    assert "not really there" not in good["fit"]
    assert real["confidence"] == "high"


def test_the_segment_map_reports_how_much_of_the_data_it_is_actually_showing():
    """A 2-D projection of a 10-question survey is a shadow, and how faithful a shadow it is
    decides whether overlap on screen means anything. Hiding that number would make the most
    persuasive chart in the app the least honest one."""
    r = sk.run_analysis(_three_group_survey().to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    caption = next(c for c in r["charts"] if c["id"] == "map")["caption"]
    assert re.search(r"carry \d+% of", caption)


def test_charts_never_draw_an_individual_respondent_by_name():
    """Every respondent is a dot on the segment map. Their id must not travel with the dot — the
    charts are the part of the report most likely to be pasted into a deck or shared onward.

    Embedded raster data is stripped before searching, and that is a correction rather than a
    loosening. Charts rasterise the dense artists — the scatter, the heatmap's mesh, the fill on a
    large study — and a base64 blob is arbitrary alphanumeric text, so a short id like "R4" turns
    up inside it by pure chance. This test used to search the raw SVG and passed on luck; adding a
    second rasterised artist was enough to produce 16 "hits", none of them real.

    Nothing is given up by stripping it. A leaked id could only arrive as a label, and matplotlib
    writes labels as real `<text>` elements (`svg.fonttype: none`) which survive the strip. No
    rasterised artist in this file draws text at all.
    """
    df = _three_group_survey()
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    markup = "".join(re.sub(r"data:image/[a-z]+;base64,[^\"']*", "", c["svg"])
                     for c in r["charts"])
    assert "<text" in markup, "stripping removed the labels too, so this test proves nothing"
    for rid in df["respondent_id"]:
        assert rid not in markup

    # And the strip must not be hiding a real leak: an id planted into a chart label has to be
    # caught. Question wording is the one respondent-adjacent string that legitimately reaches a
    # chart, so use it to prove the search still works.
    # No underscore: chart labels render underscores as spaces, so an underscored probe would
    # fail for a reason that has nothing to do with what is being tested.
    planted = df.rename(columns={"q1": "R7secret"})
    r2 = sk.run_analysis(planted.to_csv(index=False).encode(),
                         cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    visible = "".join(re.sub(r"data:image/[a-z]+;base64,[^\"']*", "", c["svg"])
                      for c in r2["charts"])
    assert "R7secret" in visible, "a string drawn as a label was not detectable after stripping"


def test_a_hostile_column_name_cannot_inject_markup_into_a_chart():
    """Question wording is drawn as chart labels, and it comes from whatever the survey tool
    exported. It has to be escaped on the way in, not trusted."""
    df = _three_group_survey()
    df = df.rename(columns={"q1": "<script>alert(1)</script>"})
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    everything = "".join(c["svg"] for c in r["charts"])
    assert "<script>" not in everything
    assert "&lt;script&gt;" in everything          # escaped, and still shown to the user


def test_categorical_surveys_are_charted_too():
    """Multiple-choice surveys go down the latent-class path, which has no centroids and no
    distances of its own. It gets the same four charts or the categorical half of the tool is
    the half nobody can check."""
    rng = np.random.default_rng(5)
    opts = {"channel": ["Instagram", "TikTok", "Email"], "why": ["Price", "Quality", "Brand"],
            "when": ["Morning", "Evening", "Weekend"]}
    rows = []
    for i in range(150):
        g = i % 3
        rows.append([f"R{i}"] + [o[g] if rng.random() < 0.8 else o[rng.integers(0, len(o))]
                                 for o in opts.values()])
    df = pd.DataFrame(rows, columns=["respondent_id", *opts])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, n_init_search=6,
                                                  n_init_final=8, stability_B=6,
                                                  run_consensus=False,
                                                  check_variable_selection=False))
    assert r["method"] == "lca"
    assert [c["id"] for c in r["charts"]] == ["map", "gorge", "fit", "k",
                                              "profiles", "heatmap"]
    prof = next(c for c in r["charts"] if c["id"] == "profiles")
    # Probabilities, not means — the caption must not tell the reader to read them as ratings.
    assert "How likely each answer is" in prof["caption"]
    assert "channel = TikTok" in prof["svg"] or "why = Price" in prof["svg"]


def test_a_chart_that_cannot_be_drawn_never_costs_the_user_the_analysis():
    """Charts are an aid, not the product. If one cannot be drawn the run must still hand over
    the report and the CSVs rather than failing in front of someone who just wants their groups."""
    class Broken:
        labels = np.array([0, 1, 0, 1])
        recommended_k = 2

        def __getattr__(self, name):               # every matrix access blows up
            raise RuntimeError("no matrix here")

    assert sk.build_charts(Broken(), "kmeans") == []


def test_the_ai_digest_contains_no_individual_respondent_data():
    """The privacy guarantee, as an executable check rather than a sentence in a README.

    Everything the Claude layer transmits is the `digest`. The claim made to users — and to
    anyone asking a GDPR question — is that it is aggregate only: no respondent identifiers, no
    individual answer rows, no free text. That claim is worth exactly as much as the test behind
    it, so this builds a dataset whose identifiers and free-text answers are unmistakable strings
    and fails if any of them survives into the payload.
    """
    rng = np.random.default_rng(19)
    n = 400
    ids = [f"RESPONDENT-UNIQUE-{i:04d}" for i in range(n)]
    # Free text is the highest-risk field: it is the one place someone writes their own name.
    comments = [f"my-secret-comment-{i:04d}" for i in range(n)]
    rows = []
    for i in range(n):
        base = [5, 1, 5, 2] if i % 2 else [1, 5, 2, 4]
        rows.append([ids[i], *[int(np.clip(round(b + rng.normal(0, 0.7)), 1, 5)) for b in base],
                     comments[i], ["Woman", "Man"][i % 2]])
    df = pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4",
                                     "open_feedback", "gender"])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))

    payload = r["digest"]
    leaked_ids = [x for x in ids if x in payload]
    leaked_text = [c for c in comments if c in payload]
    assert leaked_ids == [], f"{len(leaked_ids)} respondent identifiers reached the AI payload"
    assert leaked_text == [], f"{len(leaked_text)} free-text answers reached the AI payload"

    # And it is not empty-by-accident: the aggregate content the feature needs IS present.
    assert "Confidence" in payload and "Segment" in payload

    # Images are now attached too, which widens the guarantee: it is no longer enough that the
    # TEXT is aggregate, because a chart could carry a label. Build the actual request and check
    # every part of it — the text blocks for the strings, and the images for what they contain.
    import base64
    import json as _json

    import ai_interpret

    request = ai_interpret.build_messages(r["digest"], None, None, charts=r["charts"])
    blocks = request[0]["content"]
    images = [b for b in blocks if b["type"] == "image"]
    assert images, "the charts were not attached at all"

    # Nothing identifying in any text block, including the chart titles.
    as_text = _json.dumps([b for b in blocks if b["type"] == "text"])
    assert not [x for x in ids if x in as_text]
    assert not [c for c in comments if c in as_text]

    # And nothing identifying inside the PNGs. matplotlib writes the labels it was given as glyph
    # data, not as recoverable strings, but a PNG can also carry text chunks of metadata — so the
    # bytes are checked directly rather than assumed clean.
    for image in images:
        raw = base64.b64decode(image["source"]["data"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not actually a PNG"
        for identifier in ids[:20] + comments[:20]:
            assert identifier.encode() not in raw, f"{identifier} is embedded in a chart"

    # The charts the browser receives carry no raster copy: it draws the vector one, and shipping
    # both would add a quarter of a megabyte to every response for bytes the page never reads.
    for chart in webapp._charts_for_browser(r["charts"]):
        assert "png_b64" not in chart
        assert chart["svg"]

    # The digest is the whole of what gets transmitted, so pin that too: what ai.build_messages
    # puts on the wire is the digest and nothing else drawn from the raw file.
    # Serialise the whole message list rather than indexing into it: content is a list of blocks,
    # so a substring check against the raw object silently tests nothing, and this assertion is
    # the one standing between a respondent identifier and a third party.
    sent = json.dumps(ai.build_messages(payload, None, None))
    assert not [x for x in ids if x in sent]
    assert not [c for c in comments if c in sent]


def test_ai_falls_back_gracefully_when_the_best_request_is_unavailable(monkeypatch):
    """The chat asks for the best request shape first, then degrades rather than failing.

    The top rung uses the beta endpoint for server-side refusal fallbacks. Not every account is
    enabled for that beta and not every installed SDK knows the parameter — and in both cases the
    failure arrives as a plain 400 or a TypeError, which would otherwise surface to a marketing
    user as "Claude could not be reached". Each rung must therefore drop a capability and retry,
    so the feature works everywhere instead of only on the newest setup.
    """
    import anthropic as _real

    class _Blk:
        type, text = "text", "Target Segment 1 first."

    class _Msg:
        stop_reason, content = "end_turn", [_Blk()]

    class _Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self): return _Msg()

    def run(beta_failure):
        tried = []

        def endpoint(name):
            def stream(**kw):
                tried.append(name)
                if name == "beta":
                    raise beta_failure
                return _Stream()
            return type("E", (), {"stream": staticmethod(stream)})

        client = type("C", (), {"beta": type("B", (), {"messages": endpoint("beta")}),
                                "messages": endpoint("plain")})
        monkeypatch.setattr(_real, "Anthropic", lambda api_key=None: client)
        monkeypatch.setattr(ai, "have_sdk", lambda: True)
        text, _ = ai.chat_once("REPORT", None, None, api_key="sk-ant-test-not-real")
        return text, tried

    http_400 = _real.BadRequestError(
        "beta not enabled",
        response=type("R", (), {"status_code": 400, "headers": {}, "request": None})(),
        body=None)

    for failure in (http_400,                                  # account lacks the beta
                    TypeError("unexpected keyword 'fallbacks'"),  # older SDK, unknown kwarg
                    AttributeError("no attribute 'beta'")):       # older SDK, no beta namespace
        text, tried = run(failure)
        assert "Target Segment 1" in text, f"chat broke on {type(failure).__name__}"
        assert tried[0] == "beta" and "plain" in tried, tried


def test_one_broken_chart_does_not_take_the_others_down():
    """Charts are independent, so a failure in one must not withhold the rest.

    They were previously built as one eagerly-evaluated tuple inside a single try/except: any
    raise discarded all four, including the segment map, which is the whole point of the feature.
    A NaN centroid was enough to trigger it.
    """
    class _Seg:
        labels = np.array([0] * 30 + [1] * 30)
        recommended_k = 2
        X = np.random.default_rng(0).normal(size=(60, 4))
        # Three questions, not two: two would make the map a line rather than a cloud,
        # and this fixture has to exercise every chart for the isolation check to mean anything.
        centroids = pd.DataFrame({"q1": [1.0, 4.0], "q2": [4.0, 1.0], "q3": [2.0, 3.0]})
        diagnostics = pd.DataFrame({"k": [2, 3], "silhouette": [0.5, 0.3],
                                    "stability_ARI": [0.9, 0.6],
                                    "prediction_strength": [0.85, 0.5]})

    assert [c["id"] for c in sk.build_charts(_Seg(), "kmeans")] == \
        ["map", "fit", "k", "profiles", "heatmap"]

    # Patched on `charts`, not on `segment_kmeans`: the drawing moved into its own module, and
    # build_charts calls its neighbours directly. Replacing the re-exported alias would have
    # patched a name nothing calls, and the test would have passed while proving nothing.
    import charts

    original = charts.chart_segment_map
    charts.chart_segment_map = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        survivors = [c["id"] for c in sk.build_charts(_Seg(), "kmeans")]
    finally:
        charts.chart_segment_map = original
    assert survivors == ["fit", "k", "profiles", "heatmap"], survivors


def test_charts_survive_degenerate_data():
    """Real exports produce degenerate cases — an empty cluster leaves NaN centroids, and a
    one-group solution makes per-person fit undefined. Neither should raise: each chart returns
    None when it has nothing honest to draw, and the others still render."""
    X = np.random.default_rng(0).normal(size=(60, 4))

    # A single group: "how much better do you fit your own group than the next" has no answer.
    assert sk.chart_silhouette(X, np.zeros(60, dtype=int)) is None

    # NaN centroids: drop the affected question, chart the rest.
    partial = sk.chart_profiles(pd.DataFrame({"q1": [np.nan, 3.0], "q2": [1.0, 2.0]}))
    assert partial is not None and "q2" in partial["svg"]
    assert sk.chart_profiles(pd.DataFrame({"q1": [np.nan, np.nan]})) is None

    # A NaN span used to raise StopIteration out of the hand-written tick-spacing helper, far
    # from its cause. That helper is gone — matplotlib chooses ticks now — so the check is that
    # non-finite data still produces a chart or an honest None, never an exception.
    assert sk.chart_k_choice(pd.DataFrame({"k": [2, 3], "silhouette": [float("nan")] * 2}), 2)
    assert sk.chart_profiles(pd.DataFrame({"q1": [float("inf"), 1.0]})) is not None


def test_it_will_not_recommend_segments_too_small_to_target():
    """A segmentation exists to be acted on, so unusably small segments are not an answer.

    Separation indices happily crown very large k: on 120 respondents the criteria picked k=55 —
    segments of two people — and reported Moderate confidence. `min_segment_frac` already
    expressed the floor, but only printed a footnote *under* the finished report, so the headline
    number was chosen as if it did not exist. It is now applied before the vote.
    """
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"respondent_id": [f"R{i}" for i in range(n)],
                       "q1": rng.integers(1, 6, n), "q2": rng.integers(1, 6, n)})
    cfg = sk.SegmentationConfig(k_min=2, k_max=55, **FAST)
    r = sk.run_analysis(df.to_csv(index=False).encode(), cfg=cfg)

    assignments = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
    smallest = assignments["segment"].value_counts().min() / len(assignments)

    # The guard is applied to the search-time fit; the final fit uses more restarts and can land
    # on a slightly different local optimum, so this is a large reduction rather than a hard
    # bound — two-person segments are gone, a segment marginally under the floor can remain.
    assert r["k"] <= 20, f"k={r['k']} is still far more groups than 120 people can support"
    assert smallest > 0.03, f"smallest segment is {smallest:.1%} — fragment-sized"

    # The narrowing is stated, not silent — a reader can see which k values were taken off the
    # table and why, rather than wondering why an obvious peak was ignored. Either explanation
    # will do: the search range is now usually cut earlier, by the rule that a group needs enough
    # distinct answer patterns to be a type, so on this file nothing is left to rule out at the
    # vote. What matters is that the report says so somewhere.
    assert ("Ruled out before the vote" in r["digest"]
            or "The search stopped at" in r["digest"]), \
        "k values were dropped without the report explaining why"

    # And the residual case is still caught downstream: anything that slips under the floor in
    # the final fit is called out rather than passing silently. On this file nothing does any
    # more — requiring a handful of distinct answer patterns per group now stops the search long
    # before it reaches fragment-sized solutions, so the smallest segment is about 45% of the
    # sample. The property is what matters, not which of the two guards enforced it.
    smallest = assignments["segment"].value_counts().min() / len(assignments)
    assert smallest >= cfg.min_segment_frac or "below 5% of the sample" in r["digest"], (
        f"smallest segment is {smallest:.1%} and the report does not mention it")


def test_the_size_floor_does_not_distort_a_healthy_segmentation():
    """The floor must only remove unusable answers. On data with three real, well-populated
    groups it must not shift the recommendation — a guard that changes good results is worse
    than no guard."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(240):
        base = {0: [5, 1, 5, 2], 1: [1, 5, 2, 4], 2: [3, 3, 4, 5]}[i % 3]
        rows.append([f"R{i}"] + [int(np.clip(round(b + rng.normal(0, 0.7)), 1, 5)) for b in base])
    df = pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4"])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
    assert r["k"] == 3, r["k"]
    assert r["confidence"] == "high"
    assert "Ruled out before the vote" not in r["digest"]   # nothing needed excluding


def test_k_is_capped_by_the_number_of_distinct_answer_patterns():
    """A cluster needs a distinct point to sit on.

    Two 1-to-5 questions admit only 25 possible answer patterns, so a 120-person file cannot
    yield 55 groups however many the criteria vote for — k-means quietly returns duplicate or
    empty clusters and emits a ConvergenceWarning per fit. The old ceiling was n//2, which
    ignores distinctness entirely, so the search burned time fitting impossible solutions and
    then scored them as if they were real.
    """
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"respondent_id": [f"R{i}" for i in range(n)],
                       "q1": rng.integers(1, 6, n), "q2": rng.integers(1, 6, n)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = sk.run_analysis(df.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=55, **FAST))

    assert r["k"] <= 25, f"chose k={r['k']} with only 25 possible answer patterns"
    # The ceiling is estimated from a handful of half-splits, so an unlucky resample can still
    # come up short — the point is that this went from hundreds of impossible fits to at most a
    # couple, not that randomness was eliminated.
    impossible = [w for w in caught if "found smaller than n_clusters" in str(w.message)]
    assert len(impossible) <= 3, (
        f"{len(impossible)} fits asked for more clusters than the data can hold")


def test_actions_still_work_after_a_session_is_evicted_from_memory(tmp_path, monkeypatch):
    """Only the last few sessions stay in memory, but the buttons stay on screen.

    A user who analysed several files and then clicked Re-group on an earlier card was told to
    "analyse a survey file first". The work was safely on disk the whole time — the button was
    lying. Every session lookup now falls back to the store, so an action behaves the same
    whether or not the session happens to still be resident.
    """
    monkeypatch.setenv("SURVEY_SEGMENTER_PROJECTS", str(tmp_path / "projects"))
    store = webapp.ProjectStore()

    rng = np.random.default_rng(5)
    base = np.tile([5, 1], 40)
    df = pd.DataFrame({"respondent_id": [f"P{i}" for i in range(80)],
                       "q1": np.clip(base + rng.integers(-1, 2, 80), 1, 5),
                       "q2": np.clip(6 - base + rng.integers(-1, 2, 80), 1, 5),
                       "q3": np.clip(base + rng.integers(-1, 2, 80), 1, 5)})
    raw = df.to_csv(index=False).encode()
    r = sk.run_analysis(raw, cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))
    store.save("evicted", {"title": "wave 1", "digest": r["digest"], "files": r["files"],
                           "report_html": r["report_html"], "k": r["k"],
                           "n_people": r["n_people"], "charts": r["charts"],
                           "names": []}, raw=raw)

    # Simulate eviction: nothing in memory, everything on disk.
    recovered = store.load("evicted")
    assert recovered is not None
    assert recovered["raw"] == raw                  # re-grouping needs the original upload
    assert "typing_rule.json" in recovered["files"]  # scoring needs the rule
    assert recovered["charts"], "charts must come back too, or the card reopens empty"


def test_hopkins_is_caveated_when_it_cannot_be_trusted():
    """Hopkins reads "strong tendency to cluster" on short Likert surveys of pure noise.

    It compares distances between real points to distances from uniformly sampled ones, so it is
    inflated wherever real points coincide. Two 1-to-5 questions admit 25 answer patterns, so 120
    respondents pile onto duplicates and the statistic reports 0.78 on structureless data —
    telling the reader the opposite of the truth, in the one section devoted to "is there
    anything here at all". The number stays; the caveat sits next to it.
    """
    rng = np.random.default_rng(0)
    n = 120
    noise = pd.DataFrame({"respondent_id": [f"R{i}" for i in range(n)],
                          "q1": rng.integers(1, 6, n), "q2": rng.integers(1, 6, n)})
    r = sk.run_analysis(noise.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=8, **FAST))
    assert "Do not lean on the Hopkins number" in r["digest"]

    # It must stay quiet on data where the statistic IS meaningful — a caveat printed on every
    # run teaches people to skip it, which costs more than it saves.
    rows = []
    for i in range(240):
        base = {0: [5, 1, 5, 2, 4, 1], 1: [1, 5, 2, 4, 1, 5], 2: [3, 3, 4, 5, 2, 3]}[i % 3]
        rows.append([f"R{i}"] + [int(np.clip(round(b + rng.normal(0, 0.7)), 1, 5)) for b in base])
    wide = pd.DataFrame(rows, columns=["respondent_id"] + [f"q{i}" for i in range(1, 7)])
    r2 = sk.run_analysis(wide.to_csv(index=False).encode(),
                         cfg=sk.SegmentationConfig(k_min=2, k_max=5, **FAST))
    assert "Do not lean on the Hopkins number" not in r2["digest"]
    assert r2["confidence"] == "high"


# ------------------------------------------------------------------ MaxDiff / Hierarchical Bayes
def _simulate_maxdiff(n_resp=200, sep=1.4, n_items=15, set_size=5, n_sets=12, seed=1, n_seg=3,
                     scale_spread=0.0, non_attendance=0.0, worst_noise=0.0):
    """Simulate best-worst answers under the study's Block D design (15 items, 5 shown, 12 sets).

    By default, choices are generated from the same sequential logit the estimator assumes, with
    `n_seg` planted mind-sets, so recovery is a fair test of the sampler rather than of model fit.

    The three optional arguments break that assumption on purpose, in the ways real respondents
    actually depart from it:

    scale_spread     people differ in how consistently they choose; the model assumes one error
                     scale for everyone. A careless respondent is closer to random.
    non_attendance   the share of items a respondent ignores outright rather than weighing, so
                     their utilities are not a draw from the population distribution at all.
    worst_noise      "worst" decided partly on grounds the model does not represent; it assumes
                     worst is a reversed logit over whatever is left after best.
    """
    rng = np.random.default_rng(seed)
    centres = np.zeros((n_seg, n_items))
    block = n_items // n_seg
    for g in range(n_seg):
        centres[g, g * block:(g + 1) * block] = sep
        nxt = (g + 1) % n_seg
        centres[g, nxt * block:(nxt + 1) * block] = -sep
    who = rng.integers(0, n_seg, n_resp)
    true_b = centres[who] + rng.normal(0, 0.45, (n_resp, n_items))
    true_b -= true_b.mean(1, keepdims=True)

    # Lognormal keeps consistency positive; a spread of 0 is exactly the assumed model.
    scale = np.exp(rng.normal(0, scale_spread, n_resp)) if scale_spread else np.ones(n_resp)
    attended = np.ones((n_resp, n_items), bool)
    if non_attendance:
        for i in range(n_resp):
            attended[i] = rng.random(n_items) >= non_attendance

    design = np.zeros((n_resp, n_sets, set_size), int)
    best = np.zeros((n_resp, n_sets), int)
    worst = np.zeros((n_resp, n_sets), int)
    for i in range(n_resp):
        pool = np.repeat(np.arange(n_items), set_size * n_sets // n_items)
        rng.shuffle(pool)
        sets = pool.reshape(n_sets, set_size)
        for s in range(n_sets):
            if len(set(sets[s])) < set_size:
                sets[s] = rng.choice(n_items, set_size, replace=False)
            shown = sets[s]
            u = true_b[i, shown] * scale[i]
            u = np.where(attended[i, shown], u, 0.0)      # an ignored item reads as neutral
            p = np.exp(u - u.max()); p /= p.sum()
            bb = rng.choice(set_size, p=p)
            rem = [j for j in range(set_size) if j != bb]
            nu = -u[rem] + (rng.normal(0, worst_noise, len(rem)) if worst_noise else 0.0)
            q = np.exp(nu - nu.max()); q /= q.sum()
            best[i, s], worst[i, s] = bb, rem[rng.choice(len(rem), p=q)]
        design[i] = sets
    return design, best, worst, true_b, who


def _tidy_export(design, best, worst, prefix="use_case_"):
    """The simulated design as the tidy long table the reader documents."""
    rows = []
    for i in range(design.shape[0]):
        for s in range(design.shape[1]):
            for pos, item in enumerate(design[i, s]):
                if item < 0:
                    continue
                rows.append({"respondent_id": f"R{i:04d}", "set": s,
                             "item": f"{prefix}{item:02d}",
                             "choice": "best" if pos == best[i, s] else
                                       ("worst" if pos == worst[i, s] else "")})
    return pd.DataFrame(rows)


def test_hb_recovers_individual_utilities_it_was_never_shown():
    """The load-bearing claim: HB reconstructs each respondent's private preferences.

    Everything downstream — the segments, the personas, the build priority — rests on these
    numbers being a fair estimate of what each person actually wanted. Simulated answers are
    generated from known utilities and the estimator must find them back.
    """
    md = pytest.importorskip("maxdiff")
    design, best, worst, true_b, _ = _simulate_maxdiff(n_resp=200)
    res = md.estimate_hb(design, best, worst, [f"i{j}" for j in range(15)],
                         [f"R{j}" for j in range(200)],
                         n_draws=2000, n_burn=700, progress=False)

    per = np.array([np.corrcoef(true_b[i], res.utilities[i])[0, 1] for i in range(200)])
    assert per.mean() > 0.80, f"individual recovery only r={per.mean():.3f}"
    assert per.min() > 0.20, f"worst respondent recovered at r={per.min():.3f}"
    # Utilities are identified only up to a constant, so each respondent must sum to zero.
    assert np.allclose(res.utilities.sum(axis=1), 0, atol=1e-8)
    # A chain that accepts nearly everything or nearly nothing has not explored the posterior.
    assert 0.10 < res.acceptance_rate < 0.70, res.acceptance_rate


def test_hb_still_beats_counting_when_its_assumptions_are_wrong():
    """The honest stress test, standing in for real respondents until there are some.

    Every other HB test generates choices from the very model the estimator assumes, which is the
    right way to test a sampler and a flattering way to test a method: of course it wins when the
    world is exactly what it believes. Real people depart from it in three documented ways, so
    the data here is generated with each of them switched on.

    Swept at 200 respondents and 3000 draws, the advantage held in every scenario: +0.147 under
    the model's own assumptions, +0.140 with careless respondents, +0.189 when three items in ten
    are ignored, +0.157 when "worst" is decided on other grounds, and +0.151 with all three at
    once (counting 0.548, HB 0.698). It was widest, not narrowest, under the worst violation.
    That is the result worth having: the advantage is not an artefact of grading the model on its
    own homework.

    This test runs the least favourable case at 120 respondents and 1500 draws to stay quick,
    where the same comparison gives counting 0.546 against HB 0.680 — a margin of +0.133, so the
    +0.05 threshold below has real headroom rather than being fitted to the observed number.

    It does NOT show HB is accurate on real data. Nothing here can. It shows the advantage does
    not evaporate the moment the assumptions do.
    """
    def counting(design, best, worst, n_items):
        """The industry default: times picked best minus times picked worst."""
        out = np.zeros((len(design), n_items))
        for i in range(len(design)):
            for s in range(design.shape[1]):
                out[i, design[i, s, best[i, s]]] += 1
                out[i, design[i, s, worst[i, s]]] -= 1
        return out

    def recovery(estimated, truth):
        rs = [np.corrcoef(estimated[i], truth[i])[0, 1] for i in range(len(truth))
              if np.std(estimated[i]) > 1e-9 and np.std(truth[i]) > 1e-9]
        return float(np.mean(rs))

    # All three violations at once — the least favourable case for the estimator.
    design, best, worst, true_b, _ = _simulate_maxdiff(
        n_resp=120, seed=7, scale_spread=0.9, non_attendance=0.30, worst_noise=1.2)
    n_items = int(design.max()) + 1

    import maxdiff as md

    hb = md.estimate_hb(design, best, worst,
                              [f"item_{j}" for j in range(n_items)],
                              [f"r{i}" for i in range(len(design))],
                              n_draws=1500, n_burn=600, seed=3, progress=False).utilities
    counted = counting(design, best, worst, n_items)

    r_hb, r_count = recovery(np.asarray(hb), true_b), recovery(counted, true_b)
    assert r_hb > r_count + 0.05, (
        f"HB no longer justifies its cost under misspecification: "
        f"counting {r_count:.3f} vs HB {r_hb:.3f}")
    # And it has not collapsed to noise: it still describes individuals usefully.
    assert r_hb > 0.55, f"individual recovery fell to {r_hb:.3f}"


def test_hb_beats_counting_at_describing_an_individual():
    """Why this module exists at all.

    The instrument specifies HB and rejects best-minus-worst counting as "too coarse for
    individual-level segmentation". That is a testable claim, not a matter of taste — and it is
    the entire justification for carrying an MCMC sampler in a marketing tool.
    """
    md = pytest.importorskip("maxdiff")
    design, best, worst, true_b, _ = _simulate_maxdiff(n_resp=200)

    counts = np.zeros((200, 15))
    for i in range(200):
        for s in range(design.shape[1]):
            counts[i, design[i, s, best[i, s]]] += 1
            counts[i, design[i, s, worst[i, s]]] -= 1

    hb = md.estimate_hb(design, best, worst, [f"i{j}" for j in range(15)],
                        [f"R{j}" for j in range(200)],
                        n_draws=2000, n_burn=700, progress=False).utilities

    r_counts = np.mean([np.corrcoef(true_b[i], counts[i])[0, 1] for i in range(200)])
    r_hb = np.mean([np.corrcoef(true_b[i], hb[i])[0, 1] for i in range(200)])
    assert r_hb > r_counts + 0.05, f"HB {r_hb:.3f} vs counting {r_counts:.3f} — no real gain"


def test_a_raw_maxdiff_export_segments_correctly_end_to_end():
    """A best-worst export is not a rating grid, and clustering its raw choice codes would
    produce a confident-looking result from nonsense. Dropping the export straight into the tool
    must score it first and then recover the mind-sets that were planted in it."""
    pytest.importorskip("maxdiff")
    design, best, worst, _true_b, who = _simulate_maxdiff(n_resp=200, sep=1.4)
    raw = _tidy_export(design, best, worst).to_csv(index=False).encode()

    r = sk.run_analysis(raw, cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))

    assign = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
    order = {f"R{i:04d}": i for i in range(200)}
    labels = np.zeros(200, dtype=int)
    for _, row in assign.iterrows():
        labels[order[str(row["id"])]] = int(row["segment"])

    from sklearn.metrics import adjusted_rand_score as _ari
    assert _ari(who, labels) > 0.85, f"only recovered the planted mind-sets at ARI {_ari(who, labels):.2f}"
    # The reader must say what it did — a silent transformation of the input is not acceptable
    # in a report someone will present.
    assert "Hierarchical Bayes" in r["digest"]


def test_maxdiff_detection_does_not_fire_on_a_rating_grid():
    """A false positive here would score an ordinary survey as best-worst data and produce
    nonsense, so detection has to be specific, not merely sensitive."""
    md = pytest.importorskip("maxdiff")
    grid = pd.DataFrame({"respondent_id": ["R1", "R2"], "q1": [4, 2], "q2": [1, 5],
                         "item": ["a", "b"]})          # has 'item' but is not best-worst
    assert not md.looks_like_maxdiff(grid)
    assert md.looks_like_maxdiff(
        pd.DataFrame({"respondent_id": [], "set": [], "item": [], "choice": []}))


def test_maxdiff_reader_drops_incomplete_sets_rather_than_inventing_choices():
    """A set with no 'worst' recorded carries no worst-choice information. Guessing one would
    fabricate preference data, so the set is dropped and the loss reported."""
    md = pytest.importorskip("maxdiff")
    design, best, worst, _, _ = _simulate_maxdiff(n_resp=30, n_sets=12)
    df = _tidy_export(design, best, worst)
    victim = (df["respondent_id"] == "R0000") & (df["set"] == 0) & (df["choice"] == "worst")
    df.loc[victim, "choice"] = ""

    d2, b2, w2, items, respondents = md.read_maxdiff(df)
    assert len(items) == 15 and len(respondents) == 30
    kept_for_r0 = int((d2[respondents.index("R0000")] >= 0).any(axis=1).sum())
    assert kept_for_r0 == 11, f"expected the damaged set to be dropped, kept {kept_for_r0}"


def test_the_heatmap_shows_every_question_where_the_bars_cannot():
    """The bar chart stops at nine questions to stay legible, which on a fifteen-item MaxDiff
    block hides a third of the study. That trim is the heatmap's whole reason for existing, so
    the grid must be complete — and the bars must say where the rest went."""
    rng = np.random.default_rng(0)
    centroids = pd.DataFrame(rng.normal(0, 1, (4, 15)),
                             columns=[f"use_case_{i:02d}" for i in range(15)],
                             index=[f"Segment {i}" for i in range(4)])

    def shown(chart):
        return sum(1 for i in range(15) if f"use case {i:02d}" in chart["svg"])

    bars, heat = sk.chart_profiles(centroids), sk.chart_heatmap(centroids)
    assert shown(bars) == 9, shown(bars)
    assert shown(heat) == 15, f"the full grid hid {15 - shown(heat)} questions"
    assert "Full grid" in bars["caption"], "the bars must point at where the rest are"


def test_profile_charts_refuse_shapes_they_cannot_draw_honestly():
    """Returning None is the honest answer to data a chart cannot describe, and build_charts
    drops the chart rather than showing an empty frame.

    This used to cover the radar chart, which has been removed rather than repaired. A radar
    encodes value as distance from a centre, so the eye reads the enclosed AREA — which grows with
    the square of the values and changes completely when the questions are reordered, an order
    that carries no meaning. Three of its six labels also truncated and one overlapped the plot.
    chart_profiles now answers the same question as a distance along a shared axis, which is the
    comparison people read accurately.
    """
    c = pd.DataFrame({"a": [1.0, 4.0], "b": [4.0, 1.0], "c": [2.0, 3.0]},
                     index=["Segment 0", "Segment 1"])
    assert not hasattr(sk, "chart_radar") and not hasattr(charts, "chart_radar")
    assert sk.chart_heatmap(pd.DataFrame()) is None
    assert sk.chart_profiles(pd.DataFrame()) is None
    # A single group still has a readable profile row, so both keep working.
    assert sk.chart_heatmap(c.iloc[:1]) is not None
    assert sk.chart_profiles(c.iloc[:1]) is not None

    for chart in (sk.chart_profiles(c), sk.chart_heatmap(c)):
        assert chart["svg"].startswith("<svg") and chart["svg"].endswith("</svg>")
        # Titles are escaped by the UI, so an HTML entity here renders as literal "&mdash;".
        assert "&" not in chart["title"], chart["title"]


def test_every_chart_is_offered_on_a_normal_run():
    """All six charts should appear together — each answers a different question, and a silently
    missing one looks like the tool decided the reader did not need it."""
    r = sk.run_analysis(_three_group_survey().to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    # The order is the reading order: whether these groups are real (map, gorge, fit, k) before
    # what is in them (profiles, heatmap). It is also the tab order in the app.
    assert [c["id"] for c in r["charts"]] == ["map", "gorge", "fit", "k", "profiles", "heatmap"]
    assert sk.charts_html(r["charts"]).count("<svg") == 6


# =====================================================================================
# Mixed-type surveys: ratings and pick-any questions in one model (kprototypes.py)
# =====================================================================================
def _mixed_survey(n=300, seed=0, brand_signal=True, n_brands=1):
    """A questionnaire of the shape most real ones have: some 1-5 scales, some "pick one"."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, n)
    centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)
    df = pd.DataFrame(_likert(centres[truth] + rng.normal(0, 0.7, (n, 4))),
                      columns=[f"q{i+1}" for i in range(4)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(n)])
    names = np.array(["Nespresso", "Lavazza", "Illy"])
    for j in range(n_brands):
        picks = [names[rng.choice(3, p=(np.where(np.arange(3) == (g + j) % 3, 0.70, 0.15)
                                        if brand_signal else np.full(3, 1 / 3)))]
                 for g in truth]
        df["favourite brand" if n_brands == 1 else f"brand{j+1}"] = picks
    return df, truth


def test_gower_is_exactly_a_manhattan_distance_in_disguise():
    """The identity the whole mixed-type path is built on.

    Divide each rating by its range and replace each pick-any answer with half a one-hot
    indicator, and Manhattan distance on those coordinates reproduces Gower's distance exactly.
    That is what lets the silhouette, the cluster-tendency test, the hierarchical cross-check and
    the segment map run through the same library functions as the numeric path with nothing but a
    `metric="manhattan"` argument, instead of a second family of hand-written Gower versions.

    If this ever stops holding, those four stop measuring what they claim to, silently — hence
    testing the identity itself rather than any one of its consumers.
    """
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack([rng.integers(1, 6, n), rng.integers(1, 6, n), rng.normal(50, 10, n),
                         rng.integers(0, 4, n), rng.integers(0, 3, n)]).astype(float)
    kinds = [kp.ORDINAL, kp.ORDINAL, kp.NUMERIC, kp.NOMINAL, kp.NOMINAL]
    spec = kp.fit_spec(X, kinds)
    Xe = kp.encode(X, spec)

    direct = kp.gower_distances(Xe, Xe, spec)
    embedded = cdist(kp.gower_embedding(Xe, spec), kp.gower_embedding(Xe, spec),
                     metric="cityblock") / spec.n_vars
    assert np.abs(direct - embedded).max() < 1e-12
    assert np.allclose(np.diag(direct), 0) and direct.max() <= 1.0
    # A metric, unlike Podani's tie-corrected version — see the module docstring for why that one
    # was rejected. "Nearest prototype" is only meaningful if the triangle inequality holds.
    assert (direct[:, :, None] <= direct[:, None, :] + direct[None, :, :] + 1e-9).all()

    # The embedding must take its one-hot columns from the spec, not from the array in front of
    # it. A bootstrap resample or a reference sample can easily be missing a brand, and deriving
    # the levels locally would shift every coordinate after it and compare two different spaces.
    degenerate = Xe[:5].copy()
    degenerate[:, 3] = degenerate[0, 3]
    assert kp.gower_embedding(degenerate, spec).shape[1] == kp.gower_embedding(Xe, spec).shape[1]
    assert np.abs(kp.gower_distances(degenerate, Xe, spec)
                  - cdist(kp.gower_embedding(degenerate, spec), kp.gower_embedding(Xe, spec),
                          metric="cityblock") / spec.n_vars).max() < 1e-12


def test_kprototypes_updates_are_the_ones_that_make_it_converge():
    """Median, mode and nearest-rank level — Szepannek et al. (2024), not a port of k-means.

    Gower is an L1-type distance, so the value minimising the within-cluster total is the median,
    not the mean; for a pick-any answer it is the mode; for an ordered answer it is the level
    whose rank is closest to the cluster's median rank. Using means, which is what porting k-means
    naively would do, is what breaks the convergence proof.
    """
    X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [100.0, 0.0]])
    spec = kp.fit_spec(X, [kp.NUMERIC, kp.NOMINAL])
    proto = kp._update(X, spec)
    assert proto[0] == 1.0, "an outlier moved the prototype, so this is a mean not a median"
    assert proto[1] == 0.0, "the pick-any prototype is not the most common answer"

    # An ordered answer's prototype has to be a level somebody could have chosen.
    ord_X = np.array([[1.0], [2.0], [2.0], [5.0]])
    ord_spec = kp.fit_spec(ord_X, [kp.ORDINAL])
    assert kp._update(kp.encode(ord_X, ord_spec), ord_spec)[0] in ord_spec.ord_levels[0]

    # The objective must never rise. That is the property the update rules buy, and the only
    # direct evidence that this is the 2024 algorithm rather than something that resembles it.
    rng = np.random.default_rng(0)
    data = np.column_stack([rng.integers(1, 6, 200), rng.integers(1, 6, 200),
                            rng.integers(0, 3, 200)]).astype(float)
    s = kp.fit_spec(data, [kp.ORDINAL, kp.ORDINAL, kp.NOMINAL])
    Xe = kp.encode(data, s)
    protos = kp._seed(Xe, 3, s, np.random.default_rng(1))
    last = np.inf
    for _ in range(15):
        D = kp.gower_distances(Xe, protos, s)
        labels = D.argmin(1)
        total = float(D[np.arange(len(Xe)), labels].sum())
        assert total <= last + 1e-9, f"objective rose from {last} to {total}"
        last = total
        protos = np.vstack([kp._update(Xe[labels == c], s) for c in range(3)])

    model = kp.KPrototypes(3, s, n_init=3, random_state=0).fit(Xe)
    assert set(np.unique(model.labels_)) == {0, 1, 2}, "a segment was allowed to empty"
    assert np.array_equal(model.predict(Xe), model.labels_)


def test_a_survey_with_ratings_and_pick_any_questions_uses_both():
    """The capability gap this closes.

    Before this, a questionnaire with both kinds of question had the multiple-choice columns set
    aside with an apology, and was segmented on the ratings alone. On a study where the brand
    question is the interesting one that threw away the finding.

    Measured on this machine, three planted segments over four ratings and three brand questions:
    k-prototypes recovers them at ARI 0.96, ratings-only k-means at 0.99 when the ratings carry
    the signal, and 0.50 against 0.00 when only the brand questions do.
    """
    df, truth = _mixed_survey(brand_signal=True, n_brands=3)
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=5, **FAST))
    assert r["method"] == "kproto"
    assert r["k"] == 3 and r["confidence"] == "high"
    assert not r["chart_errors"]
    assigned = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
    assert adjusted_rand_score(truth, assigned["segment"]) > 0.75
    assert assigned["fit"].isna().sum() == 0 and assigned["fit"].between(0, 1).all()

    d = r["digest"]
    # The pick-any questions must be described as answers, not as arithmetic. "Values 0.6 below
    # average" about a brand is a statement about the order the brands were listed in.
    assert "Mostly picks" in d and "Nespresso" in d
    assert re.search(r"favourite brand|brand1", d)
    assert "Gower k-prototypes" in d
    # The panel is smaller here and the report has to say so rather than quietly showing fewer
    # corroborating numbers than the numeric path does.
    assert "Gaussian-mixture cross-check is **not run**" in d
    assert "Range standardization" not in d


def test_pick_any_answers_are_scored_by_association_not_by_their_codes():
    """Eta-squared on brand codes measures the coding, not the data.

    Renumber the brands and it changes. Cramer's V does not, which is the only honest property to
    want. It is reported SQUARED: V is correlation-like and eta-squared is variance-like, so they
    sit one square apart. Measured on matched pure noise, a random pick-any column scores V = 0.06
    — already over the 0.05 near-noise floor, so a useless question could never be flagged as one
    — against V-squared = 0.00 and eta-squared = 0.00 for a random rating.
    """
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1, 2], 100)
    perfect = labels.astype(float)
    assert sk._cramers_v(perfect, labels) == pytest.approx(1.0, abs=1e-9)
    noise = rng.integers(0, 3, 300).astype(float)
    assert sk._cramers_v(noise, labels) ** 2 < 0.05, "a random question would never be flagged"
    # Invariant to how the levels are numbered, which is the entire reason for using it.
    shuffled = np.select([perfect == 0, perfect == 1, perfect == 2], [2.0, 0.0, 1.0])
    assert sk._cramers_v(shuffled, labels) == pytest.approx(sk._cramers_v(perfect, labels))
    assert sk._cramers_v(np.zeros(300), labels) == 0.0     # one level: no association possible

    # End to end: brand questions that separate nobody must reach the near-noise verdict, because
    # that check is what protects this path from its own weakness. Measured: three useless brand
    # columns beside four real ratings cost 0.25 ARI, and dropping them lifts the silhouette from
    # 0.25 to 0.52.
    df, _ = _mixed_survey(brand_signal=False, n_brands=3)
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=5,
                                                  **{**FAST, "check_variable_selection": True}))
    assert "Near-noise items" in r["digest"], "useless pick-any questions were not flagged"
    for brand in ("brand1", "brand2", "brand3"):
        assert brand in r["digest"].split("Near-noise items")[1][:200]
    # And it must not claim to have found something. Measured: it settles on k=2 and says so.
    assert r["confidence"] == "low"


def test_new_people_can_be_scored_on_a_mixed_survey():
    """The exported rule has to survive JSON and a brand nobody in the study ever named.

    The typing rule is the operational payoff — you segment once and type everyone afterwards —
    so it has to carry the Gower spec, the code-to-answer mapping, and a distance that matches the
    one the segmentation used. A pickle would have been easier and would have tied the file to
    this version of this class, which is the opposite of what an exported rule is for.
    """
    df, _ = _mixed_survey(brand_signal=True)
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=5, **FAST))
    rule = json.loads(r["files"]["typing_rule.json"])
    assert rule["method"] == "kproto"
    assert rule["scale_params"]["scaling"] == "gower"
    assert rule["level_labels"]["favourite brand"]        # codes mean nothing without these

    original = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))["segment"]
    fresh = df.drop(columns=["respondent_id"]).head(25).copy()
    scored = sk.classify_new(rule, fresh)
    agree = (scored["segment"].to_numpy() == original.head(25).to_numpy()).mean()
    assert agree >= 0.9, f"the exported rule disagrees with the analysis on {1 - agree:.0%} of rows"

    # Somebody names a brand the study never saw. Refusing to score them would be worse than
    # placing them; it must sit a full mismatch from every known answer rather than half of one,
    # and the rest of their answers must still count.
    newcomer = fresh.head(1).copy()
    newcomer.loc[newcomer.index[0], "favourite brand"] = "A Brand Nobody Named"
    out = sk.classify_new(rule, newcomer)
    assert len(out) == 1 and out["segment"].iloc[0] in set(original)
    assert 0 < out["confidence"].iloc[0] <= 1

    # A skipped answer must fall back to the study's own typical answer, not to whoever else
    # happens to be in the upload — otherwise the same person is typed differently in a batch of
    # one than in a batch of five hundred.
    gaps = fresh.head(1).copy()
    gaps.loc[gaps.index[0], "q1"] = np.nan
    alone = sk.classify_new(rule, gaps)
    together = sk.classify_new(rule, pd.concat([gaps, fresh.tail(20)], ignore_index=True)).head(1)
    assert alone["segment"].iloc[0] == together["segment"].iloc[0]
    assert alone["confidence"].iloc[0] == together["confidence"].iloc[0]


def test_the_web_app_handles_a_mixed_survey_end_to_end(monkeypatch):
    """Upload, re-group on a hand-picked mix, then score newcomers — through the real server.

    The mixed-question path added a new method, a new distance and a new typing rule. Everything
    the team touches goes through these three endpoints, and none of them knew about kproto
    before. A unit test on run_analysis would not have caught a rule that fails to serialise, or a
    re-group whose hand-picked columns route somewhere new.
    """
    import socket
    import threading
    import time
    import urllib.request
    import json as _json

    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=sk.serve, kwargs={"port": port}, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post_json(path, obj):
        req = urllib.request.Request(base + path, data=_json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return _json.loads(urllib.request.urlopen(req, timeout=60).read().decode())

    def post_file(path, name, blob, field="file"):
        b = "----t"
        body = (f'--{b}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
                f'Content-Type: text/csv\r\n\r\n').encode() + blob + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(base + path, data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        return _json.loads(urllib.request.urlopen(req, timeout=120).read().decode())

    for _ in range(80):
        try:
            urllib.request.urlopen(base + "/", timeout=5).read()
            break
        except Exception:
            time.sleep(0.1)

    df, _ = _mixed_survey(n=160, brand_signal=True)
    blob = df.to_csv(index=False).encode()
    try:
        a = post_file("/analyze", "mixed.csv", blob)
        assert a["ok"], a
        # The brand question must be reported as grouped on, not set aside — the whole point.
        assert a["columns"]["favourite brand"] == "used", a["columns"]
        assert "typing_rule.json" in a["downloads"]
        assert a["charts"], "a mixed survey produced no charts"

        # Re-grouping on a hand-picked mix has to reach the mixed path too, not fall back.
        picked = post_json("/regroup", {"session_id": a["session_id"],
                                        "items": ["q1", "q2", "q3", "favourite brand"]})
        assert picked["ok"], picked
        assert picked["columns"]["favourite brand"] == "used"

        # Scoring newcomers against a Gower rule: the endpoint picks the classifier off the saved
        # rule, so a rule it does not recognise would silently be scored with the wrong distance.
        fresh = df.drop(columns=["respondent_id"]).head(10)
        scored = post_file(f"/score?session_id={picked['session_id']}", "new.csv",
                           fresh.to_csv(index=False).encode())
        assert scored["ok"], scored
        assert scored["n"] == 10, scored
    finally:
        with contextlib.suppress(Exception):
            urllib.request.urlopen(base + "/quit", timeout=5).read()


def test_swedish_survey_keeps_its_ordering_and_uses_its_pick_any_answers():
    """A single missing phrase fails the whole column, not just that cell.

    `_try_likert` requires every answer to map, so one unrecognised wording sends the entire
    survey down the categorical path and throws away the ordering it was measuring — the tool
    stops knowing that "Instämmer helt" is more than "Instämmer delvis". Found by sweeping
    realistic files: "Instämmer delvis inte" and "Håller delvis inte med" are the standard second
    step on Swedish five-point scales and neither was listed — which matters for any study fielded
    in Swedish, where that wording is the norm rather than the exception.
    """
    five = {
        "instämmer, partly-not wording": ["Instämmer inte alls", "Instämmer delvis inte",
                                          "Varken eller", "Instämmer delvis", "Instämmer helt"],
        "instämmer, plain wording": ["Instämmer inte alls", "Instämmer inte", "Varken eller",
                                     "Instämmer delvis", "Instämmer helt"],
        "håller med": ["Håller inte alls med", "Håller delvis inte med", "Varken eller",
                       "Håller delvis med", "Håller helt med"],
        "nöjd, ganska wording": ["Mycket missnöjd", "Ganska missnöjd",
                                 "Varken nöjd eller missnöjd", "Ganska nöjd", "Mycket nöjd"],
        "aldrig / alltid": ["Aldrig", "Sällan", "Ibland", "Ofta", "Alltid"],
        "English, unchanged": ["Strongly disagree", "Disagree", "Neutral", "Agree",
                               "Strongly agree"],
    }
    for name, scale in five.items():
        recoded = sk._try_likert(pd.Series(scale * 8))
        assert recoded is not None, f"{name} was not recognised as a rating scale"
        first = [float(recoded[pd.Series(scale * 8).eq(v).idxmax()]) for v in scale]
        assert first == sorted(first), f"{name} lost its ordering"

    # No token may mean two different numbers across the twelve scales, or which scale matches
    # first would decide the answer.
    seen = {}
    for scale in sk._LIKERT_SCALES:
        for token, value in scale.items():
            assert seen.get(token, value) == value, f"'{token}' means two different numbers"
            seen[token] = value

    # End to end, in the shape a Swedish export actually arrives in: worded scales, a Swedish id
    # column, and a Swedish brand question.
    rng = np.random.default_rng(0)
    n = 240
    truth = rng.integers(0, 3, n)
    centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)
    scale = five["instämmer, partly-not wording"]
    ratings = _likert(centres[truth] + rng.normal(0, 0.6, (n, 4)))
    df = pd.DataFrame({f"fråga {i+1}": [scale[v - 1] for v in ratings[:, i]] for i in range(4)})
    df.insert(0, "svarsnummer", [f"P{i}" for i in range(n)])
    df["favoritmärke"] = [["Löfbergs", "Zoégas", "Gevalia"][
        rng.choice(3, p=np.where(np.arange(3) == g, 0.7, 0.15))] for g in truth]

    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=5, **FAST))
    assert r["method"] == "kproto", "a worded Swedish scale fell back to the categorical path"
    assigned = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
    assert adjusted_rand_score(truth, assigned["segment"]) > 0.75
    assert any(b in r["digest"] for b in ("Löfbergs", "Zoégas", "Gevalia"))


def test_asking_for_more_groups_than_the_answers_can_support_settles_immediately():
    """The empty-cluster refill has to know when to give up.

    Restarting an emptied cluster on the worst-served respondent is right when somebody is
    genuinely unexplained. When every respondent already sits exactly on a prototype there is
    nobody to build a group around, and moving one anyway just hands them back on the next pass:
    the refill fights the assignment step, the labels never settle, and the loop runs to max_iter
    on any file holding fewer distinct answer patterns than groups asked for. Measured before the
    fix: 50 iterations for k = 3 and k = 5. After: 2.
    """
    X = np.array([[1.0, 0.0]] * 20 + [[5.0, 1.0]] * 20)      # exactly two distinct patterns
    spec = kp.fit_spec(X, [kp.ORDINAL, kp.NOMINAL])
    Xe = kp.encode(X, spec)
    for k in (2, 3, 5):
        model = kp.KPrototypes(k, spec, n_init=3, max_iter=50, random_state=0).fit(Xe)
        assert model.n_iter_ <= 5, f"k={k} did not settle ({model.n_iter_} iterations)"
        assert model.inertia_ == pytest.approx(0.0), "two exact patterns should cost nothing"
        assert sorted(np.bincount(model.labels_, minlength=k))[-2:] == [20, 20]

    # And the give-up branch must not fire on ordinary data, where the refill is doing real work.
    rng = np.random.default_rng(0)
    n = 400
    truth = rng.integers(0, 3, n)
    centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)
    data = np.hstack([_likert(centres[truth] + rng.normal(0, 0.7, (n, 4))),
                      np.array([[rng.choice(3, p=np.where(np.arange(3) == g, 0.7, 0.15))]
                                for g in truth], float)])
    s = kp.fit_spec(data, [kp.ORDINAL] * 4 + [kp.NOMINAL])
    model = kp.KPrototypes(3, s, n_init=8, random_state=0).fit(kp.encode(data, s))
    assert adjusted_rand_score(truth, model.labels_) > 0.75
    assert (np.bincount(model.labels_, minlength=3) > 0).all(), "a real segment was left empty"


def test_the_segment_map_shows_every_respondent_and_says_how_many_share_a_spot():
    """The chart that exists to let somebody check the answer must not be showing a sample.

    Two things used to stop it being the whole study, and both were invisible to the reader:

    1.  It drew a random 1,200 respondents.
    2.  Rating answers come in whole steps, so people who answered identically land on exactly
        the same coordinates and hide under one another. Measured: 3,000 people answering five
        1-5 questions occupy 422 distinct positions, so a plain scatter showed 14% of the data.

    Fixed by drawing one dot per distinct answer pattern with its area proportional to how many
    people share it. No jitter, deliberately — Wilke's *Fundamentals of Data Visualization* warns
    that jittering too much places points "in locations that are not representative of the
    underlying dataset", and this chart cannot invent coordinates to make itself readable.
    """
    rng = np.random.default_rng(0)
    n = 3000
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    truth = rng.integers(0, 3, n)
    X = _likert(centres[truth] + rng.normal(0, 0.8, (n, 5))).astype(float)
    lo, hi = X.min(0), X.max(0)
    Xs = (X - lo) / np.where(hi > lo, hi - lo, 1)
    labels = sk._km(Xs, 3, 10, 0).labels_

    # The premise: this data really does hide most of itself under a plain scatter.
    coords, _, _ = charts.pca_2d(Xs)
    distinct = len(np.unique(np.round(coords, 9), axis=0))
    assert distinct < n / 2, "pick data where respondents actually collide, or this proves nothing"

    chart = charts.chart_segment_map(Xs, labels)
    caption = chart["caption"]
    assert f"All {n:,} respondents are shown" in caption
    assert f"{distinct:,} distinct positions" in caption
    assert "Nothing is sampled" in caption
    assert "random" not in caption, "the map is describing a sample again"

    # Every respondent must be accounted for by exactly one dot.
    spots, counts = np.unique(np.round(coords, 9), axis=0, return_counts=True)
    assert counts.sum() == n and len(spots) == distinct

    # The size key has to be there, or a big dot reads as "important" rather than "many".
    assert "1 person" in chart["svg"] and f"{int(counts.max()):,} people" in chart["svg"]
    # And the share of variation belongs on the axes, not only in the prose.
    assert re.search(r"Direction 1 — \d+% of the variation", chart["svg"])

    # Continuous inputs (MaxDiff utilities) have no collisions; the chart must degrade to a
    # plain scatter rather than claiming stacking that is not there.
    smooth = rng.normal(0, 1, (400, 4))
    plain = charts.chart_segment_map(smooth, sk._km(smooth, 3, 10, 0).labels_)
    assert "distinct positions" not in plain["caption"]
    assert "All 400 respondents are shown" in plain["caption"]


def test_the_per_person_fit_chart_covers_everyone_and_stays_quick():
    """It used to draw a random 900 people, because a silhouette needs every pairwise distance.

    The pipeline already computes a per-respondent score for everybody — one minus Leisch's shadow
    value, the same number in the `fit` column of the exported file — so the chart uses that and
    covers the whole study. Drawing it then became the bottleneck instead: one rectangle per
    respondent is 50,000 patches, which took 22 seconds and produced a picture identical to
    filling the same outline once. Measured after: 0.2s and 20 KB at n = 50,000.
    """
    import time

    rng = np.random.default_rng(0)
    n = 5000
    centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)
    truth = rng.integers(0, 3, n)
    X = _likert(centres[truth] + rng.normal(0, 0.8, (n, 4))).astype(float)
    lo, hi = X.min(0), X.max(0)
    Xs = (X - lo) / np.where(hi > lo, hi - lo, 1)
    labels = sk._km(Xs, 3, 10, 0).labels_
    fit = 1.0 - sk.shadow_values(Xs, sk.segment_centres(Xs, labels))[0]

    started = time.monotonic()
    chart = charts.chart_silhouette(Xs, labels, fit=fit)
    elapsed = time.monotonic() - started
    assert f"Every one of the {n:,} respondents is drawn here" in chart["caption"]
    assert "random" not in chart["caption"], "the fit chart is describing a sample again"
    assert elapsed < 10, f"drawing every respondent took {elapsed:.1f}s"
    assert len(chart["svg"]) < 400_000, "the fill was left as vector and the file exploded"

    # Without per-person scores — the latent-class path — it still has to work, and must admit
    # when it has fallen back to a sample rather than silently implying full coverage.
    fallback = charts.chart_silhouette(Xs, labels, fit=None)
    assert fallback is not None and "random" in fallback["caption"]


def test_colour_does_exactly_one_job_per_chart():
    """The palette rework, pinned.

    Colour here does four jobs — which segment somebody is in, whether a value is above or below
    average, how much of something there is, and plain chrome — and the charts used to do all of
    them with one set of hues. Orange meant "Group 1" on four charts, "Separation (silhouette)" on
    the choice-of-k chart and "above average" on the heatmap, so a reader who learned one was
    misled by the next.
    """
    # Identity and polarity must not share hues, or "above average" and "Group 1" look alike.
    assert not (set(charts.SEG_LIGHT) & {charts.DIVERGING[1]})
    assert charts.DIVERGING[1] not in charts.SEG_LIGHT, "the diverging midpoint must read as nothing"
    assert len(set(charts.SEG_LIGHT)) == len(charts.SEG_LIGHT) == len(charts.SEG_DARK)
    assert len(charts.SEG_MARKERS) == len(charts.SEG_LIGHT), "every slot needs its own shape"
    assert len(set(charts.SEG_MARKERS)) == len(charts.SEG_MARKERS)

    rng = np.random.default_rng(0)
    n = 400
    centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)
    truth = rng.integers(0, 3, n)
    X = _likert(centres[truth] + rng.normal(0, 0.7, (n, 4))).astype(float)
    labels = sk._km(X, 3, 10, 0).labels_

    # The whole-sample gorge must not wear a segment's colour: it describes everybody, and in
    # Group 0's green it read as a chart about Group 0.
    shadow = sk.shadow_values(X, sk.segment_centres(X, labels))[0]
    gorge = charts.chart_gorge(shadow)
    for hue in charts.SEG_LIGHT:
        assert f"var(--seg-{charts.SEG_LIGHT.index(hue)}" not in gorge["svg"], (
            "the gorge histogram is wearing a segment colour")

    # The choice-of-k lines are metrics, not segments, and must not borrow identity hues either.
    diag = pd.DataFrame({"k": [2, 3, 4], "prediction_strength": [0.9, 0.95, 0.6],
                         "stability_ARI": [0.8, 0.85, 0.5], "silhouette": [0.5, 0.55, 0.3]})
    kchart = charts.chart_k_choice(diag, 3)
    assert "var(--seg-" not in kchart["svg"], "the k chart is using the segment palette"
    assert charts._METRIC_LEAD not in charts.SEG_LIGHT, (
        "the emphasis colour is one of the identity slots — the very collision being removed")
    assert "var(--chart-lead" in kchart["svg"], "the emphasis colour must follow the theme too"
    assert "decides" in kchart["svg"], "the deciding criterion should be named as such"


def test_segments_are_told_apart_by_shape_as_well_as_colour():
    """Required, not decorative.

    The segment map is a scatter, so every pair of colours is on screen at once. Measured on the
    eight-slot palette in that "all pairs" case: worst CVD separation ΔE 3.2 (green against
    orange, protanopia) and worst normal-vision ΔE 7.1 (red against orange) — far below the floor
    of 15, meaning two segments a reader with full colour vision cannot tell apart. Only the first
    three slots clear all-pairs on colour alone, so shape carries the rest. It also survives a
    photocopier, which colour does not.
    """
    rng = np.random.default_rng(0)
    n = 900
    centres = rng.integers(1, 6, (6, 5)).astype(float)
    truth = rng.integers(0, 6, n)
    X = _likert(centres[truth] + rng.normal(0, 0.7, (n, 5))).astype(float)
    labels = sk._km(X, 6, 10, 0).labels_
    chart = charts.chart_segment_map(X, labels)

    # matplotlib emits each distinct marker shape as its own path in <defs>; six groups must not
    # be drawn with one shape repeated.
    assert chart["svg"].count("<path id=") >= 6, "the map is drawing every group with one shape"
    assert len({sk.seg_marker(i) for i in range(6)}) == 6


def test_charts_carry_their_own_dark_mode():
    """A chart is downloaded, pasted into a document and printed, so its theming travels with it.

    Segment hues ship as CSS variables with the light value as the fallback, and each SVG carries
    the dark steps itself. Two scopes are declared on purpose: the media query follows the
    operating system and the [data-theme] rules follow the reader's own toggle, which has to win
    in both directions.
    """
    rng = np.random.default_rng(0)
    X = _likert(np.array([[5, 1, 5], [1, 5, 1], [3, 3, 5]], float)[
        rng.integers(0, 3, 300)] + rng.normal(0, 0.6, (300, 3))).astype(float)
    chart = charts.chart_segment_map(X, sk._km(X, 3, 10, 0).labels_)
    svg = chart["svg"]

    assert "prefers-color-scheme:dark" in svg and "[data-theme=dark]" in svg
    assert ":root:not([data-theme=light])" in svg, "an explicit light choice must beat OS dark"
    for slot in range(3):
        assert f"--seg-{slot}:{charts.SEG_LIGHT[slot]}" in svg
        assert f"--seg-{slot}:{charts.SEG_DARK[slot]}" in svg
        # The light value stays as the var()'s fallback, so a chart opened outside the app — in a
        # document, a mail client, a printed page — is coloured rather than black.
        assert f"var(--seg-{slot}, {charts.SEG_LIGHT[slot]})" in svg
    assert "--chart-surface" in svg, "separator rings need the page colour, not hard-coded white"

    # var() resolves in a CSS declaration but NOT in an SVG presentation attribute, so every
    # reference has to land inside style="…". If matplotlib ever emits `fill="…"` instead, the
    # colours silently stop following the theme.
    #
    # The spans are parsed rather than sniffed with a fixed lookback. The first version of this
    # check searched the preceding 90 characters for `style=` and passed locally while failing on
    # CI, where a longer font stack pushed the attribute out of the window — a test that depended
    # on how much text happened to sit in front of the thing it was checking.
    styled = [m.span(1) for m in re.finditer(r'style="([^"]*)"', svg)]
    for hit in re.finditer(r"var\(--seg-\d", svg):
        inside = any(lo <= hit.start() < hi for lo, hi in styled)
        assert inside, (
            "a themed colour landed in a presentation attribute, where var() does not apply")
    assert styled, "no style attributes at all — the check above would pass vacuously"


def test_the_fit_chart_compares_segments_instead_of_stacking_everybody():
    """It answers "which group is the weak one", so that has to be the thing you read first.

    The old form sorted every respondent into one tall column of bars. At any realistic sample
    size each bar was a fraction of a pixel, so it showed an outline and hid the individuals it
    claimed to be about — and comparing groups meant measuring three silhouettes against each
    other by eye. It is now one distribution per segment on a shared axis, each with its own
    median and size printed, and the whole sample's median drawn across them.

    Heights are normalised per segment on purpose, so a 40-person group's shape is as readable as
    a 900-person one; the counts are printed beside each row because shape is the question here.
    """
    def build(real):
        rng = np.random.default_rng(0)
        n = 900
        if real:
            centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)
            X = _likert(centres[rng.integers(0, 3, n)] + rng.normal(0, 0.8, (n, 4)))
        else:
            X = _likert(rng.normal(3, 1.2, (n, 4)))
        X = X.astype(float)
        labels = sk._km(X, 3, 10, 0).labels_
        fit = 1.0 - sk.shadow_values(X, sk.segment_centres(X, labels))[0]
        return charts.chart_silhouette(X, labels, fit=fit), fit, labels

    chart, fit, labels = build(real=True)
    svg = chart["svg"]
    # Every segment is named on its own row, and carries its own median and size.
    for c in range(3):
        assert f"Group {c}" in svg
        assert f"{int((labels == c).sum()):,} people" in svg
    assert "whole sample" in svg, "each row has to be readable against the study as a whole"
    assert f"Every one of the {len(fit):,} respondents is drawn here" in chart["caption"]

    # The falsification job still works: on structureless data every group should look the same
    # and sit low, which is the signature this chart exists to make obvious.
    noise_chart, noise_fit, noise_labels = build(real=False)
    real_median = float(np.median(fit))
    noise_median = float(np.median(noise_fit))
    assert noise_median < real_median - 0.25, (
        f"noise ({noise_median:.2f}) should sit far below real structure ({real_median:.2f})")
    spread = [float(np.median(noise_fit[noise_labels == c])) for c in range(3)]
    assert max(spread) - min(spread) < 0.1, "on noise the groups should be indistinguishable"
    assert noise_chart is not None

    # A segment too small to bin must not take the chart down, and must still get its own row.
    lopsided = np.array([0] * 400 + [1] * 400 + [2] * 6)
    rng = np.random.default_rng(1)
    X = _likert(rng.normal(3, 1, (len(lopsided), 4))).astype(float)
    tiny = charts.chart_silhouette(X, lopsided, fit=rng.random(len(lopsided)))
    assert tiny is not None and "6 people" in tiny["svg"]


def test_two_segments_are_never_drawn_identically():
    """Colour and shape both have eight values, so cycling them together fails silently.

    Group 8 got exactly the colour AND the shape of group 0 — two segments drawn identically,
    which is worse than either channel failing on its own because nothing on the chart hints that
    anything is wrong. Measured before the fix: groups 0/8, 1/9, 2/10 and 3/11 were
    indistinguishable in both channels at once, and `--kmax 10` reaches that.
    """
    seen = {}
    for i in range(24):
        pair = (sk.seg_colour(i), sk.seg_marker(i))
        assert pair not in seen, f"group {i} is drawn exactly like group {seen[pair]}"
        seen[pair] = i
    # The ordinary case must be untouched: the first eight keep one colour and one shape each.
    assert len({sk.seg_colour(i) for i in range(8)}) == 8
    assert len({sk.seg_marker(i) for i in range(8)}) == 8


def test_a_question_worded_like_a_colour_survives_the_charts():
    """Theme tokens are swapped in by string replacement, which cannot tell data from markup.

    A question worded "#2a78d6 is my favourite" came out of a chart reading
    "var(--seg-0, #2a78d6) is my favourite" — the label is only text in the SVG, and the
    replacement rewrote it. Respondent-supplied strings live in <text> elements, so those are
    lifted out before the swaps run and put back afterwards.

    The held strings are returned rather than parked on the module, because the server draws for
    several people at once; the concurrent half of this test is what makes that more than a claim.
    """
    import concurrent.futures

    frame = pd.DataFrame([[1.0, 4.0], [4.0, 1.0]],
                         index=["Segment 0", "Segment 1"],
                         columns=["#2a78d6 is my favourite", "plain question"])
    chart = charts.chart_profiles(frame)
    assert "#2a78d6 is my favourite" in chart["svg"], "a label was rewritten as a theme token"
    assert "var(--seg-0," in chart["svg"], "theming stopped being applied to the marks"
    assert "currentColor" in chart["svg"], "chrome stopped following the page"
    assert "\x00text" not in chart["svg"], "a masking sentinel was left in the output"

    def draw(i):
        c = pd.DataFrame([[1.0, 4.0, 2.0], [4.0, 1.0, 3.0]],
                         index=["Segment 0", "Segment 1"],
                         # No underscores: labels render them as spaces, which would make this
                         # probe fail for a reason unrelated to what it is testing.
                         columns=[f"RUN{i}alpha", f"RUN{i}beta", f"RUN{i}gamma"])
        svg = charts.chart_profiles(c)["svg"]
        return (f"RUN{i}alpha" in svg,
                [j for j in range(10) if j != i and f"RUN{j}alpha" in svg],
                "\x00text" in svg)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(draw, range(10)))
    assert all(mine for mine, _, _ in results), "a run lost its own labels"
    assert all(not others for _, others, _ in results), "a run was given another run's labels"
    assert all(not leftover for _, _, leftover in results), "a masking sentinel escaped"


def test_the_report_says_the_number_of_groups_was_searched_for():
    """The tool has always chosen k by a nine-criterion panel and never said so where anyone reads.

    The person who commissioned it asked whether it could "figure out the best k" — it already
    did, several sections below a heading a marketer has no reason to open. A finding nobody knows
    about is not a feature.

    The runner-up is the part that carries information: "3 groups, and 4 was next" is a different
    situation from "3 groups, and nothing else came close", and it is the honest answer to the
    question every client asks — you said three, what if it were four?
    """
    rng = np.random.default_rng(0)
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    df = pd.DataFrame(_likert(centres[rng.integers(0, 3, 500)] + rng.normal(0, 0.7, (500, 5))),
                      columns=[f"q{i+1}" for i in range(5)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=8, **FAST))
    plain = r["digest"].split("## ")[1]           # the plain-language box, not a later section
    assert "I tried every number of groups from 2 to 8" in plain
    assert "independent criteria" in plain
    assert str(r["k"]) in plain

    # It reports a vote, not a verdict. On structureless data some number still wins, and calling
    # that "the clear answer" would contradict the red light directly above it.
    noise = pd.DataFrame(_likert(rng.normal(3, 1.1, (500, 5))),
                         columns=[f"q{i+1}" for i in range(5)])
    noise.insert(0, "respondent_id", [f"P{i}" for i in range(len(noise))])
    rn = sk.run_analysis(noise.to_csv(index=False).encode(),
                         cfg=sk.SegmentationConfig(k_min=2, k_max=8, **FAST))
    assert rn["confidence"] == "low"
    assert "clear answer" not in rn["digest"]
    assert "these criteria agree on most" in rn["digest"]

    # The sentence itself, on inputs the pipeline would not normally produce.
    assert sk.how_k_was_chosen({}, 3, 2, 8) == ""
    assert sk.how_k_was_chosen({"a": None}, 3, 2, 8) == ""
    only = sk.how_k_was_chosen({"a": 3, "b": 3}, 3, 2, 8)
    assert "nothing else was chosen by any of them" in only
    tied = sk.how_k_was_chosen({"a": 3, "b": 4, "c": 4}, 3, 2, 8)
    assert "judgement call" in tied, "a rival with more votes must not read as settled"


def test_the_interactive_spec_and_the_drawn_chart_cannot_disagree():
    """One computation, two renderers — which is the whole reason a spec exists.

    The obvious way to add interactive charts is to write a second chart engine in TypeScript, and
    then the two slowly disagree about what the data says the first time somebody edits one of
    them. The spec is built from the SAME arrays matplotlib drew from, so the browser cannot show
    a different number of people, a different segment, or a different colour from the picture in
    the printed report.
    """
    rng = np.random.default_rng(0)
    n = 800
    centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)
    X = _likert(centres[rng.integers(0, 3, n)] + rng.normal(0, 0.8, (n, 4))).astype(float)
    labels = sk._km(X, 3, 10, 0).labels_
    chart = charts.chart_segment_map(X, labels)
    spec = chart["spec"]

    assert spec["version"] == charts.SPEC_VERSION and spec["kind"] == "segment_map"
    # Every respondent is accounted for exactly once, which is the same promise the drawing makes.
    assert sum(spec["points"]["people"]) == n == spec["people"]
    # One mark per distinct answer pattern — the same aggregation the chart drew.
    coords, _, _ = charts.pca_2d(X)
    assert len(spec["points"]["x"]) == len(np.unique(np.round(coords, 9), axis=0))
    # Parallel arrays must stay parallel, or the browser drops respondents off the chart.
    assert len({len(v) for v in spec["points"].values()}) == 1

    # Identity is sent, never re-derived on the other side: the palette carries the measured
    # colour-vision properties and the markers carry what colour cannot, so they have one home.
    for i, key in enumerate(spec["segments"]):
        assert key["index"] == i
        assert key["colour"] == sk.seg_colour(i)
        assert key["colour_dark"] == charts.SEG_DARK[i % len(charts.SEG_DARK)]
        assert key["marker"] == sk.seg_marker(i)

    # No respondent identity travels with the spec — it is the same aggregate the picture shows.
    blob = json.dumps(spec)
    assert "respondent" not in blob and "id" not in spec["points"]

    # Past the cap the chart still draws; it simply ships no spec, and the interface falls back to
    # the static picture rather than to nothing.
    assert charts.INTERACTIVE_MAX_POINTS > 0
    wide = _likert(rng.normal(3, 1.2, (300, 4))).astype(float)
    small = charts.chart_segment_map(wide, sk._km(wide, 2, 10, 0).labels_)
    assert "spec" in small, "an ordinary survey should be interactive"
    huge = np.arange(charts.INTERACTIVE_MAX_POINTS + 50, dtype=float).reshape(-1, 1)
    huge = np.hstack([huge, huge * 2])
    no_spec = charts.chart_segment_map(huge, (np.arange(len(huge)) % 2))
    assert no_spec is not None and "spec" not in no_spec
    assert no_spec["svg"].startswith("<svg"), "the static drawing must still be there"


def test_a_second_cluster_tendency_test_covers_where_hopkins_is_weak():
    """Hopkins has always been the tool's only real answer to "is there anything here".

    Adolfsson, Ackerman & Brownstein (Pattern Recognition, 2019) measured its two failure modes
    over 35,000 simulated datasets: its power falls to 32% on partially overlapping clusters, and
    it reads a handful of outliers as a group. Overlapping segments merging into one is the single
    failure mode measured in `references/kbench.py`, so the second test earns its place precisely
    there.
    """
    if not clusterability.available():
        pytest.skip("the dip test add-on is not installed")

    rng = np.random.default_rng(0)
    n = 400
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)

    def tendency(X):
        lo, hi = X.min(0), X.max(0)
        Xs = (X - lo) / np.where(hi > lo, hi - lo, 1)
        return clusterability.pca_dip(Xs, charts.pca_2d(Xs)[0][:, 0])

    # Real groups are found at every separation, including the heavy overlap where Hopkins fades.
    for spread in (0.5, 1.0, 1.4):
        found = tendency(_likert(centres[rng.integers(0, 3, n)] + rng.normal(0, spread, (n, 5))))
        assert found["p"] < 0.05, f"missed real groups at spread {spread}: {found}"

    # And it does not invent them. Both cases below are ones Hopkins reads as structure.
    assert tendency(_likert(rng.normal(3, 1.1, (n, 5))))["p"] >= 0.05
    outliers = _likert(rng.normal(3, 1.0, (n, 5)))
    outliers[:5] = 5
    assert tendency(outliers)["p"] >= 0.05, "five eccentric respondents were read as a group"


def test_the_dip_refuses_the_data_it_cannot_read_rather_than_guessing():
    """The obvious form of this test is wrong for survey data, and the guard is the whole story.

    The paper's headline recommendation is the dip on all PAIRWISE DISTANCES. Rating answers are
    whole numbers, so those distances take very few values — measured, 400 people answering five
    questions produce 79,800 distances with 50 distinct values among them — and the dip reads that
    comb as many modes, returning p = 0.0000 on data with no groups at all. The same test on
    continuous noise of identical size returns p = 0.9962. So the first principal component is
    used instead, being a weighted sum and therefore continuous.

    That still needs enough questions to sum. Measured on pure noise: two and three questions
    false-alarm, four and up are correct. The guard counts QUESTIONS rather than distinct values,
    because a low count of distinct values has two causes and only one is a fault — genuinely tight
    groups drive it down too, and guarding on it refused three well-separated groups.
    """
    if not clusterability.available():
        pytest.skip("the dip test add-on is not installed")

    rng = np.random.default_rng(0)
    n = 400

    def run(X):
        lo, hi = X.min(0), X.max(0)
        Xs = (X - lo) / np.where(hi > lo, hi - lo, 1)
        return clusterability.pca_dip(Xs, charts.pca_2d(Xs)[0][:, 0])

    # Too few questions: refused, with a reason, rather than a false alarm.
    for items in (2, 3):
        verdict = run(_likert(rng.normal(3, 1.1, (n, items))))
        assert "p" not in verdict, f"{items} questions should be refused, got {verdict}"
        assert "question" in verdict["skipped"]

    # Too few people: refused for a different, stated reason.
    assert "respondents" in run(_likert(rng.normal(3, 1.1, (20, 5))))["skipped"]

    # Tight, well-separated groups must NOT be refused — they were, when the guard counted
    # distinct values, because people in one segment genuinely give identical answers.
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    tight = run(_likert(centres[rng.integers(0, 3, n)] + rng.normal(0, 0.4, (n, 5))))
    assert "p" in tight, f"the clearest case in the set was refused: {tight}"
    assert tight["p"] < 0.05


def test_the_two_tendency_tests_are_read_against_each_other():
    """Disagreement is informative, because the two fail in opposite directions."""
    assert clusterability.agreement(0.9, None) is None
    assert clusterability.agreement(0.9, {"skipped": "nope"}) is None
    assert clusterability.agreement(0.9, {"p": 0.001})[0] == "both"
    assert clusterability.agreement(0.4, {"p": 0.9})[0] == "neither"
    # Hopkins quiet, dip loud: the signature of groups that overlap, which is Hopkins' weak spot.
    label, sentence = clusterability.agreement(0.4, {"p": 0.001})
    assert label == "dip only" and "overlap" in sentence
    # Hopkins loud, dip quiet: the signature of a few outliers, which Hopkins reads as a group.
    label, sentence = clusterability.agreement(0.9, {"p": 0.9})
    assert label == "hopkins only" and "outliers" in sentence


def test_the_report_carries_the_second_opinion():
    """It is only useful if the reader sees it, and it must say when it could not run."""
    rng = np.random.default_rng(0)
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    df = pd.DataFrame(_likert(centres[rng.integers(0, 3, 400)] + rng.normal(0, 0.7, (400, 5))),
                      columns=[f"q{i+1}" for i in range(5)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
    digest = sk.run_analysis(df.to_csv(index=False).encode(),
                             cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))["digest"]
    if clusterability.available():
        assert "Dip test (second opinion)" in digest
        # Never printed as "p 0": a test reports a probability, and zero claims a certainty no
        # statistical test has.
        assert "p 0**" not in digest and "p = 0**" not in digest
        assert "Both cluster-tendency tests agree" in digest

    # Two questions: the section has to say it did not run, not vanish silently.
    short = pd.DataFrame(_likert(rng.normal(3, 1.1, (400, 2))), columns=["q1", "q2"])
    short.insert(0, "respondent_id", [f"P{i}" for i in range(len(short))])
    brief = sk.run_analysis(short.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))["digest"]
    if clusterability.available():
        assert "not run" in brief


def test_every_module_is_actually_shipped_in_the_wheel():
    """`py-modules` in pyproject.toml is a hand-kept list, and it has now been forgotten twice.

    Both times the symptom was the same and invisible locally: every test passes, because tests
    import from the working directory, and then every *installed* copy dies at import with
    ModuleNotFoundError. `kprototypes` went out that way, and `clusterability` repeated it a few
    days later. A list that must be updated by memory will be forgotten a third time, so this
    checks it instead of trusting it.
    """
    root = Path(__file__).resolve().parent.parent
    declared = set(re.findall(r'"([a-z_]+)"',
                              re.search(r"py-modules = \[(.*?)\]",
                                        (root / "pyproject.toml").read_text(), re.S).group(1)))
    # Everything at the top level that is part of the package: not the entry points, not the build
    # script, not this suite.
    not_shipped = {"run_app", "build_app", "conftest", "setup"}
    present = {p.stem for p in root.glob("*.py")} - not_shipped
    missing = present - declared
    assert not missing, (
        f"{sorted(missing)} exist but are not in pyproject.toml's py-modules, so a `pip install` "
        "of this project would be missing them")
    stale = declared - present
    assert not stale, f"{sorted(stale)} are declared but no longer exist"


def test_the_packaged_build_installs_from_the_declared_dependencies():
    """build_app.py used to repeat the dependency list, and a second list goes stale too.

    It had already: `diptest` was added to pyproject.toml and not to the copy in build_app.py, so
    every build on a machine that did not happen to have it produced an app whose second
    cluster-tendency test silently never ran. Both CI runners were in that state.

    A list can be checked; not having one cannot go wrong. This pins that the build installs the
    project itself rather than naming packages again.
    """
    build = (Path(__file__).resolve().parent.parent / "build_app.py").read_text()
    install = re.search(r'"pip", "install",(.*?)\]', build, re.S)
    assert install, "the build no longer installs anything, which cannot be right"
    named = install.group(1)
    assert '".[' in named, "the build should install the project, not a repeated list of packages"
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "diptest"):
        assert f'"{package}"' not in named, (
            f"{package} is named directly in build_app.py; it belongs in pyproject.toml so there "
            "is one list rather than two that can disagree")


# The measured k-selection table from a 400-respondent file with three planted segments, kept
# verbatim so the case that exposed these defects cannot quietly come back. Columns are the ones
# recommend_k reads; the numbers are what the search actually produced.
_THREE_SEGMENT_DIAGNOSTICS = [
    # k  inertia   sil     CH       DB     share   gap   gap_se  stab  stab_sd   PS    PS_sd    BIC       ICL      PAC
    (2, 264.302, 0.327, 252.512, 1.190, 0.420, 0.811, 0.014, 0.658, 0.274, 0.593, 0.099, 295.480, 300.668, 0.353),
    (3, 198.594, 0.297, 233.285, 1.327, 0.272, 1.017, 0.013, 0.995, 0.005, 0.968, 0.021, 559.368, 567.658, 0.003),
    (4, 191.797, 0.201, 165.307, 2.167, 0.145, 0.991, 0.018, 0.778, 0.075, 0.512, 0.034, 934.412, 949.250, 0.193),
    (5, 186.034, 0.121, 130.557, 2.765, 0.125, 0.981, 0.015, 0.603, 0.085, 0.487, 0.030, 1345.150, 1391.240, 0.229),
]
_DIAG_COLUMNS = ["k", "inertia", "silhouette", "calinski_harabasz", "davies_bouldin",
                 "min_segment_share", "gap", "gap_se", "stability_ARI", "stability_ARI_sd",
                 "prediction_strength", "prediction_strength_sd", "gmm_BIC", "gmm_ICL",
                 "consensus_PAC"]


def _tally(signals, cfg):
    weights = sk.signal_weights(cfg)
    out = {}
    for name, k in signals.items():
        out[k] = out.get(k, 0) + weights.get(name, 1)
    return out


def test_the_stability_signal_backs_the_k_that_is_actually_stable():
    """"Largest k above the cutoff" is Tibshirani & Walther's rule for prediction strength, where
    it is right. It was also being applied to replication stability, where it is not.

    On the file below, stability ran 0.995 at k=3 and 0.778 at k=4. The old rule read both as
    "above 0.75" and handed the signal — one of the two the whole method leans on — to k=4, which
    was enough to tie the vote and lose the segmentation. The one-standard-error rule keeps only
    the k values that cannot be told apart from the best.
    """
    diag = pd.DataFrame(_THREE_SEGMENT_DIAGNOSTICS, columns=_DIAG_COLUMNS)
    cfg = sk.SegmentationConfig()
    _, _, signals = sk.recommend_k(diag, cfg)

    assert signals["global stability"] == 3, (
        f"stability voted for k={signals['global stability']} when k=3 scored 0.995 and the "
        "runner-up 0.778 — a signal weighted double must not go to a visibly worse solution")


def test_the_number_of_groups_survives_the_criteria_that_matter_most():
    """End of the same story: the recommendation itself.

    k=3 held both doubled criteria (prediction strength 0.968 against 0.593 — k=2 does not even
    clear the 0.80 cutoff the report quotes — and consensus PAC 0.003 against 0.353). k=2 held the
    separation indices. The vote tied and parsimony took k=2, and the three real segments were
    then written up as constructed noise.
    """
    diag = pd.DataFrame(_THREE_SEGMENT_DIAGNOSTICS, columns=_DIAG_COLUMNS)
    cfg = sk.SegmentationConfig()
    pick, _, signals = sk.recommend_k(diag, cfg)

    assert pick == 3, f"recommended k={pick} on a file whose structure is plainly three segments"
    assert signals["prediction strength"] == 3 and signals["consensus PAC"] == 3


def test_a_tie_is_broken_by_the_criteria_the_method_leans_on():
    """Parsimony breaks what the weighted criteria leave tied, not the other way round.

    Constructed so the weighted vote lands exactly level at 5 each, with the larger k holding both
    doubled criteria and the smaller holding only one. Choosing the smaller here would contradict
    the priority the function is built around and states in its own report.
    """
    rows = [
        (2, 300.0, 0.40, 300.0, 1.50, 0.45, 0.90, 0.01, 0.90, 0.02, 0.60, 0.05, 300.0, 305.0, 0.30),
        (3, 250.0, 0.30, 200.0, 1.00, 0.30, 0.85, 0.01, 0.70, 0.02, 0.95, 0.02, 500.0, 505.0, 0.01),
        (4, 245.0, 0.20, 150.0, 2.00, 0.20, 0.95, 0.01, 0.60, 0.02, 0.50, 0.03, 200.0, 205.0, 0.20),
    ]
    cfg = sk.SegmentationConfig()
    pick, _, signals = sk.recommend_k(pd.DataFrame(rows, columns=_DIAG_COLUMNS), cfg)

    tally = _tally(signals, cfg)
    assert tally[2] == tally[3], (
        f"this table is meant to tie 2 against 3; it scored {tally} — retune it, the tie-break "
        "is what is under test")
    assert pick == 3, "the tie went to parsimony over both of the doubled criteria"


def test_the_headline_counts_the_vote_that_actually_decided():
    """The plain-language line explains a weighted decision, so it has to report the weighted
    tally. Counting heads instead produced "5 of them picked 2" for a k that both of the criteria
    this tool trusts most had argued against — an explanation that was not one."""
    diag = pd.DataFrame(_THREE_SEGMENT_DIAGNOSTICS, columns=_DIAG_COLUMNS)
    cfg = sk.SegmentationConfig()
    pick, _, signals = sk.recommend_k(diag, cfg)
    line = sk.how_k_was_chosen(signals, pick, cfg.k_min, cfg.k_max, cfg=cfg)

    tally = _tally(signals, cfg)
    assert f"{pick} scored {tally[pick]}" in line, line
    runner = max((v, k) for k, v in tally.items() if k != pick)[1]
    assert str(tally[runner]) in line, f"the runner-up's weighted score is missing from: {line}"


def test_forcing_the_number_of_groups_works_on_the_path_everyone_uses(tmp_path):
    """--force-k was wired into every explicit --method path and left out of the automatic one,
    which is the default. So for almost every user the flag did nothing at all: the run finished,
    the report never mentioned an override, and the number in it was the tool's own. An ignored
    flag that reports success is worse than one that is missing, because the reader believes the
    answer is the one they asked for.
    """
    import argparse
    rng = np.random.default_rng(5)
    rows = []
    for i in range(180):
        base = {0: [5, 1, 5, 2], 1: [1, 5, 2, 4], 2: [3, 3, 4, 5]}[i % 3]
        rows.append([f"R{i}"] + [int(np.clip(round(b + rng.normal(0, 0.7)), 1, 5)) for b in base])
    df = pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4"])

    outdir = tmp_path / "forced"
    args = argparse.Namespace(id_col="respondent_id", outdir=str(outdir), seed=42, force_k=5)
    with contextlib.redirect_stdout(io.StringIO()):
        sk.run_auto(df, args, argparse.ArgumentParser())

    assignments = pd.read_csv(outdir / "segment_assignments.csv")
    assert assignments["segment"].nunique() == 5, (
        f"asked for 5 groups on the default path and got "
        f"{assignments['segment'].nunique()} — the flag was ignored")
    report = (outdir / "segmentation_report.md").read_text()
    assert "force_k" in report, "the report does not say the number was overridden"


def test_response_styles_do_not_become_the_segments():
    """The classic way survey segmentation goes wrong, measured rather than assumed.

    Real respondents differ in how they USE a scale as well as in what they think: some push to
    the ends, some hug the middle, some agree with everything. Cluster raw Likert answers and the
    textbook worry is that you recover scale-use groups and present them as mind-sets.

    Every respondent here has an attitude segment AND, independently, a response style, so the
    question "which one did it recover?" has an answer. Measured: Adjusted Rand Index against
    response style comes out at -0.002 — no relationship at all — while attitude is recovered.
    The worry does not materialise on this engine, and this test exists to catch it if that ever
    changes.

    What response styles DO cost is resolution: they blur the three-way structure until the
    two-way split is genuinely the more reproducible fact, so the tool merges two segments and
    drops its own confidence to Moderate rather than overclaiming. That behaviour is asserted
    below too, because quietly reporting High here would be the real failure.
    """
    rng = np.random.default_rng(17)
    attitudes = np.array([[4.6, 4.4, 4.5, 2.0, 1.8, 3.9, 1.7, 2.2, 3.9, 2.1, 2.4, 3.0],
                          [1.9, 2.6, 1.8, 4.6, 4.3, 2.8, 3.6, 3.5, 2.2, 4.4, 4.3, 3.2],
                          [3.0, 3.9, 3.1, 3.8, 2.4, 4.6, 1.9, 4.7, 3.1, 3.3, 2.6, 4.2]])
    n = 450
    attitude = rng.integers(0, 3, n)
    style = rng.integers(0, 3, n)
    rows = []
    for i in range(n):
        x = attitudes[attitude[i]] + rng.normal(0, 0.55, attitudes.shape[1])
        if style[i] == 0:
            x = 3 + 1.7 * (x - 3)        # extreme responder
        elif style[i] == 1:
            x = 3 + 0.45 * (x - 3)       # midpoint responder
        else:
            x = x + 0.9                  # acquiescent
        rows.append(np.clip(np.rint(x), 1, 5))
    df = pd.DataFrame(np.array(rows, int), columns=[f"q{i+1}" for i in range(attitudes.shape[1])])
    df.insert(0, "respondent_id", [f"R{i}" for i in range(n)])

    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
    labels = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))["segment"]

    to_style = adjusted_rand_score(style, labels)
    to_attitude = adjusted_rand_score(attitude, labels)

    assert to_style < 0.15, (
        f"the segments track how people use the scale (ARI {to_style:.3f}), not what they think — "
        "this is the failure mode the response-style literature warns about")
    assert to_attitude > to_style + 0.3, (
        f"attitude {to_attitude:.3f} vs response style {to_style:.3f}: the segments are not "
        "clearly about what people think")
    assert r["confidence"] != "high" or r["k"] == 3, (
        "merged the blurred segments and still reported high confidence")


def test_demographics_are_recognised_in_both_languages():
    """A numeric demographic code is indistinguishable from a rating scale by its values.

    Education coded 1-5 looks exactly like a six-point Likert item, so only the column NAME can
    tell them apart — which makes the name list load-bearing. It had drifted: the Swedish
    'utbildning' was present and the English 'education' was not, so a Swedish survey's education
    column was set aside correctly while an English one was clustered on. Found on the Big Five
    inventory (psych::bfi), where a 1-5 education code sat among 25 six-point personality items
    and silently became the 26th question.

    Every pair below is one concept in both languages. A word recognised in one language and not
    the other is the bug this test exists for.
    """
    pairs = [("Education", "Utbildning"), ("Gender", "Kön"), ("Age", "Ålder"),
             ("Occupation", "Yrke"), ("Household size", "Hushåll"), ("Language", "Språk"),
             ("City", "Stad"), ("Income", "Inkomst"), ("Postcode", "Postnummer"),
             ("University", "Universitet"), ("Marital status", "Civilstånd")]
    missed = [name for pair in pairs for name in pair if not sk._looks_demographic(name.lower())]
    assert not missed, f"not recognised as background traits: {missed}"

    # And the guard that keeps attitude questions out still holds: a rating item is a sentence,
    # not a label, however many demographic words it happens to contain.
    for question in ["Campus politics puts me off using an app like this",
                     "My university should do more about student income inequality"]:
        substantive = [w for w in re.findall(sk._WORD_RE, question.lower())
                       if w not in sk._LABEL_STOP]
        assert len(substantive) > 3, f"guard would misread a question as a demographic: {question}"


def test_a_numeric_demographic_does_not_join_the_rating_grid():
    """End to end: the education column must describe the segments, not help form them."""
    rng = np.random.default_rng(4)
    n = 300
    centres = {0: [5, 1, 5, 1], 1: [1, 5, 1, 5], 2: [3, 3, 5, 5]}
    who = rng.integers(0, 3, n)
    rows = []
    for i in range(n):
        base = centres[who[i]]
        rows.append([f"R{i}"] + [int(np.clip(round(b + rng.normal(0, 0.6)), 1, 5)) for b in base]
                    + [int(rng.integers(1, 6))])          # education 1-5, unrelated to attitude
    df = pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4", "education"])

    _, _, _, items, plan = sk.auto_prepare(df)
    assert "education" not in items, "education was used to form the groups"
    assert "education" in plan["demographics"], "education was not kept for profiling either"


def test_a_number_too_big_to_be_an_answer_is_flagged():
    """Some numeric columns are facts about a person, not answers they gave, and they quietly
    help define the segments.

    Found on the Chilean plebiscite survey (carData::Chile): 'population' — the size of the
    respondent's town, 3,750 to 250,000 — was clustered on alongside two real opinion variables.
    The result was eight segments, each pure on how the person voted and then split in two by
    town size. Four of the eight "mind-sets" were really "lives somewhere bigger".

    Whether a number is an answer or a circumstance is a judgement about the study, not a property
    of the data, so this warns and leaves the column in rather than guessing. What it must not do
    is stay silent.
    """
    rng = np.random.default_rng(9)
    n = 120
    df = pd.DataFrame({
        "respondent_id": [f"R{i}" for i in range(n)],
        "q1": rng.integers(1, 6, n),
        "q2": rng.integers(1, 6, n),
        "nps": rng.integers(0, 11, n),              # 0-10, a real answer scale
        "slider": rng.integers(0, 101, n),          # 0-100, also a real answer scale
        # A handful of town sizes shared across respondents, as in the real file. All-distinct
        # large integers would be a record id instead, and are correctly skipped as one.
        "population": rng.choice([3_750, 27_000, 65_000, 159_000, 250_000], n),
    })
    plan = sk.classify_columns(df)
    notes = " ".join(plan["notes"])

    assert "population" in plan["continuous"], "the column should still be usable, not dropped"
    flagged = [n_ for n_ in plan["notes"] if "population" in n_ and "far larger" in n_]
    assert flagged, f"a column running to 250,000 was presented as a rating: {notes}"

    # ...and the genuine answer scales must not be flagged, or the warning becomes noise.
    for real in ("nps", "slider", "q1"):
        assert not any(real in n_ and "far larger" in n_ for n_ in plan["notes"]), \
            f"'{real}' is a normal answer scale and must not be flagged"


def _export_body(n=200, seed=3):
    """The same 200 respondents, to be written out in each tool's format."""
    rng = np.random.default_rng(seed)
    centres = {0: [5, 1, 5, 1, 4], 1: [1, 5, 1, 5, 2], 2: [3, 3, 5, 5, 1]}
    who = rng.integers(0, 3, n)
    rows = [[int(np.clip(round(b + rng.normal(0, 0.6)), 1, 5)) for b in centres[w]] for w in who]
    df = pd.DataFrame(rows, columns=[f"Q{i+1}" for i in range(5)])
    df.insert(0, "ResponseId", [f"R_{i:05d}" for i in range(n)])
    return df, who


QUESTION_TEXT = ["I compare prices before buying", "Quality matters more than cost",
                 "I wait for a sale", "I like trying new brands", "Recommendations sway me"]


def test_a_qualtrics_export_is_read_as_answers_not_as_question_wording(tmp_path):
    """Qualtrics writes THREE header rows: short name, full question text, and a JSON blob like
    {"ImportId": "QID1"}. pandas keeps the first as the header, so the other two arrive as
    respondents.

    Measured on a 240-person export before this: the file read as 242 rows, every column came out
    as text because the question wording contaminated it, and the survey was routed to latent
    class analysis with its 1-5 scales treated as unordered categories. Nothing raised, and the
    report looked ordinary. SurveyMonkey does the same with two rows.
    """
    df, who = _export_body()
    path = tmp_path / "qualtrics.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(df.columns) + "\n")
        f.write(",".join(["Response ID"] + [f'"{q}"' for q in QUESTION_TEXT]) + "\n")
        f.write(",".join(json.dumps({"ImportId": c}).replace(",", ";") for c in df.columns) + "\n")
        df.to_csv(f, header=False, index=False)

    got = sk._read_table(str(path))
    assert len(got) == len(df), f"read {len(got)} rows from a {len(df)}-person export"

    _, method, _, items, _ = sk.auto_prepare(got)
    assert method == "kmeans", f"rating scales were routed to '{method}' as if unordered"
    assert len(items) == 5

    r = sk.run_analysis(path.read_bytes(), cfg=sk.SegmentationConfig(k_min=2, k_max=5, **FAST))
    labels = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))["segment"]
    assert adjusted_rand_score(who, labels) > 0.9, "the planted segments did not survive the export"


def test_a_surveymonkey_export_loses_neither_a_row_nor_a_respondent(tmp_path):
    """Two header rows rather than three, and the second is a sub-label under each question."""
    df, _ = _export_body()
    path = tmp_path / "surveymonkey.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(["Respondent ID"] + [f'"{q}"' for q in QUESTION_TEXT]) + "\n")
        f.write(",".join([""] + ["Response"] * 5) + "\n")
        df.to_csv(f, header=False, index=False)

    got = sk._read_table(str(path))
    assert len(got) == len(df), f"read {len(got)} rows from a {len(df)}-person export"
    assert sk.auto_prepare(got)[1] == "kmeans"


def test_a_swedish_excel_export_keeps_its_decimal_scores(tmp_path):
    """Swedish and German Excel write 4,5 for four-and-a-half and separate fields with ';'.

    The delimiter was already handled; the decimal comma was not. Such a column arrived as text
    and was then quietly dropped from the rating grid — measured on an export whose 0-10
    satisfaction score vanished from the analysis without a word.
    """
    df, _ = _export_body()
    rng = np.random.default_rng(11)
    df["Nöjdhet (0-10)"] = np.round(rng.uniform(0, 10, len(df)), 1)
    path = tmp_path / "swedish.csv"
    path.write_text(df.to_csv(sep=";", index=False, decimal=","), encoding="utf-8-sig")

    got = sk._read_table(str(path))
    assert len(got) == len(df)
    assert pd.api.types.is_numeric_dtype(got["Nöjdhet (0-10)"]), \
        "the satisfaction score came back as text and will be dropped from the grid"
    assert abs(float(got["Nöjdhet (0-10)"].mean()) - float(df["Nöjdhet (0-10)"].mean())) < 0.01
    assert "Nöjdhet (0-10)" in sk.auto_prepare(got)[3], "the score is not among the questions"

    # pandas 3 gives text columns a `str` dtype rather than `object`, and the first version of
    # this repair asked `dtype != object` — so it skipped every string column and quietly stopped
    # working. Local pandas 2.3 still said `object` and passed; CI on 3.11/3.12 did not. Pin the
    # behaviour to "is it numeric?" using an explicitly string-typed column, which reproduces the
    # same condition on either version.
    typed = pd.DataFrame({"score": pd.array(["1,3", "5,0", "6,0", "0,3"], dtype="string")})
    assert typed["score"].dtype != object                      # the condition CI hit
    repaired = sk._fix_decimal_commas(typed.copy())
    assert pd.api.types.is_numeric_dtype(repaired["score"]), \
        "a string-dtype column was skipped; the guard is testing dtype identity again"
    assert list(repaired["score"]) == [1.3, 5.0, 6.0, 0.3]


def test_header_stripping_never_eats_a_real_respondent(tmp_path):
    """The dangerous direction. Header rows are recognised by being non-numeric where the answers
    below them are numeric, so an all-categorical survey offers nothing to match and must come
    back whole — as must a numeric survey whose first respondent is perfectly ordinary.
    """
    words = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    rng = np.random.default_rng(2)
    text_only = pd.DataFrame({"respondent_id": [f"R{i}" for i in range(60)],
                              "q1": rng.choice(words, 60), "q2": rng.choice(words, 60),
                              "q3": rng.choice(words, 60)})
    p1 = tmp_path / "text.csv"; text_only.to_csv(p1, index=False)
    assert len(sk._read_table(str(p1))) == 60, "an all-text survey lost rows to header stripping"

    numeric, _ = _export_body(n=60)
    p2 = tmp_path / "numeric.csv"; numeric.to_csv(p2, index=False)
    got = sk._read_table(str(p2))
    assert len(got) == 60, "a clean numeric export lost rows"
    assert got.iloc[0]["ResponseId"] == numeric.iloc[0]["ResponseId"], "the first respondent was eaten"

    # A blank first response is a respondent, not a header row.
    gappy = numeric.copy()
    gappy.loc[0, ["Q1", "Q2", "Q3", "Q4", "Q5"]] = np.nan
    p3 = tmp_path / "gappy.csv"; gappy.to_csv(p3, index=False)
    assert len(sk._read_table(str(p3))) == 60, "a respondent who skipped every question was dropped"


def _diag_columns(digest):
    """Column names of the k-selection table in a report. Read from the table's own header row:
    the prose underneath explains every criterion by name whether or not it was computed, so
    searching the whole digest for a column name finds the explanation, not the column."""
    for line in digest.splitlines():
        if line.startswith("|") and "inertia" in line:
            return [c.strip() for c in line.strip("|").split("|")]
    return []


def test_a_large_study_still_gets_its_pairwise_diagnostics(monkeypatch):
    """Anything needing a distance between every pair of people cannot be computed over everybody
    once a study is large: the pair count grows as the square, and the consensus routine holds two
    dense n-by-n matrices. At 41,188 respondents those want 27 GB.

    The old response was to drop consensus clustering entirely above 5,000 — which silently cost
    the consensus_PAC column, one of the three criteria the k panel weights double, on exactly the
    large studies where a second opinion is worth most. Measured on the UCI bank marketing file
    (41,188 real telephone-survey responses): PAC was simply absent from the table.

    Both are now estimated from a random subsample and the report says so. The threshold is
    lowered here so the test stays quick.
    """
    monkeypatch.setattr(sk, "MAX_PAIRWISE_N", 150)
    rng = np.random.default_rng(6)
    n = 400
    centres = {0: [5, 1, 5, 1], 1: [1, 5, 1, 5], 2: [3, 3, 5, 5]}
    who = rng.integers(0, 3, n)
    rows = [[f"R{i}"] + [int(np.clip(round(b + rng.normal(0, 0.7)), 1, 5)) for b in centres[who[i]]]
            for i in range(n)]
    df = pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4"])

    fast = dict(FAST); fast["run_consensus"] = True
    cfg = sk.SegmentationConfig(k_min=2, k_max=4, consensus_H=6, **fast)
    r = sk.run_analysis(df.to_csv(index=False).encode(), cfg=cfg)

    cols = _diag_columns(r["digest"])
    assert "consensus_PAC" in cols, (
        f"PAC is missing on a large study — the panel is down one of its three doubled "
        f"criteria. Columns present: {cols}")
    assert "silhouette" in cols, "the silhouette was lost rather than sampled"

    # And it must SAY it sampled. A sampled number presented as covering the study is the exact
    # failure this project keeps meeting.
    assert "estimated on a sample" in r["digest"], "the report does not disclose the subsampling"
    assert f"random {150:,} of them" in r["digest"] or "random 150 of them" in r["digest"], \
        "the disclosure does not say how many respondents it used"


def test_a_small_study_is_not_sampled_and_says_nothing_about_it(monkeypatch):
    """The dangerous direction: an ordinary survey must be computed over everybody, and must not
    carry a footnote implying otherwise."""
    monkeypatch.setattr(sk, "MAX_PAIRWISE_N", 6000)
    rng = np.random.default_rng(7)
    n = 200
    centres = {0: [5, 1, 5, 1], 1: [1, 5, 1, 5]}
    who = rng.integers(0, 2, n)
    rows = [[f"R{i}"] + [int(np.clip(round(b + rng.normal(0, 0.7)), 1, 5)) for b in centres[who[i]]]
            for i in range(n)]
    df = pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4"])

    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))
    assert "estimated on a sample" not in r["digest"], \
        "a 200-person survey was described as sampled"
    assert sk._pairwise_sample(n, sk.SegmentationConfig()) is None


def test_the_sampling_note_names_only_columns_that_are_there(monkeypatch):
    """The note once named consensus_PAC on a run where consensus was switched off, describing a
    column the reader could not find. It reports what the table actually contains."""
    monkeypatch.setattr(sk, "MAX_PAIRWISE_N", 150)
    rng = np.random.default_rng(8)
    n = 400
    centres = {0: [5, 1, 5, 1], 1: [1, 5, 1, 5], 2: [3, 3, 5, 5]}
    who = rng.integers(0, 3, n)
    rows = [[f"R{i}"] + [int(np.clip(round(b + rng.normal(0, 0.7)), 1, 5)) for b in centres[who[i]]]
            for i in range(n)]
    df = pd.DataFrame(rows, columns=["respondent_id", "q1", "q2", "q3", "q4"])

    r = sk.run_analysis(df.to_csv(index=False).encode(),          # FAST turns consensus off
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    assert "consensus_PAC" not in _diag_columns(r["digest"]), \
        "fixture assumption: consensus is off under FAST"
    if "estimated on a sample" in r["digest"]:
        note = r["digest"].split("estimated on a sample")[1][:400]
        assert "consensus_PAC" not in note, "the note names a column that is not in the table"
        assert "silhouette" in note


def test_encoding_an_answer_scale_does_not_depend_on_how_many_levels_it_has():
    """`encode` maps each answer to its nearest known level. It did that with

        np.abs(col[:, None] - known[None, :]).argmin(1)

    which allocates one number per respondent PER LEVEL. On a five-point scale that is nothing.
    On a column of 48,842 distinct values — which is what a continuous measurement looks like once
    something has typed it as ordinal — it is 19 GB, and measured on the UCI adult file this one
    line was the entire 11 GB peak of a run.

    Two things are pinned here: that the replacement returns exactly what the scan returned,
    including its tie-breaking, and that its cost no longer scales with the number of levels. The
    size below would allocate about 13 GB the old way.
    """
    rng = np.random.default_rng(0)

    def nearest_level_scan(X, spec):
        """What the code used to do, kept as the definition of the right answer."""
        Xe = np.array(X, float, copy=True)
        for j, table in spec.ord_ranks.items():
            known = np.array(sorted(table), float)
            ranks = np.array([table[float(v)] for v in known], float)
            Xe[:, j] = ranks[np.abs(Xe[:, j][:, None] - known[None, :]).argmin(1)]
        for j, levels in spec.nom_levels.items():
            Xe[~np.isin(Xe[:, j], levels), j] = kp.UNSEEN
        return Xe

    for _ in range(60):
        n, p = int(rng.integers(5, 80)), int(rng.integers(1, 4))
        levels = int(rng.integers(1, 8))
        X = rng.integers(1, levels + 1, size=(n, p)).astype(float)
        kinds = [kp.ORDINAL if rng.random() < 0.7 else kp.NOMINAL
                 for _ in range(p)]
        spec = kp.fit_spec(X, kinds)
        # Values on levels, between them, and exactly midway — the tie case the scan resolved
        # downwards, which the replacement has to resolve the same way.
        probe = X + rng.normal(0, 0.4, X.shape)
        probe = np.where(rng.random(X.shape) < 0.25, X + 0.5, probe)
        assert np.allclose(nearest_level_scan(probe, spec), kp.encode(probe, spec),
                           equal_nan=True), "the fast path disagrees with the scan it replaced"

    n = levels = 40_000
    wide = rng.choice(np.arange(levels, dtype=float), size=(n, 1))
    spec = kp.fit_spec(wide, [kp.ORDINAL])
    started = time.time()
    out = kp.encode(wide, spec)
    assert out.shape == wide.shape
    assert time.time() - started < 30, (
        "encoding is scaling with the number of levels again — the pairwise scan is back")


def test_an_answer_list_does_not_grow_with_the_number_of_respondents():
    """How many options a question may offer used to be capped at a quarter of the sample.

    That grows with the study, which is backwards: on the 541,909-row UCI online retail file it
    permitted 135,477 "answer options", so invoice numbers (25,900 distinct), stock codes (4,070)
    and free-text product descriptions (4,223) were all clustered on as pick-any answers. The run
    had not finished after half an hour, and could not have said anything if it had — Gower scores
    two nominal answers as identical or not, so with thousands of levels nearly every pair simply
    differs.

    The ceiling is absolute now. Real option lists — brands, universities, countries — sit well
    inside it.
    """
    rng = np.random.default_rng(12)
    n = 4000
    df = pd.DataFrame({
        "respondent_id": [f"R{i}" for i in range(n)],
        "q1": rng.integers(1, 6, n),
        "q2": rng.integers(1, 6, n),
        "which_brand": rng.choice([f"Brand {i}" for i in range(20)], n),
        "home_country": rng.choice([f"Country {i}" for i in range(38)], n),
        "product_description": rng.choice([f"Item {i}" for i in range(3000)], n),
    })
    plan = sk.classify_columns(df)

    assert "which_brand" in plan["categorical"], "a 20-option question was refused"
    assert "home_country" in plan["categorical"] or "home_country" in plan["demographics"], \
        "a 38-country question was refused"
    assert "product_description" in plan["skipped"], (
        "3,000 distinct values were accepted as a question's options — the cap is scaling with "
        "the sample again")
    note = next(n_ for n_ in plan["notes"] if "product_description" in n_)
    # Not every one of the 3,000 possible values lands in 4,000 draws, so the note is checked
    # against how many distinct values are actually present.
    assert f"{df['product_description'].nunique():,}" in note and "identifier" in note, note

    # The cap must not depend on how many people answered: same columns, a much bigger study.
    big = pd.concat([df] * 10, ignore_index=True)
    big["respondent_id"] = [f"R{i}" for i in range(len(big))]
    plan_big = sk.classify_columns(big)
    assert "product_description" in plan_big["skipped"], \
        "accepted as a question purely because the study got bigger"
    assert "which_brand" in plan_big["categorical"], "a real question was dropped in a big study"


def test_a_column_flattened_by_its_outliers_is_called_out():
    """Range scaling divides by max minus min, so a couple of extreme values can flatten a column.

    Measured on the UCI online retail file: a returned order of -80,995 against a median quantity
    of 3 put **100% of 541,909 respondents inside 2% of the scale**. The segmentation then has
    almost no geometry — k-means cannot separate points that are all but coincident, so it spends
    every restart hitting its iteration limit (20 minutes on three columns) and what it returns
    describes the outliers rather than the people.

    Said, not silently corrected: whether those extremes are errors or the most interesting rows
    in the file is the reader's call, and `--scaling robust` is named for them.
    """
    rng = np.random.default_rng(21)
    n = 600
    ordinary = rng.integers(1, 6, n).astype(float)
    flattened = ordinary.copy()
    flattened[0], flattened[1] = -80_000, 80_000          # two returns, as in the real file
    df = pd.DataFrame({"respondent_id": [f"R{i}" for i in range(n)],
                       "q1": ordinary, "q2": rng.integers(1, 6, n), "quantity": flattened})
    notes = sk.classify_columns(df)["notes"]

    flagged = [n_ for n_ in notes if "within 1%" in n_]
    assert any("quantity" in n_ for n_ in flagged), (
        "a column where two values set the whole range was presented as an ordinary answer scale")
    assert not any("'q1'" in n_ or "'q2'" in n_ for n_ in flagged), \
        "an ordinary 1-5 answer scale was flagged as flattened"
    note = next(n_ for n_ in flagged if "quantity" in n_)
    assert "robust" in note, f"the note does not name the remedy: {note}"


def test_a_title_typed_above_the_column_names_is_not_the_header(tmp_path):
    """Somebody types "Customer Survey — Q1 2026 Results" in A1 and puts the column names on row 3.

    pandas takes the title as the header: one named column and the rest "Unnamed: 1", "Unnamed: 2".
    Measured on such a file, every real column name became data, the sheet read as text, and the
    survey was routed down the categorical path — 302 rows where there were 300 respondents, with
    no complaint anywhere.
    """
    rng = np.random.default_rng(3)
    n = 200
    centres = {0: [5, 1, 5, 1], 1: [1, 5, 1, 5], 2: [3, 3, 5, 5]}
    who = rng.integers(0, 3, n)
    rows = [[f"R{i}"] + [int(np.clip(round(b + rng.normal(0, 0.6)), 1, 5)) for b in centres[who[i]]]
            for i in range(n)]
    df = pd.DataFrame(rows, columns=["respondent_id", "Q1", "Q2", "Q3", "Q4"])

    path = tmp_path / "titled.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("Customer Survey — Q1 2026 Results\n\n")
        df.to_csv(f, index=False)

    got = sk._read_table(str(path))
    assert list(got.columns) == list(df.columns), f"header not found: {list(got.columns)[:4]}"
    assert len(got) == n, f"read {len(got)} rows from a {n}-respondent file"
    assert sk.auto_prepare(got)[1] == "kmeans", "rating answers were read as text"


def test_an_excel_unicode_text_export_is_readable(tmp_path):
    """UTF-16 is what Excel's "Unicode Text (*.txt)" export writes, and latin-1 decodes any byte
    without complaining — so with utf-16 missing from the attempts, such a file came back as a
    single column named 'ÿþr' full of mojibake and failed later with "no questions found", which
    tells the reader nothing about what was wrong."""
    rng = np.random.default_rng(4)
    n = 120
    df = pd.DataFrame({"respondent_id": [f"R{i}" for i in range(n)],
                       "Nöjdhet": rng.integers(1, 6, n), "Q2": rng.integers(1, 6, n),
                       "Q3": rng.integers(1, 6, n)})
    path = tmp_path / "unicode_text.csv"
    df.to_csv(path, index=False, encoding="utf-16", sep="\t")

    got = sk._read_table(str(path))
    assert len(got) == n and list(got.columns) == list(df.columns), \
        f"utf-16 export read as {got.shape} with columns {list(got.columns)[:3]}"
    assert pd.api.types.is_numeric_dtype(got["Nöjdhet"])


def test_a_select_all_question_is_split_whatever_packs_it(tmp_path):
    """Google Forms joins ticked options with a comma. A Swedish or German Excel uses a semicolon,
    because the comma is already the decimal mark. Only the comma was recognised, so
    "Spotify;Netflix" and "Netflix;Spotify" were different answers — measured, five options became
    74 pseudo-categories on a 300-person file."""
    rng = np.random.default_rng(6)
    n = 200
    apps = ["Spotify", "Netflix", "HBO", "Viaplay", "YouTube"]
    for sep in (",", ";", "|"):
        picks = [sep.join(rng.choice(apps, rng.integers(1, 4), replace=False)) for _ in range(n)]
        df = pd.DataFrame({"respondent_id": [f"R{i}" for i in range(n)],
                           "q1": rng.integers(1, 6, n), "q2": rng.integers(1, 6, n),
                           "Which do you use": picks})
        plan = sk.classify_columns(df)
        assert "Which do you use" in plan["multiselect"], f"not recognised when packed with '{sep}'"
        assert set(plan["multiselect"]["Which do you use"]) == set(apps), \
            f"wrong options for '{sep}': {plan['multiselect']['Which do you use']}"
        assert plan["multiselect_sep"]["Which do you use"] == sep


def test_more_questions_than_the_sample_can_support_is_never_called_high_confidence():
    """With many questions and few people, distances between respondents concentrate: everybody is
    roughly equidistant, real structure is diluted across the questions carrying none, and — the
    dangerous part — what survives is highly REPRODUCIBLE, because noise reproduces.

    Measured on 150 respondents answering 400 questions where only 60 carried a three-group
    signal: two groups at an Adjusted Rand Index of 0.635 against the truth, reported as **High**
    confidence on two runs out of three. Being wrong is survivable; being wrong and confident is
    the one thing this report may not do.
    """
    rng = np.random.default_rng(7)
    n, q = 120, 300
    grp = rng.integers(0, 3, n)
    signal = np.array([[5, 1, 3], [1, 5, 3], [3, 3, 5]], float)
    X = rng.normal(3, 1.0, (n, q))
    for j in range(40):
        X[:, j] = signal[grp, j % 3] + rng.normal(0, 0.6, n)
    df = pd.DataFrame(np.clip(np.rint(X), 1, 5).astype(int), columns=[f"Q{i+1}" for i in range(q)])
    df.insert(0, "respondent_id", [f"R{i}" for i in range(n)])

    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    assert r["confidence"] != "high", (
        f"{q} questions and {n} people, and the report claims high confidence — the regime where "
        "every criterion agrees on a wrong answer")
    assert "too few to pin down" in r["digest"] or "questions and only" in r["digest"], \
        "the report does not say why it is holding back"


def test_the_summary_does_not_contradict_the_tables_under_it():
    """Found by reading a report end to end as its reader, on 420 students whose three mind-sets
    the tool recovered at an Adjusted Rand Index of 0.954 — every segment genuinely real.

    Three things in the plain-language box argued with the evidence below it:

    - it said to "start with the biggest, most distinct group", and the largest segment was the one
      the table below told the reader not to build a campaign on;
    - it called a 40% share "about 1 in 3 (40%)", putting a claim and its contradiction inside one
      set of brackets;
    - the green light said the groups "are clear", five lines above a Hopkins statistic of 0.59
      described as "essentially random".
    """
    rng = np.random.default_rng(31)
    Q = ["I want to meet people outside my own course",
         "I would use an app to find people for a night out",
         "Most social apps feel fake to me",
         "I prefer meeting people through friends I already have",
         "I go out more than twice a week",
         "I like planning things rather than deciding last minute"]
    centres = [[4.6, 4.5, 2.2, 2.0, 4.4, 2.4], [2.1, 2.3, 4.4, 4.5, 2.2, 3.9],
               [3.9, 3.4, 3.2, 3.1, 3.0, 4.2]]
    rows, truth = [], []
    for g, (m, c) in enumerate(zip([170, 140, 110], centres)):
        rows.append(np.clip(np.rint(rng.normal(c, 0.75, size=(m, len(Q)))), 1, 5))
        truth += [g] * m
    df = pd.DataFrame(np.vstack(rows).astype(int), columns=Q)
    df.insert(0, "respondent_id", [f"S{i}" for i in range(len(df))])

    r = sk.run_analysis(df.to_csv(index=False).encode(),
                        cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
    digest = r["digest"]

    assert "most distinct" not in digest, (
        "the summary asserts the biggest group is the most distinct, which nothing here tested")
    assert "Do not build a campaign" not in digest, (
        "a segment is being condemned on the split direction, which every solution forces")

    # No share may be described as a fraction it is not. Checked directly on the helper, over
    # every whole percentage, rather than hoping this run happens to produce a 40% segment.
    for pct in range(1, 100):
        phrase = sk._fraction_phrase(pct / 100)
        if "1 in " in phrase:
            d = int(phrase.split("1 in ")[1].split(" ")[0])
            assert abs(pct / 100 - 1 / d) <= 0.03, f"{pct}% described as {phrase}"

    # The green light may only claim what it measured: reproducibility, not separation.
    if "🟢" in digest:
        assert "groups are clear" not in digest, (
            "the confidence light claims the groups are separated; it is built from stability")


def test_a_segment_is_not_condemned_for_dividing_when_asked_for_more_groups():
    """At any k, asking for one more group forces the solution to split something. Whichever
    segment it splits scored about 0.5 in that direction, the weaker of the two directions was
    reported, and a genuine segment was labelled 'dissolves'.

    Measured: the largest and cleanest of three real mind-sets held together perfectly under
    merging (1.00), scored 0.56 under splitting, and the report told the reader not to spend money
    on it.
    """
    para = sk.persistence_paragraph(
        {0: {"merges": 1.0, "splits": 0.51}, 1: {"merges": 1.0, "splits": 1.0}},
        ["Night-out crowd", "Homebodies"])
    assert "holds together" in para
    assert "scatters" not in para and "dissolves" not in para, (
        "a segment that survives merging intact is being called unreal because k+1 split it")
    assert "finer detail is available" in para

    # And the direction that IS evidence still works: a segment whose members scatter when groups
    # are merged was never a unit, and must be called out.
    weak = sk.persistence_paragraph({0: {"merges": 0.30}, 1: {"merges": 0.95}}, ["Fragile", "Solid"])
    assert "scatters" in weak


def _report_tables(md):
    """Every markdown table in a report, as (header cells, [row cells])."""
    out, cur = [], None
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cur is None:
                cur = [cells, []]
            elif set("".join(cells)) <= set("-: "):
                continue
            else:
                cur[1].append(cells)
        elif cur is not None:
            out.append(cur); cur = None
    if cur is not None:
        out.append(cur)
    return out


def _report_complaints(md, k, n):
    """What a careful reader would catch: the document disagreeing with its own numbers."""
    bad, tables = [], _report_tables(md)

    m = re.search(r"looked at \*\*([\d,]+) people\*\*", md)
    if m and int(m.group(1).replace(",", "")) != n:
        bad.append(f"headline says {m.group(1)} people, the run has {n:,}")
    m = re.search(r"found \*\*(\d+) (?:group|class|segment)", md)
    if m and int(m.group(1)) != k:
        bad.append(f"headline says {m.group(1)} groups, the run produced {k}")

    sizes = [t for t in tables if t[0][:1] == [""] and "n" in t[0]]
    if sizes:
        col = sizes[0][0].index("n")          # by name: columns get inserted over time
        total = sum(int(r[col]) for r in sizes[0][1])
        if total != n:
            bad.append(f"segment sizes sum to {total:,}, the run has {n:,}")
        if len(sizes[0][1]) != k:
            bad.append(f"sizes table lists {len(sizes[0][1])} groups, the run produced {k}")

    for d, pct in re.findall(r"about 1 in (\d+) \((\d+)%\)", md):
        if abs(int(pct) / 100 - 1 / int(d)) > 0.035:
            bad.append(f"'{pct}%' described as 1 in {d}")

    named = {r[0] for t in tables for r in t[1]
             if t[0][:1] == ["segment"] and not re.fullmatch(r"Segment \d+", r[0])}
    generic = {r[0] for t in tables for r in t[1]
               if t[0][:1] == ["segment"] and re.fullmatch(r"Segment \d+", r[0])}
    if named and generic and not any("suggested name" in t[0] for t in tables):
        bad.append("segments labelled both ways with no table mapping one to the other")

    if "scatters" in md and re.search(r"Start with the (biggest|largest)", md):
        bad.append("says to start with the biggest group while flagging one as scatters")
    if "🟢" in md and "essentially random" in md and "groups are clear" in md:
        bad.append("green light claims clear groups above an 'essentially random' score")

    # Two sentences about the same replication number must not disagree. Found on the categorical
    # path: a split-half of 0.577 was called "partly reproduces" immediately above "the division
    # does not survive being repeated on half the sample".
    if re.search(r"reproduces(,| well)", md) and "does not survive being repeated" in md:
        bad.append("one replication number described both as reproducing and as not surviving")
    return bad


def test_the_report_agrees_with_itself_in_every_regime():
    """The blind spot, closed.

    Three separate defects reached main while every test passed, and all three were plainly
    visible in the generated report: the k-selection fault in v1.5.2 (a table contradicting the
    prose above it), the wide-questionnaire fault (a confidence light contradicting its own
    evidence), and the persistence fault (a summary recommending the segment the table below
    condemned). The suite never caught any of them because every test asked whether the pipeline
    RAN, never whether the document it produced held together.

    This reads the report and checks it against its own numbers, across the regimes that produce
    materially different reports — real structure, none at all, overlapping, very unequal sizes,
    and a two-group answer where the persistence table has only one direction available.
    """
    def likert(a):
        return np.clip(np.rint(a), 1, 5).astype(int)

    separated = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    overlapping = np.array([[4, 2, 4, 2, 3], [2, 4, 2, 4, 3], [3, 3, 4, 4, 2]], float)
    regimes = {}
    r0 = np.random.default_rng(0)
    who = r0.integers(0, 3, 400)
    regimes["three real groups"] = likert(separated[who] + r0.normal(0, 0.5, (400, 5)))
    regimes["no structure"] = likert(r0.normal(3, 1.1, (400, 5)))
    w = r0.integers(0, 3, 400)
    regimes["overlapping"] = likert(overlapping[w] + r0.normal(0, 1.1, (400, 5)))
    w = r0.choice(3, 400, p=[0.8, 0.15, 0.05])
    regimes["very unequal sizes"] = likert(separated[w] + r0.normal(0, 0.5, (400, 5)))
    w = r0.integers(0, 2, 400)
    regimes["two groups"] = likert(separated[:2][w] + r0.normal(0, 0.5, (400, 5)))

    # The categorical path builds a different report from different machinery, and until now had
    # two of the eleven pieces of evidence the numeric path gives — so it needs checking too.
    words = ["Yes", "No", "Maybe"]
    rcat = np.random.default_rng(3)
    grp = rcat.integers(0, 3, 400)
    regimes["all multiple-choice"] = pd.DataFrame(
        {f"pick{i+1}": [words[(grp[j] + i) % 3] if rcat.random() < 0.85 else rcat.choice(words)
                        for j in range(400)] for i in range(4)})

    failures = {}
    for name, X in regimes.items():
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=[f"q{i+1}" for i in range(X.shape[1])])
        df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
        with contextlib.redirect_stdout(io.StringIO()):
            r = sk.run_analysis(df.to_csv(index=False).encode(),
                                cfg=sk.SegmentationConfig(k_min=2, k_max=5, **FAST))
        a = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
        col = "segment" if "segment" in a.columns else "class"
        complaints = _report_complaints(r["digest"], a[col].nunique(), len(a))
        if complaints:
            failures[name] = complaints
    assert not failures, "reports contradict themselves: " + json.dumps(failures, indent=2)

    # A checker that cannot fail proves nothing, so put each real defect back and confirm it is
    # caught. Every string below was in a report that shipped.
    caught = _report_complaints(
        "We looked at **400 people** and found **3 groups**:\n"
        "- **A** (about 1 in 3 (40%) of people)\n"
        "**Confidence: 🟢 High.** the groups are clear\n"
        "Hopkins statistic = **0.59** — essentially random\n"
        "Start with the biggest, most distinct group.\n"
        "\n| segment | reading |\n|:--|:--|\n| Night crowd | scatters |\n"
        "\n| segment | mean_Jaccard |\n|:--|--:|\n| Segment 0 | 0.9 |\n"
        "\n|    |   n |   share |\n|:--|--:|--:|\n| Segment 0 | 250 | 0.6 |\n",
        k=3, n=400)
    joined = " | ".join(caught)
    for expected in ("40%", "sizes sum to", "sizes table lists", "labelled both ways",
                     "essentially random", "start with the biggest"):
        assert expected in joined, f"the checker misses {expected!r}: {caught}"


def test_the_report_actually_contains_tables():
    """Every table in the report goes through DataFrame.to_markdown, which needs `tabulate`.

    It was declared an optional extra described as "prettier Markdown tables", and that was wrong
    in a way nobody would notice on a machine that had it: without tabulate the report contains no
    tables at all. Measured with it blocked, eight markdown tables become zero and eight <table>
    elements in the HTML become zero, so segment sizes, the stability checks, the centroids and the
    whole k-selection panel arrive as run-together text. Neither CI nor the app build installed the
    extra, so it was present only where it happened to be — the third capability to be declared
    optional and then be quietly missing everywhere else, after the dip test and the decimal-comma
    repair.
    """
    assert "|" in _md_probe(), (
        "DataFrame.to_markdown is not producing a markdown table — `tabulate` is missing from this "
        "environment, and the report will contain no tables")

    rng = np.random.default_rng(0)
    centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)
    who = rng.integers(0, 3, 200)
    X = np.clip(np.rint(centres[who] + rng.normal(0, 0.6, (200, 4))), 1, 5).astype(int)
    df = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(4)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(200)])
    with contextlib.redirect_stdout(io.StringIO()):
        r = sk.run_analysis(df.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))

    assert r["digest"].count("|:--") >= 4, "the report's markdown tables are not tables"
    assert r["report_html"].count("<table") >= 4, (
        "the HTML report has no tables in it — a reader gets run-together text where the segment "
        "sizes and stability checks should be")


def _md_probe():
    return sk._md(pd.DataFrame({"a": [1], "b": [2]}))


def test_a_multiple_choice_survey_gets_the_same_evidence_as_a_rating_one():
    """A survey made entirely of pick-any questions goes down the latent-class path, and that path
    was giving the reader almost nothing to judge the result by.

    Measured against the numeric report, eleven pieces of evidence to two: no split-half
    replication, no statement of which kind of segmentation this is, no table of which classes sit
    next to each other, and no explanation of the per-person fit column — even though the
    neighbours table was **already being computed** and then dropped, under a comment saying it
    existed so the categorical half would not be a poor relation.

    Split-half replication is the one that matters most: the confidence light is built from it, and
    without it a multiple-choice survey had no answer at all to "would this come back again".
    """
    rng = np.random.default_rng(11)
    n = 400
    words = ["Yes", "No", "Maybe"]
    grp = rng.integers(0, 3, n)
    df = pd.DataFrame({f"pick{i+1}": [words[(grp[j] + i) % 3] if rng.random() < 0.85
                                      else rng.choice(words) for j in range(n)]
                       for i in range(4)})
    df.insert(0, "respondent_id", [f"P{i}" for i in range(n)])

    with contextlib.redirect_stdout(io.StringIO()):
        r = sk.run_analysis(df.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
    digest = r["digest"]

    for phrase, what in [
            ("fresh sample find the same", "split-half replication"),
            ("segmentation** —", "which of the three kinds of segmentation this is"),
            ("sit next to each other", "which classes border each other"),
            ("`fit` column", "how to read the per-person fit"),
            ("mean_Jaccard", "per-class bootstrap stability")]:
        assert phrase in digest, f"the categorical report does not give the reader {what}"

    # The assignments must carry that fit, not merely mention it.
    a = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
    assert "fit" in a.columns and a["fit"].notna().all()

    # And the two sentences about replication must agree with each other — 0.577 was once
    # "partly reproduces" directly above "does not survive being repeated on half the sample".
    assert not (re.search(r"reproduces(,| well)", digest)
                and "does not survive being repeated" in digest), \
        "the report describes one replication number two contradictory ways"


def test_a_no_answer_code_in_a_follow_up_file_is_not_scored_silently(tmp_path):
    """Survey exports routinely code "no answer" as 99, 999 or -99, and the typing rule accepted
    such a value without a word.

    Nothing rejects it: it is scaled with the study's own parameters, lands far outside the space,
    and drags the respondent to whichever segment is extreme on that item. Measured on a 250-person
    follow-up with 99 in one question, **35 of the 60 affected people were put in the wrong
    segment** and agreement with the truth fell from 0.967 to 0.593.

    The confidence column already dropped sharply for them (0.34 against 0.72), so the signal was
    there — what was missing was any reason for the reader to go looking. Counted rather than
    corrected, because whether 99 means "no answer" or is a real value is a fact about the
    questionnaire, not about the data.
    """
    Q = ["I compare prices", "Quality over cost", "I wait for sales", "I try new brands"]
    centres = np.array([[5, 1, 5, 2], [1, 5, 1, 5], [3, 3, 5, 4]], float)

    def survey(n, seed):
        r = np.random.default_rng(seed)
        w = r.integers(0, 3, n)
        X = np.clip(np.rint(centres[w] + r.normal(0, 0.6, (n, 4))), 1, 5).astype(int)
        d = pd.DataFrame(X, columns=Q)
        d.insert(0, "respondent_id", [f"R{i}" for i in range(n)])
        return d, w

    train, _ = survey(400, 1)
    path = tmp_path / "train.csv"; train.to_csv(path, index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        sk.Segmenter(sk.SegmentationConfig(k_min=2, k_max=5, **FAST)).run(
            str(path), id_col="respondent_id", outdir=str(tmp_path / "out"))
    rule = json.loads((tmp_path / "out" / "typing_rule.json").read_text())

    new, truth = survey(250, 99)
    clean = sk.classify_new(rule, new, id_col="respondent_id")
    col = "answers_off_the_original_scale"
    assert col in clean.columns
    assert int(clean[col].sum()) == 0, "a clean follow-up must raise no flags at all"
    assert adjusted_rand_score(truth, clean["segment"]) > 0.9

    dirty = new.copy()
    dirty.loc[dirty.index[:60], Q[1]] = 99          # "no answer"
    dirty.loc[dirty.index[200:210], Q[3]] = -99     # the other common code
    scored = sk.classify_new(rule, dirty, id_col="respondent_id")
    flagged = scored[col] > 0

    assert set(np.flatnonzero(flagged.to_numpy())) == set(range(60)) | set(range(200, 210)), (
        "the flag does not identify exactly the respondents whose answers were off the scale")
    assert scored.loc[flagged, "confidence"].mean() < scored.loc[~flagged, "confidence"].mean(), \
        "an off-scale answer should show up as low confidence too"

    # A respondent who simply skipped a question is NOT off the scale — a blank is a blank.
    gappy = new.copy(); gappy.loc[gappy.index[:20], Q[0]] = np.nan
    assert int(sk.classify_new(rule, gappy, id_col="respondent_id")[col].sum()) == 0, \
        "a skipped answer was reported as an off-scale one"


def test_it_says_when_the_chosen_k_fails_the_cutoff_it_quotes():
    """The report calls prediction strength "the column to trust most" and quotes Tibshirani &
    Walther's 0.80. It could then recommend a k below that line without a word.

    Found on a file whose answers round onto a few tight patterns — every mind-set becomes a
    handful of well-separated satellites, so the separation indices favour a fine split. k=6 won
    the weighted vote at a prediction strength of 0.74 while k=2 scored a perfect 1.00. Both
    readings are defensible there, which is precisely why the reader should be told they disagree
    instead of being shown one number.

    The answer is unchanged; only the claim made for it is.
    """
    cols = ["k", "inertia", "silhouette", "calinski_harabasz", "davies_bouldin",
            "min_segment_share", "gap", "gap_se", "stability_ARI", "stability_ARI_sd",
            "prediction_strength", "prediction_strength_sd", "gmm_BIC", "gmm_ICL", "consensus_PAC"]
    # k=6 wins on separation and stability; its prediction strength is under the cutoff.
    rows = [
        (2, 300.0, 0.507, 200.0, 1.50, 0.42, 0.80, 0.01, 0.812, 0.05, 1.000, 0.01, 500.0, 505.0, 0.30),
        (3, 260.0, 0.563, 210.0, 1.40, 0.23, 0.82, 0.01, 0.849, 0.05, 0.871, 0.02, 480.0, 485.0, 0.20),
        (6, 180.0, 0.693, 320.0, 0.90, 0.10, 0.95, 0.01, 0.927, 0.02, 0.740, 0.05, 400.0, 405.0, 0.01),
    ]
    cfg = sk.SegmentationConfig()
    pick, rationale, _ = sk.recommend_k(pd.DataFrame(rows, columns=cols), cfg)
    assert pick == 6, "fixture assumption: the separation indices should win here"
    assert "Prediction strength at k = 6 is 0.74" in rationale, rationale[-400:]
    assert "below the 0.80" in rationale
    assert "k = 2 reaches 1.00" in rationale, "it should name the k that does clear the cutoff"

    # And it must stay quiet when the winner clears the line, or the note becomes noise.
    fine = [
        (2, 300.0, 0.507, 200.0, 1.50, 0.42, 0.80, 0.01, 0.812, 0.05, 0.990, 0.01, 500.0, 505.0, 0.30),
        (3, 260.0, 0.400, 150.0, 1.90, 0.23, 0.70, 0.01, 0.700, 0.05, 0.600, 0.02, 600.0, 605.0, 0.20),
    ]
    pick2, rationale2, _ = sk.recommend_k(pd.DataFrame(fine, columns=cols), cfg)
    assert pick2 == 2 and "Prediction strength at k" not in rationale2


def test_a_large_study_stays_within_a_sane_memory_budget(tmp_path):
    """A guard against the shape of defect that made a 48,842-person file need 11 GB.

    That one was a single line mapping each answer to its nearest known level by comparing it
    against every level at once: nothing on a five-point scale, and 19 GB per column once a
    continuous measurement had been typed as an ordinal one with tens of thousands of distinct
    values. Nothing failed — the run completed, slowly, and would have taken a 16 GB laptop down.

    The file below reproduces exactly that shape (four rating questions, one continuous
    measurement with a distinct value per respondent, one pick-any answer, which routes to the
    mixed-type path) at a size where the old code would allocate well over a gigabyte per column
    and the current code allocates none of it.

    Measured in a subprocess, because peak RSS is a high-water mark for the whole process and the
    test suite has already run a lot by this point.
    """
    n = 12_000
    script = tmp_path / "run.py"
    script.write_text(
        "import sys, resource, warnings, numpy as np, pandas as pd\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "warnings.filterwarnings('ignore')\n"
        "import segment_kmeans as sk\n"
        "r = np.random.default_rng(0)\n"
        f"n = {n}\n"
        "centres = np.array([[5,1,5,1],[1,5,1,5],[3,3,5,5]], float)\n"
        "w = r.integers(0, 3, n)\n"
        "X = np.clip(np.rint(centres[w] + r.normal(0, .6, (n, 4))), 1, 5).astype(int)\n"
        "d = pd.DataFrame(X, columns=['q1','q2','q3','q4'])\n"
        "d['spend'] = np.round(r.uniform(0, 5000, n), 2)      # a distinct value per respondent\n"
        "d['brand'] = np.where(w == 0, 'Alpha', np.where(w == 1, 'Beta', 'Gamma'))\n"
        "d.insert(0, 'respondent_id', [f'R{i}' for i in range(n)])\n"
        "sk.run_analysis(d.to_csv(index=False).encode(),\n"
        "                cfg=sk.SegmentationConfig(k_min=2, k_max=3, gap_B=4, stability_B=4,\n"
        "                                          ps_splits=3, jaccard_B=8, n_init_final=3,\n"
        "                                          n_init_search=3, run_consensus=False,\n"
        "                                          check_variable_selection=False))\n"
        "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "print(peak / (1e9 if sys.platform == 'darwin' else 1e6))\n")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-1500:]
    peak_gb = float(out.stdout.strip().splitlines()[-1])
    # 1.2 GB, chosen by measuring both sides rather than picked: the current code peaks at about
    # 0.48 GB here, and restoring the old per-level scan peaks at 2.39 GB. The first threshold
    # tried was 2.5 GB, which the OLD code passed — a guard that would have let the very defect it
    # was written for straight through.
    assert peak_gb < 1.2, (
        f"{n:,} respondents needed {peak_gb:.2f} GB, against about 0.48 GB expected. Something is "
        "allocating per respondent per level again — see kprototypes.encode, which once did "
        "exactly that and cost 11 GB on a 48,842-person file")


def test_no_chart_quietly_describes_fewer_people_than_the_study():
    """The charts exist so a reader can disagree with the write-up. That only works if they show
    everybody.

    They used to draw a random sample and say so in small print — a sample cannot falsify
    anything — and the fix was to plot one dot per distinct answer pattern with its area
    proportional to how many people share it. This checks the arithmetic actually holds: at every
    size, the dots on the map, the bars of the gorge plot and the rows of the fit chart must
    account for exactly as many people as the report claims to have analysed.

    Checked across sizes because the failure mode is silent sampling above some threshold, which is
    invisible on a small file.
    """
    centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
    for n in (300, 1500, 4000):
        rng = np.random.default_rng(3)
        who = rng.integers(0, 3, n)
        X = np.clip(np.rint(centres[who] + rng.normal(0, 0.6, (n, 5))), 1, 5).astype(int)
        df = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(5)])
        df.insert(0, "respondent_id", [f"P{i}" for i in range(n)])
        with contextlib.redirect_stdout(io.StringIO()):
            r = sk.run_analysis(df.to_csv(index=False).encode(),
                                cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))
        specs = {c["id"]: (c["spec"] if isinstance(c.get("spec"), dict) else json.loads(c["spec"]))
                 for c in r["charts"] if c.get("spec")}
        people = r["n_people"]

        if "map" in specs:
            pts = specs["map"]["points"]
            assert sum(pts["people"]) == people, (
                f"the segment map accounts for {sum(pts['people'])} of {people} respondents")
            assert specs["map"]["people"] == people
            assert len(pts["x"]) == len(pts["people"]) == len(pts["segment"]), \
                "the map's columns are different lengths, so a dot has no count or no group"
        if "gorge" in specs:
            assert sum(specs["gorge"]["counts"]) == people, (
                f"the gorge plot counts {sum(specs['gorge']['counts'])} of {people} respondents")
        if "fit" in specs:
            covered = sum(int(row["people"]) for row in specs["fit"]["rows"])
            assert covered == people, f"the fit chart covers {covered} of {people} respondents"
            assert specs["fit"]["sampled"] == 0, "the fit chart fell back to a sample"

        # And every chart that shows segments must show all of them.
        for cid in ("map", "profiles", "heatmap", "fit"):
            if cid in specs and "segments" in specs[cid]:
                assert len(specs[cid]["segments"]) == r["k"], (
                    f"the {cid} chart shows {len(specs[cid]['segments'])} of {r['k']} groups")


def test_naming_the_groups_reaches_every_file_that_names_them():
    """Once a team names its segments, every file that refers to them should use those names.

    It did not: `segment_assignments.csv` gained a `group_name` column and a `group_names.csv`
    appeared, while `group_profiles.csv` — the file describing what each group is like — still
    said "Segment 0/1/2". One thing under two names, in two files a reader opens side by side,
    which is the fault the report itself was cleaned of earlier.

    The number stays everywhere, because that is what the files join on. What was missing was the
    name beside it.
    """
    rng = np.random.default_rng(5)
    n = 240
    centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)
    who = rng.integers(0, 3, n)
    X = np.clip(np.rint(centres[who] + rng.normal(0, 0.6, (n, 4))), 1, 5).astype(int)
    df = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(4)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(n)])
    with contextlib.redirect_stdout(io.StringIO()):
        r = sk.run_analysis(df.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=4, **FAST))

    # The exported profiles must say what their first column is, or a spreadsheet shows a blank
    # heading over the segment labels.
    profiles = pd.read_csv(io.StringIO(r["files"]["group_profiles.csv"]))
    assert not str(profiles.columns[0]).startswith("Unnamed"), \
        f"the profiles file has a nameless first column: {list(profiles.columns)[:3]}"

    # Now apply names the way the app does, and check both files carry them.
    assign = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))
    groups = sorted(assign["segment"].unique())
    names = [f"Mind-set {chr(65 + i)}" for i in range(len(groups))]
    mapping = dict(zip(groups, names))

    frame = pd.read_csv(io.StringIO(r["files"]["group_profiles.csv"]))
    label = frame.columns[0]
    numbers = frame[label].astype(str).str.extract(r"(\d+)")[0].astype("Int64")
    assert numbers.notna().all(), "the profiles rows do not carry a segment number to join on"
    frame.insert(1, "suggested name", [mapping[int(v)] for v in numbers])

    assert list(frame["suggested name"]) == [mapping[g] for g in groups]
    assert len(frame) == len(groups), "the profiles file does not have a row per group"
    # And the number survives alongside the name, because that is the join key.
    assert numbers.tolist() == list(groups)


def test_the_multiple_comparison_correction_matches_the_definition():
    """The report profiles segments against every background trait, so it asks many questions of
    one dataset and must correct for that or it will find a "significant" difference in noise.

    Benjamini-Hochberg is a step-up procedure and easy to get subtly wrong — an off-by-one in the
    rank, or rejecting only the hypotheses that individually clear the line rather than everything
    up to the largest that does. Checked against the definition computed independently, over
    random p-values including ties and values sitting exactly on a rejection threshold.
    """
    def reference(pvals, alpha=0.05):
        """BH adjusted p-values, from the definition: p_adj(i) = min over j >= i of (m/j)·p(j),
        made monotone, then compared with alpha. A different route to the same answer."""
        items = sorted(pvals.items(), key=lambda kv: kv[1])
        m = len(items)
        adjusted, running = {}, 1.0
        for i in range(m, 0, -1):
            name, p = items[i - 1]
            running = min(running, (m / i) * p)
            adjusted[name] = running
        return {name: adjusted[name] <= alpha for name, _ in items}

    rng = np.random.default_rng(0)
    for trial in range(600):
        m = int(rng.integers(1, 15))
        p = rng.uniform(0, 1, m)
        if trial % 3 == 0:                       # enrich so rejections actually happen
            p[: max(1, m // 3)] = rng.uniform(0, 0.05, max(1, m // 3))
        if trial % 7 == 0 and m > 2:             # ties
            p[1] = p[0]
        names = {f"v{i}": float(v) for i, v in enumerate(p)}
        assert sk._fdr_bh(names) == reference(names), (
            f"disagrees with the definition on {sorted(round(v, 6) for v in names.values())}")

    # And the textbook behaviour, worked through by hand. m = 4, alpha = 0.05, so the thresholds
    # are 0.0125, 0.025, 0.0375, 0.05. Sorted: a=0.001, c=0.011, d=0.02, b=0.9. The first three
    # each clear their own threshold and the fourth does not, so three are rejected — note that d
    # at 0.02 is rejected although 0.02 > 0.0125, because the threshold rises with the rank. That
    # is the whole point of a step-up procedure, and getting it wrong is how this is usually got
    # wrong. (My first version of this line asserted d was NOT rejected; the code was right.)
    got = sk._fdr_bh({"a": 0.001, "b": 0.9, "c": 0.011, "d": 0.02})
    assert got == {"a": True, "c": True, "d": True, "b": False}, got

    # A hypothesis whose p-value is above its own rank threshold is still rejected when a later
    # one passes: the classic step-up case, and the one a naive per-test comparison gets wrong.
    stepup = sk._fdr_bh({"x": 0.024, "y": 0.001})               # m = 2 -> 0.025, 0.05
    assert stepup == {"x": True, "y": True}, stepup
    assert sk._fdr_bh({"only": 0.04}) == {"only": True}
    assert sk._fdr_bh({"only": 0.06}) == {"only": False}


def test_a_useless_pick_any_question_is_flagged_like_a_useless_rating():
    """Variable importance decides which questions the report tells a team to drop, so the two
    measures behind it have to agree on what "useless" looks like.

    A rating is scored by eta-squared, a share of variance. A pick-any question cannot be, because
    eta-squared on its codes would depend on the arbitrary order the answers were listed in, so it
    is scored by Cramér's V — which is correlation-like, one square away from a variance share. Left
    unsquared, a random pick-any column scores about 0.06 and clears the 0.05 near-noise floor, so
    a genuinely useless question could never be flagged as one.

    The existing coverage tests the rating path only, which is not where that fault was.
    """
    rng = np.random.default_rng(9)
    n = 600
    g = rng.integers(0, 3, n)
    centres = np.array([[5, 1], [1, 5], [3, 3]], float)
    real = np.clip(np.rint(centres[g] + rng.normal(0, 0.6, (n, 2))), 1, 5).astype(int)
    brands = np.array(["Alpha", "Beta", "Gamma"])
    raw = pd.DataFrame({
        "q_real": real[:, 0],
        "q_noise": rng.integers(1, 6, n),
        "brand_real": np.where(rng.random(n) < 0.85, brands[g], rng.choice(brands, n)),
        "brand_noise": rng.choice(brands, n),
    })
    # Code the pick-any columns the way the pipeline does, and tell it which they are.
    coded = raw.copy()
    levels = {}
    for col in ("brand_real", "brand_noise"):
        cats = sorted(raw[col].unique())
        levels[col] = {i: c for i, c in enumerate(cats)}
        coded[col] = raw[col].map({c: i for i, c in enumerate(cats)}).astype(float)
    cfg = sk.SegmentationConfig(var_kinds={"brand_real": kp.NOMINAL, "brand_noise": kp.NOMINAL,
                                           "q_real": kp.ORDINAL, "q_noise": kp.ORDINAL},
                                level_labels=levels)

    vi = sk.variable_importance(coded, g, cfg).set_index("item")
    assert vi.loc["brand_real", "measure"].startswith("Cramer"), "the pick-any path was not used"

    # THE INVARIANT, stated exactly: the number reported for a pick-any question is Cramér's V
    # SQUARED. Asserting only that noise scores low and signal scores high has no teeth here —
    # measured, a bias-corrected V already crushes random data to about 0.02, and at strong signal
    # 0.87 unsquared and 0.76 squared both look plausible beside an eta-squared of 0.90. The first
    # version of this test passed with and without the squaring, which is no test at all.
    for col in ("brand_real", "brand_noise"):
        raw_v = sk._cramers_v(coded[col].to_numpy(float), g)
        assert vi.loc[col, "eta_squared"] == round(raw_v ** 2, 3), (
            f"{col} reports {vi.loc[col, 'eta_squared']} where V squared is {raw_v ** 2:.3f} "
            f"(V itself is {raw_v:.3f}) — the two measures are one square apart and the bands do "
            "not transfer between them")

    # Where it actually bites: a question with middling association. Unsquared it looks like a
    # driver, squared it reads as the weak signal it is, and the two land in different bands.
    middling = np.where(rng.random(n) < 0.45, brands[g], rng.choice(brands, n))
    mid = pd.DataFrame({"brand_mid": pd.Series(middling).map(
        {c: i for i, c in enumerate(sorted(brands))}).astype(float)})
    v_mid = sk._cramers_v(mid["brand_mid"].to_numpy(float), g)
    assert 0.2 < v_mid < 0.75, f"fixture drifted: V = {v_mid:.3f}, retune the agreement rate"
    reported = sk.variable_importance(
        mid, g, sk.SegmentationConfig(var_kinds={"brand_mid": kp.NOMINAL},
                                      level_labels={"brand_mid": levels["brand_real"]})
    ).set_index("item").loc["brand_mid", "eta_squared"]
    assert reported == round(v_mid ** 2, 3)
    assert reported < v_mid, "the reported figure is not the squared one"

    # And the sanity checks: real questions drive, useless ones are flagged for dropping.
    assert vi.loc["brand_real", "eta_squared"] > 0.5 and "drives" in vi.loc["brand_real", "role"]
    assert vi.loc["brand_noise", "eta_squared"] < 0.05
    assert "near-noise" in vi.loc["brand_noise", "role"]
    assert vi.loc["q_noise", "eta_squared"] < 0.05


def test_a_best_worst_export_is_read_in_the_words_real_exports_use():
    """A MaxDiff set is only usable if both the best and the worst answer can be identified in it.
    Any set where they cannot is dropped — silently — so an unrecognised word in the choice column
    empties the whole file and the run fails having read nothing.

    Only the literal English "best" and "worst" were recognised. A Swedish study writing "bäst"
    and "sämst" — which is this tool's own user base, and a language it goes to lengths for
    elsewhere — lost every observation, and the failure named no cause.
    """
    items = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        u = rng.normal(0, 1, len(items))
        for s in range(8):
            shown = rng.choice(len(items), 4, replace=False)
            v = u[shown] + rng.gumbel(0, 1, 4)
            best, worst = shown[v.argmax()], shown[v.argmin()]
            for it in shown:
                rows.append({"respondent_id": f"R{i}", "set": s, "item": items[it],
                             "choice": "best" if it == best else "worst" if it == worst else ""})
    base = pd.DataFrame(rows)

    for label, mapping in [
            ("English", {"best": "best", "worst": "worst", "": ""}),
            ("Swedish", {"best": "bäst", "worst": "sämst", "": ""}),
            ("most/least", {"best": "Most", "worst": "Least", "": ""}),
            ("Norwegian", {"best": "beste", "worst": "verste", "": ""})]:
        df = base.assign(choice=base["choice"].map(mapping))
        with contextlib.redirect_stdout(io.StringIO()):
            utilities = _maxdiff.utilities_from_export(df).as_frame()
        assert utilities.shape == (60, len(items)), f"{label} export lost respondents or items"
        assert np.isfinite(utilities.to_numpy()).all(), f"{label} produced non-finite utilities"

    # Words that mean nothing here must still fail — but say why, rather than handing back a
    # sentinel. The message has to name the choice column, since that is the cause every time.
    nonsense = base.assign(choice=base["choice"].map({"best": "yes", "worst": "no", "": ""}))
    with pytest.raises(ValueError) as caught:
        with contextlib.redirect_stdout(io.StringIO()):
            _maxdiff.utilities_from_export(nonsense)
    explained = sk._explain_run_error(str(caught.value))
    assert "choice column" in explained and "best" in explained, explained
    assert not explained.startswith("_"), "the reader is being shown an internal sentinel"


def test_the_places_the_version_is_written_by_hand_all_agree():
    """A release edits the version in three separate files, and nothing was checking that all three
    moved. The failure is quiet and it survives the smoke test: the app runs perfectly and stamps
    the wrong number into every report footer and `run_manifest.json`, so a result cannot be traced
    back to the code that produced it — which is the whole reason the version is recorded.

    The first version of this test covered two of the three. `frontend/package.json` was already a
    full release behind by the time the third was noticed, which is the argument for checking every
    file that carries the number rather than the ones that came to mind.
    """
    root = Path(__file__).resolve().parent.parent
    declared = {}

    found = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    assert found, "pyproject.toml has no version line to compare against"
    declared["pyproject.toml"] = found.group(1)

    found = re.search(r'"version"\s*:\s*"([^"]+)"',
                      (root / "frontend" / "package.json").read_text())
    assert found, "frontend/package.json has no version field to compare against"
    declared["frontend/package.json"] = found.group(1)

    wrong = {f: v for f, v in declared.items() if v != sk.__version__}
    assert not wrong, (f"segment_kmeans.__version__ is {sk.__version__} but {wrong}; a release "
                       f"moved some of them and not the others")


def _homogeneous_best_worst(order, n_resp=140, n_task=9, seed=5):
    """A best-worst export from one population with a KNOWN preference order.

    Deliberately not `_simulate_maxdiff`, which plants several mind-sets: a mixture has no single
    true ranking to check the headline table against, and that table is what these tests are about.
    """
    rng = np.random.default_rng(seed)
    names = [f"item_{i:02d}" for i in range(len(order))]
    truth = np.asarray(order, dtype=float)
    rows = []
    for r in range(n_resp):
        for t in range(n_task):
            shown = rng.choice(len(names), 4, replace=False)
            u = truth[shown] + rng.gumbel(0, 1, 4)
            best, worst = shown[u.argmax()], shown[u.argmin()]
            for i in shown:
                rows.append({"respondent_id": f"R{r:04d}", "task": t,
                             "item": names[i],
                             "choice": "best" if i == best else
                                       ("worst" if i == worst else "")})
    return pd.DataFrame(rows), names, truth


def test_the_headline_ranking_recovers_the_order_the_sample_preferred():
    """The question a best-worst study is fielded to answer, which the tool used to discard.

    It scored the export, handed the utilities to the segmenter, and reported the groups — while
    never saying which items the sample actually wanted. The ranking was computed and thrown away.
    """
    md = pytest.importorskip("maxdiff")
    df, names, truth = _homogeneous_best_worst([2.0, 1.2, 0.4, -0.4, -1.2, -2.0])
    with contextlib.redirect_stdout(io.StringIO()):
        est = md.utilities_from_export(df, n_draws=1200, n_burn=400, progress=False)
    rank = est.ranking()

    want = [n for _, n in sorted(zip(-truth, names))]
    assert list(rank["item"]) == want, f"ranked {list(rank['item'])}, planted order was {want}"
    # The scores must be monotone with the planted strengths, not merely in the right order by
    # accident of sorting: a table that ranks correctly but with meaningless spacing invites
    # "twice as important" readings that the numbers do not support.
    planted = rank["item"].map(dict(zip(names, truth)))
    assert np.corrcoef(rank["utility"], planted)[0, 1] > 0.99


def test_a_ranking_the_data_cannot_support_is_reported_as_tied():
    """The honesty half. A ranking prints an order whether or not one exists, and a reader acts on
    the order, not on the standard errors nobody put in the table.

    Near-identical items on a small sample must come back flagged rather than silently ordered.
    """
    md = pytest.importorskip("maxdiff")
    # Three items essentially tied at the top, and a deliberately thin study.
    df, _names, _truth = _homogeneous_best_worst([0.55, 0.5, 0.45, -0.5, -0.55, -0.6],
                                                 n_resp=45, n_task=5, seed=11)
    with contextlib.redirect_stdout(io.StringIO()):
        est = md.utilities_from_export(df, n_draws=1200, n_burn=400, progress=False)
    rank = est.ranking()

    unresolved = [bool(v) for v in rank["separated_from_next"][:-1]]
    assert not all(unresolved), ("every neighbouring pair was called clearly separated on a study "
                                 "of 45 people with three items 0.05 apart")
    assert pd.isna(rank["separated_from_next"].iloc[-1]), (
        "the last item has nothing below it, so 'not separated' would be a claim about nothing")
    # And the report must say it in words, not only in a column a non-statistician will skim past.
    prose = sk._maxdiff_ranking_section(est)
    assert "not settled" in prose and "chance that pair really is the right way round" in prose
    # The probability itself has to reach the reader, not a verdict derived from it. Reducing it
    # gave a pair at 0.58 and a pair at 0.93 the same three words, which is what this replaced.
    assert "chance it beats the next" in prose
    unsure = [p for p in rank["prob_ahead"].dropna() if p < md.MaxDiffResult.ORDER_CERTAINTY]
    assert unsure, "the fixture stopped producing an unsettled pair; it no longer tests anything"
    assert any(f"{p * 100:.0f}%" in prose for p in unsure), (
        "no unsettled pair had its probability printed anywhere in the section")


def test_a_well_separated_ranking_is_not_hedged_into_uselessness():
    """The guard on the guard: a warning that fires on clean data would train the reader to ignore
    it, which is worse than not having it."""
    md = pytest.importorskip("maxdiff")
    df, _n, _t = _homogeneous_best_worst([2.5, 1.5, 0.5, -0.5, -1.5, -2.5], n_resp=200, n_task=10)
    with contextlib.redirect_stdout(io.StringIO()):
        est = md.utilities_from_export(df, n_draws=1200, n_burn=400, progress=False)
    prose = sk._maxdiff_ranking_section(est)
    assert "not settled" not in prose, "hedged a ranking with a full point between each item"
    assert "at least 95% certainty" in prose


def test_a_best_worst_study_can_download_what_it_was_fielded_to_measure():
    """Reporting the ranking on screen is half of it. The utilities are the asset — re-running the
    sampler to get them back is the expensive part of the whole analysis."""
    pytest.importorskip("maxdiff")
    df, names, _truth = _homogeneous_best_worst([1.8, 1.0, 0.2, -0.6, -1.2, -1.8], n_resp=90,
                                                n_task=7)
    with contextlib.redirect_stdout(io.StringIO()):
        r = sk.run_analysis(df.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))

    assert "## What matters most, overall" in r["digest"]
    assert "What matters most" in r["report_html"], "the section never reached the rendered report"

    items = pd.read_csv(io.StringIO(r["files"]["item_utilities.csv"]))
    assert set(items["item"]) == set(names)
    assert list(items["rank"]) == sorted(items["rank"]), "the export is not in ranked order"

    people = pd.read_csv(io.StringIO(r["files"]["respondent_utilities.csv"]))
    assert len(people) == 90, "one row per respondent, so a follow-up analysis needs nothing else"
    assert set(names).issubset(people.columns)

    # The download and the report must agree. Two numbers for one thing is the fault this project
    # has fixed twice already, in the report tables and in the segment names.
    top_in_table = items.sort_values("rank").iloc[0]["item"]
    assert f"**{top_in_table}** comes out strongest" in r["digest"]


def test_a_rating_grid_gets_no_ranking_section_or_utility_downloads():
    """The section is meaningless for a survey that was never a best-worst exercise, and offering
    an empty 'item_utilities.csv' would be worse than offering nothing."""
    rng = np.random.default_rng(4)
    cen = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3]], float)
    w = rng.integers(0, 2, 160)
    X = np.clip(np.rint(cen[w] + rng.normal(0, .6, (160, 5))), 1, 5).astype(int)
    grid = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(5)])
    grid.insert(0, "respondent_id", [f"P{i}" for i in range(160)])

    with contextlib.redirect_stdout(io.StringIO()):
        r = sk.run_analysis(grid.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))
    assert "What matters most, overall" not in r["digest"]
    assert "item_utilities.csv" not in r["files"]
    assert "respondent_utilities.csv" not in r["files"]


def test_every_file_the_app_can_hand_over_has_a_human_label():
    """The download chips read from a map in the TypeScript, and a file with no entry there falls
    through to its raw filename — `respondent_utilities.csv` sitting in a row of plain-English
    buttons. It has happened before, to the two files the interface itself creates, and it happened
    again the moment this release added two more.

    Checking it from the Python side is the only place both halves are visible: the names are
    decided here and rendered there, and nothing else crosses that boundary to notice a mismatch.
    """
    pytest.importorskip("maxdiff")
    df, _names, _truth = _homogeneous_best_worst([1.5, 0.8, 0.1, -0.6, -1.0, -1.5], n_resp=70,
                                                 n_task=6)
    with contextlib.redirect_stdout(io.StringIO()):
        produced = set(sk.run_analysis(df.to_csv(index=False).encode(),
                                       cfg=sk.SegmentationConfig(k_min=2, k_max=3,
                                                                 **FAST))["files"])
    # The two the interface creates after the fact, which no analysis run returns.
    produced |= {"group_names.csv", "scored_new_people.csv"}

    labels = (Path(__file__).resolve().parent.parent
              / "frontend" / "src" / "lib" / "labels.ts").read_text()
    block = re.search(r"DOWNLOAD_LABEL[^{]*\{(.*?)\n\}", labels, re.S)
    assert block, "could not find DOWNLOAD_LABEL in labels.ts — has it been renamed?"
    labelled = set(re.findall(r"['\"]([\w.]+\.(?:csv|json))['\"]\s*:", block.group(1)))

    missing = sorted(produced - labelled)
    assert not missing, (f"{missing} can be downloaded but has no entry in DOWNLOAD_LABEL, so the "
                         f"button will show the raw filename")


def test_the_probability_is_read_from_the_right_pair_of_items():
    """`ranking()` sorts the items but the posterior draws stay in their original column order, so
    the columns must be reordered to match before any pair is compared.

    Getting that wrong is silent: the table still looks right, every probability is in [0, 1], and
    the numbers are simply about the wrong pairs. So this fixes draws by hand, in an order where
    the ranking is NOT the storage order, and checks the arithmetic against values worked out
    independently of the implementation.
    """
    md = pytest.importorskip("maxdiff")
    # Stored order a, b, c — ranked order b, c, a, so a correct reordering is not the identity.
    draws = np.array([[0.0, 2.0, 1.0],
                      [0.0, 2.0, 1.0],
                      [0.0, 1.0, 2.0],
                      [0.0, 2.0, 1.0]])
    res = md.MaxDiffResult(
        utilities=np.zeros((1, 3)), item_names=["a", "b", "c"], respondent_ids=["R1"],
        population_mean=draws.mean(axis=0), acceptance_rate=0.3, n_draws=4, n_burn=0,
        population_draws=draws)
    rank = res.ranking()

    assert list(rank["item"]) == ["b", "c", "a"], "the fixture no longer ranks out of storage order"
    # P(b > c) across the four draws: 2>1, 2>1, 1>2, 2>1 -> three of four.
    assert rank.loc[0, "prob_ahead"] == pytest.approx(0.75)
    # P(c > a): 1>0, 1>0, 2>0, 1>0 -> all four.
    assert rank.loc[1, "prob_ahead"] == pytest.approx(1.0)
    assert pd.isna(rank.loc[2, "prob_ahead"]), "the last item has nothing below it to beat"
    # And the 95% line is applied to those probabilities, not to anything else.
    assert bool(rank.loc[0, "separated_from_next"]) is False      # 0.75 is under the line
    assert bool(rank.loc[1, "separated_from_next"]) is True       # 1.00 clears it


def test_the_opening_sentence_does_not_claim_more_than_the_table_below_it():
    """A section that announces a winner and then explains that nothing was separated is the
    report arguing with itself — the exact fault fixed twice before, in the confidence wording and
    in the persistence table.

    On items with no real differences the ranking still has a first row, because sorting noise
    produces an order. The prose must not present that row as a finding.
    """
    md = pytest.importorskip("maxdiff")

    flat, _n, _t = _homogeneous_best_worst([0.0] * 6, n_resp=60, n_task=6, seed=2)
    with contextlib.redirect_stdout(io.StringIO()):
        est = md.utilities_from_export(flat, n_draws=1200, n_burn=400, progress=False)
    assert not any(bool(v) for v in est.ranking()["separated_from_next"][:-1]), (
        "the fixture separated something; it can no longer test the no-differences wording")
    prose = sk._maxdiff_ranking_section(est)
    assert "did not separate these items at all" in prose
    assert "should not be read as a ranking" in prose
    assert "comes out strongest" not in prose, "named a winner on data with no differences in it"

    clean, _n2, _t2 = _homogeneous_best_worst([2.0, 1.0, 0.0, -1.0, -2.0, -3.0], n_resp=200,
                                              n_task=10, seed=3)
    with contextlib.redirect_stdout(io.StringIO()):
        est2 = md.utilities_from_export(clean, n_draws=1200, n_burn=400, progress=False)
    prose2 = sk._maxdiff_ranking_section(est2)
    assert "comes out strongest" in prose2, "hedged a ranking with a full point between each item"
    assert "did not separate these items" not in prose2


def test_the_ai_digest_stays_aggregate_on_a_best_worst_study_too():
    """The privacy guarantee covered only rating grids, and best-worst is a different route.

    A MaxDiff export is read by a different reader, rebuilt into a different frame — respondent
    ids move through the INDEX of the utilities table rather than a column — and now carries a
    report section that did not exist when the original guarantee was written. None of that is
    exercised by the rating-grid test, so the claim "only aggregates are transmitted" was unproven
    for exactly the newest path.
    """
    pytest.importorskip("maxdiff")
    rng = np.random.default_rng(19)
    items = ["Price", "Speed", "Support", "Design", "Range", "Eco"]
    truth = np.array([1.5, 0.9, 0.3, -0.3, -0.9, -1.5])
    ids = [f"RESPONDENT-UNIQUE-{i:04d}" for i in range(90)]
    secrets = [f"my-secret-comment-{i:04d}" for i in range(90)]
    rows = []
    for r in range(90):
        for t in range(7):
            shown = rng.choice(len(items), 4, replace=False)
            u = truth[shown] + rng.gumbel(0, 1, 4)
            best, worst = shown[u.argmax()], shown[u.argmin()]
            for i in shown:
                rows.append({"respondent_id": ids[r], "task": t, "item": items[i],
                             "choice": "best" if i == best else
                                       ("worst" if i == worst else ""),
                             "open_feedback": secrets[r]})
    with contextlib.redirect_stdout(io.StringIO()):
        r = sk.run_analysis(pd.DataFrame(rows).to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))

    payload = r["digest"]
    assert [x for x in ids if x in payload] == [], "respondent identifiers reached the AI payload"
    assert [c for c in secrets if c in payload] == [], "free text reached the AI payload"
    # Not empty-by-accident: the aggregate this study exists to produce IS in there, and the item
    # names are questionnaire wording rather than anything belonging to a person.
    assert "What matters most" in payload
    assert all(i in payload for i in items)


def _real_world_shaped_export(n_resp=60, seed=3):
    """A best-worst export in the shape real published data actually takes.

    Column names and coding copied from the `bwsTools` example dataset (350 respondents asked
    which issues facing the country matter most and least): the item column is called `issue`, the
    set column `block`, the choice column `value`, and the pick is coded **1 / -1 / 0** rather than
    the words best and worst. That file was refused outright before this shape was handled.
    """
    rng = np.random.default_rng(seed)
    issues = ["healthcare", "economy", "education", "guns", "taxes", "crime"]
    strength = np.array([2.0, 1.5, 0.5, -0.4, -1.2, -2.0])
    rows = []
    for person in range(n_resp):
        for block in range(8):
            shown = rng.choice(len(issues), 4, replace=False)
            u = strength[shown] + rng.gumbel(0, 1, 4)
            best, worst = shown[u.argmax()], shown[u.argmin()]
            for i in shown:
                rows.append({"id": person, "block": block, "issue": issues[i],
                             "value": 1 if i == best else (-1 if i == worst else 0)})
    return pd.DataFrame(rows), issues, strength


def test_a_best_worst_export_coded_with_numbers_is_read():
    """Real published best-worst data codes the pick as 1 / -1 / 0, and names its columns for the
    subject rather than for the method. Only the English words best/worst were recognised, and only
    a fixed alias list of column names, so a genuine dataset was refused with an error naming no
    cause."""
    md = pytest.importorskip("maxdiff")
    df, issues, strength = _real_world_shaped_export()

    assert md.looks_like_maxdiff(df), "a real-world best-worst export was not recognised as one"
    with contextlib.redirect_stdout(io.StringIO()):
        est = md.utilities_from_export(df, n_draws=1200, n_burn=400, progress=False)
    assert len(est.respondent_ids) == 60, "respondents were dropped while reading a valid file"

    order = list(est.ranking()["item"])
    assert order == [n for _, n in sorted(zip(-strength, issues))], (
        f"read the file but recovered the wrong order: {order}")


def test_numeric_choice_coding_is_only_read_when_it_is_unambiguous():
    """1 and -1 mean best and worst. 1 and 2 might mean best and worst, or first and second
    choice, or an ordinary two-point rating — and inventing preferences out of a number nobody
    said was a preference is worse than refusing the file."""
    md = pytest.importorskip("maxdiff")
    df, _issues, _s = _real_world_shaped_export(n_resp=20)

    recoded = df.copy()
    recoded["value"] = recoded["value"].map({1: 1, -1: 2, 0: 0})
    assert not md.looks_like_maxdiff(recoded), (
        "read 1/2 as best/worst, which is a guess about what a number means")


def test_widening_the_column_aliases_did_not_make_detection_fire_on_ordinary_surveys():
    """`value`, `code` and `issue` had to be accepted to read real exports, and those words appear
    in plenty of tables that are nothing to do with best-worst. Detection therefore reads what is
    IN the choice column, not only what it is called — a false positive here would put an ordinary
    survey through a preference sampler and return confident nonsense."""
    md = pytest.importorskip("maxdiff")
    rng = np.random.default_rng(8)

    # Every column name matches, but `value` holds ordinary 1-5 ratings.
    ratings = pd.DataFrame({"id": rng.integers(0, 50, 300), "block": rng.integers(0, 5, 300),
                            "issue": rng.choice(["a", "b", "c"], 300),
                            "value": rng.integers(1, 6, 300)})
    assert not md.looks_like_maxdiff(ratings), "a table of 1-5 ratings was read as best-worst data"

    # And the plain rating grid, which has none of the columns.
    grid = pd.DataFrame({"respondent_id": ["R1", "R2"], "q1": [4, 2], "q2": [1, 5]})
    assert not md.looks_like_maxdiff(grid)

    # But a best-worst file whose choice column is EMPTY is still a best-worst file — a broken one.
    # Demanding recognisable marks unconditionally sent it down the rating-grid path instead, where
    # the reader's specific "no complete sets" message is replaced by generic advice about columns.
    blank = pd.DataFrame({"respondent_id": ["R1", "R1"], "set": [1, 1],
                          "item": ["a", "b"], "choice": ["", ""]})
    assert md.looks_like_maxdiff(blank), (
        "an empty best-worst export is no longer recognised, so the reader cannot explain itself")


def test_a_best_worst_file_missing_a_column_is_explained_not_signalled():
    """The sentinel reached a user verbatim — "Technical detail: _MAXDIFF_MISSING:item" — next to
    generic advice to check the file has one row per person, which is the opposite of what a
    best-worst export looks like. Found by feeding a real published dataset."""
    md = pytest.importorskip("maxdiff")
    df, _issues, _s = _real_world_shaped_export(n_resp=10)
    with pytest.raises(ValueError) as caught:
        with contextlib.redirect_stdout(io.StringIO()):
            md.utilities_from_export(df.rename(columns={"issue": "subject_matter"}))

    explained = sk._explain_run_error(str(caught.value))
    assert "_MAXDIFF" not in explained, "an internal sentinel is being shown to the reader"
    assert "best-worst" in explained and "one row per item" in explained.lower()
    assert "one row per person" not in explained.split("not one row per person")[0], (
        "still telling the reader a best-worst file should have one row per person")


def _wide_best_worst(n_resp=80, seed=4):
    """A best-worst export the way Qualtrics and Sawtooth write one: a row per PERSON, a column
    per (task, item), and a small code in each cell — commonly 3 for the item picked best, 1 for
    worst, 2 for the others shown."""
    rng = np.random.default_rng(seed)
    items = ["Price", "Speed", "Support", "Design", "Range", "Eco"]
    strength = np.array([2.0, 1.2, 0.4, -0.4, -1.2, -2.0])
    rows = []
    for person in range(n_resp):
        rec = {"ResponseId": f"R_{person:04d}"}
        for task in range(1, 7):
            shown = rng.choice(len(items), 4, replace=False)
            u = strength[shown] + rng.gumbel(0, 1, 4)
            best, worst = shown[u.argmax()], shown[u.argmin()]
            for i in shown:
                rec[f"Q{task}_{i + 1}...-{items[i]}"] = 3 if i == best else (1 if i == worst else 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def test_a_best_worst_export_saved_one_row_per_person_is_refused_not_guessed():
    """The silent failure this catches is the worst kind the tool can produce.

    Qualtrics and Sawtooth write MaxDiff wide, and nothing in such a file says it is a preference
    exercise. It was therefore read as an ordinary rating grid and the response CODES were
    clustered as if they were scores: an 80-person export came back as two confident segments with
    no warning anywhere.

    It is refused rather than read because the layout is recoverable but the POLARITY is not.
    Whether 3 means best or 1 means best is a fact about how the survey was built, and guessing it
    would invert every ranking the tool then produced. `choicetools`, which does read these files,
    makes the analyst state it for the same reason.
    """
    pytest.importorskip("maxdiff")
    wide = _wide_best_worst()
    with pytest.raises(ValueError) as caught:
        with contextlib.redirect_stdout(io.StringIO()):
            sk.run_analysis(wide.to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))

    explained = sk._explain_run_error(str(caught.value))
    assert "_MAXDIFF" not in explained, "an internal sentinel is being shown to the reader"
    # The message has to be actionable: name the problem, and show the shape that works.
    assert "one row per" in explained.lower()
    assert "respondent_id, task, item, choice" in explained
    assert "upside down" in explained, "does not say why guessing the codes would be dangerous"


@pytest.mark.parametrize("label, build", [
    # Every one of these is an ORDINARY survey with Qualtrics-style matrix naming. Refusing any of
    # them would be worse than the defect being fixed: a valid file turned away.
    ("5-point matrix", lambda r: {f"Q{q}_{i}": r.integers(1, 6, 300)
                                  for q in (1, 2, 3) for i in range(1, 6)}),
    # Three points is the dangerous one: "exactly one lowest and one highest" happens by chance.
    ("3-point matrix", lambda r: {f"Q{q}_{i}": r.integers(1, 4, 300)
                                  for q in (1, 2, 3) for i in range(1, 5)}),
    ("binary pick-any", lambda r: {f"Q{q}_{i}": r.integers(0, 2, 300)
                                   for q in (1, 2, 3) for i in range(1, 6)}),
    ("0-10 sliders", lambda r: {f"S{q}_{i}": r.integers(0, 11, 300)
                                for q in (1, 2, 3) for i in range(1, 5)}),
])
def test_the_wide_guard_does_not_turn_away_ordinary_matrix_surveys(label, build):
    md = pytest.importorskip("maxdiff")
    df = pd.DataFrame(build(np.random.default_rng(2)))
    assert not md.looks_like_wide_best_worst(df), (
        f"an ordinary {label} with matrix column names was refused as a best-worst export")


def test_an_unsettled_estimate_says_so_instead_of_looking_confident():
    """The sampler's mixing gets worse as a study grows, and nothing in its output reveals it.

    Measured before this was added: 300 respondents and 12 items produced four independent chains
    that disagreed at R-hat 1.13, while the tool reported utilities and 95% intervals with no hint
    that anything was unsettled. An MCMC estimate that has not converged still yields a tidy
    ranking and a tight-looking interval — that is exactly why it needs saying out loud.
    """
    md = pytest.importorskip("maxdiff")
    df, _names, _truth = _homogeneous_best_worst([1.5, 0.9, 0.3, -0.3, -0.9, -1.5], n_resp=60,
                                                 n_task=6)
    # A deliberately truncated chain: too short to settle, which is the state being reported on.
    with contextlib.redirect_stdout(io.StringIO()):
        bad = md.utilities_from_export(df, n_draws=140, n_burn=100, thin=1, progress=False)
    assert bad.rhat is not None, "no convergence diagnostic was computed at all"

    if bad.converged is False:                       # what a short chain should look like
        prose = sk._maxdiff_ranking_section(bad)
        assert "have not fully settled" in prose
        assert f"{bad.rhat:.2f}" in prose, "the diagnostic is mentioned but its value is not"

    # And a chain long enough to settle must NOT carry the warning: a caveat that fires on good
    # data teaches the reader to ignore it.
    with contextlib.redirect_stdout(io.StringIO()):
        good = md.utilities_from_export(df, n_draws=6000, n_burn=2000, progress=False)
    assert good.converged is True, f"a full-length chain still reports R-hat {good.rhat:.3f}"
    assert "have not fully settled" not in sk._maxdiff_ranking_section(good)


def test_sampling_length_is_chosen_by_measurement_not_by_a_constant():
    """A fixed 6,000 draws was ample for a small study and not enough for a larger one, and the
    shortfall was silent. The chain now grows until split-R-hat says it has settled.

    The escalation must not fire on studies that do not need it — that would make every ordinary
    run several times slower for nothing.
    """
    md = pytest.importorskip("maxdiff")
    small, _n, _t = _homogeneous_best_worst([1.5, 0.9, 0.3, -0.3, -0.9, -1.5], n_resp=45, n_task=5)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = md.utilities_from_export(small, progress=False)
    assert res.converged is True and res.rhat < md.RHAT_TARGET
    assert "sampling" not in out.getvalue().lower() or "instead of" not in out.getvalue(), (
        "a small study escalated when it did not need to, which costs every run time for nothing")

    # A caller that pins its own length means it, and must not be silently overridden.
    with contextlib.redirect_stdout(io.StringIO()):
        pinned = md.utilities_from_export(small, n_draws=800, n_burn=300, progress=False)
    assert pinned.n_draws == 800, "an explicitly requested chain length was overridden"
    assert pinned.rhat is not None, "a pinned run still needs its diagnostic reported"


def test_the_tool_runs_on_a_machine_with_no_locale_set(tmp_path):
    """A Linux box with no LANG — a container, a cron job, a minimal CI image — reports its
    encoding as ASCII. The report opens with a red/amber/green confidence circle and is full of
    em-dashes and Nordic characters, so the FIRST thing printed used to raise UnicodeEncodeError
    and destroy a run that had already finished computing:

        'ascii' codec can't encode character '\\U0001f534' in position 77

    Nothing on macOS shows this, because macOS always reports UTF-8. It is exactly the class of
    defect that survives every local test and fails on somebody else's machine, so it is tested
    the only way that means anything: by running the real command with the locale stripped out.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("LANG", "LC_ALL", "LC_CTYPE", "PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONCOERCECLOCALE"] = "0"          # stop Python quietly upgrading C to UTF-8 for us
    env["PYTHONUTF8"] = "0"
    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(root / "segment_kmeans.py"),
         str(root / "examples" / "example_survey.csv"), "--outdir", str(out)],
        capture_output=True, text=True, env=env, cwd=str(root), timeout=900)

    assert proc.returncode == 0, (
        f"the tool died under a bare locale:\n{proc.stdout[-1500:]}\n{proc.stderr[-800:]}")
    assert "codec can't encode" not in (proc.stdout + proc.stderr)

    # And the FILES must be UTF-8 whatever the console could manage: a report is read later, by a
    # browser, on a different machine, and mojibake there would be permanent.
    report = out / "segmentation_report.md"
    assert report.exists(), "no report was written"
    text = report.read_bytes().decode("utf-8")          # raises if it is not valid UTF-8
    assert any(ch in text for ch in "🔴🟡🟢"), (
        "the confidence light did not survive into the report, so the characters were stripped "
        "rather than written")


# --------------------------------------------------------------------------------------------
# Study planner: how many respondents does a planned survey need?
# --------------------------------------------------------------------------------------------

def _plan_fixture(moderate_hits, subtle_hits=(0, 0, 0), sizes=(100, 200, 400), wrong_at=None):
    """A plan dict shaped like `plan_study` returns, without paying for the simulations.

    The advice functions are where the reasoning lives and they are pure, so they can be checked
    against hand-made evidence in milliseconds. Running the real sweep to test a sentence would
    cost minutes and would test the sampler instead of the sentence.
    """
    import planner
    cells = []
    for regime, hits in (("obvious", (4,) * len(sizes)), ("moderate", moderate_hits),
                         ("subtle", subtle_hits)):
        for n_people, right in zip(sizes, hits):
            cells.append({"regime": regime,
                          "separation": dict(planner.REGIMES)[regime],
                          "n_people": n_people, "runs": 4, "right_k": right,
                          "hit_rate": right / 4, "mean_ari": 0.5,
                          "confidently_wrong": 1 if wrong_at == n_people else 0})
    return {"cells": cells, "n_questions": 6, "n_segments": 3, "scale": 5,
            "sizes": list(sizes), "seeds": 4}


def test_the_planner_recommends_the_smallest_sample_that_actually_works():
    """The recommendation is the point of the whole exercise, so it is checked against evidence
    where the right answer is known by construction rather than against a live simulation."""
    planner = pytest.importorskip("planner")
    # Moderately distinct segments start being found reliably at 200.
    advice = planner.recommend(_plan_fixture(moderate_hits=(2, 4, 4)))
    assert advice["recommended_n"] == 200, "did not pick the smallest size that met the bar"

    # Nothing works at any size: it must say so rather than recommending the largest tried.
    none_work = planner.recommend(_plan_fixture(moderate_hits=(1, 2, 2)))
    assert none_work["recommended_n"] is None


def test_the_planner_says_when_more_people_would_not_help():
    """The most valuable thing it can say. If the differences are below the method's resolution,
    a bigger sample measures a weak signal more precisely — it does not make it strong — and
    someone about to spend a budget needs to hear that rather than a number."""
    planner = pytest.importorskip("planner")
    hopeless = _plan_fixture(moderate_hits=(2, 4, 4), subtle_hits=(0, 0, 0))
    prose = planner.render(hopeless)
    assert "will not rescue subtle differences" in prose
    assert "sharper questions" in prose

    # And when subtle IS reachable, that warning must not fire — a caveat that always appears
    # teaches the reader to skip it.
    reachable = _plan_fixture(moderate_hits=(2, 4, 4), subtle_hits=(0, 4, 4))
    assert "will not rescue subtle differences" not in planner.render(reachable)


def test_the_planner_warns_where_the_tool_can_be_confidently_wrong():
    """Measured while building this: at 100 people with subtle differences, roughly one simulated
    study in ten reported the WRONG number of segments and still called it high confidence. The
    tool judges whether a grouping REPRODUCES, and on heavily overlapping data a merged pair
    reproduces perfectly well — so this is a limit to disclose, not a contradiction to hide."""
    planner = pytest.importorskip("planner")
    prose = planner.render(_plan_fixture(moderate_hits=(2, 4, 4), wrong_at=100))
    assert "WRONG number of segments" in prose and "100 people" in prose
    assert "provisional" in prose
    # Silent when it did not happen.
    assert "WRONG number of segments" not in planner.render(_plan_fixture(moderate_hits=(2, 4, 4)))


def test_the_planner_never_promises_to_know_whether_your_segments_exist():
    """The one thing it cannot do, and the thing a sample-size number most invites people to
    assume. It must be stated in every rendering, not only when the news is bad."""
    planner = pytest.importorskip("planner")
    for hits in [(4, 4, 4), (0, 0, 0), (2, 4, 4)]:
        prose = planner.render(_plan_fixture(moderate_hits=hits))
        assert "cannot tell you whether your segments exist" in prose


def test_planted_separation_behaves_the_way_the_regimes_claim():
    """The three regimes are only meaningful if they are actually ordered in difficulty. This
    checks the DATA rather than the pipeline — cheap, and it catches a generator that silently
    stopped varying separation, which would make every column of the table identical."""
    planner = pytest.importorskip("planner")
    from sklearn.metrics import silhouette_score
    scores = []
    for _label, separation in planner.REGIMES:
        frame, truth = planner.simulate_study(separation, 300, seed=1)
        items = frame.drop(columns=["respondent_id"]).to_numpy(float)
        scores.append(silhouette_score(items, truth))
    assert scores[0] > scores[1] > scores[2], (
        f"the regimes are not ordered by how separable they are: {scores}")
    assert scores[0] > 0.3, "the 'obvious' regime is not obviously separated"


def test_the_planner_runs_end_to_end_on_a_small_sweep():
    """One real sweep, kept deliberately tiny, so the wiring between the simulator, the pipeline
    and the renderer is exercised rather than assumed."""
    planner = pytest.importorskip("planner")
    plan = planner.plan_study(n_questions=5, n_segments=3, sizes=(120,), seeds=1)
    assert len(plan["cells"]) == len(planner.REGIMES)
    obvious = next(c for c in plan["cells"] if c["regime"] == "obvious")
    assert obvious["right_k"] == 1, "failed to find three clearly separated segments"
    assert "Planning a study" in planner.render(plan)


def test_the_planted_separation_is_the_separation_actually_delivered():
    """The planner's whole vocabulary rests on this number meaning what it says.

    The first version used the figure directly as a step size and DOCUMENTED it as Cohen's d. It is
    not: the centre pattern rotates, so one adjacent pair differs by a single step on some
    questions and by two on others, and the mean difference overshoots. Measured, asking for 2.0
    delivered 2.83 at three segments and 3.11 at four — and the module docstring, the README and
    the changelog all inherited the wrong figure.
    """
    planner = pytest.importorskip("planner")
    for n_segments in (2, 3, 4):
        for asked in (1.0, 0.6):
            frame, truth = planner.simulate_study(asked, 4000, n_questions=6,
                                                  n_segments=n_segments, seed=0)
            items = frame.drop(columns=["respondent_id"]).to_numpy(float)
            means = np.array([items[truth == c].mean(axis=0) for c in range(n_segments)])
            within = np.mean([items[truth == c].std(axis=0, ddof=1) for c in range(n_segments)])
            gaps = [np.abs(means[c] - means[c + 1]).mean() for c in range(n_segments - 1)]
            delivered = float(np.mean(gaps)) / within
            assert abs(delivered - asked) / asked < 0.2, (
                f"asked for d={asked} with {n_segments} segments and the data actually has "
                f"d={delivered:.2f}")


def test_a_design_that_cannot_fit_the_answer_scale_is_reported_not_clipped():
    """Five segments two standard deviations apart do not fit on a 1-5 scale — there is nowhere to
    put the outer ones. Clipping them to the ends produced a study far less separable than
    requested while the table still called it "obvious", which is a wrong answer wearing a
    confident label. It is a fact about the answer scale, and no sample size changes it."""
    planner = pytest.importorskip("planner")
    with pytest.raises(ValueError) as caught:
        planner.simulate_study(2.0, 200, n_questions=6, n_segments=5, scale=5)
    assert "_PLAN_TOO_TIGHT" in str(caught.value)

    # And the sweep must survive it — reported as not-applicable, with the reason in words.
    plan = planner.plan_study(n_segments=5, sizes=(120,), seeds=1)
    impossible = [c for c in plan["cells"] if c.get("impossible")]
    assert impossible, "a design that cannot fit the scale was silently simulated anyway"
    prose = planner.render(plan)
    assert "n/a" in prose and "nowhere on the scale" in prose
    assert "not of your sample size" in prose


def test_the_recommendation_is_not_a_size_that_only_got_lucky():
    """Recovery is not monotonic in sample size, so "the first size that clears the bar" can name
    one that got lucky while larger samples do worse above it.

    The rule has to fail in two directions at once, which is why both cases are here. Demanding
    that EVERY larger size also clears the bar looked right and was too strict: with a handful of
    repeats per cell, one unlucky draw anywhere above the answer suppressed the recommendation and
    the tool announced that no sample size was reliable for a perfectly good design. Measured in
    the app's cheaper sweep, that is exactly what it did.
    """
    planner = pytest.importorskip("planner")

    # Genuinely lucky: it passes once and everything above it collapses. Must be refused.
    lucky = _plan_fixture(moderate_hits=(4, 0, 0), sizes=(100, 200, 400))
    assert planner.recommend(lucky)["recommended_n"] != 100, (
        "recommended a size that passed once while every larger sample failed")

    # One dip between two clean results is sampling noise, not a reason to withhold an answer.
    noisy = _plan_fixture(moderate_hits=(4, 2, 4), sizes=(100, 200, 400))
    assert planner.recommend(noisy)["recommended_n"] == 100, (
        "a single unlucky cell above the answer suppressed the recommendation entirely")

    sustained = _plan_fixture(moderate_hits=(2, 4, 4), sizes=(100, 200, 400))
    assert planner.recommend(sustained)["recommended_n"] == 200


def test_the_default_sample_sizes_cover_where_the_answer_changes():
    """The first version swept 100 to 800 and so measured almost entirely inside the flat part of
    the curve: every regime gave the same result in six of eight columns, and the table looked
    authoritative while saying nothing. Recovery actually turns over between roughly 40 and 150."""
    planner = pytest.importorskip("planner")
    assert min(planner.DEFAULT_SIZES) <= 75, (
        "the sweep starts above the range where sample size actually decides the answer")
    assert any(n <= 150 for n in planner.DEFAULT_SIZES)
    assert planner.DEFAULT_SEEDS >= 5, (
        "fewer than five repeats cannot separate 'nearly always' from 'usually', which is exactly "
        "the distinction the recommendation turns on")


# --------------------------------------------------------------------------------------------
# TURF: which few items to launch
# --------------------------------------------------------------------------------------------

def test_turf_picks_across_crowds_where_a_ranking_would_not():
    """The reason TURF exists at all. A ranking says what people like ON AVERAGE; if tastes divide,
    the three best-liked items can all appeal to the same crowd while everybody else is left with
    nothing. Reach counts people rather than preference, so it should sacrifice a well-liked item
    for one that covers the crowd nobody else is serving."""
    tf = pytest.importorskip("turf")
    rng = np.random.default_rng(3)
    names = ["A", "B", "C", "D", "E"]
    utilities = rng.normal(0, 0.4, (600, 5))
    majority = rng.random(600) < 0.6
    utilities[majority, 0] += 3.0            # the big crowd loves A, B and C
    utilities[majority, 1] += 2.8
    utilities[majority, 2] += 2.6
    utilities[~majority, 3] += 3.2           # everyone else only wants D
    utilities[~majority, 4] += 3.0

    average_favourites = set(np.argsort(-utilities.mean(axis=0))[:2])
    assert average_favourites == {0, 1}, "the fixture no longer has A and B as the top-rated pair"

    result = tf.turf(utilities, names, size=2, splits=10)
    assert set(result["indices"]) & {3, 4}, (
        f"picked {result['items']} — every choice serves the same crowd, which is the mistake "
        "reach is supposed to prevent")


def test_turf_reports_how_much_of_its_headline_is_luck():
    """A maximum taken over every combination flatters whichever one suited this sample. Measured
    on a hundred people over twenty items, the in-sample figure overstates reach by more than
    twenty points — the difference between a launch decision and a mistake — so the holdout is not
    an optional refinement, it is the number worth quoting."""
    tf = pytest.importorskip("turf")
    rng = np.random.default_rng(1)
    utilities = rng.normal(0, 1.0, (100, 20))          # pure taste noise: every lead is luck
    result = tf.turf(utilities, [f"i{j}" for j in range(20)], size=3, splits=25)

    assert result["holdout_reach"] < result["in_sample_reach"], (
        "the holdout matched the in-sample figure on pure noise, which cannot be right")
    assert result["optimism"] > 0.05, (
        f"optimism measured at only {result['optimism']:.3f} on noise, where the entire apparent "
        "advantage of the winning combination is sampling luck")


def test_turf_search_is_exact_when_it_can_afford_to_be():
    """Exhaustive where the binomial allows it, greedy beyond — and the result says which, because
    a reader is entitled to know whether "best" means best or merely good."""
    tf = pytest.importorskip("turf")
    rng = np.random.default_rng(0)
    small = tf.turf(rng.normal(0, 1, (200, 8)), [f"i{j}" for j in range(8)], size=3, splits=5)
    assert small["search"] == "exhaustive"

    hit = tf.reach_matrix(rng.normal(0, 1, (200, 8)))
    combo, reach, _, _ = tf.best_combination(hit, 3)
    every = [tf._reach_of(hit, c) for c in __import__("itertools").combinations(range(8), 3)]
    assert reach == pytest.approx(max(every)), "the exhaustive search did not return the maximum"


def test_a_best_worst_study_is_told_which_items_to_launch():
    """End to end: the section and the download, on a study where tastes genuinely divide."""
    pytest.importorskip("turf")
    pytest.importorskip("maxdiff")
    rng = np.random.default_rng(11)
    # Ten items rather than eight, because "reached" means "in your own top three" and a person
    # therefore accepts three of however many are on the list. At eight items that is 38% of
    # everything and the best set of three reaches 98% of people whatever the tastes are, so the
    # section is now refused as arithmetic rather than a finding. See _turf_section.
    items = ["Fast delivery", "Low price", "Good support", "Wide range", "Eco packaging",
             "Long warranty", "Known brand", "Easy returns", "Gift wrapping", "Loyalty points"]
    crowd_a = np.array([2.5, 2.2, 2.0, 0.0, -1.5, -1.5, -1.5, -2.0, -2.2, -1.8])
    crowd_b = np.array([-1.5, -1.0, -1.5, 0.0, 2.6, 2.4, -1.0, 1.0, -1.6, -1.4])
    rows = []
    for person in range(200):
        truth = crowd_a if person % 10 < 6 else crowd_b
        for task in range(9):
            shown = rng.choice(len(items), 4, replace=False)
            u = truth[shown] + rng.gumbel(0, 1, 4)
            best, worst = shown[u.argmax()], shown[u.argmin()]
            for i in shown:
                rows.append({"respondent_id": f"R{person:03d}", "task": task, "item": items[i],
                             "choice": "best" if i == best else
                                       ("worst" if i == worst else "")})
    with contextlib.redirect_stdout(io.StringIO()):
        r = sk.run_analysis(pd.DataFrame(rows).to_csv(index=False).encode(),
                            cfg=sk.SegmentationConfig(k_min=2, k_max=3, **FAST))

    assert "## Which 3 to launch" in r["digest"]
    assert "what_to_launch.csv" in r["files"], "the decision was reported but not downloadable"
    launch = pd.read_csv(io.StringIO(r["files"]["what_to_launch.csv"]))
    assert len(launch) == 3 and set(launch["item"]).issubset(items)
    # The set must span both crowds — that is the entire point.
    assert set(launch["item"]) & {"Eco packaging", "Long warranty"}, (
        f"chose {list(launch['item'])}, all of which serve the 60% crowd")


def test_the_luck_warning_stays_quiet_when_there_is_no_luck_to_report():
    """It read "Expect about 100%, not 100%" and "the 0-point difference" whenever the holdout
    agreed exactly. A caveat that fires with nothing to caveat reads as broken, and teaches the
    reader to skip the one that matters."""
    pytest.importorskip("turf")
    md = pytest.importorskip("maxdiff")

    class _Fixed:
        """An estimate whose reach cannot be optimistic: identical people, so any split agrees.

        Nine items rather than five — not decoration. "Reached" means the item is in that
        person's own top three, so with five items every person accepts 60% of the list and the
        best set of three necessarily reaches everybody; the section refuses that shape now
        because the answer is arithmetic rather than evidence.
        """
        item_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        respondent_ids = [f"R{i}" for i in range(200)]
        utilities = np.tile(np.arange(9.0, 0.0, -1.0), (200, 1))

    prose, result = sk._turf_section(_Fixed())
    assert result["optimism"] == pytest.approx(0.0, abs=1e-9)
    assert "not 100%" not in prose and "0-point difference" not in prose
    assert "holds up" in prose
    _ = md


# --------------------------------------------------------------------------------------------
# Building the best-worst questionnaire itself
# --------------------------------------------------------------------------------------------

def test_a_design_shows_every_item_and_every_pair():
    """What a best-worst design has to get right, and what each failure costs.

    An item shown less often is estimated less precisely, so its place in the ranking partly
    reflects how often it was asked about. A PAIR never shown together is worse: those two items
    are then compared only through other items, and no amount of analysis afterwards recovers the
    comparison the questionnaire never made.
    """
    d = pytest.importorskip("design")
    built, report = d.make_design(n_items=8, items_per_set=4, sets_per_respondent=10,
                                  n_respondents=40, seed=0)

    # 10 x 4 slots divides by 8 items, so exposure can be — and must be — exactly even.
    low, high = report["times_each_item_shown"]
    assert report["evenly_divisible"] and low == high, (
        f"item exposure ran {low}-{high} on a shape that divides evenly")
    assert report["never_paired"] == 0, (
        f"{report['never_paired']} pairs never appear together, so those comparisons cannot be made")

    # No screen may show the same item twice — it would ask someone to choose between a thing and
    # itself, and the reader would silently drop the set as malformed.
    for person in built:
        for task in person:
            assert len(set(task)) == len(task), f"an item appears twice on one screen: {task}"

    # And people must not all get the same questionnaire, or fatigue on screen three becomes an
    # opinion about whatever is always on screen three.
    first_screens = {tuple(person[0]) for person in built}
    assert len(first_screens) > 1, "every respondent was given an identical arrangement"


def test_a_design_this_builds_is_one_the_estimator_can_recover_truth_from():
    """The only test that matters for a design: does a study built this way actually work?

    Balance statistics are a proxy. This closes the loop — build the questionnaire, simulate people
    answering it from known preferences, run the real estimator, and check the ranking that comes
    back is the one that went in.
    """
    d = pytest.importorskip("design")
    md = pytest.importorskip("maxdiff")
    names = [f"item {i + 1}" for i in range(10)]
    truth = np.linspace(2.0, -2.0, 10)
    built, _report = d.make_design(10, 4, 10, 200, seed=1)

    rng = np.random.default_rng(7)
    rows = []
    for person, tasks in enumerate(built):
        for task_number, task in enumerate(tasks, start=1):
            drawn = truth[task] + rng.gumbel(0, 1, len(task))
            best, worst = task[int(drawn.argmax())], task[int(drawn.argmin())]
            for item in task:
                rows.append({"respondent_id": f"R{person:04d}", "task": task_number,
                             "item": names[item],
                             "choice": "best" if item == best else
                                       ("worst" if item == worst else "")})
    with contextlib.redirect_stdout(io.StringIO()):
        est = md.utilities_from_export(pd.DataFrame(rows), n_draws=2500, n_burn=800, progress=False)

    recovered = [names.index(name) for name in est.ranking()["item"]]
    from scipy.stats import spearmanr
    rho, _p = spearmanr(recovered, range(len(names)))
    assert rho > 0.9, f"a study built from this design recovered the planted order at only {rho:.3f}"


def test_a_design_says_when_its_shape_cannot_be_balanced():
    """Some shapes cannot be even, and that is arithmetic rather than a flaw. Nine screens of four
    is thirty-six slots, which does not divide by ten items, so some items MUST be shown once more
    than others. Saying so is better than a reader assuming perfect balance they did not get."""
    d = pytest.importorskip("design")
    _built, report = d.make_design(n_items=10, items_per_set=4, sets_per_respondent=9,
                                   n_respondents=30, seed=0)
    assert not report["evenly_divisible"]
    assert "cannot be perfectly even" in d.render(report)

    _even, even_report = d.make_design(n_items=8, items_per_set=4, sets_per_respondent=10,
                                       n_respondents=30, seed=0)
    assert "cannot be perfectly even" not in d.render(even_report), (
        "warned about uneven exposure on a shape that divides exactly")


@pytest.mark.parametrize("kwargs, expect", [
    (dict(n_items=4, items_per_set=4), "fewer items than you have"),
    (dict(n_items=10, items_per_set=1), "at least two items"),
])
def test_an_impossible_questionnaire_is_explained_in_words(kwargs, expect):
    """A screen showing everything asks nobody to choose, and a screen of one has nothing to
    compare. Both are refused before any work happens, in language that says what to change."""
    d = pytest.importorskip("design")
    with pytest.raises(ValueError) as caught:
        d.make_design(sets_per_respondent=5, n_respondents=5, **kwargs)
    explained = sk._explain_run_error(str(caught.value))
    assert "_DESIGN" not in explained, "an internal sentinel is being shown to the reader"
    assert expect in explained


def test_the_design_endpoint_builds_a_fieldable_questionnaire(monkeypatch):
    """Build a questionnaire through the real server and check the CSV is one you could field.

    The design was command-line-only for a release, which in a tool built for people who do not use
    a command line means it may as well not have existed. What matters here is not that the
    endpoint answers — it is that what comes back is the design itself: every item present, no
    screen showing the same item twice, and the row count exactly what was asked for. A handler
    that returned a plausible report and a truncated CSV would pass a shallower test and waste a
    real study.
    """
    import csv
    import io
    import json as _json
    import socket
    import threading
    import time
    import urllib.request

    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=sk.serve, kwargs={"port": port}, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post_json(path, obj):
        req = urllib.request.Request(base + path, data=_json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return _json.loads(urllib.request.urlopen(req, timeout=120).read().decode())

    for _ in range(60):                                        # wait for the server to come up
        try:
            urllib.request.urlopen(base + "/", timeout=5).read()
            break
        except Exception:
            time.sleep(0.1)

    names = [f"Benefit {chr(65 + i)}" for i in range(8)]
    got = post_json("/design", {"items": names, "per_screen": 3, "screens": 6, "people": 25})
    assert got["ok"] is True, got.get("error")

    rows = list(csv.DictReader(io.StringIO(got["csv"])))
    assert len(rows) == 25 * 6 * 3, "the CSV is not the size that was asked for"
    assert {r["item"] for r in rows} == set(names), "an item never appears in the questionnaire"
    assert len({r["respondent_id"] for r in rows}) == 25

    screens = {}
    for r in rows:
        screens.setdefault((r["respondent_id"], r["task"]), []).append(r["item"])
    assert all(len(v) == 3 for v in screens.values()), "a screen has the wrong number of items"
    assert all(len(set(v)) == 3 for v in screens.values()), "a screen shows the same item twice"

    # Different people must see different arrangements, or fatigue on screen three becomes an
    # opinion about whatever is always on screen three.
    arrangements = {tuple(sorted(v)) for (person, task), v in screens.items() if task == "1"}
    assert len(arrangements) > 1, "every respondent got the identical questionnaire"

    assert got["report"]["n_items"] == 8 and got["report"]["never_paired"] == 0
    assert "_DESIGN" not in got["prose"]


@pytest.mark.parametrize("body,expect", [
    ({"items": ["only", "two"]}, "at least three"),
    ({"items": [f"item {i}" for i in range(45)]}, "45 items"),
    ({"items": ["a", "b", "c", "d"], "per_screen": 9}, "between 2 and 6"),
    ({"items": ["a", "b", "c"], "per_screen": 3}, "nothing left to compare"),
    ({"items": ["a", "b", "c", "d", "e"], "screens": 40}, "between 2 and 15"),
    ({"items": ["a", "b", "c", "d", "e"], "people": 5000}, "between 20 and 300"),
    ({"items": ["a", "b", "c", "d", "e"], "screens": "ten"}, "whole numbers"),
])
def test_the_design_endpoint_refuses_impossible_questionnaires_in_plain_words(body, expect,
                                                                             monkeypatch):
    """Every refusal has to say what to do instead, and never leak an internal sentinel.

    These are the shapes a person actually types — nine items on a screen, three of three items,
    five thousand versions — and each one is a question about the study rather than a mistake, so
    the answer has to read like one.
    """
    import json as _json
    import socket
    import threading
    import time
    import urllib.request

    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=sk.serve, kwargs={"port": port}, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/", timeout=5).read()
            break
        except Exception:
            time.sleep(0.1)

    req = urllib.request.Request(base + "/design", data=_json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    got = _json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    assert got["ok"] is False
    assert expect in got["error"], got["error"]
    assert "_DESIGN" not in got["error"] and "Traceback" not in got["error"]


def test_the_design_endpoint_collapses_items_listed_twice(monkeypatch):
    """A duplicated line must not become an item competing against itself.

    Pasting a list is the normal way in, and a list pasted from a slide or an email routinely has
    the same benefit twice in different case. Left alone it produces a questionnaire where two
    identical options appear on one screen and the ranking has to split preference between them.
    """
    import json as _json
    import socket
    import threading
    import time
    import urllib.request

    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=sk.serve, kwargs={"port": port}, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/", timeout=5).read()
            break
        except Exception:
            time.sleep(0.1)

    body = {"items": ["Free delivery", "free delivery ", "Lower prices", "", "Longer returns"],
            "per_screen": 2, "screens": 4, "people": 20}
    req = urllib.request.Request(base + "/design", data=_json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    got = _json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    assert got["ok"] is True, got.get("error")
    assert got["items"] == ["Free delivery", "Lower prices", "Longer returns"]
    assert got["report"]["n_items"] == 3


def test_a_swap_never_changes_the_total_number_of_pairings():
    """The fast objective is only correct because the total is fixed; check that it is.

    The optimiser minimises the SUM OF SQUARES of the pair counts and calls it minimising the
    variance. That step is valid only while the mean is constant — which holds because a swap moves
    items between screens without changing how many pairs a screen contains. If that ever stopped
    being true the search would quietly optimise the wrong thing and still report a balance figure,
    so it is checked rather than assumed.
    """
    import design as d
    import numpy as np

    for seed in range(3):
        built, _ = d.make_design(9, 3, 5, 20, seed=seed)
        pairs = d._pair_counts(built, 9)
        off = pairs[~np.eye(9, dtype=bool)]
        # Every screen contributes exactly C(items_per_set, 2) unordered pairs, counted twice.
        assert int(off.sum()) == 20 * 5 * (3 * 2 // 2) * 2


@pytest.mark.parametrize("body,expect", [
    # The one that failed SILENTLY: iterating a string yields letters, so the app built a
    # questionnaire asking people to choose between d, e, l, i, v, r and y — and reported it as a
    # perfectly balanced seven-item design.
    ({"items": "delivery"}, "as a list"),
    ({"items": [{"a": 1}, ["b"], None, "real item"]}, "has to be text"),
    ({"items": ["x" * 5000, "b", "c", "d"]}, "under 200"),
])
def test_the_design_endpoint_refuses_items_that_are_not_items(body, expect, monkeypatch):
    """Malformed item lists are refused rather than coerced into something plausible.

    `str()` will happily turn a dict into a string and a bare string into a list of letters, and
    every one of those produces a design that looks correct in the report and is nonsense in the
    field. Refusing costs a message; coercing costs a study.
    """
    import json as _json
    import socket
    import threading
    import time
    import urllib.request

    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=sk.serve, kwargs={"port": port}, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/", timeout=5).read()
            break
        except Exception:
            time.sleep(0.1)

    req = urllib.request.Request(base + "/design", data=_json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    got = _json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    assert got["ok"] is False, "a malformed item list was accepted"
    assert expect in got["error"], got["error"]


def test_an_item_carrying_a_newline_cannot_split_a_cell_in_the_export(monkeypatch):
    """Whitespace inside an item is collapsed, because a newline corrupts most platform imports.

    The panel cannot produce one — it splits on newlines — but the endpoint is reachable directly,
    and a broken import is worse than a visibly wrong one because it looks like the design's fault.
    """
    import csv
    import io
    import json as _json
    import socket
    import threading
    import time
    import urllib.request

    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=sk.serve, kwargs={"port": port}, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/", timeout=5).read()
            break
        except Exception:
            time.sleep(0.1)

    body = {"items": ['Free "next-day",\ndelivery', "Lower prices", "Longer returns", "Live chat"],
            "per_screen": 2, "screens": 3, "people": 20}
    req = urllib.request.Request(base + "/design", data=_json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    got = _json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    assert got["ok"] is True, got.get("error")
    assert got["items"][0] == 'Free "next-day", delivery'
    rows = list(csv.DictReader(io.StringIO(got["csv"])))
    assert len(rows) == 20 * 3 * 2
    assert {r["item"] for r in rows} == set(got["items"])


def test_segment_names_that_are_not_names_are_refused(monkeypatch, tmp_path):
    """The same coercion the design endpoint had, in the one place nothing downstream can catch it.

    A segment name is free text, so no later validation can tell a real name from a coerced one:
    "AB" sent instead of ["A", "B"] named two segments A and B, and a list holding an object named
    one of them "{'x': 1}". Those land in group_names.csv and in every export built from it, which
    is what a colleague opens.
    """
    import json as _json
    import socket
    import threading
    import time
    import urllib.request

    monkeypatch.setenv("SURVEY_SEGMENTER_PROJECTS", str(tmp_path / "projects"))
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=sk.serve, kwargs={"port": port}, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(80):
        try:
            urllib.request.urlopen(base + "/", timeout=5).read()
            break
        except Exception:
            time.sleep(0.1)

    rng = np.random.default_rng(0)
    rows = ["id,q1,q2,q3,q4"]
    for i in range(120):
        centre = 2 if i % 2 else 6
        rows.append(f"{i}," + ",".join(
            str(int(np.clip(rng.normal(centre, 1), 1, 7))) for _ in range(4)))
    boundary = "----t"
    blob = ("\n".join(rows)).encode()
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="s.csv"\r\n'
            f'Content-Type: text/csv\r\n\r\n').encode() + blob + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(base + "/analyze", data=body, method="POST",
                                 headers={"Content-Type":
                                          f"multipart/form-data; boundary={boundary}"})
    session_id = _json.loads(urllib.request.urlopen(req, timeout=300).read().decode())["session_id"]

    def name_with(names):
        r = urllib.request.Request(
            base + "/name", data=_json.dumps({"session_id": session_id, "names": names}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        return _json.loads(urllib.request.urlopen(r, timeout=60).read().decode())

    spelt_as_a_string = name_with("AB")
    assert spelt_as_a_string["ok"] is False, "a bare string was split into one name per letter"
    assert "as a list" in spelt_as_a_string["error"]

    with_objects = name_with([{"x": 1}, None])
    assert with_objects["ok"] is False
    assert "has to be text" in with_objects["error"]

    fine = name_with(["Loyal  Fans ", "Price Hunters"])
    assert fine["ok"] is True, fine.get("error")
    assert fine["names"] == ["Loyal Fans", "Price Hunters"], "whitespace was not tidied"


def test_turf_says_when_the_winning_set_is_a_coin_toss():
    """A tie at the top must be reported, not resolved silently by the order of the item list.

    Reach counts people, so it can only land on multiples of 1/n: sixty respondents give sixty-one
    possible values for hundreds of candidate combinations, and exact ties at the best value are
    ordinary rather than freakish. Before this, whichever tied set came first in `itertools`
    order — that is, in the order the items happened to appear in the file — was printed as the
    finding. Reordering the item list changed the recommendation in 8 of 25 simulated studies at
    sixty people and ten items, with identical reach both times.
    """
    tf = pytest.importorskip("turf")

    def study(n_people, n_items, seed):
        r = np.random.default_rng(seed)
        centres = r.normal(0, 1, (3, n_items))
        return centres[r.integers(0, 3, n_people)] + r.normal(0, 0.8, (n_people, n_items))

    names = [f"item{i}" for i in range(10)]

    # A study whose top is genuinely shared, and one whose top is not: both have to be said
    # correctly, because a tie warning that never switches off is noise.
    tied = tf.turf(study(60, 10, 0), names, size=3, splits=5)
    assert tied["n_tied"] > 1
    assert len(tied["tied_items"]) == tied["n_tied"]
    reaches = {round(tf._reach_of(tf.reach_matrix(study(60, 10, 0)),
                                  [names.index(x) for x in combo]), 12)
               for combo in tied["tied_items"]}
    assert len(reaches) == 1, "sets reported as tied do not actually reach the same number"

    alone = tf.turf(study(60, 10, 3), names, size=3, splits=5)
    assert alone["n_tied"] == 1 and alone["tied_items"] == []

    # And the consequence itself: with the tie reported, a reordered item list still tells the
    # reader the same thing, even when the single named set differs.
    utilities = study(60, 10, 0)
    order = np.random.default_rng(99).permutation(10)
    shuffled = tf.turf(utilities[:, order], [names[i] for i in order], size=3, splits=5)
    assert shuffled["reach"] == tied["reach"]
    assert shuffled["n_tied"] == tied["n_tied"], "the tie count must not depend on item order"


def test_both_turf_searches_break_a_tie_the_same_way():
    """Greedy and exhaustive must not disagree about which of two equal items to take.

    The greedy branch sorted on `(reach, index)` descending, so a tie went to the HIGHEST item
    index, while the exhaustive branch keeps the first combination it meets, which is the lowest.
    Two code paths for one question, quietly giving different answers depending only on the item
    count that decided which path ran.
    """
    tf = pytest.importorskip("turf")
    # Items 0 and 1 are exact duplicates, so any choice between them is a tie by construction.
    hit = np.zeros((40, 5), dtype=bool)
    hit[:20, 0] = True
    hit[:20, 1] = True
    hit[20:30, 2] = True
    hit[30:, 3] = True

    exhaustive, _, how, _ = tf.best_combination(hit, 3)
    assert how == "exhaustive"

    monkey = tf.MAX_EXHAUSTIVE
    try:
        tf.MAX_EXHAUSTIVE = 0                      # force the greedy path on the same data
        greedy, _, how_greedy, ties = tf.best_combination(hit, 3)
    finally:
        tf.MAX_EXHAUSTIVE = monkey
    assert how_greedy == "greedy"
    assert ties is None, "greedy cannot count ties and must not claim to"
    assert greedy == exhaustive, "the two searches disagree about a tie"


def test_the_turf_optimism_is_the_gap_the_report_actually_shows():
    """The stated gap must equal headline minus holdout — the two numbers the reader can see.

    It used to be `in_sample - holdout`, where in_sample is the reach on the half the combination
    was CHOSEN from. Both are half-sample quantities, so it measured the optimism of a study half
    this size and pinned it on the full-sample headline. The report printed "Expect about 93%, not
    95%" immediately above "the 3-point difference": arithmetic a reader can check, and it did not
    add up. The 96% those points were measured from appeared nowhere in the report.
    """
    tf = pytest.importorskip("turf")
    r = np.random.default_rng(3)
    centres = r.normal(0, 1, (3, 20))
    utilities = centres[r.integers(0, 3, 100)] + r.normal(0, 0.8, (100, 20))
    res = tf.turf(utilities, [f"item{i}" for i in range(20)], size=3, splits=20)

    assert res["optimism"] == pytest.approx(res["reach"] - res["holdout_reach"], abs=1e-12)
    # And it is genuinely a different number from the old definition, so this test would have
    # failed before rather than passing either way.
    assert res["optimism"] != pytest.approx(res["in_sample_reach"] - res["holdout_reach"], abs=1e-6)


def test_turf_optimism_tracks_the_real_error_against_a_known_population():
    """The honesty correction has to be checked against an outside truth, not against itself.

    Pure noise is the strongest case to check: with no taste groups at all, everything the search
    finds is luck, so the headline's error is large and known to be entirely optimism. Draw a
    sample from a big population, run TURF on the sample, then look up what the chosen combination
    really reaches in the population.
    """
    tf = pytest.importorskip("turf")
    reported, real = [], []
    for seed in range(6):
        population = np.random.default_rng(seed).normal(0, 1, (20000, 20))
        population_hit = tf.reach_matrix(population)
        index = np.random.default_rng(500 + seed).choice(len(population), 100, replace=False)
        res = tf.turf(population[index], [f"i{j}" for j in range(20)], size=3, splits=30)
        reported.append(res["optimism"])
        real.append(res["reach"] - tf._reach_of(population_hit, res["indices"]))

    reported, real = float(np.mean(reported)), float(np.mean(real))
    assert real > 0.05, "the setup is meant to produce a badly optimistic headline"
    # Within four points of the truth. The old definition was out by roughly double that here, and
    # in the opposite direction on structured data — it called a pessimistic headline optimistic.
    assert abs(reported - real) < 0.04, f"reported {reported:.3f} against a real error of {real:.3f}"


@pytest.mark.parametrize("n_items,expect_section", [(5, False), (8, False), (9, True), (14, True)])
def test_turf_refuses_item_lists_too_short_for_reach_to_mean_anything(n_items, expect_section):
    """Below about nine items the answer is arithmetic wearing a finding's clothes.

    Someone counts as reached when the item is among their own top three, so each person accepts
    three of however many items are on the list. At five items that is 60% of everything: measured
    over twelve simulated studies of 200 people, the best set of three reached 100.0% every single
    time, with ten combinations tied for it. The section used to run at five items and report that
    100% as a result.

    Best reach for a set of three, measured: 5 items 100.0%, 6 items 99.8%, 8 items 98.4%,
    10 items 95.1%, 20 items 83.9%.
    """
    pytest.importorskip("turf")

    class _Est:
        item_names = [f"item{i}" for i in range(n_items)]
        respondent_ids = [f"R{i}" for i in range(200)]
        utilities = np.random.default_rng(2).normal(0, 1, (200, n_items))

    produced = sk._turf_section(_Est()) is not None
    assert produced is expect_section


def test_turf_says_when_fewer_items_would_reach_the_same_people():
    """Recommending three when one does the job is an expensive way to be right.

    The incremental column shows this row by row, but the headline still names the full set, and a
    reader who reads headlines walks away about to fund two items that add nobody.
    """
    pytest.importorskip("turf")

    class _Dominant:
        """One item everybody takes, then eight nobody needs."""
        item_names = [f"item{i}" for i in range(9)]
        respondent_ids = [f"R{i}" for i in range(200)]
        # Item 0 tops everyone's list; the rest are shuffled noise below it.
        _rng = np.random.default_rng(5)
        utilities = np.column_stack([np.full(200, 10.0), _rng.normal(0, 1, (200, 8))])

    prose, result = sk._turf_section(_Dominant())
    assert result["alone"][0] == pytest.approx(1.0), "item 0 should reach everyone by itself"
    assert "would do" in prose, "the report named three items without saying one was enough"
    assert "1 of these 3 would do" in prose
