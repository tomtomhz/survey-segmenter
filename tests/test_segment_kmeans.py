"""
Test suite for segment_kmeans.py — encodes the professional guarantees as executable tests.

Run:  pytest              (from tools/survey-segmenter/)

The two tests that matter most:
  - test_recovers_planted_structure : on data with real segments, it finds them.
  - test_rejects_structureless_noise : on random noise, it does NOT manufacture stable segments.
A segmentation tool that fails either of those is worse than useless, because it will mislead.
"""
import io
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
import charts
import kprototypes as kp
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


def test_short_label_uses_whole_words():
    lbl = sk._short_label("I want to meet people in real life")
    assert lbl.endswith(("people", "real", "life")) and "want" in lbl        # no mid-word cut
    assert not lbl.endswith("rea")


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
    # triples must split to make a third group; the other survives whole.
    assert max(got.values()) == 1.0 and min(got.values()) >= 2 / 3

    # A segment index with no members must not produce a score or a division by zero.
    lopsided = sk.stability_across_solutions(X, range(2, 4), 3, labels, 5, np.random.default_rng(0))
    assert 2 not in lopsided

    # No neighbouring solution to compare against means no claim, not a fabricated one.
    assert sk.stability_across_solutions(X, [2], 2, labels, 5, np.random.default_rng(0)) == {}

    assert sk.persistence_paragraph({}, []) == ""
    holds = sk.persistence_paragraph({0: 0.95, 1: 0.88}, ["Loyal", "Curious"])
    assert "holds its shape" in holds and "Loyal" in holds
    assert "does not survive" in sk.persistence_paragraph({0: 0.95, 1: 0.2}, ["A", "B"])

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
    assert "survive a different number" in real["digest"]
    assert min(real["persistence"].values()) >= 0.7, real["persistence"]
    assert "Every segment survives" in real["digest"]

    noise = analyse(_likert(rng.normal(3, 1.1, (400, 5))))
    assert min(noise["persistence"].values()) < 0.7, (
        "segments found in structureless data should not survive a change in k")


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
    ai.save_api_key("  sk-ant-abc  ")
    assert ai.load_api_key() == "sk-ant-abc"                    # trimmed
    assert ai.key_source() == "this app's Settings"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    assert ai.load_api_key() == "sk-ant-env"                    # env wins over the file
    assert "environment" in ai.key_source()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ai.clear_api_key()
    assert ai.load_api_key() is None


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
    assert list(scored.columns) == ["respondent_id", "segment", "confidence"]
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

    # The exclusion is stated, not silent — a reader can see which k values were taken off the
    # table and why, rather than wondering why an obvious peak was ignored.
    assert "Ruled out before the vote" in r["digest"]

    # And the residual case is still caught downstream: anything that slips under the floor in
    # the final fit is called out in the report rather than passing silently.
    assert "below 5% of the sample" in r["digest"]


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
    step on Swedish five-point scales and neither was listed. This matters because the original sponsor fields
    Swedish surveys.
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
    for hit in re.finditer(r"var\(--seg-\d", svg):
        prefix = svg[max(0, hit.start() - 90):hit.start()]
        assert "style=" in prefix.rsplit(">", 1)[-1], (
            "a themed colour landed in a presentation attribute, where var() does not apply")


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
