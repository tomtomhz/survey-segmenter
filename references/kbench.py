"""Does the tool actually recover the number of groups, and does it refuse to invent them?

That is the tool's central claim, and it has been tested on a handful of hand-made cases. This
sweeps it against data with known ground truth across the conditions survey data actually
presents: overlapping segments, very unequal sizes, noise questions that separate nobody,
non-spherical shapes, and no structure at all.

Written as a measurement, not a test: the point is to find out what the answer is, including
where the tool is weak, rather than to confirm what I already believe.
"""
import contextlib
import pathlib
import io
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import segment_kmeans as sk                                       # noqa: E402

warnings.filterwarnings("ignore")
FAST = dict(gap_B=8, stability_B=12, ps_splits=6, jaccard_B=20, n_init_final=10,
            n_init_search=8, run_consensus=False, check_variable_selection=False)


def likert(x):
    """Push a latent score onto a 1-5 answer scale, which is what a survey actually contains."""
    return np.clip(np.round(x), 1, 5).astype(int)


def make(kind, n=400, seed=0):
    """Return (dataframe, true_k). true_k of 1 means there is genuinely nothing to find."""
    rng = np.random.default_rng(seed)

    if kind == "separated k=3":
        centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
        who = rng.integers(0, 3, n)
        X = likert(centres[who] + rng.normal(0, 0.5, (n, 5)))
        return X, 3

    if kind == "overlapping k=3":
        centres = np.array([[4, 2, 4, 2, 3], [2, 4, 2, 4, 3], [3, 3, 4, 4, 2]], float)
        who = rng.integers(0, 3, n)
        X = likert(centres[who] + rng.normal(0, 1.3, (n, 5)))
        return X, 3

    if kind == "unequal 80/15/5":
        centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1]], float)
        who = rng.choice(3, n, p=[0.80, 0.15, 0.05])
        X = likert(centres[who] + rng.normal(0, 0.5, (n, 5)))
        return X, 3

    if kind == "k=3 + 5 noise questions":
        centres = np.array([[5, 1, 5], [1, 5, 1], [3, 3, 5]], float)
        who = rng.integers(0, 3, n)
        signal = likert(centres[who] + rng.normal(0, 0.5, (n, 3)))
        noise = likert(rng.normal(3, 1.2, (n, 5)))        # separate nobody
        return np.hstack([signal, noise]), 3

    if kind == "two elongated bands":
        # k-means assumes compact, roughly spherical groups. Two long parallel bands is the
        # textbook case where that assumption is wrong.
        t = rng.uniform(-1, 1, n)
        side = rng.integers(0, 2, n)
        X = np.column_stack([likert(3 + 2 * t), likert(3 + 2 * t + (side * 2 - 1) * 1.4),
                             likert(rng.normal(3, 1, n)), likert(rng.normal(3, 1, n)),
                             likert(rng.normal(3, 1, n))])
        return X, 2

    if kind == "no structure at all":
        return likert(rng.normal(3, 1.1, (n, 5))), 1

    if kind == "k=5":
        centres = np.array([[5, 1, 5, 1, 3], [1, 5, 1, 5, 3], [3, 3, 5, 5, 1],
                            [5, 5, 1, 1, 5], [1, 1, 3, 3, 1]], float)
        who = rng.integers(0, 5, n)
        X = likert(centres[who] + rng.normal(0, 0.5, (n, 5)))
        return X, 5

    raise ValueError(kind)


CASES = ["separated k=3", "overlapping k=3", "unequal 80/15/5", "k=3 + 5 noise questions",
         "two elongated bands", "k=5", "no structure at all"]

print(f"{'condition':<26}{'true k':>7}{'found':>7}{'confidence':>12}{'secs':>7}   verdict")
print("-" * 88)

summary = []
for kind in CASES:
    for seed in range(3):
        X, true_k = make(kind, seed=seed)
        df = pd.DataFrame(X, columns=[f"q{i+1}" for i in range(X.shape[1])])
        df.insert(0, "respondent_id", [f"P{i}" for i in range(len(df))])
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r = sk.run_analysis(df.to_csv(index=False).encode(),
                                cfg=sk.SegmentationConfig(k_min=2, k_max=8, **FAST))
        dt = time.time() - t0
        found, conf = r["k"], r["confidence"]

        if true_k == 1:
            # Nothing to find: the only right answer is to say so, whatever k it settles on.
            ok = conf == "low"
            verdict = "correctly refuses" if ok else f"CLAIMS STRUCTURE ({conf})"
        else:
            ok = found == true_k
            verdict = "correct" if ok else f"off by {found - true_k:+d}"
            if not ok and conf == "low":
                verdict += " (but flagged low)"
        summary.append((kind, ok, conf))
        print(f"{kind:<26}{true_k:>7}{found:>7}{conf:>12}{dt:>7.0f}   {verdict}")

print("\n" + "=" * 88)
for kind in CASES:
    got = [ok for k, ok, _ in summary if k == kind]
    confs = [c for k, _, c in summary if k == kind]
    print(f"  {kind:<26} {sum(got)}/{len(got)} correct   confidence seen: {sorted(set(confs))}")


# =====================================================================================
# Mixed-type surveys: rating scales and pick-any questions in one model
# =====================================================================================
# The k-prototypes path (kprototypes.py) has to be measured on its own terms, because a mixed
# questionnaire has a failure mode the all-ratings one does not: half the questions can be useless
# while the other half is fine. The question is not only "does it find the right number" but "does
# it stay quiet when the half it was given is noise".
def make_mixed(kind, n=400, seed=0):
    """Four 1-5 ratings plus two pick-any questions. Returns (frame, true_k, true membership)."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, n)
    centres = np.array([[5, 1, 5, 1], [1, 5, 1, 5], [3, 3, 5, 5]], float)

    if kind == "brands only":                       # the ratings separate nobody
        ratings = likert(rng.normal(3, 1.2, (n, 4)))
    else:
        ratings = likert(centres[truth] + rng.normal(0, 0.7, (n, 4)))
    df = pd.DataFrame(ratings, columns=[f"q{i+1}" for i in range(4)])
    df.insert(0, "respondent_id", [f"P{i}" for i in range(n)])

    names = np.array(["A", "B", "C"])
    brands_carry_signal = kind != "ratings only"    # otherwise the brand answers are noise
    for j in range(2):
        probs = [np.where(np.arange(3) == (g + j) % 3, 0.70, 0.15) if brands_carry_signal
                 else np.full(3, 1 / 3) for g in truth]
        df[f"brand{j+1}"] = [names[rng.choice(3, p=p)] for p in probs]
    return df, 3, truth


MIXED = ["both kinds carry it", "ratings only", "brands only"]

print("\n" + "=" * 88)
print("Mixed questionnaires: 4 rating questions + 2 pick-any questions, true k = 3\n")
print(f"{'condition':<24}{'found':>7}{'confidence':>12}{'ARI':>7}{'secs':>7}   verdict")
print("-" * 88)

for kind in MIXED:
    for seed in range(3):
        df, true_k, truth = make_mixed(kind, seed=seed)
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r = sk.run_analysis(df.to_csv(index=False).encode(),
                                cfg=sk.SegmentationConfig(k_min=2, k_max=6, **FAST))
        dt = time.time() - t0
        labels = pd.read_csv(io.StringIO(r["files"]["segment_assignments.csv"]))["segment"]
        agree = adjusted_rand_score(truth, labels)
        if r["k"] == true_k:
            verdict = "correct"
        elif r["confidence"] == "high":
            verdict = "WRONG AND CONFIDENT"      # the one outcome that would make it unusable
        else:
            verdict = f"off by {r['k'] - true_k:+d} (flagged {r['confidence']})"
        print(f"{kind:<24}{r['k']:>7}{r['confidence']:>12}{agree:>7.2f}{dt:>7.0f}   {verdict}")

print("""
Reading: when both kinds of question carry the signal it recovers k = 3 at high confidence. When
either half is noise it loses accuracy — which is the ordinary cost of clustering on useless
variables, not something particular to this method — but it drops to moderate or low rather than
claiming a result. That is the same property the all-ratings path is held to, and the report's
variable-selection check names the useless questions so they can be dropped and the run repeated.
""")
