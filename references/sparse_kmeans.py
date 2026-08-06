"""Sparse k-means — implemented, measured, and NOT adopted. Kept so the finding can be rechecked.

    THIS MODULE IS NOT PART OF THE TOOL. It lives in references/ because the measurements below
    are the useful output, not the code. Read `STATE-OF-THE-ART.md` before reinstating it.

Witten & Tibshirani's *A Framework for Feature Selection in Clustering* (JASA, 2010) is the
standard answer to a real problem this tool has: every question counts equally when respondents
are grouped, and questions that separate nobody drag membership accuracy down. It was built,
measured against the tool's own conditions, and rejected on the evidence. Four findings, in the
order that decided it:

1.  **No gain where it matters.** On ordinary survey conditions plain k-means already recovers the
    planted groups at ARI 1.00, and so does this. A gain appears only when noise questions
    outnumber real ones and the groups are weak: +0.13 at two real questions against ten noise,
    +0.15 at two against fourteen.

2.  **Actively dangerous as the clustering method.** On PURE NOISE it lifts the silhouette from
    0.13 to 0.58 with six questions, and from 0.05 to 0.35 with eighteen. It selects whichever
    questions happen to split the noise best and then weights them up, so structureless data comes
    out looking well separated. That is precisely the confidently-wrong failure the whole tool is
    built to avoid. The stability gates still caught it — split-half ARI stayed at 0.26-0.45
    against 1.00 for real structure — but a headline separation number that flatters noise is not
    something to ship and rely on a downstream check to contain.

3.  **The weights cannot tell real from noise.** Share of total weight held by the top three
    questions: 98-99% on real structure, and 94-100% on pure noise. It concentrates weight
    whether or not there is anything to concentrate on, so "these questions drive the
    segmentation" would read identically on a study with no segments.

4.  **It adds nothing over what the tool already reports.** At identifying which questions carry
    the segmentation — the one thing it does well when structure exists — eta-squared ranked every
    real question above every noise question in 5 of 5 runs across four conditions, and so did
    this. Sparse weights matched it and never beat it.

So: a published method, correctly implemented, that solves a problem this tool does not have with a
tool it already owns, at the cost of a statistic that flatters noise. Reproduce with
`python3 references/sparse_kmeans.py`.

The implementation below is left intact and correct.

Original description follows.

Which questions actually deserve a vote, learned from the data rather than assumed.

Every question currently counts equally when respondents are grouped. That is a real assumption and
it has a measured cost: three questions that separate nobody, sitting beside six that do, drag
membership accuracy down by 0.25 ARI. The report already names the offenders through the
variable-selection check, and re-runs without them for comparison — but that is all-or-nothing per
question, and the honest answer is usually that a question matters *somewhat*.

This is Witten & Tibshirani, *A Framework for Feature Selection in Clustering* (JASA, 2010), the
standard answer. It gives each question a weight and finds the clustering and the weights together:

    maximise   sum_j w_j * (between-cluster sum of squares for question j)
    subject to ||w||₂ ≤ 1,  ||w||₁ ≤ s,  w_j ≥ 0

The L1 bound is what makes it a *selection* rather than a rescaling: as s tightens, weights are
driven to exactly zero and those questions drop out entirely. Alternating between the two halves —
cluster with the current weights, then re-weight given the clustering — is the whole algorithm, and
each half has a closed form.

**The tuning parameter is chosen, not picked.** s decides how many questions survive, and a
hand-set value would be exactly the sort of undefendable number this tool rejects elsewhere. Witten
& Tibshirani give a permutation gap: for each candidate s, compare the objective on the real data
against the same objective on data whose questions have been shuffled independently, which destroys
any relationship between them while keeping each question's own distribution. The s with the
largest gap is the one where the real data most outruns its own shuffled twin.

**This does not silently change anyone's answer.** It is reported beside the shipped segmentation,
the way the variable-selection check is, because dropping questions can manufacture structure as
easily as it can reveal it and the analyst is the one who knows whether a question is conceptually
load-bearing.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def _bcss_per_feature(X, labels):
    """Between-cluster sum of squares for each question separately.

    This is the quantity the weights are proportional to: a question on which the groups sit far
    apart earns a large one, and a question everybody answers the same way earns nearly nothing.
    """
    grand = X.mean(0)
    total = ((X - grand) ** 2).sum(0)
    within = np.zeros(X.shape[1])
    for c in np.unique(labels):
        members = X[labels == c]
        if len(members):
            within += ((members - members.mean(0)) ** 2).sum(0)
    # Between = total - within, floored at zero: tiny negatives are floating-point noise, and a
    # negative weight would mean "this question argues against its own grouping", which is not a
    # thing the objective allows.
    return np.maximum(total - within, 0.0)


def _soft_threshold(x, delta):
    return np.sign(x) * np.maximum(np.abs(x) - delta, 0.0)


def _weights_for(a, bound):
    """The w maximising w·a subject to ||w||₂ ≤ 1, ||w||₁ ≤ bound, w ≥ 0.

    Closed form given the threshold: soft-threshold a, then scale to unit length. The threshold
    itself is found by bisection, because ||w||₁ falls monotonically as it rises — so there is
    exactly one value that meets the budget, and no search heuristics are involved.
    """
    a = np.maximum(a, 0.0)
    if not np.any(a > 0):
        return np.full(a.size, 1.0 / np.sqrt(a.size))

    def scaled(delta):
        w = _soft_threshold(a, delta)
        norm = np.linalg.norm(w)
        return w / norm if norm > 0 else w

    if np.abs(scaled(0.0)).sum() <= bound:
        return scaled(0.0)                       # the budget does not bind; no question is cut
    lo, hi = 0.0, float(a.max())
    for _ in range(60):                          # 60 halvings is far past double precision
        mid = (lo + hi) / 2
        if np.abs(scaled(mid)).sum() > bound:
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def fit(X, k, bound, rng, n_init=10, max_iter=15):
    """Alternate between clustering and re-weighting until neither moves.

    Returns (weights, labels, objective). The clustering step is ordinary k-means on the data with
    each question scaled by the square root of its weight — which is exactly what weighting a sum
    of squared distances means, so no separate weighted k-means is needed.
    """
    X = np.asarray(X, float)
    p = X.shape[1]
    w = np.full(p, 1.0 / np.sqrt(p))
    labels = None
    for _ in range(max_iter):
        scaled = X * np.sqrt(w)
        if np.allclose(scaled, 0):
            break
        labels = KMeans(n_clusters=k, n_init=n_init,
                        random_state=int(rng.integers(1e9))).fit(scaled).labels_
        new_w = _weights_for(_bcss_per_feature(X, labels), bound)
        if np.allclose(new_w, w, atol=1e-6):
            w = new_w
            break
        w = new_w
    if labels is None:
        labels = KMeans(n_clusters=k, n_init=n_init,
                        random_state=int(rng.integers(1e9))).fit(X).labels_
    return w, labels, float(w @ _bcss_per_feature(X, labels))


def choose_bound(X, k, rng, n_permutations=8, n_init=5, candidates=None):
    """Witten & Tibshirani's permutation gap: how far the real data outruns its shuffled twin.

    Shuffling each question independently keeps every question's own distribution intact and
    destroys only the relationships between them — so the comparison isolates structure rather than
    spread. The candidate s with the largest gap wins.

    The candidate range starts just above 1 because s ≤ 1 forces a single question to carry
    everything, and ends at sqrt(p), the point at which the L1 bound stops binding at all and this
    reduces to ordinary k-means.
    """
    X = np.asarray(X, float)
    p = X.shape[1]
    if candidates is None:
        candidates = np.linspace(1.2, max(np.sqrt(p), 1.3), 8)

    permuted = []
    for _ in range(n_permutations):
        shuffled = np.column_stack([rng.permutation(X[:, j]) for j in range(p)])
        permuted.append(shuffled)

    best, best_gap, rows = None, -np.inf, []
    for s in candidates:
        _, _, real = fit(X, k, s, rng, n_init=n_init)
        nulls = [fit(shuffled, k, s, rng, n_init=n_init)[2] for shuffled in permuted]
        # Logs, as the paper specifies: the objective scales with the data, and the log makes the
        # comparison a ratio rather than a difference.
        gap = np.log(real + 1e-12) - float(np.mean(np.log(np.array(nulls) + 1e-12)))
        rows.append({"bound": float(s), "gap": float(gap)})
        if gap > best_gap:
            best, best_gap = float(s), float(gap)
    return best, rows


def weighting(X, k, rng, n_init=10, n_permutations=8):
    """The whole thing: choose the bound, fit, and report what survived.

    Returns None when there is nothing to weight — fewer than three questions leaves no room for
    selection to mean anything.
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] < 3 or len(X) < k * 2:
        return None
    bound, gaps = choose_bound(X, k, rng, n_permutations=n_permutations, n_init=max(3, n_init // 3))
    w, labels, objective = fit(X, k, bound, rng, n_init=n_init)
    # Report weights on a scale a reader can use: the largest question is 1.0 and the rest are
    # fractions of it. The absolute values are constrained to unit length and mean nothing on
    # their own.
    top = w.max()
    relative = w / top if top > 0 else w
    return {"weights": w, "relative": relative, "labels": labels, "bound": bound,
            "objective": objective, "dropped": [int(j) for j in np.flatnonzero(w <= 0)],
            "gaps": gaps}


if __name__ == "__main__":
    # Reproduces the four findings in the docstring. About a minute.
    import warnings
    from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
    warnings.filterwarnings("ignore")

    def planted(seed, n_real, n_noise, spread, n=400):
        rng = np.random.default_rng(seed)
        truth = rng.integers(0, 3, n)
        centres = np.array([[5, 1, 5, 1, 3, 2], [1, 5, 1, 5, 3, 4],
                            [3, 3, 5, 5, 1, 5]], float)[:, :n_real]
        real = np.clip(np.round(centres[truth] + rng.normal(0, spread, (n, n_real))), 1, 5) \
            if n_real else np.empty((n, 0))
        noise = np.clip(np.round(rng.normal(3, 1.2, (n, n_noise))), 1, 5) \
            if n_noise else np.empty((n, 0))
        X = np.hstack([real, noise])
        lo, hi = X.min(0), X.max(0)
        return (X - lo) / np.where(hi > lo, hi - lo, 1), truth

    print("1. Does it recover the planted groups better?")
    print(f"   {'condition':<34}{'plain':>8}{'sparse':>8}")
    for tag, nr, nn, sd in (("6 real, no noise", 6, 0, .7), ("3 real + 9 noise", 3, 9, .7),
                            ("2 real + 10 noise, weak", 2, 10, 1.2)):
        P, S = [], []
        for seed in range(3):
            X, truth = planted(seed, nr, nn, sd)
            P.append(ari(truth, KMeans(3, n_init=10, random_state=0).fit(X).labels_))
            S.append(ari(truth, weighting(X, 3, np.random.default_rng(seed),
                                          n_permutations=6)["labels"]))
        print(f"   {tag:<34}{np.mean(P):>8.2f}{np.mean(S):>8.2f}")

    print("\n2. On PURE NOISE, does it flatter the separation?")
    print(f"   {'questions':<34}{'plain sil':>10}{'sparse sil':>12}")
    for items in (6, 12, 18):
        P, S = [], []
        for seed in range(3):
            X, _ = planted(seed, 0, items, 0)
            P.append(silhouette_score(X, KMeans(3, n_init=10, random_state=0).fit(X).labels_))
            r = weighting(X, 3, np.random.default_rng(seed), n_permutations=6)
            S.append(silhouette_score(X * np.sqrt(r["weights"]), r["labels"]))
        print(f"   {items:<34}{np.mean(P):>10.2f}{np.mean(S):>12.2f}")

    print("\n3. Can the weights tell real from noise? (share held by the top three)")
    for tag, nr, nn in (("3 real + 9 noise", 3, 9), ("PURE NOISE, 12 questions", 0, 12)):
        share = []
        for seed in range(3):
            X, _ = planted(seed, nr, nn, .7)
            w = weighting(X, 3, np.random.default_rng(seed), n_permutations=6)["weights"]
            share.append(np.sort(w)[::-1][:3].sum() / max(w.sum(), 1e-12))
        print(f"   {tag:<34}{np.mean(share):>9.0%}")
