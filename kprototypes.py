"""Grouping people when the survey mixes rating scales with pick-any questions.

Until now a survey had to pick a side. All rating questions went to k-means; all multiple-choice
questions went to latent class analysis; and a survey containing both had the smaller half set
aside with an apologetic note. That is the largest capability gap the tool had, because real
questionnaires are mixed almost by default — a few 1-5 agreement scales, one "which of these do
you buy", one "how often do you travel".

The published answer is k-prototypes under Gower's distance:

    Szepannek, Aschenbruck & Wilhelm, *Clustering large mixed-type data with ordinal variables*,
    Advances in Data Analysis and Classification 19(3), 2024, and its implementation as
    `kproto(type = "gower")` in the R package `clustMixType`.

Three things make it the right choice rather than one more variant:

1.  **Every question contributes on the same scale.** Gower scores each variable between 0 and 1
    in its own natural way, so a rating question and a brand question can be averaged without
    inventing an exchange rate between them. Huang's original k-prototypes needs a `gamma`
    weight to trade numeric distance against categorical mismatches, and nobody in a marketing
    meeting can defend the number they picked for it.

2.  **Ordinal answers are treated as ordinal.** A 1-5 agreement scale is not a measurement and it
    is not a set of unrelated labels. Here it becomes ranks: each level is replaced by its
    midrank over all respondents, and distance is the range-normalised gap between midranks. The
    order is respected, and the spacing follows how many people actually sit between two answers
    rather than assuming the gap from "agree" to "strongly agree" equals the gap from "neutral"
    to "agree".

3.  **It converges, provably.** This is the 2024 paper's contribution and the reason the update
    rules below are not the obvious ones. Gower is an L1-type distance, so the value minimising
    the within-cluster sum is the **median**, not the mean; for a nominal variable it is the
    **mode**; for an ordinal variable it is the level whose rank is closest to the cluster's
    median rank. With those three, the update step can never increase the objective, the
    assignment step cannot either, and there are finitely many partitions — so the algorithm
    stops. Using means here, as a naive port of k-means would, breaks that and can oscillate.

**On Podani's two versions.** Podani (1999) extended Gower to ordinal characters in two forms.
The tie-corrected non-metric version (his equation 2b) is the more commonly cited, and it is the
wrong one here: it subtracts the within-tie spread from every gap, which on survey data with a
handful of levels and hundreds of respondents collapses adjacent answers to near-zero distance
while leaving the extremes a full 1 apart. Worked through for three equally-used levels, it gives
d(1,3) = 1.00 against d(1,2) + d(2,3) = 0.065 — a violation of the triangle inequality so large
that "closest prototype" stops meaning anything, which is precisely why Podani calls it
non-metric. This module uses his metric version (equation 3), plain midranks over the rank range,
which is also what `clustMixType` uses under `type = "gower"`.

The distance is deliberately computed against prototypes only — O(n·k·p) — never as a full
pairwise matrix, which at O(n²·p) is what stops most Gower implementations well before the sample
sizes this tool has to handle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUMERIC = "numeric"
ORDINAL = "ordinal"
NOMINAL = "nominal"
KINDS = (NUMERIC, ORDINAL, NOMINAL)

# Distances are built in blocks of rows rather than one (n, k, p) array, which at 100,000
# respondents would be gigabytes for no benefit.
_ROW_BLOCK = 4096

#: Stand-in for an answer to a pick-any question that nobody gave when the segmentation was built,
#: which can only happen when scoring new people. It has to be a level of its own rather than a
#: blank: someone who names a brand the original sample never mentioned genuinely differs from
#: everyone, and should sit a full mismatch away from every known answer rather than half of one.
UNSEEN = -np.inf


@dataclass
class GowerSpec:
    """Per-variable constants, fitted once on the whole sample and then reused everywhere.

    Fitting these on the full data and holding them fixed is what makes a distance computed on a
    bootstrap resample comparable to one computed on the whole file. Refitting the ranges per
    subsample would quietly rescale the space under the stability tests, which exist precisely to
    measure whether the answer moves.
    """

    kinds: tuple[str, ...]
    #: Divisor per column putting every difference in 0..1. 1.0 where the column never varies.
    scale: np.ndarray
    #: The typical answer to each question in the original study, on the raw answer scale: the
    #: median for a rating, UNSEEN for a pick-any. What a skipped answer falls back to when
    #: scoring new people, so that somebody's segment does not depend on who else happened to be
    #: uploaded alongside them.
    centre: np.ndarray = field(default_factory=lambda: np.array([], float))
    #: Column indices, split by how their distance is computed. Numeric and ordinal share a rule
    #: once ordinal levels have become midranks, so they travel together here.
    span_cols: np.ndarray = field(default_factory=lambda: np.array([], int))
    match_cols: np.ndarray = field(default_factory=lambda: np.array([], int))
    #: For each ordinal column, the sorted distinct midranks its levels map to. A prototype has
    #: to land on a level a respondent could actually have chosen, so updates snap to these.
    ord_levels: dict[int, np.ndarray] = field(default_factory=dict)
    #: For each ordinal column, value -> midrank, so new data can be encoded the same way.
    ord_ranks: dict[int, dict[float, float]] = field(default_factory=dict)
    #: For each nominal column, the levels seen when the spec was fitted, plus UNSEEN. Held here
    #: rather than recomputed per array so that a bootstrap resample, a reference sample and a new
    #: respondent all embed into the same columns — otherwise a level absent from one of them
    #: would shift every coordinate after it and quietly compare two different spaces.
    nom_levels: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return len(self.kinds)

    def to_json(self):
        """Plain JSON, so a saved typing rule stays a readable file rather than a pickle.

        A pickle would bind the exported rule to this exact class in this exact version, which is
        the opposite of what an exported rule is for: somebody should be able to open it, read
        what it says, and still use it after the tool has moved on. UNSEEN is -inf, which JSON has
        no literal for, so it travels as a string and comes back as a float.
        """
        return {"kinds": list(self.kinds), "scale": [float(v) for v in self.scale],
                "centre": ["-inf" if np.isneginf(v) else float(v) for v in self.centre],
                "ord_ranks": {str(j): {str(k): float(v) for k, v in t.items()}
                              for j, t in self.ord_ranks.items()},
                "nom_levels": {str(j): ["-inf" if np.isneginf(v) else float(v) for v in levels]
                               for j, levels in self.nom_levels.items()}}

    @classmethod
    def from_json(cls, blob):
        kinds = tuple(blob["kinds"])
        ord_ranks = {int(j): {float(k): float(v) for k, v in t.items()}
                     for j, t in blob.get("ord_ranks", {}).items()}
        return cls(kinds=kinds, scale=np.asarray(blob["scale"], float),
                   centre=np.array([UNSEEN if v == "-inf" else float(v)
                                    for v in blob.get("centre", [])], float),
                   span_cols=np.array([j for j, k in enumerate(kinds) if k != NOMINAL], int),
                   match_cols=np.array([j for j, k in enumerate(kinds) if k == NOMINAL], int),
                   ord_levels={j: np.array(sorted(t.values()), float)
                               for j, t in ord_ranks.items()},
                   ord_ranks=ord_ranks,
                   nom_levels={int(j): np.array(
                       [UNSEEN if v == "-inf" else float(v) for v in levels], float)
                       for j, levels in blob.get("nom_levels", {}).items()})


def _midranks(col):
    """Map each distinct value to its midrank over the sample: value -> mean rank of its ties.

    Midranks rather than level numbers is the whole of "treated as ordinal". On a 1-5 scale where
    almost everyone answers 4 or 5, the 4-to-5 gap is genuinely large in rank terms and the 1-to-2
    gap is small, because hardly anyone sits between them. Podani (1999), metric version.
    """
    values = np.asarray(col, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    table = {}
    for v in np.unique(values):
        table[float(v)] = float(ranks[values == v].mean())
    return table


def fit_spec(X, kinds):
    """Measure the constants Gower needs. X is (n, p); kinds names each column's type."""
    X = np.asarray(X, float)
    kinds = tuple(kinds)
    if X.ndim != 2 or X.shape[1] != len(kinds):
        raise ValueError(f"kinds describes {len(kinds)} columns, data has {X.shape}")
    bad = sorted(set(kinds) - set(KINDS))
    if bad:
        raise ValueError(f"unknown variable kind(s): {bad}")

    scale = np.ones(len(kinds), float)
    centre = np.full(len(kinds), UNSEEN, float)
    span, match, ord_levels, ord_ranks, nom_levels = [], [], {}, {}, {}
    for j, kind in enumerate(kinds):
        if kind == NOMINAL:
            match.append(j)
            nom_levels[j] = np.append(np.unique(X[:, j]), UNSEEN)
            continue
        centre[j] = float(np.median(X[:, j])) if len(X) else 0.0
        span.append(j)
        col = X[:, j]
        if kind == ORDINAL:
            table = _midranks(col)
            ord_ranks[j] = table
            ord_levels[j] = np.array(sorted(table.values()), float)
            col = np.array([table[float(v)] for v in col], float)
        spread = float(col.max() - col.min()) if len(col) else 0.0
        # A question everybody answered identically carries no information. Leaving the divisor at
        # 1 makes its contribution exactly zero rather than a division by zero.
        scale[j] = spread if spread > 0 else 1.0
    return GowerSpec(kinds=kinds, scale=scale, span_cols=np.array(span, int),
                     match_cols=np.array(match, int), ord_levels=ord_levels, ord_ranks=ord_ranks,
                     nom_levels=nom_levels, centre=centre)


def encode(X, spec):
    """Put the data in the form the distance works on: ordinal levels become their midranks.

    An ordinal value never seen when the spec was fitted — which only happens when scoring new
    people — is placed at the nearest known level rather than rejected, because refusing to score
    somebody over one unusual answer is worse than placing them slightly imprecisely. An
    unrecognised nominal value has no "nearest", so it becomes UNSEEN instead.

    Call this on raw answers, once. It is deliberately not idempotent: encoding replaces ordinal
    levels with midranks, which are themselves numbers, so encoding twice would snap those
    midranks onto the level scale and quietly corrupt the column.
    """
    Xe = np.array(X, float, copy=True)
    for j, table in spec.ord_ranks.items():
        known = np.array(sorted(table), float)
        ranks = np.array([table[float(v)] for v in known], float)
        col = Xe[:, j]
        # Nearest known level, by binary search. This was
        #     np.abs(col[:, None] - known[None, :]).argmin(1)
        # which allocates one number per respondent PER LEVEL. Harmless on a five-point scale;
        # 19 GB on a column holding 48,842 distinct values, which is what a continuous measurement
        # looks like once something has typed it as ordinal. Measured on the UCI adult file, this
        # single line accounted for the whole 11 GB peak of a run.
        #
        # `known` is sorted, so searchsorted gives the insertion point and the answer is whichever
        # neighbour is nearer — the same level the scan picked, keeping its habit of taking the
        # lower one when a value falls exactly between two, in O(n log levels) time and O(n) space.
        if len(known) == 1:
            Xe[:, j] = ranks[0]
            continue
        pos = np.clip(np.searchsorted(known, col), 1, len(known) - 1)
        take_left = np.abs(col - known[pos - 1]) <= np.abs(col - known[pos])
        Xe[:, j] = ranks[np.where(take_left, pos - 1, pos)]
    for j, levels in spec.nom_levels.items():
        # A pick-any answer has no nearest neighbour to fall back on, so anything unrecognised
        # becomes its own level rather than being forced onto whichever code sorts closest.
        Xe[~np.isin(Xe[:, j], levels), j] = UNSEEN
    return Xe


def gower_distances(Xe, prototypes, spec):
    """(n, k) Gower distances from every respondent to every prototype, each in 0..1.

    The mean over variables of: the range-normalised absolute gap for numeric and ordinal
    columns, and 1-if-different for nominal ones.
    """
    Xe = np.asarray(Xe, float)
    P = np.asarray(prototypes, float)
    if P.ndim == 1:
        P = P[None, :]
    n, k = len(Xe), len(P)
    out = np.empty((n, k), float)
    span, match = spec.span_cols, spec.match_cols
    inv = 1.0 / spec.scale[span] if len(span) else None
    for lo in range(0, n, _ROW_BLOCK):
        hi = min(lo + _ROW_BLOCK, n)
        total = np.zeros((hi - lo, k), float)
        if len(span):
            total += (np.abs(Xe[lo:hi, None, span] - P[None, :, span]) * inv).sum(2)
        if len(match):
            total += (Xe[lo:hi, None, match] != P[None, :, match]).sum(2)
        out[lo:hi] = total / spec.n_vars
    return out


def gower_embedding(Xe, spec):
    """Coordinates on which plain Manhattan distance reproduces Gower's distance exactly.

    Gower is not a metric people usually think of as having coordinates, but it does: divide each
    numeric or ordinal column by its range, and replace each nominal column by half a one-hot
    indicator. Two respondents differing on a nominal question then sit 0.5 + 0.5 = 1 apart on
    that block, which is the simple matching distance, and every other column is already its
    normalised gap. So

        gower(i, j) == |z_i - z_j|_1 / (number of variables)

    exactly, which `test_gower_is_exactly_a_manhattan_distance_in_disguise` pins.

    This matters well beyond tidiness. It means the silhouette, the cluster-tendency test, the
    hierarchical cross-check and the segment map can all run on a mixed-type segmentation using
    the same library functions as the numeric path, asking only for `metric="manhattan"`, instead
    of a second family of hand-written Gower versions that would need their own tests and would
    drift from the originals.
    """
    Xe = np.asarray(Xe, float)
    blocks = []
    for j, kind in enumerate(spec.kinds):
        if kind == NOMINAL:
            levels = spec.nom_levels[j]
            blocks.append((Xe[:, j][:, None] == levels[None, :]).astype(float) * 0.5)
        else:
            blocks.append(Xe[:, j][:, None] / spec.scale[j])
    return np.hstack(blocks) if blocks else np.zeros((len(Xe), 0))


def reference_sample(Xe, spec, m, rng):
    """A "no structure here" sample: every question answered independently of every other.

    Each variable keeps its own observed support — a rating stays inside its range, a brand
    question still picks one of the brands people actually named — but the answers stop travelling
    together. That is exactly the null the cluster-tendency test and the gap statistic need, and
    building it column by column is Tibshirani-Walther-Hastie's reference method (a). Uniform over
    a bounding box, the numeric shortcut, is not available here: no box contains only legal
    combinations once a column is a brand rather than a number.
    """
    Xe = np.asarray(Xe, float)
    out = np.empty((m, spec.n_vars), float)
    for j, kind in enumerate(spec.kinds):
        col = Xe[:, j]
        if kind == NUMERIC:
            out[:, j] = rng.uniform(col.min(), col.max(), m)
        else:
            # Ordinal levels are midranks and nominal levels are codes; either way a respondent
            # can only ever hold one of the values actually observed.
            out[:, j] = rng.choice(np.unique(col), m)
    return out


def _update(members, spec):
    """The prototype minimising Gower distance to a set of members. Szepannek et al. (2024).

    Median for numeric, mode for nominal, and for ordinal the level whose rank sits closest to the
    members' median rank. These are the exact minimisers for their respective terms, which is what
    makes the algorithm provably stop rather than merely usually stop.
    """
    proto = np.empty(spec.n_vars, float)
    for j, kind in enumerate(spec.kinds):
        col = members[:, j]
        if kind == NOMINAL:
            values, counts = np.unique(col, return_counts=True)
            proto[j] = values[counts.argmax()]
        elif kind == ORDINAL:
            levels = spec.ord_levels[j]
            proto[j] = levels[np.abs(levels - np.median(col)).argmin()]
        else:
            proto[j] = np.median(col)
    return proto


def _seed(Xe, k, spec, rng):
    """k-means++ seeding, with Gower in place of squared Euclidean.

    Prototypes start as real respondents, which for a mixed data set is the only starting point
    guaranteed to be a legal combination of answers — an averaged one need not be.
    """
    n = len(Xe)
    first = int(rng.integers(n))
    chosen = [first]
    closest = gower_distances(Xe, Xe[[first]], spec)[:, 0]
    while len(chosen) < k:
        weights = closest ** 2
        total = weights.sum()
        # Every remaining point already sits on a prototype: the data has fewer distinct answer
        # patterns than requested groups, so pick the rest at random rather than divide by zero.
        pick = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=weights / total))
        chosen.append(pick)
        closest = np.minimum(closest, gower_distances(Xe, Xe[[pick]], spec)[:, 0])
    return Xe[chosen].copy()


def _fit_once(Xe, k, spec, rng, max_iter):
    protos = _seed(Xe, k, spec, rng)
    labels = np.zeros(len(Xe), int)
    for it in range(max_iter):
        D = gower_distances(Xe, protos, spec)
        new = D.argmin(1)
        if it and np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = Xe[labels == c]
            if len(members):
                protos[c] = _update(members, spec)
            else:
                # An emptied cluster would silently reduce k. Restart it on the respondent least
                # well served by the prototypes there are, which is where a group is most needed.
                far = int(D[np.arange(len(Xe)), labels].argmax())
                if D[far, labels[far]] <= 0:
                    # Everyone already sits exactly on a prototype, so there is no unexplained
                    # respondent to build a group around. Moving one anyway just hands them back
                    # next iteration: the reseeding fights the assignment step, labels never
                    # settle, and the loop runs to max_iter every time on any file with fewer
                    # distinct answer patterns than groups asked for. Leave it empty instead —
                    # the honest answer is that this many groups do not exist here.
                    continue
                protos[c] = Xe[far]
                labels[far] = c
    D = gower_distances(Xe, protos, spec)
    labels = D.argmin(1)
    return protos, labels, float(D[np.arange(len(Xe)), labels].sum()), it + 1


class KPrototypes:
    """Gower k-prototypes, exposing the same surface as scikit-learn's KMeans.

    `labels_`, `cluster_centers_`, `inertia_` and `predict()` are named to match KMeans because
    every stability check in this tool — prediction strength, split-half replication, bootstrap
    Jaccard, the final fit — goes through one base learner. Matching the interface means those
    checks apply to a mixed-type segmentation unchanged, rather than a second, less tested path
    growing up beside them.

    `inertia_` is the total Gower distance to the assigned prototype: an L1-type objective, not a
    sum of squares, so it is not comparable with a k-means inertia and is only ever used to
    compare restarts of this algorithm against each other.
    """

    def __init__(self, n_clusters, spec, n_init=10, max_iter=100, random_state=0):
        self.n_clusters = int(n_clusters)
        self.spec = spec
        self.n_init = max(1, int(n_init))
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

    def fit(self, Xe, y=None):
        Xe = np.asarray(Xe, float)
        if len(Xe) < self.n_clusters:
            raise ValueError(f"{len(Xe)} respondents cannot form {self.n_clusters} groups")
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            got = _fit_once(Xe, self.n_clusters, self.spec, rng, self.max_iter)
            if best is None or got[2] < best[2]:
                best = got
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, Xe):
        return gower_distances(np.asarray(Xe, float), self.cluster_centers_, self.spec).argmin(1)

    def fit_predict(self, Xe, y=None):
        return self.fit(Xe).labels_
