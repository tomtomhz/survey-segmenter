"""Which few things should you actually launch? (TURF: total unduplicated reach and frequency.)

A best-worst study says which items people want on average. That is not the same question as which
SET to offer, and the difference is the whole point of this module. The three highest-scoring
flavours might all appeal to the same people; a lower-scoring fourth might be the only thing the
rest of your customers will touch. Reach counts people, not preference, so it answers "what do we
put on the menu" where the ranking answers "what do people like".

**The number everyone quotes is biased, and this says by how much.**

TURF searches every combination and reports the best one. A maximum taken over hundreds of
candidates is optimistic by construction: some of the winner's apparent lead is real preference and
some is the luck of this particular sample, and the search deliberately seeks out whichever
combination got the most of both. Quote the raw figure to a board and you are quoting the sample's
best case as though it were the world's average.

So every result here carries a **holdout** estimate as well. The combination is chosen on one half
of the respondents and its reach is measured on the other half, repeated over many splits. The gap
between the two is the optimism, stated rather than absorbed.

The gap is far larger than "a rounding error", which is the assumption worth killing. Measured on
simulated studies, choosing three items (percentage points of overstatement):

                        real taste groups        pure noise
        100 people, 10 items      9.5                15.6
        100 people, 20 items     22.3                20.7
        300 people, 20 items      7.8                11.7
      1,000 people, 10 items      2.4                 4.9

Two things follow. It gets worse as the item list grows — which is exactly when TURF is most
tempting — and at a hundred respondents with twenty items the headline overstates reach by more
than twenty points, which is the difference between a launch decision and a mistake. It shrinks
with sample size because there is less luck left to find.

**What "reached" means, and why it is a choice rather than a fact.** A person is counted as reached
by an item when that item is among their own top few. It has to be a rule of that kind: utilities
are relative and centred, so "likes it" has no absolute meaning, and a threshold like "utility
above zero" would simply count how many items sit above each person's personal average. The default
is the top three, which is the convention in commercial practice, and it is reported alongside the
answer because a different rule gives a different winner and the reader is entitled to know which
one produced theirs.
"""
from __future__ import annotations

import itertools

import numpy as np

#: How many of a respondent's own favourites count as "would take it". Three is the commercial
#: convention. It is stated in the output because the answer depends on it.
DEFAULT_TOP_N = 3

#: Above this many combinations, stop enumerating and go greedy. Exhaustive search is exact and
#: worth having whenever it is affordable — at twenty items choosing three it is barely a thousand
#: candidates — but it grows as a binomial and a large item list would otherwise hang the app.
MAX_EXHAUSTIVE = 200_000


def reach_matrix(utilities, top_n=DEFAULT_TOP_N):
    """Who would take what: a people x items matrix of True where the item is in that person's top N.

    Ties are broken by taking the first `top_n` in argsort order, which matters only when a
    respondent rates several items identically — with continuous utilities from the sampler that is
    vanishingly rare, and any tie-break is arbitrary anyway.
    """
    u = np.asarray(utilities, dtype=float)
    if u.ndim != 2:
        raise ValueError("utilities must be a respondents x items matrix")
    take = max(1, min(int(top_n), u.shape[1]))
    # argsort descending, then mark the first `take` columns for each row.
    order = np.argsort(-u, axis=1)[:, :take]
    hit = np.zeros(u.shape, dtype=bool)
    np.put_along_axis(hit, order, True, axis=1)
    return hit


def _reach_of(hit, combo):
    """Share of people reached by at least one item in `combo`. Unduplicated: each person once."""
    return float(hit[:, list(combo)].any(axis=1).mean())


def best_combination(hit, size):
    """The set of `size` items reaching the most people, and how it was found.

    Exhaustive while that is affordable, greedy beyond. Greedy is not guaranteed optimal, and the
    result says which was used rather than letting a reader assume the answer is exact.
    """
    n_items = hit.shape[1]
    size = max(1, min(int(size), n_items))
    from math import comb
    if comb(n_items, size) <= MAX_EXHAUSTIVE:
        best, best_reach = None, -1.0
        for combo in itertools.combinations(range(n_items), size):
            score = _reach_of(hit, combo)
            if score > best_reach:
                best, best_reach = combo, score
        return list(best), best_reach, "exhaustive"

    chosen = []
    for _ in range(size):
        candidates = [i for i in range(n_items) if i not in chosen]
        gains = [(_reach_of(hit, chosen + [i]), i) for i in candidates]
        gains.sort(reverse=True)
        chosen.append(gains[0][1])
    return sorted(chosen), _reach_of(hit, chosen), "greedy"


def holdout_reach(hit, size, splits=40, seed=0):
    """How well the winning combination does on people it was NOT chosen from.

    This is the honest number. Choosing and scoring on the same respondents reports a maximum over
    every candidate combination, which flatters whichever one happened to suit this sample; picking
    on one half and measuring on the other removes exactly that advantage.

    Returns (mean holdout reach, mean in-sample reach on the same halves). The DIFFERENCE is the
    optimism — reporting the holdout figure alone would hide how much the headline overstates.
    """
    rng = np.random.default_rng(seed)
    n_people = hit.shape[0]
    if n_people < 20:
        return float("nan"), float("nan")
    out, inside = [], []
    for _ in range(splits):
        order = rng.permutation(n_people)
        half = n_people // 2
        pick_on, score_on = hit[order[:half]], hit[order[half:]]
        combo, in_reach, _ = best_combination(pick_on, size)
        inside.append(in_reach)
        out.append(_reach_of(score_on, combo))
    return float(np.mean(out)), float(np.mean(inside))


def turf(utilities, item_names, size=3, top_n=DEFAULT_TOP_N, splits=40, seed=0):
    """The whole answer: which items to launch, who they reach, and how much of that is luck.

    `incremental` is what each chosen item ADDS over the ones before it, in the order chosen. That
    is usually the most actionable column: an item contributing a percentage point or two is
    carrying almost nobody the others do not already carry, and is a candidate to drop.
    """
    hit = reach_matrix(utilities, top_n=top_n)
    combo, reach, how = best_combination(hit, size)
    held, in_sample = holdout_reach(hit, size, splits=splits, seed=seed)

    # Order the winners by what each adds on top of those already counted.
    remaining, ordered, running = list(combo), [], 0.0
    while remaining:
        gains = [(_reach_of(hit, [i for i, _ in ordered] + [c]), c) for c in remaining]
        gains.sort(reverse=True)
        total, pick = gains[0]
        ordered.append((pick, total - running))
        running = total
        remaining.remove(pick)

    return {
        "items": [item_names[i] for i, _ in ordered],
        "indices": [i for i, _ in ordered],
        "reach": reach,
        "incremental": [gain for _, gain in ordered],
        "alone": [float(hit[:, i].mean()) for i, _ in ordered],
        "holdout_reach": held,
        "in_sample_reach": in_sample,
        "optimism": (in_sample - held) if in_sample == in_sample else float("nan"),
        "search": how,
        "top_n": top_n,
        "size": len(combo),
        "n_people": int(hit.shape[0]),
        "n_items": int(hit.shape[1]),
    }
