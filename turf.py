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
between the headline and that holdout is the optimism, stated rather than absorbed.

**These numbers are measured against a known truth, not against each other.** Each cell draws a
sample from a 40,000-person population, runs the whole thing on the sample, and then looks up what
the chosen combination really reaches in the population — so "real error" is a fact rather than an
estimate. Ten draws per cell, choosing three items, in percentage points:

                              gap this reports   headline's real error
    real taste groups
        100 people, 10 items         3.6                 +4.6
        100 people, 20 items         5.0                 +2.9
        300 people, 20 items         2.1                 -1.0
      1,000 people, 10 items         0.2                 +0.3
    pure noise
        100 people, 10 items        10.6                 +9.0
        100 people, 20 items        14.8                +14.0
        300 people, 20 items         7.8                 +8.1
      1,000 people, 10 items         3.5                 +3.2

Three things follow. The reported gap tracks the real error rather than merely gesturing at it,
which is what makes it quotable. It is largest when the structure is weakest — on data with no
taste groups at all the headline overstates by nine to fourteen points, because there is nothing
but luck for the search to find. And it shrinks with sample size, for the same reason.

**An earlier version of this table was wrong, and how it was wrong is worth keeping.** It reported
9.5 to 22.3 points, because the gap was computed as `in_sample - holdout` — both of them
half-sample figures, so it measured the optimism of a study half this size and attributed it to the
full-sample headline. The report then printed "Expect about 93%, not 95%" directly above "the
3-point difference", two sentences that do not agree, with the 96% those 3 points came from
appearing nowhere. The lesson is the one this project keeps relearning: a plausible number nobody
checked against an outside truth is not evidence.

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


#: How many equally-best combinations to keep hold of, so the report can show a few without
#: carrying thousands of them around when almost everything ties.
MAX_TIES_KEPT = 12


def best_combination(hit, size):
    """The set of `size` items reaching the most people, how it was found, and what else tied.

    Exhaustive while that is affordable, greedy beyond. Greedy is not guaranteed optimal, and the
    result says which was used rather than letting a reader assume the answer is exact.

    **The ties are returned because they are the difference between an answer and a coin toss.**
    Reach is a count of people, so it lands on multiples of 1/n: with sixty respondents there are
    only sixty-one possible values and hundreds of candidate combinations, so exact ties at the top
    are ordinary rather than freakish. Measured on simulated studies with real taste groups, the
    best reach was shared by more than one combination in 14 of 30 studies at sixty people and ten
    items, and 10 of 30 at a hundred people and twenty items.

    Whichever tied combination is reported is then decided by the order the items happened to
    appear in the file — and reordering the list changed the recommendation in 8 of 25 studies.
    That is not a tie-break worth improving, because every tie-break is arbitrary; the honest move
    is to say a tie happened and let the reader choose on grounds this data does not contain, like
    cost or brand fit.
    """
    n_items = hit.shape[1]
    size = max(1, min(int(size), n_items))
    from math import comb
    if comb(n_items, size) <= MAX_EXHAUSTIVE:
        best, best_reach, tied = None, -1.0, []
        for combo in itertools.combinations(range(n_items), size):
            score = _reach_of(hit, combo)
            if score > best_reach:
                best, best_reach, tied = combo, score, [combo]
            elif score == best_reach and len(tied) < MAX_TIES_KEPT:
                tied.append(combo)
        return list(best), best_reach, "exhaustive", [list(c) for c in tied]

    chosen = []
    for _ in range(size):
        candidates = [i for i in range(n_items) if i not in chosen]
        # Sorted on (-reach, index) so a tie resolves to the lowest item index, matching what the
        # exhaustive branch returns. It was `sort(reverse=True)`, which took the HIGHEST index on a
        # tie, so the two search paths silently disagreed about which of two equal items to pick.
        gains = sorted(((-_reach_of(hit, chosen + [i]), i) for i in candidates))
        chosen.append(gains[0][1])
    # Greedy never enumerates the alternatives, so it cannot count the ties. Reported as unknown
    # rather than as zero: "no ties" and "not looked" are different claims.
    return sorted(chosen), _reach_of(hit, chosen), "greedy", None


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
        combo, in_reach, _, _ = best_combination(pick_on, size)
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
    combo, reach, how, tied = best_combination(hit, size)
    held, in_sample = holdout_reach(hit, size, splits=splits, seed=seed)

    # Order the winners by what each adds on top of those already counted. Sorted on (-total,
    # index) for the same reason as the greedy search: a tie here decided the ORDER items are
    # listed in, and "reverse=True" resolved it to the highest index, which is a different
    # arbitrary answer from the one the search itself gives.
    remaining, ordered, running = list(combo), [], 0.0
    while remaining:
        gains = sorted((-_reach_of(hit, [i for i, _ in ordered] + [c]), c) for c in remaining)
        total, pick = -gains[0][0], gains[0][1]
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
        # The gap between the HEADLINE and the holdout — the two figures the report actually shows.
        #
        # It used to be `in_sample - held`, and both of those are half-sample quantities: the
        # optimism of a study half this size, quietly attributed to a full-sample number that never
        # had it. The report then printed "Expect about 93%, not 95%" directly above "the 3-point
        # difference", which do not agree, and the 96% the 3 points were measured from appeared
        # nowhere. Measured against a known population, `reach - held` is also the closer estimate
        # of the headline's real error in three of the four shapes tried; see the table above.
        "optimism": (reach - held) if held == held else float("nan"),
        "search": how,
        # None when the search was greedy and never looked. 1 means the winner is genuinely alone;
        # anything more means the reported set is one of several that reach exactly as many people,
        # and which one was printed came down to the order of the item list.
        "n_tied": (len(tied) if tied is not None else None),
        "tied_items": ([[item_names[i] for i in c] for c in tied[:MAX_TIES_KEPT]]
                       if tied is not None and len(tied) > 1 else []),
        "tie_capped": bool(tied is not None and len(tied) >= MAX_TIES_KEPT),
        "top_n": top_n,
        "size": len(combo),
        "n_people": int(hit.shape[0]),
        "n_items": int(hit.shape[1]),
    }
