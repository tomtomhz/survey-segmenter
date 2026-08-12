"""Build the best-worst questionnaire itself: which items appear on which screen, for whom.

Until this existed the tool could read a MaxDiff study but not help you run one. That is a strange
place to stop — the design decides what the data can possibly say, and getting it wrong is not
recoverable by any amount of analysis afterwards. A questionnaire where two items never appear
together cannot tell you which of them people prefer, however many respondents answer it.

**What a design has to get right, and what each failure costs.**

* **Every item shown about equally often.** An item seen half as much as the rest is estimated half
  as precisely, and its position in the final ranking partly reflects how often it was asked about.
* **Every PAIR shown together about equally often.** This is the one people forget. Best-worst data
  is comparisons; a pair that never co-occurs is compared only through other items, and a pair that
  co-occurs constantly is measured far more sharply than the rest. Imbalance here distorts the
  ranking in a way that looks like preference.
* **A different arrangement per respondent.** Everyone answering the identical screens ties the
  design to any order effect: if screen three is always the same four items, fatigue on screen
  three becomes an opinion about those items.

**Why this is not simply a textbook BIBD.** A balanced incomplete block design gives exact pairwise
balance, but only exists for particular combinations of item count, set size and number of sets —
and a real study fixes those from how long people will sit still, not from what number theory
allows. So this searches for the most balanced design of the shape you actually want, by swapping
items between sets while the balance improves, and then REPORTS what it achieved rather than
claiming a balance it may not have reached.

**The honesty is in the report.** Every design comes back with its worst item-frequency spread and
its worst pair spread, and the wording says plainly when a shape cannot be balanced — usually
because the slots per respondent are not a multiple of the item count, which is arithmetic and no
amount of searching fixes it.
"""
from __future__ import annotations

import numpy as np


def _pair_counts(design, n_items):
    """How often each pair of items appears on the same screen, across the whole design."""
    counts = np.zeros((n_items, n_items), dtype=int)
    for respondent in design:
        for task in respondent:
            for a_index, a in enumerate(task):
                for b in task[a_index + 1:]:
                    counts[a, b] += 1
                    counts[b, a] += 1
    return counts


def make_design(n_items, items_per_set=4, sets_per_respondent=10, n_respondents=200, seed=0,
                passes=2):
    """Build a best-worst design and measure how balanced it actually came out.

    Each respondent gets their own arrangement, built by dealing items out in a rotating order so
    every item appears as evenly as the arithmetic allows, then improved by swapping items between
    that person's screens while the swap reduces overall imbalance.

    Returns (design, report). `design` is a list per respondent of lists per screen of item indices.
    """
    if items_per_set < 2:
        raise ValueError("_DESIGN_SET_TOO_SMALL")
    if n_items <= items_per_set:
        raise ValueError(f"_DESIGN_TOO_FEW_ITEMS:{n_items}:{items_per_set}")

    rng = np.random.default_rng(seed)
    slots = sets_per_respondent * items_per_set
    design = []
    for _ in range(n_respondents):
        # Deal from a repeatedly reshuffled deck: this alone gets item frequency close to even,
        # because an item cannot appear twice before every other item has appeared once.
        deck = []
        while len(deck) < slots:
            order = rng.permutation(n_items)
            deck.extend(order.tolist())
        person = [deck[i * items_per_set:(i + 1) * items_per_set]
                  for i in range(sets_per_respondent)]
        # A screen must not show the same item twice; the deck can straddle a reshuffle boundary.
        for task in person:
            seen = set()
            for position, item in enumerate(task):
                if item in seen:
                    spare = [c for c in range(n_items) if c not in seen and c not in task]
                    if spare:
                        task[position] = int(rng.choice(spare))
                seen.add(task[position])
        design.append(person)

    # Swap items between one person's screens while it improves pair balance.
    #
    # Scored INCREMENTALLY, which is the difference between usable and not. Rebuilding the whole
    # pair matrix for every candidate swap took twenty-four seconds for sixty respondents and grows
    # with the sample, so a real study would never finish. Two observations make it cheap: a swap
    # moves items between screens without changing how often any item appears, so item balance is
    # invariant and drops out of the objective entirely; and only the pairs inside the two screens
    # being touched can change, which is a few dozen entries rather than the entire design.
    #
    # Measured on twelve items over sixty respondents, this brings the worst pair spread from
    # 37-71 down to 51-59 — the imbalance that most distorts a ranking — and a second pass changes
    # nothing, so one is the default.
    # The OBJECTIVE is maintained incrementally as well, which is the second half of the same idea
    # and matters far more as the item count grows. Rebuilding a standard deviation over the whole
    # pair matrix costs O(items^2) per candidate swap while the swap itself touches a few dozen
    # entries: sixty items over five hundred people took five minutes, nearly all of it spent
    # re-reading numbers that had not changed.
    #
    # A swap moves items between screens without changing how many pairs a screen contains, so the
    # TOTAL of the pair counts is invariant — and with the mean fixed, minimising the variance is
    # exactly minimising the sum of squares. So that sum is all that has to be carried, and each
    # changed entry updates it in constant time. Locked in by
    # test_a_swap_never_changes_the_total_number_of_pairings, because the whole shortcut rests on it.
    pairs = _pair_counts(design, n_items)
    off_mask = ~np.eye(n_items, dtype=bool)
    sumsq = float((pairs[off_mask].astype(float) ** 2).sum())

    def apply_pairs(task, item, delta):
        """Add `delta` to every pair between `item` and the rest of `task`, keeping `sumsq` true."""
        nonlocal sumsq
        for other in task:
            if other != item:
                old = int(pairs[item, other])
                new = old + delta
                pairs[item, other] = new
                pairs[other, item] = new
                sumsq += 2.0 * float(new * new - old * old)

    best = sumsq
    for _ in range(passes):
        improved = False
        for person in design:
            for a in range(len(person)):
                for b in range(a + 1, len(person)):
                    for i in range(len(person[a])):
                        for j in range(len(person[b])):
                            x, y = person[a][i], person[b][j]
                            if x == y or y in person[a] or x in person[b]:
                                continue
                            apply_pairs(person[a], x, -1)
                            apply_pairs(person[b], y, -1)
                            person[a][i], person[b][j] = y, x
                            apply_pairs(person[a], y, +1)
                            apply_pairs(person[b], x, +1)
                            now = sumsq
                            if now < best - 1e-9:
                                best, improved = now, True
                            else:                                   # put it back
                                apply_pairs(person[a], y, -1)
                                apply_pairs(person[b], x, -1)
                                person[a][i], person[b][j] = x, y
                                apply_pairs(person[a], x, +1)
                                apply_pairs(person[b], y, +1)
        if not improved:
            break

    return design, describe(design, n_items)


def describe(design, n_items):
    """What the design achieved — reported, not assumed."""
    flat = np.array([i for respondent in design for task in respondent for i in task])
    shown = np.bincount(flat, minlength=n_items)
    pairs = _pair_counts(design, n_items)
    off = pairs[~np.eye(n_items, dtype=bool)]
    per_person = len(design[0]) * len(design[0][0])
    return {
        "n_respondents": len(design),
        "n_items": n_items,
        "sets_per_respondent": len(design[0]),
        "items_per_set": len(design[0][0]),
        "times_each_item_shown": (int(shown.min()), int(shown.max())),
        "exposures_per_respondent": per_person / n_items,
        "pair_appearances": (int(off.min()), int(off.max())),
        "never_paired": int((off == 0).sum() // 2),
        # Arithmetic, not search: if the slots each respondent sees are not a multiple of the item
        # count, some items MUST appear more often than others no matter how the screens are built.
        "evenly_divisible": per_person % n_items == 0,
    }


def to_frame(design, item_names):
    """The design as the long table a survey platform or a colleague can actually use."""
    import pandas as pd
    rows = []
    for person, tasks in enumerate(design):
        for task_number, task in enumerate(tasks, start=1):
            for position, item in enumerate(task, start=1):
                rows.append({"respondent_id": f"R{person + 1:04d}", "task": task_number,
                             "position": position, "item": item_names[item]})
    return pd.DataFrame(rows)


def render(report):
    """The design's own account of itself, in the terms that decide whether it is usable."""
    low, high = report["times_each_item_shown"]
    pair_low, pair_high = report["pair_appearances"]
    lines = [
        f"A best-worst questionnaire for {report['n_items']} items: "
        f"{report['sets_per_respondent']} screens per person, {report['items_per_set']} items on "
        f"each, {report['n_respondents']} people.",
        "",
        f"  each item shown       {low}-{high} times across the study "
        f"(~{report['exposures_per_respondent']:.1f} times per person)",
        f"  each pair together    {pair_low}-{pair_high} times",
        "",
    ]
    if report["never_paired"]:
        lines += [f"**{report['never_paired']} pairs of items never appear together.** Those two "
                  f"items can only be compared through other items, which is weaker evidence. More "
                  f"screens per person, or more people, would fix it.", ""]
    else:
        lines += ["Every pair of items appears together somewhere, so every comparison the study "
                  "needs to make is supported directly.", ""]
    if not report["evenly_divisible"]:
        lines += [f"*The {report['sets_per_respondent']} x {report['items_per_set']} slots each "
                  f"person sees is not a multiple of {report['n_items']} items, so exposure cannot "
                  f"be perfectly even — some items are shown once more than others. That is "
                  f"arithmetic rather than a flaw in the design, and it evens out across "
                  f"respondents.*", ""]
    return "\n".join(lines)
