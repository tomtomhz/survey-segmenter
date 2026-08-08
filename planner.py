"""How many people do you need? Answered by simulating the study you are about to run.

Every survey starts with the same question and it is normally answered by folklore — "300 is
standard", "√(n/2)", whatever the last agency charged for. This answers it by measurement: it
builds surveys with a known number of segments planted in them, at the shape you describe, and
runs **the real pipeline** over them at a range of sample sizes to see when the tool actually
recovers what is there.

**Why simulate rather than use a formula.** A closed-form power calculation would have to assume
the model is right — spherical, equal-sized, well-separated clusters and a method that finds them.
This tool's own measured weakness is precisely overlapping segments, and a formula that assumed
otherwise would confidently under-quote the sample you need. Running the pipeline itself means the
answer includes the method's real behaviour, weaknesses and all.

**What it cannot tell you.** Whether your segments exist at all. Nothing can, before fielding. This
answers the narrower and answerable question: *if* segments of a given distinctness are there, how
many respondents does this tool need to find them. The output says so rather than implying more.

**The three regimes, and why these numbers.** Distinctness is expressed as Cohen's d between
adjacent segment centres on the questions that separate them — a standard effect size, so it can be
stated plainly rather than as an arbitrary knob. The three values were chosen by measuring, not by
picking round numbers; a sweep of d against sample size gave (four seeds, six questions, three
planted segments, share of runs finding the right number):

        d      n=100   n=200   n=400     mean ARI
        2.0     4/4     4/4     4/4        0.99
        1.2     4/4     4/4     4/4        0.87
        1.0     3/4     4/4     4/4        0.72
        0.8     2/4     4/4     4/4        0.53
        0.6     0/4     0/4     1/4        0.24

So the useful trio is 2.0, 1.0 and 0.6, because each says something different and only the middle
one is about sample size:

* **2.0 — obvious differences.** Recovered at every size tried. If your segments are this distinct,
  sample size is not your problem and you are choosing it for precision of the profiles, not for
  finding the groups.
* **1.0 — moderate.** This is the regime where the number of respondents genuinely decides the
  answer, and therefore the one the recommendation is built on.
* **0.6 — subtle.** Not recovered at ANY size tried, including 400. More people does not rescue it.
  That is the most useful thing the planner says, because it is the case where spending more money
  buys nothing and the honest advice is to sharpen the questionnaire instead.

Deliberately not covered: fielding cost, incidence and screen-out rates, and best-worst designs.
Those are separate jobs and pretending to model them here would be guessing.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

#: Distinctness of the planted segments, as Cohen's d between adjacent centres on the separating
#: questions. See the module docstring for the measurements behind these three values.
REGIMES = (("obvious", 2.0), ("moderate", 1.0), ("subtle", 0.6))

#: Sample sizes to try. Chosen to bracket the range a real study actually chooses between; the
#: planner reports the whole curve rather than a single number, because where it flattens is the
#: information — a size beyond that buys precision, not discovery.
DEFAULT_SIZES = (100, 200, 300, 400, 600, 800)

#: Repeats per cell. One run of one size tells you almost nothing: the same design and the same
#: number of people recovers the truth on some samples and not others, which IS the finding, and a
#: single draw would report whichever way one coin landed.
DEFAULT_SEEDS = 4

#: A shorter, cheaper validation panel. The planner runs the pipeline dozens of times, and the full
#: resampling settings would take an hour to answer a question worth four minutes. Measured against
#: full settings on the same designs, the recovered number of segments does not change.
_FAST = dict(gap_B=6, stability_B=10, ps_splits=5, jaccard_B=15, n_init_final=8,
             n_init_search=6, run_consensus=False, check_variable_selection=False)


def simulate_study(separation, n_people, n_questions=6, n_segments=3, scale=5, seed=0):
    """One synthetic survey with `n_segments` groups planted `separation` standard deviations apart.

    Each question separates a different pair of segments, which is what a real questionnaire looks
    like: no single item tells the groups apart on its own, and the structure is in the pattern
    across them. Answers are rounded onto the response scale and clipped to it, because a survey
    contains whole numbers between 1 and 5 rather than latent scores — and that discreteness is
    itself part of what limits how well any method can separate people.
    """
    rng = np.random.default_rng(seed)
    step = float(separation)
    middle = (scale + 1) / 2.0
    centres = np.full((n_segments, n_questions), middle)
    for question in range(n_questions):
        for segment in range(n_segments):
            offset = ((segment + question) % n_segments) - (n_segments - 1) / 2.0
            centres[segment, question] = middle + step * offset
    who = rng.integers(0, n_segments, n_people)
    answers = np.clip(np.round(centres[who] + rng.normal(0, 1.0, (n_people, n_questions))),
                      1, scale).astype(int)
    frame = pd.DataFrame(answers, columns=[f"q{i + 1}" for i in range(n_questions)])
    frame.insert(0, "respondent_id", [f"P{i:05d}" for i in range(n_people)])
    return frame, who


def _run_once(frame, truth, n_segments):
    """Put one simulated survey through the real pipeline and score what came back."""
    import segment_kmeans as sk
    from sklearn.metrics import adjusted_rand_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            result = sk.run_analysis(
                frame.to_csv(index=False).encode(),
                cfg=sk.SegmentationConfig(k_min=2, k_max=max(6, n_segments + 3), **_FAST))
    assigned = pd.read_csv(io.StringIO(result["files"]["segment_assignments.csv"]))
    order = assigned.sort_values("id", key=lambda s: s.str[1:].astype(int))
    return {
        "found_k": int(result["k"]),
        "right_k": int(result["k"]) == n_segments,
        "ari": float(adjusted_rand_score(truth, order["segment"].to_numpy())),
        "confidence": result.get("confidence", "unknown"),
    }


def plan_study(n_questions=6, n_segments=3, scale=5, sizes=DEFAULT_SIZES, seeds=DEFAULT_SEEDS,
               progress=None):
    """Sweep sample size against distinctness and return what each combination actually recovers.

    Returns a dict with `cells` (one row per regime and size) and the inputs it was given, so the
    caller can render it or hand it to a report. No advice is computed here — `recommend()` turns
    the measurements into words, and keeping the two apart means the numbers can be checked without
    arguing about the prose.
    """
    cells = []
    total = len(REGIMES) * len(sizes)
    done = 0
    for label, separation in REGIMES:
        for n_people in sizes:
            runs = [_run_once(*simulate_study(separation, n_people, n_questions, n_segments,
                                              scale, seed=seed), n_segments)
                    for seed in range(seeds)]
            right = sum(r["right_k"] for r in runs)
            cells.append({
                "regime": label,
                "separation": separation,
                "n_people": n_people,
                "runs": len(runs),
                "right_k": right,
                "hit_rate": right / len(runs),
                "mean_ari": float(np.mean([r["ari"] for r in runs])),
                # Worth keeping: a run that finds the WRONG number of segments and still calls it
                # high confidence is the failure this tool most wants to avoid, and the planner is
                # where a user can be warned it is possible at a given size.
                "confidently_wrong": sum(1 for r in runs
                                         if not r["right_k"] and r["confidence"] == "high"),
            })
            done += 1
            if progress:
                progress(done, total, cells[-1])
    return {"cells": cells, "n_questions": n_questions, "n_segments": n_segments,
            "scale": scale, "sizes": list(sizes), "seeds": seeds}


def recommend(plan, threshold=0.9):
    """Turn the sweep into the sentence someone planning a study actually needs.

    The recommendation is built on the MODERATE regime alone, and that is a deliberate choice. The
    obvious regime is recovered at every size, so it would recommend the smallest sample tried; the
    subtle regime is recovered at none, so it would recommend the largest and still not work. Only
    the middle regime is actually a question about sample size.
    """
    moderate = sorted((c for c in plan["cells"] if c["regime"] == "moderate"),
                      key=lambda c: c["n_people"])
    subtle = sorted((c for c in plan["cells"] if c["regime"] == "subtle"),
                    key=lambda c: c["n_people"])
    enough = next((c["n_people"] for c in moderate if c["hit_rate"] >= threshold), None)
    subtle_ever = any(c["hit_rate"] >= threshold for c in subtle)
    risky = [c for c in plan["cells"] if c["confidently_wrong"]]
    return {
        "recommended_n": enough,
        "largest_tried": max(plan["sizes"]),
        "subtle_reachable": subtle_ever,
        "confidently_wrong_below": min((c["n_people"] for c in risky), default=None),
    }


def render(plan):
    """The sweep as something a person reads before spending money on fieldwork.

    Deliberately plain text rather than a chart: this is read once, in a terminal, while deciding a
    number. The table is the evidence and the paragraph is the answer, in that order, because
    anyone spending a budget on it should be able to see what the advice rests on.
    """
    advice = recommend(plan)
    sizes = plan["sizes"]
    lines = [
        f"Planning a study: {plan['n_questions']} questions on a 1-{plan['scale']} scale, "
        f"expecting {plan['n_segments']} segments.",
        f"Each cell is {plan['seeds']} simulated studies put through the real analysis.",
        "",
        "How often the tool found the right number of segments:",
        "",
    ]
    # Build each cell as a string BEFORE padding it. Formatting "{a}/{b:<12}" pads only b, so the
    # counts ran into the column beside them and the table came out as "1002/2".
    head = f"  {'people':>7}" + "".join(f"{label:>12}" for label, _ in REGIMES)
    lines += [head, "  " + "-" * (len(head) - 2)]
    for n_people in sizes:
        row = f"  {n_people:>7}"
        for label, _ in REGIMES:
            cell = next(c for c in plan["cells"]
                        if c["regime"] == label and c["n_people"] == n_people)
            score = f"{cell['right_k']}/{cell['runs']}"
            row += f"{score:>12}"
        lines.append(row)
    lines += ["",
              "  obvious  = segments differ by about 2 standard deviations (you could spot them "
              "by eye)",
              "  moderate = about 1 standard deviation — the realistic case, and the one sample "
              "size decides",
              "  subtle   = about 0.6 — real, but close to the limit of what any method separates",
              ""]

    if advice["recommended_n"] is None:
        lines += [f"**No sample size tried reached a reliable answer even for moderately distinct "
                  f"segments.** The largest tried was {advice['largest_tried']}. Either the "
                  f"questionnaire needs to separate people more sharply — more questions, or "
                  f"questions people genuinely disagree about — or the groups you are looking for "
                  f"may not be distinct enough to find."]
    else:
        lines += [f"**Field about {advice['recommended_n']} people.** Below that, moderately "
                  f"distinct segments start being missed rather than found. Above it you are "
                  f"buying more precise profiles, not a better chance of discovering the groups."]

    lines.append("")
    if not advice["subtle_reachable"]:
        lines += [f"**More people will not rescue subtle differences.** At the subtle level the "
                  f"tool did not reliably find the right number at ANY size tried, up to "
                  f"{advice['largest_tried']}. If you think your groups differ only slightly, "
                  f"spend the budget on sharper questions rather than on more respondents — a "
                  f"bigger sample measures a weak signal more precisely, it does not make it "
                  f"strong."]
    if advice["confidently_wrong_below"] is not None:
        lines += ["",
                  f"**One caution.** At {advice['confidently_wrong_below']} people, at least one "
                  f"simulated study reported the WRONG number of segments while still calling the "
                  f"result high confidence. The tool judges whether a grouping reproduces, and on "
                  f"heavily overlapping data a merged pair reproduces perfectly well. Treat "
                  f"segment counts from small samples as provisional."]
    lines += ["",
              "This says how many people the tool needs to FIND segments of a given distinctness. "
              "It cannot tell you whether your segments exist — nothing can, before you field."]
    return "\n".join(lines)
