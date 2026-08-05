"""Is there anything to segment at all? A second, independent answer.

The tool has always asked this — the Hopkins statistic from the beginning, and the gap statistic at
k=1. This adds the test the literature says those two most need beside them, and it is guarded by
what it can and cannot do on survey data.

**Why a second opinion.** Adolfsson, Ackerman & Brownstein, *To Cluster, or Not to Cluster: An
Analysis of Clusterability Methods* (Pattern Recognition, 2019), compares the available methods
across 35,000 simulated datasets. Two findings decide it:

* **Hopkins loses power exactly where this tool is weakest.** On partially overlapping clusters its
  power falls to 32%, while dip-based tests keep near-perfect power. Overlapping segments merging
  into one is the single failure mode measured in `references/kbench.py`.
* **Hopkins treats outliers as clusters**, where dip-based methods do not. Five eccentric
  respondents should not turn a study into a four-segment one.

Measured here on this tool's own data, the second point shows up immediately: on pure noise with
five outliers Hopkins reads 0.60 — over its own "some tendency to cluster" line — while the dip
returns p = 0.57 and correctly finds nothing.

**Two versions of this test were tried, and the obvious one is wrong for survey data.**

The paper's headline recommendation is *dip-dist*: the dip test on all pairwise distances. It does
not transfer to rating scales, and the reason is worth stating so nobody re-adds it. Answers on a
1-5 scale are discrete, so the distances between respondents take a small number of values: on 400
people answering five questions there are 79,800 distances and **50 distinct values among them**.
The dip reads that comb of spikes as many modes and returns p = 0.0000 on data with no groups
whatsoever. The identical test on genuinely continuous noise of the same size returns p = 0.9962.
The paper's simulations are continuous Gaussians; nothing in that literature was run against a
five-point questionnaire.

What is used instead is the paper's other evaluated variant, *PCA-dip*: the dip test on the first
principal component. A principal component is a weighted sum of every question, so it is continuous
even when each answer is not, and the comb disappears.

**Even that has a floor, which is enforced rather than assumed.** With very few questions the sum
is still too lumpy and the same false alarm returns. Measured on pure noise, by question count:

    questions   dip p
        2       0.0000    false alarm
        3       0.0000    false alarm
        4       0.2401    correct
        5       0.9365    correct
       12       0.9952    correct

So the test runs from four questions upward and says why when it does not. That is the same limit
that makes Hopkins unreliable on very short questionnaires — on the 2-question noise above Hopkins
reads 1.00 — which is worth knowing: both tests fail together there, and neither rescues the other.

The guard counts questions rather than the distinct values that are the actual mechanism, and the
first version got that wrong. Guarding on distinct values refused the clearest case in the whole
test set — three well-separated groups — because tight real groups drive that count down too:
people in one segment genuinely do give identical answers. A low count has two causes and only one
of them is a fault, so the guard names the fault.

Measured across nine conditions, with the guard in place: three groups at every separation from
clean to heavy overlap detected (p = 0.0000 in all three, including where Hopkins falls to 0.66),
five groups detected, and pure noise at 4, 5 and 12 questions plus noise-with-outliers all
correctly finding nothing.

The statistic itself comes from the `diptest` package rather than from here. Hartigan & Hartigan's
(1985) algorithm is subtle enough that a re-derivation is a liability: the first attempt scored a
plain normal sample at 0.25, the maximum possible, because it measured distance from *convexity*
rather than from *unimodality* — a unimodal distribution is convex and then concave, so those are
not the same thing. The reference scores the same sample 0.0091.
"""
from __future__ import annotations

import numpy as np

try:                        # optional, like the AI layer: absent means this check is skipped
    import diptest as _hartigan
except Exception:           # pragma: no cover - exercised by the missing-dependency test
    _hartigan = None

#: The first principal component has to be effectively continuous before the dip means anything,
#: and what makes it continuous is having enough questions to sum. Measured on pure noise, the test
#: false-alarms at two and three questions and is correct from four upwards.
#:
#: The guard counts QUESTIONS rather than distinct values, though distinct values are the mechanism
#: — because a low count of them has two possible causes and only one is a fault. Too few questions
#: is the artefact; genuinely tight groups also drive it down, because people in one segment really
#: do give identical answers. Guarding on the symptom refused the clearest case in the whole test
#: set, three well-separated groups, which is precisely the data the test exists to confirm.
MIN_QUESTIONS = 4

#: Fewer respondents than this and the component has too little shape to test either way.
MIN_RESPONDENTS = 40


def available():
    """Whether the dip test can run at all. Nothing else in the panel depends on it."""
    return _hartigan is not None


def pca_dip(X, first_component):
    """Adolfsson et al.'s PCA-dip: Hartigan's dip test on the first principal component.

    `first_component` is passed in rather than computed here so this uses the SAME projection the
    segment map draws — a reader comparing the two is then looking at one thing, not two.

    Returns a dict, or None with a `skipped` reason when the test cannot be trusted on this data.
    """
    if _hartigan is None:
        return {"skipped": "the dip test add-on is not installed"}
    X = np.asarray(X, float)
    n = len(X)
    if n < MIN_RESPONDENTS:
        return {"skipped": f"only {n} respondents; the dip test needs at least "
                           f"{MIN_RESPONDENTS} to say anything"}
    pc = np.asarray(first_component, float).ravel()
    pc = pc[np.isfinite(pc)]
    if pc.size < MIN_RESPONDENTS or np.ptp(pc) <= 0:
        return {"skipped": "every respondent projects to the same point"}

    questions = X.shape[1] if X.ndim > 1 else 1
    if questions < MIN_QUESTIONS:
        return {"skipped": f"only {questions} question(s) — with this few, whole-number answers "
                           "leave the first principal component too lumpy to test, and it reports "
                           "groups that are not there",
                "questions": int(questions)}

    statistic, p = _hartigan.diptest(np.asarray(pc, float))
    return {"p": float(p), "dip": float(statistic), "questions": int(questions),
            "distinct_share": round(len(np.unique(np.round(pc, 9))) / pc.size, 3),
            "n_used": int(pc.size)}


def dip_reading(p):
    """What a dip p-value means, in the tool's own terms."""
    if p < 0.01:
        return "clearly more than one group in these answers"
    if p < 0.05:
        return "more than one group, on the usual 5% threshold"
    if p < 0.10:
        return "a hint of more than one group, short of the usual threshold"
    return "one single spread of answers, with no separate groups showing"


def agreement(hopkins, dip):
    """How the two cluster-tendency tests line up, which is the point of having both.

    They fail in different ways — Hopkins overstates when a few outliers are present and
    understates when groups overlap; the dip is conservative throughout. Agreement is worth more
    than either verdict alone, and disagreement is informative rather than embarrassing.

    Returns (label, sentence) or None when the dip could not run.
    """
    if not dip or "p" not in dip:
        return None
    hopkins_says = hopkins >= 0.60
    dip_says = dip["p"] < 0.05
    if hopkins_says and dip_says:
        return ("both", "Both cluster-tendency tests agree there is structure here — the "
                        "strongest evidence available before any grouping is attempted.")
    if not hopkins_says and not dip_says:
        return ("neither", "Neither cluster-tendency test finds structure. Any groups below were "
                           "constructed by the method rather than found among your respondents.")
    if dip_says:
        return ("dip only", "The dip test finds more than one group where the Hopkins statistic "
                            "does not. Hopkins is known to lose power when groups overlap rather "
                            "than sit apart, so this pattern points to segments that shade into "
                            "one another — read the segment map before deciding.")
    return ("hopkins only", "The Hopkins statistic suggests structure but the dip test does not. "
                            "Hopkins is known to read a handful of unusual respondents as a "
                            "group, so check the segment map for a small knot of outliers before "
                            "treating this as real.")
