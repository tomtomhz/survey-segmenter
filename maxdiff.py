"""
Hierarchical Bayes estimation of individual-level MaxDiff utilities.

Why this module exists
----------------------
MaxDiff (best-worst scaling) asks respondents to pick the best and worst item from small sets.
Counting "times best minus times worst" gives a usable ranking for the SAMPLE, but it is far too
coarse to describe an INDIVIDUAL: with four exposures per item a respondent's count for any item
can only take a handful of values, so everyone collapses onto a few identical score patterns and
a segmentation run on those counts is mostly clustering rounding error.

Hierarchical Bayes fixes this by estimating each respondent's utilities while borrowing strength
from the population: a respondent with little information is pulled toward the group mean, one
with informative answers is allowed to differ. That is what makes individual-level scores stable
enough to cluster on, and it is the input the study's instrument specifies.

The model
---------
Sequential best-then-worst multinomial logit, the standard MaxDiff likelihood:

    P(best = j | set S)              = exp(b_j) / sum_{k in S} exp(b_k)
    P(worst = l | set S, best = j)   = exp(-b_l) / sum_{k in S\\{j}} exp(-b_k)

with a normal population prior over respondents:

    b_i ~ MVN(mu, Sigma),   mu ~ diffuse normal,   Sigma ~ Inverse-Wishart

Estimated by Gibbs sampling: mu and Sigma have conjugate draws, and each respondent's b_i is
updated with a random-walk Metropolis step. The Metropolis step is vectorised across respondents,
so cost scales with the number of MCMC draws rather than with sample size — a few thousand
respondents costs little more than a few hundred.

Identification: utilities are only defined up to an additive constant per respondent (adding 1 to
every item changes no choice probability), so each draw is centred to sum to zero. Reported scores
are the posterior mean per respondent.

Deliberately implemented in numpy alone. PyMC or Stan would be more general, but they cannot be
frozen into the packaged desktop app, and a segmentation tool a marketer cannot install is not a
tool. Correctness is defended by a parameter-recovery test rather than by the library's reputation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MaxDiffResult:
    """Estimated utilities plus the diagnostics needed to judge whether to trust them."""
    utilities: np.ndarray          # (n_respondents, n_items) posterior mean, centred
    item_names: list               # length n_items
    respondent_ids: list           # length n_respondents
    population_mean: np.ndarray    # (n_items,) posterior mean of mu — the sample-level ranking
    acceptance_rate: float         # Metropolis acceptance; healthy is roughly 0.2-0.5
    n_draws: int
    n_burn: int
    #: 95% credible interval on the population ranking, same scale and centring as
    #: `population_mean`. Two items whose intervals overlap are not separated by this study.
    population_low: np.ndarray = field(default=None, repr=False)
    population_high: np.ndarray = field(default=None, repr=False)
    #: (n_kept, n_items) retained posterior draws of the population ranking, centred like
    #: `population_mean`. Kept because whether one item beats another is a question about the JOINT
    #: posterior, and marginal intervals cannot answer it: these utilities are centred, so they are
    #: correlated by construction and overlapping intervals do not imply an unresolved ordering.
    population_draws: np.ndarray = field(default=None, repr=False)
    rescaled: np.ndarray = field(default=None, repr=False)  # 0-100 per respondent, for reading

    def as_frame(self):
        """Utilities as a respondent-by-item table, ready to hand to the segmenter."""
        import pandas as pd
        return pd.DataFrame(self.utilities, index=self.respondent_ids, columns=self.item_names)

    #: How sure the study must be that one item beats the next before its printed position is
    #: treated as established. The conventional 95%, chosen for the reason it is conventional
    #: elsewhere: it is the line readers already have an intuition for.
    ORDER_CERTAINTY = 0.95

    def ranking(self):
        """The study's headline answer: items ordered by how much the sample wants them.

        One row per item, best first, carrying the credible interval, the probability that each
        item really does beat the one below it, and a flag for whether that clears
        `ORDER_CERTAINTY`.

        **Why the probability rather than overlapping intervals.** The first version of this
        compared each item's lower bound against the next item's upper bound — the comparison a
        reader makes on sight of a table, and easy to defend. It is also the wrong question, in two
        ways that matter:

        * These utilities are centred, so the items are correlated *by construction*: one going up
          pushes the rest down. Reading two marginal intervals side by side quietly assumes an
          independence the model does not have.
        * It answers yes or no. Measured on a deliberately thin study, the interval rule returned
          "too close to call" for a pair at probability 0.579 and for another at 0.929 — the same
          three words for a coin flip and for a finding most people would act on. In the second
          case "we cannot tell" was simply false.

        The joint draws answer the question that was actually being asked, and cost nothing: the
        sampler already produced them.
        """
        import pandas as pd
        lo = self.population_low
        hi = self.population_high
        out = pd.DataFrame({
            "item": self.item_names,
            "utility": self.population_mean,
            "low": lo if lo is not None else np.nan,
            "high": hi if hi is not None else np.nan,
        }).sort_values("utility", ascending=False)
        order = out.index.to_numpy()          # original positions, now in ranked order
        out = out.reset_index(drop=True)

        draws = self.population_draws
        if draws is not None and len(out) > 1:
            d = np.asarray(draws)[:, order]   # columns reordered to match the table
            ahead = (d[:, :-1] > d[:, 1:]).mean(axis=0)
            out["prob_ahead"] = np.append(np.round(ahead, 4), np.nan)
            # Nullable boolean, not plain bool: the last row has no item below it, so its value is
            # genuinely missing rather than False. pandas 3 refuses to put NA into a bool column,
            # which is the correct objection — "not established" and "nothing to establish it
            # against" are different claims and the dtype should not let them blur.
            sep = pd.Series(np.append(ahead >= self.ORDER_CERTAINTY, False)).astype("boolean")
            sep.iloc[-1] = pd.NA
            out["separated_from_next"] = sep
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out


def _loglik(beta, design, best_pos, worst_pos, mask):
    """Log-likelihood of every respondent's best/worst choices under their own utilities.

    Vectorised over respondents AND sets. `design` holds item indices with -1 padding for ragged
    set sizes; `mask` marks the real entries. Padding is pushed to -inf so it cannot win a choice
    and contributes nothing to the log-sum-exp.
    """
    n_resp, n_sets, set_size = design.shape
    idx = np.where(mask, design, 0)                                  # safe gather
    u = np.take_along_axis(beta[:, None, :], idx.reshape(n_resp, -1)[:, None, :], axis=2)
    u = u.reshape(n_resp, n_sets, set_size)
    u = np.where(mask, u, -np.inf)

    # Best: softmax over the whole shown set.
    ll = np.take_along_axis(u, best_pos[..., None], axis=2)[..., 0]
    ll = ll - _logsumexp(u)

    # Worst: softmax over the NEGATED utilities of what remains after the best is removed.
    neg = np.where(mask, -u, -np.inf)
    remaining = neg.copy()
    np.put_along_axis(remaining, best_pos[..., None], -np.inf, axis=2)
    ll = ll + np.take_along_axis(neg, worst_pos[..., None], axis=2)[..., 0] - _logsumexp(remaining)

    return np.where(np.isfinite(ll), ll, 0.0).sum(axis=1)            # (n_resp,)


def _logsumexp(a):
    """Row-wise log-sum-exp over the last axis, stable against the -inf padding."""
    m = np.max(a, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(a - m).sum(axis=-1, keepdims=True)))[..., 0]


def estimate_hb(design, best_pos, worst_pos, item_names, respondent_ids,
                n_draws=6000, n_burn=2000, thin=5, seed=42, progress=True):
    """Estimate individual-level utilities by Hierarchical Bayes.

    design      (n_resp, n_sets, set_size) item indices shown; -1 pads ragged sets
    best_pos    (n_resp, n_sets) POSITION within the set that was chosen best
    worst_pos   (n_resp, n_sets) POSITION within the set that was chosen worst

    Defaults are deliberately generous: this runs once per study, and a segmentation built on
    under-converged utilities is worse than a slow one.
    """
    design = np.asarray(design, dtype=int)
    best_pos = np.asarray(best_pos, dtype=int)
    worst_pos = np.asarray(worst_pos, dtype=int)
    mask = design >= 0
    n_resp, n_sets, _ = design.shape
    n_items = len(item_names)
    if n_resp < 2:
        raise ValueError("Hierarchical Bayes needs more than one respondent — it works by "
                         "borrowing strength across the sample.")

    rng = np.random.default_rng(seed)

    # Identification. Utilities are only defined up to an additive constant per respondent, so one
    # item must be pinned. Sampling all K and re-centring each draw looks equivalent but is not:
    # centred vectors lie on a K-1 dimensional plane, which makes the population covariance
    # singular by construction — the Inverse-Wishart is then being asked for a draw from a
    # degenerate distribution, and Sigma is only kept finite by the ridge added before each
    # inverse. Pinning the LAST item at zero and sampling K-1 free utilities keeps Sigma
    # genuinely full rank. Everything is re-expanded and centred once at the end, for reporting.
    n_free = n_items - 1
    if n_free < 1:
        raise ValueError("MaxDiff needs at least two items to compare.")

    beta = np.zeros((n_resp, n_free))
    mu = np.zeros(n_free)
    sigma = np.eye(n_free) * 0.5
    step = np.full(n_resp, 0.35)          # per-respondent random-walk scale, tuned during burn-in

    prior_df = n_free + 3                 # weakly informative Inverse-Wishart
    prior_scale = np.eye(n_free) * prior_df

    def expand(b):
        """K-1 free utilities -> K item utilities, with the pinned item at zero."""
        return np.concatenate([b, np.zeros((b.shape[0], 1))], axis=1)

    cur_ll = _loglik(expand(beta), design, best_pos, worst_pos, mask)
    kept, accepted, proposed = [], 0, 0
    kept_mu = []

    for it in range(n_draws):
        # --- Metropolis update of every respondent's utilities, all at once ---
        chol = np.linalg.cholesky(sigma + np.eye(n_free) * 1e-8)
        prop = beta + (rng.standard_normal((n_resp, n_free)) @ chol.T) * step[:, None]

        prop_ll = _loglik(expand(prop), design, best_pos, worst_pos, mask)
        d_cur = beta - mu
        d_prop = prop - mu
        prec = np.linalg.inv(sigma + np.eye(n_free) * 1e-8)
        log_prior_cur = -0.5 * np.einsum("ij,jk,ik->i", d_cur, prec, d_cur)
        log_prior_prop = -0.5 * np.einsum("ij,jk,ik->i", d_prop, prec, d_prop)

        take = np.log(rng.random(n_resp)) < (prop_ll + log_prior_prop) - (cur_ll + log_prior_cur)
        beta = np.where(take[:, None], prop, beta)
        cur_ll = np.where(take, prop_ll, cur_ll)
        accepted += int(take.sum())
        proposed += n_resp

        # Adapt the step size during burn-in only; after that the chain must be homogeneous.
        if it < n_burn and it % 50 == 49:
            rate = take.mean()
            step *= 1.15 if rate > 0.35 else (0.87 if rate < 0.15 else 1.0)
            step = np.clip(step, 0.02, 2.0)

        # --- Conjugate draws for the population parameters ---
        bbar = beta.mean(axis=0)
        mu = bbar + np.linalg.cholesky(sigma / n_resp + np.eye(n_free) * 1e-10) @ \
            rng.standard_normal(n_free)
        dev = beta - mu
        scale = prior_scale + dev.T @ dev
        sigma = _inv_wishart(scale, prior_df + n_resp, rng)

        if it >= n_burn and (it - n_burn) % thin == 0:
            kept.append(beta.copy())
            kept_mu.append(mu.copy())

        if progress and it % max(1, n_draws // 10) == 0:
            print(f"  HB draw {it:>5}/{n_draws}  acceptance {accepted / max(proposed, 1):.2f}")

    if not kept:
        raise ValueError("No posterior draws were kept — increase n_draws above n_burn.")

    # Expand back to K items and centre once, for reporting. Centring here is safe: it happens
    # after sampling, so it cannot make the covariance the sampler used singular.
    utilities = expand(np.mean(kept, axis=0))
    utilities -= utilities.mean(axis=1, keepdims=True)

    # The population ranking, kept WITH its uncertainty rather than collapsed to a point. Every
    # retained draw of mu is expanded and centred exactly as the mean was, so the interval is on
    # the same scale as the number it brackets. Centring is linear, so the mean of the centred
    # draws is identical to the centred mean this previously computed — the ranking does not move;
    # the spread simply stops being thrown away.
    #
    # This matters because the whole point of Hierarchical Bayes here is that it knows how sure it
    # is. Two items half a point apart with intervals that overlap completely are not ranked, and a
    # report that prints them in order without saying so invites a decision the data cannot carry.
    mu_draws = np.concatenate([np.asarray(kept_mu, dtype=float),
                               np.zeros((len(kept_mu), 1))], axis=1)
    mu_draws -= mu_draws.mean(axis=1, keepdims=True)
    pop_mean = mu_draws.mean(axis=0)
    pop_low, pop_high = np.percentile(mu_draws, [2.5, 97.5], axis=0)

    # A 0-100 rescale per respondent, purely for human reading; clustering uses the raw utilities.
    lo = utilities.min(axis=1, keepdims=True)
    rng_span = utilities.max(axis=1, keepdims=True) - lo
    rescaled = 100.0 * (utilities - lo) / np.where(rng_span > 0, rng_span, 1.0)

    return MaxDiffResult(
        utilities=utilities,
        item_names=list(item_names),
        respondent_ids=list(respondent_ids),
        population_mean=pop_mean,
        population_low=pop_low,
        population_high=pop_high,
        population_draws=mu_draws,
        acceptance_rate=accepted / max(proposed, 1),
        n_draws=n_draws,
        n_burn=n_burn,
        rescaled=rescaled,
    )


def _inv_wishart(scale, df, rng):
    """Draw from an Inverse-Wishart via the Bartlett decomposition of its Wishart inverse."""
    p = scale.shape[0]
    inv_scale = np.linalg.inv(scale + np.eye(p) * 1e-10)
    chol = np.linalg.cholesky((inv_scale + inv_scale.T) / 2 + np.eye(p) * 1e-10)
    A = np.zeros((p, p))
    A[np.diag_indices(p)] = np.sqrt(rng.chisquare(df - np.arange(p)))
    A[np.tril_indices(p, -1)] = rng.standard_normal(p * (p - 1) // 2)
    W = chol @ A @ A.T @ chol.T
    return np.linalg.inv(W + np.eye(p) * 1e-10)


# =====================================================================================
# Reading a MaxDiff export
# =====================================================================================
# Survey platforms export MaxDiff in many shapes, but all of them can be reshaped into one tidy
# table: one row per item shown, saying which set it was in and whether it was picked best or
# worst. Asking for that shape is far more robust than trying to guess Qualtrics' column naming,
# and it is a shape any analyst can produce with a pivot.

_MAXDIFF_COLUMNS = {
    "respondent": ("respondent_id", "respondent", "resp_id", "id", "sys_respnum", "person",
                   "panelist", "uid"),
    "set":        ("set", "task", "block", "screen", "question", "set_id", "trial"),
    # "issue" comes from real published data: the bwsTools example asks which issues facing the
    # country matter most and least, and named its column that. The others are the words a
    # questionnaire uses for whatever is being compared.
    "item":       ("item", "item_id", "concept", "attribute", "statement", "issue", "option",
                   "feature", "brand", "message", "alternative"),
    # "value" and "code" are generic enough to appear in tables that are not best-worst at all,
    # which is why detection now also reads what is IN the column — see `looks_like_maxdiff`.
    "choice":     ("choice", "selection", "answer", "response", "best_worst", "bw", "value",
                   "code", "pick", "chosen"),
}


def _normalise_choice(series):
    """The choice column as 'best' / 'worst' / '', whichever way the export encoded it.

    Two encodings are in the wild and only one was handled. Words — best/worst, most/least, and
    their Nordic equivalents — are in `_CHOICE_WORDS`. The other is numeric, and it is what real
    published data uses: the bwsTools example dataset codes **1 for best, -1 for worst and 0 for
    merely shown**, and being unable to read it meant a genuine best-worst study was refused with
    an error naming no cause.

    Numeric coding is only accepted when it is unambiguous — the values are a subset of {-1, 0, 1}
    and BOTH 1 and -1 appear. A column of 1s and 2s is left alone: 2 could mean worst, or second
    choice, or a rating, and guessing would fabricate preference data out of an ordinary number.
    """
    import pandas as pd
    numeric = pd.to_numeric(series, errors="coerce")
    seen = set(numeric.dropna().unique())
    if seen and seen <= {-1.0, 0.0, 1.0} and {1.0, -1.0} <= seen:
        return numeric.map({1.0: "best", -1.0: "worst"}).fillna("")
    return series.astype(str).str.strip().str.lower().map(_CHOICE_WORDS).fillna("")


def looks_like_maxdiff(df) -> bool:
    """True when the table is a tidy best-worst export rather than a rating grid.

    Two conditions, not one. The four columns must be identifiable, AND the choice column must
    actually contain best-worst answers. The content check exists because the aliases had to widen
    to read real exports — 'value', 'code', 'issue' — and those words appear in plenty of tables
    that are not best-worst at all. A false positive here is expensive: an ordinary survey would be
    put through a preference sampler and come back as confident nonsense.
    """
    cols = {str(c).strip().lower() for c in df.columns}
    if sum(any(a in cols for a in alts) for alts in _MAXDIFF_COLUMNS.values()) != 4:
        return False
    lower = {str(c).strip().lower(): c for c in df.columns}
    choice = next(lower[a] for a in _MAXDIFF_COLUMNS["choice"] if a in lower)
    marks = set(_normalise_choice(df[choice]).unique()) - {""}
    if marks:
        return True
    # Nothing recognisable in the column. If it is EMPTY, this is still a best-worst file — a
    # broken one — and saying so is far more useful than silently treating it as a rating grid:
    # the reader's own message names the missing bests and worsts. If instead the column is full of
    # something else, such as 1-5 ratings, it is not a best-worst file and must not be scored as
    # one. Requiring recognisable marks unconditionally cost that error message, which is the
    # regression this distinction exists to avoid.
    raw = df[choice].astype(str).str.strip().str.lower()
    return bool((~raw.isin(["", "nan", "none"])).sum() == 0)


# What a best-worst export actually writes in its choice column. Only the literal English "best"
# and "worst" were recognised, and anything else made every set look incomplete: each one was
# dropped in silence and the run then failed with a "too few" error that named no cause. A Swedish
# study writing "bäst" and "sämst" — this tool's own users — lost every observation that way, and
# "most"/"least" is at least as common in English tooling as best/worst.
_CHOICE_WORDS = {
    "best": "best", "worst": "worst",
    "most": "best", "least": "worst",
    "b": "best", "w": "worst",
    "most important": "best", "least important": "worst",
    "most preferred": "best", "least preferred": "worst",
    "bäst": "best", "sämst": "worst",           # Swedish
    "basta": "best", "bast": "best", "samst": "worst", "sämsta": "worst",
    "bästa": "best", "viktigast": "best", "minst viktig": "worst",
    "beste": "best", "verste": "worst",         # Norwegian / Danish
    "bedste": "best", "værst": "worst", "vaerst": "worst",
    "paras": "best", "huonoin": "worst",        # Finnish
    "": "", "nan": "", "none": "",
}


def read_maxdiff(df):
    """Tidy long MaxDiff table -> (design, best_pos, worst_pos, item_names, respondent_ids).

    Expected columns (case-insensitive, several aliases accepted):

        respondent_id | set | item | choice

    where `choice` is 'best' / 'worst' / blank for the items merely shown. Every set must record
    exactly one best and one worst; a set missing either is dropped with a note rather than
    silently guessed at, because inventing a choice would fabricate preference data.
    """
    lower = {str(c).strip().lower(): c for c in df.columns}
    pick = {}
    for role, alts in _MAXDIFF_COLUMNS.items():
        match = next((lower[a] for a in alts if a in lower), None)
        if match is None:
            raise ValueError(f"_MAXDIFF_MISSING:{role}")
        pick[role] = match

    d = df[[pick["respondent"], pick["set"], pick["item"], pick["choice"]]].copy()
    d.columns = ["respondent", "set", "item", "choice"]
    d["choice"] = _normalise_choice(d["choice"])

    items = sorted(d["item"].astype(str).unique())
    item_ix = {n: i for i, n in enumerate(items)}
    respondents = list(dict.fromkeys(d["respondent"].astype(str)))
    resp_ix = {r: i for i, r in enumerate(respondents)}

    grouped = list(d.groupby(["respondent", "set"], sort=False))
    set_size = max(len(g) for _, g in grouped)
    per_resp = {}
    dropped = 0
    for (r, _s), g in grouped:
        pos_best = np.where(g["choice"].to_numpy() == "best")[0]
        pos_worst = np.where(g["choice"].to_numpy() == "worst")[0]
        if len(pos_best) != 1 or len(pos_worst) != 1 or pos_best[0] == pos_worst[0]:
            dropped += 1
            continue
        row = [item_ix[str(x)] for x in g["item"]]
        row += [-1] * (set_size - len(row))
        per_resp.setdefault(str(r), []).append((row, int(pos_best[0]), int(pos_worst[0])))
    if dropped:
        print(f"NOTE: skipped {dropped} MaxDiff set(s) without exactly one best and one worst.")

    usable = [r for r in respondents if per_resp.get(r)]
    # Losing a respondent entirely is a different matter from losing a set, and has to be said
    # out loud. Someone who skipped the exercise, or whose every set came back malformed, simply
    # disappears here — and a study that reports 400 respondents when 380 were analysed has an
    # inaccurate sample size in the write-up. The count is the only place it can be noticed.
    lost = len(respondents) - len(usable)
    if lost:
        print(f"NOTE: {lost} of {len(respondents)} respondents gave no usable best-worst answers "
              f"at all and are not in the utilities; {len(usable)} were analysed. Check the "
              "export if that number is higher than you expect.")
    if len(usable) < 2:
        raise ValueError("_MAXDIFF_TOO_FEW")
    n_sets = max(len(v) for v in per_resp.values())

    design = np.full((len(usable), n_sets, set_size), -1, dtype=int)
    best_pos = np.zeros((len(usable), n_sets), dtype=int)
    worst_pos = np.zeros((len(usable), n_sets), dtype=int)
    for r in usable:
        i = usable.index(r)
        for s, (row, bp, wp) in enumerate(per_resp[r]):
            design[i, s] = row
            best_pos[i, s], worst_pos[i, s] = bp, wp
    _ = resp_ix
    return design, best_pos, worst_pos, items, usable


def utilities_from_export(df, **kw):
    """One call: tidy MaxDiff export -> respondent-by-item utility table ready to segment."""
    design, best_pos, worst_pos, items, respondents = read_maxdiff(df)
    n_sets = int((design >= 0).any(axis=2).sum(axis=1).max())
    exposures = float((design >= 0).sum() / (len(respondents) * len(items)))
    print(f"MaxDiff detected: {len(respondents)} respondents, {len(items)} items, "
          f"up to {n_sets} sets each (~{exposures:.1f} exposures per item). "
          "Estimating individual-level utilities by Hierarchical Bayes...")
    if exposures < 3:
        print(f"NOTE: only ~{exposures:.1f} exposures per item. Individual-level utilities get "
              "unreliable below about 3; treat per-respondent scores as indicative.")
    return estimate_hb(design, best_pos, worst_pos, items, respondents, **kw)
