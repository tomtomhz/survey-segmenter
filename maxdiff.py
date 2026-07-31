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
    rescaled: np.ndarray = field(default=None, repr=False)  # 0-100 per respondent, for reading

    def as_frame(self):
        """Utilities as a respondent-by-item table, ready to hand to the segmenter."""
        import pandas as pd
        return pd.DataFrame(self.utilities, index=self.respondent_ids, columns=self.item_names)


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
    pop_mean = np.concatenate([np.mean(kept_mu, axis=0), [0.0]])
    pop_mean -= pop_mean.mean()

    # A 0-100 rescale per respondent, purely for human reading; clustering uses the raw utilities.
    lo = utilities.min(axis=1, keepdims=True)
    rng_span = utilities.max(axis=1, keepdims=True) - lo
    rescaled = 100.0 * (utilities - lo) / np.where(rng_span > 0, rng_span, 1.0)

    return MaxDiffResult(
        utilities=utilities,
        item_names=list(item_names),
        respondent_ids=list(respondent_ids),
        population_mean=pop_mean,
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
    "respondent": ("respondent_id", "respondent", "resp_id", "id", "sys_respnum"),
    "set":        ("set", "task", "block", "screen", "question", "set_id"),
    "item":       ("item", "item_id", "concept", "attribute", "statement"),
    "choice":     ("choice", "selection", "answer", "response", "best_worst", "bw"),
}


def looks_like_maxdiff(df) -> bool:
    """True when the table is a tidy best-worst export rather than a rating grid."""
    cols = {str(c).strip().lower() for c in df.columns}
    have = sum(any(a in cols for a in alts) for alts in _MAXDIFF_COLUMNS.values())
    return have == 4


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
    d["choice"] = d["choice"].astype(str).str.strip().str.lower()

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
