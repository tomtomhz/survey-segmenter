"""Drawing the result.

Division of labour, which is the standard one: **scikit-learn computes, matplotlib draws.**
`segment_kmeans.py` runs `KMeans` / `GaussianMixture` and the validation statistics; nothing in
this file decides anything about the segmentation, it only renders what was already found.

This replaces 684 lines that built SVG by concatenating f-strings — path data, tick positions and
text anchors worked out by hand. That worked, but every chart re-derived axes, scaling and
legends from scratch, and adding one meant writing another few dozen lines of coordinate
arithmetic. matplotlib does that part properly.

Two things the old engine did well are kept, because losing them would be a regression:

* **Charts follow the page theme.** All chrome — axes, ticks, labels, grid — is drawn in one
  sentinel colour which is swapped for `currentColor` on the way out, so a single SVG is legible
  on the light ground and the dark one. Only the segment hues are fixed, and those are Okabe-Ito,
  which stays distinguishable for the ~8% of men with colour-vision deficiency.
* **Output is vector.** The report is read on screen, printed, and mailed around as a PDF.

New here: every chart is also rendered to PNG, because Claude cannot see an SVG. That is what
lets the interpretation be based on the same picture the reader is looking at.
"""
import base64
import io
import os
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                      # no display; this runs inside a local web server
import matplotlib.pyplot as plt            # noqa: E402  (backend must be set first)
from matplotlib.colors import LinearSegmentedColormap    # noqa: E402
from matplotlib.backends import backend_agg, backend_svg  # noqa: E402

# Imported by name on purpose, and referenced below so nothing strips them.
#
# matplotlib loads the writer for an output format lazily, inside savefig. A bundler doing static
# analysis therefore sees only the Agg backend selected above and leaves backend_svg out — and
# the packaged app then analyses a survey perfectly and draws nothing at all, failing six times
# with ModuleNotFoundError: No module named 'matplotlib.backends.backend_svg'. That is what the
# first packaged build after moving to matplotlib actually did.
#
# A --hidden-import flag in build_app.py would fix that one build command. An import here fixes
# it for every way this module is ever packaged.
_REQUIRED_BACKENDS = (backend_agg, backend_svg)

# Okabe-Ito, extended. Chosen for colour-vision deficiency rather than for looking pretty: these
# stay distinguishable under the common forms, which matters when colour IS the group identity.
SEG_COLOURS = ("#46785C", "#D55E00", "#0072B2", "#CC79A7", "#E69F00",
               "#56B4E9", "#7A5195", "#8C6D3F", "#3F7F7F", "#9C4029")

# Everything that is chrome rather than data is drawn in this colour and rewritten as
# `currentColor` when the SVG is emitted. It has to be a value matplotlib will not produce by
# itself and that appears nowhere in the palette above.
_INK = "#010203"

# Text is laid out with DejaVu Sans — matplotlib ships it, so every machine and the packaged app
# measure identically — and then the SVG is told to *render* with the page's own stack, so a chart
# looks like part of the interface rather than a pasted-in figure. Metrics come from DejaVu either
# way, which is why the figures are saved with generous padding: a slightly wider rendering font
# must not clip a label.
#
# `svg.fonttype: none` is what makes this possible at all: it keeps text as real <text> elements
# instead of embedding glyph outlines, which also keeps the files roughly an order of magnitude
# smaller and leaves the labels selectable and searchable in the PDF.
_METRIC_FONT = "DejaVu Sans"
_FONT_STACK = ("ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
               "Helvetica, Arial, sans-serif")

_STYLE = {
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": [_METRIC_FONT],
    "font.size": 11,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "text.color": _INK,
    "axes.labelcolor": _INK,
    "axes.edgecolor": _INK,
    "xtick.color": _INK,
    "ytick.color": _INK,
    "axes.titlecolor": _INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": _INK,
    "grid.alpha": 0.13,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.autolayout": False,
}


# Populated by the most recent build_charts call; see the note there.
last_errors = []


def seg_colour(index):
    return SEG_COLOURS[int(index) % len(SEG_COLOURS)]


def _label(names, index):
    """The name the reader gave a group, falling back to its number."""
    if names and index < len(names) and names[index]:
        return str(names[index])
    return f"Group {index}"


def _num(x, places=2):
    """A number a non-statistician can read: no scientific notation, no trailing noise."""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:.{places}f}".rstrip("0").rstrip(".") or "0"


def _short(text, limit):
    """Question codes are long and the axis is not. Underscores read as spaces."""
    s = str(text).replace("_", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


class _Figure:
    """A matplotlib figure that renders itself to the two forms the app needs.

    Used as a context manager so the figure is always closed: matplotlib keeps every unclosed
    figure alive in a global registry, and a long-running server that leaks them will eventually
    exhaust memory over a day of use.
    """

    def legend_below(self, ax, ncol):
        """Put the key under the whole figure rather than under the axes.

        Anchoring it to the axes means the offset is in axes-height fractions, so the same number
        is a small gap on a short chart and a large one on a tall chart. Letting the layout engine
        place it removes the guesswork — and stops the radar's bottom spoke label landing on top
        of the key.
        """
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            self.fig.legend(handles, labels, loc="outside lower center", ncol=ncol, fontsize=10,
                            frameon=False, handletextpad=0.5, columnspacing=1.6)

    def __init__(self, width=8.0, height=4.4, dpi=110):
        self._context = plt.rc_context(_STYLE)
        self._context.__enter__()
        self.fig = plt.figure(figsize=(width, height), dpi=dpi, layout='constrained')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        plt.close(self.fig)
        self._context.__exit__(*exc)
        return False

    def svg(self):
        buf = io.StringIO()
        self.fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.15)
        out = buf.getvalue()
        # Drop matplotlib's XML preamble and DOCTYPE: this is embedded in an HTML document, not
        # served as a standalone file, and a stray <?xml ...?> mid-page is a parse error.
        start = out.find("<svg")
        out = out[start:] if start >= 0 else out
        # One swap makes the chrome theme-aware. Case-insensitive because matplotlib writes hex
        # in both cases depending on where the colour lands.
        for form in (_INK, _INK.upper(), _INK.lower()):
            out = out.replace(form, "currentColor")
        # Render with the interface's typeface rather than the one used for measuring.
        # The quotes matter: matplotlib emits `font: 11px 'DejaVu Sans'`, so replacing only the
        # name leaves the whole stack inside one pair of quotes and the browser reads it as a
        # single family with a very strange name, matches nothing, and silently falls back.
        out = out.replace(f"'{_METRIC_FONT}'", _FONT_STACK).replace(_METRIC_FONT, _FONT_STACK)
        # Let it scale to its container rather than sitting at a fixed pixel width.
        out = out.replace("<svg ", '<svg class="chart" preserveAspectRatio="xMidYMid meet" ', 1)
        return out.strip()

    def png(self):
        """A raster copy, for Claude. Rendered on a light ground because a transparent PNG
        composites onto black in some viewers and the axis labels vanish."""
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2,
                         facecolor="#FBFAF3", dpi=110)
        return buf.getvalue()


def _finish(figure, chart_id, title, caption, with_png=True):
    """Package a drawn figure into the shape the interface and the report expect."""
    chart = {"id": chart_id, "title": title, "svg": figure.svg(), "caption": caption}
    if with_png:
        chart["png_b64"] = base64.b64encode(figure.png()).decode("ascii")
    return chart


def onehot_matrix(Xcat, level_counts):
    """Indicator coding of categorical answers, so pick-any data can be projected and measured in
    a Euclidean space at all. Scaled by 1/sqrt(levels) so an item with many options does not
    dominate the picture purely by having more columns."""
    cols = []
    for j, levels in enumerate(level_counts):
        M = np.zeros((len(Xcat), int(levels)))
        M[np.arange(len(Xcat)), Xcat[:, j].astype(int)] = 1.0
        cols.append(M / np.sqrt(max(int(levels), 1)))
    return np.hstack(cols) if cols else np.zeros((len(Xcat), 1))


def pca_2d(X):
    """Project onto the first two principal components, returning (coords, share of variance).

    SVD of the centred matrix rather than an eigendecomposition of the covariance: numerically
    better behaved, and the same answer. The share is what tells the reader how much of the real
    structure this flat picture is actually showing.
    """
    A = np.asarray(X, float)
    A = A - A.mean(0)
    if A.shape[1] == 1:                    # one item: spread it out so the points are visible
        return np.column_stack([A[:, 0], np.zeros(len(A))]), 1.0
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    var = S ** 2
    total = float(var.sum())
    coords = U[:, :2] * S[:2]
    return coords, (float(var[:2].sum() / total) if total > 0 else 0.0)


def chart_segment_map(X, labels, names=None, max_points=1200, seed=0):
    """The one chart that can falsify the whole result.

    Every respondent is a dot, coloured by the group they were put in. Real segments show as
    separated clumps; a segmentation imposed on structureless data shows as one cloud sliced into
    pie wedges — instantly obvious here and invisible in any table of fit statistics.

    The shaded regions behind the dots are the decision boundaries: which group a new person would
    be assigned to if they landed at that spot. They make the pie-slice failure unmistakable,
    because imposed groups produce boundaries that cut straight through a single dense cloud.
    """
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    coords, kept = pca_2d(X)
    k = int(labels.max()) + 1 if len(labels) else 0
    if k <= 0 or len(coords) == 0:
        return None

    # Centroids from ALL respondents, even when the scatter itself is thinned for file size.
    cents = np.array([coords[labels == c].mean(0) if (labels == c).any() else [np.nan, np.nan]
                      for c in range(k)])
    idx = np.arange(len(coords))
    thinned = len(idx) > max_points
    if thinned:
        idx = np.random.default_rng(seed).choice(len(coords), max_points, replace=False)

    with _Figure() as figure:
        ax = figure.fig.add_subplot(111)

        lo, hi = coords.min(0), coords.max(0)
        span = np.where((hi - lo) > 0, hi - lo, 1.0)
        lo, hi = lo - span * 0.07, hi + span * 0.07

        # Decision regions: nearest centroid over a grid, which is exactly how k-means assigns.
        live = [c for c in range(k) if np.isfinite(cents[c]).all()]
        if len(live) > 1:
            gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 320),
                                 np.linspace(lo[1], hi[1], 320))
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            d = np.stack([((grid - cents[c]) ** 2).sum(1) for c in live])
            region = np.array(live)[d.argmin(0)].reshape(gx.shape)
            ax.pcolormesh(gx, gy, region, shading="nearest", alpha=0.10,
                          cmap=LinearSegmentedColormap.from_list(
                              "segments", [seg_colour(c) for c in range(k)], N=k),
                          vmin=-0.5, vmax=k - 0.5, rasterized=True)

        for c in range(k):
            here = idx[labels[idx] == c]
            if len(here):
                ax.scatter(coords[here, 0], coords[here, 1], s=22, c=seg_colour(c),
                           alpha=0.62, linewidths=0, label=_label(names, c))
        for c in live:
            ax.scatter(*cents[c], s=190, c=seg_colour(c), edgecolors="white", linewidths=2,
                       zorder=5)
            ax.annotate(str(c), cents[c], color="white", fontsize=10, fontweight="bold",
                        ha="center", va="center", zorder=6)

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel("Direction 1 →")
        ax.set_ylabel("← Direction 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        figure.legend_below(ax, ncol=min(k, 5))

        pct = int(round(kept * 100))
        if kept >= 0.7:
            trust = (f"These two directions carry {pct}% of everything that varies between your "
                     "respondents, so this picture is a fair likeness — believe what you see.")
        elif kept >= 0.4:
            trust = (f"These two directions carry {pct}% of the variation. Groups that look "
                     "separated here really are; groups that overlap here might still differ on "
                     "something this flat view cannot show.")
        else:
            trust = (f"Careful: these two directions carry only {pct}% of the variation, so this "
                     "is a poor likeness of your data. Overlap here is weak evidence either way "
                     "— lean on the stability numbers instead.")

        caption = ("Every dot is one respondent, placed so people who answered alike sit close "
                   "together, and coloured by the group they were assigned to. Big numbered dots "
                   "are the group centres, and the faint shaded regions show which group a new "
                   "person landing there would be put into. <strong>What you want to see:</strong> "
                   "colours forming their own clumps that the shaded regions follow. "
                   "<strong>What is a warning sign:</strong> one continuous cloud with the "
                   "boundaries cut across it like slices of a pie &mdash; that means the groups "
                   "were imposed on the data rather than found in it. " + trust)
        if thinned:
            caption += (f" (Showing a random {max_points:,} of {len(coords):,} respondents so the "
                        "chart stays quick to draw; the group centres use everyone.)")

        return _finish(figure, "map", "The segment map — do these groups actually separate?",
                       caption)


def chart_silhouette(X, labels, names=None, max_rows=900):
    """How well each individual person fits the group they were put in.

    The conventional silhouette plot: one bar per respondent, sorted within group. A negative bar
    is someone who sits closer to a different group than their own.
    """
    from sklearn.metrics import silhouette_samples

    X = np.asarray(X, float)
    labels = np.asarray(labels)
    k = int(labels.max()) + 1 if len(labels) else 0
    if k < 2 or len(np.unique(labels)) < 2:
        return None
    rows = np.arange(len(X))
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(0).choice(len(X), max_rows, replace=False))
    try:
        sv = silhouette_samples(X[rows], labels[rows])
    except Exception:
        return None
    mean = float(np.mean(sv))

    with _Figure(height=4.6) as figure:
        ax = figure.fig.add_subplot(111)
        y = 0
        for c in range(k):
            here = np.sort(sv[labels[rows] == c])
            if not len(here):
                continue
            ax.barh(np.arange(y, y + len(here)), here, height=1.0, color=seg_colour(c),
                    alpha=0.85, linewidth=0, label=_label(names, c))
            y += len(here) + max(4, len(sv) // 60)      # a gap so groups read as blocks

        ax.axvline(0, color=_INK, linewidth=1, alpha=0.5)
        ax.axvline(mean, color="#9C4029", linewidth=1.4, linestyle="--")
        # Pinned to the top of the axes, not to a data row: the y axis is inverted so that group
        # 0 reads first, which would otherwise drop this label onto the last group's bars.
        ax.annotate(f"average {_num(mean)}", xy=(mean, 1.0),
                    xycoords=("data", "axes fraction"), color="#9C4029", fontsize=10,
                    ha="right" if mean > 0 else "left", va="bottom",
                    xytext=(-4 if mean > 0 else 4, 4), textcoords="offset points")
        ax.set_ylim(-2, y)
        ax.invert_yaxis()          # group 0 at the top, matching the key and the other charts
        ax.set_yticks([])
        ax.set_xlabel("how well each person fits their group →")
        figure.legend_below(ax, ncol=min(k, 5))

        n_neg = int((sv < 0).sum())
        misfits = ("No respondent is actually misfiled — everyone sits closer to their own group "
                   "than to any other." if n_neg == 0 else
                   f"{n_neg} of {len(sv):,} respondents ({n_neg / len(sv):.0%}) sit closer to a "
                   "different group than their own.")
        # The count of misfits alone reads reassuringly on data with no structure: a partition of
        # pure noise can have almost no NEGATIVE scores while every score sits near zero, which
        # means the groups barely separate. The average is what decides that, on Kaufman &
        # Rousseeuw's conventional reading, so the verdict leads with it.
        if mean >= 0.5:
            verdict = (f"Average fit is {_num(mean)} — the groups are substantially separated, "
                       "and these are safe to treat as real segments. " + misfits)
        elif mean >= 0.25:
            verdict = (f"Average fit is {_num(mean)} — weak separation. The bars being mostly "
                       "positive is not enough on its own: at this level the groups are a "
                       "reasonable working split rather than a natural boundary. " + misfits)
        else:
            verdict = (f"<strong>Average fit is only {_num(mean)}.</strong> That is the signature "
                       "of structure that is not really there: the bars can look tidy and mostly "
                       "positive while nobody is much closer to their own group than to the next "
                       "one. Do not read these as natural segments. " + misfits)

        return _finish(figure, "fit", "Who actually belongs — fit of every respondent", verdict)


def chart_k_choice(diag, rec_k):
    """The elbow plot's honest cousin: every quality measure at every number of groups tried.

    An elbow alone invites reading a bend that is not there. Plotting the stability measures
    against the 0.80 line makes "no number of groups reproduces" visible rather than arguable.
    """
    if diag is None or len(diag) == 0 or "k" not in diag:
        return None
    series = [("prediction_strength", "Prediction strength", "#46785C"),
              ("stability_ARI", "Reproducibility (ARI)", "#0072B2"),
              ("silhouette", "Separation (silhouette)", "#D55E00")]
    present = [(col, name, colour) for col, name, colour in series if col in diag]
    if not present:
        return None

    with _Figure(height=4.0) as figure:
        ax = figure.fig.add_subplot(111)
        ks = list(diag["k"])
        best = 0.0
        for col, name, colour in present:
            values = pd.to_numeric(diag[col], errors="coerce")
            ax.plot(ks, values, marker="o", markersize=5, linewidth=2, color=colour, label=name)
            finite = values[np.isfinite(values)]
            if len(finite):
                best = max(best, float(np.nanmax(finite)))

        # 0.80 is the conventional threshold for "this partition reproduces", so it is drawn
        # rather than left for the reader to hold in their head while squinting at the lines.
        ax.axhline(0.80, color=_INK, linewidth=1, linestyle=":", alpha=0.55)
        ax.annotate("0.80 — reproduces", (ks[-1], 0.80), fontsize=9.5, va="bottom", ha="right",
                    xytext=(0, 3), textcoords="offset points", alpha=0.75)
        if rec_k in ks:
            ax.axvline(rec_k, color=_INK, linewidth=1, alpha=0.22)
            ax.annotate(f"chosen: {rec_k}", (rec_k, ax.get_ylim()[1]), fontsize=10, ha="center",
                        va="top", xytext=(0, -4), textcoords="offset points")
        ax.set_xticks(ks)
        ax.set_xlabel("number of groups")
        ax.set_ylim(min(0.0, ax.get_ylim()[0]), max(1.0, ax.get_ylim()[1]))
        figure.legend_below(ax, ncol=3)

        read = ("The chosen number sits at a clear peak above the 0.80 line — that is a real "
                "answer from the data." if best >= 0.8 else
                "Nothing reaches the 0.80 line, which means no number of groups reproduces "
                "strongly. Treat the groups as a working hypothesis, not a finding, and lean on "
                "judgement about how many you can actually act on.")
        return _finish(figure, "k", "Was the number of groups a clear call?",
                       "Each line is a different test of quality, run for every number of groups "
                       "the tool tried. Higher is better for all three. " + read)


def _separating(centroids, limit):
    """The questions that actually distinguish the groups, most first.

    Ranked by spread across groups: a question every group answers the same way is not evidence
    of anything, however important it sounds.
    """
    C = centroids.astype(float)
    # Columns that are entirely missing separate nothing, and drawing a bar for them produces a
    # chart that is wrong rather than empty. An all-NaN frame therefore yields no chart at all.
    C = C.loc[:, C.notna().any(axis=0)]
    if C.empty:
        return [], 0
    spread = (C.max(axis=0) - C.min(axis=0)).fillna(0).sort_values(ascending=False)
    keep = [c for c in spread.index[:limit]]
    return keep, max(0, len(spread) - len(keep))


def chart_profiles(centroids, names=None, max_items=9, kind="means"):
    """What actually distinguishes the groups, on the original answer scale."""
    if centroids is None or centroids.empty:
        return None
    keep, trimmed = _separating(centroids, max_items)
    if not keep:
        return None
    data = centroids[keep]
    k = len(data.index)

    with _Figure(height=max(3.4, 0.42 * len(keep) + 1.6)) as figure:
        ax = figure.fig.add_subplot(111)
        y = np.arange(len(keep))
        height = 0.8 / max(k, 1)
        for i in range(k):
            ax.barh(y + i * height - 0.4 + height / 2, data.iloc[i].values, height=height,
                    color=seg_colour(i), label=_label(names, i), linewidth=0)
        ax.set_yticks(y)
        ax.set_yticklabels([_short(c, 26) for c in keep])
        ax.invert_yaxis()
        ax.set_xlabel("share of the group" if kind == "shares" else "average answer")
        ax.grid(axis="y", visible=False)
        figure.legend_below(ax, ncol=min(k, 5))

        if kind == "shares":
            caption = ("How likely each answer is within each group, for the answers that separate "
                       "the groups most. A bar at 0.90 means nine in ten of that group gave that "
                       "answer.")
        else:
            caption = ("Average answer per group, on your original answer scale, for the questions "
                       "that separate the groups most.")
        caption += (" Bars of visibly different lengths are a real difference you can write a "
                    "brief around; bars of near-identical length mean that question does not "
                    "distinguish anybody, whatever the report calls it.")
        if trimmed > 0:
            caption += (f" ({trimmed} more question{'' if trimmed == 1 else 's'} separated the "
                        "groups less and would not fit legibly here. The <strong>Full grid</strong> "
                        "tab shows every question with nothing left out.)")
        return _finish(figure, "profiles", "What makes the groups different", caption)


def chart_radar(centroids, names=None, max_axes=12, kind="means"):
    """Each group as an outline, so its shape can be read at a glance.

    Declines below three spokes or two groups: a two-spoke radar is a line, and one outline has
    nothing to be compared with. Drawing it anyway would imply a comparison that is not there.
    """
    if centroids is None or centroids.empty or len(centroids.index) < 2:
        return None
    keep, trimmed = _separating(centroids, max_axes)
    if len(keep) < 3:
        return None
    data = centroids[keep].astype(float)

    # Each spoke scaled to its own range: questions on different scales would otherwise let one
    # dominate the shape entirely, and shape is the whole point of this chart.
    lo, hi = data.min(axis=0), data.max(axis=0)
    span = (hi - lo).replace(0, 1.0)
    scaled = (data - lo) / span

    angles = np.linspace(0, 2 * np.pi, len(keep), endpoint=False)
    closed = np.concatenate([angles, angles[:1]])

    with _Figure(width=7.4, height=5.4) as figure:
        ax = figure.fig.add_subplot(111, polar=True)
        for i in range(len(data.index)):
            values = np.concatenate([scaled.iloc[i].values, scaled.iloc[i].values[:1]])
            ax.plot(closed, values, linewidth=2, color=seg_colour(i), label=_label(names, i))
            ax.fill(closed, values, color=seg_colour(i), alpha=0.12)
        ax.set_xticks(angles)
        ax.set_xticklabels([_short(c, 18) for c in keep], fontsize=9.5)
        ax.set_yticks([])
        ax.set_ylim(0, 1.08)
        ax.spines["polar"].set_alpha(0.25)
        figure.legend_below(ax, ncol=min(len(data.index), 4))

        caption = ("Each outline is one group, drawn across the questions that separate the groups "
                   "most. A spoke reaching further out means that group scores higher on it. "
                   "<strong>What to look for:</strong> outlines with genuinely different shapes "
                   "are distinct personas you can write separate briefs for. Outlines that nest "
                   "neatly inside one another describe the same people at different intensities "
                   "&mdash; one message with a volume knob, not several audiences. Each spoke is "
                   "scaled to its own range, so shapes are comparable but distances between "
                   "spokes are not.")
        if trimmed > 0:
            caption += (f" ({trimmed} more question{'' if trimmed == 1 else 's'} separated the "
                        "groups less and would have crowded the spokes; the <strong>Full "
                        "grid</strong> tab has all of them.)")
        return _finish(figure, "radar", "Group shapes — the persona view", caption)


def chart_heatmap(centroids, names=None, kind="means"):
    """Every question against every group, with nothing trimmed.

    This exists because the bar chart stops at nine questions to stay legible, which on a
    fifteen-item block hides a third of the study. Diverging around each question's own mean, so
    a strong colour means "unusual for this question" rather than "large number".
    """
    if centroids is None or centroids.empty:
        return None
    data = centroids.astype(float)
    items = list(data.columns)
    if not items:
        return None
    centred = data - data.mean(axis=0)
    limit = float(np.nanmax(np.abs(centred.values))) or 1.0

    with _Figure(width=8.0, height=max(2.6, 0.30 * len(items) + 1.8)) as figure:
        ax = figure.fig.add_subplot(111)
        cmap = LinearSegmentedColormap.from_list(
            "divergent", ["#0072B2", "#F2F0E6", "#D55E00"])
        mesh = ax.pcolormesh(centred.values.T, cmap=cmap, vmin=-limit, vmax=limit,
                             edgecolors="none")
        ax.set_xticks(np.arange(len(data.index)) + 0.5)
        ax.set_xticklabels([_label(names, i) for i in range(len(data.index))],
                           rotation=30, ha="right", fontsize=9.5)
        ax.set_yticks(np.arange(len(items)) + 0.5)
        ax.set_yticklabels([_short(c, 30) for c in items], fontsize=9.5)
        ax.invert_yaxis()
        ax.grid(False)
        bar = figure.fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
        bar.set_label("below average          above average", fontsize=9.5)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=9)

        n = len(items)
        what = "the share giving that answer" if kind == "shares" else "the average answer"
        caption = (f"The full grid: {what} for each group on all {n} question"
                   f"{'' if n == 1 else 's'}, with nothing left out. Colour shows how far that "
                   "group sits from the average <em>for that question</em>, so a strong colour "
                   "means the group is genuinely unusual there. <strong>Read it by scanning for "
                   "the strongest colours</strong> &mdash; those cells are what distinguishes a "
                   "group. A row of pale cells is a group with no distinctive profile, which is "
                   "worth knowing before it becomes a persona.")
        return _finish(figure, "heatmap", "Every group against every question", caption)


def build_charts(seg, method, names=None):
    """Draw everything this result supports, skipping whatever it does not.

    Each chart is built inside its own try: they are independent, and a failure in one must not
    withhold the rest. This was a real bug once — the charts were built as a single eagerly
    evaluated tuple, so one NaN centroid discarded all of them, the segment map included.
    """
    out = []
    # Why a chart is missing, kept rather than only printed. The packaged app is built
    # --windowed, which discards stdout entirely, so a print is invisible exactly where the
    # failure is most likely — a dependency that did not survive being bundled. The first
    # packaged build after moving to matplotlib produced no charts at all and said nothing
    # about why. `last_errors` is read by build_app.py's smoke test and reported to the user.
    last_errors.clear()

    # Only when a diagnostic log is configured, so normal runs stay quiet. A hang leaves no
    # traceback and no failed chart — the only way to know which one stopped is a line before it.
    trace = bool(os.environ.get("SEG_LOG"))

    def attempt(label, fn):
        started = time.monotonic()
        if trace:
            print(f"chart: starting {label}")
        try:
            chart = fn()
        except Exception as e:
            reason = f"{label}: {type(e).__name__}: {e}"
            last_errors.append(reason)
            print(f"NOTE: could not draw the '{label}' chart ({type(e).__name__}: {e}); "
                  "the rest of the report is unaffected.")
            return
        if trace:
            print(f"chart: {label} took {time.monotonic() - started:.1f}s")
        if chart:
            out.append(chart)

    try:
        if method == "lca":
            # Categorical answers are not points in a space until they are coded as one.
            X = onehot_matrix(seg.Xcat, seg.level_counts)
            # A latent class is described by how likely each ANSWER is within it, so pivot the
            # probability table into the same class-by-column shape the bar chart already draws.
            # "What actually differs between these groups" is the chart a marketer needs most,
            # and leaving the categorical path without one made it the weaker half of the tool.
            pf = seg.profiles_frame()
            pf = pf.assign(col=pf["item"].astype(str) + " = " + pf["level"].astype(str))
            centroids = pf.pivot(index="class", columns="col", values="probability")
        else:
            X, centroids = seg.X, seg.centroids
        labels = np.asarray(seg.labels)
    except Exception as e:          # the shared inputs failed; there is nothing to draw at all
        last_errors.append(f"preparing the data: {type(e).__name__}: {e}")
        print(f"NOTE: could not prepare the charts ({type(e).__name__}: {e}); "
              "the report itself is unaffected.")
        return []

    kind = "shares" if method == "lca" else "means"
    attempt("segment map", lambda: chart_segment_map(X, labels, names))
    attempt("per-person fit", lambda: chart_silhouette(X, labels, names))
    attempt("choice of k", lambda: chart_k_choice(seg.diagnostics, int(seg.recommended_k)))
    if centroids is not None:
        attempt("group profiles", lambda: chart_profiles(centroids, names, kind=kind))
        attempt("group shapes", lambda: chart_radar(centroids, names, kind=kind))
        attempt("full grid", lambda: chart_heatmap(centroids, names, kind=kind))
    return out
