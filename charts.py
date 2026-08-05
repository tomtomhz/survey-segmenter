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
import re
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

# =====================================================================================
# The colour system
# =====================================================================================
# Colour here does four different jobs, and the previous palette did all of them with one set of
# hues. Orange meant "Group 1" on four charts, "Separation (silhouette)" on the choice-of-k chart,
# and "above average" on the heatmap. A reader who learns one of those is misled by the next. Each
# job now has its own encoding and they do not overlap:
#
#   identity   which segment somebody is in         SEG_LIGHT / SEG_DARK, plus a marker shape
#   polarity   above or below the average answer    DIVERGING, warm against cool, grey middle
#   magnitude  how much of something                one hue, light to dark
#   neutral    the sample as a whole, and chrome    the ink colour at low emphasis
#
# The identity palette is not eyeballed. It is checked with a validator against this app's own
# surfaces, and the previous one FAILED: three hues sat below the chroma floor and read as grey,
# and the worst adjacent pair was slots 1 and 2 — Group 0 against Group 1, the most common
# comparison in the whole tool — at CVD ΔE 7.9. The comment it carried, that these "stay
# distinguishable under the common forms", was not true as measured. The true Okabe-Ito set it
# claimed to be does pass (ΔE 15.8); it had been edited into failing by swapping in a muted green
# and appending low-chroma extras.
#
# Slot ORDER is part of the safety rather than decoration: the checks run on adjacent pairs, so
# reordering changes whether the palette passes. Do not reorder these to taste.
SEG_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
             "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# The same eight hues stepped for a dark surface — chosen for it, not flipped into it. Measured on
# the app's own dark card (#222724) all eight clear the dark lightness band and 3:1 contrast,
# which the light steps do not. Charts used to ship one fixed hex for both grounds.
SEG_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767")

# Marker shape per segment: a requirement, not a flourish. The segment map is a scatter, so every
# pair of colours is on screen at once — the "all pairs" case, not the "adjacent pairs" case a bar
# chart or a legend gets. Measured at eight groups: worst all-pairs CVD ΔE 3.2 (green against
# orange, protanopia) and worst normal-vision ΔE 7.1 (red against orange). Seven is far below the
# floor of 15 — two segments a reader with full colour vision cannot tell apart. Only the first
# three slots clear all-pairs on colour alone. Shape carries the identity colour cannot, and
# survives a photocopier and a bad projector as well.
SEG_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

# Above and below average, for the heatmap only. Warm against cool so the poles read as opposite,
# with a genuinely neutral middle so "no difference" reads as nothing. Never used for identity.
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

# The page behind the chart. Marks that overlap are separated by a ring in THIS colour rather than
# in white, so the ring reads as a gap rather than as an outline — on a dark page a white ring
# around every dot looks like a deliberate stroke, which is not what it is. Swapped for a CSS
# variable on the way out exactly like the segment hues, so it follows the reader's theme.
_SURFACE = "#FBFAF3"
_SURFACE_DARK = "#222724"

# The app's own accent, and deliberately NOT one of the identity slots — that was the mistake
# this rework exists to fix, and the first attempt at this line repeated it by reaching for the
# palette's violet. Emphasis is a different job from identity, so it gets a colour from outside
# the set: the green the interface already uses for "this is the thing to look at".
_METRIC_LEAD = "#46785C"
_METRIC_LEAD_DARK = "#7FB48F"

SEG_COLOURS = SEG_LIGHT       # the old name; everything reads it through seg_colour()

# Injected into every chart. Both scopes are declared on purpose: the media query follows the
# operating system, and the [data-theme] rules follow the reader's own toggle in the app, which
# has to win in BOTH directions — hence the :not() guard letting an explicit light choice beat an
# OS set to dark.
_LIGHT_VARS = ("".join(f"--seg-{i}:{c};" for i, c in enumerate(SEG_LIGHT))
               + f"--chart-surface:{_SURFACE};--chart-lead:{_METRIC_LEAD};")
_DARK_VARS = ("".join(f"--seg-{i}:{c};" for i, c in enumerate(SEG_DARK))
              + f"--chart-surface:{_SURFACE_DARK};--chart-lead:{_METRIC_LEAD_DARK};")
_THEME_STYLE = ("<style>"
                f"svg.chart{{{_LIGHT_VARS}}}"
                "@media (prefers-color-scheme:dark){:root:not([data-theme=light]) svg.chart{"
                f"{_DARK_VARS}}}}}"
                f":root[data-theme=dark] svg.chart{{{_DARK_VARS}}}"
                "</style>")

# Everything that is chrome rather than data is drawn in this colour and rewritten as
# `currentColor` when the SVG is emitted. It has to be a value matplotlib will not produce by
# itself and that appears nowhere in the palette above.
_INK = "#010203"

# Marker AREA in points^2 for the segment map's count-weighted dots: the low end is a single
# respondent, the high end whatever the busiest answer pattern holds. Area rather than radius
# because area is how people read "how many" (Cleveland; Wilke), and a floor rather than a pure
# proportion because one person sitting alone still has to be visible beside a stack of hundreds.
_COUNT_DOT_MIN = 16.0
_COUNT_DOT_MAX = 460.0

# Above this many people in one group the per-respondent fit chart switches from drawing a
# rectangle each to filling the same outline as a single polygon. Purely a drawing decision — the
# values plotted are identical either way — and it exists because the patch count is what made an
# uncapped chart slow, not the arithmetic.
_BARS_MAX = 400

# Marks are kept as vector up to these counts and rasterised above them. The threshold is a real
# trade-off, not tidiness: only VECTOR marks carry the CSS variables, so only vector marks follow
# the reader into dark mode. Rasterised marks bake the light palette into a bitmap, and measured
# on the app's dark card that leaves the violet slot at 1.77:1 contrast.
#
# Measured SVG weight for the map, drawn once per distinct answer pattern:
#     207 marks   32 KB rasterised    85 KB vector
#     698 marks   72 KB              258 KB
#   2,962 marks   90 KB            1,011 KB
#  49,975 marks   91 KB           16,915 KB
#
# So vector below ~1,500 marks costs a few hundred KB and buys correct dark mode; above it the
# file grows without bound. Ordinary survey work — a few hundred to a few thousand people on a
# short questionnaire — sits comfortably under the line, which is exactly the case that matters.
_VECTOR_MARKS_MAX = 1500
_VECTOR_FILL_MAX = 3000

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


def seg_colour(index):
    """The light-mode hex for a segment.

    Drawn with the light value and swapped for `var(--seg-N, <that value>)` on the way out, so the
    page can hand the chart its dark step. The fallback in the var() is what keeps a chart saved
    to a file on its own still looking right.
    """
    return SEG_LIGHT[int(index) % len(SEG_LIGHT)]


def seg_marker(index):
    """The marker shape for a segment. See SEG_MARKERS for why this is not optional.

    Colour and shape both have eight values, so cycling them together would hand group 8 exactly
    the colour AND the shape of group 0 — two segments drawn identically, which is worse than
    either channel failing alone. Measured before this: groups 0/8, 1/9, 2/10 and 3/11 were
    indistinguishable in both channels at once, reachable with `--kmax 10`.
    #
    Advancing the shape by one extra step on each wrap makes the (colour, shape) PAIR unique out
    to 64 groups, which is far past anything a survey can support. The first eight are untouched,
    so the ordinary case still gets its own colour and its own shape.
    """
    index = int(index)
    slots = len(SEG_MARKERS)
    return SEG_MARKERS[(index + index // slots) % slots]


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

    def legend_below(self, ax, ncol, handles=None, labels=None):
        """Put the key under the whole figure rather than under the axes.

        Anchoring it to the axes means the offset is in axes-height fractions, so the same number
        is a small gap on a short chart and a large one on a tall chart. Letting the layout engine
        place it removes the guesswork — and stops the radar's bottom spoke label landing on top
        of the key.
        """
        if handles is None or labels is None:
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
        # No metadata block. matplotlib embeds an RDF/Dublin Core record naming itself and
        # stamping the creation time — about 600 bytes per chart for nothing a reader wants, and
        # a timestamp that makes otherwise identical output differ byte for byte.
        self.fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.15,
                         metadata={"Date": None, "Creator": None, "Format": None, "Type": None})
        out, held_text = _protect_text(buf.getvalue())
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
        # Segment hues become CSS variables so the page can supply the dark step. matplotlib must
        # be given a real colour to draw with, so it draws in the light value and that value then
        # becomes the var()'s own fallback — which is what keeps a chart saved to a file, or opened
        # outside this app, looking right rather than black.
        for slot, light in enumerate(SEG_LIGHT):
            for form in (light, light.upper()):
                out = out.replace(form, f"var(--seg-{slot}, {light})")
        for form in (_SURFACE, _SURFACE.upper(), _SURFACE.lower()):
            out = out.replace(form, f"var(--chart-surface, {_SURFACE})")
        for form in (_METRIC_LEAD, _METRIC_LEAD.upper(), _METRIC_LEAD.lower()):
            out = out.replace(form, f"var(--chart-lead, {_METRIC_LEAD})")
        # Let it scale to its container rather than sitting at a fixed pixel width.
        out = out.replace("<svg ", '<svg class="chart" preserveAspectRatio="xMidYMid meet" ', 1)
        # The dark steps travel inside the chart rather than living in the app's stylesheet. A
        # chart gets downloaded, pasted into a document and printed; carrying its own theming is
        # what makes it right in those places too, not only inside this interface.
        cut = out.index(">") + 1
        return _restore_text(out[:cut] + _THEME_STYLE + out[cut:], held_text).strip()

    def png(self):
        """A raster copy, for Claude. Rendered on a light ground because a transparent PNG
        composites onto black in some viewers and the axis labels vanish."""
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2,
                         facecolor="#FBFAF3", dpi=110)
        return buf.getvalue()


#: Colours are swapped for theme tokens by plain string replacement over the whole document,
#: which is simple and has one sharp edge: it does not know data from markup. A question worded
#: "#2a78d6 is my favourite" came out of a chart reading "var(--seg-0, #2a78d6) is my favourite",
#: because the label is just text in the SVG and the replacement could not tell.
#:
#: Respondent-supplied strings only ever appear inside <text> elements, so those are lifted out
#: before any replacement runs and put back afterwards. Everything the swaps are actually aiming
#: at — style declarations and presentation attributes — stays in place and still gets rewritten.
_TEXT_NODE = re.compile(r"(<text\b[^>]*>)(.*?)(</text>)", re.DOTALL)
_TEXT_MARK = "\x00text%d\x00"


def _protect_text(svg):
    """Return (masked svg, the text that was lifted out).

    The held strings are RETURNED rather than parked on the module or the function, because two
    people can be analysing surveys in the same process at the same time. Shared mutable state
    here would let one run's labels reappear inside another's chart — the same shape as the
    global error list that once made three healthy runs report failures they never had.
    """
    held = []

    def stash(m):
        held.append(m.group(2))
        return m.group(1) + (_TEXT_MARK % (len(held) - 1)) + m.group(3)

    return _TEXT_NODE.sub(stash, svg), held


def _restore_text(svg, held):
    for i, body in enumerate(held):
        svg = svg.replace(_TEXT_MARK % i, body)
    return svg


#: Bumped when the shape of a chart spec changes in a way the interface must notice. The browser
#: renderer checks it and falls back to the static drawing rather than reading a spec it does not
#: understand — an old saved project must not render as a broken chart.
SPEC_VERSION = 1

#: Above this many distinct marks a chart ships WITHOUT a spec and the interface shows the static
#: drawing instead. Not a data limit — the static chart still contains every respondent either way
#: — but a limit on what an interactive layer can carry, since each mark becomes a live DOM node.
#:
#: Set from measurement rather than instinct. Payloads: 6 KB at 207 marks, 72 KB at 2,962,
#: 1,212 KB at 49,975 — the last is the real objection, along with fifty thousand nodes. At the
#: cap itself, measured in a browser on 5,896 marks: 0.06 ms per pointer move, because the marks
#: are memoised and hit-testing is a log-time Delaunay lookup, so hovering costs a ring and a line
#: of text rather than the chart. One-off render at that size is ~119 ms.
#:
#: If you raise this, measure the pointer-move cost again first — an interactive chart that lags
#: behind the cursor, or names the wrong dot, is worse than a crisp static one.
INTERACTIVE_MAX_POINTS = 6000


def _finish(figure, chart_id, title, caption, with_png=True, spec=None):
    """Package a drawn figure into the shape the interface and the report expect.

    A chart can carry a `spec`: the numbers behind the picture, in a form the browser can draw
    interactively. Both renderings come from ONE computation, which is the entire point — the
    obvious way to add interactive charts is to write a second chart engine in TypeScript, and
    then the two slowly disagree about what the data says. Here matplotlib and the browser consume
    the same spec, so they cannot.

    The static drawing is not replaced by it. It is what goes into the printed report, the PDF, and
    the PNG that Claude reads, and it is the fallback whenever the interface cannot use a spec.

    A spec carries the same aggregate the picture shows — positions, counts, segment numbers — and
    never a respondent id or a free-text answer, exactly like everything else that leaves this
    process.
    """
    chart = {"id": chart_id, "title": title, "svg": figure.svg(), "caption": caption}
    if with_png:
        chart["png_b64"] = base64.b64encode(figure.png()).decode("ascii")
    if spec is not None:
        chart["spec"] = {"version": SPEC_VERSION, **spec}
    return chart


def _segment_key(count, names):
    """The per-segment identity both renderers share: name, colour in each theme, marker shape.

    Sent rather than re-derived in TypeScript. Duplicating the palette on the other side is how
    the two renderings drift the day somebody edits one of them, and the palette is the part with
    the measured colour-vision properties — it should have exactly one home.
    """
    return [{"index": c, "label": _label(names, c),
             "colour": SEG_LIGHT[c % len(SEG_LIGHT)],
             "colour_dark": SEG_DARK[c % len(SEG_DARK)],
             "marker": seg_marker(c)} for c in range(count)]


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
        return np.column_stack([A[:, 0], np.zeros(len(A))]), 1.0, np.array([1.0, 0.0])
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    var = S ** 2
    total = float(var.sum())
    coords = U[:, :2] * S[:2]
    kept = float(var[:2].sum() / total) if total > 0 else 0.0
    # Per-axis as well as combined: the two directions rarely carry equal weight, and a reader
    # judging whether a horizontal separation is meaningful needs to know what that axis is worth
    # on its own, not just what the pair are worth together.
    each = (var[:2] / total) if total > 0 else np.zeros(2)
    return coords, kept, np.pad(np.asarray(each, float), (0, 2 - len(each)))


def chart_segment_map(X, labels, names=None, seed=0):
    """The one chart that can falsify the whole result.

    Every respondent is on it, coloured by the group they were put in. Real segments show as
    separated clumps; a segmentation imposed on structureless data shows as one cloud sliced into
    pie wedges — instantly obvious here and invisible in any table of fit statistics.

    **Every respondent means every respondent.** Two things used to stop that being true, and both
    bit hardest on exactly the short questionnaires this tool is usually pointed at:

    1.  It drew a random 1,200 and admitted so in small print. A sample cannot falsify anything.
    2.  Rating answers are discrete, so respondents land on a *finite grid* of positions and stack
        invisibly on top of one another. Measured: 3,000 people answering five 1-5 questions
        occupy 422 distinct positions, so a plain scatter shows 14% of the data and hides the
        rest without saying so. With twelve questions it is 99% and the problem disappears — this
        is a short-survey problem, and short surveys are the common case.

    So each dot is drawn once per distinct answer pattern, with its **area proportional to how
    many people share it**. That is exact: no jitter, no invented positions. Jitter is the usual
    remedy and it is rejected here deliberately — Wilke states the danger plainly in *Fundamentals
    of Data Visualization*, that jittering too much "end[s] up placing points in locations that
    are not representative of the underlying dataset", and a chart whose whole job is to let
    somebody check the answer must not invent coordinates to do it.

    The shaded regions behind the dots are the decision boundaries: which group a new person would
    be assigned to if they landed at that spot. They make the pie-slice failure unmistakable,
    because imposed groups produce boundaries that cut straight through a single dense cloud. They
    are drawn in two dimensions while the grouping happened in as many dimensions as there are
    questions, so the caption reports how often the picture's own rule reproduces the real
    assignment instead of quietly assuming it does. Measured on ordinary survey data that is
    99.5-100%, but it is a property of the data, not a guarantee.
    """
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    coords, kept, spread = pca_2d(X)
    k = int(labels.max()) + 1 if len(labels) else 0
    if k <= 0 or len(coords) == 0:
        return None

    cents = np.array([coords[labels == c].mean(0) if (labels == c).any() else [np.nan, np.nan]
                      for c in range(k)])

    # Collapse respondents sitting at exactly the same spot into one marker carrying the count.
    # Rounding first is what makes "exactly" reliable: two identical answer patterns can differ in
    # the last bit or two after the projection's arithmetic, and would otherwise be drawn as two
    # touching dots that read as one slightly darker one.
    spots, first, counts = np.unique(np.round(coords, 9), axis=0,
                                     return_index=True, return_counts=True)
    spot_labels = labels[first]
    stacked = int(counts.max())

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
            # Recede as the number of groups grows. At k=6 six tinted regions behind six colours
            # and six shapes is three encodings of the same fact competing for the same pixels;
            # the regions are there to expose the pie-slice failure, not to identify anybody.
            ax.pcolormesh(gx, gy, region, shading="nearest", alpha=max(0.05, 0.13 - 0.012 * k),
                          cmap=LinearSegmentedColormap.from_list(
                              "segments", [seg_colour(c) for c in range(k)], N=k),
                          vmin=-0.5, vmax=k - 0.5, rasterized=True)

        # Area proportional to the number of people at that spot: the perceptually correct
        # encoding for a count, and the reason a lone respondent stays visible next to a stack of
        # two hundred rather than being scaled into nothing.
        area = _COUNT_DOT_MIN + (_COUNT_DOT_MAX - _COUNT_DOT_MIN) * (
            (counts - 1) / (stacked - 1) if stacked > 1 else np.zeros(len(counts)))
        for c in range(k):
            here = spot_labels == c
            if here.any():
                # Shape as well as colour. See SEG_MARKERS: on a scatter every pair of hues is on
                # screen together, and at eight groups the worst pair measures ΔE 3.2 under
                # protanopia — indistinguishable. Shape is what actually separates them, and it
                # keeps working on a photocopy.
                ax.scatter(spots[here, 0], spots[here, 1], s=area[here], c=seg_colour(c),
                           marker=seg_marker(c), alpha=0.62, linewidths=0,
                           label=_label(names, c), rasterized=len(spots) > _VECTOR_MARKS_MAX)
        for c in live:
            # A plain white badge with the segment's colour as its ring, whatever shape that
            # segment's respondents wear. White text written straight onto the marker worked
            # while every marker was a fat circle; on a plus, a star or a downward triangle there
            # is no solid centre to write on and the number became unreadable. The shape identity
            # is already carried by the hundreds of dots around it — the centre only has to be
            # findable and named.
            ax.scatter(*cents[c], s=330, c=_SURFACE, marker="o",
                       edgecolors=seg_colour(c), linewidths=2.6, zorder=6)
            ax.annotate(str(c), cents[c], color=seg_colour(c), fontsize=10, fontweight="bold",
                        ha="center", va="center", zorder=7)

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        # The share of the variation each direction carries belongs on the axis it describes, not
        # only in the caption: somebody reading the picture on its own has to be able to see how
        # much of the real spread they are actually looking at.
        ax.set_xlabel(f"Direction 1 — {spread[0]:.0%} of the variation →")
        ax.set_ylabel(f"← Direction 2 — {spread[1]:.0%}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        handles, texts = ax.get_legend_handles_labels()
        if stacked > 1:
            # Without a key, a big dot reads as "important" rather than "many". Sizes are the
            # real extremes in this data, so the reader can calibrate against something true.
            from matplotlib.lines import Line2D
            for count, size in ((1, _COUNT_DOT_MIN), (stacked, _COUNT_DOT_MAX)):
                # Plain circles in ink: the key is about how MANY, and borrowing a segment's
                # shape or hue here would read as though it were about a particular group.
                handles.append(Line2D([], [], marker="o", linestyle="none", color=_INK,
                                      alpha=0.45, markersize=float(np.sqrt(size))))
                texts.append(f"{count:,} {'person' if count == 1 else 'people'}")
        figure.legend_below(ax, ncol=min(len(texts), 5), handles=handles, labels=texts)

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

        # How often the picture's own rule — nearest centre, in these two dimensions — reproduces
        # the assignment actually made using every question. Stated rather than assumed, because
        # the shading claims to show where a new person would land and that claim is only true to
        # the extent this number is high.
        faithful = 1.0
        if len(live) > 1:
            d2 = np.stack([((coords - cents[c]) ** 2).sum(1) for c in live])
            faithful = float((np.array(live)[d2.argmin(0)] == labels).mean())

        counted = (f"All {len(coords):,} respondents are shown. Rating answers only come in whole "
                   f"steps, so people who answered identically sit on exactly the same spot: "
                   f"these {len(coords):,} people occupy {len(spots):,} distinct positions, and a "
                   f"dot's area is how many of them share it (the busiest holds {stacked:,}). "
                   "Nothing is sampled and no dot has been nudged to make it visible."
                   if stacked > 1 else
                   f"All {len(coords):,} respondents are shown, each on their own spot &mdash; "
                   "nothing is sampled.")

        caption = ("Every respondent is on this chart, coloured by the group they were assigned "
                   "to. Big numbered dots are the group centres, and the faint shaded regions "
                   "show which group a new person landing there would be put into. "
                   "<strong>What you want to see:</strong> colours forming their own clumps that "
                   "the shaded regions follow. <strong>What is a warning sign:</strong> one "
                   "continuous cloud with the boundaries cut across it like slices of a pie "
                   "&mdash; that means the groups were imposed on the data rather than found in "
                   "it. " + counted + " " + trust)
        if faithful < 0.95:
            caption += (f" One caution about the shading: the groups were formed using all your "
                        f"questions, and this is a flat drawing of that. Judging by position "
                        f"alone here would put {(1 - faithful):.0%} of your respondents in a "
                        "different group from the one they are actually in, so read the coloured "
                        "regions as a rough guide rather than as the real boundary.")

        # The spec is built from the SAME arrays the drawing above used — spots, counts,
        # spot_labels, cents — rather than recomputed. That is what makes the interactive chart
        # and the static one incapable of disagreeing: there is one projection, one aggregation,
        # one set of centres, and two renderers reading them.
        spec = {
            "kind": "segment_map",
            "points": {"x": [round(float(v), 5) for v in spots[:, 0]],
                       "y": [round(float(v), 5) for v in spots[:, 1]],
                       "segment": [int(v) for v in spot_labels],
                       "people": [int(v) for v in counts]},
            "centroids": [{"segment": int(c), "x": round(float(cents[c][0]), 5),
                           "y": round(float(cents[c][1]), 5)} for c in live],
            "segments": _segment_key(k, names),
            "axes": {"x_share": round(float(spread[0]), 4),
                     "y_share": round(float(spread[1]), 4),
                     "kept": round(float(kept), 4)},
            "people": int(len(coords)),
            "busiest_spot": stacked,
            # How often the picture's own nearest-centre rule reproduces the real assignment. The
            # interactive layer surfaces this the same way the caption does, so hovering a dot
            # never implies more precision than the projection has.
            "faithful": round(float(faithful), 4),
        }
        return _finish(figure, "map", "The segment map — do these groups actually separate?",
                       caption,
                       spec=spec if len(spots) <= INTERACTIVE_MAX_POINTS else None)


def chart_silhouette(X, labels, names=None, max_rows=900, metric="euclidean", fit=None):
    """How well each individual person fits the group they were put in.

    The conventional silhouette plot: one bar per respondent, sorted within group. A low bar is
    somebody sitting near the boundary, about equally at home in a different group.

    `fit` is the per-respondent score the pipeline already computed for every single person — one
    minus Leisch's shadow value, the same number that goes into the `fit` column of the exported
    file. Prefer it, for two reasons. It covers **everyone**: the fallback below is scikit-learn's
    silhouette, which needs every pairwise distance, so it had to be capped at a random 900
    respondents and this chart then quietly described a sample of the study rather than the study.
    And it is the same number the reader can look up per person in the CSV, so the picture and the
    file cannot disagree.

    The fallback is kept for the latent-class path, which has probabilities rather than centres and
    so has no shadow value to use.
    """
    labels = np.asarray(labels)
    k = int(labels.max()) + 1 if len(labels) else 0
    if k < 2 or len(np.unique(labels)) < 2:
        return None

    if fit is not None and len(np.asarray(fit)) == len(labels):
        sv = np.asarray(fit, float)
        rows = np.arange(len(labels))
        keep = np.isfinite(sv)
        rows, sv = rows[keep], sv[keep]
        if not len(sv):
            return None
        sampled = 0
    else:
        from sklearn.metrics import silhouette_samples

        X = np.asarray(X, float)
        rows = np.arange(len(X))
        if len(rows) > max_rows:
            rows = np.sort(np.random.default_rng(0).choice(len(X), max_rows, replace=False))
        try:
            sv = silhouette_samples(X[rows], labels[rows], metric=metric)
        except Exception:
            return None
        sampled = len(X) - len(rows)
    mean = float(np.mean(sv))

    groups = [(c, np.sort(sv[labels[rows] == c])) for c in range(k)]
    groups = [(c, v) for c, v in groups if len(v)]
    if not groups:
        return None

    with _Figure(height=max(3.4, 0.78 * len(groups) + 1.9)) as figure:
        ax = figure.fig.add_subplot(111)
        # One distribution per segment on a shared axis, rather than every respondent's bar
        # stacked into one column. The old form drew a solid block: at any realistic sample size
        # each bar was a fraction of a pixel tall, so it showed an outline and hid the individuals
        # it claimed to be about — and the comparison this chart exists for, WHICH group is the
        # weak one, meant measuring three silhouettes against each other by eye.
        #
        # Heights are normalised per segment, so a 40-person group's shape is as readable as a
        # 900-person group's. That is a deliberate choice and the sizes are printed beside each
        # row, because shape is the question here and absolute counts are answered elsewhere.
        # Bin count follows the smallest group, so a 40-person segment is not drawn as a comb of
        # single-respondent spikes next to a smooth 900-person one.
        smallest = min(len(v) for _, v in groups)
        bins = int(np.clip(np.sqrt(smallest) * 1.15, 10, 26))
        edges = np.linspace(float(np.min(sv)), float(np.max(sv)), bins + 1)
        if edges[-1] <= edges[0]:
            edges = np.linspace(edges[0] - 0.5, edges[0] + 0.5, bins + 1)
        mids = (edges[:-1] + edges[1:]) / 2
        # Row 0 sits at the TOP, laid out by arithmetic rather than by inverting the axis. Fills
        # grow upward from their own baseline, so each one reads as a distribution standing on its
        # line; inverting the axis instead left them hanging downward and dropped every label onto
        # the row beneath.
        for row, (c, values) in enumerate(groups):
            base = len(groups) - 1 - row
            counts, _ = np.histogram(values, bins=edges)
            shape = counts / (counts.max() or 1) * 0.72
            ax.fill_between(mids, base, base + shape, step="mid", color=seg_colour(c), alpha=0.82,
                            linewidth=0, label=_label(names, c))
            # A hairline under each ridge, so a row reads as a distribution standing on its own
            # line rather than as bars floating in the panel.
            ax.plot([edges[0], edges[-1]], [base, base], color=_INK, alpha=0.28, linewidth=0.8,
                    zorder=2)
            middle = float(np.median(values))
            # The segment's own median, and its size. Those two numbers are the whole reading:
            # where this group sits, and how much of the sample it speaks for.
            ax.plot([middle, middle], [base, base + 0.76], color=_SURFACE, linewidth=3, zorder=4)
            ax.plot([middle, middle], [base, base + 0.76], color=_INK, linewidth=1.1, zorder=5)
            ax.annotate(f"{_num(middle)}  ({len(values):,} people)", (middle, base + 0.78),
                        fontsize=9, ha="center", va="bottom", zorder=6)

        # The whole sample's typical value, so each row can be read against the study as a whole.
        ax.axvline(mean, color=_INK, linewidth=1, alpha=0.35, zorder=1)
        ax.annotate(f"whole sample {_num(mean)}", xy=(mean, 1.0),
                    xycoords=("data", "axes fraction"), fontsize=9.5, alpha=0.75,
                    ha="right" if mean > np.mean(edges[[0, -1]]) else "left",
                    va="bottom", xytext=(-5 if mean > np.mean(edges[[0, -1]]) else 5, 3),
                    textcoords="offset points")
        ax.set_yticks([len(groups) - 1 - row for row in range(len(groups))])
        ax.set_yticklabels([_label(names, c) for c, _ in groups], fontsize=9.5)
        ax.set_ylim(-0.12, len(groups) - 1 + 1.02)
        ax.set_xlabel("how well each person fits their group →")
        ax.grid(axis="y", visible=False)
        # Rows are labelled directly, so a legend would repeat itself.
        ax.set_ylabel("")

        # A silhouette can go negative (closer to another group); a fit score bottoms out at 0
        # (exactly between two). Count whichever this is, and say which.
        stranded = int((sv < 0).sum()) if sampled or fit is None else int((sv < 0.15).sum())
        if stranded == 0:
            misfits = ("No respondent is actually misfiled — everyone sits closer to their own "
                       "group than to any other.")
        elif fit is not None and not sampled:
            misfits = (f"{stranded} of {len(sv):,} respondents ({stranded / len(sv):.0%}) sit "
                       "almost exactly between two groups — they could belong to either, so "
                       "filter on the `fit` column before spending money on a list.")
        else:
            misfits = (f"{stranded} of {len(sv):,} respondents ({stranded / len(sv):.0%}) sit "
                       "closer to a different group than their own.")
        misfits += (f" Every one of the {len(sv):,} respondents is drawn here."
                    if not sampled else
                    f" (Drawn from a random {len(sv):,} respondents — measuring this the exact "
                    f"way needs every pair of respondents compared, which is not affordable for "
                    f"all {len(sv) + sampled:,} of them.)")
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

        # Same bins the ridges were drawn from, so the interactive version can answer what the
        # picture can only gesture at: how many people in THIS group scored between here and here
        # — which is what somebody filtering a list before spending money actually needs.
        spec = {
            "kind": "fit",
            "edges": [round(float(v), 4) for v in edges],
            "segments": _segment_key(int(labels.max()) + 1, names),
            "rows": [{"segment": int(c),
                      "counts": [int(v) for v in np.histogram(values, bins=edges)[0]],
                      "median": round(float(np.median(values)), 4),
                      "people": int(len(values))} for c, values in groups],
            "overall_median": round(float(mean), 4),
            "sampled": int(sampled),
        }
        return _finish(figure, "fit", "Who actually belongs — fit of every respondent", verdict,
                       spec=spec)


#: The deciding criterion on the choice-of-k chart keeps the accent; the corroborating ones are
#: drawn in ink. These lines are METRICS, not segments — colouring them from the identity palette
#: made orange mean "Group 1" on four charts and "Separation (silhouette)" on this one.
_METRIC_INK = _INK
_METRIC_MARKS = ("s", "^", "D", "v")


def chart_k_choice(diag, rec_k):
    """The elbow plot's honest cousin: every quality measure at every number of groups tried.

    An elbow alone invites reading a bend that is not there. Plotting the stability measures
    against the 0.80 line makes "no number of groups reproduces" visible rather than arguable.
    """
    if diag is None or len(diag) == 0 or "k" not in diag:
        return None
    series = [("prediction_strength", "Prediction strength", "#46785C"),
              ("stability_ARI", "Reproducibility (ARI)", _METRIC_INK),
              ("silhouette", "Separation (silhouette)", _METRIC_INK)]
    present = [(col, name, colour) for col, name, colour in series if col in diag]
    if not present:
        return None

    with _Figure(height=4.0) as figure:
        ax = figure.fig.add_subplot(111)
        ks = list(diag["k"])
        best = 0.0
        # Emphasis rather than eight hues: prediction strength is the criterion the panel weighs
        # most, so it leads and the corroborating lines recede into ink. Three equally loud
        # coloured lines invited the reader to pick whichever one supported the answer they
        # wanted, and borrowed the segment palette to do it.
        for idx, (col, name, colour) in enumerate(present):
            values = pd.to_numeric(diag[col], errors="coerce")
            lead = col == "prediction_strength"
            # Marker shape separates the two recessive lines. Receding them both to the same ink
            # made the legend the only way to tell reproducibility from separation, which is no
            # way to read a chart.
            ax.plot(ks, values, marker="o" if lead else _METRIC_MARKS[idx % len(_METRIC_MARKS)],
                    markersize=5.5 if lead else 4.5,
                    linewidth=2.4 if lead else 1.6,
                    color=_METRIC_LEAD if lead else _METRIC_INK,
                    alpha=1.0 if lead else 0.45, zorder=3 if lead else 2,
                    label=name + (" — decides" if lead else ""))
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
            # Below the axis, not inside the plot. Pinned to the top it landed exactly where the
            # lines cross on a typical run and became unreadable — the label sat on the data it
            # was describing.
            ax.annotate(f"chosen: {rec_k}", (rec_k, 0), xycoords=("data", "axes fraction"),
                        fontsize=10, fontweight="bold", ha="center", va="top",
                        xytext=(0, -20), textcoords="offset points")
        ax.set_xticks(ks)
        ax.set_xlabel("number of groups")
        ax.set_ylim(min(0.0, ax.get_ylim()[0]), max(1.0, ax.get_ylim()[1]))
        figure.legend_below(ax, ncol=3)

        read = ("The chosen number sits at a clear peak above the 0.80 line — that is a real "
                "answer from the data." if best >= 0.8 else
                "Nothing reaches the 0.80 line, which means no number of groups reproduces "
                "strongly. Treat the groups as a working hypothesis, not a finding, and lean on "
                "judgement about how many you can actually act on.")
        spec = {
            "kind": "k_choice",
            "ks": [int(v) for v in ks],
            "chosen": int(rec_k),
            "cutoff": 0.80,
            # One entry per criterion drawn, so hovering a number of groups can report every
            # measure at once instead of asking the reader to trace three lines back to an axis.
            "series": [{"key": col, "label": name,
                        "lead": col == "prediction_strength",
                        "values": [None if not np.isfinite(v) else round(float(v), 4)
                                   for v in pd.to_numeric(diag[col], errors="coerce")]}
                       for col, name, _ in present],
        }
        return _finish(figure, "k", "Was the number of groups a clear call?",
                       "Each line is a different test of quality, run for every number of groups "
                       "the tool tried. Higher is better for all three. " + read, spec=spec)


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
    """What actually distinguishes the groups, on the original answer scale.

    A Cleveland dot plot rather than grouped bars, for two reasons that both bit the old version.

    **Bars have to start at zero, and a rating scale does not.** Drawn from 0 on a 1-5 scale,
    every bar carried a meaningless stub from 0 to 1 and the differences that matter — usually
    within one or two steps — were squeezed into the far end. A dot has no baseline to be honest
    about, so the axis can show the answer scale as it really is.

    **A dot plot compares groups within a question, which is the question being asked.** Grouped
    bars put the reader's eye on lengths from a shared left edge; the reading anybody actually
    wants is "how far apart are these groups on this question", and that is a distance between
    dots on one line. The connecting rule makes that distance the visible thing.
    """
    if centroids is None or centroids.empty:
        return None
    keep, trimmed = _separating(centroids, max_items)
    if not keep:
        return None
    data = centroids[keep]
    k = len(data.index)

    with _Figure(height=max(3.4, 0.46 * len(keep) + 1.6)) as figure:
        ax = figure.fig.add_subplot(111)
        y = np.arange(len(keep))
        values = data.to_numpy(float)
        # The rule joining a question's dots: it is the spread on that question, so a reader can
        # see at a glance which questions actually pull the groups apart.
        for j, row in enumerate(values.T):
            ax.plot([np.nanmin(row), np.nanmax(row)], [y[j], y[j]], color=_INK, alpha=0.22,
                    linewidth=2.5, solid_capstyle="round", zorder=1)
        # Nudge apart, vertically, any groups that answered a question almost identically. Two
        # groups landing on the same value would otherwise draw one dot on top of another and the
        # chart would silently show five groups where there are six. The offset is along the
        # category axis, where position carries no meaning — the value each dot reports is
        # untouched, which is the difference between separating marks and jittering data.
        offsets = np.zeros_like(values)
        if k > 1 and np.isfinite(values).any():
            reach = float(np.nanmax(values) - np.nanmin(values)) or 1.0
            for j in range(values.shape[1]):
                order = np.argsort(values[:, j])
                lane, previous = 0, -np.inf
                for i in order:
                    if values[i, j] - previous < reach * 0.045:
                        lane += 1
                    else:
                        lane = 0
                    offsets[i, j] = lane
                    previous = values[i, j]
            # Centre each question's stack on its own row so the rule still runs through it.
            offsets -= offsets.mean(0)
            offsets *= 0.13

        for i in range(k):
            ax.scatter(values[i], y + offsets[i], s=118, color=seg_colour(i),
                       marker=seg_marker(i), label=_label(names, i), zorder=3,
                       edgecolors=_SURFACE, linewidths=1.4)
        ax.set_yticks(y)
        ax.set_yticklabels([_short(c, 26) for c in keep])
        ax.set_ylim(len(keep) - 0.5, -0.5)
        # Start the axis at the bottom of the answer scale, not at zero. On 1-5 data the old
        # zero-based bars spent a fifth of their length on a region no respondent can occupy.
        if kind != "shares" and np.isfinite(values).any():
            # Guarded: a column that is entirely missing makes nanmin/nanmax NaN, and matplotlib
            # rejects a NaN limit outright — which took the whole chart down rather than just the
            # nicer axis. Fall back to the default limits, which are always drawable.
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
            if np.isfinite(lo) and np.isfinite(hi):
                pad = max(0.25, (hi - lo) * 0.12)
                ax.set_xlim(lo - pad, hi + pad)
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
        caption += (" Each row is one question and each mark is one group, joined by a rule "
                    "showing how far apart they are. A long rule is a real difference you can "
                    "write a brief around; dots sitting almost on top of one another mean that "
                    "question does not distinguish anybody, whatever the report calls it.")
        if trimmed > 0:
            caption += (f" ({trimmed} more question{'' if trimmed == 1 else 's'} separated the "
                        "groups less and would not fit legibly here. The <strong>Full grid</strong> "
                        "tab shows every question with nothing left out.)")
        return _finish(figure, "profiles", "What makes the groups different", caption)


# There was a radar ("spider") chart here, showing each group as a polygon over the questions. It
# was removed rather than fixed. A radar encodes value as distance from a centre, so the eye reads
# the enclosed AREA — which grows with the square of the values and, worse, changes entirely when
# the questions are reordered, an order that carries no meaning at all. Three of its six labels
# also truncated ("I want premium qu…") and one overlapped the plot. The Cleveland dot plot in
# chart_profiles answers the same question — how far apart are these groups on this item — as a
# distance along a common axis, which is the comparison people actually read accurately.
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
        # DIVERGING, not the identity palette. The old scale ran between two of the very hues
        # that mean "Group 1" and "Group 2" elsewhere in the report, so orange meant a segment on
        # four charts and "above average" on this one. Warm against cool with a genuinely neutral
        # middle, kept clear of the identity slots.
        cmap = LinearSegmentedColormap.from_list("divergent", list(DIVERGING))
        mesh = ax.pcolormesh(centred.values.T, cmap=cmap, vmin=-limit, vmax=limit,
                             edgecolors="none")
        ax.set_xticks(np.arange(len(data.index)) + 0.5)
        # Horizontal: with a handful of short group names there is nothing to avoid, and angled
        # text is harder to read for no gain. Only tilt when the names are genuinely long.
        _heads = [_label(names, i) for i in range(len(data.index))]
        _tilt = len(_heads) > 6 or max((len(h) for h in _heads), default=0) > 12
        ax.set_xticklabels([_short(h, 18) for h in _heads],
                           rotation=30 if _tilt else 0,
                           ha="right" if _tilt else "center", fontsize=9.5)
        ax.set_yticks(np.arange(len(items)) + 0.5)
        ax.set_yticklabels([_short(c, 30) for c in items], fontsize=9.5)
        ax.invert_yaxis()
        ax.grid(False)

        # The value in the cell. Nobody can read a number off a colour, and this grid is small
        # enough to label every cell — which is the one case where labelling everything is right
        # rather than chaos. The colour then does what colour is good at: showing the pattern at
        # a glance, while the number answers "by how much".
        if len(items) * len(data.index) <= 90:
            for col, item in enumerate(items):
                for row in range(len(data.index)):
                    value = centred.values[row, col]
                    ax.text(row + 0.5, col + 0.5, _num(data.values[row, col], 1),
                            ha="center", va="center", fontsize=8.5,
                            color="white" if abs(value) > limit * 0.55 else _INK)

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
        # Same contract as the map: the numbers behind the picture, built from the arrays that
        # were just drawn. The interactive version can then report a cell's exact value and how
        # far from that question's average it sits, which colour can only suggest.
        spec = {
            "kind": "heatmap",
            "items": [str(c) for c in items],
            "segments": _segment_key(len(data.index), names),
            # values[row][col] is a group's answer on a question; deviation is the same cell
            # measured against that question's own average, which is what the colour encodes.
            "values": [[round(float(v), 3) for v in row] for row in data.to_numpy(float)],
            "deviation": [[round(float(v), 3) for v in row] for row in centred.to_numpy(float)],
            "limit": round(float(limit), 3),
            "kind_of_value": "share" if kind == "shares" else "average",
        }
        return _finish(figure, "heatmap", "Every group against every question", caption,
                       spec=spec)


def chart_gorge(shadow, names=None):
    """The distribution of how stranded people are between segments — Leisch's gorge plot.

    For every respondent, 2·d(closest) / [d(closest) + d(second closest)]. Near 0 they sit
    squarely inside their own segment; near 1 they are halfway between two and could belong to
    either. Piled up on the left with a dip in the middle — a gorge — means the segments genuinely
    separate. One lump sitting near the right means nobody is clearly anywhere and the split is
    imposed.

    This is the tool's central question answered in a single picture, and it needs no statistical
    training to read. Leisch introduces it in *Neighborhood Graphs, Stripes and Shadow Plots*
    (2010), noting it is "similar both in spirit and interpretation to the well known silhouette
    plots" — but a shape, rather than a sorted bar for every respondent.
    """
    shadow = np.asarray(shadow, float)
    shadow = shadow[np.isfinite(shadow)]
    if len(shadow) < 20:
        return None

    with _Figure(height=3.9) as figure:
        ax = figure.fig.add_subplot(111)
        # Ink, not a segment colour. This histogram is the whole sample, so wearing Group 0's
        # hue implied it was about Group 0 — the same colour-means-two-things error the palette
        # rework exists to remove.
        ax.hist(shadow, bins=np.linspace(0, 1, 41), color=_INK, alpha=0.55, linewidth=0)
        median = float(np.median(shadow))
        ax.axvline(median, color="#9C4029", linewidth=1.4, linestyle="--")
        ax.annotate(f"typical respondent {median:.2f}", xy=(median, 1.0),
                    xycoords=("data", "axes fraction"), color="#9C4029", fontsize=10,
                    ha="right" if median > 0.5 else "left", va="bottom",
                    xytext=(-4 if median > 0.5 else 4, 4), textcoords="offset points")
        ax.set_xlim(0, 1)
        ax.set_xlabel("← sits firmly in one segment        stranded between two →")
        ax.set_ylabel("respondents")
        ax.grid(axis="x", visible=False)

        stranded = float((shadow > 0.8).mean())
        firm = float((shadow < 0.4).mean())
        if firm >= 0.5 and median < 0.55:
            verdict = ("<strong>This is the shape you want.</strong> Most respondents sit firmly "
                       "inside one segment, and the tail off to the right is the handful on the "
                       "boundaries.")
        elif median > 0.75:
            verdict = ("<strong>This is the shape of a split that was imposed rather than "
                       "found.</strong> The typical respondent is nearly equidistant from two "
                       "segments, which is what happens when there are no real groups to find.")
        else:
            verdict = ("<strong>A mixed picture.</strong> A core of each segment sits firmly "
                       "inside it, but a substantial share of people could reasonably have gone "
                       "either way.")
        caption = (f"Every respondent, measured by how much closer they are to their own segment "
                   f"than to the next nearest. {firm:.0%} sit firmly in one segment; "
                   f"{stranded:.0%} are essentially on a boundary. " + verdict +
                   " Two humps with a dip between them — a gorge — is the signature of segments "
                   "that genuinely separate.")
        return _finish(figure, "gorge", "Does anyone actually belong to one segment?", caption)


def build_charts(seg, method, names=None, errors=None):
    """Draw everything this result supports, skipping whatever it does not.

    Each chart is built inside its own try: they are independent, and a failure in one must not
    withhold the rest. This was a real bug once — the charts were built as a single eagerly
    evaluated tuple, so one NaN centroid discarded all of them, the segment map included.
    """
    out = []
    # Why a chart is missing, collected into a list the CALLER owns rather than a module global.
    #
    # It was a global, and the server is threaded: two people analysing at once clobbered each
    # other's list. Measured on 18 concurrent runs where a third of them failed — three healthy
    # runs were told charts had failed that never did, four failed runs were given no reason at
    # all, and five were handed someone else's failures alongside their own. Whoever asks for
    # the charts owns the account of what went wrong with them.
    #
    # Kept at all because the packaged app is built --windowed, which discards stdout: a printed
    # reason is invisible exactly where failures are most likely. The first packaged build after
    # moving to matplotlib drew nothing and said nothing about why.
    errors = [] if errors is None else errors
    del errors[:]

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
            errors.append(reason)
            print(f"NOTE: could not draw the '{label}' chart ({type(e).__name__}: {e}); "
                  "the rest of the report is unaffected.")
            return
        if trace:
            print(f"chart: {label} took {time.monotonic() - started:.1f}s")
        if chart:
            out.append(chart)

    metric = "euclidean"
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
        elif method == "kproto":
            # Codes standing for brands are not coordinates. The Gower embedding is (see
            # kprototypes.gower_embedding), and on it plain Manhattan distance reproduces the
            # distance the segmentation was actually built on — so the map projects a real space
            # and the per-person bars measure what put each person where they are.
            import kprototypes
            X = kprototypes.gower_embedding(np.asarray(seg.X, float), seg.cfg.gower_spec)
            metric = "manhattan"
            # A pick-any column holds its segment's most common ANSWER, which is text; the
            # profile, radar and grid charts plot quantities, so they show the rating questions.
            centroids = seg.centroids.select_dtypes(include=[np.number])
            if centroids.empty:
                centroids = None
        else:
            X, centroids = seg.X, seg.centroids
        labels = np.asarray(seg.labels)
    except Exception as e:          # the shared inputs failed; there is nothing to draw at all
        errors.append(f"preparing the data: {type(e).__name__}: {e}")
        print(f"NOTE: could not prepare the charts ({type(e).__name__}: {e}); "
              "the report itself is unaffected.")
        return []

    kind = "shares" if method == "lca" else "means"
    # The order is the order somebody should read them in, and it is also the tab order in the
    # app. Two questions, asked in the only sequence that makes sense: FIRST whether these groups
    # are real at all, and only then what they contain. Describing a segmentation before
    # establishing it exists is how a partition of noise acquires a persona.
    #
    #   are they real?    the map, the gorge, per-person fit, the choice of k
    #   what are they?    what differs between them, then the full grid
    #
    # The gorge used to sit last, after the profile charts — a "does this hold up" chart placed
    # where a reader has already stopped asking.
    _shadow = getattr(seg, "shadow", None)
    # 1 - shadow is the `fit` column of the exported file, computed for every respondent. Passing
    # it means that chart covers everyone and cannot disagree with the CSV. The latent-class path
    # has no shadow value, so it falls back to the sampled silhouette inside the chart.
    _fit = None if _shadow is None else 1.0 - np.asarray(_shadow, float)

    attempt("segment map", lambda: chart_segment_map(X, labels, names))
    if _shadow is not None:
        attempt("gorge", lambda: chart_gorge(_shadow, names))
    attempt("per-person fit",
            lambda: chart_silhouette(X, labels, names, metric=metric, fit=_fit))
    attempt("choice of k", lambda: chart_k_choice(seg.diagnostics, int(seg.recommended_k)))
    if centroids is not None:
        attempt("group profiles", lambda: chart_profiles(centroids, names, kind=kind))
        attempt("full grid", lambda: chart_heatmap(centroids, names, kind=kind))
    return out
