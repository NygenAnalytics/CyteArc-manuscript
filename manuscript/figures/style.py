"""Shared style for every manuscript figure, and the lint that holds figures to it.

This module defines:
  - the colour tokens: neutrals with fixed roles, one colour and marker per tool (ENTITIES),
    a pastel tint and a strong tone per module of Fig. 1 (MODULES), six colours for groups of
    cells (GROUPS), the sequential, neutral and diverging colormaps and the ramp of the
    biological case;
  - the type scale (8, 7, 6.5 and 6 pt) and the mark sizes;
  - small drawing helpers (panel letters with headers, legends, failure states, tints);
  - lint(), which save() runs before writing a figure.

Saturated colour is data, pastel is structure, ink is words. A tool keeps its colour and
marker in every figure; a failure state is shown by form and a word, never by colour alone.
Figures are drawn at printed size (160 or 80 mm wide), so point sizes are printed sizes.
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb  # noqa: E402

# Neutrals: text and axes, baseline or comparator data, context magnitude, grid, schematic card.
INK = "#1B2A3A"
SLATE = "#7D8590"
LIGHT = "#CBD0D6"
GRID = "#E6E8EB"
PAPER = "#F4F6F8"
WHITE = "#FFFFFF"

# Entity colours: the Okabe and Ito palette plus Tol's wine, validated as a set on white.
BLUE = "#0072B2"
VERMILION = "#D55E00"
ORANGE = "#E69F00"
GREEN = "#009E73"
WINE = "#882255"
SKY = "#56B4E9"
PINK = "#CC79A7"


@dataclass(frozen=True)
class Entity:
    label: str
    colour: str
    marker: str


# Fixed order: legends list tools in this order. A variant of a tool (Scanpy on Dask, Seurat
# with BPCells) keeps its colour and marker and is drawn open or dashed.
ENTITIES = {
    "cytearc": Entity("CyteArc", BLUE, "o"),
    "scanpy": Entity("Scanpy", VERMILION, "s"),
    "seurat": Entity("Seurat", ORANGE, "^"),
    "bpcells": Entity("BPCells", GREEN, "D"),
    "celltypist": Entity("CellTypist", WINE, "P"),
    "scanvi": Entity("scANVI", SKY, "v"),
    "symphony": Entity("Symphony", PINK, "h"),
}
# Tools outside the registry (reference implementations, one-off comparators) are slate and
# named on the category axis or labelled directly.
COMPARATOR = SLATE


@dataclass(frozen=True)
class Module:
    name: str
    tint: str
    strong: str


# The modules of Fig. 1; tints fill schematic cards only, strong tones draw their borders.
MODULES = {
    "stream": Module("Stream", "#FFF5EA", "#C28C4B"),
    "record": Module("Record", "#F8F4FF", "#6D5496"),
    "extend": Module("Extend", "#EFFAF0", "#2F7442"),
    "map": Module("Map", "#EBF9FF", "#48A4CD"),
}

# Groups of cells (clusters, cell types) in thumbnails and panels without tools: six hues
# validated as a set on white with all pairs compared, and kept apart from CyteArc blue.
GROUPS = ("#9B4630", "#B0A03C", "#52CAA5", "#69A1E8", "#7D4B92", "#C46B90")

# Ordered categories of the biological case: one hue, validated as an ordinal ramp.
CONTROL = SLATE
LESS_FIBROTIC = "#E49581"
MORE_FIBROTIC = "#923D28"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "cytearc_sequential", ["#F7FAFD", "#B1D2F4", "#73A9E1", "#3C7EBE", "#1F5587", "#0E375C"]
)
# Magnitude in panels that compare tools, where the blue ramp would read as CyteArc: the neutrals.
NEUTRAL = LinearSegmentedColormap.from_list("cytearc_neutral", ["#F7F8FA", LIGHT, SLATE, INK])
DIVERGING = LinearSegmentedColormap.from_list("cytearc_diverging", [BLUE, "#EEF0F2", VERMILION])
COLORMAPS = {SEQUENTIAL.name, NEUTRAL.name, DIVERGING.name}

MM = 1 / 25.4  # inches per millimetre
FULL_WIDTH = 160 * MM  # the text width of the manuscript and the journal's double column
SINGLE_WIDTH = 80 * MM
MAX_HEIGHT = 225 * MM

# Type scale in points; nothing else is allowed.
LETTER = 8.0
LABEL = 7.0
TICK = 6.5
NOTE = 6.0
SMALL = NOTE
SIZES = (NOTE, TICK, LABEL, LETTER)

# Marks.
LW_DATA = 1.0
LW_AXIS = 0.6
LW_THIN = 0.5
LW_PRIMARY = 1.2
LW_MARKER = 0.7
LW_BAR = 0.65
LW_FRAME = 0.55
LW_CELL = 0.38
MARKER = 12  # scatter area in square points, about 1.4 mm across
MARKER_MS = 3.6  # the same marker as a Line2D marker size
REPLICATE = 5
BAR = 0.62
GAP = 0.8  # white gap between stacked parts, in points
HATCH = "////"
FONT = "TeX Gyre Heros"

TEXT_COLOURS = (INK, SLATE, WHITE)
BOLD_ROLES = ("panel-letter", "card-title")


def tint(colour: str, share: float) -> str:
    """The colour mixed with white: share 1 is the colour itself, 0 is white."""
    rgb = np.asarray(to_rgb(colour))
    return to_hex(share * rgb + (1 - share) * 1.0)


# The source stores of a composite store (Fig. 1d): cells grouped by source, which must read apart at a
# glance both as swatches and as the pale tints that fill most of their counts. A brighter rust, the GROUPS
# mint and a royal blue, validated as a set on white (CIEDE2000, all pairs: at least 44.8, and 26.0 under
# simulated deutan, protan and tritan vision; their tints at SOURCE_FLOOR at least 17.6). The blue is kept
# apart from CyteArc blue (12.3) and the middle of the sequential ramp (12.9), the rust from Scanpy
# vermilion (12.9); never tool hues. The counts of each source are drawn from its tint at SOURCE_FLOOR
# through its hue to ink, so that cells keep their source's colour in the composite.
SOURCE_RUST = "#C4472C"
SOURCE_BLUE = "#2D5BD8"
SOURCE_HUES = (SOURCE_RUST, GROUPS[2], SOURCE_BLUE)
SOURCE_FLOOR = 0.22
SOURCE_MAPS = tuple(
    LinearSegmentedColormap.from_list(f"source_{i + 1}", [tint(h, SOURCE_FLOOR), h, INK], N=64)
    for i, h in enumerate(SOURCE_HUES)
)
COLORMAPS |= {m.name for m in SOURCE_MAPS}

# CyteArc's memory budgets (Fig. 2d): one blue ramp, darker for a larger budget, in even steps of lightness
# (L* 24, 52 and 78 for 24, 8 and 4 GiB): neighbours differ by 23-24 (CIEDE2000) and by at least 19.7 under
# simulated deutan, protan and tritan vision, and the lightest keeps a contrast of 1.8 on white. 192 GiB,
# Tahoe-100M's budget, is the darkest; the figure rings it in BLUE to set it apart from 24 GiB.
BUDGET_COLOURS = {192: "#061A2E", 24: "#0B3B66", 8: "#2F7FC4", 4: "#8CC8EE"}


def parts(colour: str, n: int) -> list[str]:
    """Tints of one colour for the parts of one quantity: the full colour at the baseline, then lighter."""
    if n == 1:
        return [colour]
    return [tint(colour, share) for share in np.linspace(1.0, 0.3, n)]


def apply(*, dense: bool = False) -> None:
    global NOTE, SMALL
    NOTE = TICK if dense else 6.0
    SMALL = NOTE
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT],
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT,
            "mathtext.it": f"{FONT}:italic",
            "mathtext.bf": f"{FONT}:bold",
            "font.size": TICK,
            "axes.titlesize": TICK,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlepad": 3,
            "axes.labelsize": LABEL,
            "axes.labelpad": 2.5,
            "xtick.labelsize": TICK,
            "ytick.labelsize": TICK,
            "legend.fontsize": TICK,
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "legend.borderaxespad": 0.2,
            "legend.borderpad": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": LW_AXIS,
            "xtick.major.width": LW_AXIS,
            "ytick.major.width": LW_AXIS,
            "xtick.minor.width": LW_THIN,
            "ytick.minor.width": LW_THIN,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "lines.linewidth": LW_DATA,
            "lines.markersize": MARKER_MS,
            "lines.markeredgewidth": LW_MARKER if dense else 0.6,
            "scatter.edgecolors": INK if dense else "face",
            "patch.linewidth": LW_BAR if dense else LW_THIN,
            "patch.edgecolor": INK if dense else "black",
            "patch.force_edgecolor": dense,
            "hatch.linewidth": 0.6,
            "grid.color": GRID,
            "grid.linewidth": LW_THIN,
            "errorbar.capsize": 0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def letter(fig, x: float, y: float, text: str, header: str | None = None) -> None:
    """Panel letter at the top-left of the panel cell; y is the top of the panel cell."""
    _ = header
    height_pt = fig.get_figheight() * 72
    baseline = y - 0.72 * LETTER / height_pt
    fig.text(
        x,
        baseline,
        text,
        fontsize=LETTER,
        fontweight="bold",
        va="baseline",
        ha="left",
        color=INK,
        gid="panel-letter",
    )


def add_axes(fig, x: float, y: float, w: float, h: float, **kwargs):
    """A data axes at (x, y, w, h) millimetres from the bottom-left corner of the figure."""
    fw, fh = fig.get_figwidth() / MM, fig.get_figheight() / MM
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh], **kwargs)


def mm_above(ax, mm: float) -> float:
    """Axes y coordinate the given millimetres above the top of the plot (for legends and notes)."""
    return 1 + mm * MM / (ax.get_position().height * ax.figure.get_figheight())


def panel_label(fig, x: float, top: float, text: str, header: str | None = None) -> None:
    """Panel letter and header with the top of the panel cell at (x, top) millimetres."""
    letter(fig, x * MM / fig.get_figwidth(), top * MM / fig.get_figheight(), text, header)


def panel_title(fig, x: float, top: float, text: str) -> None:
    """A short title above a data panel, used only where the dataset or comparison is not obvious."""
    fig.text(
        x * MM / fig.get_figwidth(),
        top * MM / fig.get_figheight(),
        text,
        fontsize=LABEL,
        fontweight="bold",
        va="baseline",
        ha="left",
        color=INK,
        gid="card-title",
    )


def percent_axis(ax, axis: str = "both") -> None:
    from matplotlib.ticker import PercentFormatter

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def plain_log_axis(ax, axis: str = "x") -> None:
    """Log axis with plain numbers (0.1, 1, 10) and no minor labels."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    def fmt(value, _):
        return f"{value:,.0f}" if value >= 1 else f"{value:g}"

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(FuncFormatter(fmt))
    target.set_minor_formatter(NullFormatter())


def note(ax, x: float, y: float, text: str, **kwargs) -> None:
    """Secondary annotation in axes coordinates: 6 pt, slate."""
    kwargs.setdefault("fontsize", NOTE)
    kwargs.setdefault("color", SLATE)
    kwargs.setdefault("transform", ax.transAxes)
    ax.text(x, y, text, **kwargs)


def category_axis(ax, axis: str = "y") -> None:
    """A category axis carries names only: no spine and no ticks."""
    if axis == "y":
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    else:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)


def frame_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(LW_FRAME)
    ax.tick_params(top=False, right=False)


def heatmap_cells(ax, shape: tuple[int, int]) -> None:
    rows, columns = shape
    ax.set_xticks(np.arange(columns + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=LIGHT, linewidth=LW_CELL)
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)
    frame_axes(ax)


def category_raster(
    counts: np.ndarray, colours: list[str], *, seed: int = 0, pad: int | None = 6
) -> np.ndarray:
    """An embedding binned on a grid as an RGB image, from counts of cells per category, row and column: each occupied
    bin takes the colour of one of its cells drawn at random, as a scatter of every cell in random order would show
    it, faded toward white where few cells fall. The image is cropped to the occupied bins and pad bins around them,
    or keeps the whole grid when pad is None, so that each bin is the pixel at its row and column."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3 or len(colours) != counts.shape[0]:
        raise ValueError(
            f"counts of shape {counts.shape} need one colour per category, not {len(colours)}"
        )
    palette = np.array([to_rgb(colour) for colour in colours])
    total = counts.sum(axis=0)
    occupied = total > 0
    drawn = np.random.default_rng(seed).uniform(0.0, 1.0, int(occupied.sum())) * total[occupied]
    category = (np.cumsum(counts[:, occupied], axis=0) < drawn).sum(axis=0)
    density = np.log1p(total[occupied])
    share = np.clip(0.3 + 0.7 * density / np.quantile(density, 0.98), 0.3, 1.0)[:, None]
    image = np.ones(total.shape + (3,))
    image[occupied] = share * palette[category] + (1 - share)
    if pad is None:
        return image
    rows, columns = np.nonzero(occupied)
    return image[
        max(rows.min() - pad, 0) : rows.max() + pad + 1,
        max(columns.min() - pad, 0) : columns.max() + pad + 1,
    ]


def bar_style(colour: str) -> dict:
    return {"facecolor": tint(colour, 0.78), "edgecolor": INK, "linewidth": LW_BAR}


def entity_handle(
    key: str,
    *,
    variant: bool = False,
    line: bool = True,
    colour: str | None = None,
    marker: str | None = None,
):
    """Legend handle for a tool: its colour and marker, open and dashed for a variant."""
    e = ENTITIES[key] if key in ENTITIES else Entity(key, COMPARATOR, "o")
    c = colour or e.colour
    return plt.Line2D(
        [],
        [],
        color=c,
        marker=marker or e.marker,
        ms=MARKER_MS,
        mfc="white" if variant else c,
        mec=c,
        lw=LW_DATA if line else 0,
        ls="--" if (variant and line) else "-",
    )


def points(
    ax,
    xs,
    ys,
    colour: str,
    marker: str = "o",
    *,
    variant: bool = False,
    size: float = MARKER,
    zorder: float = 3,
    **kwargs,
):
    """Markers by the grammar: filled with a white ring (main) or open (variant), of equal visual size."""
    if variant:
        kwargs.setdefault("facecolor", WHITE)
        kwargs.setdefault("edgecolor", colour)
        kwargs.setdefault("linewidths", LW_MARKER)
        return ax.scatter(
            xs,
            ys,
            s=size,
            marker=marker,
            zorder=zorder,
            **kwargs,
        )
    kwargs.setdefault("facecolor", colour)
    kwargs.setdefault("edgecolor", WHITE)
    kwargs.setdefault("linewidths", 0.5)
    return ax.scatter(
        xs,
        ys,
        s=size * 1.4,
        marker=marker,
        zorder=zorder,
        **kwargs,
    )


def legend_above(
    ax, handles, labels, *, ncol: int | None = None, y: float = 1.02, x: float = 0.0, **kwargs
):
    """Legend as one row above the plot, in registry order, never over the data."""
    return ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(x, y),
        ncol=ncol or len(labels),
        borderaxespad=0.0,
        **kwargs,
    )


def state_label(ax, x: float, y: float, state: str, **kwargs) -> None:
    """The word of a failure state, in ink, beside its mark."""
    words = {"oom": "out of memory", "crash": "crashed", "na": "n/a"}
    kwargs.setdefault("fontsize", TICK)
    kwargs.setdefault("color", INK)
    kwargs.setdefault("va", "center")
    ax.text(x, y, words.get(state, state), **kwargs)


def oom_bar(
    ax,
    position: float,
    length: float,
    colour: str,
    *,
    horizontal: bool = True,
    left: float = 0.0,
    width: float = BAR,
):
    """A run that ran out of memory: hatched in the tool's colour, outlined."""
    kwargs = {"color": "white", "edgecolor": colour, "hatch": HATCH, "lw": LW_THIN}
    if horizontal:
        return ax.barh(position, length, left=left, height=width, **kwargs)
    return ax.bar(position, length, bottom=left, width=width, **kwargs)


def timeout_bar(
    ax,
    position: float,
    length: float,
    colour: str,
    *,
    horizontal: bool = True,
    left: float = 0.0,
    width: float = BAR,
):
    """A run that did not finish in time: dotted outline to the time limit, unfilled."""
    kwargs = {"color": "white", "edgecolor": colour, "lw": LW_DATA, "ls": ":"}
    if horizontal:
        return ax.barh(position, length, left=left, height=width, **kwargs)
    return ax.bar(position, length, bottom=left, width=width, **kwargs)


def crash_mark(ax, x: float, y: float, **kwargs):
    """A run that crashed: a cross in ink where the value would be."""
    return ax.scatter([x], [y], marker="x", s=14, color=INK, linewidths=0.9, zorder=5, **kwargs)


class FigureLintError(RuntimeError):
    def __init__(self, stem: str, problems: list[str]):
        self.stem = stem
        self.problems = problems
        super().__init__(f"{stem}: {len(problems)} grammar violations\n  " + "\n  ".join(problems))


_TOKENS = [INK, SLATE, LIGHT, GRID, PAPER, *(e.colour for e in ENTITIES.values())]
_TOKENS += (
    [m.tint for m in MODULES.values()]
    + [m.strong for m in MODULES.values()]
    + [LESS_FIBROTIC, MORE_FIBROTIC, *GROUPS, SOURCE_RUST, SOURCE_BLUE, *BUDGET_COLOURS.values()]
)


def _is_token(rgb) -> bool:
    """A token, or a token mixed with white (a tint or a mark drawn with alpha on white)."""
    rgb = np.asarray(rgb[:3], dtype=float)
    if np.all(rgb > 0.985):
        return True
    for token in _TOKENS:
        t = np.asarray(to_rgb(token))
        room = 1 - t
        mask = room > 0.03
        shares = (1 - rgb[mask]) / room[mask]
        if (
            np.all(np.abs(rgb[~mask] - 1) < 0.03)
            and np.ptp(shares) < 0.035
            and -0.02 <= shares.mean() <= 1.02
        ):
            return True
    return False


def _hexes(colours) -> list:
    array = (
        np.atleast_2d(np.asarray(colours, dtype=float))
        if len(np.shape(colours))
        else np.empty((0, 4))
    )
    return [c for c in array if c.size >= 3 and (c.size < 4 or c[3] > 0)]


def _near(rgb, token: str) -> bool:
    return bool(np.max(np.abs(np.asarray(rgb[:3]) - np.asarray(to_rgb(token)))) < 0.01)


def _corners(box) -> np.ndarray:
    """A box's four corners in display coordinates."""
    return np.array([[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]])


def _turned_outline(t, renderer, box, inset: float) -> np.ndarray | None:
    """The four corners of a tilted text's box as drawn, in display coordinates, each side moved in by inset; None for
    text at a multiple of 90 degrees, whose upright box is its outline. The upright box of a turned rectangle shares
    its centre, so the outline is the text's unturned box, measured at rotation 0, turned about that centre."""
    rotation = t.get_rotation()
    if rotation % 90 == 0:
        return None
    t.set_rotation(0)
    flat = t.get_window_extent(renderer)
    t.set_rotation(rotation)
    half_w, half_h = flat.width / 2 - inset, flat.height / 2 - inset
    theta = np.radians(rotation)
    along = np.array([np.cos(theta), np.sin(theta)])
    across = np.array([-np.sin(theta), np.cos(theta)])
    centre = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
    return np.array(
        [
            centre - half_w * along - half_h * across,
            centre + half_w * along - half_h * across,
            centre + half_w * along + half_h * across,
            centre - half_w * along + half_h * across,
        ]
    )


def _outlines_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two convex outlines overlap: true unless an edge normal of either separates their projections."""
    for outline in (a, b):
        for start, end in zip(outline, np.roll(outline, -1, axis=0)):
            normal = np.array([start[1] - end[1], end[0] - start[0]])
            on_a, on_b = a @ normal, b @ normal
            if on_a.max() <= on_b.min() or on_b.max() <= on_a.min():
                return False
    return True


def lint(fig) -> list[str]:
    """Every grammar rule a program can check; returns the violations, empty when clean. Text overlaps are judged on
    each text's box as drawn: the upright box, or for tilted text its box turned with it."""
    from matplotlib.collections import Collection
    from matplotlib.image import AxesImage
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.text import Text

    problems: list[str] = []
    width_mm, height_mm = fig.get_figwidth() / MM, fig.get_figheight() / MM
    if min(abs(width_mm - 160), abs(width_mm - 80)) > 0.5:
        problems.append(f"canvas: width {width_mm:.1f} mm, not 80 or 160 mm")
    if height_mm > 225.5:
        problems.append(f"canvas: height {height_mm:.1f} mm, over 225 mm")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    canvas = fig.bbox

    # Tick labels exist for ticks outside the view limits but are not drawn; skip those.
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            try:
                drawn = {id(t) for t in axis._update_ticks()}
            except AttributeError:
                continue
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if id(tick) not in drawn or not ax.axison:
                    hidden.update({id(tick.label1), id(tick.label2)})

    texts = []
    for t in fig.findobj(Text):
        if id(t) in hidden or not t.get_visible() or not t.get_text().strip() or t.get_alpha() == 0:
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        label = t.get_text().strip().replace("\n", " ")[:40]
        size = t.get_fontsize()
        if not any(abs(size - s) < 0.05 for s in SIZES):
            problems.append(f"type: '{label}' at {size:g} pt")
        if t.get_fontname() != FONT:
            problems.append(f"type: '{label}' in {t.get_fontname()}")
        weight = t.get_fontweight()
        if (
            weight == "bold" or (isinstance(weight, int) and weight >= 600)
        ) and t.get_gid() not in BOLD_ROLES:
            problems.append(f"type: '{label}' bold outside a panel letter or card title")
        if not any(_near(to_rgb(t.get_color()), c) for c in TEXT_COLOURS):
            problems.append(f"type: '{label}' in {to_hex(t.get_color())}, not ink, slate or white")
        box = t.get_window_extent(renderer)
        if (
            box.x0 < canvas.x0 - pt
            or box.x1 > canvas.x1 + pt
            or box.y0 < canvas.y0 - pt
            or box.y1 > canvas.y1 + pt
        ):
            problems.append(f"layout: '{label}' leaves the canvas")
        padded = box.expanded(1, 1).padded(-0.35 * pt)
        texts.append((t, label, padded, _turned_outline(t, renderer, box, 0.35 * pt)))

    for i, (a, label_a, box_a, outline_a) in enumerate(texts):
        for b, label_b, box_b, outline_b in texts[i + 1 :]:
            if not box_a.overlaps(box_b):
                continue
            if outline_a is None and outline_b is None:
                problems.append(f"layout: '{label_a}' overlaps '{label_b}'")
            elif _outlines_overlap(
                _corners(box_a) if outline_a is None else outline_a,
                _corners(box_b) if outline_b is None else outline_b,
            ):
                problems.append(f"layout: '{label_a}' overlaps '{label_b}'")

    def check_width(what: str, lw: float) -> None:
        if lw > 0 and not (0.25 <= lw <= 2.0 + 1e-6):
            problems.append(f"marks: {what} line width {lw:g} pt")

    def check_colour(what: str, colours) -> None:
        for c in _hexes(colours):
            if not _is_token(c):
                problems.append(
                    f"colour: {what} uses {to_hex(c[:3])}, not a token or a tint of one"
                )
                return

    for ax in fig.axes:
        if not ax.get_visible():
            continue
        schematic = not ax.axison
        name = ax.get_label() or (ax.get_title(loc="left") or "axes")
        for line in ax.get_lines():
            if not line.get_visible():
                continue
            if line.get_linestyle() not in ("None", "", " ", "none"):
                check_width(f"{name} line", line.get_linewidth())
                check_colour(f"{name} line", [to_rgb(line.get_color())])
            if line.get_marker() not in (None, "None", "", " ", "none"):
                for c in (line.get_markerfacecolor(), line.get_markeredgecolor()):
                    if c not in ("none", "None") and c is not None:
                        check_colour(f"{name} marker", [to_rgb(c)])
        for patch in ax.patches:
            if not patch.get_visible() or not isinstance(patch, Patch):
                continue
            if patch.get_fill():
                check_colour(f"{name} patch", [patch.get_facecolor()])
            if patch.get_edgecolor()[3] > 0:
                check_width(f"{name} patch edge", patch.get_linewidth())
                check_colour(f"{name} patch edge", [patch.get_edgecolor()])
        for coll in ax.collections:
            if not coll.get_visible() or not isinstance(coll, Collection):
                continue
            if coll.get_array() is not None:
                if coll.get_cmap().name not in COLORMAPS:
                    problems.append(f"colour: {name} colormap {coll.get_cmap().name}")
            else:
                check_colour(f"{name} collection", coll.get_facecolors())
            edges = coll.get_edgecolors()
            widths = np.atleast_1d(coll.get_linewidths())
            if len(edges) and np.any(np.asarray(edges)[:, 3] > 0) and np.any(widths > 0):
                for lw in widths:
                    check_width(f"{name} collection edge", float(lw))
                if coll.get_array() is None:
                    check_colour(f"{name} collection edge", edges)
        for image in ax.images:
            if isinstance(image, AxesImage) and image.get_cmap().name not in COLORMAPS:
                problems.append(f"colour: {name} image colormap {image.get_cmap().name}")

        if schematic:
            continue
        # Data marks in display coordinates: marker vertices, scatter offsets, bar rectangles.
        points = []
        for line in ax.get_lines():
            if (
                line.get_visible()
                and line.get_marker() not in (None, "None", "", " ", "none")
                and len(line.get_xydata())
            ):
                points.append(line.get_transform().transform(line.get_xydata()))
        for coll in ax.collections:
            offsets = coll.get_offsets() if hasattr(coll, "get_offsets") else []
            if coll.get_visible() and len(offsets) and coll.get_array() is None:
                points.append(coll.get_offset_transform().transform(offsets))
        # Line paths, sampled every point along each segment, so labels cannot sit on a line.
        for line in ax.get_lines():
            if (
                not line.get_visible()
                or line.get_linestyle() in ("None", "", " ", "none")
                or len(line.get_xydata()) < 2
            ):
                continue
            xy = line.get_transform().transform(line.get_xydata())
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                n = max(2, int(np.hypot(x1 - x0, y1 - y0) / pt))
                points.append(np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)]))
        points = np.vstack(points) if points else np.empty((0, 2))
        bars = [
            p.get_window_extent(renderer)
            for p in ax.patches
            if isinstance(p, Rectangle)
            and p.get_visible()
            and p.get_fill()
            and not _near(p.get_facecolor(), WHITE)
            and p.get_width()
            and p.get_height()
        ]
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            box = legend.get_window_extent(renderer).padded(-pt)
            inside = (
                (points[:, 0] > box.x0)
                & (points[:, 0] < box.x1)
                & (points[:, 1] > box.y0)
                & (points[:, 1] < box.y1)
                if len(points)
                else []
            )
            if np.any(inside) or any(box.overlaps(b.padded(-0.5 * pt)) for b in bars):
                problems.append(f"layout: the legend of '{name}' covers data")
        for t in ax.texts:
            if (
                not t.get_visible()
                or not t.get_text().strip()
                or _near(to_rgb(t.get_color()), WHITE)
            ):
                continue
            box = t.get_window_extent(renderer).padded(-0.5 * pt)
            label = t.get_text().strip().replace("\n", " ")[:40]
            if len(points) and np.any(
                (points[:, 0] > box.x0)
                & (points[:, 0] < box.x1)
                & (points[:, 1] > box.y0)
                & (points[:, 1] < box.y1)
            ):
                problems.append(f"layout: '{label}' covers a marker or a line")
            elif any(box.overlaps(b) for b in bars):
                problems.append(f"layout: '{label}' covers a bar")
    return problems


def save(fig, stem: str, *, folder=None, allow: dict[str, str] | None = None) -> None:
    """Lint the figure, then write it as PDF (for LaTeX) and PNG (for review).

    allow maps a substring of a violation to the reason it is accepted; accepted violations are
    printed, any other one stops the save.
    """
    from pathlib import Path

    MANUSCRIPT = Path(__file__).resolve().parents[2]

    allow = allow or {}
    problems = lint(fig)
    accepted = [p for p in problems if any(key in p for key in allow)]
    refused = [p for p in problems if p not in accepted]
    for p in accepted:
        reason = next(allow[key] for key in allow if key in p)
        print(f"accepted ({reason}): {p}")
    if refused:
        raise FigureLintError(stem, refused)
    folder = folder or MANUSCRIPT / "publication" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = folder / f"{stem}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None)
        print(f"wrote {path.relative_to(MANUSCRIPT.parent)}")
