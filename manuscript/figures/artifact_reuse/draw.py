"""Draw the figure on reusing stored artifacts: one function per panel, and the panels assembled with their letters.

Each panel function draws one panel into the plot it is given and is named for what the panel shows. The name is also
the panel's SVG group ID and the name of its data file, <name>.json, which prepare.py writes.
CELLS gives each panel's printed size, and LAYOUT places the panels, which sets their letters. Run from the repository
root:

  uv run --locked python -B -m manuscript.figures.artifact_reuse.draw            # every panel and the figure
  uv run --locked python -B -m manuscript.figures.artifact_reuse.draw <panel>    # one panel

Renders go to .build/figures/artifact_reuse/: each panel alone and without a letter as panels/<panel>.png, .pdf and
.svg; the lettered figure as artifact_reuse.png, .pdf and .svg; and each panel's letter in letters.json, which every
run writes.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects, transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from scipy.ndimage import uniform_filter

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from manuscript.figures import glyphs as g  # noqa: E402
from manuscript.figures import style as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
BUILD = REPO / ".build" / "figures" / HERE.name
ALONE = {"canvas: width": "a panel alone is drawn at the size it prints at in the figure"}


# ---- shared helpers -----------------------------------------------------------------------------


def width_mm(ax) -> float:
    return ax.get_position().width * ax.figure.get_figwidth() / fs.MM


def height_mm(ax) -> float:
    return ax.get_position().height * ax.figure.get_figheight() / fs.MM


def plain(value, _=None) -> str:
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


def legend_above(
    ax, handles, labels, *, ncol: int, left_mm: float = 0.0, top_mm: float | None = None, **spacing
):
    """A legend above the plot, starting left_mm from the plot's left edge (negative: further left): its bottom
    1.8 mm above the plot or, given top_mm, its top that far above the plot."""
    if top_mm is None:
        loc, y = "lower left", 1.0 + 1.8 / height_mm(ax)
    else:
        loc, y = "upper left", 1.0 + top_mm / height_mm(ax)
    return ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=(left_mm / width_mm(ax), y),
        borderaxespad=0.0,
        ncol=ncol,
        **{
            "columnspacing": 0.9,
            "labelspacing": 0.25,
            "handlelength": 1.6,
            "handletextpad": 0.35,
            **spacing,
        },
    )


def log_axis(axis, ticks: list[float], labels: dict[float, str] | None = None) -> None:
    """A log axis labelled at ticks, with plain numbers unless labels names them, and no minor ticks."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    if labels is None:
        axis.set_major_formatter(FuncFormatter(plain))
    else:
        axis.set_major_formatter(FuncFormatter(lambda value, _: labels.get(value, "")))
    axis.set_minor_formatter(NullFormatter())


def category_ticks(ax, positions: list[float], names: list[str], *, labels: bool) -> None:
    """An x axis of named categories at fixed positions, with the names written only when labels is set."""
    ax.xaxis.set_major_locator(FixedLocator(positions))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda _, i: names[i] if labels else ""))
    ax.xaxis.set_minor_locator(NullLocator())


def sibling_axes(ax, left_mm: float, bottom_mm: float, w_mm: float, h_mm: float):
    """A second plot of the same panel, placed in millimetres from the lower-left corner of ax's plot."""
    pos = ax.get_position()
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    other = fs.add_axes(fig, pos.x0 * fw + left_mm, pos.y0 * fh + bottom_mm, w_mm, h_mm)
    other.set_gid(f"{ax.get_gid()}_{len(fig.axes)}")
    return other


EVEN = {"o": 1.0, "s": 0.9, "D": 0.8, "^": 1.25, "h": 1.05, "P": 1.05}


def marker_ms(marker: str) -> float:
    return fs.MARKER_MS * EVEN.get(marker, 1.0)


def method_point(ax, x, y, colour, marker, *, z=4):
    ax.plot(
        [x],
        [y],
        ls="none",
        marker=marker,
        ms=marker_ms(marker),
        mfc=colour,
        mec=fs.INK,
        mew=fs.LW_MARKER,
        zorder=z,
    )


def method_handle(colour: str, marker: str, *, primary: bool = False, ls="-") -> Line2D:
    return Line2D(
        [],
        [],
        color=colour,
        marker=marker,
        ms=marker_ms(marker),
        mfc=colour,
        mec=fs.INK,
        mew=fs.LW_MARKER,
        lw=fs.LW_PRIMARY if primary else fs.LW_DATA,
        ls=ls,
    )


# ---- outcome matrices: how each method's run ended at each size, under the plots that share its x axis -----

# The marks of figure 1's matrix: a muted dot for a completed run, and an ink mark for each way a run stopped.
OUTCOME_MARKS = {
    "pass": {
        "s": 5.5,
        "marker": "o",
        "facecolor": fs.LIGHT,
        "edgecolor": fs.SLATE,
        "linewidths": 0.4,
    },
    "oom": {"s": 14.0, "marker": "x", "color": fs.INK, "linewidths": 0.85},
    "time": {"s": 16.0, "marker": "+", "color": fs.INK, "linewidths": 0.9},
    "limit": {
        "s": 11.0,
        "marker": "v",
        "facecolor": fs.WHITE,
        "edgecolor": fs.INK,
        "linewidths": 0.65,
    },
}
OUTCOME_HANDLES = {
    "pass": {"marker": "o", "ms": 2.6, "mfc": fs.LIGHT, "mec": fs.SLATE, "mew": 0.4},
    "oom": {"marker": "x", "ms": 4.0, "color": fs.INK, "mew": 0.85},
    "time": {"marker": "+", "ms": 4.4, "color": fs.INK, "mew": 0.9},
    "limit": {"marker": "v", "ms": 3.5, "mfc": fs.WHITE, "mec": fs.INK, "mew": 0.65},
}
OUTCOME_NAMES = {
    "pass": "pass",
    "limit": "R index limit",
    "time": "out of time",
    "oom": "out of memory",
}
MATRIX_ROW_MM = 2.5
# The gap between a plot and the matrix under it, wide enough to keep the plot's ticks off the matrix's frame.
MATRIX_GAP_MM = 2.0
# The matrix's legend top sits this far below the matrix, under its x tick labels and axis label.
MATRIX_LEGEND_BELOW_MM = 7.0
# The left edge of the method legends above the time plots and of the outcome legends under the matrices, from the
# plot's left edge, so that each legend starts under the panel letter and stays inside the panel's cell.
LEGEND_LEFT_MM = -7.5


def outcome_matrix(
    ax,
    rows: list[tuple[str, list[tuple[float, str]]]],
    *,
    left_mm: float,
    legend_columns: int | None = None,
):
    """The outcome of each row's runs, one row per method from the top, directly below ax and on its x positions;
    returns the matrix's axes, with a legend of the outcomes it holds under it, starting left_mm from the plot's
    left edge, on one row or in legend_columns columns."""
    pos = ax.get_position()
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    n = len(rows)
    h = MATRIX_ROW_MM * n
    grid = fs.add_axes(fig, pos.x0 * fw, pos.y0 * fh - h - MATRIX_GAP_MM, pos.width * fw, h)
    grid.set_gid(f"{ax.get_gid()}_outcomes")
    grid.set_xlim(ax.get_xlim())
    grid.set_ylim(-0.55, n - 0.45)
    held = set()
    for i, (_, runs) in enumerate(rows):
        y = n - 1 - i
        if i % 2 == 1:
            grid.axhspan(y - 0.5, y + 0.5, facecolor=fs.PAPER, edgecolor="none", zorder=0)
        for x, state in runs:
            grid.scatter([x], [y], zorder=3, **OUTCOME_MARKS[state])
            held.add(state)
    grid.set_yticks(range(n))
    grid.set_yticklabels([name for name, _ in reversed(rows)], fontsize=fs.NOTE)
    grid.tick_params(axis="y", length=0, pad=1.5)
    fs.frame_axes(grid)
    states = [s for s in OUTCOME_NAMES if s in held]
    columns = legend_columns or len(states)
    rows = -(-len(states) // columns)
    # A legend fills its columns first; this order shows the outcomes row by row.
    states = [
        states[r * columns + c]
        for c in range(columns)
        for r in range(rows)
        if r * columns + c < len(states)
    ]
    grid.legend(
        [Line2D([], [], ls="none", **OUTCOME_HANDLES[s]) for s in states],
        [OUTCOME_NAMES[s] for s in states],
        loc="upper left",
        bbox_to_anchor=(left_mm / width_mm(grid), -MATRIX_LEGEND_BELOW_MM / height_mm(grid)),
        borderaxespad=0.0,
        ncol=columns,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.3,
    )
    return grid


# ---- pipeline runs as metro lines ----------------------------------------------------------------

RECORD = fs.MODULES["record"].strong
PARAMETER_FILL = fs.MODULES["record"].tint
STATION_R = 2.2
# A parameter's tag runs out of its artifact's circle: its width, of which the circle covers the first 1.8 mm.
TAG_W, TAG_H = 10.0, 3.6
# Every run's line, end and bar are ink: the runs are told apart by their names and where their lines go.
RUN_INK = fs.INK
METRO_LW = 1.0
# In millimetres: the distance between parallel run lines, between the clusterings of one graph, and between the
# branches of the neighbor counts.
METRO_STEP = 0.72
LEAF_STEP = 4.6
BRANCH_STEP = 11.5
# The notes under the card's artifacts sit this far under the artifacts' centres, to their baselines, and the arrows
# between the card's parts are this long.
CARD_NOTE_MM = 4.6
CARD_ARROW_MM = 4.0


def label(ax, x, y, s, *, ha="center", va="center_baseline", size=fs.TICK, colour=fs.INK, **kw):
    kw.setdefault("zorder", 8)
    return ax.text(x, y, s, fontsize=size, color=colour, ha=ha, va=va, **kw)


def note(ax, x, y, s, **kw):
    kw.setdefault("size", fs.NOTE)
    kw.setdefault("colour", fs.SLATE)
    return label(ax, x, y, s, **kw)


def station(ax, x, y, name, *, detail=False):
    """An artifact: a circle with its alias; detail rings the artifact whose provenance record the panel opens."""
    ax.add_patch(
        Circle((x, y), STATION_R, facecolor="white", edgecolor=fs.INK, lw=fs.LW_AXIS, zorder=5)
    )
    if detail:
        ax.add_patch(
            Circle(
                (x, y), STATION_R + 0.6, facecolor="none", edgecolor=RECORD, lw=fs.LW_THIN, zorder=4
            )
        )
    label(ax, x, y, name, va="center")


def parameter(ax, x, y, text, *, width=10.5):
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 2.0),
            width,
            4.0,
            boxstyle="round,pad=0,rounding_size=0.7",
            facecolor=PARAMETER_FILL,
            edgecolor=RECORD,
            lw=fs.LW_AXIS,
            zorder=5,
        )
    )
    label(ax, x, y, text, va="center")


def tagged_station(ax, x, y, name, text, *, detail=False):
    """An artifact and the parameter that made it, as one glyph: the parameter's tag runs out of the artifact's
    circle on the side its inputs come from."""
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.4, y - TAG_H / 2),
            TAG_W,
            TAG_H,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor=PARAMETER_FILL,
            edgecolor=RECORD,
            lw=fs.LW_AXIS,
            zorder=4.6,
        )
    )
    label(ax, x + (STATION_R + 0.4 + TAG_W) / 2, y, text, va="center")
    station(ax, x, y, name, detail=detail)


def bracket(ax, x0, x1, y, text) -> float:
    """A bracket under a row of artifacts, from x0 to x1, naming the runs that share them; returns its text's
    baseline."""
    ax.plot([x0, x0, x1, x1], [y, y - 0.8, y - 0.8, y], color=fs.SLATE, lw=fs.LW_THIN, zorder=3)
    baseline = y - 3.2
    note(ax, (x0 + x1) / 2, baseline, text, va="baseline")
    return baseline


def run_names(leaves: list[dict]) -> str:
    """The runs that share a bracket's artifacts, in letter order: one run, two, the first to the last of a series of
    consecutive letters, or a list."""
    names = sorted(leaf["run"] for leaf in leaves)
    if len(names) == 1:
        return f"Run {names[0]}"
    if len(names) == 2:
        return f"Runs {names[0]} and {names[1]}"
    if [ord(name) - ord(names[0]) for name in names] != list(range(len(names))):
        return f"Runs {', '.join(names[:-1])} and {names[-1]}"
    return f"Runs {names[0]} to {names[-1]}"


# The pipeline's steps, left to right; each runs on the output of the step to its right.
STEPS = (
    "Leiden",
    "Graph",
    "Neighbors",
    "ANN\nindex",
    "PCA",
    "Normalize\ncounts",
    "Variable\ngenes",
    "Filter\ncells",
)


# The last station of the shared chain sits this far from the panel's right edge, so that the pipeline's box around
# its chip ends level with the card, 0.5 mm from the edge.
CHAIN_END_MM = 4.85


def metro_columns(w: float, stations: int) -> dict:
    """The x positions: the shared chain's stations from the right edge, then the fork, the neighbors with their
    neighbor count, the graphs and the clusterings with their resolution to the left."""
    step = 14.5
    xs = tuple(w - CHAIN_END_MM - (stations - 1 - i) * step for i in range(stations))
    fork = xs[0] - 6.0
    # The neighbor count's tag ends 1.5 mm before its lines bend towards the shared chain.
    kx = fork - 5.0 - 1.5 - 0.4 - TAG_W
    gx = kx - 14.0
    lx = gx - 24.0
    return {"xs": xs, "fork": fork, "kx": kx, "gx": gx, "lx": lx}


def pipeline_strip(ax, c, y) -> float:
    """The pipeline: its steps as chips on one rail, each above the artifacts it writes, with an arrowhead in every
    gap pointing the way the steps run (right to left, from the cells to Leiden), all in a dotted box labelled on
    its top edge. Returns the box's lower edge."""
    xs_step = (c["lx"], c["gx"], c["kx"], *c["xs"])
    h, pad, inset = 6.0, 0.9, 1.1
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    edges = []
    for x, name in zip(xs_step, STEPS, strict=True):
        t = label(ax, x, y, name, va="center", linespacing=1.0, size=fs.NOTE, zorder=7)
        bb = t.get_window_extent(renderer)
        (a, _), (b, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        w = (b - a) + 2 * pad
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=1.2",
                facecolor=fs.PAPER,
                edgecolor=fs.LIGHT,
                lw=fs.LW_AXIS,
                zorder=6,
            )
        )
        edges.append((x - w / 2, x + w / 2))
    ax.plot([xs_step[0], xs_step[-1]], [y, y], color=fs.SLATE, lw=fs.LW_THIN, zorder=5)
    for (_, left_end), (right_start, _) in zip(edges[:-1], edges[1:]):
        ax.plot(
            [(left_end + right_start) / 2],
            [y],
            ls="none",
            color=fs.SLATE,
            marker="<",
            ms=3.2,
            mec="none",
            zorder=6,
        )
    x0, x1 = edges[0][0] - inset, edges[-1][1] + inset
    y0, y1 = y - h / 2 - inset, y + h / 2 + inset
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0,rounding_size=1.6",
            facecolor="none",
            edgecolor=fs.SLATE,
            lw=fs.LW_THIN,
            ls=(0, (1.0, 1.4)),
            zorder=4,
        )
    )
    t = label(ax, x0 + 3.0, y1, "Pipeline", ha="left", va="center", size=fs.NOTE, colour=fs.SLATE)
    t.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.8})  # breaks the dotted edge
    return y0


def provenance_card(ax, rx0, rx1, ry0, ry1, record: dict) -> None:
    """The provenance record of one artifact, in a card: its input artifacts, the operation and parameters that made
    it, and the artifact over the first characters of its random artifact ID in the store."""
    g.card(ax, rx0, ry0, rx1, ry1, module="record", z=0)
    # The row reaches from the ring around the artifact above it to the notes' descenders under it, in the card's
    # middle.
    row = (ry1 - (STATION_R + 0.6) + ry0 + CARD_NOTE_MM + 0.6) / 2
    edge, gap = 3.0, 0.6
    first = rx0 + edge + STATION_R
    second = first + 6.0
    # The artifact ID under the artifact is measured as drawn, so that it keeps a clear margin to the card's edge.
    digest = note(
        ax, 0.0, row - CARD_NOTE_MM, f"id: {record['digest'][:8]}…", colour=fs.INK, va="baseline"
    )
    half = (
        digest.get_window_extent(ax.figure.canvas.get_renderer()).width / ax.figure.dpi / fs.MM / 2
    )
    out = rx1 - max(edge + STATION_R, half + 1.2)
    digest.set_x(out)
    op0 = second + STATION_R + 2 * gap + CARD_ARROW_MM
    op1 = out - STATION_R - 2 * gap - CARD_ARROW_MM
    for x, item in zip((first, second), record["inputs"], strict=True):
        station(ax, x, row, item["alias"])
    note(ax, (first + second) / 2, row - CARD_NOTE_MM, "inputs", va="baseline")
    g.arrow(ax, (second + STATION_R + gap, row), (op0 - gap, row), lw=0.7, head=3.5, z=4)
    ax.add_patch(
        FancyBboxPatch(
            (op0, row - 2.4),
            op1 - op0,
            4.8,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor="white",
            edgecolor=fs.LIGHT,
            lw=fs.LW_AXIS,
            zorder=4,
        )
    )
    px = op1 - 6.2
    qx = (op0 + px - 5.0) / 2
    label(ax, qx, row, record["operation"].replace("_", " "), va="center")
    parameter(ax, px, row, f"k = {record['parameters']['k']}", width=10.0)
    note(ax, qx, row - CARD_NOTE_MM, "operation", va="baseline")
    note(ax, px, row - CARD_NOTE_MM, "parameters", va="baseline")
    g.arrow(ax, (op1 + gap, row), (out - STATION_R - gap, row), lw=0.7, head=3.5, z=4)
    station(ax, out, row, record["alias"], detail=True)


def offset_path(points, offsets):
    """A polyline shifted sideways by a per-segment offset (mm, to the left of travel), with mitred joins; where two
    segments in line take different offsets, the line steps across at their joint."""
    pts = np.asarray(points, float)
    lines = []
    for (p, q), d in zip(zip(pts[:-1], pts[1:]), offsets, strict=True):
        t = (q - p) / np.linalg.norm(q - p)
        n = np.array([-t[1], t[0]])
        lines.append((p + d * n, q + d * n, t))
    out = [lines[0][0]]
    for (a0, a1, ta), (b0, b1, tb) in zip(lines[:-1], lines[1:]):
        m = np.array([ta, -tb]).T
        if abs(np.linalg.det(m)) < 1e-9:
            out.append(a1)
            if not np.allclose(a1, b0):
                out.append(b0)
            continue
        u = np.linalg.solve(m, b0 - a0)
        out.append(a0 + u[0] * ta)
    out.append(lines[-1][1])
    return np.array(out)


def run_line(ax, points, offsets, colour) -> None:
    path = offset_path(points, offsets)
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=colour,
        lw=METRO_LW,
        solid_joinstyle="miter",
        solid_capstyle="butt",
        zorder=3,
    )


# In millimetres: the height of a run's bar, the length of the longest bar, and the gap between the bars' zero and the
# run names they grow out of.
# The runs' minutes: CyteArc's bars, as thick as a run line's name is tall at most, on an axis of runs at their zero
# whose ticks point at the names, RUN_BAR_GAP left of them.
RUN_BAR_H = 2.2
RUN_BAR_LONGEST_MM = 10.0
RUN_BAR_GAP = 1.6
RUN_TICK_MM = 0.7
RUN_BAR_COLOUR = fs.ENTITIES["cytearc"].colour
MAP_LEFT_MM = 4.0


def run_bars(ax, runs: list[dict], names: list) -> None:
    """Each run's minutes as a bar in CyteArc's blue, level with the run's name and growing leftwards from an axis of
    runs at zero, whose tick points at the name; the longest bar is RUN_BAR_LONGEST_MM long. Under the lowest bar, the
    axis of minutes grows the same way from the same corner, with a tick every ten minutes and its unit right of the
    zero."""
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    right = (
        min(inv.transform(name.get_window_extent(renderer).corners())[:, 0].min() for name in names)
        - RUN_BAR_GAP
    )
    minutes = [run["runSeconds"] / 60 for run in runs]
    scale = RUN_BAR_LONGEST_MM / max(minutes)
    for run, value in zip(runs, minutes, strict=True):
        ax.add_patch(
            Rectangle(
                (right - value * scale, run["y"] - RUN_BAR_H / 2),
                value * scale,
                RUN_BAR_H,
                zorder=3,
                **fs.bar_style(RUN_BAR_COLOUR),
            )
        )
    axis_y = min(run["y"] for run in runs) - RUN_BAR_H / 2 - 0.8
    top = max(run["y"] for run in runs) + RUN_BAR_H / 2 + 0.8
    # The two axes meet at zero minutes, below the lowest run: minutes leftwards, runs upwards.
    ax.plot(
        [right - max(minutes) * scale, right, right],
        [axis_y, axis_y, top],
        color=fs.INK,
        lw=fs.LW_AXIS,
        solid_joinstyle="miter",
        solid_capstyle="butt",
        zorder=3,
    )
    for run in runs:
        ax.plot(
            [right, right + RUN_TICK_MM],
            [run["y"], run["y"]],
            color=fs.INK,
            lw=fs.LW_AXIS,
            solid_capstyle="butt",
            zorder=3,
        )
    baseline = axis_y - 3.0
    for tick in np.arange(0.0, max(minutes), 10.0):
        x = right - tick * scale
        ax.plot([x, x], [axis_y, axis_y - 0.7], color=fs.INK, lw=fs.LW_AXIS, zorder=3)
        label(ax, x, baseline, f"{tick:g}", va="baseline")
    label(ax, right + 1.3, baseline, "minutes", ha="left", va="baseline", size=fs.LABEL)


def provenance_runs(ax, data: dict) -> None:
    """The published analysis of a mounted ten-million-cell embryo atlas, and the clusterings a later job made from
    its stored artifacts, as pipeline runs: each run an ink line from the filtered cells to its Leiden clustering,
    named at its end, the runs in letter order from the top, and the minutes its job spent a bar growing
    leftwards out of its name. The later runs share the published run's artifacts up to its graph (other resolutions)
    or its neighbor index (another neighbor count), running side by side through them. Each artifact made with a
    parameter carries it in a tag, and a bracket names the runs that share the artifacts above it. The card under the
    shared chain opens the provenance record of one artifact, from below the chain's bracket down to the lowest
    bracket's baseline."""
    w, h = width_mm(ax), height_mm(ax)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_axis_off()
    c = metro_columns(w, len(data["chain"]))
    xs, fork, kx, gx, lx = (c[k] for k in ("xs", "fork", "kx", "gx", "lx"))
    strip_bottom = pipeline_strip(ax, c, h - 6.2)
    branches = data["branches"]
    runs = [leaf for branch in branches for leaf in branch["leaves"]]
    if sum(leaf["published"] for leaf in runs) != 1:
        raise ValueError("the runs must hold one published run")
    s = METRO_STEP
    # On the shared chain the runs run side by side, centred on the artifacts, in their order from the top.
    trunk = {leaf["run"]: ((len(runs) - 1) / 2 - i) * s for i, leaf in enumerate(runs)}
    # Each branch's clusterings fan out from its graph, centred on it.
    fans = [
        [(len(branch["leaves"]) - 1) / 2 - i for i in range(len(branch["leaves"]))]
        for branch in branches
    ]
    # The highest clustering's tag clears the pipeline by 1.5 mm.
    yc = strip_bottom - 1.5 - TAG_H / 2 - max(fans[0]) * LEAF_STEP
    start, bend = lx - 4.4, lx + TAG_W + 2.4
    names = []
    for bi, (branch, fan) in enumerate(zip(branches, fans, strict=True)):
        gy = yc - bi * BRANCH_STEP
        branch["y"] = gy
        for leaf, place in zip(branch["leaves"], fan, strict=True):
            ly = gy + place * LEAF_STEP
            leaf["y"] = ly
            if bi == 0:
                # The first branch continues the shared chain, its runs keeping their places on it.
                pts = [(start, ly), (bend, ly), (gx, gy), (xs[-1], yc)]
                offsets = [0.0, 0.0, trunk[leaf["run"]]]
            else:
                pts = [
                    (start, ly),
                    (bend, ly),
                    (gx, gy),
                    (fork - 5.0, gy),
                    (fork, yc),
                    (xs[-1], yc),
                ]
                offsets = [0.0, 0.0, place * s, trunk[leaf["run"]], trunk[leaf["run"]]]
            run_line(ax, pts, offsets, RUN_INK)
            # The run's end of the line, as a metro line's terminus, with the run's name.
            ax.plot(
                [start, start],
                [ly - 0.9, ly + 0.9],
                color=RUN_INK,
                lw=fs.LW_PRIMARY,
                solid_capstyle="butt",
                zorder=3,
            )
            names.append(label(ax, start - 1.2, ly, f"Run {leaf['run']}", ha="right", va="center"))
    run_bars(ax, runs, names)
    for x, artifact in zip(reversed(xs), data["chain"], strict=True):
        station(ax, x, yc, artifact["alias"])
    shared = bracket(ax, xs[0] - 2.2, xs[-1] + 2.2, yc - 3.6, f"shared by all {data['runs']} runs")
    lowest = shared
    for branch in branches:
        gy = branch["y"]
        tagged_station(
            ax,
            kx,
            gy,
            branch["neighbors"]["alias"],
            f"k = {branch['k']}",
            detail=branch["neighbors"]["alias"] == data["record"]["alias"],
        )
        station(ax, gx, gy, branch["graph"]["alias"])
        lowest = bracket(ax, gx - 2.2, kx + 0.4 + TAG_W, gy - 3.4, run_names(branch["leaves"]))
        for leaf in branch["leaves"]:
            tagged_station(ax, lx, leaf["y"], leaf["alias"], f"r = {leaf['resolution']:g}")
    if lowest < 1.0:
        raise ValueError(f"the panel is {h:.1f} mm tall; its runs need {h - lowest + 1.0:.1f} mm")
    # The card's top clears the descenders of the shared chain's bracket text, and its bottom is level with the lowest
    # bracket's baseline. A dotted arrow runs from the card's left edge to the underside of the artifact's tag.
    rx0 = xs[0] - 2.2
    provenance_card(ax, rx0, w - 0.5, lowest, shared - 1.3, data["record"])
    record_y = next(
        branch["y"]
        for branch in branches
        if branch["neighbors"]["alias"] == data["record"]["alias"]
    )
    ax.add_patch(
        FancyArrowPatch(
            (rx0, record_y - 2.4),
            (kx + TAG_W - 0.6, record_y - TAG_H / 2 - 0.3),
            arrowstyle="-|>",
            mutation_scale=5.5,
            color=RECORD,
            lw=fs.LW_AXIS,
            ls=(0, (1.2, 1.2)),
            connectionstyle="arc3,rad=-0.2",
            zorder=6,
        )
    )


# ---- the blood reference ladder -----------------------------------------------------------------


@dataclass(frozen=True)
class Method:
    name: str
    entity: str
    marker: str


METHODS = {
    "unintegrated": Method("CyteArc", "cytearc", "o"),
    "seurat_map": Method("Seurat", "seurat", "^"),
    "symphony": Method("Symphony", "symphony", "h"),
    "celltypist_hvg": Method("CellTypist", "celltypist", "P"),
}
# The order the mapping panels list the methods in, in legends, outcome matrices, rows and side by side within each
# group of the disagreement panels: the comparators, then CyteArc.
METHOD_ORDER = ("seurat_map", "symphony", "celltypist_hvg", "unintegrated")
if sorted(METHOD_ORDER) != sorted(METHODS):
    raise ValueError("METHOD_ORDER must list each method of METHODS once")
LADDER_X = {
    "reference_500000": 0.0,
    "reference_1500000": 1.0,
    "reference_3000000": 2.0,
    "reference_all": 3.0,
}


# The ladder column's y labels start this far left of its plots, where the reference's map above them starts.
LADDER_YLABEL_LEFT_MM = 8.5
# The CITE-seq plots are short, so their y labels take two lines, start further left to clear the tick labels and sit
# below the plots' middles to clear the letters above them.
CITE_YLABEL_LEFT_MM = 11.5
CITE_YLABEL_LOW_MM = 1.2


def ladder_ylabel(
    ax, text: str, *, low_mm: float = 0.0, left_mm: float = LADDER_YLABEL_LEFT_MM
) -> None:
    """The y label of a plot of a stacked column, the ladder's, the in-place or the CITE-seq series', its left edge
    left_mm left of the plot, the same in every plot of the column whatever its tick labels' width; centred on the
    plot, or low_mm below its middle."""
    ax.set_ylabel(text, labelpad=1.5)
    width = (
        ax.yaxis.label.get_window_extent(ax.figure.canvas.get_renderer()).width
        / ax.figure.dpi
        / fs.MM
    )
    ax.yaxis.set_label_coords((width - left_mm) / width_mm(ax), 0.5 - low_mm / height_mm(ax))


def ladder_ticks(ax, sizes: list[dict], *, labels: bool) -> None:
    ax.set_xlim(-0.5, 3.5)
    category_ticks(
        ax,
        [LADDER_X[s["part"]] for s in sizes],
        [f"{s['cells'] / 1e6:.1f}" for s in sizes],
        labels=labels,
    )


def method_line(ax, key: str, xs, ys, *, z=3):
    method = METHODS[key]
    colour = fs.ENTITIES[method.entity].colour
    if len(xs) > 1:
        ax.plot(
            xs,
            ys,
            color=colour,
            lw=fs.LW_PRIMARY if method.entity == "cytearc" else fs.LW_DATA,
            zorder=z,
        )
    for x, y in zip(xs, ys):
        method_point(ax, x, y, colour, method.marker, z=z + 1)


# Cells left out of the named groups, drawn in a light grey and named on no map.
UNNAMED = fs.tint(fs.SLATE, 0.45)
# The reference's cell types in groups: the five commonest types in GROUPS hues, named on the map, and the rest.
TYPE_GROUPS = (
    ("T cells", ("T cells",), fs.GROUPS[3]),
    ("monocytes", ("Monocytes",), fs.GROUPS[0]),
    ("NK and ILC", ("NK and ILC",), fs.GROUPS[4]),
    ("B cells", ("B cells",), fs.GROUPS[2]),
    ("dendritic cells", ("Dendritic cells",), fs.GROUPS[1]),
    (None, ("Plasma cells", "Platelets", "Progenitors", "Erythroid"), UNNAMED),
)
# A group's name sits where the group is densest among the bins it dominates, smoothed over this many bins.
LABEL_SMOOTHING_BINS = 9
# The clear margin between two names on a map, in points, and the wider one around a text the names avoid, such as a
# map's title, so that no name reads as part of it.
LABEL_MARGIN_PT = 2.5
AVOID_MARGIN_PT = 4.0


def type_map(ax, raster: str, groups) -> np.ndarray:
    """A UMAP binned by cell type, each bin taking the group of one of its cells drawn at random and fading toward white
    where few cells fall, on its whole square grid; the groups name every type of the raster. Returns the cells of each
    group per bin."""
    with np.load(REPO / raster, allow_pickle=False) as binned:
        counts = binned["typeCells"].astype(float)
        types = [str(t) for t in binned["types"]]
    members = [t for _, group, _ in groups for t in group]
    if set(types) - set(members) or len(members) != len(set(members)):
        raise ValueError(
            f"the groups must name each of {sorted(types)} once, they hold {sorted(members)}"
        )
    grouped = np.stack(
        [
            sum((counts[types.index(t)] for t in group if t in types), np.zeros(counts.shape[1:]))
            for _, group, _ in groups
        ]
    )
    image = fs.category_raster(grouped, [colour for *_, colour in groups], pad=None)
    ax.imshow(
        image, origin="lower", cmap=fs.SEQUENTIAL, interpolation="nearest", aspect="auto", zorder=2
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return grouped


def map_labels(ax, grouped: np.ndarray, groups, *, avoid=()) -> None:
    """Each named group's name on the map, in ink with a white halo, on its own cells: from the bin where the group is
    densest among the bins it dominates, the name takes the nearby place, on growing rings around that bin, where it
    covers most of the group's bins without covering another name, a text of avoid or leaving the map. The largest
    groups are placed first, as their names fit inside them. A group whose name finds no such place on any of its own
    bins goes unnamed on that map, rather than named over other cells or empty space."""
    smooth = np.stack(
        [uniform_filter(layer, size=LABEL_SMOOTHING_BINS, mode="constant") for layer in grouped]
    )
    total = smooth.sum(axis=0)
    side = grouped.shape[1]
    reach = 0.3 * side
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer).padded(-2.0)
    to_data = ax.transData.inverted()
    named = sorted(
        (
            (float(cells.sum()), name, layer)
            for (name, _, _), cells, layer in zip(groups, grouped, smooth)
            if name is not None
        ),
        key=lambda entry: -entry[0],
    )
    placed = [
        t.get_window_extent(renderer).padded(AVOID_MARGIN_PT * ax.figure.dpi / 72) for t in avoid
    ]
    for _, name, layer in named:
        share = np.divide(layer, total, out=np.zeros_like(layer), where=total > 0)
        row, column = np.unravel_index(np.argmax(layer * share), layer.shape)
        # The group's own bins, summed over any box in constant time.
        own = np.zeros((side + 1, side + 1))
        own[1:, 1:] = ((share > 0.5) & (layer > 0)).cumsum(axis=0).cumsum(axis=1)
        text = ax.text(
            float(column),
            float(row),
            name,
            ha="center",
            va="center",
            fontsize=fs.NOTE,
            color=fs.INK,
            linespacing=1.0,
            zorder=4,
            path_effects=[patheffects.withStroke(linewidth=1.8, foreground=fs.WHITE)],
        )
        best, best_score, best_covered = (float(column), float(row)), -np.inf, 0.0
        candidates = [(float(column), float(row))] + [
            (column + radius * np.cos(angle), row + radius * np.sin(angle))
            for radius in np.linspace(0.02, 0.3, 15) * side
            for angle in np.radians(np.arange(0, 360, 22.5))
        ]
        for x, y in candidates:
            text.set_position((x, y))
            box = text.get_window_extent(renderer)
            if not (
                box.x0 >= frame.x0
                and box.x1 <= frame.x1
                and box.y0 >= frame.y0
                and box.y1 <= frame.y1
            ) or any(box.overlaps(other) for other in placed):
                continue
            (x0, y0), (x1, y1) = to_data.transform(box.get_points())
            c0, c1 = (
                int(np.clip(np.floor(x0 + 0.5), 0, side)),
                int(np.clip(np.ceil(x1 + 0.5), 0, side)),
            )
            r0, r1 = (
                int(np.clip(np.floor(y0 + 0.5), 0, side)),
                int(np.clip(np.ceil(y1 + 0.5), 0, side)),
            )
            area = max((c1 - c0) * (r1 - r0), 1)
            covered = own[r1, c1] - own[r0, c1] - own[r1, c0] + own[r0, c0]
            score = covered / area - 0.5 * np.hypot(x - column, y - row) / reach
            if score > best_score:
                best, best_score, best_covered = (x, y), score, covered
        if best_covered == 0:
            text.remove()
            continue
        text.set_position(best)
        # Names keep a clear margin between them, so that no two read as one.
        placed.append(text.get_window_extent(renderer).padded(LABEL_MARGIN_PT * ax.figure.dpi / 72))


def cell_count(cells: int) -> str:
    """A cell count for a title: millions with two decimals, or thousands."""
    return f"{cells / 1e6:.2f}M" if cells >= 1e6 else f"{cells / 1e3:.0f}K"


def ladder_reference_umap(ax, data: dict) -> None:
    """The ladder reference's cells in the UMAP of CyteArc's Harmony analysis of all ten blood studies, coloured by
    type group, with each named group's name on the map; the map spans the ladder's plots below it with their y
    labels, from the labels' left edge to the plots' right edge."""
    map_labels(ax, type_map(ax, data["raster"], TYPE_GROUPS), TYPE_GROUPS)


def ladder_time(ax, data: dict) -> None:
    """Time to build the reference and annotate the query, in minutes on a log10 scale, against reference size; the
    matrix under the ladder's disagreement plot gives how each run ended. The legend names the ladder's methods, in
    two rows of two."""
    ax.set_yscale("log")
    ax.set_ylim(2, 200)
    log_axis(ax.yaxis, [3, 10, 30, 100])
    ladder_ticks(ax, data["sizes"], labels=False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for key in METHODS:
        runs = data["methods"][key]
        method_line(ax, key, [LADDER_X[r["part"]] for r in runs], [r["seconds"] / 60 for r in runs])
    ladder_ylabel(ax, "minutes (log10)")
    # Every run maps the same query, which neither the axes nor the reference map name.
    plot_name(ax, f"{cell_count(data['queryCells'])} query cells", corner="lower right")
    # The legend fills its columns first, so the methods are handed over so that its rows read in METHOD_ORDER.
    order = [METHOD_ORDER[i] for i in (0, 2, 1, 3)]
    legend_above(
        ax,
        [
            method_handle(fs.ENTITIES[m.entity].colour, m.marker, primary=m.entity == "cytearc")
            for m in (METHODS[key] for key in order)
        ],
        [METHODS[key].name for key in order],
        ncol=2,
        left_mm=LEGEND_LEFT_MM,
        top_mm=LADDER_LEGEND_TOP_MM,
        columnspacing=1.3,
        handlelength=1.9,
    )


# The ladder's legend top sits this far above the ladder's time plot, level with the panel letter.
LADDER_LEGEND_TOP_MM = 6.8


def ladder_memory(ax, data: dict) -> None:
    """Each method's container peak against reference size on a log2 scale: each run built its reference before
    annotating the query, and its peak covers both, as its time does. R's sparse matrices cannot index the two larger
    references, which stopped Seurat and Symphony, as the matrix under the ladder's disagreement plot shows."""
    ax.set_yscale("log", base=2)
    ax.set_ylim(6, 330)
    log_axis(ax.yaxis, [8, 16, 32, 64, 128, 256])
    ladder_ticks(ax, data["sizes"], labels=False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for key in METHODS:
        runs = [r for r in data["methods"][key] if r["outcome"] == "pass"]
        method_line(ax, key, [LADDER_X[r["part"]] for r in runs], [r["peakGiB"] for r in runs])
    ladder_ylabel(ax, "peak (GiB, log2)")


# ---- disagreement with the author labels -------------------------------------------------------------------------

# A value is written this far beyond its point's marker or interval, whichever reaches further.
VALUE_GAP_MM = 0.6
VALUE_WIDTH_MM = g.text_width("00.0", fs.NOTE)
# The largest marker's half height, which a value must clear.
MARKER_HALF_MM = max(marker_ms(m.marker) for m in METHODS.values()) * 25.4 / 72 / 2


def marker_half_mm(marker: str) -> float:
    return marker_ms(marker) * 25.4 / 72 / 2


# The methods sit side by side at each size of the disagreement plot, this far apart in x units, so that their
# intervals stay apart.
LADDER_DODGE = 0.24


def ladder_accuracy(ax, data: dict) -> None:
    """Disagreement of the transferred labels with the author labels of the ladder's query at each reference size,
    100 minus agreement, on an axis from zero, each point on its donor-bootstrap 95% interval: one line per method as in
    the time and memory plots above, the methods side by side at each size in METHOD_ORDER, CyteArc's drawn last so
    that it stays on top. Under it, the matrix of how each method's run ended at each size closes the ladder's
    column."""
    highest = 100 * max(r["high"] for runs in data["methods"].values() for r in runs)
    top = 5 * np.ceil(1.05 * highest / 5)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(FixedLocator(list(np.arange(0, top + 1e-9, 5))))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ladder_ticks(ax, data["sizes"], labels=False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for i, key in enumerate(METHOD_ORDER):
        runs = data["methods"][key]
        shift = (i - (len(METHOD_ORDER) - 1) / 2) * LADDER_DODGE
        xs = [LADDER_X[r["part"]] + shift for r in runs]
        colour = fs.ENTITIES[METHODS[key].entity].colour
        for x, r in zip(xs, runs, strict=True):
            ax.plot(
                [x, x],
                [100 * r["low"], 100 * r["high"]],
                color=colour,
                lw=fs.LW_THIN,
                solid_capstyle="butt",
                zorder=2.5,
            )
        method_line(ax, key, xs, [100 * r["disagreement"] for r in runs])
    ladder_ylabel(ax, "disagreement (%)")
    plot_name(ax, "vs author labels", corner="upper right")
    matrix = outcome_matrix(
        ax,
        [
            (
                METHODS[key].name,
                [(LADDER_X[r["part"]], r["outcome"]) for r in data["outcomes"][key]],
            )
            for key in METHOD_ORDER
        ],
        left_mm=LEGEND_LEFT_MM,
    )
    ladder_ticks(matrix, data["sizes"], labels=True)
    matrix.set_xlabel("reference cells (M)", labelpad=1.5)


# ---- the held-out kidney and lung studies -----------------------------------------------------------

# Each held-out reference's harmonized labels in groups, as its UMAP colours them: the kidney's nephron segments, vessels
# and immune cells, with the rest unnamed; the lung's compartments, as the lung atlas groups its finer labels.
TISSUE_GROUPS = {
    "kidney": (
        ("proximal\ntubule", ("Proximal tubule",), fs.GROUPS[0]),
        (
            "loop of\nHenle",
            ("Thin descending limb", "Thin ascending limb", "Thick ascending limb"),
            fs.GROUPS[1],
        ),
        (
            "distal and\ncollecting",
            (
                "Distal convoluted tubule",
                "Connecting tubule",
                "Principal cells",
                "Intercalated cells",
            ),
            fs.GROUPS[2],
        ),
        ("endothelial", ("Endothelial cells",), fs.GROUPS[5]),
        ("immune", ("B lineage", "T and NK cells", "Myeloid cells"), fs.GROUPS[3]),
        (
            None,
            (
                "Podocytes",
                "Parietal epithelium",
                "Stromal cells",
                "Neural cells",
                "Urothelial cells",
            ),
            UNNAMED,
        ),
    ),
    "lung": (
        ("alveolar", ("AT1", "AT2"), fs.GROUPS[0]),
        (
            "airway",
            ("Basal", "Multiciliated lineage", "Secretory", "Submucosal Secretory", "Rare"),
            fs.GROUPS[2],
        ),
        (
            "endothelial",
            (
                "EC arterial",
                "EC capillary",
                "EC venous",
                "Lymphatic EC differentiating",
                "Lymphatic EC mature",
                "Lymphatic EC proliferating",
            ),
            fs.GROUPS[5],
        ),
        (
            "stromal",
            (
                "Fibroblasts",
                "Myofibroblasts",
                "Smooth muscle FAM83D+",
                "SM activated stress response",
            ),
            fs.GROUPS[1],
        ),
        ("myeloid", ("Macrophages", "Monocytes", "Dendritic cells", "Mast cells"), fs.GROUPS[4]),
        ("lymphoid", ("B cell lineage", "T cell lineage", "Innate lymphoid cell NK"), fs.GROUPS[3]),
    ),
}
# The disagreement strip beside each held-out map: its gap from the map, which holds its axis, and its width, in mm.
STRIP_GAP_MM = 8.0
STRIP_W_MM = 12.0

# The held-out panels' methods in the ladder's colours and order.
HELD_OUT_METHODS = {
    "seurat_map": METHODS["seurat_map"],
    "symphony": METHODS["symphony"],
    "celltypist": METHODS["celltypist_hvg"],
    "unintegrated": METHODS["unintegrated"],
}
if list(HELD_OUT_METHODS.values()) != [METHODS[key] for key in METHOD_ORDER]:
    raise ValueError("HELD_OUT_METHODS must list its methods in METHOD_ORDER")


def disagreement_strip(ax, data: dict) -> None:
    """Disagreement with the author labels on the held-out query, one method beside the next in the ladder's order and
    markers, as points on donor-bootstrap 95% intervals with their values written above, on an axis from zero shared
    by the kidney and lung strips; the query's study and size under it."""
    if sorted(data["methods"]) != sorted(HELD_OUT_METHODS):
        raise ValueError(
            f"the strip draws {sorted(HELD_OUT_METHODS)}, its data holds {sorted(data['methods'])}"
        )
    h = height_mm(ax)
    high = 100 * data["scaleHigh"]
    ax.set_ylim(0, high * h / (h - (MARKER_HALF_MM + VALUE_GAP_MM + VALUE_WIDTH_MM + 0.3)))
    ax.yaxis.set_major_locator(
        FixedLocator(list(np.arange(0, high + 1e-9, 5 if high <= 20 else 10)))
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(-0.6, len(HELD_OUT_METHODS) - 0.4)
    ax.set_xticks([])
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    per_mm = ax.get_ylim()[1] / h
    above = transforms.offset_copy(
        ax.transData, fig=ax.figure, y=VALUE_GAP_MM * fs.MM, units="inches"
    )
    for x, (key, method) in enumerate(HELD_OUT_METHODS.items()):
        colour = fs.ENTITIES[method.entity].colour
        run = data["methods"][key]
        y = 100 * run["disagreement"]
        ax.plot(
            [x, x],
            [100 * run["low"], 100 * run["high"]],
            color=colour,
            lw=fs.LW_THIN,
            solid_capstyle="butt",
            zorder=3,
        )
        method_point(ax, x, y, colour, method.marker)
        ax.text(
            x,
            max(100 * run["high"], y + marker_half_mm(method.marker) * per_mm),
            f"{y:.1f}",
            transform=above,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=fs.NOTE,
            color=fs.INK,
        )
    ax.set_ylabel("disagreement (%)", labelpad=1.5)
    query = data["query"]
    ax.set_xlabel(
        f"{query['study']}\n{cell_count(query['cells'])} cells", labelpad=1.5, fontsize=fs.NOTE
    )


def held_out(ax, data: dict) -> None:
    """The held-out tissue's reference UMAP, coloured by label group with each named group on the map, and beside it
    the strip of each method's disagreement on the held-out query."""
    groups = TISSUE_GROUPS[data["tissue"]]
    map_labels(ax, type_map(ax, data["umap"]["raster"], groups), groups)
    strip = sibling_axes(ax, width_mm(ax) + STRIP_GAP_MM, 0.0, STRIP_W_MM, height_mm(ax))
    disagreement_strip(strip, data)
    fs.frame_axes(strip)


def held_out_kidney(ax, data: dict) -> None:
    """The held-out kidney study (Reck), mapped onto a kidney reference of two other datasets."""
    held_out(ax, data)


def held_out_lung(ax, data: dict) -> None:
    """The held-out lung study (Madissoon), mapped onto the rest of the lung atlas's core."""
    held_out(ax, data)


# ---- the in-place column: annotating blood stores from the stored reference, job by job --------------------------

CYTEARC_BLUE = fs.ENTITIES["cytearc"].colour
# The jobs the column's plots space evenly. Their labels under the column run on one line, tilted, and shorten the two
# Sound Life jobs' names.
IN_PLACE_JOBS = 8
IN_PLACE_LABEL_ANGLE = 45
IN_PLACE_SHORT_NAMES = {"Sound Life, CMV+": "SL CMV+", "Sound Life, CMV−": "SL CMV−"}
# How far below its plot's middle the last plot's y label is centred: the label is longer than the plot, and this
# keeps it 2 mm clear of the memory plot's label above.
IN_PLACE_LAST_LABEL_LOW_MM = 3.7
# The memory plot's circles have areas in proportion to the cells per store; a circle of the other plots' size stands
# for this many.
CELLS_PER_PLAIN_CIRCLE = 1e6


def job_label(job: dict) -> str:
    """A job's short name and the cells it mapped, in millions, on one line."""
    return f"{IN_PLACE_SHORT_NAMES.get(job['name'], job['name'])}, {job['cells'] / 1e6:.1f}M"


def store_axes(ax, jobs: list[dict], *, top: float, step: float, names: bool) -> list[int]:
    """The column's axes: the jobs from the fewest cells up along x, labelled under the last plot only, and an axis
    from zero with its ticks every step; returns the jobs' positions."""
    if len(jobs) != IN_PLACE_JOBS:
        raise ValueError(
            f"the in-place column is sized for {IN_PLACE_JOBS} jobs, its data holds {len(jobs)}"
        )
    xs = list(range(len(jobs)))
    ax.set_xlim(-0.5, len(jobs) - 0.5)
    ax.set_xticks(xs)
    if names:
        ax.set_xticklabels(
            [job_label(job) for job in jobs],
            rotation=IN_PLACE_LABEL_ANGLE,
            ha="right",
            va="top",
            rotation_mode="anchor",
            fontsize=fs.NOTE,
        )
    else:
        ax.set_xticklabels([])
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(FixedLocator(list(np.arange(0, top + 1e-9, step))))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    return xs


def job_points(ax, xs: list[int], per_job: list[list[float]], *, areas=None) -> None:
    """A point per job, at the mean over its stores, on a bar over their range where it has several; no line joins
    the points, since the jobs are not a scale. areas, where given, scale each point's area from the plain size."""
    for i, (x, values) in enumerate(zip(xs, per_job, strict=True)):
        if len(values) > 1:
            ax.plot(
                [x, x],
                [min(values), max(values)],
                color=CYTEARC_BLUE,
                lw=fs.LW_THIN,
                solid_capstyle="butt",
                zorder=2.5,
            )
        scale = 1.0 if areas is None else float(np.sqrt(areas[i]))
        ax.plot(
            [x],
            [float(np.mean(values))],
            ls="none",
            marker="o",
            ms=marker_ms("o") * scale,
            mfc=CYTEARC_BLUE,
            mec=fs.INK,
            mew=fs.LW_MARKER,
            zorder=4,
        )


def store_annotation_time(ax, data: dict) -> None:
    """Minutes to annotate each store from the stored reference, opening it included, job by job from the fewest cells
    up, on an axis from zero. A job opened the reference once, and each of its stores is charged the whole opening."""
    jobs = data["jobs"]
    minutes = [[(job["openingSeconds"] + s) / 60 for s in job["mappingSeconds"]] for job in jobs]
    highest = max(max(values) for values in minutes)
    xs = store_axes(ax, jobs, top=float(5 * np.ceil(1.05 * highest / 5)), step=10.0, names=False)
    job_points(ax, xs, minutes)
    ladder_ylabel(ax, "minutes")


def store_annotation_memory(ax, data: dict) -> None:
    """Each store's container peak while it was mapped with label transfer onto the stored reference, in GiB, job by
    job from the fewest cells up, on an axis from zero; each point's area in proportion to the cells per store, the
    mean over a job's stores where it has several."""
    jobs = data["jobs"]
    highest = max(max(job["peaksGiB"]) for job in jobs)
    xs = store_axes(ax, jobs, top=float(4 * np.ceil(1.05 * highest / 4)), step=4.0, names=False)
    job_points(
        ax,
        xs,
        [job["peaksGiB"] for job in jobs],
        areas=[job["cells"] / len(job["peaksGiB"]) / CELLS_PER_PLAIN_CIRCLE for job in jobs],
    )
    ladder_ylabel(ax, "peak (GiB)")


def store_disagreement(ax, data: dict) -> None:
    """Disagreement of transferred with author labels as a bar per job, from the fewest cells up, on an axis from
    zero, with each job's short name and the cells it mapped under it, which closes the column. Its y label, longer
    than the plot, is centred IN_PLACE_LAST_LABEL_LOW_MM low, clear of the memory plot's label."""
    jobs = data["jobs"]
    highest = max(100 * j["disagreement"] for j in jobs)
    xs = store_axes(ax, jobs, top=float(5 * np.ceil(1.05 * highest / 5)), step=5.0, names=True)
    ax.bar(
        xs,
        [100 * j["disagreement"] for j in jobs],
        width=fs.BAR,
        zorder=2,
        **fs.bar_style(CYTEARC_BLUE),
    )
    ladder_ylabel(ax, "disagreement (%)", low_mm=IN_PLACE_LAST_LABEL_LOW_MM)


# ---- joint RNA and protein graphs ------------------------------------------------------

CITE_X = {100000: 0.0, 300000: 1.0, 600000: 2.0, 1000000: 3.0, 1457527: 4.0}
CITE_TICKS = ("0.1", "0.3", "0.6", "1.0", "1.46")
# The graphs in the order the CITE-seq plots list them, in the legend and the outcome matrix, as the mapping panels
# list their methods: Seurat's, then CyteArc's, which is drawn last so that it stays on top.
WNN = {
    "seurat": ("Seurat WNN", fs.ENTITIES["seurat"].colour, "^"),
    "cytearc": ("CyteArc WNN", fs.ENTITIES["cytearc"].colour, "o"),
}


# Inside a CITE-seq plot, a short name of what it measures sits this far from the corner it takes.
PLOT_NAME_INSET_MM = 1.0


def plot_name(ax, text: str, *, corner: str, drop_mm: float = 0.0) -> None:
    """A short name of what the plot measures in small type inside its empty corner, "upper right" or "lower right",
    with a white halo over the grid; drop_mm lowers a name in the upper right, away from a point near the corner."""
    upper = corner == "upper right"
    if corner not in ("upper right", "lower right"):
        raise ValueError(f"corner must be upper right or lower right, not {corner!r}")
    ax.text(
        1 - PLOT_NAME_INSET_MM / width_mm(ax),
        1 - (PLOT_NAME_INSET_MM + drop_mm) / height_mm(ax)
        if upper
        else PLOT_NAME_INSET_MM / height_mm(ax),
        text,
        transform=ax.transAxes,
        ha="right",
        va="top" if upper else "bottom",
        fontsize=fs.NOTE,
        color=fs.INK,
        zorder=5,
        path_effects=[patheffects.withStroke(linewidth=1.8, foreground=fs.WHITE)],
    )


# The minutes of the first analysis and of re-clustering share one log10 axis, so that the two plots read together.
CITE_MINUTES = (0.05, 600.0)
CITE_MINUTE_TICKS = [0.1, 1, 10, 100]
# On the CITE-seq column's short plots, labels at every other tick, so that no two touch.
CITE_MINUTE_LABELS = {0.1: "0.1", 10: "10"}


def cite_ticks(ax, *, labels: bool) -> None:
    ax.set_xlim(-0.5, len(CITE_X) - 0.5)
    category_ticks(ax, list(CITE_X.values()), list(CITE_TICKS), labels=labels)


def wnn_line(ax, tool: str, xs, ys, *, ranges=None) -> None:
    """A graph's line with a point at each size; ranges, where given, are drawn as bars behind the points."""
    _, colour, marker = WNN[tool]
    if ranges is not None:
        for x, (low, high) in zip(xs, ranges, strict=True):
            ax.plot(
                [x, x], [low, high], color=colour, lw=fs.LW_THIN, solid_capstyle="butt", zorder=2.5
            )
    ax.plot(xs, ys, color=colour, lw=fs.LW_PRIMARY if tool == "cytearc" else fs.LW_DATA, zorder=3)
    for x, y in zip(xs, ys):
        method_point(ax, x, y, colour, marker)


def multimodal_time(ax, data: dict) -> None:
    """Time of the first joint RNA and protein analysis on 100,000 to 1,457,527 CITE-seq cells, in minutes on a log10
    scale, for Seurat where its first analysis completed and for CyteArc's WNN workflow; the matrix under the CITE-seq
    agreement plot gives how each run ended. The legend names the graphs of the CITE-seq column, one to a row."""
    sizes = data["sizes"]
    ax.set_yscale("log")
    ax.set_ylim(*CITE_MINUTES)
    log_axis(ax.yaxis, CITE_MINUTE_TICKS, CITE_MINUTE_LABELS)
    cite_ticks(ax, labels=False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    done = [s for s in sizes if s["seurat"]["outcome"] == "pass"]
    wnn_line(
        ax,
        "seurat",
        [CITE_X[s["cells"]] for s in done],
        [s["seurat"]["initialSeconds"] / 60 for s in done],
    )
    wnn_line(
        ax,
        "cytearc",
        [CITE_X[s["cells"]] for s in sizes],
        [s["cytearc"]["initialSeconds"] / 60 for s in sizes],
    )
    ladder_ylabel(ax, "minutes\n(log10)", left_mm=CITE_YLABEL_LEFT_MM, low_mm=CITE_YLABEL_LOW_MM)
    plot_name(ax, "first analysis", corner="lower right")
    legend_above(
        ax,
        [
            method_handle(colour, marker, primary=tool == "cytearc")
            for tool, (_, colour, marker) in WNN.items()
        ],
        [name for name, _, _ in WNN.values()],
        ncol=2,
        left_mm=CITE_LEGEND_LEFT_MM,
        top_mm=CITE_LEGEND_MM - 0.5,
        columnspacing=0.8,
        handlelength=1.3,
        handletextpad=0.3,
    )


def multimodal_reclustering(ax, data: dict) -> None:
    """Time to cluster each tool's stored WNN graph again at a new resolution, on the first analysis's minutes axis:
    CyteArc from its stored artifacts at every size, and Seurat from its stored graph where its first analysis made one;
    the matrix under the CITE-seq agreement plot gives how each run ended."""
    sizes = data["sizes"]
    ax.set_yscale("log")
    ax.set_ylim(*CITE_MINUTES)
    log_axis(ax.yaxis, CITE_MINUTE_TICKS, CITE_MINUTE_LABELS)
    cite_ticks(ax, labels=False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    stored = [s for s in sizes if s["seurat"]["storedSeconds"] is not None]
    wnn_line(
        ax,
        "seurat",
        [CITE_X[s["cells"]] for s in stored],
        [s["seurat"]["storedSeconds"] / 60 for s in stored],
    )
    wnn_line(
        ax,
        "cytearc",
        [CITE_X[s["cells"]] for s in sizes],
        [s["cytearc"]["storedSeconds"] / 60 for s in sizes],
    )
    ladder_ylabel(ax, "minutes\n(log10)", left_mm=CITE_YLABEL_LEFT_MM, low_mm=CITE_YLABEL_LOW_MM)
    # Lowered into the empty right, between Seurat's last point and CyteArc's line.
    plot_name(ax, "re-clustering", corner="upper right", drop_mm=2.5)


def multimodal_memory(ax, data: dict) -> None:
    """The container peak of each completed first analysis on a log2 scale (CyteArc's over its first analysis and
    both revisions); the matrix under the CITE-seq agreement plot gives how each run ended."""
    sizes = data["sizes"]
    ax.set_yscale("log", base=2)
    ax.set_ylim(6, 200)
    log_axis(ax.yaxis, [8, 16, 32, 64, 128], {8: "8", 32: "32", 128: "128"})
    cite_ticks(ax, labels=False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    done = [s for s in sizes if s["seurat"]["outcome"] == "pass"]
    wnn_line(
        ax, "seurat", [CITE_X[s["cells"]] for s in done], [s["seurat"]["peakGiB"] for s in done]
    )
    wnn_line(
        ax, "cytearc", [CITE_X[s["cells"]] for s in sizes], [s["cytearc"]["peakGiB"] for s in sizes]
    )
    ladder_ylabel(ax, "GiB\n(log2)", left_mm=CITE_YLABEL_LEFT_MM, low_mm=CITE_YLABEL_LOW_MM)
    # Lowered under Seurat's last point.
    plot_name(ax, "whole-run peak", corner="upper right", drop_mm=2.8)


def cluster_agreement(ax, data: dict) -> None:
    """ARI with the author labels of each WNN graph's partition cut to the authors' number of cell types, on an axis
    from zero: one line per graph through the mean over its Leiden seeds, with their range as a bar behind each point.
    Under it, the matrix of how each tool's run ended at each size closes the CITE-seq column."""
    sizes = data["sizes"]
    highest = max(run["ari"] for s in sizes for runs in s["graphs"].values() for run in runs)
    top = 0.1 * np.ceil(1.1 * highest / 0.1)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(FixedLocator(list(np.arange(0, top + 1e-9, 0.1))))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: ("0" if v < 1e-9 else f"{v:.1f}") if round(v * 10) % 2 == 0 else ""
        )
    )
    cite_ticks(ax, labels=False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for tool in WNN:
        made = [s for s in sizes if f"{tool}_wnn" in s["graphs"]]
        aris = [[run["ari"] for run in s["graphs"][f"{tool}_wnn"]] for s in made]
        wnn_line(
            ax,
            tool,
            [CITE_X[s["cells"]] for s in made],
            [float(np.mean(found)) for found in aris],
            ranges=[(min(found), max(found)) for found in aris],
        )
    # The plot's name says what the index measures; a longer label would run into the memory plot's above it.
    ladder_ylabel(ax, "ARI", left_mm=CITE_YLABEL_LEFT_MM, low_mm=CITE_YLABEL_LOW_MM)
    plot_name(ax, "vs author labels", corner="lower right")
    # Both rows are WNN graphs, so the matrix names only the tools.
    matrix = outcome_matrix(
        ax,
        [
            (
                fs.ENTITIES["seurat"].label,
                [(CITE_X[s["cells"]], s["seurat"]["outcome"]) for s in sizes],
            ),
            (fs.ENTITIES["cytearc"].label, [(CITE_X[s["cells"]], "pass") for s in sizes]),
        ],
        left_mm=CITE_LEGEND_LEFT_MM,
        legend_columns=2,
    )
    cite_ticks(matrix, labels=True)
    matrix.set_xlabel("CITE-seq cells (M)", labelpad=1.5)


# ---- the CITE-seq graphs as maps, and the weights WNN gives each modality ------------------------------------------

# The atlas's author labels in groups: the commonest lineages in GROUPS hues, named on the maps as on the blood map,
# with CD4 and CD8 T cells apart, as the protein panel tells them apart, and the rest in grey; the empty label marks
# cells that no rule of the atlas labels.
CITE_TYPE_GROUPS = (
    ("CD4 T", ("CD4 Naive", "CD4 Memory", "Treg"), fs.GROUPS[3]),
    ("CD8 T", ("CD8 Naive", "CD8 Memory"), fs.GROUPS[5]),
    ("NK", ("NK", "NK_CD56bright"), fs.GROUPS[4]),
    ("B cells", ("B naive", "B memory"), fs.GROUPS[2]),
    ("monocytes", ("CD14 Mono", "CD16 Mono"), fs.GROUPS[0]),
    ("DCs", ("cDC", "pDC"), fs.GROUPS[1]),
    (
        None,
        ("MAIT", "gdT", "Proliferating lymphocyte", "Plasmablast", "Platelet", "HSPC", ""),
        UNNAMED,
    ),
)
CITE_GRAPHS = {"RNA": "RNA", "ADT": "ADT", "WNN": "WNN"}
GRAPH_MAP_GAP_MM = 1.5


def graph_umaps(ax, data: dict) -> None:
    """The CITE-seq cells in the UMAP of each graph, top to bottom the RNA, protein and WNN graphs, each binned by author
    label group with the group names on the map and the graph named in its corner. Each modality's map starts from
    k-means of its own coordinates, and the WNN map from the RNA one."""
    side = width_mm(ax)
    maps = [ax] + [
        sibling_axes(ax, 0.0, -i * (side + GRAPH_MAP_GAP_MM), side, side)
        for i in range(1, len(CITE_GRAPHS))
    ]
    for axis, (graph, title) in zip(maps, CITE_GRAPHS.items(), strict=True):
        grouped = type_map(axis, data["rasters"][graph], CITE_TYPE_GROUPS)
        corner = axis.text(
            0.04,
            0.96,
            title,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=fs.TICK,
            color=fs.INK,
            zorder=5,
            path_effects=[patheffects.withStroke(linewidth=1.8, foreground=fs.WHITE)],
        )
        map_labels(axis, grouped, CITE_TYPE_GROUPS, avoid=[corner])
        if axis is not ax:
            fs.frame_axes(axis)


# The COMBAT T-cell subtypes in one order, with the short name and the colour each takes in every T-cell panel: CD4 and
# CD8 naive cells take the CD4 T and CD8 T hues of the CITE-seq maps.
T_CELL_SUBTYPES = {
    "CD4 naive": ("CD4 naive", fs.GROUPS[3]),
    "CD4 central memory": ("CD4 memory", fs.GROUPS[2]),
    "Treg": ("Treg", fs.GROUPS[1]),
    "CD8 naive": ("CD8 naive", fs.GROUPS[5]),
    "CD8 effector": ("CD8 effector", fs.GROUPS[0]),
    "MAIT": ("MAIT", fs.GROUPS[4]),
    "Vδ2 γδ T": ("γδ T", fs.SOURCE_BLUE),
}
# The subtypes' names on the map, on two lines where one line is wider than the subtype's cells.
T_CELL_MAP_NAMES = {"CD8 effector": "CD8\neffector"}
T_CELL_GROUPS = (
    *(
        (T_CELL_MAP_NAMES.get(short, short), (name,), colour)
        for name, (short, colour) in T_CELL_SUBTYPES.items()
    ),
    (None, ("unlabelled",), UNNAMED),
)
# The ADT heatmaps' title, from the study each shows.
T_CELL_ADT_TITLE = "{} ADT"
# The share axis of the reference comparison.
SHARE_TICKS = [0, 50, 100]


def t_cell_map(ax, data: dict) -> None:
    """COMBAT's T cells in the UMAP of their WNN graph, binned by named subtype and each subtype named on the map; the
    cells the reference leaves unlabelled in grey."""
    grouped = type_map(ax, data["raster"], T_CELL_GROUPS)
    map_labels(ax, grouped, T_CELL_GROUPS)


def share_rows(ax, names: list[str]) -> np.ndarray:
    """An axis of shares from 0 to 100% with one named row per entry from the top; returns the rows' heights."""
    ys = np.arange(len(names))[::-1]
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.xaxis.set_major_locator(FixedLocator(SHARE_TICKS))
    ax.yaxis.set_major_locator(FixedLocator(list(ys)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: names[len(names) - 1 - int(round(v))]))
    ax.tick_params(axis="y", length=0, pad=1.5, labelsize=fs.NOTE)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    return ys


def t_cell_reference_comparison(ax, data: dict) -> None:
    """For each subtype both references name, the share of Hao's T cells of that author label that receive it by RNA
    alone: from COMBAT's WNN clusters, CyteArc's marker filled, and from its RNA clusters, drawn open, joined per
    subtype; the legend at the right of the header row."""
    colour = fs.ENTITIES["cytearc"].colour
    rows = data["rows"]
    names = [short for short, _ in T_CELL_SUBTYPES.values()]
    if [row["subtype"] for row in rows] != names:
        raise ValueError(
            f"the comparison's rows {[row['subtype'] for row in rows]} must be the subtypes {names} in order"
        )
    ys = share_rows(ax, names)
    for y, row in zip(ys, rows, strict=True):
        ax.plot(
            [100 * row["rna"], 100 * row["wnn"]], [y, y], color=fs.SLATE, lw=fs.LW_THIN, zorder=2
        )
        ax.plot(
            [100 * row["rna"]],
            [y],
            ls="none",
            marker="o",
            ms=marker_ms("o"),
            mfc=fs.WHITE,
            mec=colour,
            mew=fs.LW_DATA,
            zorder=3,
        )
        method_point(ax, 100 * row["wnn"], y, colour, "o")
    ax.set_xlabel("agreement (%)", labelpad=1.5)
    open_handle = Line2D(
        [], [], ls="none", marker="o", ms=marker_ms("o"), mfc=fs.WHITE, mec=colour, mew=fs.LW_DATA
    )
    # The legend's top is level with the header's, its right edge with the plot's.
    ax.legend(
        [method_handle(colour, "o", primary=True, ls="none"), open_handle],
        ["WNN", "RNA"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0 + (CITE_HEADER_MM - 1.0) / height_mm(ax)),
        borderaxespad=0.0,
        ncol=2,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def protein_heatmap(ax, medians: list[list[float]], markers: list[str]):
    """One study's heatmap: each subtype's median per-cell CLR of each protein's ADT, as standard deviations from the
    subtypes' mean for that protein, blue below and vermilion above, clipped at two, with the proteins named under it.
    Returns the image."""
    values = np.asarray(medians, dtype=float)
    spread = values.std(axis=0)
    if values.shape != (len(T_CELL_SUBTYPES), len(markers)) or not np.all(spread > 0):
        raise ValueError(
            f"a heatmap needs each of {len(markers)} proteins for each of {len(T_CELL_SUBTYPES)} subtypes, varying "
            f"between them; it got {values.shape} values, spread {spread.round(3).tolist()}"
        )
    image = ax.imshow(
        (values - values.mean(axis=0)) / spread,
        cmap=fs.DIVERGING,
        vmin=-2.0,
        vmax=2.0,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(markers)), markers, rotation=90)
    ax.set_yticks(range(len(values)))
    ax.tick_params(length=0, pad=1.5, labelsize=fs.NOTE)
    fs.heatmap_cells(ax, values.shape)
    return image


def t_cell_proteins(ax, data: dict) -> None:
    """Each subtype's defining surface proteins, as ADT: left in Hao's T cells, the query, grouped by the subtype each
    received by RNA alone, so that the mapping never read their ADT, under the comparison of Hao's labels; right in
    COMBAT's T cells, the reference whose subtypes these are, T_CELL_HEATMAP_OFFSET_MM right of Hao's and under the map
    of COMBAT's T cells. The heatmaps share the subtypes' names at the left and the colour key between them, and each is
    titled with its study where the headers of its column start."""
    if data["subtypes"] != list(T_CELL_SUBTYPES):
        raise ValueError(
            f"the heatmaps' rows {data['subtypes']} must be the subtypes {list(T_CELL_SUBTYPES)} in order"
        )
    reference_ax = sibling_axes(ax, T_CELL_HEATMAP_OFFSET_MM, 0.0, width_mm(ax), height_mm(ax))
    title_left = (
        HEADER_INDENT_MM - T_CELL_ROW_NAMES_MM,
        MAP_COLUMN_LEFT
        + HEADER_INDENT_MM
        - (CITE_LEFT + CITE_PLOT_LEFT + T_CELL_HEATMAP_OFFSET_MM),
    )
    for axis, role, left_mm in zip(
        (ax, reference_ax), ("query", "reference"), title_left, strict=True
    ):
        image = protein_heatmap(axis, data[role]["medianClr"], data["markers"])
        axis.text(
            left_mm / width_mm(axis),
            1.0 + (T_CELL_HEADER_MM - HEADER_BASELINE_MM) / height_mm(axis),
            T_CELL_ADT_TITLE.format(data[role]["study"]),
            transform=axis.transAxes,
            fontsize=fs.LABEL,
            fontweight="bold",
            va="baseline",
            ha="left",
            color=fs.INK,
            gid="card-title",
        )
    ax.set_yticklabels([T_CELL_SUBTYPES[s][0] for s in data["subtypes"]])
    reference_ax.tick_params(axis="y", labelleft=False)
    # The colour key: a slim bar as tall as the heatmaps, with its tick labels centred in the gap between them.
    gap = T_CELL_HEATMAP_OFFSET_MM - width_mm(ax)
    key = ax.inset_axes(
        [
            1 + (gap - T_CELL_KEY_MM) / 2 / width_mm(ax),
            0.0,
            T_CELL_KEY_BAR_MM / width_mm(ax),
            1.0,
        ]
    )
    ax.figure.colorbar(image, cax=key, ticks=[-2, 0, 2])
    key.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: {-2: "\u22122", 2: "+2"}.get(round(v), "0"))
    )
    key.tick_params(labelsize=fs.NOTE, length=1.2, pad=0.8, width=fs.LW_AXIS)
    fs.frame_axes(key)


PANELS = {
    draw.__name__: draw
    for draw in (
        provenance_runs,
        ladder_reference_umap,
        ladder_time,
        ladder_memory,
        ladder_accuracy,
        held_out_kidney,
        held_out_lung,
        store_annotation_time,
        store_annotation_memory,
        store_disagreement,
        multimodal_time,
        multimodal_reclustering,
        multimodal_memory,
        cluster_agreement,
        t_cell_map,
        t_cell_proteins,
        graph_umaps,
        t_cell_reference_comparison,
    )
}
DATA_FILE: dict[str, str] = {}

# Short headers beside panel letters, for the panels whose subject their axes and legend do not name; the schematic
# carries its own titles.
HEADERS = {
    "ladder_reference_umap": lambda data: (
        f"Reference studies, {cell_count(data['rasterCells'])} cells"
    ),
    "held_out_kidney": lambda data: f"Kidney, {cell_count(data['reference']['cells'])} cells",
    "held_out_lung": lambda data: f"Lung, {cell_count(data['reference']['cells'])} cells",
    # The column is too narrow for the word cells.
    "graph_umaps": lambda data: f"CITE-seq, {cell_count(data['cells'])}",
    "t_cell_map": lambda data: f"COMBAT, {cell_count(data['cells'])}",
    "t_cell_reference_comparison": lambda data: f"Hao, {cell_count(data['overall']['haoCells'])}",
    "store_annotation_time": lambda data: (
        f"{sum(len(job['mappingSeconds']) for job in data['jobs'])} blood stores, "
        f"{cell_count(sum(job['cells'] for job in data['jobs']))} cells"
    ),
}
HEADER_INDENT_MM = 4.8
# A header's baseline sits this far under the top edge of its panel's cell.
HEADER_BASELINE_MM = 3.1


# ---- layout and rendering -----------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """A panel's printed size in millimetres, and its plot inside it as left, bottom, width and height from
    the cell's lower-left corner. The top-left corner stays free for the letter the figure adds."""

    width: float
    height: float
    plot: tuple[float, float, float, float]


# The left column is as wide as the ladder's plots reach, with 1 mm to spare, and the middle column starts there.
LEFT_COLUMN_W = 40.5
# The stacked plots start this far right of their cells' left edges, leaving room for their tick and y labels.
STACK_PLOT_LEFT = 12.5
# The plots of the stacked columns: the ladder's, and the in-place column's, which end level with lung's disagreement
# strip.
LADDER_PLOT_W = 27.0
HELD_OUT_PLOT = (2.0, 6.5, 25.0, 25.0)
IN_PLACE_PLOT_W = HELD_OUT_PLOT[0] + HELD_OUT_PLOT[2] + STRIP_GAP_MM + STRIP_W_MM - STACK_PLOT_LEFT
# The in-place column's plots, one height for all three. Above the first plot, room for the column's title; each
# cell holds half the gap between its plot and the next, the ladder's 3 mm; under the last plot, room for the jobs'
# tilted labels.
IN_PLACE_PLOT_H = 10.4
IN_PLACE_TITLE_MM = 4.0
IN_PLACE_HALF_GAP_MM = 1.5
IN_PLACE_LABELS_MM = 16.3
IN_PLACE_CELL_W = STACK_PLOT_LEFT + IN_PLACE_PLOT_W + 2.0
IN_PLACE_PLOT = (STACK_PLOT_LEFT, IN_PLACE_HALF_GAP_MM, IN_PLACE_PLOT_W, IN_PLACE_PLOT_H)
# The CITE-seq column starts where the middle column's widest cells, the in-place column's, end, and the map column
# where the CITE-seq cells end. The maps are as wide as kidney's and lung's, and panel a's card and pipeline end level
# with them.
CITE_LEFT = LEFT_COLUMN_W + IN_PLACE_CELL_W
# The CITE-seq column is as wide as a stacked column whose plots are as much wider than the ladder's as they have more
# sizes, less a tenth. Its plots start T_CELL_ROW_NAMES_MM into it, so that the names of the T-cell subtypes under the
# column fit left of plots that start where its plots start, and they end 1 mm short of its right edge.
CITE_CELL_W = STACK_PLOT_LEFT + 0.9 * LADDER_PLOT_W * len(CITE_X) / len(LADDER_X) + 1.0
T_CELL_ROW_NAMES_MM = 13.5
CITE_PLOT_LEFT = T_CELL_ROW_NAMES_MM
CITE_PLOT_W = CITE_CELL_W - CITE_PLOT_LEFT - 1.0
# The CITE-seq legends start where a stacked column's legends start, under the panel letters.
CITE_LEGEND_LEFT_MM = LEGEND_LEFT_MM - (CITE_PLOT_LEFT - STACK_PLOT_LEFT)
MAP_COLUMN_LEFT = CITE_LEFT + CITE_CELL_W
MAP_COLUMN_W = 160.0 - MAP_COLUMN_LEFT
# The CITE-seq maps are smaller than kidney's and lung's, and they end 1.5 mm short of the figure's edge.
GRAPH_MAP_W = 22.0
MAP_RIGHT_MM = 24.5
GRAPH_MAP_LEFT = MAP_RIGHT_MM - GRAPH_MAP_W
RIGHT_EDGE_MM = MAP_COLUMN_LEFT + MAP_RIGHT_MM
# The height of a header row above a panel's plot or map.
CITE_HEADER_MM = 5.5
# The map column: its header, then the three maps 1.5 mm apart.
GRAPH_UMAPS_H = CITE_HEADER_MM + 3 * GRAPH_MAP_W + 2 * GRAPH_MAP_GAP_MM + 1.5
# The CITE-seq plots end level with the maps: k holds its one-row legend above its plot and n its matrix and outcome
# legend below, each cell holds half the 3 mm gap to the next plot, and the plots share the height left.
CITE_LEGEND_MM = 4.5
CITE_MATRIX_MM = 21.6
CITE_PLOT_H = (GRAPH_UMAPS_H - CITE_LEGEND_MM - 9.0 - CITE_MATRIX_MM) / 4
# Under k to o, down to the figure's bottom edge, the T-cell panels in two rows. Above, the comparison of references on
# Hao's T cells continues the CITE-seq column with a plot as wide as its plots, and the map of COMBAT's T cells
# continues the map column at the size of its maps, the map's top level with the comparison's plot. Below, across both
# columns, the subtypes' ADT in Hao and in COMBAT: Hao's heatmap under the comparison's plot, starting
# where it starts, and COMBAT's under the map, as wide as the maps. The comparison's rows are T_CELL_ROW_MM apart; the
# heatmaps' are T_CELL_PROTEIN_ROW_MM apart, under a header row and over the proteins' names.
T_CELL_LEFT = CITE_LEFT
T_CELL_W = 160.0 - T_CELL_LEFT
T_CELL_BLOCK_H = 132.5 - GRAPH_UMAPS_H
T_CELL_ROW_MM = 2.2
T_CELL_PROTEIN_ROW_MM = 2.0
T_CELL_HEADER_MM = 4.6
T_CELL_PROTEIN_LABELS_MM = 9.3
T_CELL_BOTTOM_H = (
    T_CELL_HEADER_MM + len(T_CELL_SUBTYPES) * T_CELL_PROTEIN_ROW_MM + T_CELL_PROTEIN_LABELS_MM
)
T_CELL_TOP_H = T_CELL_BLOCK_H - T_CELL_BOTTOM_H
# The map's and the comparison's plot's top edge, a header row under the top of their cells.
T_CELL_PLOT_TOP_MM = T_CELL_TOP_H - CITE_HEADER_MM
# How far right of Hao's heatmap COMBAT's starts, level with the maps.
T_CELL_HEATMAP_OFFSET_MM = MAP_COLUMN_LEFT + GRAPH_MAP_LEFT - (CITE_LEFT + CITE_PLOT_LEFT)
# The heatmaps' colour key: a bar T_CELL_KEY_BAR_MM wide that with its tick labels takes T_CELL_KEY_MM.
T_CELL_KEY_BAR_MM = 1.2
T_CELL_KEY_MM = 4.5
CITE_COLUMN = (
    "multimodal_time",
    "multimodal_reclustering",
    "multimodal_memory",
    "cluster_agreement",
)
CELLS = {
    "provenance_runs": Cell(160.0, 37.5, (0.0, 0.0, RIGHT_EDGE_MM + 0.5, 37.5)),
    "ladder_reference_umap": Cell(LEFT_COLUMN_W, 42.0, (MAP_LEFT_MM, 1.5, 35.5, 35.5)),
    "ladder_time": Cell(LEFT_COLUMN_W, 26.0, (STACK_PLOT_LEFT, 1.5, LADDER_PLOT_W, 17.0)),
    "ladder_memory": Cell(LEFT_COLUMN_W, 20.0, (STACK_PLOT_LEFT, 1.5, LADDER_PLOT_W, 17.0)),
    "ladder_accuracy": Cell(LEFT_COLUMN_W, 44.5, (STACK_PLOT_LEFT, 26.0, LADDER_PLOT_W, 17.0)),
    "held_out_kidney": Cell(48.5, 36.5, HELD_OUT_PLOT),
    "held_out_lung": Cell(48.5, 36.5, HELD_OUT_PLOT),
    "store_annotation_time": Cell(
        IN_PLACE_CELL_W, IN_PLACE_TITLE_MM + IN_PLACE_PLOT_H + IN_PLACE_HALF_GAP_MM, IN_PLACE_PLOT
    ),
    "store_annotation_memory": Cell(
        IN_PLACE_CELL_W, IN_PLACE_PLOT_H + 2 * IN_PLACE_HALF_GAP_MM, IN_PLACE_PLOT
    ),
    "store_disagreement": Cell(
        IN_PLACE_CELL_W,
        IN_PLACE_LABELS_MM + IN_PLACE_PLOT_H + IN_PLACE_HALF_GAP_MM,
        (STACK_PLOT_LEFT, IN_PLACE_LABELS_MM, IN_PLACE_PLOT_W, IN_PLACE_PLOT_H),
    ),
    "multimodal_time": Cell(
        CITE_CELL_W,
        CITE_LEGEND_MM + CITE_PLOT_H + 1.5,
        (CITE_PLOT_LEFT, 1.5, CITE_PLOT_W, CITE_PLOT_H),
    ),
    "multimodal_reclustering": Cell(
        CITE_CELL_W, CITE_PLOT_H + 3.0, (CITE_PLOT_LEFT, 1.5, CITE_PLOT_W, CITE_PLOT_H)
    ),
    "multimodal_memory": Cell(
        CITE_CELL_W, CITE_PLOT_H + 3.0, (CITE_PLOT_LEFT, 1.5, CITE_PLOT_W, CITE_PLOT_H)
    ),
    "cluster_agreement": Cell(
        CITE_CELL_W,
        CITE_MATRIX_MM + CITE_PLOT_H + 1.5,
        (CITE_PLOT_LEFT, CITE_MATRIX_MM, CITE_PLOT_W, CITE_PLOT_H),
    ),
    "graph_umaps": Cell(
        MAP_COLUMN_W,
        GRAPH_UMAPS_H,
        (GRAPH_MAP_LEFT, 1.5 + 2 * (GRAPH_MAP_W + GRAPH_MAP_GAP_MM), GRAPH_MAP_W, GRAPH_MAP_W),
    ),
    "t_cell_map": Cell(
        MAP_COLUMN_W,
        T_CELL_TOP_H,
        (GRAPH_MAP_LEFT, T_CELL_PLOT_TOP_MM - GRAPH_MAP_W, GRAPH_MAP_W, GRAPH_MAP_W),
    ),
    "t_cell_reference_comparison": Cell(
        CITE_CELL_W,
        T_CELL_TOP_H,
        (
            CITE_PLOT_LEFT,
            T_CELL_PLOT_TOP_MM - len(T_CELL_SUBTYPES) * T_CELL_ROW_MM,
            CITE_PLOT_W,
            len(T_CELL_SUBTYPES) * T_CELL_ROW_MM,
        ),
    ),
    "t_cell_proteins": Cell(
        T_CELL_W,
        T_CELL_BOTTOM_H,
        (
            T_CELL_ROW_NAMES_MM,
            T_CELL_PROTEIN_LABELS_MM,
            GRAPH_MAP_W,
            len(T_CELL_SUBTYPES) * T_CELL_PROTEIN_ROW_MM,
        ),
    ),
}
# Where each panel's cell sits: its lower-left corner, in millimetres from the figure's lower-left corner. The pipeline
# runs span the top, ending level with the CITE-seq maps. Under them, the left column holds the blood reference
# ladder: the reference's UMAP over the ladder's time, memory and disagreement, 3 mm apart, all spanning one x range
# and ending in the matrix of how each run ended. Beside it, the held-out kidney study over the lung study, each a
# UMAP of its reference with its methods' disagreement beside it, the kidney map's top level with the blood map's and
# lung tucked under kidney, beside kidney's strip labels. Under lung, the in-place column, built as the ladder's and
# ending level with lung's strip: the minutes, memory and disagreement of annotating the blood stores from the stored
# reference, job by job. Beside the middle column, level with the maps, the CITE-seq column, built as the ladder's: the
# first analysis's minutes, re-clustering's minutes on the same axis, memory and agreement, ending in its own matrix.
# At the right, the RNA, protein and WNN graphs' maps of the largest CITE-seq run. Under k to o, the T-cell panels:
# under o, the map of COMBAT's T cells by WNN subtype, which takes the letter after o's; under n, the comparison of WNN
# and RNA references on Hao's T cells; and across the bottom, the subtypes' ADT in Hao and in COMBAT. The
# panels take their letters in this order.
LAYOUT = {
    "provenance_runs": (0.0, 132.5),
    "ladder_reference_umap": (0.0, 90.5),
    "ladder_time": (0.0, 64.5),
    "ladder_memory": (0.0, 44.5),
    "ladder_accuracy": (0.0, 0.0),
    "held_out_kidney": (LEFT_COLUMN_W, 96.0),
    "held_out_lung": (LEFT_COLUMN_W, 63.0),
    "store_annotation_time": (LEFT_COLUMN_W, 47.1),
    "store_annotation_memory": (LEFT_COLUMN_W, 33.7),
    "store_disagreement": (LEFT_COLUMN_W, 5.5),
    # The CITE-seq column stacks down from the maps' top edge.
    **{
        name: (CITE_LEFT, 132.5 - sum(CELLS[n].height for n in CITE_COLUMN[: i + 1]))
        for i, name in enumerate(CITE_COLUMN)
    },
    "graph_umaps": (MAP_COLUMN_LEFT, 132.5 - GRAPH_UMAPS_H),
    "t_cell_map": (MAP_COLUMN_LEFT, T_CELL_BOTTOM_H),
    "t_cell_reference_comparison": (CITE_LEFT, T_CELL_BOTTOM_H),
    "t_cell_proteins": (T_CELL_LEFT, 0.0),
}
# Cells that may overlap, with why their contents do not: lung's letter, title and plots sit under kidney's map and
# left of kidney's strip labels, which fill only the right of kidney's lower margin.
OVERLAPPING_CELLS = {
    frozenset(
        {"held_out_kidney", "held_out_lung"}
    ): "lung tucks under kidney's map, left of its strip labels"
}


def check_layout() -> None:
    if sorted(LAYOUT) != sorted(PANELS) or sorted(CELLS) != sorted(PANELS):
        raise ValueError(
            f"LAYOUT places {sorted(LAYOUT)} and CELLS sizes {sorted(CELLS)}; both must hold each panel of {sorted(PANELS)} once"
        )
    boxes = {}
    for name, (left, bottom) in LAYOUT.items():
        cell = CELLS[name]
        if left < 0 or bottom < 0 or left + cell.width > 160 + 1e-9:
            raise ValueError(
                f"{name} reaches past the figure's 160 mm width or below its bottom edge"
            )
        boxes[name] = (left, bottom, left + cell.width, bottom + cell.height)
    if max(box[3] for box in boxes.values()) > 225:
        raise ValueError("the figure is taller than 225 mm")
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = boxes[a], boxes[b]
            if frozenset({a, b}) in OVERLAPPING_CELLS:
                continue
            if ax0 < bx1 - 1e-9 and bx0 < ax1 - 1e-9 and ay0 < by1 - 1e-9 and by0 < ay1 - 1e-9:
                raise ValueError(f"the cells of {a} and {b} overlap")


def accepted_exceptions() -> dict[str, str]:
    """The departures from the figure grammar this figure accepts, each with its reason."""
    return {}


def letters() -> dict[str, str]:
    """Each panel's letter: the panels in LAYOUT's order take a, b, c and on."""
    return {name: chr(97 + i) for i, name in enumerate(LAYOUT)}


def write_letters() -> Path:
    path = BUILD / "letters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"figure": HERE.name, "letters": letters()}, indent=1) + "\n")
    return path


def draw_panel(fig, name: str, left: float, bottom: float):
    """Draw one panel with its cell's lower-left corner at (left, bottom) millimetres in the figure."""
    x, y, w, h = CELLS[name].plot
    ax = fs.add_axes(fig, left + x, bottom + y, w, h)
    stem = DATA_FILE.get(name, name)
    ax.set_gid(name)
    PANELS[name](ax, json.loads((HERE / f"{stem}.json").read_text()))
    if ax.axison:
        fs.frame_axes(ax)
    return ax


# Panels whose legend or label runs across the panels beside them, which a panel drawn alone does not hold.
SPANNING_TEXT = {
    "ladder_time": {
        "leaves the canvas": "its legend names the methods of every mapping panel and runs across them"
    },
    "t_cell_reference_comparison": {
        "leaves the canvas": "its last tick label, 100, reaches into the empty left margin of the map column"
    },
}


def render_panel(name: str) -> None:
    cell = CELLS[name]
    fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
    draw_panel(fig, name, 0.0, 0.0)
    allow = {**ALONE, **SPANNING_TEXT.get(name, {}), **accepted_exceptions()}
    fs.save(fig, name, folder=BUILD / "panels", allow=allow)
    plt.close(fig)


def render_figure() -> None:
    height = max(bottom + CELLS[name].height for name, (_, bottom) in LAYOUT.items())
    fig = plt.figure(figsize=(fs.FULL_WIDTH, height * fs.MM))
    letter_by_name = letters()
    for name, (left, bottom) in LAYOUT.items():
        cell = CELLS[name]
        draw_panel(fig, name, left, bottom)
        top_y = bottom + cell.height - 1.0
        fs.panel_label(fig, left + 1.0, top_y, letter_by_name[name])
        if name in HEADERS:
            header = HEADERS[name]
            if callable(header):
                header = header(
                    json.loads((HERE / f"{DATA_FILE.get(name, name)}.json").read_text())
                )
            fs.panel_title(
                fig, left + HEADER_INDENT_MM, bottom + cell.height - HEADER_BASELINE_MM, header
            )
    fs.save(fig, HERE.name, folder=BUILD, allow=accepted_exceptions())
    plt.close(fig)


def main(argv: list[str]) -> None:
    fs.apply()
    check_layout()
    write_letters()
    if len(argv) > 1:
        for name in argv[1:]:
            if name not in PANELS:
                raise SystemExit(f"unknown panel {name!r}; choose from {', '.join(PANELS)}")
            render_panel(name)
        return
    for name in PANELS:
        render_panel(name)
    render_figure()


if __name__ == "__main__":
    main(sys.argv)
