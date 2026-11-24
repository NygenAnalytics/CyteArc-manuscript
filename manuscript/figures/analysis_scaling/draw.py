"""Draw the figure on analysis at scale: one function per panel, and the panels assembled with their letters.

Each panel function draws one panel into the plot it is given and is named for what the panel shows. The
name is also the panel's SVG group ID and the name of its data file, <name>.json,
which prepare.py writes. CELLS gives each panel's printed size, and LAYOUT places the panels in rows, which
sets their letters. Run from the repository root:

  uv run --locked python -B -m manuscript.figures.analysis_scaling.draw            # every panel and the figure
  uv run --locked python -B -m manuscript.figures.analysis_scaling.draw <panel>    # one panel

Renders go to .build/figures/analysis_scaling/: each panel alone and without a letter as panels/<panel>.png,
.pdf and .svg; the lettered figure as analysis_scaling.png, .pdf and .svg; and each panel's letter in
letters.json, which every run writes.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

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
    1.8 mm above the plot or, given top_mm, its top that far above the plot. spacing overrides the legend's
    column spacing, handle length or padding."""
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
            "columnspacing": 0.65,
            "labelspacing": 0.25,
            "handlelength": 1.2,
            "handletextpad": 0.35,
            **spacing,
        },
    )


def log_axis(axis, ticks: list[float], minor: list[float] | None = None) -> None:
    """A log axis labelled with plain numbers at ticks, with no minor labels. Minor ticks sit at minor, none for an
    empty list, or between the decades when minor is None."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(
        FixedLocator(minor) if minor is not None else LogLocator(base=10, subs=np.arange(2, 10))
    )
    axis.set_major_formatter(FuncFormatter(plain))
    axis.set_minor_formatter(NullFormatter())


def range_bar(ax, x: float, low: float, high: float, cap: float = 0.08) -> None:
    """The range of the runs, lowest to highest, as an ink error bar with short caps."""
    ax.plot([x, x], [low, high], color=fs.INK, lw=fs.LW_AXIS, solid_capstyle="butt", zorder=5)
    for y in (low, high):
        ax.plot(
            [x - cap, x + cap], [y, y], color=fs.INK, lw=fs.LW_AXIS, solid_capstyle="butt", zorder=5
        )


def range_bar_across(ax, y: float, low: float, high: float, cap: float = 0.08) -> None:
    """range_bar for a horizontal plot: the range of the runs along x at height y."""
    ax.plot([low, high], [y, y], color=fs.INK, lw=fs.LW_AXIS, solid_capstyle="butt", zorder=5)
    for x in (low, high):
        ax.plot(
            [x, x], [y - cap, y + cap], color=fs.INK, lw=fs.LW_AXIS, solid_capstyle="butt", zorder=5
        )


# A memory limit (container size or budget): thin dashed ink, so tool hues mean only tools.
LIMIT_LINE = {"color": fs.INK, "lw": fs.LW_THIN, "ls": (0, (4, 2))}
# Every CyteArc bar in the figure: the 8 GiB step of the budget ramp.
CYTEARC_BAR = {"facecolor": fs.BUDGET_COLOURS[8], "edgecolor": fs.INK, "linewidth": fs.LW_BAR}
# Panel b's legend top sits this far above its plot, just under the title; panel c's outcome legend sits this far
# below its matrix, under the cells axis.
LEGEND_TOP_MM = 11.0
STATUS_LEGEND_BELOW_MM = 8.6

EVEN = {"o": 1.0, "s": 0.9, "D": 0.8, "^": 1.25}


def marker_ms(marker: str) -> float:
    """Legend marker size matching fs.points' filled markers (area 1.4 x MARKER x EVEN)."""
    return fs.MARKER_MS * float(np.sqrt(1.4 * EVEN[marker]))


def mean_point(
    ax, x: float, y: float, colour: str, marker: str, *, open_face: bool = False
) -> None:
    """The mean of the runs as a marker with an ink ring, filled, or open for a variant."""
    face = {"facecolor": fs.WHITE} if open_face else {}
    fs.points(
        ax,
        [x],
        [y],
        colour,
        marker,
        size=fs.MARKER * EVEN[marker],
        zorder=4,
        edgecolor=colour if open_face else fs.INK,
        linewidths=fs.LW_MARKER,
        **face,
    )


# ---- panel a: paired count store, exploded shard/chunk/unit anatomy, and streaming queue --------

SHARD = {"color": fs.INK, "lw": fs.LW_THIN, "solid_capstyle": "butt", "zorder": 5.2}
CHUNK = {"color": fs.SLATE, "lw": 0.35, "ls": (0, (1.2, 1.2)), "zorder": 5.1}
UNIT = {"facecolor": "none", "edgecolor": fs.BLUE, "lw": fs.LW_PRIMARY, "zorder": 5.5}
# The gene-major shard the middle of panel a enlarges, and its enlarged copy: a heavier ink outline.
ENLARGED_SHARD = {"facecolor": "none", "edgecolor": fs.INK, "lw": fs.LW_DATA, "zorder": 5.45}


def layout_matrix(
    ax, x: float, y: float, w: float, h: float, values: np.ndarray, kind: str
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float] | None]:
    """The count matrix (cells as rows, genes as columns) cut into shards, chunks and one read unit.
    Returns the box (x, y, width, height) of the highlighted read unit and, for the gene-major matrix, the box of
    the shard the middle of the panel enlarges: the lower shard of the middle column, which the read unit crosses."""
    ax.imshow(
        values,
        extent=(x, x + w, y, y + h),
        cmap=fs.SEQUENTIAL,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
        zorder=5,
    )
    shard = None
    if kind == "cell":
        edges = np.linspace(y + h, y, 5)
        for e in edges[1:-1]:
            ax.plot([x, x + w], [e, e], **SHARD)
        for xx in np.linspace(x, x + w, 4)[1:-1]:
            ax.plot([xx, xx], [y, y + h], **CHUNK)
        unit = (x, edges[2], w, edges[1] - edges[2])
    else:
        xs = np.linspace(x, x + w, 4)
        ys = np.linspace(y + h, y, 3)
        for xx in xs[1:-1]:
            ax.plot([xx, xx], [y, y + h], **SHARD)
        for yy in ys[1:-1]:
            ax.plot([x, x + w], [yy, yy], **SHARD)
        fine_x = np.linspace(x, x + w, 10)
        for xx in fine_x[1:-1]:
            if np.min(np.abs(xs - xx)) > 1e-6:
                ax.plot([xx, xx], [y, y + h], **CHUNK)
        for yy in np.linspace(y + h, y, 5)[1:-1]:
            if np.min(np.abs(ys - yy)) > 1e-6:
                ax.plot([x, x + w], [yy, yy], **CHUNK)
        sub = w / 9
        # A gene-wise read unit is one band of genes across every cell: the middle chunk column of the middle
        # shard column. The enlarged shard is that column's lower shard, the fullest of the sample's shards.
        unit = (x + 4 * sub, y, sub, h)
        shard = (x + w / 3, y, w / 3, h / 2)
        ax.add_patch(Rectangle(shard[:2], *shard[2:], **ENLARGED_SHARD))
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="none", edgecolor=fs.INK, lw=fs.LW_AXIS, zorder=5.3)
    )
    ax.add_patch(Rectangle(unit[:2], *unit[2:], **UNIT))
    return unit, shard


def store_layout(ax, data: dict) -> None:
    """Visual schematic of CyteArc's dual-orientation store, one gene-major shard enlarged to show its chunks and
    the read unit fetched from it, and the rolling queue of units in the memory budget feeding the CPU pool."""
    w, h = width_mm(ax), height_mm(ax)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_axis_off()

    heatmap = np.asarray(data["heatmap"], dtype=float)

    def say(x, y, text, *, colour=fs.INK, size=fs.TICK, ha="center", va="center", **kw):
        kw.setdefault("zorder", 8)
        return ax.text(x, y, text, fontsize=size, color=colour, ha=ha, va=va, **kw)

    y_bot = 1.8
    y_top = h - 4.8

    # 1. Left: the CyteArc store in cloud storage, with both orientations. The folder starts clear of the panel
    # letter, which the figure prints at the cell's top-left corner.
    fx0, fx1 = 3.5, 54.5
    g.folder(
        ax,
        fx0,
        y_bot,
        fx1,
        y_top,
        label="CyteArc store",
        tab_h=3.4,
        size=fs.TICK,
        fill=fs.WHITE,
        z=2,
    )
    cy = y_top + 1.7
    g.cloud(ax, 25.5, cy + 0.1, 4.8, z=4)
    say(29.2, cy, "cloud storage", colour=fs.SLATE, size=fs.NOTE, ha="left")

    # The two matrices share the row of the enlarged shard in the middle: same top, bottom and labels.
    row_y, row_h = y_bot + 10.5, 17.5
    mw, mh, my = 18.0, row_h, row_y
    cell_x, gene_x = 8.5, 33.5
    layout_matrix(ax, cell_x, my, mw, mh, heatmap, "cell")
    _, shard = layout_matrix(ax, gene_x, my, mw, mh, heatmap, "gene")
    say(cell_x + mw / 2, my + mh + 2.0, "genes", colour=fs.SLATE, size=fs.NOTE)
    say(gene_x + mw / 2, my + mh + 2.0, "genes", colour=fs.SLATE, size=fs.NOTE)
    say(cell_x - 2.5, my + mh / 2, "cells", colour=fs.SLATE, size=fs.NOTE, rotation=90)
    say(gene_x - 2.5, my + mh / 2, "cells", colour=fs.SLATE, size=fs.NOTE, rotation=90)
    say(cell_x + mw / 2, my - 2.4, "cell-major", size=fs.TICK)
    say(gene_x + mw / 2, my - 2.4, "gene-major", size=fs.TICK)

    # 2. Middle: the outlined gene-major shard, enlarged. Dotted lines run from its corners to the enlarged copy,
    # which shows the same counts: grey, except the read unit's chunks in blue, which HTTP range requests fetch
    # and the decoder turns into a dense unit.
    zx0, zx1 = 58.0, 106.5
    g.card(ax, zx0, y_bot, zx1, y_top + 1.2, z=2)
    say((zx0 + zx1) / 2, y_top - 1.5, "Hierarchy & range read", size=fs.TICK)

    sx, sy, sw, sh = zx0 + 3.0, row_y, 16.5, row_h
    shard_x, shard_y, shard_w, shard_h = shard
    for corner_y, enlarged_y in ((shard_y + shard_h, sy + sh), (shard_y, sy)):
        ax.plot(
            [shard_x + shard_w, sx],
            [corner_y, enlarged_y],
            color=fs.SLATE,
            lw=0.4,
            ls=":",
            zorder=5.35,
        )

    rows, columns = heatmap.shape
    shard_rows, shard_columns = slice(rows // 2, rows), slice(columns // 3, 2 * columns // 3)
    unit_columns = slice(4 * columns // 9, 5 * columns // 9)
    c_xs = np.linspace(sx, sx + sw, 4)
    c_ys = np.linspace(sy, sy + sh, 3)
    ax.imshow(
        heatmap[shard_rows, shard_columns],
        extent=(sx, sx + sw, sy, sy + sh),
        cmap=fs.NEUTRAL,
        vmin=0,
        vmax=1.5,
        interpolation="nearest",
        aspect="auto",
        zorder=5,
    )
    ux0, ux1 = c_xs[1], c_xs[2]
    ax.imshow(
        heatmap[shard_rows, unit_columns],
        extent=(ux0, ux1, sy, sy + sh),
        cmap=fs.SEQUENTIAL,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
        zorder=5.05,
    )
    for cxx in c_xs[1:-1]:
        ax.plot([cxx, cxx], [sy, sy + sh], **CHUNK)
    for cyy in c_ys[1:-1]:
        ax.plot([sx, sx + sw], [cyy, cyy], **CHUNK)
    ax.add_patch(Rectangle((sx, sy), sw, sh, **ENLARGED_SHARD))
    ax.add_patch(
        Rectangle(
            (ux0, sy),
            ux1 - ux0,
            sh,
            facecolor="none",
            edgecolor=fs.BLUE,
            lw=fs.LW_PRIMARY,
            zorder=5.6,
        )
    )
    # Direct labels: the shard by name, and arrows to one chunk and to the read unit.
    say(sx + sw / 2, sy + sh + 2.0, "shard (one file)", size=fs.NOTE)
    bottom_chunk_y = (c_ys[0] + c_ys[1]) / 2
    chunk_x, unit_x = (c_xs[0] + c_xs[1]) / 2, (ux0 + ux1) / 2
    say(chunk_x, sy - 2.6, "chunk", size=fs.NOTE)
    g.arrow(ax, (chunk_x, sy - 1.6), (chunk_x, bottom_chunk_y), lw=fs.LW_THIN, head=3.0, z=8)
    say(unit_x + 2.0, sy - 5.2, "read unit", size=fs.NOTE)
    g.arrow(ax, (unit_x, sy - 4.2), (unit_x, bottom_chunk_y), lw=fs.LW_THIN, head=3.0, z=8)

    # The read unit's chunks in this shard, fetched by HTTP range requests and decoded into a dense unit. The decoded
    # unit keeps the shard's margin from the card's right edge, and the two gaps between the three are equal.
    rw, rh = 6.5, 4.2
    dw, dh = 9.0, sh
    gap = (zx1 - (sx - zx0) - sx - sw - rw - dw) / 2
    rx0 = sx + sw + gap
    dx0 = rx0 + rw + gap
    for ci in range(2):
        cy_chunk = (c_ys[ci] + c_ys[ci + 1]) / 2
        g.arrow(ax, (ux1 + 0.4, cy_chunk), (rx0 - 0.4, cy_chunk), lw=fs.LW_THIN, head=3.0, z=6)
        ax.add_patch(
            Rectangle(
                (rx0, cy_chunk - rh / 2),
                rw,
                rh,
                facecolor=fs.tint(fs.BLUE, 0.65),
                edgecolor=fs.INK,
                lw=fs.LW_THIN,
                zorder=5,
            )
        )
        g.arrow(ax, (rx0 + rw + 0.4, cy_chunk), (dx0 - 0.4, cy_chunk), lw=fs.LW_THIN, head=3.0, z=6)

    ax.imshow(
        heatmap[shard_rows, unit_columns],
        extent=(dx0, dx0 + dw, sy, sy + dh),
        cmap=fs.SEQUENTIAL,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
        zorder=5,
    )
    ax.add_patch(
        Rectangle(
            (dx0, sy), dw, dh, facecolor="none", edgecolor=fs.BLUE, lw=fs.LW_PRIMARY, zorder=5.5
        )
    )
    say(rx0 + rw / 2, sy - 2.4, "HTTP GET", colour=fs.SLATE, size=fs.NOTE)
    say(rx0 + rw / 2, sy - 5.0, "chunks", colour=fs.SLATE, size=fs.NOTE)
    say(dx0 + dw / 2, sy - 2.4, "decoded", colour=fs.INK, size=fs.NOTE)
    say(dx0 + dw / 2, sy - 5.0, "unit", colour=fs.INK, size=fs.NOTE)

    # 3. Right: Rolling queue of N units inside the RAM budget feeding a unified compute pool
    px0, px1 = 110.0, w - 1.0
    g.card(ax, px0, y_bot, px1, y_top + 1.2, module="stream", z=2)
    say((px0 + px1) / 2, y_top - 1.5, "Memory-bounded stream", size=fs.TICK)

    bx0, bx1 = px0 + 2.8, px0 + 23.5
    by0, by1 = y_bot + 7.5, y_top - 5.2
    # The connecting arrows run straight at the decoded unit's height and match the arrows inside the read box.
    bridge_y = sy + dh / 2
    g.arrow(ax, (dx0 + dw + 0.5, bridge_y), (bx0 - 0.4, bridge_y), lw=fs.LW_THIN, head=3.0, z=6)

    ax.add_patch(
        Rectangle(
            (bx0, by0),
            bx1 - bx0,
            by1 - by0,
            facecolor=fs.WHITE,
            edgecolor=fs.INK,
            lw=fs.LW_DATA,
            ls=(0, (3, 2)),
            zorder=3,
        )
    )
    say((bx0 + bx1) / 2, y_bot + 3.4, "memory budget", colour=fs.INK, size=fs.NOTE)

    # Five stacked units inside the memory budget, with three dots between unit 3 and unit N.
    u_slots = np.linspace(by1 - 3.2, by0 + 3.2, 5)
    u_labels = ("unit 1", "unit 2", "unit 3", "dots", "unit N")
    for uy_c, ulab in zip(u_slots, u_labels):
        if ulab == "dots":
            for dy_dot in (-0.9, 0.0, 0.9):
                ax.add_patch(
                    Circle(
                        ((bx0 + bx1) / 2, uy_c + dy_dot),
                        0.28,
                        facecolor=fs.SLATE,
                        edgecolor="none",
                        zorder=5,
                    )
                )
        else:
            ax.add_patch(
                Rectangle(
                    (bx0 + 1.4, uy_c - 1.65),
                    bx1 - bx0 - 2.8,
                    3.3,
                    facecolor=fs.tint(fs.BLUE, 0.28),
                    edgecolor=fs.BLUE,
                    lw=fs.LW_AXIS,
                    zorder=4,
                )
            )
            say((bx0 + bx1) / 2, uy_c, ulab, size=fs.NOTE)

    # One arrow from each unit to the CPU pool: the units in the budget are processed side by side.
    wx0, wx1 = bx1 + 5.2, px1 - 2.4
    wy0, wy1 = by0 + 1.5, by1 - 1.5
    for uy_c, ulab in zip(u_slots, u_labels):
        if ulab != "dots":
            g.arrow(ax, (bx1 + 0.5, uy_c), (wx0 - 0.5, uy_c), lw=fs.LW_THIN, head=3.0, z=6)
    ax.add_patch(
        FancyBboxPatch(
            (wx0, wy0),
            wx1 - wx0,
            wy1 - wy0,
            boxstyle="round,pad=0,rounding_size=1.0",
            facecolor=fs.WHITE,
            edgecolor=fs.INK,
            lw=fs.LW_AXIS,
            zorder=4,
        )
    )
    say((wx0 + wx1) / 2, wy1 - 2.8, "CPU pool", size=fs.NOTE)
    # The passes that run on the streamed units, one line each under the pool's name and a rule.
    ax.plot([wx0 + 1.6, wx1 - 1.6], [wy1 - 4.6] * 2, color=fs.LIGHT, lw=fs.LW_THIN, zorder=4.5)
    operations = ("filtering", "HVGs", "normalization", "PCA", "marker search")
    top, bottom = wy1 - 4.6, wy0 + 0.6
    pitch = (top - bottom) / len(operations)
    for idx, operation in enumerate(operations):
        say((wx0 + wx1) / 2, top - pitch * (idx + 0.5), operation, size=fs.NOTE)


# ---- panels b & c: scaling curves + unified failure-reason grid ---------------------------------


class Arm(NamedTuple):
    key: str
    entity: str
    marker: str
    name: str
    short: str


ARMS = (
    Arm("cytearc", "cytearc", "o", "CyteArc", "CyteArc"),
    Arm("bpcells_cached", "bpcells", "D", "BPCells", "BPCells"),
    Arm("scanpy_dask_covariance_csc", "scanpy", "D", "Scanpy (Dask)", "Scanpy-D"),
    Arm("seurat_bpcells", "seurat", "s", "Seurat (BPCells)", "Seurat-B"),
    Arm("scanpy", "scanpy", "^", "Scanpy (memory)", "Scanpy"),
    Arm("census_scanpy", "scanpy", "s", "Scanpy (Census)", "Census"),
    Arm("seurat", "seurat", "^", "Seurat (memory)", "Seurat"),
)
ARM = {arm.key: arm for arm in ARMS}
POSITION = {
    "embryo_1000000": 0.0,
    "brain_1306127": 1.0,
    "hbca_neurons": 2.0,
    "mssm_cohort": 3.0,
    "embryo_10000000": 4.0,
}
TAHOE_X = 5.0


def cell_outcome(arm_key: str, dataset: str, entry: dict) -> str:
    """Classify each (tool, dataset) outcome into one of five states:
    'pass' : completed
    'oom'  : ran out of RAM (336 GiB limit)
    'time' : ran out of time (>20.5h / >24h limit)
    'err'  : software / format error (>2^31 index limit or uwot segfault)
    'none' : not run, such as the comparators on Tahoe-100M without a run there
    """
    if dataset == "tahoe" and arm_key == "cytearc":
        return "pass"
    st = entry.get("state", "done")
    if st == "done":
        return "pass"
    if st == "not run":
        return "none"
    if st == "oom":
        return "oom"
    if st == "timeout":
        return "time"
    return "err"


def outcome_entry(data: dict, arm_key: str, dataset: str) -> dict:
    """The outcome record of one grid cell. On Tahoe-100M it is the configuration's comparator run, and a
    configuration without one was not run."""
    if dataset == "tahoe":
        return data["tahoe"]["outcomes"].get(arm_key, {"state": "not run"})
    return data["arms"][arm_key][dataset]


def draw_status_grid(ax, data: dict):
    """The outcome of each toolkit configuration on each dataset, a row per configuration, directly below ax and
    on its x positions; returns the matrix's axes."""
    pos = ax.get_position()
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    x_mm = pos.x0 * fw
    w_mm = pos.width * fw
    g_h_mm = 17.5
    g_y_mm = pos.y0 * fh - g_h_mm - 1.0
    gax = fs.add_axes(fig, x_mm, g_y_mm, w_mm, g_h_mm)
    gax.set_gid(f"{ax.get_gid()}_grid")

    order = sorted(POSITION, key=POSITION.get)
    gax.set_xlim(-0.42, TAHOE_X + 0.42)
    n_rows = len(ARMS)
    gax.set_ylim(-0.55, n_rows - 0.45)

    ticks = [POSITION[d] for d in order] + [TAHOE_X]
    labels = [data["datasets"][d]["tick"] for d in order] + [data["tahoe"]["tick"]]
    gax.xaxis.set_major_locator(FixedLocator(ticks))
    gax.xaxis.set_major_formatter(FuncFormatter(lambda _, i: labels[i]))
    gax.xaxis.set_minor_locator(NullLocator())
    gax.set_xlabel("cells")

    gax.set_yticks(range(n_rows))
    gax.set_yticklabels([a.short for a in reversed(ARMS)], fontsize=fs.NOTE)
    gax.tick_params(axis="y", length=0, pad=1.5)

    for r in range(n_rows):
        if r % 2 == 1:
            gax.axhspan(r - 0.5, r + 0.5, facecolor=fs.PAPER, edgecolor="none", zorder=0)

    all_cols = [(d, POSITION[d]) for d in order] + [("tahoe", TAHOE_X)]
    for row_idx, arm in enumerate(ARMS):
        y = n_rows - 1 - row_idx
        for d, x in all_cols:
            kind = cell_outcome(arm.key, d, outcome_entry(data, arm.key, d))
            if kind == "pass":
                # Small muted dot so completed runs don't crowd the failure marks
                gax.scatter(
                    [x],
                    [y],
                    s=5.5,
                    marker="o",
                    facecolor=fs.LIGHT,
                    edgecolor=fs.SLATE,
                    linewidths=0.4,
                    zorder=2,
                )
            elif kind == "oom":
                # Out of RAM: an ink x; outcomes differ by shape, since tool hues mark only tools
                gax.scatter(
                    [x],
                    [y],
                    s=14.0,
                    marker="x",
                    color=fs.INK,
                    linewidths=0.85,
                    zorder=4,
                )
            elif kind == "time":
                # Out of time: an ink +
                gax.scatter(
                    [x],
                    [y],
                    s=16.0,
                    marker="+",
                    color=fs.INK,
                    linewidths=0.9,
                    zorder=4,
                )
            elif kind == "none":
                # Not run: a short slate dash
                gax.scatter([x], [y], s=12.0, marker="_", color=fs.SLATE, linewidths=0.8, zorder=2)
            else:
                # Crash / >2^31 index limit / error: open ink triangle
                gax.scatter(
                    [x],
                    [y],
                    s=11.0,
                    marker="v",
                    facecolor=fs.WHITE,
                    edgecolor=fs.INK,
                    linewidths=0.65,
                    zorder=3,
                )

    fs.frame_axes(gax)
    return gax


def analysis_time(ax, data: dict) -> None:
    """Analysis time of each toolkit configuration on the benchmark datasets and Tahoe-100M. Panel c sits directly
    below on the same x positions and carries the outcome matrix the two panels share."""
    measured = [
        v["minutes"] for rows in data["arms"].values() for v in rows.values() if "minutes" in v
    ]
    low = min(m["low"] for m in measured) / 1.5
    high = max([m["high"] for m in measured] + [data["tahoe"]["minutes"]["high"]]) * 1.5
    ticks = [
        factor * 10.0**power
        for power in range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
        for factor in (1, 5)
        if low <= factor * 10.0**power <= high
    ]
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    log_axis(ax.yaxis, ticks)
    configurations_by_cells(ax, data, "minutes", tahoe=data["tahoe"]["minutes"], hide_xticks=True)
    ax.set_ylabel("analysis time (min, log10)")
    configuration_legend(ax, data["legendOrder"])


def configurations_by_cells(
    ax, data: dict, field: str, *, tahoe: dict | None = None, hide_xticks: bool = False
) -> None:
    """Each configuration's summary under field on the datasets along the cells axis, and CyteArc's summary on
    Tahoe-100M (tahoe) at the right end."""
    order = sorted(POSITION, key=POSITION.get)
    ax.set_xlim(-0.42, TAHOE_X + 0.42)
    ticks = [POSITION[d] for d in order] + [TAHOE_X]
    labels = [data["datasets"][d]["tick"] for d in order] + [data["tahoe"]["tick"]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    if hide_xticks:
        ax.xaxis.set_major_formatter(NullFormatter())
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda _, i: labels[i]))
        ax.set_xlabel("cells")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    cap = 0.08 * 64.0 / width_mm(ax)
    for arm in ARMS:
        colour = fs.ENTITIES[arm.entity].colour
        xs, ys = [], []
        for dataset in order:
            row = data["arms"][arm.key][dataset]
            if field not in row:
                continue
            mean_point(ax, POSITION[dataset], row[field]["mean"], colour, arm.marker)
            if row[field]["n"] > 1:
                range_bar(ax, POSITION[dataset], row[field]["low"], row[field]["high"], cap=cap)
            xs.append(POSITION[dataset])
            ys.append(row[field]["mean"])
        if arm.key == "cytearc" and tahoe is not None:
            mean_point(ax, TAHOE_X, tahoe["mean"], colour, arm.marker)
            if tahoe["n"] > 1:
                range_bar(ax, TAHOE_X, tahoe["low"], tahoe["high"], cap=cap)
            xs.append(TAHOE_X)
            ys.append(tahoe["mean"])
        if len(xs) > 1:
            width = fs.LW_PRIMARY if arm.entity == "cytearc" else fs.LW_DATA
            ax.plot(xs, ys, color=colour, lw=width, zorder=2)


def configuration_legend(ax, order: list[str]) -> None:
    """The toolkit configurations in the given order, in three rows across the top of panel b."""
    shown = [ARM[key] for key in order]
    handles = [
        Line2D(
            [],
            [],
            color=fs.ENTITIES[a.entity].colour,
            marker=a.marker,
            ms=marker_ms(a.marker),
            mfc=fs.ENTITIES[a.entity].colour,
            mec=fs.INK,
            mew=fs.LW_MARKER,
            lw=fs.LW_PRIMARY if a.entity == "cytearc" else fs.LW_DATA,
        )
        for a in shown
    ]
    legend_above(
        ax,
        handles,
        [a.name for a in shown],
        ncol=3,
        left_mm=-11.3,
        top_mm=LEGEND_TOP_MM,
        columnspacing=0.5,
        handlelength=1.0,
        handletextpad=0.3,
    )


def peak_memory(ax, data: dict) -> None:
    """Peak container memory of each toolkit configuration on the benchmark datasets and Tahoe-100M on a log2 scale,
    above the matrix of each configuration's outcome on each dataset, which panels b and c share."""
    ax.set_yscale("log", base=2)
    ax.set_ylim(2.8, 420)
    log_axis(ax.yaxis, [4, 8, 16, 32, 64, 128, 256], minor=[])
    configurations_by_cells(ax, data, "peakGiB", tahoe=data["tahoe"]["peakGiB"], hide_xticks=True)
    ax.set_ylabel("peak memory (GiB, log2)")
    grid = draw_status_grid(ax, data)
    # The outcome states, in two rows under the matrix's cells axis.
    status_handles = [
        Line2D([], [], ls="none", marker="o", ms=2.6, mfc=fs.LIGHT, mec=fs.SLATE, mew=0.4),
        Line2D([], [], ls="none", marker="x", ms=4.0, color=fs.INK, mew=0.85),
        Line2D([], [], ls="none", marker="+", ms=4.4, color=fs.INK, mew=0.9),
        Line2D([], [], ls="none", marker="v", ms=3.5, mfc=fs.WHITE, mec=fs.INK, mew=0.65),
        Line2D([], [], ls="none", marker="_", ms=3.6, color=fs.SLATE, mew=0.8),
    ]
    grid.legend(
        status_handles,
        ["pass", "out of RAM", "out of time", "limit / crash", "not run"],
        loc="upper center",
        bbox_to_anchor=(0.5, -STATUS_LEGEND_BELOW_MM / height_mm(grid)),
        borderaxespad=0.0,
        ncol=3,
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.35,
    )


# ---- panels f and g: the Tahoe-100M run's stages, and its UMAP by cell line with its clusters' purity ------


# The UMAP's cell-line colours start from the figure's twelve categorical colours.
UMAP_SEED_COLOURS = (*fs.GROUPS, fs.BLUE, fs.SKY, fs.GREEN, fs.ORANGE, fs.WINE, fs.PINK)
# sRGB's primaries and D65 white in CIE XYZ, for CIELAB distances between colours.
SRGB_TO_XYZ = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
)
D65_WHITE = np.array([0.9505, 1.0, 1.089])
# Two cell lines border each other on the UMAP when this many pairs of neighbouring dense pixels join them.
BORDER_PIXELS = 40
# The cell-line names on the UMAP sit a step below the type scale's 6 pt so they cover less of the map.
MAP_LABEL_PT = 5.0
# The UMAP's cell-line colours are mixed with this share of white, here and in the purity pies, so the map draws
# less attention than the panels around it.
UMAP_FADE = 0.3


def cielab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB coordinates of sRGB colours, one colour per row with channels from 0 to 1."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ SRGB_TO_XYZ.T / D65_WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.column_stack(
        [116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])]
    )


def categorical_palette(count: int) -> np.ndarray:
    """count distinct sRGB colours: the seed colours, then colours added one at a time, each the candidate farthest in
    CIELAB from every colour chosen so far, among candidates neither too light to read on white, nor too grey, nor
    more saturated than most seed colours (CIELAB chroma 28 to 60)."""
    seeds = np.array([to_rgb(c) for c in UMAP_SEED_COLOURS])
    steps = np.linspace(0.0, 1.0, 21)
    grid = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    lab = cielab(grid)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = (lab[:, 0] >= 35) & (lab[:, 0] <= 78) & (chroma >= 28) & (chroma <= 60)
    candidates, candidate_lab = grid[keep], lab[keep]
    chosen = list(seeds[:count])
    nearest = np.linalg.norm(candidate_lab[:, None] - cielab(seeds)[None], axis=2).min(axis=1)
    while len(chosen) < count:
        best = int(np.argmax(nearest))
        chosen.append(candidates[best])
        nearest = np.minimum(nearest, np.linalg.norm(candidate_lab - candidate_lab[best], axis=1))
    return np.array(chosen)


def cell_line_colours(dominant: np.ndarray, counts: np.ndarray, lines: int) -> np.ndarray:
    """One colour per cell line, no two alike. The lines take colours largest first: a line whose bordering lines
    hold colours already takes the free colour farthest from theirs, any other line the next free colour."""
    if dominant.max() >= lines:
        raise ValueError(
            f"the raster names cell line {dominant.max()}, beyond the {lines} lines of the run"
        )
    lab = cielab(palette := categorical_palette(lines))
    # Borders between the dense bodies of the islands: neighbouring pixels, each above the median count of the
    # occupied pixels, whose commonest cell lines differ.
    dense = np.where(counts > np.median(counts[counts > 0]), dominant, -1)
    border = np.zeros((lines, lines))
    for a, b in ((dense[:, :-1], dense[:, 1:]), (dense[:-1, :], dense[1:, :])):
        joined = (a >= 0) & (b >= 0) & (a != b)
        np.add.at(border, (a[joined], b[joined]), 1)
    border += border.T
    occupied = dominant >= 0
    size = np.bincount(dominant[occupied], weights=counts[occupied], minlength=lines)
    colour = np.full(lines, -1)
    free = list(range(lines))
    for line in np.argsort(-size, kind="stable"):
        neighbours = [n for n in np.flatnonzero(border[line] >= BORDER_PIXELS) if colour[n] >= 0]
        if neighbours:
            gaps = np.linalg.norm(lab[free][:, None] - lab[colour[neighbours]][None], axis=2).min(
                axis=1
            )
            pick = free[int(np.argmax(gaps))]
        else:
            pick = free[0]
        colour[line] = pick
        free.remove(pick)
    return palette[colour]


def tahoe_umap(ax, data: dict) -> None:
    """Density raster of the Tahoe-100M UMAP embedding, each pixel coloured by its commonest cell line, one colour
    per line, with the largest cell lines named where their cells sit, and beside it a strip of the Leiden clusters'
    purity."""
    with np.load(REPO / data["raster"]) as r:
        counts = r["cells"]
        dom = r["commonestCellLine"]

    colours = (1 - UMAP_FADE) * cell_line_colours(dom, counts, data["cellLines"]) + UMAP_FADE
    paper_rgb = np.array(to_rgb(fs.WHITE), dtype=float)

    h_img, w_img = counts.shape
    rgb_img = np.broadcast_to(paper_rgb, (h_img, w_img, 3)).copy()
    mask = (counts > 0) & (dom >= 0)
    line_colours = colours[dom[mask]]
    log_c = np.log1p(counts[mask].astype(float))
    q98 = float(np.quantile(log_c, 0.98)) if log_c.size else 1.0
    share = np.clip(0.32 + 0.68 * (log_c / max(q98, 1e-6)), 0.32, 1.0)[:, None]
    rgb_img[mask] = share * line_colours + (1.0 - share) * 1.0

    # Crop tight around non-empty cells so the UMAP fills the axes with minimal white margin
    nz_y, nz_x = np.nonzero(mask)
    pad_px = 45
    y0_px = max(0, int(nz_y.min()) - pad_px)
    y1_px = min(h_img, int(nz_y.max()) + pad_px)
    x0_px = max(0, int(nz_x.min()) - pad_px)
    x1_px = min(w_img, int(nz_x.max()) + pad_px)
    cropped = rgb_img[y0_px:y1_px, x0_px:x1_px][::-1]

    ax.imshow(
        cropped,
        cmap=fs.SEQUENTIAL,
        interpolation="bilinear",
        aspect="auto",
        zorder=2,
    )
    # Each named cell line at the cell-weighted median row and column of its pixels, moved into the cropped and
    # flipped image.
    for label in data["labels"]:
        ax.text(
            label["column"] - x0_px,
            y1_px - 1 - label["row"],
            label["name"],
            ha="center",
            va="center",
            fontsize=MAP_LABEL_PT,
            color=fs.INK,
            zorder=4,
            path_effects=[patheffects.withStroke(linewidth=1.0, foreground=fs.WHITE)],
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", labelpad=1.2)
    ax.set_ylabel("UMAP 2", labelpad=1.2)

    pos = ax.get_position()
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    strip_ax = fs.add_axes(fig, pos.x1 * fw + 1.2, pos.y0 * fh, 8.5, pos.height * fh)
    strip_ax.set_gid("tahoe_cluster_purity")
    strip_ax.set_label("tahoe_cluster_purity")
    cluster_purity(strip_ax, data["clusters"], colours)


def cluster_purity(ax, clusters: list[dict], colours: np.ndarray) -> None:
    """One circle per Leiden cluster, purest at the top, at the share of the cluster's cells from its commonest cell
    line, sized by the cluster's cells and filled with that line's UMAP colour."""
    ordered = sorted(clusters, key=lambda c: (-c["commonestCellLineFraction"], -c["cells"]))
    ax.scatter(
        [100.0 * c["commonestCellLineFraction"] for c in ordered],
        range(len(ordered)),
        s=[PURITY_SIZE[0] + PURITY_SIZE[1] * c["cells"] / 1e6 for c in ordered],
        c=[colours[c["composition"][0]["cellLine"]] for c in ordered],
        edgecolors=fs.INK,
        linewidths=fs.LW_CELL,
        zorder=3,
        clip_on=False,
    )
    ax.set_xlim(*PURITY_LIMITS)
    ax.set_ylim(len(ordered) - 1 + PURITY_PAD_ROWS, -PURITY_PAD_ROWS)
    ax.set_xticks(PURITY_TICKS)
    ax.set_yticks([])
    ax.tick_params(axis="x", pad=1.5, labelsize=fs.NOTE)
    for tick in PURITY_TICKS:
        ax.axvline(tick, color=fs.SLATE, lw=fs.LW_THIN, ls=(0, (1, 1.5)), zorder=1)
    # The title ends at the strip's right edge so that it clears the panel to the right.
    ax.set_title("cluster\npurity (%)", fontsize=fs.NOTE, loc="right", pad=1.6, linespacing=1.0)
    fs.frame_axes(ax)


# The purity strip's axis, padded on both sides so that no circle meets the frame, with dotted lines at round ticks
# from just under the lowest purity (21%) to 100%; rows of padding above and below; the circles' area in points
# squared: a floor plus a share per million cells.
PURITY_LIMITS = (10.0, 110.0)
PURITY_TICKS = (20, 60, 100)
PURITY_PAD_ROWS = 1.2
PURITY_SIZE = (2.0, 4.5)


def tahoe_stages(ax, data: dict) -> None:
    """The minutes and the peak container memory of each stage of the 95.6M-cell Tahoe-100M run, in two charts that
    share the stage rows, in the stage-time panel's order: minutes on a log10 axis, as in that panel, and memory on a
    log2 axis, as in the other memory panels. ax is the minutes chart; the memory chart follows to its right."""
    rows = data["rows"]
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    pos = ax.get_position()
    memory_ax = fs.add_axes(
        fig, pos.x1 * fw + TAHOE_CHART_GAP_MM, pos.y0 * fh, pos.width * fw, pos.height * fh
    )
    memory_ax.set_gid("tahoe_stage_memory")
    stage_bars(
        ax, rows, "minutes", "time", "min (log10)", base=10, limits=(8, 3000), ticks=[10, 100]
    )
    stage_bars(
        memory_ax,
        rows,
        "peakGiB",
        "memory",
        "GiB (log2)",
        base=2,
        limits=(8, 256),
        ticks=[16, 64],
    )
    ax.set_yticklabels([row["stage"] for row in reversed(rows)], fontsize=fs.NOTE)
    memory_ax.set_yticklabels([])


TAHOE_CHART_GAP_MM = 2.0


def stage_bars(
    ax,
    rows: list[dict],
    field: str,
    title: str,
    xlabel: str,
    *,
    base: int,
    limits: tuple[float, float],
    ticks: list[float],
) -> None:
    """One CyteArc bar per Tahoe-100M stage to the mean of the replicates under field, top to bottom in the rows'
    order, with the replicates' range as a whisker and the mean as a label past it, under a short title; the caller
    names the rows."""
    top = len(rows) - 1
    for i, row in enumerate(rows):
        summary = row[field]
        ax.barh(top - i, summary["mean"], height=0.62, **CYTEARC_BAR, zorder=3)
        if summary["n"] > 1:
            range_bar_across(ax, top - i, summary["low"], summary["high"], cap=0.16)
        ax.annotate(
            f"{summary['mean']:.0f}",
            (max(summary["mean"], summary["high"]), top - i),
            xytext=(1.5, 0),
            textcoords="offset points",
            fontsize=fs.NOTE,
            color=fs.INK,
            va="center",
        )
    ax.set_xscale("log", base=base)
    ax.set_xlim(*limits)
    # A log10 axis carries minor ticks between its decades, as in panel b; a log2 axis has none.
    log_axis(ax.xaxis, ticks, minor=None if base == 10 else [])
    ax.set_ylim(-0.6, top + 0.6)
    ax.set_yticks(range(len(rows)))
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.grid(True, axis="x", which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=fs.TICK, loc="left", pad=2.0)
    fs.frame_axes(ax)


# ---- panel d: 10M-cell stage breakdown across CyteArc, BPCells, and Scanpy (Dask) ---------------

STAGE_TOOLS = (
    ("cytearc", "cytearc", "CyteArc"),
    ("bpcells_cached", "bpcells", "BPCells"),
    ("scanpy_dask_covariance_csc", "scanpy", "Scanpy (Dask)"),
)


def tool_bar(entity: str) -> dict:
    """A tool's bar: CyteArc's bar colour for CyteArc, a tint of the tool's colour for the others."""
    return CYTEARC_BAR if entity == "cytearc" else fs.bar_style(fs.ENTITIES[entity].colour)


STAGE_TIME_LIMITS = (0.8, 1100.0)
# Where a stage a tool never finished takes its +, in minutes on the log axis: just above the floor.
UNFINISHED_MARK_Y = 1.5


def stage_times(ax, data: dict) -> None:
    """Minutes of each analysis stage on ten million cells across CyteArc, BPCells and Scanpy (Dask), as vertical
    bars grouped by stage in the order the workflow runs them. Above each group, BPCells' time over CyteArc's; below
    it, the stage's name, which the figure joins to the stage's span on the resource trace."""
    stages = data["stages"]
    other = data["otherTools10M"]
    offsets, width = (-0.27, 0.0, 0.27), 0.25
    # Bars rise from the log axis's floor rather than from zero, so each bar ends where the plot does.
    floor = STAGE_TIME_LIMITS[0]
    for x, stage in enumerate(stages):
        name = stage["stage"]
        heights = []
        for offset, (key, entity, _) in zip(offsets, STAGE_TOOLS):
            if key in ("cytearc", "bpcells_cached"):
                run = stage[key]
                ax.bar(
                    x + offset,
                    run["mean"] - floor,
                    bottom=floor,
                    width=width,
                    zorder=3,
                    **tool_bar(entity),
                )
                if run["n"] > 1:
                    range_bar(ax, x + offset, run["low"], run["high"], cap=0.07)
                heights.append(run["high"])
            elif other[key].get(name) is not None:
                ax.bar(
                    x + offset,
                    other[key][name] - floor,
                    bottom=floor,
                    width=width,
                    zorder=3,
                    **tool_bar(entity),
                )
                heights.append(other[key][name])
            elif key == "scanpy_dask_covariance_csc" and name == "markers":
                # Scanpy (Dask) never finished marker search, so the slot has no bar, only the out-of-time + of
                # panel c's outcome grid, just above the axis floor.
                ax.scatter(
                    [x + offset],
                    [UNFINISHED_MARK_Y],
                    s=16.0,
                    marker="+",
                    color=fs.INK,
                    linewidths=0.9,
                    zorder=4,
                )
        fs.note(
            ax,
            x,
            max(heights) * 1.3,
            f"×{stage['ratio']:.1f}",
            transform=ax.transData,
            ha="center",
            va="bottom",
            color=fs.INK,
        )
    ax.set_yscale("log")
    ax.set_ylim(*STAGE_TIME_LIMITS)
    log_axis(ax.yaxis, [1, 10, 100, 1000])
    ax.set_xlim(-0.45, len(stages) - 0.55)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    stage_names(ax, [stage["stage"] for stage in stages])
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.set_ylabel("min (log10)")
    handles = [Patch(**tool_bar(entity)) for _, entity, _ in STAGE_TOOLS]
    # The legend starts under the panel's title, clear of panel b's legend to its left.
    legend_above(ax, handles, [name for _, _, name in STAGE_TOOLS], ncol=3, left_mm=-6.2)


# ---- panel e: one run's memory, CPU and network, once a second ---------------------------------------------------

TRACE_LANE_GAP_MM = 1.6
# The lanes' names read upward in one column, ending this far left of the axes, clear of the tick labels. Each hangs
# from its lane's top, so the network name, longer than its lane, runs on below it.
TRACE_UNIT_MM = 5.9
# The network lane's log axis starts here; rates below it, idle seconds among them, are drawn on it.
NETWORK_FLOOR_MBPS = 0.1
# The stage names under the stage-time bars: this far below the plot, and at least this far apart.
STAGE_NAME_DROP_MM = 1.0
STAGE_NAME_GAP_MM = 0.6
# The trace's stage bands and the dotted lines where each stage starts.
TRACE_BAND = fs.tint(fs.SLATE, 0.2)
TRACE_EDGE = {"color": fs.SLATE, "lw": fs.LW_THIN, "ls": (0, (1, 1.5))}


def resource_trace(ax, data: dict) -> None:
    """One run's container memory up to its peak, CPU cores busy against the container's limit and network receive
    rate, once a second, in three lanes on one time axis. Bands shade every other stage, dotted lines mark where each
    stage starts; the figure's guides from the stage-time panel name them. ax is the memory lane; the other two follow
    below it."""
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    pos = ax.get_position()
    x_mm, y_mm, w_mm, h_mm = pos.x0 * fw, pos.y0 * fh, pos.width * fw, pos.height * fh
    cpu_ax = fs.add_axes(fig, x_mm, y_mm - (h_mm + TRACE_LANE_GAP_MM), w_mm, h_mm)
    network_ax = fs.add_axes(fig, x_mm, y_mm - 2 * (h_mm + TRACE_LANE_GAP_MM), w_mm, h_mm)
    cpu_ax.set_gid("resource_trace_cpu")
    network_ax.set_gid("resource_trace_network")
    samples, rates = data["samples"], data["rates"]
    network = np.maximum(rates["networkMBps"], NETWORK_FLOOR_MBPS)
    lanes = (
        (ax, samples["minutes"], samples["memoryGiB"], "peak", "memory\nGiB"),
        (cpu_ax, rates["minutes"], rates["cpuCores"], "limit", "CPU\ncores"),
        (network_ax, rates["minutes"], network, "log", "network\nMB/s (log10)"),
    )
    edges = sorted({span["start"] for span in data["stages"][1:]})
    for lane, minutes, values, scale, name in lanes:
        for number, span in enumerate(data["stages"]):
            if number % 2 == 0:
                lane.axvspan(span["start"], span["end"], color=TRACE_BAND, lw=0, zorder=0)
        for edge in edges:
            lane.axvline(edge, zorder=1, **TRACE_EDGE)
        lane.plot(minutes, values, color=fs.INK, lw=fs.LW_THIN, zorder=3)
        if scale == "peak":
            # Memory starts at zero and marks its peak.
            peak = max(values)
            lane.set_ylim(0, peak * 1.15)
            lane.set_yticks([peak])
            lane.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
        elif scale == "limit":
            # CPU starts at zero and marks the container's limit.
            lane.axhline(data["cpuLimit"], color=fs.SLATE, lw=fs.LW_THIN, ls=":", zorder=2)
            lane.set_ylim(0, data["cpuLimit"] * 1.15)
            lane.set_yticks([data["cpuLimit"]])
            lane.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        else:
            # The network rate on a log axis from its floor, marked at the floor and the top decade it reaches.
            lane.set_yscale("log")
            lane.set_ylim(NETWORK_FLOOR_MBPS, max(network) * 2)
            lane.set_yticks([NETWORK_FLOOR_MBPS, 10.0 ** np.floor(np.log10(max(network)))])
            lane.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        lane.yaxis.set_minor_locator(NullLocator())
        # The small lanes carry their names and marks at the smallest type size.
        lane.set_ylabel(
            name, rotation=90, ha="right", va="bottom", fontsize=fs.NOTE, linespacing=1.0
        )
        lane.tick_params(axis="y", labelsize=fs.NOTE)
        lane.yaxis.set_label_coords(-TRACE_UNIT_MM / w_mm, 1.0)
        lane.set_xlim(0, samples["minutes"][-1])
        lane.tick_params(axis="x", labelbottom=lane is network_ax)
        if lane is not ax:
            fs.frame_axes(lane)
    network_ax.set_xlabel("minutes")


def stage_names(ax, names: list[str]) -> None:
    """Each stage's name under its group of bars, in one row, taken from the right: a name that would touch the name
    after it moves left by the least it takes, so every other name stays centred under its bars."""
    below = blended_transform_factory(ax.transData, ax.transAxes)
    texts = [
        ax.text(
            x,
            -STAGE_NAME_DROP_MM / height_mm(ax),
            name,
            transform=below,
            ha="center",
            va="top",
            fontsize=fs.NOTE,
            color=fs.INK,
            gid="stage-name",
        )
        for x, name in enumerate(names)
    ]
    renderer = ax.figure.canvas.get_renderer()
    low, high = ax.get_xlim()
    per_mm = (high - low) / width_mm(ax)
    halves = [
        t.get_window_extent(renderer).width / ax.figure.dpi * 25.4 * per_mm / 2 for t in texts
    ]
    gap = STAGE_NAME_GAP_MM * per_mm
    centres = [float(t.get_position()[0]) for t in texts]
    for i in range(len(centres) - 2, -1, -1):
        limit = centres[i + 1] - halves[i + 1] - gap - halves[i]
        centres[i] = min(centres[i], limit)
    for text, centre in zip(texts, centres):
        text.set_x(centre)


def guide_stages(fig) -> None:
    """A fan from the middle of each stage name under the stage-time panel down to the resource trace: two dotted
    guides to the times where the stage starts and ends, which the trace's own dotted lines carry on through its
    lanes, and between them the shade of the stage's band on the trace, for the stages the trace shades."""
    stage_ax = next(a for a in fig.axes if a.get_gid() == "stage_times")
    trace_ax = next(a for a in fig.axes if a.get_gid() == "resource_trace")
    names = [t for t in stage_ax.texts if t.get_gid() == "stage-name"]
    spans = json.loads((HERE / "resource_trace.json").read_text())["stages"]
    if [t.get_text() for t in names] != [span["stage"] for span in spans]:
        raise ValueError("the stage-time panel and the resource trace list different stages")
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    gap = 0.4 / (fig.get_figheight() / fs.MM)
    bottom = trace_ax.get_position().y1
    for number, (name, span) in enumerate(zip(names, spans)):
        (x0, y0), (x1, _) = to_figure.transform(name.get_window_extent(renderer).get_points())
        apex = ((x0 + x1) / 2, y0 - gap)
        start, end = (
            to_figure.transform(trace_ax.transData.transform((time, 1.0)))[0]
            for time in (span["start"], span["end"])
        )
        if number % 2 == 0:
            fig.add_artist(
                Polygon(
                    [apex, (start, bottom), (end, bottom)],
                    closed=True,
                    transform=fig.transFigure,
                    facecolor=TRACE_BAND,
                    lw=0,
                    zorder=0.5,
                )
            )
        for x in (start, end):
            fig.add_artist(
                Line2D([apex[0], x], [apex[1], bottom], transform=fig.transFigure, **TRACE_EDGE)
            )


# ---- panel h: memory decoupling across 1M, 10M, and 95.6M cells ---------------------------------


def step_memory(ax, data: dict) -> None:
    """Memory decoupling across 1M, 10M, and 95.6M cells on a log2 GiB scale: raw uncompressed CSR size
    grows from 20 GiB to 1,560 GiB, O(N) graph stages scale linearly with cell count, and streamed count
    passes (HVG, normalization, PCA) stay bounded under the memory planner."""
    scales = data["scales"]
    xs = np.arange(len(scales), dtype=float)

    csr_ys = [r["csrGiB"] for r in scales]
    graph_ys = [r["graphPeakGiB"] for r in scales]
    stream_ys = [r["streamedPeakGiB"] for r in scales]

    ax.plot(xs, csr_ys, color=fs.SLATE, lw=fs.LW_DATA, ls="--", zorder=2)
    for x, y in zip(xs, csr_ys):
        mean_point(ax, x, y, fs.SLATE, "s", open_face=True)

    ax.plot(xs, graph_ys, color=fs.BLUE, lw=fs.LW_PRIMARY, ls="-", zorder=3)
    for x, y in zip(xs, graph_ys):
        mean_point(ax, x, y, fs.BLUE, "D", open_face=True)

    ax.plot(xs, stream_ys, color=fs.BLUE, lw=fs.LW_PRIMARY, ls="-", zorder=4)
    for x, y in zip(xs, stream_ys):
        mean_point(ax, x, y, fs.BLUE, "o", open_face=False)

    ax.set_yscale("log", base=2)
    ax.set_xlim(-0.38, len(scales) - 0.62)
    # The floor leaves room for the whole 1M markers, whose graph peak is under 4 GiB.
    ax.set_ylim(2.5, 2400.0)
    log_axis(ax.yaxis, [8, 32, 128, 512, 2048], minor=[4, 16, 64, 256, 1024])
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in scales])
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("cells")
    ax.set_ylabel("memory (GiB, log2)")

    handles = [
        Line2D(
            [],
            [],
            color=fs.SLATE,
            lw=fs.LW_DATA,
            ls="--",
            marker="s",
            ms=marker_ms("s") * 0.9,
            mfc=fs.WHITE,
            mec=fs.SLATE,
            mew=fs.LW_MARKER,
        ),
        Line2D(
            [],
            [],
            color=fs.BLUE,
            lw=fs.LW_PRIMARY,
            ls="-",
            marker="D",
            ms=marker_ms("D") * 0.9,
            mfc=fs.WHITE,
            mec=fs.BLUE,
            mew=fs.LW_MARKER,
        ),
        Line2D(
            [],
            [],
            color=fs.BLUE,
            lw=fs.LW_PRIMARY,
            ls="-",
            marker="o",
            ms=marker_ms("o") * 0.9,
            mfc=fs.BLUE,
            mec=fs.INK,
            mew=fs.LW_MARKER,
        ),
    ]
    legend_above(
        ax,
        handles,
        ["raw CSR", "graph O(N)", "streamed"],
        ncol=2,
        left_mm=-11.5,
    )


# ---- panel i: memory budget compliance sweep ----------------------------------------------------


def budget_sweep(ax, data: dict) -> None:
    """Memory under a budget: the peak container memory of the one-million-cell runs from input to PCA under 4, 8,
    16 and 24 GiB (bars, three runs each), the budget itself, and on a second axis the minutes the runs took."""
    rows = data["rows"]
    xs = list(range(len(rows)))
    for x, row in enumerate(rows):
        peak = row["containerGiB"]
        ax.bar(x, peak["mean"], width=0.52, zorder=2, **CYTEARC_BAR)
        range_bar(ax, x, peak["low"], peak["high"], cap=0.12)
        ax.plot([x - 0.34, x + 0.34], [row["budgetGiB"], row["budgetGiB"]], zorder=5, **LIMIT_LINE)
    ax.set_yscale("log", base=2)
    ax.set_xlim(-0.58, len(rows) - 0.42)
    ax.set_ylim(2.2, 34.0)
    log_axis(ax.yaxis, [4, 8, 16, 32], minor=[])
    ax.set_xticks(xs)
    ax.set_xticklabels([str(row["budgetGiB"]) for row in rows])
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("budget (GiB)")
    ax.set_ylabel("memory (GiB, log2)")

    minutes = ax.twinx()
    minutes.set_gid("budget_sweep_minutes")
    minutes.plot(xs, [row["minutes"]["mean"] for row in rows], zorder=7, **TIME_LINE)
    # The minutes run above the bars' tops, clear of the budget marks.
    minutes.set_ylim(0.0, 6.6)
    minutes.set_yticks([0, 2, 4, 6])
    minutes.set_ylabel("min")
    # The second axis only adds the minutes scale on the right; the main axes draw the frame and the budgets.
    minutes.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False
    )
    for spine in minutes.spines.values():
        spine.set_visible(False)
    minutes.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)

    handles = [
        Patch(**CYTEARC_BAR),
        Line2D([], [], **LIMIT_LINE),
        Line2D([], [], **TIME_LINE),
    ]
    legend_above(ax, handles, ["peak", "budget", "time"], ncol=2, left_mm=-11.0)


# Panel h: the minutes of the runs, an ink line on its own axis.
TIME_LINE = {
    "color": fs.INK,
    "lw": fs.LW_DATA,
    "marker": "o",
    "ms": marker_ms("o") * 0.7,
    "mfc": fs.INK,
    "mec": fs.INK,
}


# ---- panel j: analysis core-time against cells, 100,000 to 10,000,000 -------------------------------

MEAN_DOT = 5.0  # scatter area of the mean in square points, about 0.8 mm across


def core_time_scaling(ax, data: dict) -> None:
    """CyteArc's analysis core-time on embryo samples of 100,000 to 10,000,000 cells and on the 95.6-million-cell
    Tahoe-100M atlas on log axes: for each size the range of its three runs as a capped bar with the mean as a small
    ink dot on it, and the power law fitted to the runs as a dotted line, with its exponent. The dot is small so that a range of a few percent stays visible. Both
    axes are log10, with the decades labelled and minor ticks between them."""
    sizes = data["sizes"]
    for size in sizes:
        cells, summary = size["cells"], size["coreMinutes"]
        ax.plot(
            [cells, cells], [summary["low"], summary["high"]], color=fs.INK, lw=fs.LW_AXIS, zorder=4
        )
        for value in (summary["low"], summary["high"]):
            ax.plot(
                [cells / 1.18, cells * 1.18], [value, value], color=fs.INK, lw=fs.LW_AXIS, zorder=4
            )
        ax.scatter([cells], [summary["mean"]], s=MEAN_DOT, color=fs.INK, lw=0, zorder=5)
    slope = data["slope"]
    ends = np.array([sizes[0]["cells"], sizes[-1]["cells"]], dtype=float)
    ax.plot(
        ends,
        10 ** (slope["intercept"] + slope["value"] * np.log10(ends)),
        color=fs.INK,
        lw=fs.LW_THIN,
        ls=":",
        zorder=2,
    )
    fs.note(ax, 0.05, 0.95, f"exponent {slope['value']:.2f}", va="top", ha="left", color=fs.INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # A short margin at each end leaves each decade room for its label, 10M and 100M included.
    ax.set_xlim(sizes[0]["cells"] / 1.4, sizes[-1]["cells"] * 1.6)
    means = [size["coreMinutes"]["mean"] for size in sizes]
    ax.set_ylim(min(means) / 2, max(means) * 3)
    names = {1e5: "100k", 1e6: "1M", 1e7: "10M", 1e8: "100M"}
    ax.xaxis.set_major_locator(FixedLocator(list(names)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: names[value]))
    # numticks keeps LogLocator from dropping the minor ticks of a short axis.
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    log_axis(ax.yaxis, [10, 100, 1000, 10000])
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("cells (log10)")
    ax.set_ylabel("core-minutes (log10)")


PANELS = {
    draw.__name__: draw
    for draw in (
        store_layout,
        analysis_time,
        peak_memory,
        stage_times,
        resource_trace,
        tahoe_stages,
        tahoe_umap,
        step_memory,
        budget_sweep,
        core_time_scaling,
    )
}
DATA_FILE: dict[str, str] = {}

# Short headers beside panel letters so readers immediately grasp what each panel shows.
HEADERS = {
    "store_layout": "Dual-orientation store and memory-bounded streaming",
    "analysis_time": "End-to-end runtime (1M–95.6M)",
    "peak_memory": "Peak container memory",
    "tahoe_umap": "Tahoe-100M",
    "stage_times": "10M-cell stage time",
    "tahoe_stages": "Tahoe-100M stages",
    "step_memory": "Memory vs. atlas scale",
    "budget_sweep": "Memory under a budget",
    "core_time_scaling": "Core-time vs. cells",
}


# ---- layout and rendering -----------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """A panel's printed size in millimetres, and its plot inside it as left, bottom, width and height from
    the cell's lower-left corner. The top-left corner stays free for the letter the figure adds."""

    width: float
    height: float
    plot: tuple[float, float, float, float]


CELLS = {
    "store_layout": Cell(160.0, 44.0, (0.0, 0.0, 160.0, 44.0)),
    "analysis_time": Cell(57.0, 49.6, (11.5, 1.0, 44.5, 33.0)),
    "peak_memory": Cell(57.0, 72.5, (11.5, 34.0, 44.5, 33.0)),
    "tahoe_umap": Cell(47.0, 45.0, (4.5, 4.6, 30.5, 35.4)),
    "tahoe_stages": Cell(47.0, 35.1, (11.5, 9.0, 16.0, 15.5)),
    "stage_times": Cell(56.0, 40.0, (11.0, 4.5, 44.5, 23.0)),
    "resource_trace": Cell(56.0, 40.1, (11.0, 26.5, 44.5, 7.6)),
    "step_memory": Cell(34.3, 42.0, (12.5, 9.0, 20.8, 20.0)),
    "budget_sweep": Cell(34.3, 42.0, (11.5, 9.0, 16.5, 20.0)),
    "core_time_scaling": Cell(34.4, 42.0, (12.0, 9.0, 19.4, 20.0)),
}
# Where each panel's cell sits: its lower-left corner, in millimetres from the figure's lower-left corner. Panels b
# and c share one column under panel a, c directly below b on the same plot width, so the two share the cells axis
# and the outcome matrix under c. To their right, above the bottom row, the 10M stage times sit over the resource
# trace, and to the right of those the Tahoe-100M stage charts sit over its UMAP, so the panels read down each
# column in the order the text first cites them. The resource trace's plot shares the stage-time plot's left and
# right edges, and its cell reaches up beside the guides between them to hold its letter clear of the lane names.
# The panels take their letters in this order.
LAYOUT = {
    "store_layout": (0.0, 122.1),
    "analysis_time": (0.0, 72.5),
    "peak_memory": (0.0, 0.0),
    "stage_times": (57.0, 82.1),
    "resource_trace": (57.0, 42.0),
    "tahoe_stages": (113.0, 87.0),
    "tahoe_umap": (113.0, 42.0),
    "step_memory": (57.0, 0.0),
    "budget_sweep": (91.3, 0.0),
    "core_time_scaling": (125.6, 0.0),
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
            if ax0 < bx1 - 1e-9 and bx0 < ax1 - 1e-9 and ay0 < by1 - 1e-9 and by0 < ay1 - 1e-9:
                raise ValueError(f"the cells of {a} and {b} overlap")


def accepted_exceptions() -> dict[str, str]:
    """The departures from the figure grammar this figure accepts, each with its reason."""
    names = [
        label["name"] for label in json.loads((HERE / "tahoe_umap.json").read_text())["labels"]
    ]
    return {
        **{
            f"type: '{name}' at {MAP_LABEL_PT:g} pt": "cell-line names small enough to cover little of the UMAP"
            for name in names
        },
        "colour: tahoe_cluster_purity": "the purity circles take the cell lines' UMAP colours",
    }


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
    fs.frame_axes(ax)
    return ax


def render_panel(name: str) -> None:
    cell = CELLS[name]
    fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
    draw_panel(fig, name, 0.0, 0.0)
    fs.save(fig, name, folder=BUILD / "panels", allow={**ALONE, **accepted_exceptions()})
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
        if name in HEADERS and name != "store_layout":
            fs.panel_title(fig, left + 4.8, top_y - 2.1, HEADERS[name])
    guide_stages(fig)
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
