"""Draw the supplementary figure on how joint RNA and protein graphs agree with author labels and with Seurat: one
function per panel, and the panels assembled with their letters.

Each panel function draws one panel into the plot it is given and is named for what the panel shows. The name is also
the panel's SVG group ID and the name of its data file, <name>.json, which prepare.py writes. CELLS gives each panel's
printed size, and LAYOUT places the panels, which sets their letters. Run from the repository root:

  uv run --locked python -B -m manuscript.figures.multimodal_graph_agreement.draw            # every panel and the figure
  uv run --locked python -B -m manuscript.figures.multimodal_graph_agreement.draw <panel>    # one panel

Renders go to .build/figures/multimodal_graph_agreement/: each panel alone and without a letter as panels/<panel>.png,
.pdf and .svg; the lettered figure as multimodal_graph_agreement.png, .pdf and .svg; and each panel's letter in
letters.json, which every run writes.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from manuscript.figures import style as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
BUILD = REPO / ".build" / "figures" / HERE.name
ALONE = {"canvas: width": "a panel alone is drawn at the size it prints at in the figure"}


# ---- shared helpers -----------------------------------------------------------------------------


def width_mm(ax) -> float:
    return ax.get_position().width * ax.figure.get_figwidth() / fs.MM


def height_mm(ax) -> float:
    return ax.get_position().height * ax.figure.get_figheight() / fs.MM


# A panel's legend sits in one row under its header: its top this far below the top of the cell, from this far right of
# the cell's left edge.
LEGEND_TOP_MM = 4.2
LEGEND_LEFT_MM = 1.5


def legend_row(ax, handles, labels) -> None:
    """A legend of lines in one row under the panel's header, from the cell's left edge."""
    cell = CELLS[ax.get_gid()]
    left, bottom, _, height = cell.plot
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(
            (LEGEND_LEFT_MM - left) / width_mm(ax),
            1 + (cell.height - bottom - height - LEGEND_TOP_MM) / height_mm(ax),
        ),
        borderaxespad=0.0,
        ncol=len(labels),
        columnspacing=0.9,
        handlelength=1.8,
        handletextpad=0.35,
    )


def category_ticks(ax, positions: list[float], names: list[str], *, labels: bool) -> None:
    """An x axis of named categories at fixed positions, with the names written only when labels is set."""
    ax.xaxis.set_major_locator(FixedLocator(positions))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda _, i: names[i] if labels else ""))
    ax.xaxis.set_minor_locator(NullLocator())


def share_axis(axis, top: float, ticks: list[float]) -> None:
    """A share from zero, labelled at ticks."""
    (axis.axes.set_ylim if axis.axis_name == "y" else axis.axes.set_xlim)(0, top)
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def sibling_axes(ax, left_mm: float, bottom_mm: float, w_mm: float, h_mm: float):
    """A second plot of the same panel, placed in millimetres from the lower-left corner of ax's plot."""
    pos = ax.get_position()
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    other = fs.add_axes(fig, pos.x0 * fw + left_mm, pos.y0 * fh + bottom_mm, w_mm, h_mm)
    other.set_gid(f"{ax.get_gid()}_{len(fig.axes)}")
    return other


def facets(ax, n: int, gap_mm: float) -> list:
    """The panel's plot and n - 1 more of its size in a row to its right, gap_mm apart."""
    out = [ax]
    for i in range(1, n):
        other = sibling_axes(ax, i * (width_mm(ax) + gap_mm), 0.0, width_mm(ax), height_mm(ax))
        fs.frame_axes(other)
        out.append(other)
    return out


def facet_title(ax, text: str) -> None:
    ax.text(
        0.0, 1.0 + 0.9 / height_mm(ax), text, transform=ax.transAxes, fontsize=fs.NOTE, va="bottom"
    )


def row_xlabel(row: list, text: str) -> None:
    """One x axis label centred under a row of facets."""
    first, last = row[0], row[-1]
    span = (last.get_position().x1 - first.get_position().x0) * first.figure.get_figwidth() / fs.MM
    first.set_xlabel(text, labelpad=1.5)
    first.xaxis.set_label_coords(span / 2 / width_mm(first), -5.4 / height_mm(first))


def note_in_legend_row(ax, panel, text: str) -> None:
    """A slate note in the row of panel's legend, right-aligned with the right edge of ax's plot."""
    cell = CELLS[panel.get_gid()]
    _, bottom, _, height = cell.plot
    # The legend's text is centred this far under the legend's top: its border pad and half a line of its type.
    centre_mm = LEGEND_TOP_MM + (0.2 + 0.5) * fs.TICK * 25.4 / 72
    ax.text(
        1.0,
        1.0 + (cell.height - bottom - height - centre_mm) / height_mm(ax),
        text,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=fs.NOTE,
        color=fs.SLATE,
    )


def grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)


EVEN = {"o": 1.0, "s": 0.9, "D": 0.8, "^": 1.25, "h": 1.05, "P": 1.05}


def marker_ms(marker: str) -> float:
    return fs.MARKER_MS * EVEN.get(marker, 1.0)


# ---- the graphs, as every panel draws them ------------------------------------------------------------------------

BLUE = fs.ENTITIES["cytearc"].colour
ORANGE = fs.ENTITIES["seurat"].colour


@dataclass(frozen=True)
class Series:
    """How one graph is drawn: its name, colour and marker, and its line. CyteArc's SNN fusion graph is a variant of
    its WNN graph, drawn hollow and dashed; the single-modality graphs, a baseline, are slate."""

    name: str
    colour: str
    marker: str
    hollow: bool = False
    ls: str = "-"
    lw: float = fs.LW_DATA


SERIES = {
    "cytearc_wnn": Series("CyteArc WNN", BLUE, "o", lw=fs.LW_PRIMARY),
    "seurat_wnn": Series("Seurat WNN", ORANGE, "^"),
    "cytearc_snn": Series("CyteArc SNN fusion", BLUE, "o", hollow=True, ls="--"),
    "rna": Series("RNA only", fs.SLATE, "s"),
    "adt": Series("ADT only", fs.SLATE, "D", hollow=True),
}
# Drawn from the back: the baselines first, CyteArc's WNN graph last and on top.
DRAW_ORDER = ("adt", "rna", "seurat_wnn", "cytearc_snn", "cytearc_wnn")
LEGEND_ORDER = ("cytearc_wnn", "seurat_wnn", "cytearc_snn", "rna", "adt")
# The graphs side by side at each size, so that close values stay apart.
DODGE = {"cytearc_wnn": -0.2, "cytearc_snn": -0.1, "seurat_wnn": 0.0, "rna": 0.1, "adt": 0.2}


def point(ax, x, y, key: str, *, z: float = 4, hollow: bool | None = None) -> None:
    s = SERIES[key]
    hollow = s.hollow if hollow is None else hollow
    ax.plot(
        [x],
        [y],
        ls="none",
        marker=s.marker,
        ms=marker_ms(s.marker),
        mfc=fs.WHITE if hollow else s.colour,
        mec=s.colour if hollow else fs.INK,
        mew=fs.LW_MARKER,
        zorder=z,
    )


def handle(key: str) -> Line2D:
    s = SERIES[key]
    return Line2D(
        [],
        [],
        color=s.colour,
        lw=s.lw,
        ls=s.ls,
        marker=s.marker,
        ms=marker_ms(s.marker),
        mfc=fs.WHITE if s.hollow else s.colour,
        mec=s.colour if s.hollow else fs.INK,
        mew=fs.LW_MARKER,
    )


def series(ax, xs, ys, key: str, *, lows=None, highs=None, z: float = 3) -> None:
    """One graph as a line through its points, with each point's range over Leiden seeds when lows and highs are
    given."""
    s = SERIES[key]
    if len(xs) > 1:
        ax.plot(xs, ys, color=s.colour, lw=s.lw, ls=s.ls, zorder=z)
    if lows is not None:
        for x, low, high in zip(xs, lows, highs, strict=True):
            ax.plot([x, x], [low, high], color=s.colour, lw=fs.LW_THIN, zorder=z)
    for x, y in zip(xs, ys, strict=True):
        point(ax, x, y, key, z=z + 1)


def graph_legend(ax) -> None:
    legend_row(ax, [handle(k) for k in LEGEND_ORDER], [SERIES[k].name for k in LEGEND_ORDER])


# ---- the COMBAT and Stephenson CITE-seq series ---------------------------------------------------------------------

CITE_X = {100000: 0.0, 300000: 1.0, 600000: 2.0, 1000000: 3.0, 1457527: 4.0}
CITE_TICKS = ("0.1", "0.3", "0.6", "1.0", "1.46")
CITE_AXIS = "COMBAT and Stephenson cells (M)"
# Each metric of the agreement panel, with its facet title.
METRICS = {
    "ari": "ARI",
    "ari_combat": "ARI in COMBAT",
    "ari_stephenson": "ARI in Stephenson",
    "nmi": "NMI",
    "mean_f1_all_types": "F1, all types",
    "mean_f1_protein_defined_types": "F1, protein types",
}


def cite_agreement_by_size(ax, data: dict) -> None:
    """At each size, each graph's agreement with the author labels when cut to the authors' number of cell types:
    ARI over all cells and within each study's cells, NMI, and the mean F1 over all types and over the ten types the
    authors defined by surface protein; the mean of three Leiden seeds with their range."""
    row = facets(ax, len(METRICS), 2.6)
    sizes = data["sizes"]
    for facet, (metric, title) in zip(row, METRICS.items(), strict=True):
        share_axis(facet.yaxis, 0.82, [0, 0.2, 0.4, 0.6, 0.8])
        facet.set_xlim(-0.55, 4.55)
        category_ticks(facet, list(CITE_X.values()), list(CITE_TICKS), labels=True)
        facet.tick_params(axis="x", labelsize=fs.NOTE)
        grid(facet)
        for key in DRAW_ORDER:
            held = [s for s in sizes if key in s["graphs"]]
            values = [s["graphs"][key][metric] for s in held]
            series(
                facet,
                [CITE_X[s["cells"]] + DODGE[key] for s in held],
                [v["mean"] for v in values],
                key,
                lows=[v["low"] for v in values],
                highs=[v["high"] for v in values],
            )
        facet_title(facet, title)
        if facet is not ax:
            facet.tick_params(axis="y", labelleft=False)
    ax.set_ylabel("score", labelpad=1.5)
    row_xlabel(row, CITE_AXIS)
    graph_legend(ax)
    note_in_legend_row(row[-1], ax, f"{data['cellTypes']} clusters, mean and range of three seeds")


PANELS = {draw.__name__: draw for draw in (cite_agreement_by_size,)}
DATA_FILE: dict[str, str] = {}
# Short headers beside panel letters, naming what each panel measures.
HEADERS = {"cite_agreement_by_size": "Agreement with author labels"}
HEADER_INDENT_MM = 4.8


# ---- layout and rendering -----------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """A panel's printed size in millimetres, and its plot inside it as left, bottom, width and height from
    the cell's lower-left corner. The top-left corner stays free for the letter the figure adds."""

    width: float
    height: float
    plot: tuple[float, float, float, float]


CELLS = {
    "cite_agreement_by_size": Cell(160.0, 40.0, (10.0, 9.0, 22.5, 19.5)),
}
# Where each panel's cell sits: its lower-left corner, in millimetres from the figure's lower-left corner. One row: the
# COMBAT and Stephenson series, every metric at the authors' number of types by size. The panels take their letters in
# this order.
LAYOUT = {
    "cite_agreement_by_size": (0.0, 0.0),
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
        if name in HEADERS:
            fs.panel_title(fig, left + HEADER_INDENT_MM, top_y - 2.1, HEADERS[name])
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
