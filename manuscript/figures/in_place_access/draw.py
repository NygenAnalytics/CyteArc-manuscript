"""Draw the supplementary figure on using stores in place: one function per panel, and the panels assembled with their
letters.

Each panel function draws one panel into the plot it is given and is named for what the panel shows. The name is also
the panel's SVG group ID and the name of its data file, <name>.json, which prepare.py writes.
CELLS gives each panel's printed size, and LAYOUT places the panels, which sets their letters. Run from the repository
root:

  uv run --locked python -B -m manuscript.figures.in_place_access.draw            # every panel and the figure
  uv run --locked python -B -m manuscript.figures.in_place_access.draw <panel>    # one panel

Renders go to .build/figures/in_place_access/: each panel alone and without a letter as panels/<panel>.png, .pdf and
.svg; the lettered figure as in_place_access.png, .pdf and .svg; and each panel's letter in letters.json, which every
run writes.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

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


def plain(value, _=None) -> str:
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


# A panel's legend sits in one row under its header: its top this far below the top of the cell, from this far right of
# the cell's left edge.
LEGEND_TOP_MM = 4.2
LEGEND_LEFT_MM = 1.5


def legend_row(ax, handles, labels) -> None:
    """The panel's legend in one row under its header, from the cell's left edge."""
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
        handlelength=0.9,
        handletextpad=0.35,
    )


# A panel's header starts this far from its cell's left edge, after the letter.
HEADER_INDENT_MM = 4.8
# The plots of the row share a frame: their bottom and top edges, in millimetres from the bottom of their cells. The top
# leaves room for the letter, the header and a legend under it.
ROW_FRAME_MM = (9.0, 37.5)


# Every CyteArc bar: the 8 GiB step of the budget ramp, as in the analysis-scaling figure; a tool outside the registry,
# such as AnnData, or a download, is slate.
CYTEARC_BAR = {"facecolor": fs.BUDGET_COLOURS[8], "edgecolor": fs.INK, "linewidth": fs.LW_BAR}
COMPARATOR_BAR = fs.bar_style(fs.COMPARATOR)


# ---- the in-place screen of the human blood stores, against downloading source files ---------------------

# Stores read in place take CyteArc's bar colour, as the query's bars do; downloads of source H5AD files are slate
# squares, as the download's bars are.
SCREEN_POINT = CYTEARC_BAR["facecolor"]
DOWNLOAD_POINT = COMPARATOR_BAR["facecolor"]
# The marks' areas in square points.
SCREEN_POINT_AREA = 7.0
DOWNLOAD_POINT_AREA = 13.0
SCREEN_CELLS = (1e2, 2e7)
SCREEN_LANE_GAP_MM = 2.0
SCREEN_LEGEND = ("CyteBase", "H5AD download")
CELL_TICKS = {1e3: "1k", 1e5: "100k", 1e7: "10M"}
SCREEN_LANES = (
    (
        "seconds",
        "seconds",
        (1.0, 3000.0),
        {1.0: "1", 10.0: "10", 100.0: "100", 1000.0: "1,000"},
        "seconds\n(log10)",
    ),
    (
        "responseBytes",
        "bytes",
        (3e5, 3e11),
        {1e6: "1 MB", 1e8: "100 MB", 1e10: "10 GB"},
        "bytes\n(log10)",
    ),
)


def blood_screen(ax, data: dict) -> None:
    """For each store the screen read without waiting on Hugging Face's request quota, by its cells on a log10 axis,
    the seconds the screen took for it and the bytes it received, in two lanes on log10 axes; among them, in the same
    units, the measured downloads of three whole source H5AD files. ax is the seconds lane; the bytes lane follows below
    it."""
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    pos = ax.get_position()
    x_mm, y_mm, w_mm, h_mm = pos.x0 * fw, pos.y0 * fh, pos.width * fw, pos.height * fh
    bytes_ax = fs.add_axes(fig, x_mm, y_mm - h_mm - SCREEN_LANE_GAP_MM, w_mm, h_mm)
    bytes_ax.set_gid("blood_screen_bytes")
    stores, downloads = data["stores"], data["downloads"]
    for lane, (store_field, field, limits, ticks, name) in zip((ax, bytes_ax), SCREEN_LANES):
        lane.scatter(
            [s["cells"] for s in stores],
            [s[store_field] for s in stores],
            s=SCREEN_POINT_AREA,
            facecolor=SCREEN_POINT,
            edgecolor=fs.WHITE,
            linewidths=0.4,
            zorder=4,
        )
        lane.scatter(
            [d["cells"] for d in downloads],
            [d[field] for d in downloads],
            marker="s",
            s=DOWNLOAD_POINT_AREA,
            facecolor=DOWNLOAD_POINT,
            edgecolor=fs.INK,
            linewidths=fs.LW_MARKER,
            zorder=5,
        )
        lane.set_xscale("log")
        lane.set_yscale("log")
        lane.set_xlim(*SCREEN_CELLS)
        lane.set_ylim(*limits)
        decade_axis(lane.yaxis, limits, ticks)
        lane.xaxis.set_major_locator(FixedLocator(list(CELL_TICKS)))
        lane.xaxis.set_major_formatter(FuncFormatter(lambda value, _: CELL_TICKS[value]))
        lane.xaxis.set_minor_locator(NullLocator())
        lane.grid(True, which="major")
        lane.set_axisbelow(True)
        lane.set_ylabel(name, labelpad=1.5)
        fs.frame_axes(lane)
    ax.tick_params(axis="x", labelbottom=False)
    bytes_ax.set_xlabel("cells (log10)")
    # The legend's marks are the plotted ones: a scatter's area in square points is a marker's size squared.
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=SCREEN_POINT_AREA**0.5,
            mfc=SCREEN_POINT,
            mec=fs.WHITE,
            mew=0.4,
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="s",
            ms=DOWNLOAD_POINT_AREA**0.5,
            mfc=DOWNLOAD_POINT,
            mec=fs.INK,
            mew=fs.LW_MARKER,
        ),
    ]
    # The stores read in place and the downloaded files carry the names the query panel gives them.
    legend_row(ax, handles, list(SCREEN_LEGEND))


# ---- every store of the release, by its cells and its nonzero counts -------------------------------------------

STORE_CELLS = (1e2, 3e7)
STORE_NONZEROS = (1e5, 3e10)
# Every decade takes a tick, with minor ticks between; these decades carry labels.
CELL_LABELS = {1e2: "100", 1e4: "10k", 1e6: "1M"}
NONZERO_LABELS = {1e6: "1M", 1e8: "100M", 1e10: "10B"}


def decade_axis(axis, limits: tuple[float, float], labels: dict[float, str]) -> None:
    """A log10 axis with a tick at every decade, labels at the given ones, and minor ticks between decades."""
    low, high = (int(np.floor(np.log10(v))) for v in limits)
    decades = [10.0**k for k in range(low, high + 1)]
    axis.set_major_locator(FixedLocator(decades))
    axis.set_major_formatter(FuncFormatter(lambda value, _: labels.get(value, "")))
    axis.set_minor_locator(
        FixedLocator([d * m for d in decades for m in range(2, 10) if d * m <= limits[1]])
    )
    axis.set_minor_formatter(NullFormatter())


def release_storage(ax, data: dict) -> None:
    """Every store of the CyteBase release by its cells and its estimated nonzero counts, both on log10 axes; the
    header gives the release's cells."""
    stores = data["stores"]
    ax.scatter(
        [s["cells"] for s in stores],
        [s["nonzeros"] for s in stores],
        s=2.2,
        facecolor=fs.tint(CYTEARC_BAR["facecolor"], 0.7),
        edgecolor="none",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*STORE_CELLS)
    ax.set_ylim(*STORE_NONZEROS)
    decade_axis(ax.xaxis, STORE_CELLS, CELL_LABELS)
    decade_axis(ax.yaxis, STORE_NONZEROS, NONZERO_LABELS)
    ax.grid(False)
    ax.set_xlabel("observations per store (log10)")
    ax.set_ylabel("nonzeros (log10)", labelpad=1.5)


PANELS = {
    draw.__name__: draw
    for draw in (
        release_storage,
        blood_screen,
    )
}
DATA_FILE: dict[str, str] = {}

# Short headers beside panel letters, for the panels whose subject their axes and legend do not name.
HEADERS = {
    "release_storage": "CyteBase (178M observations)",
    "blood_screen": "Blood store screen",
}


# ---- layout and rendering -----------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """A panel's printed size in millimetres, and its plot inside it as left, bottom, width and height from
    the cell's lower-left corner. The top-left corner stays free for the letter the figure adds."""

    width: float
    height: float
    plot: tuple[float, float, float, float]


ROW_HEIGHT_MM = 46.0
FRAME_HEIGHT_MM = ROW_FRAME_MM[1] - ROW_FRAME_MM[0]
# The screen's two lanes fill the row's frame: the seconds lane on top, the bytes lane SCREEN_LANE_GAP_MM under it.
SCREEN_LANE_MM = (FRAME_HEIGHT_MM - SCREEN_LANE_GAP_MM) / 2
CELLS = {
    "release_storage": Cell(56.0, ROW_HEIGHT_MM, (12.5, ROW_FRAME_MM[0], 39.0, FRAME_HEIGHT_MM)),
    "blood_screen": Cell(
        104.0,
        ROW_HEIGHT_MM,
        (17.0, ROW_FRAME_MM[1] - SCREEN_LANE_MM, 84.0, SCREEN_LANE_MM),
    ),
}
# Where each panel's cell sits: its lower-left corner, in millimetres from the figure's lower-left corner; the two
# panels share one row, and take their letters in this order.
LAYOUT = {
    "release_storage": (0.0, 0.0),
    "blood_screen": (56.0, 0.0),
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
