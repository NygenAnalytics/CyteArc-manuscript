"""Draw the supplementary figure on count-layout scaling: range reads per pass on each layout and the exponents of pass
time on each layout.

Run from the repository root:

  uv run --locked python -B -m manuscript.figures.layout_scaling.draw            # every panel and the figure
  uv run --locked python -B -m manuscript.figures.layout_scaling.draw <panel>    # one panel
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from manuscript.figures import style as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
BUILD = REPO / ".build" / "figures" / HERE.name
ALONE = {"canvas: width": "a panel alone is drawn at the size it prints at in the figure"}


def width_mm(ax) -> float:
    return ax.get_position().width * ax.figure.get_figwidth() / fs.MM


def height_mm(ax) -> float:
    return ax.get_position().height * ax.figure.get_figheight() / fs.MM


def plain(value, _=None) -> str:
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


# A legend sits in one row under the panel headers: its top this far below the top of the cell, from this far right of
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


def log_axis(axis, ticks: list[float], minor: list[float] | None = None) -> None:
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(
        FixedLocator(minor) if minor else LogLocator(base=10, subs=np.arange(2, 10))
    )
    axis.set_major_formatter(FuncFormatter(plain))
    axis.set_minor_formatter(NullFormatter())


def range_bar_across(ax, y: float, low: float, high: float, cap: float = 0.08) -> None:
    ax.plot([low, high], [y, y], color=fs.INK, lw=fs.LW_AXIS, solid_capstyle="butt", zorder=5)
    for x in (low, high):
        ax.plot(
            [x, x], [y - cap, y + cap], color=fs.INK, lw=fs.LW_AXIS, solid_capstyle="butt", zorder=5
        )


# Both layouts are CyteArc's, in its blue: the scaled layout, the default, as a filled circle on a solid line; the
# fixed layout, its variant, as an open circle on a dashed line. Both layout panels draw these marks, and one legend
# names them.
LAYOUTS = (("fixed", "fixed layout", True), ("scaled", "scaled layout", False))


def mark_style(variant: bool) -> dict:
    return {
        "marker": "o",
        "ms": fs.MARKER_MS,
        "mfc": fs.WHITE if variant else fs.BLUE,
        "mec": fs.BLUE if variant else fs.INK,
        "mew": fs.LW_MARKER,
    }


def line_style(variant: bool) -> dict:
    return {
        "color": fs.BLUE,
        "lw": fs.LW_DATA if variant else fs.LW_PRIMARY,
        "ls": "--" if variant else "-",
    }


def mean_point(ax, x: float, y: float, *, variant: bool) -> None:
    ax.plot([x], [y], ls="none", zorder=4, **mark_style(variant))


def layout_legend(ax) -> None:
    """The one legend of the two layout panels."""
    legend_row(
        ax,
        [Line2D([], [], **line_style(variant), **mark_style(variant)) for _, _, variant in LAYOUTS],
        [label for _, label, _ in LAYOUTS],
    )


def read_requests(ax, data: dict) -> None:
    """Range reads of the variable-gene pass on the fixed and the scaled count layouts at five and ten
    million cells, with how many times as many the doubled cell count needed on each layout."""
    reads = {(row["layout"], row["cells"]): row["rangeReads"] / 1000 for row in data["rows"]}
    sizes = sorted({row["cells"] for row in data["rows"]})
    xs = list(range(len(sizes)))
    for layout, _, variant in LAYOUTS:
        ys = [reads[(layout, cells)] for cells in sizes]
        ax.plot(xs, ys, zorder=2, **line_style(variant))
        for x, y in zip(xs, ys):
            mean_point(ax, x, y, variant=variant)
        fs.note(
            ax,
            xs[-1] + 0.12,
            ys[-1],
            f"×{ys[-1] / ys[0]:.1f}",
            transform=ax.transData,
            va="center",
            color=fs.INK,
        )
    ax.set_yscale("log")
    ax.set_ylim(3.5, 140)
    log_axis(ax.yaxis, [5, 10, 20, 50, 100])
    ax.set_xlim(-0.35, xs[-1] + 0.65)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{cells / 1e6:g}M" for cells in sizes])
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("cells")
    ax.set_ylabel("range reads\n(thousands, log10)", linespacing=1.05)
    layout_legend(ax)


def pass_exponents(ax, data: dict) -> None:
    """The exponent b in time ∝ cells^b of each count pass between five and ten million cells on the fixed and the
    scaled layouts: b from the mean times as a marker, its range over pairs of runs as an error bar, and linear
    scaling as a dashed line."""
    names = list(dict.fromkeys(row["name"] for row in data["rows"]))
    top = len(names) - 1
    ax.axvline(1.0, color=fs.INK, lw=fs.LW_THIN, ls=(0, (4, 2)), zorder=1)
    for row in data["rows"]:
        variant = row["layout"] == "fixed"
        y = top - names.index(row["name"]) + (0.17 if variant else -0.17)
        range_bar_across(ax, y, row["low"], row["high"], cap=0.09)
        mean_point(ax, row["exponent"], y, variant=variant)
    fs.note(ax, 1.015, top + 0.42, "linear", transform=ax.transData, va="center", color=fs.INK)
    ax.set_ylim(-0.55, top + 0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=fs.NOTE)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.75, 1.65)
    ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("exponent $b$ (time $\\propto$ cells$^{b}$)")


PANELS = {
    draw.__name__: draw
    for draw in (
        read_requests,
        pass_exponents,
    )
}


@dataclass(frozen=True)
class Cell:
    width: float
    height: float
    plot: tuple[float, float, float, float]


# The plots of the row share a frame: their bottom and top edges, in millimetres from the bottom of their cells. The top
# leaves room for the letter, the header and the legend under it.
ROW_HEIGHT_MM = 46.0
ROW_FRAME_MM = (9.0, 37.5)
FRAME_HEIGHT_MM = ROW_FRAME_MM[1] - ROW_FRAME_MM[0]
CELLS = {
    "read_requests": Cell(72.0, ROW_HEIGHT_MM, (16.0, ROW_FRAME_MM[0], 50.0, FRAME_HEIGHT_MM)),
    "pass_exponents": Cell(88.0, ROW_HEIGHT_MM, (24.0, ROW_FRAME_MM[0], 58.0, FRAME_HEIGHT_MM)),
}
LAYOUT = (("read_requests", "pass_exponents"),)
# Short headers beside the panel letters, naming what each panel measures.
HEADERS = {
    "read_requests": "Range reads per pass",
    "pass_exponents": "Pass time scaling",
}
HEADER_INDENT_MM = 4.8


def check_layout() -> None:
    placed = [name for row in LAYOUT for name in row]
    if (
        sorted(placed) != sorted(PANELS)
        or sorted(CELLS) != sorted(PANELS)
        or sorted(HEADERS) != sorted(PANELS)
    ):
        raise ValueError(
            f"LAYOUT places {placed}, CELLS sizes {sorted(CELLS)} and HEADERS names {sorted(HEADERS)}; each must hold "
            f"each panel of {sorted(PANELS)} once"
        )
    for row in LAYOUT:
        width = sum(CELLS[name].width for name in row)
        if width > 160:
            raise ValueError(f"the row {row} is {width} mm wide; a row fits in the figure's 160 mm")


def letters() -> dict[str, str]:
    return {name: chr(97 + i) for i, name in enumerate(name for row in LAYOUT for name in row)}


def write_letters() -> Path:
    path = BUILD / "letters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"figure": HERE.name, "letters": letters()}, indent=1) + "\n")
    return path


def draw_panel(fig, name: str, left: float, bottom: float):
    x, y, w, h = CELLS[name].plot
    ax = fs.add_axes(fig, left + x, bottom + y, w, h)
    ax.set_gid(name)
    PANELS[name](ax, json.loads((HERE / f"{name}.json").read_text()))
    fs.frame_axes(ax)
    return ax


def render_panel(name: str) -> None:
    cell = CELLS[name]
    fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
    draw_panel(fig, name, 0.0, 0.0)
    fs.save(fig, name, folder=BUILD / "panels", allow=ALONE)
    plt.close(fig)


def render_figure() -> None:
    heights = [max(CELLS[name].height for name in row) for row in LAYOUT]
    total = sum(heights)
    fig = plt.figure(figsize=(fs.FULL_WIDTH, total * fs.MM))
    marks = letters()
    top = total
    for row, height in zip(LAYOUT, heights):
        left = 0.0
        for name in row:
            draw_panel(fig, name, left, top - CELLS[name].height)
            fs.panel_label(fig, left + 1.0, top - 1.0, marks[name])
            fs.panel_title(fig, left + HEADER_INDENT_MM, top - 3.1, HEADERS[name])
            left += CELLS[name].width
        top -= height
    fs.save(fig, HERE.name, folder=BUILD)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    names = sys.argv[1:] if argv is None else argv
    unknown = [name for name in names if name not in PANELS]
    if unknown:
        raise SystemExit(f"no panel named {', '.join(unknown)}; the panels are {', '.join(PANELS)}")
    check_layout()
    fs.apply()
    print(f"wrote {write_letters().relative_to(REPO)}")
    for name in names or PANELS:
        render_panel(name)
    if not names:
        render_figure()


if __name__ == "__main__":
    main()
