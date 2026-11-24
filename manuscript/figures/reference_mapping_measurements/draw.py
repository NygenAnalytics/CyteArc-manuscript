"""Draw the supplementary figure of reference mapping measurements: one function per panel, and the panels assembled
with their letters.

Each panel function draws one panel into the plot it is given and is named for what the panel shows. The name is also
the panel's SVG group ID and the name of its data file, <name>.json, which prepare.py writes.
CELLS gives each panel's printed size, and LAYOUT places the panels, which sets their letters. Run from the repository
root:

  uv run --locked python -B -m manuscript.figures.reference_mapping_measurements.draw            # every panel and the figure
  uv run --locked python -B -m manuscript.figures.reference_mapping_measurements.draw <panel>    # one panel

Renders go to .build/figures/reference_mapping_measurements/: each panel alone and without a letter as
panels/<panel>.png, .pdf and .svg; the lettered figure as reference_mapping_measurements.png, .pdf and .svg; and each
panel's letter in letters.json, which every run writes.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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


def log_axis(axis, ticks: list[float], labels: dict[float, str] | None = None) -> None:
    """A log axis labelled at ticks, with plain numbers unless labels names them, and no minor ticks."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    if labels is None:
        axis.set_major_formatter(FuncFormatter(plain))
    else:
        axis.set_major_formatter(FuncFormatter(lambda value, _: labels.get(value, "")))
    axis.set_minor_formatter(NullFormatter())


def linear_axis(axis, ticks) -> None:
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def category_ticks(axis, positions: list[float], names: list[str]) -> None:
    """An axis of named categories at fixed positions."""
    axis.set_major_locator(FixedLocator(positions))
    axis.set_major_formatter(FuncFormatter(lambda _, i: names[i] if 0 <= i < len(names) else ""))
    axis.set_minor_locator(NullLocator())


def sibling_axes(ax, left_mm: float, bottom_mm: float, w_mm: float, h_mm: float):
    """A second plot of the same panel, placed in millimetres from the lower-left corner of ax's plot."""
    pos = ax.get_position()
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    other = fs.add_axes(fig, pos.x0 * fw + left_mm, pos.y0 * fh + bottom_mm, w_mm, h_mm)
    other.set_gid(f"{ax.get_gid()}_{len(fig.axes)}")
    return other


def value_axis(ax) -> None:
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


# ---- methods --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Method:
    name: str
    entity: str
    marker: str
    # A variant of a tool keeps its colour and marker and is drawn open, its lines dashed.
    variant: bool = False

    @property
    def colour(self) -> str:
        return fs.ENTITIES[self.entity].colour


METHODS = {
    "seurat_map": Method("Seurat", "seurat", "^"),
    "symphony": Method("Symphony", "symphony", "h"),
    "celltypist_hvg": Method("CellTypist, variable genes", "celltypist", "P"),
    "celltypist": Method("CellTypist, all genes", "celltypist", "P", variant=True),
    "unintegrated": Method("CyteArc", "cytearc", "o"),
    "integrated": Method("CyteArc, integrated", "cytearc", "o", variant=True),
}
# The order the panels draw the methods in: the comparators, then CyteArc, drawn last so that it stays on top; each
# tool's variant beside it.
LADDER_ORDER = (
    "seurat_map",
    "symphony",
    "celltypist_hvg",
    "celltypist",
    "unintegrated",
    "integrated",
)
# The order the legend lists the methods in: the tools in the order of the style's registry, each with its variant after
# it.
LEGEND_ORDER = (
    "unintegrated",
    "integrated",
    "seurat_map",
    "celltypist_hvg",
    "celltypist",
    "symphony",
)
for order in (LADDER_ORDER, LEGEND_ORDER):
    if sorted(order) != sorted(METHODS):
        raise ValueError("each method order must list each method of METHODS once")

EVEN = {"o": 1.0, "s": 0.9, "D": 0.8, "^": 1.25, "h": 1.05, "P": 1.05}


def marker_ms(marker: str) -> float:
    return fs.MARKER_MS * EVEN.get(marker, 1.0)


def marker_style(key: str) -> dict:
    m = METHODS[key]
    return {
        "marker": m.marker,
        "ms": marker_ms(m.marker),
        "mfc": fs.WHITE if m.variant else m.colour,
        "mec": m.colour if m.variant else fs.INK,
        "mew": fs.LW_MARKER,
    }


def line_style(key: str) -> dict:
    m = METHODS[key]
    primary = m.entity == "cytearc" and not m.variant
    return {
        "color": m.colour,
        "lw": fs.LW_PRIMARY if primary else fs.LW_DATA,
        "ls": "--" if m.variant else "-",
    }


def method_point(ax, key: str, x, y, *, z=4):
    ax.plot([x], [y], ls="none", zorder=z, **marker_style(key))


def method_handle(key: str) -> Line2D:
    return Line2D([], [], **line_style(key), **marker_style(key))


def method_line(ax, key: str, xs, ys, *, z=3):
    if len(xs) > 1:
        ax.plot(xs, ys, zorder=z, **line_style(key))
    for x, y in zip(xs, ys):
        method_point(ax, key, x, y, z=z + 1)


# The legend of every method on the page runs in one row across the top of the figure, above the panel letters: its top
# this far below the figure's top edge, from this far right of its left edge. The panels' cells start LEGEND_BAND_MM
# below the top.
LEGEND_TOP_MM = 1.0
LEGEND_LEFT_MM = 1.5
LEGEND_BAND_MM = 5.0


def method_legend(fig) -> None:
    """The legend of every method the figure draws, in one row across its top."""
    width, height = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    fig.legend(
        [method_handle(k) for k in LEGEND_ORDER],
        [METHODS[k].name for k in LEGEND_ORDER],
        loc="upper left",
        bbox_to_anchor=(LEGEND_LEFT_MM / width, 1 - LEGEND_TOP_MM / height),
        borderaxespad=0.0,
        ncol=len(LEGEND_ORDER),
        columnspacing=0.9,
        handlelength=1.8,
        handletextpad=0.35,
    )


# ---- registering a reference and what each method saves -----------------------------------------------------------

ARM_OFFSET = 0.22
# The registration plots: the gap between the bytes read and the seconds, and the seconds plot's width.
REGISTRATION_GAP_MM = 4.0
REGISTRATION_TIME_W_MM = 20.0


def registration_reads(ax, data: dict) -> None:
    """Stored analysis artifacts read while registering each finished CyteArc analysis as a mapping reference, and the
    time registration took, for the held-out blood, kidney and lung references, one point per arm; no registration read
    a count byte."""
    references = list(dict.fromkeys(row["reference"] for row in data["rows"]))
    centre = {name: float(len(references) - 1 - i) for i, name in enumerate(references)}
    offset = {"unintegrated": ARM_OFFSET, "integrated": -ARM_OFFSET}
    if any(row["countReadBytes"] for row in data["rows"]):
        raise ValueError("the panel states that no registration read a count byte")
    time = sibling_axes(
        ax, width_mm(ax) + REGISTRATION_GAP_MM, 0.0, REGISTRATION_TIME_W_MM, height_mm(ax)
    )
    for plot in (ax, time):
        plot.set_ylim(-0.6, len(references) - 0.4)
        plot.grid(True, axis="x")
        plot.set_axisbelow(True)
        plot.tick_params(axis="y", length=0)
    for row in data["rows"]:
        y = centre[row["reference"]] + offset[row["arm"]]
        method_point(ax, row["arm"], row["readBytes"] / 1e9, y)
        method_point(time, row["arm"], row["seconds"], y)
    ax.set_xlim(0, 1.0)
    linear_axis(ax.xaxis, [0, 0.5, 1.0])
    ax.set_yticks([centre[r] for r in references])
    ax.set_yticklabels(references, fontsize=fs.NOTE)
    ax.set_xlabel("artifacts read (GB)", labelpad=1.5)
    # The note sits under the x label, its top this far below the plot.
    ax.text(
        0.0,
        -7.3 / height_mm(ax),
        "count bytes read: 0",
        transform=ax.transAxes,
        fontsize=fs.NOTE,
        color=fs.SLATE,
        va="top",
    )
    time.set_xlim(0, 3.5)
    linear_axis(time.xaxis, [0, 1, 2, 3])
    time.set_yticks([centre[r] for r in references])
    time.tick_params(axis="y", labelleft=False)
    time.set_xlabel("seconds", labelpad=1.5)
    fs.frame_axes(time)


# Byte ticks of the saved references, in megabytes.
SAVED_TICKS = {
    0.1: "0.1 MB",
    1.0: "1 MB",
    10.0: "10 MB",
    100.0: "100 MB",
    1e3: "1 GB",
    1e4: "10 GB",
    1e5: "100 GB",
}
# Within a reference, each method's offset: the comparators' saved files differ from CyteArc's by orders of magnitude,
# but CellTypist's models lie next to CyteArc's descriptors, so the four sit apart, about 2 mm from one to the next.
SAVED_OFFSET = {
    "seurat_map": 0.0,
    "symphony": 0.0,
    "celltypist_hvg": -0.27,
    "celltypist": -0.09,
    "unintegrated": 0.09,
    "integrated": 0.27,
}
# The saved sizes' plots: the gap between the held-out tissues and the ladder, and the ladder plot's width.
SAVED_GAP_MM = 3.5
SAVED_LADDER_W_MM = 38.0
LADDER_X = {
    "reference_500000": 0.0,
    "reference_1500000": 1.0,
    "reference_3000000": 2.0,
    "reference_all": 3.0,
}


def size_label(cells: int) -> str:
    return f"{cells / 1e6:.1f}"


def saved_reference_size(ax, data: dict) -> None:
    """The bytes each method saves to reuse a reference, on a log10 scale: CyteArc's descriptor written at
    registration, CellTypist's model, Symphony's RDS and Seurat's RDS with its neighbor index; on the left the held-out
    tissues, on the right each ladder size at which the method completed."""
    ladder = sibling_axes(ax, width_mm(ax) + SAVED_GAP_MM, 0.0, SAVED_LADDER_W_MM, height_mm(ax))
    for plot in (ax, ladder):
        plot.set_yscale("log")
        plot.set_ylim(0.04, 3e5)
        log_axis(plot.yaxis, list(SAVED_TICKS), SAVED_TICKS)
        value_axis(plot)
    tissues = data["tissues"]
    for key in LADDER_ORDER:
        for i, tissue in enumerate(tissues):
            saved = data["heldOut"][tissue]
            if key in saved:
                method_point(ax, key, i + SAVED_OFFSET[key], saved[key] / 1e6)
        runs = data["ladder"][key]
        method_line(
            ladder,
            key,
            [LADDER_X[r["part"]] + SAVED_OFFSET[key] for r in runs],
            [r["bytes"] / 1e6 for r in runs],
        )
    ax.set_xlim(-0.6, len(tissues) - 0.4)
    category_ticks(ax.xaxis, list(range(len(tissues))), tissues)
    ax.tick_params(axis="x", labelsize=fs.NOTE)
    ax.set_xlabel("held-out", labelpad=1.5)
    ax.set_ylabel("bytes (log10)", labelpad=1.5)
    ladder.set_xlim(-0.6, 3.6)
    category_ticks(ladder.xaxis, [0, 1, 2, 3], [size_label(s["cells"]) for s in data["sizes"]])
    ladder.tick_params(axis="y", labelleft=False)
    ladder.set_xlabel("reference cells (M)", labelpad=1.5)
    fs.frame_axes(ladder)


# ---- the blood reference ladder -----------------------------------------------------------------------------------


def ladder_ticks(ax, sizes: list[dict]) -> None:
    ax.set_xlim(-0.5, 3.5)
    category_ticks(
        ax.xaxis, [LADDER_X[s["part"]] for s in sizes], [size_label(s["cells"]) for s in sizes]
    )
    ax.set_xlabel("reference cells (M)", labelpad=1.5)


def ladder_lines(ax, data: dict, value) -> None:
    for key in LADDER_ORDER:
        runs = data["methods"][key]
        method_line(ax, key, [LADDER_X[r["part"]] for r in runs], [value(r) for r in runs])


def ladder_annotation_time(ax, data: dict) -> None:
    """Seconds to annotate the query from the stored reference, on a log10 scale: registering the finished analysis and
    annotating for CyteArc, loading the saved reference and annotating for the comparators."""
    ax.set_yscale("log")
    ax.set_ylim(1, 6000)
    log_axis(ax.yaxis, [1, 10, 100, 1000])
    ladder_ticks(ax, data["sizes"])
    value_axis(ax)
    ladder_lines(ax, data, lambda r: r["storedSeconds"])
    ax.set_ylabel("seconds (log10)", labelpad=1.5)


def ladder_rare_type_missed(ax, data: dict) -> None:
    """Rare-type cells missed: 100 minus the mean recall of the four types below 1% of the reference pool (erythroid,
    plasma cells, platelets and progenitors), on an axis from zero."""
    ax.set_ylim(0, 60)
    linear_axis(ax.yaxis, [0, 20, 40, 60])
    ladder_ticks(ax, data["sizes"])
    value_axis(ax)
    ladder_lines(ax, data, lambda r: 100 * (1 - r["value"]))
    ax.set_ylabel("missed (%)", labelpad=1.5)


PANELS = {
    draw.__name__: draw
    for draw in (
        registration_reads,
        saved_reference_size,
        ladder_annotation_time,
        ladder_rare_type_missed,
    )
}
# Panels drawn from another panel's data file.
DATA_FILE: dict[str, str] = {}

# Short headers beside panel letters, naming what each panel measures where its axes do not.
HEADERS = {
    "registration_reads": "Registering a reference",
    "saved_reference_size": "Saved to reuse",
    "ladder_annotation_time": "Annotate from stored",
    "ladder_rare_type_missed": "Rare types",
}
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
    "registration_reads": Cell(66.0, 36.0, (14.0, 12.0, 26.0, 18.5)),
    "saved_reference_size": Cell(94.0, 36.0, (15.0, 7.5, 34.0, 23.0)),
    "ladder_annotation_time": Cell(80.0, 32.0, (10.5, 7.0, 66.0, 20.0)),
    "ladder_rare_type_missed": Cell(80.0, 32.0, (10.5, 7.0, 66.0, 20.0)),
}
# Where each panel's cell sits: its lower-left corner, in millimetres from the figure's lower-left corner. The top row
# holds registration and what each method saves; under it, filling the page's width, the blood reference ladder's
# annotation time from a stored reference beside its rare-type cells missed. The legend of every method runs across the
# top of the figure, above the panels. The panels take their letters in this order.
LAYOUT = {
    "registration_reads": (0.0, 32.0),
    "saved_reference_size": (66.0, 32.0),
    "ladder_annotation_time": (0.0, 0.0),
    "ladder_rare_type_missed": (80.0, 0.0),
}


def check_layout() -> None:
    if sorted(LAYOUT) != sorted(PANELS) or sorted(CELLS) != sorted(PANELS):
        raise ValueError(
            f"LAYOUT places {sorted(LAYOUT)} and CELLS sizes {sorted(CELLS)}; both must hold each panel of "
            f"{sorted(PANELS)} once"
        )
    boxes = {}
    for name, (left, bottom) in LAYOUT.items():
        cell = CELLS[name]
        if left < 0 or bottom < 0 or left + cell.width > 160 + 1e-9:
            raise ValueError(
                f"{name} reaches past the figure's 160 mm width or below its bottom edge"
            )
        boxes[name] = (left, bottom, left + cell.width, bottom + cell.height)
    if max(box[3] for box in boxes.values()) + LEGEND_BAND_MM > 225:
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
    fig = plt.figure(figsize=(fs.FULL_WIDTH, (height + LEGEND_BAND_MM) * fs.MM))
    method_legend(fig)
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
