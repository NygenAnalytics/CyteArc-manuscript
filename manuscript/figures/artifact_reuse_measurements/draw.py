"""Draw the supplementary figure on reusing stored artifacts: one function per panel, and the panels assembled with their
letters.

Each panel function draws one panel into the plot it is given and is named for what the panel shows. The name is also
the panel's SVG group ID and the name of its data file, <name>.json, which prepare.py writes.
CELLS gives each panel's printed size, and LAYOUT places the panels, which sets their letters. Run from the repository
root:

  uv run --locked python -B -m manuscript.figures.artifact_reuse_measurements.draw            # all panels, the figure
  uv run --locked python -B -m manuscript.figures.artifact_reuse_measurements.draw <panel>    # one panel

Renders go to .build/figures/artifact_reuse_measurements/: each panel alone and without a letter as
panels/<panel>.png, .pdf and .svg; the lettered figure as artifact_reuse_measurements.png, .pdf and .svg; and each
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
from manuscript.figures import glyphs as g  # noqa: E402
from manuscript.figures import style as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
BUILD = REPO / ".build" / "figures" / HERE.name
ALONE = {"canvas: width": "a panel alone is drawn at the size it prints at in the figure"}

# The top of each cell holds the panel letter and header; a legend, where a panel has one, sits under them, its top this
# far below the top of the cell.
HEADER_BAND_MM = 4.2


# ---- shared helpers -----------------------------------------------------------------------------


def width_mm(ax) -> float:
    return ax.get_position().width * ax.figure.get_figwidth() / fs.MM


def height_mm(ax) -> float:
    return ax.get_position().height * ax.figure.get_figheight() / fs.MM


def plain(value, _=None) -> str:
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


def log_axis(axis, ticks: list[float]) -> None:
    """A log axis labelled at ticks with plain numbers, and no minor ticks."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(plain))
    axis.set_minor_formatter(NullFormatter())


def linear_axis(axis, ticks: list[float]) -> None:
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def duration(seconds: float) -> str:
    """A time as the text writes it: seconds under two minutes, then minutes."""
    if seconds < 1:
        return f"{seconds:.2f} s"
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 120:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    return f"{minutes:.2f} min" if minutes < 10 else f"{minutes:.1f} min"


def gigabytes(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n < 1e10 else f"{n / 1e9:.1f} GB"


def size(n: float) -> str:
    """Bytes in decimal units, as the text gives them; nothing written is 0."""
    if n == 0:
        return "0"
    if n < 1e6:
        return f"{n / 1e3:.0f} kB"
    if n < 1e7:
        return f"{n / 1e6:.1f} MB"
    if n < 1e9:
        return f"{n / 1e6:.0f} MB"
    return gigabytes(n)


def sibling_axes(ax, left_mm: float, w_mm: float, *, down_mm: float = 0.0):
    """A second plot of the same panel and of ax's height, left_mm right of ax's plot and down_mm below it."""
    pos = ax.get_position()
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    other = fs.add_axes(fig, pos.x0 * fw + left_mm, pos.y0 * fh - down_mm, w_mm, pos.height * fh)
    other.set_gid(f"{ax.get_gid()}_{len(fig.axes)}")
    return other


# ---- marks: a run without reuse is a slate square; CyteArc reusing stored artifacts is blue ------------------------

BLUE = fs.ENTITIES["cytearc"].colour
EVEN = {"o": 1.0, "s": 0.9, "D": 0.8}
# Each kind of mark: its marker, fill and edge. A step computed without reuse (a published run, a fresh composite, a
# complete rerun) is a slate square; CyteArc computing a new artifact on stored ones is a filled blue circle, and
# returning a stored artifact an open one. Diamonds mark the bytes read from a source and results recovered. The
# legends name the circles "computed" and "stored" in every panel, and each square for the run it stands for.
MARKS = {
    "computed": ("s", fs.SLATE, fs.INK),
    "new": ("o", BLUE, fs.INK),
    "stored": ("o", fs.WHITE, BLUE),
    "recovered": ("D", fs.WHITE, BLUE),
    "read": ("D", BLUE, fs.INK),
}


def marker_ms(marker: str) -> float:
    return fs.MARKER_MS * EVEN.get(marker, 1.0)


def mark_style(kind: str) -> dict:
    marker, face, edge = MARKS[kind]
    return {
        "marker": marker,
        "ms": marker_ms(marker),
        "mfc": face,
        "mec": edge,
        "mew": fs.LW_MARKER,
    }


def mark(ax, x: float, y: float, kind: str) -> None:
    """One value; CyteArc's marks sit over the slate ones."""
    ax.plot(
        [x],
        [y],
        ls="none",
        zorder=4 if kind == "computed" else 5,
        clip_on=False,
        **mark_style(kind),
    )


def colour_of(kind: str) -> str:
    return fs.SLATE if kind == "computed" else BLUE


def summary_mark(ax, summary: dict, y: float, kind: str) -> None:
    """The mean of several runs with the range of their values."""
    ax.plot(
        [summary["low"], summary["high"]],
        [y, y],
        color=colour_of(kind),
        lw=fs.LW_DATA,
        solid_capstyle="butt",
        zorder=3,
    )
    mark(ax, summary["mean"], y, kind)


def handle(kind: str, *, line: bool = False) -> Line2D:
    return Line2D(
        [], [], ls="-" if line else "none", color=colour_of(kind), lw=fs.LW_DATA, **mark_style(kind)
    )


def rows(ax, names: list[str]) -> list[float]:
    """Named rows from the top, the names on the left at 6 pt without ticks; returns each row's y."""
    n = len(names)
    ys = [float(n - 1 - i) for i in range(n)]
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=fs.NOTE)
    ax.tick_params(axis="y", length=0, pad=1.5)
    return ys


def rule(ax, y: float) -> None:
    """A light rule between the steps and their total."""
    ax.axhline(y, color=fs.LIGHT, lw=fs.LW_THIN, zorder=1)


def x_grid(ax) -> None:
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)


def seconds_axis(
    ax, low: float, high: float, ticks: list[float], label: str = "seconds (log10)"
) -> None:
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    log_axis(ax.xaxis, ticks)
    x_grid(ax)
    ax.set_xlabel(label, labelpad=1.5)


COLUMN_GAP_MM = 1.8


def columns(
    ax, entries: list[tuple[str, list[str]]], ys: list[float], *, header_y: float | None = None
) -> None:
    """Columns of values right of the plot, each right-aligned under its header, which may be empty, and as wide as its
    widest text; the headers sit above the plot, or at data y header_y when only the rows below it hold values."""
    right = width_mm(ax)
    for header, texts in entries:
        right += COLUMN_GAP_MM + max(g.text_width(t, fs.NOTE) for t in [header, *texts])
        x = right / width_mm(ax)
        if header:
            ax.text(
                x,
                1 + 0.8 / height_mm(ax) if header_y is None else header_y,
                header,
                transform=ax.transAxes if header_y is None else ax.get_yaxis_transform(),
                fontsize=fs.NOTE,
                color=fs.SLATE,
                ha="right",
                va="bottom" if header_y is None else "center",
            )
        for y, text in zip(ys, texts):
            if text:
                ax.text(
                    x,
                    y,
                    text,
                    transform=ax.get_yaxis_transform(),
                    fontsize=fs.NOTE,
                    color=fs.INK,
                    ha="right",
                    va="center",
                )


LEGEND_LEFT_MM = 1.5


def legend(ax, entries: list[tuple[str, str]], *, line: bool = False) -> None:
    """A legend in one row under the panel's header, from the cell's left edge."""
    cell = CELLS[ax.get_gid()]
    band = cell.height - cell.plot[1] - cell.plot[3]
    ax.legend(
        [handle(kind, line=line) for kind, _ in entries],
        [label for _, label in entries],
        loc="upper left",
        bbox_to_anchor=(
            (LEGEND_LEFT_MM - cell.plot[0]) / width_mm(ax),
            1 + (band - HEADER_BAND_MM) / height_mm(ax),
        ),
        borderaxespad=0.0,
        ncol=len(entries),
        columnspacing=0.9,
        handlelength=1.8 if line else 0.9,
        handletextpad=0.35,
    )


# ---- the mounted ten-million-cell embryo store ---------------------------------------------------------------------


def mounted_store_requests(ax, data: dict) -> None:
    """Each request on the published analysis of the mounted ten-million-cell embryo store, on 8 CPUs, beside the
    published run's stage it repeats or parallels, on 16 CPUs, on a log10 axis of seconds; the last row is the published
    analysis from cell filtering through marker search. The columns give the bytes each request read from the source
    store and wrote to the user store."""
    requests = data["requests"]
    ys = rows(ax, [r["name"] for r in requests] + ["initial analysis"])
    seconds_axis(ax, 0.3, 8000, [1, 10, 100, 1000])
    for y, r in zip(ys, requests):
        if r["publishedSeconds"] is not None:
            mark(ax, r["publishedSeconds"], y, "computed")
        mark(ax, r["seconds"], y, r["kind"])
    mark(ax, data["publishedAnalysisSeconds"], ys[-1], "computed")
    rule(ax, ys[-1] + 0.5)
    columns(
        ax,
        [
            ("read", [size(r["readBytes"]) for r in requests] + [""]),
            ("written", [size(r["writtenBytes"]) for r in requests] + [""]),
        ],
        ys,
    )
    legend(
        ax,
        [
            ("computed", f"initial, {data['cpus']['published']} CPUs"),
            ("stored", "stored"),
            ("new", "computed"),
        ],
    )


def endothelial_store_size(ax, data: dict) -> None:
    """The published embryo store, its parts and its source H5AD against the user store mounted on it, after the
    mount, after the endothelial reanalysis and after the requests on the whole atlas, and the bytes all steps read
    from the source store, on a log10 axis."""
    sizes = data["sizes"]
    ys = rows(ax, [s["name"] for s in sizes] + ["read from the source"])
    ax.set_xscale("log")
    ax.set_xlim(0.3, 300)
    log_axis(ax.xaxis, [1, 10, 100])
    x_grid(ax)
    ax.set_xlabel("GB (log10)", labelpad=1.5)
    for y, s in zip(ys, sizes):
        mark(ax, s["bytes"] / 1e9, y, "computed" if s["kind"] == "source" else "new")
    mark(ax, data["readBytes"] / 1e9, ys[-1], "read")
    columns(ax, [("", [gigabytes(s["bytes"]) for s in sizes] + [gigabytes(data["readBytes"])])], ys)
    legend(ax, [("computed", "source"), ("new", "user store"), ("read", "read, all steps")])


# ---- the neighbor-count fork on the stored ten-million-cell embryo analysis ----------------------------------------


def neighbor_count_fork_stages(ax, data: dict) -> None:
    """Each stage of the stored analysis of the ten-million-cell store beside the fork's step for it, a lookup of a
    stored artifact or a recomputation, and the totals under a rule, on a log10 axis of seconds."""
    stages = data["stages"]
    ys = rows(ax, [s["name"] for s in stages] + ["total"])
    seconds_axis(ax, 5, 6000, [10, 100, 1000])
    rule(ax, ys[-1] + 0.5)
    for y, s in zip(ys, stages):
        mark(ax, s["storedSeconds"], y, "computed")
        kind = {"lookup": "stored", "computed": "new"}.get(s["fork"])
        if kind:
            mark(ax, s["forkSeconds"], y, kind)
    totals = data["totals"]
    mark(ax, totals["storedSeconds"], ys[-1], "computed")
    mark(ax, totals["forkSeconds"], ys[-1], "new")
    # The stored analysis is the published run the fork started from.
    legend(
        ax,
        [
            ("computed", f"initial, {data['cpus']['stored']} CPUs"),
            ("stored", "stored"),
            ("new", "computed"),
        ],
    )


# ---- Harmony integration on the four-study COVID-19 composite ----------------------------------------------------


def composite_integration_stages(ax, data: dict) -> None:
    """Each stage of Harmony integration on a fresh composite beside the same stage after the unintegrated analysis,
    which returned the stored upstream artifacts, and the time of each integration under a rule, on a log10 axis of
    seconds: the mean and range of three replicates."""
    stages = data["stages"]
    ys = rows(ax, [s["name"] for s in stages] + ["total"])
    seconds_axis(ax, 0.01, 400, [0.01, 0.1, 1, 10, 100])
    for y, s in zip(ys, stages):
        summary_mark(ax, s["fresh"], y, "computed")
        summary_mark(ax, s["reuse"], y, "stored" if s["stored"] else "new")
    rule(ax, ys[-1] + 0.5)
    summary_mark(ax, data["totals"]["fresh"], ys[-1], "computed")
    summary_mark(ax, data["totals"]["reuse"], ys[-1], "new")
    columns(
        ax,
        [
            ("fresh", [""] * len(stages) + [duration(data["totals"]["fresh"]["mean"])]),
            ("reuse", [""] * len(stages) + [duration(data["totals"]["reuse"]["mean"])]),
        ],
        ys,
        header_y=ys[-1] + 0.85,
    )
    legend(ax, [("computed", "fresh composite"), ("stored", "stored"), ("new", "computed")])


# The peak plot sits under the counts plot, this far below it, with the same rows.
RESOURCE_GAP_MM = 8.0


def composite_integration_resources(ax, data: dict) -> None:
    """The count bytes each analysis of the composite read from its four source stores, and under it its container
    peak, the mean and range of three replicates, on axes from zero."""
    arms = data["arms"]
    peak = sibling_axes(ax, 0.0, width_mm(ax), down_mm=height_mm(ax) + RESOURCE_GAP_MM)
    for plot in (ax, peak):
        ys = rows(plot, [a["name"] for a in arms])
        # The row names stand off the plot, clear of the point at zero on its left edge.
        plot.tick_params(axis="y", pad=3.0)
    ax.set_xlim(0, 3)
    linear_axis(ax.xaxis, [0, 1, 2, 3])
    peak.set_xlim(0, 9)
    linear_axis(peak.xaxis, [0, 3, 6, 9])
    for plot, label in ((ax, "counts read (GB)"), (peak, "peak (GiB)")):
        x_grid(plot)
        plot.set_xlabel(label, labelpad=1.5)
    for y, a in zip(ys, arms):
        kind = "new" if a["arm"] == "integrationReusingUpstream" else "computed"
        read = {key: a["readBytes"][key] / 1e9 for key in ("low", "mean", "high")}
        summary_mark(ax, read, y, kind)
        summary_mark(peak, a["peakGiB"], y, kind)
    columns(ax, [("", [size(a["readBytes"]["mean"]) for a in arms])], ys)
    columns(peak, [("", [f"{a['peakGiB']['mean']:.1f} GiB" for a in arms])], ys)
    fs.frame_axes(peak)


# ---- the lung reference revision -----------------------------------------------------------------------------------


def reference_update_steps(ax, data: dict) -> None:
    """Each step of updating the Natri query store to the revised reference, where the v1 results are stored, beside
    the same step of a complete rerun of both versions in a fresh store, and the totals under a rule, on a log10 axis
    of seconds; the columns give the bytes each step added to the store."""
    steps = data["rows"]
    ys = rows(ax, [s["name"] for s in steps] + ["total"])
    seconds_axis(ax, 1, 3000, [1, 10, 100, 1000])
    for y, s in zip(ys, steps):
        for seconds in s["rerun"]:
            mark(ax, seconds, y, "computed")
        for seconds in s["update"]:
            mark(ax, seconds, y, s["updateKind"])
    rule(ax, ys[-1] + 0.5)
    mark(ax, data["rerun"]["seconds"], ys[-1], "computed")
    mark(ax, data["update"]["seconds"], ys[-1], "new")
    columns(
        ax,
        [
            (
                "update",
                [size(s["updateAddedBytes"]) if s["update"] else "" for s in steps]
                + [size(data["update"]["addedBytes"])],
            ),
            (
                "rerun",
                [size(s["rerunAddedBytes"]) if s["rerun"] else "" for s in steps]
                + [size(data["rerun"]["addedBytes"])],
            ),
        ],
        ys,
    )
    legend(
        ax,
        [
            ("computed", "rerun"),
            ("stored", "stored"),
            ("new", "computed"),
            ("recovered", "recovered"),
        ],
    )


# ---- a revised clustering of the CITE-seq WNN analysis -------------------------------------------------------------

CITE_TICKS = ("0.1", "0.3", "0.6", "1.0", "1.46")
# The time plot runs on past the largest size to hold the times there at the ends of their lines; the peak plot sits
# CITE_GAP_MM right of it, with the same width per size.
CITE_LABEL_ROOM = 0.85
CITE_GAP_MM = 19.0


def cite_axis(ax, *, room: float = 0.0) -> None:
    ax.set_xlim(-0.5, len(CITE_TICKS) - 0.5 + room)
    ax.xaxis.set_major_locator(FixedLocator(list(range(len(CITE_TICKS)))))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda _, i: CITE_TICKS[i]))
    ax.set_xlabel("CITE-seq cells (M)", labelpad=1.5)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def cite_line(ax, ys: list[float], kind: str) -> None:
    ax.plot(
        range(len(ys)),
        ys,
        color=colour_of(kind),
        lw=fs.LW_PRIMARY if kind == "new" else fs.LW_DATA,
        zorder=3,
    )
    for x, y in enumerate(ys):
        mark(ax, x, y, kind)


def multimodal_revision(ax, data: dict) -> None:
    """Revising the WNN clustering of 100,000 to 1,457,527 CITE-seq cells to resolution 1.2 from the stored RNA and
    protein graphs, against recomputing both modalities from the counts: time in minutes on a log10 axis on the left,
    with the times at the largest size beside it, and peak memory on an axis from zero on the right."""
    sizes = data["sizes"]
    if [s["cells"] for s in sizes] != [100000, 300000, 600000, 1000000, 1457527]:
        raise ValueError("the CITE-seq panel draws the five sizes of its tick labels")
    sizes_w = len(CITE_TICKS)
    peak = sibling_axes(
        ax, width_mm(ax) + CITE_GAP_MM, width_mm(ax) * sizes_w / (sizes_w + CITE_LABEL_ROOM)
    )
    ax.set_yscale("log")
    ax.set_ylim(0.08, 30)
    log_axis(ax.yaxis, [0.1, 1, 10])
    peak.set_ylim(0, 12)
    linear_axis(peak.yaxis, [0, 4, 8, 12])
    cite_axis(ax, room=CITE_LABEL_ROOM)
    cite_axis(peak)
    ax.set_ylabel("minutes (log10)", labelpad=1.5)
    peak.set_ylabel("peak (GiB)", labelpad=1.5)
    for kind, key in (("computed", "fresh"), ("new", "reuse")):
        cite_line(ax, [s[f"{key}Seconds"] / 60 for s in sizes], kind)
        cite_line(peak, [s[f"{key}PeakGiB"] for s in sizes], kind)
    # The times at the largest size, at the ends of their lines.
    last = sizes[-1]
    for key in ("fresh", "reuse"):
        ax.text(
            len(sizes) - 1 + 0.3,
            last[f"{key}Seconds"] / 60,
            duration(last[f"{key}Seconds"]),
            fontsize=fs.NOTE,
            color=fs.INK,
            ha="left",
            va="center",
        )
    fs.frame_axes(peak)
    legend(ax, [("computed", "recomputed from counts"), ("new", "from stored graphs")], line=True)


PANELS = {
    draw.__name__: draw
    for draw in (
        mounted_store_requests,
        endothelial_store_size,
        neighbor_count_fork_stages,
        composite_integration_stages,
        composite_integration_resources,
        reference_update_steps,
        multimodal_revision,
    )
}

# Short headers beside panel letters: each panel's dataset, which its axes and legend do not name.
HEADERS = {
    "mounted_store_requests": "Requests on the mounted 10M-cell store",
    "endothelial_store_size": "Remote store and user store",
    "neighbor_count_fork_stages": "Fork from k = 11 to 15, 10M-cell store",
    "composite_integration_stages": "Harmony on the COVID-19 composite",
    "composite_integration_resources": "Composite: counts read and peak",
    "reference_update_steps": "Natri store: reference update",
    "multimodal_revision": "CITE-seq: WNN clustering revised",
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
    "mounted_store_requests": Cell(80.0, 40.0, (26.5, 7.2, 28.0, 25.0)),
    "endothelial_store_size": Cell(80.0, 40.0, (26.5, 7.2, 34.0, 25.0)),
    "neighbor_count_fork_stages": Cell(80.0, 48.0, (23.0, 7.2, 40.0, 33.0)),
    "composite_integration_stages": Cell(80.0, 40.0, (17.0, 7.2, 34.0, 25.0)),
    # The counts plot; the peak plot sits RESOURCE_GAP_MM under it, level with the bottom of the plot beside it.
    "composite_integration_resources": Cell(80.0, 40.0, (26.5, 23.7, 34.0, 8.5)),
    "reference_update_steps": Cell(80.0, 40.0, (23.5, 7.2, 30.0, 25.0)),
    # The time plot; the peak plot sits to its right.
    "multimodal_revision": Cell(160.0, 36.0, (12.0, 7.5, 64.0, 20.0)),
}
# Where each panel's cell sits: its lower-left corner, in millimetres from the figure's lower-left corner. Two columns,
# read row by row in the order the text cites the panels, each row's cells aligned at the top: the requests on the
# mounted embryo store beside the neighbor-count fork; the mounted store's sizes beside Harmony on the composite; the
# composite's counts read and peak beside the lung reference revision; and across the bottom, the revised CITE-seq
# clustering. The panels take their letters in this order.
LAYOUT = {
    "mounted_store_requests": (0.0, 124.0),
    "neighbor_count_fork_stages": (80.0, 116.0),
    "endothelial_store_size": (0.0, 76.0),
    "composite_integration_stages": (80.0, 76.0),
    "composite_integration_resources": (0.0, 36.0),
    "reference_update_steps": (80.0, 36.0),
    "multimodal_revision": (0.0, 0.0),
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
    ax.set_gid(name)
    PANELS[name](ax, json.loads((HERE / f"{name}.json").read_text()))
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
