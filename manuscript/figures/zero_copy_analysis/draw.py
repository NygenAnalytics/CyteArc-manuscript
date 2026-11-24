"""Draw the figure on zero-copy analysis: one function per panel, and the panels assembled with their letters.

Each panel function draws one panel into the plot it is given and is named for what the panel shows. The name is also
the panel's SVG group ID and the name of its data file, <name>.json, which prepare.py writes.
CELLS gives each panel's printed size, and LAYOUT places the panels, which sets their letters. Run from the repository
root:

  uv run --locked python -B -m manuscript.figures.zero_copy_analysis.draw            # every panel and the figure
  uv run --locked python -B -m manuscript.figures.zero_copy_analysis.draw <panel>    # one panel

Renders go to .build/figures/zero_copy_analysis/: each panel alone and without a letter as panels/<panel>.png, .pdf and
.svg; the lettered figure as zero_copy_analysis.png, .pdf and .svg; and each panel's letter in letters.json, which every
run writes.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from manuscript.figures import glyphs as g  # noqa: E402
from manuscript.figures import style as fs  # noqa: E402
from manuscript.figures import thumbnails as th  # noqa: E402

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
            "columnspacing": 0.65,
            "labelspacing": 0.25,
            "handlelength": 1.2,
            "handletextpad": 0.35,
            **spacing,
        },
    )


def mm_text(ax, x, y, text, *, colour=fs.INK, size=fs.NOTE, ha="center", va="center", **kw):
    """Text on a panel drawn in millimetres."""
    kw.setdefault("zorder", 8)
    return ax.text(x, y, text, fontsize=size, color=colour, ha=ha, va=va, **kw)


def p_text(p: float) -> str:
    """A P value with two significant digits, its trailing zero kept: P = 0.010, P = 0.0047; under 0.001 in powers of
    ten: P = 7.1 × 10⁻⁵."""
    if p >= 0.001:
        return f"P = {p:#.2g}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"P = {mantissa} × 10$^{{\\mathregular{{{int(exponent)}}}}}$"


def log_minor_ticks(axis) -> None:
    """A log axis's unlabelled ticks at 2 to 9 times each power of ten, shorter than its labelled ones. The locator's
    tick budget is fixed: left to size it from a short axis, matplotlib drops every minor tick."""
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    axis.set_minor_formatter(NullFormatter())
    axis.set_tick_params(which="minor", length=1.6, width=fs.LW_THIN)


def byte_size(value: float, *, decimals: int = 1) -> str:
    """Bytes in decimal units, as the text writes them: MB under a gigabyte, GB from there."""
    if value < 1e9:
        return f"{value / 1e6:.0f} MB"
    return f"{value / 1e9:.{decimals}f} GB" if value < 1e11 else f"{value / 1e9:.0f} GB"


# A panel's header starts this far from its cell's left edge, after the letter; panels without a header put their
# legend there.
HEADER_INDENT_MM = 4.8
# The plots of the row under the schematic share a frame: their bottom and top edges, in millimetres from the bottom of
# their cells.
ROW_FRAME_MM = (7.0, 31.0)


# Every CyteArc bar: the 8 GiB step of the budget ramp, as in the analysis-scaling figure; a tool outside the registry,
# such as AnnData, or a download, is slate.
CYTEARC_BAR = {"facecolor": fs.BUDGET_COLOURS[8], "edgecolor": fs.INK, "linewidth": fs.LW_BAR}
COMPARATOR_BAR = fs.bar_style(fs.COMPARATOR)


# Every bar chart draws its bars this wide, with this gap between neighbours, in millimetres on an x axis that runs in
# millimetres across its frame, so bars read alike from panel to panel. A bar too short to see is drawn this tall, which
# the captions state.
BAR_MM = 2.0
BAR_GAP_MM = 0.5
MIN_BAR_MM = 0.5
# Each comparator's value over CyteArc's: a bracket from CyteArc's bar to the comparator's, the nearest lowest, the first
# this far above the tallest bar, each next one this far above the last, with room for its label.
RATIO_CLEARANCE_MM = 1.2
RATIO_STEP_MM = 3.4
RATIO_LABEL_MM = 2.6


def bar_centres(ax, groups: list[int], group_gap_mm: float = 0.0) -> list[list[float]]:
    """The centres of each group's bars, in millimetres from the frame's left edge, the groups side by side
    group_gap_mm apart and centred in the frame; sets the x axis to millimetres."""
    width = width_mm(ax)
    spans = [n * BAR_MM + (n - 1) * BAR_GAP_MM for n in groups]
    left = (width - sum(spans) - group_gap_mm * (len(groups) - 1)) / 2
    if left < 0:
        raise ValueError(f"{sum(groups)} bars need more than the frame's {width:.1f} mm")
    centres = []
    for n, span in zip(groups, spans):
        centres.append([left + BAR_MM / 2 + i * (BAR_MM + BAR_GAP_MM) for i in range(n)])
        left += span + group_gap_mm
    ax.set_xlim(0, width)
    return centres


def shown_height(ax, value: float, top: float) -> float:
    """A bar's drawn height on an axis from 0 to top: its value, or MIN_BAR_MM when the value would be shorter."""
    return max(value, MIN_BAR_MM * top / height_mm(ax))


def ratio_brackets(
    ax, centres: list[float], ratios: list[float], highest: float, top: float
) -> None:
    """Over the bars, each later bar's value over the first's, as a bracket from the first bar to its own, the nearest
    lowest; raises when the stack would not fit under top."""
    per_mm = top / height_mm(ax)
    for i, (x, ratio) in enumerate(zip(centres[1:], ratios)):
        y = highest + (RATIO_CLEARANCE_MM + i * RATIO_STEP_MM) * per_mm
        pair_bracket(ax, centres[0], x, y, 0.6 * per_mm, [(f"×{amount(ratio)}", fs.INK)])
    needed = RATIO_CLEARANCE_MM + (len(ratios) - 1) * RATIO_STEP_MM + RATIO_LABEL_MM
    if highest + needed * per_mm > top:
        raise ValueError(f"the ratios' brackets need an axis above {top}")


def tool_bar(entity: str) -> dict:
    if entity == "cytearc":
        return CYTEARC_BAR
    if entity in fs.ENTITIES:
        return fs.bar_style(fs.ENTITIES[entity].colour)
    return COMPARATOR_BAR


# ---- a remote store mounted under a local user store, and stores combined into a composite -----------

# The heatmap's rows taken for each of the composite's three sources, each cut into blocks of BLOCK_ROWS rows as a
# source's stored chunks cut its cells. The composite orders its rows as CyteArc's row plan does: the blocks of all
# sources shuffled together, (source, block) in the order below, and each block's rows permuted within it.
SOURCE_ROWS = (slice(0, 8), slice(16, 24), slice(32, 40))
BLOCK_ROWS = 4
BLOCK_ORDER = ((1, 0), (0, 1), (2, 0), (0, 0), (2, 1), (1, 1))
WITHIN_BLOCK_ORDER = (2, 0, 3, 1)
# The composite's sketch names four genes, heatmap columns that each source expresses differently. Each source stores
# them in its own order; the composite reads them all in the order A to D.
GENE_COLUMNS = (0, 2, 6, 9)
GENE_NAMES = "ABCD"
SOURCE_GENE_ORDERS = ((1, 0, 3, 2), (2, 0, 3, 1), (3, 1, 0, 2))


def composite_rows(heat: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """The composite's rows of the four named genes, block by block in BLOCK_ORDER with each block's rows permuted,
    and the source of each row."""
    rows, sources = [], []
    for source, block in BLOCK_ORDER:
        values = heat[SOURCE_ROWS[source]][block * BLOCK_ROWS : (block + 1) * BLOCK_ROWS]
        rows.append(np.clip(values[list(WITHIN_BLOCK_ORDER)][:, list(GENE_COLUMNS)], 0, 1))
        sources += [source] * BLOCK_ROWS
    return np.vstack(rows), sources


def counts_matrix(ax, x, y, w, h, values, *, kind: str, bands: int = 3, cmap=None, z=5) -> None:
    """A small count matrix, cells as rows and genes as columns, cut into bands of cells (cell-major) or of genes
    (gene-major) by ink lines, or uncut for any other kind."""
    ax.imshow(
        values,
        extent=(x, x + w, y, y + h),
        cmap=cmap or fs.SEQUENTIAL,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
        zorder=z,
    )
    if kind == "cell":
        for yy in np.linspace(y, y + h, bands + 1)[1:-1]:
            ax.plot([x, x + w], [yy, yy], color=fs.INK, lw=fs.LW_THIN, zorder=z + 0.1)
    elif kind == "gene":
        for xx in np.linspace(x, x + w, bands + 1)[1:-1]:
            ax.plot([xx, xx], [y, y + h], color=fs.INK, lw=fs.LW_THIN, zorder=z + 0.1)
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="none", edgecolor=fs.INK, lw=fs.LW_AXIS, zorder=z + 0.2)
    )


def gene_names(ax, x, top, w, order) -> None:
    """The genes of a matrix's columns, in its order, just above its top edge."""
    step = w / len(order)
    for i, gene in enumerate(order):
        mm_text(ax, x + (i + 0.5) * step, top + 1.25, GENE_NAMES[gene], size=fs.NOTE)


# The published analysis of the CyteBase store as a chain of artifacts, in the order the analysis made them. A new
# clustering in the user store reads the first REUSED of them where they are and writes the rest anew.
CHAIN = ("genes", "loadings", "ANN index", "graph", "clusters", "markers")
REUSED = 4
# The chain's thumbnails: this wide and tall, this far apart, from this left edge.
NODE_MM = 7.0
NODE_STEP_MM = 10.4
CHAIN_LEFT_MM = 35.0
# The virtual composite's half of the schematic starts this far from its left edge.
COMPOSITE_LEFT_MM = 100.0
# The overlay mount's stores end this far from the schematic's left edge.
MOUNT_RIGHT_MM = 97.0
# The colour strips that mark each source store's cells, in the stores and in the composite, are this wide.
STRIP_MM = 1.8


def selected_genes(ax, x, y, w, h, *, z=3) -> None:
    """Gene selection: each gene's mean against its spread, the genes chosen for their excess spread in ink."""
    rng = np.random.default_rng(3)
    mean = rng.uniform(0.0, 1.0, 40)
    trend = 0.25 + 0.5 * mean
    spread = trend + rng.normal(0.0, 0.07, 40)
    chosen = spread - trend >= np.sort(spread - trend)[-9]
    th.frame(ax, x, y, w, h, z=z)
    px = x + w * (0.12 + 0.76 * mean)
    py = y + h * (0.12 + 0.76 * (spread - spread.min()) / np.ptp(spread))
    ax.scatter(px[~chosen], py[~chosen], s=0.8, color=fs.LIGHT, linewidths=0, zorder=z + 0.1)
    ax.scatter(px[chosen], py[chosen], s=1.4, color=fs.INK, linewidths=0, zorder=z + 0.2)


def ann_index(ax, x, y, w, h, *, z=3) -> None:
    """An approximate nearest-neighbour index as the layers of a navigable graph: seven cells on the bottom layer, four
    of them on the next and two on the top, each layer's cells linked in a row and down to themselves below."""
    th.frame(ax, x, y, w, h, z=z)
    positions = x + w * np.linspace(0.14, 0.86, 7)
    levels = [y + h * share for share in (0.2, 0.5, 0.8)]
    layers = [positions, positions[[0, 2, 4, 6]], positions[[1, 5]]]
    for i, (cells, level) in enumerate(zip(layers, levels)):
        ax.plot(cells, [level] * len(cells), color=fs.SLATE, lw=0.35, zorder=z + 0.1)
        if i:
            for cx in cells:
                ax.plot(
                    [cx, cx],
                    [levels[i - 1], level],
                    color=fs.LIGHT,
                    lw=0.35,
                    ls=(0, (1.0, 1.0)),
                    zorder=z + 0.05,
                )
        ax.scatter(cells, [level] * len(cells), s=1.3, color=fs.INK, linewidths=0, zorder=z + 0.2)


def marker_dots(ax, x, y, w, h, *, clustering: str, z=3) -> None:
    """The marker dot plot of a clustering of the thumbnails' dataset: a row per cluster and a column per marker
    gene, with the gene that marks the second clustering's split cluster added for it."""
    data = th.dataset()
    labels = data[clustering]
    genes = [*th.MARKERS, 10] if clustering == "second" else list(th.MARKERS)
    groups = np.unique(labels)
    share = [[(data["logc"][labels == c, gene] > 0).mean() for gene in genes] for c in groups]
    mean = [[data["logc"][labels == c, gene].mean() for gene in genes] for c in groups]
    th.dots(ax, x, y, w, h, share, mean, z=z, max_size=5.0)


def chain_node(ax, name: str, x, y, *, clustering: str = "first", z=3) -> None:
    """One artifact of the published analysis as its thumbnail, NODE_MM square, lower-left corner at (x, y)."""
    s = NODE_MM
    if name == "genes":
        selected_genes(ax, x, y, s, s, z=z)
    elif name == "loadings":
        th.loadings(ax, x + 1.5, y, s - 3.0, s, z=z)
    elif name == "ANN index":
        ann_index(ax, x, y, s, s, z=z)
    elif name == "graph":
        th.graph(ax, x, y, s, s, k=2, size=0.5, z=z)
    elif name == "clusters":
        th.embedding(ax, x, y, s, s, clustering=clustering, size=0.35, z=z)
    elif name == "markers":
        marker_dots(ax, x, y, s, s, clustering=clustering, z=z)
    else:
        raise ValueError(f"no thumbnail for the artifact {name!r}")


def chain_arrows(ax, y, *, z=6) -> None:
    """The steps between the chain's thumbnails, at height y."""
    for i in range(len(CHAIN) - 1):
        left = CHAIN_LEFT_MM + i * NODE_STEP_MM + NODE_MM
        g.arrow(
            ax,
            (left + 0.35, y),
            (left + NODE_STEP_MM - NODE_MM - 0.35, y),
            lw=fs.LW_THIN,
            head=2.6,
            z=z,
        )


def ghost(ax, x, y, w, h, *, z) -> None:
    """An artifact the user store reads where it is: its thumbnail faded under a dashed edge."""
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=fs.WHITE, edgecolor="none", alpha=0.62, zorder=z)
    )
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="none",
            edgecolor=fs.SLATE,
            lw=fs.LW_THIN,
            ls=(0, (2.0, 1.2)),
            zorder=z + 0.1,
        )
    )


def new_edge(ax, x, y, w, h, *, z) -> None:
    """The edge of an artifact the user store wrote: CyteArc's blue, as new results take in the glyphs."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=fs.BLUE, lw=0.9, zorder=z))


def metadata_tables(ax, x, y, w, *, height: float = 4.4, added: bool = False, z=5) -> None:
    """The cell table over the gene table, each height tall and 0.7 mm apart; with added, the cell table carries a new
    column of labels written in the user store."""
    top = y + height + 0.7
    th.table(ax, x, top, w, height, kind="cells", rows=4, z=z)
    th.table(ax, x, y, w, height, kind="genes", rows=4, z=z)
    if added:
        header = height / 5
        column = 1.7
        ax.add_patch(
            Rectangle(
                (x + w, top + height - header),
                column,
                header,
                facecolor=fs.PAPER,
                edgecolor="none",
                zorder=z,
            )
        )
        th.strip(ax, x + w, top, column, height - header, [fs.GROUPS[c] for c in (0, 5, 1, 2)], z=z)
        new_edge(ax, x + w, top, column, height, z=z + 0.3)


def mount_overlay(ax, data: dict) -> None:
    """Schematic of a mounted store and of a virtual composite, in two rows: CyteBase stores in cloud storage above,
    and below them a local CyteArc store and a virtual composite that maps into it. Left, the overlay mount: the
    CyteBase store holds its metadata, its counts and its published analysis as a chain of artifacts, read-only; the
    local store under it copies the metadata, references the counts, reads the published graph to cluster anew and
    computes the new clusters' markers from the referenced counts. Right, three CyteBase stores hold the same genes in
    different orders, each cut into blocks; the zero-copy virtual composite indexes their cells with the genes in one
    order and the blocks of all three shuffled together, and the local store references it as it references a single
    store's counts."""
    w, h = width_mm(ax), height_mm(ax)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_axis_off()
    heat = np.asarray(data["heatmap"], dtype=float)
    extend = fs.MODULES["extend"]
    title_y = h - 3.1
    cloud_y, local_y = (title_y - 16.8, title_y - 3.8), (1.6, title_y - 23.3)
    names_y, verbs_y = cloud_y[1] - 1.5, (local_y[1] + cloud_y[0]) / 2 + 0.6
    for x, title in ((HEADER_INDENT_MM, "Overlay mount"), (COMPOSITE_LEFT_MM, "Virtual composite")):
        ax.text(
            x,
            title_y,
            title,
            fontsize=fs.LABEL,
            fontweight="bold",
            color=fs.INK,
            va="baseline",
            gid="card-title",
        )
    nodes = [CHAIN_LEFT_MM + i * NODE_STEP_MM for i in range(len(CHAIN))]
    centre = NODE_MM / 2
    matrix_h, table_h = 8.7, 4.0

    # 1. The CyteBase store in cloud storage, read-only: its metadata, its counts and the chain of its published
    # analysis, each named above.
    g.folder(
        ax,
        1.0,
        cloud_y[0],
        MOUNT_RIGHT_MM,
        cloud_y[1],
        label="CyteBase store",
        tab_h=3.0,
        size=fs.TICK,
        z=2,
    )
    g.cloud(ax, 27.0, cloud_y[1] + 1.6, 4.6, z=4)
    mm_text(ax, 30.4, cloud_y[1] + 1.5, "cloud storage", colour=fs.SLATE, ha="left")
    g.lock(ax, 46.6, cloud_y[1] + 1.5, s=1.9, z=6)
    matrix_y = cloud_y[0] + 1.1
    node_y = matrix_y + (matrix_h - NODE_MM) / 2
    metadata_tables(ax, 4.0, matrix_y, 7.0, height=table_h)
    mm_text(ax, 7.5, names_y, "metadata")
    mw = 8.5
    counts_x = (14.0, 14.0 + mw + 1.5)
    counts_matrix(ax, counts_x[0], matrix_y, mw, matrix_h, heat[:24, :12], kind="cell")
    counts_matrix(ax, counts_x[1], matrix_y, mw, matrix_h, heat[:24, :12], kind="gene")
    mm_text(ax, counts_x[0] + mw + 0.75, names_y, "counts")
    for x, name in zip(nodes, CHAIN):
        chain_node(ax, name, x, node_y, z=4)
        mm_text(ax, x + centre, names_y, name)
    chain_arrows(ax, node_y + centre)
    g.arrow(
        ax,
        (counts_x[1] + mw + 0.35, node_y + centre),
        (nodes[0] - 0.35, node_y + centre),
        lw=fs.LW_THIN,
        head=2.6,
        z=6,
    )

    # 2. The local CyteArc store under it: the copied metadata with a column of new labels; the counts, referenced
    # where they are; and the user's reanalysis, a new clustering made by reading the published graph and its markers
    # computed from the referenced counts.
    g.folder(
        ax,
        1.0,
        local_y[0],
        MOUNT_RIGHT_MM,
        local_y[1],
        label="local CyteArc store",
        tab_h=3.0,
        size=fs.TICK,
        fill=extend.tint,
        edge=extend.strong,
        side="right",
        z=2,
    )
    g.laptop(ax, MOUNT_RIGHT_MM + 3.1, local_y[1] + 0.3, 4.2, z=4)
    mm_text(ax, MOUNT_RIGHT_MM + 6.2, local_y[1] + 1.6, "local disk", colour=fs.SLATE, ha="left")
    inner_y = local_y[1] - 1.9 - matrix_h
    metadata_tables(ax, 4.0, inner_y, 7.0, height=table_h, added=True)
    for x, kind in zip(counts_x, ("cell", "gene")):
        counts_matrix(ax, x, inner_y, mw, matrix_h, heat[:24, :12], kind=kind)
        ghost(ax, x, inner_y, mw, matrix_h, z=5.5)
    graph = nodes[REUSED - 1]
    clusters_x, markers_x = graph, graph - NODE_STEP_MM
    new_y = inner_y + (matrix_h - NODE_MM) / 2
    chain_node(ax, "clusters", clusters_x, new_y, clustering="second", z=4)
    chain_node(ax, "markers", markers_x, new_y, clustering="second", z=4)
    for x in (clusters_x, markers_x):
        new_edge(ax, x, new_y, NODE_MM, NODE_MM, z=4.6)
    mid = new_y + centre
    g.arrow(ax, (counts_x[1] + mw + 0.4, mid), (markers_x - 0.4, mid), lw=fs.LW_THIN, head=2.6, z=6)
    g.arrow(
        ax,
        (clusters_x - 0.35, mid),
        (markers_x + NODE_MM + 0.35, mid),
        lw=fs.LW_THIN,
        head=2.6,
        z=6,
    )
    mm_text(
        ax,
        (markers_x + clusters_x + NODE_MM) / 2,
        new_y - 1.9,
        "user reanalysis",
        colour=fs.SLATE,
    )
    # What crosses from the cloud to the disk: the metadata copied, the counts referenced, the graph read.
    top, bottom = cloud_y[0] + 0.4, inner_y + matrix_h + 0.4
    g.arrow(ax, (7.5, top), (7.5, bottom), lw=fs.LW_THIN, head=3.0, z=6)
    mm_text(ax, 8.8, verbs_y, "copied", colour=fs.SLATE, ha="left")
    counts_mid = counts_x[0] + mw + 0.75
    g.arrow(ax, (counts_mid, top), (counts_mid, bottom), lw=fs.LW_THIN, head=3.0, dashed=True, z=6)
    mm_text(ax, counts_mid + 1.3, verbs_y, "referenced", colour=fs.SLATE, ha="left")
    g.arrow(
        ax,
        (graph + centre, node_y - 0.4),
        (clusters_x + centre, new_y + NODE_MM + 0.4),
        lw=fs.LW_THIN,
        head=3.0,
        z=6,
    )
    mm_text(ax, graph + centre - 1.3, verbs_y, "read", colour=fs.SLATE, ha="right")

    # 3. Three CyteBase stores, each with the genes A to D in its own order, its cells cut into blocks and marked by a
    # colour; beside the local store, the zero-copy virtual composite takes their cells block by block, the blocks of
    # all three shuffled together and the genes in one order, and the local store references it.
    sw, gap = 18.0, 2.25
    tops = []
    for s, (rows, order) in enumerate(zip(SOURCE_ROWS, SOURCE_GENE_ORDERS)):
        sx = COMPOSITE_LEFT_MM + s * (sw + gap)
        g.folder(ax, sx, cloud_y[0], sx + sw, cloud_y[1], tab=5.0, tab_h=1.8, z=2)
        columns = [GENE_COLUMNS[gene] for gene in order]
        mx, mws, my = sx + 3.6, sw - 5.0, matrix_y + 0.2
        counts_matrix(ax, mx, my, mws, matrix_h - 0.4, heat[rows][:, columns], kind="cell", bands=2)
        ax.add_patch(
            Rectangle(
                (mx - STRIP_MM - 0.4, my),
                STRIP_MM,
                matrix_h - 0.4,
                facecolor=fs.SOURCE_HUES[s],
                edgecolor="none",
                zorder=5,
            )
        )
        gene_names(ax, mx, my + matrix_h - 0.4, mws, order)
        tops.append(sx + sw / 2)
    mm_text(ax, tops[1] + 4.0, title_y + 0.7, "CyteBase stores", ha="center")
    g.lock(ax, tops[1] + 15.0, title_y + 0.7, s=1.9, z=6)
    virtual, sources = composite_rows(heat)
    vw, vy0 = 14.0, local_y[0] + 0.6
    vh = local_y[1] - 1.6 - vy0
    vx0 = MOUNT_RIGHT_MM + 17.0
    strip_x = vx0 - STRIP_MM - 0.4
    # The composite holds no counts of its own: its matrix has only the dashed edge drawn below, not a matrix's frame.
    ax.imshow(
        virtual,
        extent=(vx0, vx0 + vw, vy0, vy0 + vh),
        cmap=fs.SEQUENTIAL,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
        zorder=5,
    )
    th.strip(ax, strip_x, vy0, STRIP_MM, vh, [fs.SOURCE_HUES[s] for s in sources], z=5)
    # White gaps part the blocks, across the strip and the matrix, so the shuffled blocks read as units.
    row_h = vh / len(sources)
    for b in range(1, len(BLOCK_ORDER)):
        yy = vy0 + vh - b * BLOCK_ROWS * row_h
        ax.plot(
            [strip_x, vx0 + vw], [yy, yy], color=fs.WHITE, lw=1.0, solid_capstyle="butt", zorder=5.5
        )
    ax.add_patch(
        Rectangle(
            (vx0, vy0),
            vw,
            vh,
            facecolor="none",
            edgecolor=fs.INK,
            lw=fs.LW_AXIS,
            ls=(0, (2, 1.4)),
            zorder=6,
        )
    )
    gene_names(ax, vx0, vy0 + vh, vw, range(len(GENE_NAMES)))
    for i, (line, colour) in enumerate(
        (
            ("zero-copy virtual composite", fs.INK),
            ("genes aligned", fs.SLATE),
            ("blocks shuffled across stores", fs.SLATE),
        )
    ):
        mm_text(ax, vx0 + vw + 2.0, vy0 + vh - 2.2 - 2.6 * i, line, colour=colour, ha="left")
    # The sources' cells are referenced, not copied: dashed, merging from the three stores into the composite, which
    # the local store references in turn.
    merge_y = cloud_y[0] - 1.6
    composite_mid = vx0 + vw / 2
    for x in tops:
        ax.plot(
            [x, x],
            [cloud_y[0] - 0.3, merge_y],
            color=fs.SLATE,
            lw=fs.LW_THIN,
            ls=(0, (2.0, 1.4)),
            zorder=3,
        )
    ax.plot(
        [tops[0], tops[-1]],
        [merge_y, merge_y],
        color=fs.SLATE,
        lw=fs.LW_THIN,
        ls=(0, (2.0, 1.4)),
        zorder=3,
    )
    g.arrow(
        ax,
        (composite_mid, merge_y),
        (composite_mid, vy0 + vh + 2.4),
        lw=fs.LW_THIN,
        head=2.6,
        dashed=True,
        z=6,
    )
    link_y = vy0 + vh * 0.45
    g.arrow(
        ax,
        (strip_x - 0.4, link_y),
        (MOUNT_RIGHT_MM + 0.4, link_y),
        lw=fs.LW_THIN,
        head=3.0,
        dashed=True,
        z=6,
    )
    mm_text(ax, (MOUNT_RIGHT_MM + strip_x) / 2, link_y + 1.9, "referenced", colour=fs.SLATE)


# ---- what the remote Tabula Sapiens query returned, and what it took against a download -----------------------

# The UMAP's background shows every cell in slate tints by density; the B cells take the sequential ramp by MS4A1 from
# this share of the ramp up, so that no B-cell pixel is near white.
RAMP_FLOOR = 0.25
DENSITY_SHARE = (0.1, 0.55)
# The MS4A1 scale above the map: its bar's left edge and size, in millimetres from the map's top-left corner.
SCALE_BAR = (10.0, 2.0, 11.0, 1.3)
# The time and the data of the CyteBase query against querying CELLxGENE Census and downloading the source H5AD, in two
# charts right of the map and as tall as it, each this far from the map's left edge and this wide, under one legend.
COST_LEFT_MM = (34.0, 53.5)
COST_WIDTH_MM = 10.0
# Each chart: its title, the scale from the record's units to the axis's, and the axis's top and ticks. Time is in
# minutes, as everywhere in the figure.
COST_CHARTS = {
    "seconds": ("time (min)", 1 / 60, 5.5, (0, 2, 4)),
    "bytes": ("data (GB)", 1e-9, 70.0, (0, 20, 40, 60)),
}
# The CyteBase query in CyteArc's blue, the download in the comparators' slate, as the in-place access figure draws
# downloads, and the Census query in light grey.
QUERY_BARS = {
    "cytebase": CYTEARC_BAR,
    "census": fs.bar_style(fs.LIGHT),
    "download": COMPARATOR_BAR,
}


def tabula_query(ax, data: dict) -> None:
    """The UMAP of the 1.14 million Tabula Sapiens cells as the remote query read it, every cell in grey by density,
    and the B cells it selected coloured by the mean MS4A1 value of each pixel, with the MS4A1 scale above the map;
    right of it, the query's time and data against downloading the source H5AD, under one legend."""
    with np.load(REPO / data["raster"]) as raster:
        cells, b_cells, b_sum = raster["cells"], raster["bCells"], raster["bMs4a1Sum"]
    image = np.ones(cells.shape + (3,))
    occupied = cells > 0
    density = np.log1p(cells[occupied].astype(float))
    density /= np.quantile(density, 0.98)
    share = np.clip(
        DENSITY_SHARE[0] + (DENSITY_SHARE[1] - DENSITY_SHARE[0]) * density, *DENSITY_SHARE
    )
    image[occupied] = share[:, None] * np.array(to_rgb(fs.SLATE)) + (1 - share[:, None])
    selected = b_cells > 0
    level = np.clip(b_sum[selected] / b_cells[selected] / data["ms4a1Top"], 0, 1)
    image[selected] = fs.SEQUENTIAL(RAMP_FLOOR + (1 - RAMP_FLOOR) * level)[:, :3]

    rows, columns = np.nonzero(occupied)
    pad = 6
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, cells.shape[0] - 1)
    c0, c1 = max(columns.min() - pad, 0), min(columns.max() + pad, cells.shape[1] - 1)
    # Row 0 of the raster is the lowest UMAP 2 value, so the image is drawn from the lower-left corner.
    ax.imshow(
        image[r0 : r1 + 1, c0 : c1 + 1],
        origin="lower",
        cmap=fs.SEQUENTIAL,
        interpolation="nearest",
        aspect="auto",
        zorder=2,
    )
    label = data["bCellLabel"]
    ax.text(
        label["column"] - c0 + 16,
        label["row"] - r0 - 24,
        "B cells",
        ha="left",
        va="center",
        fontsize=fs.NOTE,
        color=fs.INK,
        zorder=4,
        path_effects=[patheffects.withStroke(linewidth=1.0, foreground=fs.WHITE)],
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", labelpad=1.2)
    ax.set_ylabel("UMAP 2", labelpad=1.2)

    # The MS4A1 scale above the map, read as a legend: name, low end, bar, high end.
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    pos = ax.get_position()
    left, above, bar_w, bar_h = SCALE_BAR
    bar = fs.add_axes(fig, pos.x0 * fw + left, pos.y1 * fh + above, bar_w, bar_h)
    bar.set_gid("tabula_query_scale")
    bar.imshow(
        np.linspace(RAMP_FLOOR, 1.0, 64)[None, :],
        cmap=fs.SEQUENTIAL,
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=(0, 1, 0, 1),
    )
    bar.set_xticks([])
    bar.set_yticks([])
    fs.frame_axes(bar)
    for x, text, ha in ((-0.06, "0", "right"), (1.06, f"{data['ms4a1Top']:.1f}", "left")):
        bar.text(
            x,
            0.5,
            text,
            transform=bar.transAxes,
            ha=ha,
            va="center",
            fontsize=fs.NOTE,
            color=fs.INK,
        )
    bar.text(
        -left / bar_w,
        0.5,
        "MS4A1",
        transform=bar.transAxes,
        ha="left",
        va="center",
        fontsize=fs.NOTE,
        color=fs.INK,
    )

    # The query's cost beside the map: each query's bar is the mean of three runs, the download one run, and brackets
    # give each other row's value over the CyteBase query's.
    charts = []
    for field, name, chart_left in zip(("seconds", "bytes"), ("time", "data"), COST_LEFT_MM):
        chart = fs.add_axes(
            fig, pos.x0 * fw + chart_left, pos.y0 * fh, COST_WIDTH_MM, pos.height * fh
        )
        chart.set_gid(f"tabula_query_{name}")
        cost_bars(chart, data["rows"], field, *COST_CHARTS[field])
        charts.append(chart)
    legend_above(
        charts[0],
        [Patch(**QUERY_BARS[row["key"]]) for row in data["rows"]],
        [row["label"] for row in data["rows"]],
        ncol=2,
        left_mm=-4.0,
        handlelength=0.8,
        columnspacing=0.6,
        labelspacing=0.15,
    )


def cost_bars(
    ax,
    rows: list[dict],
    field: str,
    title: str,
    scale: float,
    top: float,
    ticks: tuple[float, ...],
) -> None:
    """One bar per row under field on a linear axis, the CyteBase query first; over them, each other row's value over
    the query's as a bracket."""
    (centres,) = bar_centres(ax, [len(rows)])
    values = [row[field]["mean"] * scale for row in rows]
    for x, row, value in zip(centres, rows, values):
        ax.bar(x, shown_height(ax, value, top), width=BAR_MM, zorder=3, **QUERY_BARS[row["key"]])
    ratio = "timeOverQuery" if field == "seconds" else "dataOverQuery"
    ratio_brackets(ax, centres, [row[ratio] for row in rows[1:]], max(values), top)
    ax.set_xticks([])
    ax.set_ylim(0, top)
    ax.set_yticks(list(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.set_ylabel(title, labelpad=1.5)
    fs.frame_axes(ax)


# ---- the interferon response of each lineage, healthy against disease -----------------------------------------

# Each study's short name, by the start of its store's name, and each disease's; a row of the interferon panel names
# its study and disease, and a row of the progenitor forests its study.
STUDY_NAMES_BY_STORE = {
    "wilk_2020_": "Wilk",
    "perez_2022_": "Perez",
    "cambridge_2021_": "Cambridge",
    "jin_2021_individual_single_cell_rna_seq_pbmc_data_arunachalam": "Arunach.",
    "jin_2021_individual_single_cell_rna_seq_data_schulte_schrepping": "Schulte-S.",
    "ivanova_2023_": "Ivanova",
    "rabadam_2024_": "Rabadam",
    "van_der_wijst_2021_": "v.d. Wijst",
    "ren_2021_": "Ren",
    "lee_2020_": "Lee",
    "liu_2021_": "Liu",
    "ahern_2022_": "COMBAT",
    "yoshida_2022_": "Yoshida",
    "he_2025_": "ALTRA",
    "gong_2025_all_cells": "Gong",
    "gong_2025_human_immune_health_atlas": "Gong atlas",
    "gong_2025_sound_life": "Sound Life",
    "binvignat_2024_": "Binvignat",
    "heimlich_2024_": "Heimlich",
    "marteau_2026_": "Marteau",
    "rodriguezubreva_2022_": "Rodríguez",
    "unpub_0000_human_pbmc_glaucoma": "Glaucoma atlas",
    "kock_2025_aida": "AIDA",
    "yazar_2022_": "Yazar",
    "fachrul_2026_": "Fachrul",
}
DISEASE_NAMES = {
    "COVID-19": "COVID",
    "systemic lupus erythematosus": "SLE",
    "influenza": "flu",
    "juvenile dermatomyositis": "JDM",
    "cytomegalovirus infection": "CMV",
    "rheumatoid arthritis": "RA",
    "clonal hematopoiesis": "CHIP",
    "colorectal cancer": "CRC",
    "colon adenocarcinoma": "colon ca.",
    "common variable immunodeficiency": "CVID",
    "open-angle glaucoma": "glaucoma",
    "post-COVID-19 disorder": "PCC",
    "respiratory system disorder": "resp.",
    "sepsis": "sepsis",
}
LINEAGE_NAMES = {
    "T cells": "T",
    "NK and ILC": "NK",
    "B cells": "B",
    "Monocytes": "Mo",
    "Dendritic cells": "DC",
}
# Healthy donors take a blue and donors with the disease a rust, two group hues validated as a pair. Each side's ridge
# is the density of its T-cell scores, peaking this many rows high: a line in its hue over a light fill of it.
HEALTHY_RIDGE = fs.GROUPS[3]
DISEASE_RIDGE = fs.GROUPS[0]
RIDGE_FILL = 0.25
RIDGE_PEAK_ROWS = 1.4
SCORE_RANGE = (-3.0, 9.0)
# The AUC matrix sits this far left of the ridges, one column per lineage this wide, with the rows' names on its left; a
# lineage a store does not hold leaves its cell paper-coloured.
MATRIX_GAP_MM = 1.2
MATRIX_COLUMN_MM = 3.3
MISSING_CELL = fs.PAPER


def study_name(store: str) -> str:
    names = [name for prefix, name in STUDY_NAMES_BY_STORE.items() if store.startswith(prefix)]
    if len(names) != 1:
        raise ValueError(
            f"the store {store!r} matches {len(names)} study names; it needs exactly one"
        )
    return names[0]


def row_label(row: dict) -> str:
    """A row's study and disease in short; a study named for its disease keeps its name alone."""
    study = study_name(row["stores"][0])
    disease = "+".join(DISEASE_NAMES[label] for label in row["disease"].split(" || "))
    return study if disease.lower() in study.lower().split() else f"{study} {disease}"


def density_outline(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """A Gaussian kernel density of values on grid with Scott's bandwidth, scaled to peak at 1."""
    bandwidth = max(values.std(ddof=1), 0.3) * len(values) ** (-1 / 5)
    density = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2).sum(axis=1)
    return density / density.max()


def interferon_score(ax, data: dict) -> None:
    """Each study and disease of the screen on one row, interferon diseases above the others: on the left, the AUC of
    disease against healthy donors in each lineage; on the right, in the same rows, the density of the T-cell
    interferon scores of its healthy donors (blue) and of its donors with the disease (rust), as ridges."""
    rows = data["rows"]
    lineages = data["lineages"]
    n = len(rows)
    matrix_mm = len(lineages) * MATRIX_COLUMN_MM
    label_offset_pt = -(MATRIX_GAP_MM + matrix_mm + 0.5) * 72 * fs.MM
    grid = np.linspace(*SCORE_RANGE, 400)
    for i, row in enumerate(rows):
        base = n - 1 - i
        z = 2 + i * 0.1
        ax.plot(SCORE_RANGE, [base, base], color=fs.GRID, lw=fs.LW_THIN, zorder=z - 0.05)
        if row["ridge"] is not None:
            for side, colour in (("healthy", HEALTHY_RIDGE), ("cases", DISEASE_RIDGE)):
                height = density_outline(np.array(row["ridge"][side]), grid) * RIDGE_PEAK_ROWS
                shown = height > 0.01 * RIDGE_PEAK_ROWS
                ax.fill_between(
                    grid[shown],
                    base,
                    base + height[shown],
                    facecolor=fs.tint(colour, RIDGE_FILL),
                    edgecolor="none",
                    alpha=0.75,
                    zorder=z,
                )
                ax.plot(grid[shown], base + height[shown], color=colour, lw=0.7, zorder=z + 0.02)
        ax.annotate(
            row_label(row),
            (0.0, base),
            xycoords=("axes fraction", "data"),
            xytext=(label_offset_pt, 0.2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=fs.NOTE,
            color=fs.INK,
        )
    # The line between the last row of an interferon disease and the first control: on the boundary of their rows,
    # which each run from their base to the next row's base, so it passes under one label and over the next.
    controls = next((i for i, row in enumerate(rows) if not row["interferon"]), n)
    split = n - controls
    top = n - 1 + RIDGE_PEAK_ROWS + 0.1
    ax.set_xlim(*SCORE_RANGE)
    ax.set_xticks([0, 4, 8])
    ax.set_ylim(-0.4, top)
    ax.set_yticks([])
    ax.axhline(split, color=fs.INK, lw=fs.LW_THIN, ls=(0, (2, 1.5)), zorder=9)
    ax.axvline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlabel("T-cell score (SD)", labelpad=1.0)
    # Over the ridges, their legend; the screen's cells and donors are in the panel's title.
    legend_above(
        ax,
        [
            Patch(facecolor=fs.tint(colour, RIDGE_FILL), edgecolor=colour, lw=0.7)
            for colour in (HEALTHY_RIDGE, DISEASE_RIDGE)
        ],
        ["healthy", "disease"],
        ncol=1,
        handlelength=1.0,
        labelspacing=0.15,
    )
    # The matrix: one cell per row and lineage, in the ridges' rows, left of them.
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    pos = ax.get_position()
    matrix_left = pos.x0 - (MATRIX_GAP_MM + matrix_mm) / fw
    matrix = fig.add_axes([matrix_left, pos.y0, matrix_mm / fw, pos.height])
    matrix.set_gid("interferon_score_lineages")
    values = np.full((n, len(lineages)), np.nan)
    for i, row in enumerate(rows):
        for j, lineage in enumerate(lineages):
            if row["auc"][lineage] is not None:
                values[n - 1 - i, j] = row["auc"][lineage]
    colours = fs.DIVERGING.with_extremes(bad=MISSING_CELL)
    matrix.imshow(
        np.ma.masked_invalid(values),
        cmap=colours,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0, len(lineages), 0, n),
        zorder=2,
    )
    for j in range(1, len(lineages)):
        matrix.axvline(j, color=fs.WHITE, lw=0.3, zorder=3)
    for b in range(1, n):
        matrix.axhline(b, color=fs.WHITE, lw=0.3, zorder=3)
    matrix.axhline(split, color=fs.INK, lw=fs.LW_THIN, ls=(0, (2, 1.5)), zorder=9)
    matrix.set_xlim(0, len(lineages))
    matrix.set_ylim(*ax.get_ylim())
    matrix.set_yticks([])
    matrix.set_xticks(np.arange(len(lineages)) + 0.5)
    matrix.set_xticklabels([LINEAGE_NAMES[lineage] for lineage in lineages], fontsize=fs.NOTE)
    matrix.xaxis.tick_top()
    matrix.tick_params(axis="x", length=0, pad=1.0)
    for spine in matrix.spines.values():
        spine.set_visible(False)
    # The colour scale under the matrix: AUC from 0, healthy higher, to 1, disease higher.
    scale = fig.add_axes([matrix_left, pos.y0 - 2.0 / fh, matrix_mm / fw, 1.0 / fh])
    scale.set_gid("interferon_score_scale")
    scale.imshow(np.linspace(0, 1, 64)[None, :], cmap=colours, aspect="auto", extent=(0, 1, 0, 1))
    scale.set_yticks([])
    scale.set_xticks([0, 0.5, 1])
    scale.set_xticklabels(["0", ".5", "1"], fontsize=fs.NOTE)
    scale.tick_params(axis="x", length=1.0, pad=0.8)
    scale.set_xlabel("AUC", labelpad=0.5)
    for spine in scale.spines.values():
        spine.set_linewidth(fs.LW_THIN)
        spine.set_color(fs.INK)


# ---- the ten-study composite: its UMAP by study, and its disk and memory against copies ----------------------

# The composite's studies by the cohort analysis's dataset keys: the collection's first author or the cohort it
# republished, as the interferon panel names datasets, with the atlas or project where the study has one.
STUDY_NAMES = {
    "aida": "Kock (AIDA)",
    "yazar": "Yazar",
    "combat": "COMBAT",
    "covid_atlas": "Cambridge",
    "ivanova": "Ivanova",
    "hao": "Hao",
    "arthritis": "Binvignat",
    "jdm": "Rabadam",
    "schulte_schrepping": "Schulte-S.",
    "lee": "Lee",
}
# One colour per study, largest study first: the six group hues, then four other tokens far from them and from one
# another.
STUDY_COLOURS = (*fs.GROUPS, fs.MODULES["extend"].strong, fs.WINE, fs.INK, fs.SLATE)
# The study list beside the map: a swatch and a name per row, this far apart.
STUDY_ROW_MM = 2.3
# The routes in the order the charts draw them, with their names and bars: CyteArc's blue, BPCells' green and Scanpy's
# orange as in the analysis-scaling figure and the storage chart; AnnData in memory, which concatenates the files without
# analysing them, takes a lighter tint of the orange. It sits between the two routes of similar disk, so their labels stay apart.
ROUTE_ORDER = ("composite", "bpcells", "in_memory", "on_disk")
ROUTE_NAMES = {
    "composite": "CyteArc",
    "bpcells": "BPCells",
    "on_disk": "Scanpy (Dask)",
    "in_memory": "AnnData (memory)",
}
ROUTE_BARS = {
    "composite": CYTEARC_BAR,
    "bpcells": tool_bar("bpcells"),
    "on_disk": tool_bar("scanpy"),
    "in_memory": {
        "facecolor": fs.tint(fs.ENTITIES["scanpy"].colour, 0.4),
        "edgecolor": fs.INK,
        "linewidth": fs.LW_BAR,
    },
}
# Each chart: its title, the scale from the record's GiB to the axis's unit, the axis's top and ticks; the tops leave
# room over the bars for the ratios' brackets. Disk is in decimal gigabytes, as storage is sold; memory in GiB, as
# containers are sized.
GIB_IN_GB = 1024**3 / 1e9
FOOTPRINT_CHARTS = {
    "disk": ("local disk used (GB)", GIB_IN_GB, 250.0, (0, 100, 200)),
    "memory": ("peak memory (GiB)", 1.0, 320.0, (0, 100, 200, 300)),
}


def study_map(ax, data: dict) -> list[str]:
    """The composite's UMAP from CyteArc's Harmony analysis: each pixel takes the study of one of its cells drawn at
    random, as a scatter of every cell in random order would show it, faded toward white where few cells fall; returns
    the studies, largest first, in the colours the map gives them."""
    with np.load(REPO / data["raster"]) as raster:
        counts = raster["studyCells"].astype(float)
        studies = [str(study) for study in raster["studies"]]
    order = [s["study"] for s in sorted(data["studies"], key=lambda s: -s["cells"])]
    if sorted(order) != sorted(studies) or len(order) > len(STUDY_COLOURS):
        raise ValueError(f"the map's studies {studies} do not match the panel's {order}")
    image = fs.category_raster(counts, [STUDY_COLOURS[order.index(study)] for study in studies])
    ax.imshow(
        image, origin="lower", cmap=fs.SEQUENTIAL, interpolation="nearest", aspect="auto", zorder=2
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", labelpad=1.2)
    ax.set_ylabel("UMAP 2", labelpad=1.2)
    return order


def amount(value: float) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def footprint_chart(chart, routes: dict, group: str) -> None:
    """One route per bar, CyteArc first, memory bars with the range of three runs; over the bars, each other route's
    value over CyteArc's as a bracket."""
    title, scale, top, ticks = FOOTPRINT_CHARTS[group]
    (centres,) = bar_centres(chart, [len(ROUTE_ORDER)])
    values, highs = [], []
    for x, key in zip(centres, ROUTE_ORDER):
        route = routes[key]
        if group == "disk":
            value = high = sum(part["GiB"] for part in route["disk"]) * scale
        else:
            value, high = route["memoryGiB"]["mean"], route["memoryGiB"]["high"]
            range_bar(chart, x, route["memoryGiB"]["low"], high, cap=0.5)
        chart.bar(x, shown_height(chart, value, top), width=BAR_MM, zorder=3, **ROUTE_BARS[key])
        values.append(value)
        highs.append(high)
    ratio_brackets(chart, centres, [value / values[0] for value in values[1:]], max(highs), top)
    chart.set_xticks([])
    chart.set_ylim(0, top)
    chart.set_yticks(list(ticks))
    chart.yaxis.set_major_formatter(FuncFormatter(plain))
    chart.grid(True, axis="y", which="major")
    chart.set_axisbelow(True)
    chart.set_ylabel(title, labelpad=1.5)
    fs.frame_axes(chart)


# Between the study list and the charts, how well the Harmony analysis mixes the studies and keeps the cell types apart,
# against the unintegrated analysis, as a narrow chart: the two metrics along x, each named under its column, on a
# scale from 0 to 1 along y. The batch entropy is the entropy of the studies among each cell's neighbours of its own
# cell type over the entropy of that type's studies; the kNN purity, the share of each cell's neighbours that share its
# cell type. In each column, the unintegrated analysis's value open in slate left of the column's centre and the
# Harmony analysis's filled in CyteArc's blue right of it, INTEGRATION_DODGE apart from the centre, so neither hides the
# other; the chart stops INTEGRATION_LEGEND_MM under the charts' top for its legend over it.
INTEGRATION_METRICS = (("studyMixing", "batch\nentropy"), ("sameType", "kNN\npurity"))
INTEGRATION_PHASES = (
    ("unintegrated", "unintegrated", fs.SLATE),
    ("integrated", "Harmony", CYTEARC_BAR["facecolor"]),
)
INTEGRATION_DODGE = 0.102


def integration_strip(ax, integration: dict) -> None:
    """The batch entropy and the kNN purity of the unintegrated and the Harmony analyses, one column each, with the
    legend over the chart."""
    xs = list(range(len(INTEGRATION_METRICS)))
    for key, _ in INTEGRATION_METRICS:
        values = [integration[phase][key] for phase, _, _ in INTEGRATION_PHASES]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{key} values {values} are not between 0 and 1")
    for (phase, _, colour), side, variant in zip(INTEGRATION_PHASES, (-1, 1), (True, False)):
        fs.points(
            ax,
            [x + side * INTEGRATION_DODGE for x in xs],
            [integration[phase][key] for key, _ in INTEGRATION_METRICS],
            colour,
            variant=variant,
            zorder=3,
        )
    ax.set_xlim(-0.42, len(xs) - 0.58)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in INTEGRATION_METRICS], fontsize=fs.NOTE)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.0, 1.06)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    fs.frame_axes(ax)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=fs.MARKER_MS, mfc=fs.WHITE, mec=fs.SLATE),
        plt.Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=fs.MARKER_MS,
            mfc=CYTEARC_BAR["facecolor"],
            mec=fs.WHITE,
        ),
    ]
    legend_above(
        ax,
        handles,
        [label for _, label, _ in INTEGRATION_PHASES],
        ncol=1,
        left_mm=-2.5,
        handlelength=0.8,
        handletextpad=0.35,
        labelspacing=0.12,
    )


# The panel left to right, in millimetres from its cell's left edge: the map (the cell's plot), the study list, the
# metrics' chart, then the disk and memory charts. All three charts stand on PANEL_FOOT_MM, leaving room under them for
# the metrics' names and the routes' legend; the disk and memory charts reach PANEL_TOP_MM, under the cell's top edge,
# so no empty band sits between them and the row above, and the metrics' chart stops INTEGRATION_LEGEND_MM lower.
STUDY_LIST_GAP_MM = 1.5
INTEGRATION_LEFT_MM, INTEGRATION_WIDTH_MM = 46.5, 14.5
FOOTPRINT_LEFT_MM = (70.0, 91.0)
FOOTPRINT_WIDTH_MM = 12.5
PANEL_FOOT_MM, PANEL_TOP_MM = 9.5, 36.5
INTEGRATION_LEGEND_MM = 9.0


def composite_umap(ax, data: dict) -> None:
    """The composite's UMAP from CyteArc's Harmony analysis, every cell coloured by its study, and beside it the
    studies, largest first, each with its colour; right of them, the analysis's batch entropy and kNN purity against the
    unintegrated analysis's, and the local disk and peak memory of each route to the cohort, under one legend."""
    fig = ax.figure
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    pos = ax.get_position()
    cell_left = pos.x0 * fw - CELLS["composite_umap"].plot[0]
    cell_foot = pos.y0 * fh - CELLS["composite_umap"].plot[1]
    order = study_map(ax, data["umap"])
    names = g.mm_axes(fig, pos.x1 * fw + STUDY_LIST_GAP_MM, pos.y0 * fh, 18.0, pos.height * fh)
    names.set_gid("composite_umap_studies")
    top = pos.height * fh - 0.8
    for i, study in enumerate(order):
        y = top - i * STUDY_ROW_MM
        names.add_patch(
            Rectangle(
                (0.0, y - 0.8),
                1.6,
                1.6,
                facecolor=STUDY_COLOURS[i],
                edgecolor="none",
                zorder=3,
                clip_on=False,
            )
        )
        mm_text(names, 2.4, y, STUDY_NAMES[study], ha="left")
    strip = fs.add_axes(
        fig,
        cell_left + INTEGRATION_LEFT_MM,
        cell_foot + PANEL_FOOT_MM,
        INTEGRATION_WIDTH_MM,
        PANEL_TOP_MM - INTEGRATION_LEGEND_MM - PANEL_FOOT_MM,
    )
    strip.set_gid("composite_umap_integration")
    integration_strip(strip, data["integration"])
    routes = {route["key"]: route for route in data["routes"]}
    charts = []
    for group, left in zip(("disk", "memory"), FOOTPRINT_LEFT_MM):
        chart = fs.add_axes(
            fig,
            cell_left + left,
            cell_foot + PANEL_FOOT_MM,
            FOOTPRINT_WIDTH_MM,
            PANEL_TOP_MM - PANEL_FOOT_MM,
        )
        chart.set_gid(f"composite_umap_{group}")
        footprint_chart(chart, routes, group)
        charts.append(chart)
    charts[0].legend(
        [Patch(**ROUTE_BARS[key]) for key in ROUTE_ORDER],
        [ROUTE_NAMES[key] for key in ROUTE_ORDER],
        loc="upper left",
        bbox_to_anchor=(-3.0 / width_mm(charts[0]), -1.2 / height_mm(charts[0])),
        borderaxespad=0.0,
        ncol=2,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.12,
        columnspacing=0.8,
    )


# ---- circulating progenitors pulled out of the blood corpus ----------------------------------------------------

# Each state's colour on the map and its short name there; cells that passed the selection without being progenitors
# share one light grey as mature cells.
STATE_COLOURS = {
    "HSC/MPP": fs.GROUPS[0],
    "erythroid": fs.GROUPS[1],
    "lymphoid-primed": fs.GROUPS[2],
    "cycling": fs.GROUPS[3],
    "myeloid": fs.GROUPS[4],
    "megakaryocyte": fs.GROUPS[5],
    "basophil/mast": fs.WINE,
    "dendritic": fs.INK,
    "B-lymphoid": fs.MODULES["extend"].strong,
    "mature": fs.LIGHT,
}
STATE_NAMES = {
    "HSC/MPP": "HSC/MPP",
    "erythroid": "Ery",
    "lymphoid-primed": "Lym",
    "cycling": "Cyc",
    "myeloid": "My",
    "megakaryocyte": "MK",
    "basophil/mast": "Ba",
    "dendritic": "DC",
    "B-lymphoid": "B",
    "mature": "mature",
}
# The funnels: the datasets of all blood cells and, beside them, the datasets of CD34+ cells only. Each level is a band
# as wide as the square root of its cells, FUNNEL_WIDEST_MM for the most cells read, joined to the next by a band of its
# tint; where 25,000 cells of each CD34+ dataset were drawn at random, the join is outlined, not filled. Each count sits
# inside its band when the band is wide enough and right of it otherwise, with the share of the cells read that the
# selection kept under the second level's count. Down the left, an arrow per step with what it did.
FUNNEL_LEVELS = ("cellsRead", "selected", "progenitors")
FUNNEL_STEPS = ("screen for\nprogenitors", "integrate and\ncluster")
FUNNEL_WIDEST_MM = 17.0
FUNNEL_BAND_MM = 3.0
# Each level's centre, in millimetres from the plot's foot, and each funnel's centre from its left.
FUNNEL_LEVELS_MM = (41.0, 23.5, 6.0)
FUNNEL_CENTRES_MM = (21.0, 32.2)
# A band at least this wide holds its count; a count beside a band keeps this gap, and a share sits this far under it.
FUNNEL_INSIDE_MM = 8.0
FUNNEL_TEXT_GAP_MM = 0.8
FUNNEL_SHARE_DROP_MM = 2.6
# The step arrows' x and their gap to the bands they join.
FUNNEL_ARROW_MM = 1.5
FUNNEL_ARROW_GAP_MM = 0.8
FUNNEL_BAND_TINT, FUNNEL_JOIN_TINT = 0.55, 0.22
FUNNEL_ARROW = {
    "arrowstyle": "-|>",
    "lw": fs.LW_THIN,
    "shrinkA": 0,
    "shrinkB": 0,
    "mutation_scale": 5,
}
BLOOD_SOURCE = fs.GROUPS[3]
REFERENCE_SOURCE = fs.SLATE
SOURCES = (("blood", BLOOD_SOURCE, "all blood"), ("enriched", REFERENCE_SOURCE, "CD34+ only"))
SAMPLED = {"blood": False, "enriched": True}


def cell_count(cells: int) -> str:
    if cells >= 1e6:
        return f"{cells / 1e6:.1f}M"
    if cells >= 1e5:
        return f"{cells / 1e3:.0f}k"
    return f"{cells / 1e3:.1f}k"


def share_text(share: float) -> str:
    """A share as a percentage with two significant digits: 0.10%, 2.4%, 79%."""
    percent = share * 100
    decimals = max(0, 1 - int(np.floor(np.log10(percent))))
    return f"{percent:.{decimals}f}%"


def funnel_width(cells: float, widest: float) -> float:
    """A band's width in millimetres: FUNNEL_WIDEST_MM for ``widest`` cells, and in proportion to the square root of
    its cells below that."""
    if not 0 < cells <= widest:
        raise ValueError(
            f"a funnel band holds {cells} cells, not between 0 and the widest band's {widest}"
        )
    return FUNNEL_WIDEST_MM * float(np.sqrt(cells / widest))


def progenitor_funnel(ax, data: dict) -> None:
    """From every cell the screen read to the cells in progenitor states, as two funnels, the datasets of all blood
    cells and those of CD34+ cells only, with each level's cells and each step's action."""
    w, h = width_mm(ax), height_mm(ax)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    sources = data["sources"]
    widest = max(sources[source]["cellsRead"] for source, _, _ in SOURCES)
    levels, half = FUNNEL_LEVELS_MM, FUNNEL_BAND_MM / 2
    for (source, colour, label), centre in zip(SOURCES, FUNNEL_CENTRES_MM):
        cells = [sources[source][key] for key in FUNNEL_LEVELS]
        widths = [funnel_width(n, widest) for n in cells]
        mm_text(ax, centre, levels[0] + half + 1.0, label, colour=fs.SLATE, va="bottom")
        for i, (y, width) in enumerate(zip(levels, widths)):
            left, right = centre - width / 2, centre + width / 2
            ax.fill(
                [left, right, right, left],
                [y - half, y - half, y + half, y + half],
                facecolor=fs.tint(colour, FUNNEL_BAND_TINT),
                edgecolor=colour,
                lw=fs.LW_THIN,
                zorder=3,
            )
            if width >= FUNNEL_INSIDE_MM:
                mm_text(ax, centre, y, cell_count(cells[i]), size=fs.LABEL)
            else:
                x = right + FUNNEL_TEXT_GAP_MM
                mm_text(ax, x, y, cell_count(cells[i]), size=fs.LABEL, ha="left")
                if FUNNEL_LEVELS[i] == "selected":
                    share = share_text(sources[source]["selectedShare"])
                    mm_text(ax, x, y - FUNNEL_SHARE_DROP_MM, share, colour=fs.SLATE, ha="left")
            if i + 1 < len(levels):
                below, narrow = levels[i + 1], widths[i + 1] / 2
                sampled = SAMPLED[source] and FUNNEL_LEVELS[i] == "selected"
                ax.fill(
                    [left, right, centre + narrow, centre - narrow],
                    [y - half, y - half, below + half, below + half],
                    facecolor="none" if sampled else fs.tint(colour, FUNNEL_JOIN_TINT),
                    edgecolor=colour if sampled else "none",
                    lw=fs.LW_THIN,
                    ls=(0, (1.5, 1.5)),
                    zorder=2,
                )
    for upper, lower, step in zip(levels, levels[1:], FUNNEL_STEPS):
        start, end = upper - half - FUNNEL_ARROW_GAP_MM, lower + half + FUNNEL_ARROW_GAP_MM
        ax.annotate(
            "",
            xy=(FUNNEL_ARROW_MM, end),
            xytext=(FUNNEL_ARROW_MM, start),
            arrowprops={**FUNNEL_ARROW, "color": fs.INK},
        )
        mm_text(ax, FUNNEL_ARROW_MM + 1.2, (start + end) / 2, step, ha="left")


def progenitor_raster(ax, data: dict, key: str, colours: list[str]) -> None:
    with np.load(REPO / data["raster"]) as raster:
        counts = raster[key].astype(float)
    image = fs.category_raster(counts, colours, pad=None)
    rows, columns = counts.shape[1:]
    ax.imshow(
        image,
        origin="lower",
        cmap=fs.SEQUENTIAL,
        interpolation="nearest",
        extent=(0, columns, 0, rows),
        zorder=2,
    )
    # The frame takes the map's own shape at the foot of the plot's box, so no empty band sits inside it.
    ax.set_aspect("equal", adjustable="box", anchor="S")
    ax.set_xticks([])
    ax.set_yticks([])


def progenitor_states(ax, data: dict) -> None:
    """The analysed progenitors' UMAP, each bin taking the state of one of its cells, each state named at the median of
    its cells; mature cells, which passed the selection without being progenitors, in grey."""
    order = [state["state"] for state in data["states"]]
    progenitor_raster(ax, data, "states", [STATE_COLOURS[state] for state in order])
    for state in data["states"]:
        text = ax.text(
            state["column"] + 0.5,
            state["row"] + 0.5,
            STATE_NAMES[state["state"]],
            ha="center",
            va="center",
            fontsize=fs.NOTE,
            color=fs.INK,
            zorder=6,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=1.6, foreground=fs.WHITE)])
    # The source map under this one carries the shared UMAP 1 label.
    ax.set_ylabel("UMAP 2", labelpad=1.0)


SOURCES_LEGEND_TOP_MM = 2.85


def progenitor_sources(ax, data: dict) -> None:
    """The same map, each bin taking the source of one of its cells: the datasets of all blood cells or the datasets
    of CD34+ cells only."""
    progenitor_raster(ax, data, "sources", [BLOOD_SOURCE, REFERENCE_SOURCE])
    # The legend sits between the header and the map, its top SOURCES_LEGEND_TOP_MM over the map.
    legend_above(
        ax,
        [Patch(facecolor=colour, edgecolor="none") for _, colour, _ in SOURCES],
        [label for _, _, label in SOURCES],
        ncol=2,
        left_mm=-0.6,
        top_mm=SOURCES_LEGEND_TOP_MM,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.4,
        borderpad=0.0,
        fontsize=fs.NOTE,
    )
    ax.set_xlabel("UMAP 1", labelpad=1.0)
    ax.set_ylabel("UMAP 2", labelpad=1.0)


# Per study, the progenitors per 10,000 blood cells of its healthy donors and of its donors with COVID-19, in the colours
# of the interferon ridges, joined by the study's line; the caption names the studies. Over both, a bracket with the
# P value of the odds ratio pooled over the studies, which the caption gives.
COVID_GROUPS = (
    ("healthyPer10k", "healthyDonors", "healthy", HEALTHY_RIDGE),
    ("covidPer10k", "covidDonors", "COVID-19", DISEASE_RIDGE),
)
RATE_LIMITS = (0.25, 2000.0)
RATE_TICKS = (1, 10, 100)
# Each point's area grows with its study-condition's donors, DONOR_AREA square points per donor before the grammar's
# marker scale; the key over the plot shows DONOR_KEY donors.
DONOR_AREA = 0.3
DONOR_KEY = (10, 50, 100)


def donor_area(donors: int) -> float:
    """A point's scatter size, in square points before fs.points scales it, for a study-condition's donors."""
    if donors < 1:
        raise ValueError(f"a study-condition with {donors} donors has no point to draw")
    return DONOR_AREA * donors


def progenitor_covid(ax, data: dict) -> None:
    """Per study, progenitors per 10,000 blood cells in healthy donors and in donors with COVID-19, one line per study,
    under a bracket with the P value of the pooled odds ratio."""
    studies = data["covid"]
    for study in studies:
        rates = [study[key] for key, _, _, _ in COVID_GROUPS]
        if not all(RATE_LIMITS[0] < rate < RATE_LIMITS[1] for rate in rates):
            raise ValueError(
                f"a study's progenitor rates {rates} fall outside the axis {RATE_LIMITS}"
            )
        ax.plot([0, 1], rates, color=fs.SLATE, lw=fs.LW_THIN, zorder=2)
    for x, (key, donors, _, colour) in enumerate(COVID_GROUPS):
        fs.points(
            ax,
            [x] * len(studies),
            [study[key] for study in studies],
            colour,
            size=np.array([donor_area(study[donors]) for study in studies]),
        )
    ax.set_yscale("log")
    ax.set_ylim(*RATE_LIMITS)
    ax.set_yticks(list(RATE_TICKS))
    fs.plain_log_axis(ax, "y")
    log_minor_ticks(ax.yaxis)
    ax.set_xlim(-0.32, 1.32)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label for _, _, label, _ in COVID_GROUPS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("progenitors\nper 10k cells", labelpad=1.0)
    # The bracket sits RATIO_CLEARANCE_MM over the highest rate, its ends dropping 0.6 mm, on the log axis.
    pool = data["selections"]["default"]
    per_mm = np.log10(RATE_LIMITS[1] / RATE_LIMITS[0]) / height_mm(ax)
    highest = max(study[key] for study in studies for key, _, _, _ in COVID_GROUPS)
    y = highest * 10 ** (RATIO_CLEARANCE_MM * per_mm)
    pair_bracket(
        ax,
        0,
        1,
        y,
        y - y / 10 ** (0.6 * per_mm),
        [(p_text(pool["p"]), fs.INK)],
    )
    # The size key over the plot: a slate point for each of DONOR_KEY donors, drawn as fs.points scales its markers.
    handles = [
        plt.Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=2 * np.sqrt(1.4 * donor_area(donors) / np.pi),
            mfc=fs.SLATE,
            mec=fs.WHITE,
            mew=0.5,
        )
        for donors in DONOR_KEY
    ]
    labels = [str(donors) for donors in DONOR_KEY[:-1]] + [f"{DONOR_KEY[-1]} donors"]
    legend_above(
        ax,
        handles,
        labels,
        ncol=len(DONOR_KEY),
        left_mm=-10.0,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.8,
    )


# Each disease's odds ratio of a donor's cell being a progenitor against the healthy donors of its own studies, pooled
# over them where several have it, with its 95% interval: one row per disease from the highest ratio down, named as in
# the interferon rows, COVID-19 and sepsis in their colours in the panels that follow and every other disease in slate,
# about a dashed line at an odds ratio of 1.
ODDS_LIMITS = (0.05, 40.0)
ODDS_TICKS = (0.1, 1, 10)
DISEASE_COLOURS = {"COVID-19": DISEASE_RIDGE, "sepsis": fs.WINE}


def progenitor_diseases(ax, data: dict) -> None:
    """Per disease, the odds ratio of a cell being a progenitor against healthy donors of the same studies, with its
    95% interval, from the highest ratio at the top."""
    diseases = data["diseases"]
    unnamed = sorted({entry["disease"] for entry in diseases} - set(DISEASE_NAMES))
    if unnamed:
        raise ValueError(f"diseases without a name in DISEASE_NAMES: {unnamed}")
    rows = len(diseases)
    for i, entry in enumerate(diseases):
        if not ODDS_LIMITS[0] < entry["low"] <= entry["high"] < ODDS_LIMITS[1]:
            raise ValueError(
                f"the interval of {entry['disease']}, {entry['low']:.3g} to {entry['high']:.3g}, falls outside "
                f"the axis {ODDS_LIMITS}"
            )
        y = rows - 1 - i
        colour = DISEASE_COLOURS.get(entry["disease"], fs.SLATE)
        ax.plot(
            [entry["low"], entry["high"]],
            [y, y],
            color=colour,
            lw=fs.LW_AXIS,
            solid_capstyle="butt",
            zorder=2,
        )
        fs.points(ax, [entry["ratio"]], [y], colour)
    ax.axvline(1.0, color=fs.SLATE, lw=fs.LW_THIN, ls=(0, (2.0, 1.6)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(*ODDS_LIMITS)
    ax.set_xticks(list(ODDS_TICKS))
    fs.plain_log_axis(ax, "x")
    log_minor_ticks(ax.xaxis)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(
        [DISEASE_NAMES[entry["disease"]] for entry in reversed(diseases)], fontsize=fs.NOTE
    )
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel("odds ratio", labelpad=1.0)


# COMBAT's donors by the cohort the authors recorded: healthy volunteers, COVID-19 from mild to critical in deepening
# tints of the COVID-19 rust, and sepsis in its wine; each donor's progenitors per 10,000 blood cells a point, spread
# across its cohort's column, and the cohort's own rate, its donors' cells pooled, a bar. A donor without a progenitor
# sits open on the axis floor. Over the healthy volunteers and the COVID-19 cohorts, a bracket with the P value of the
# odds ratio per step of severity, which the caption gives with the odds ratio of sepsis.
COHORT_COLOURS = {
    "healthy": HEALTHY_RIDGE,
    "mild": fs.tint(DISEASE_RIDGE, 0.45),
    "severe": fs.tint(DISEASE_RIDGE, 0.75),
    "critical": DISEASE_RIDGE,
    "sepsis": fs.WINE,
}
COHORT_STEPS = ("healthy", "mild", "severe", "critical")
COHORT_LIMITS = (0.5, 5000.0)
COHORT_TICKS = (1, 10, 100, 1000)
# Half the width of a column's spread of points, and of its bar, in columns; the spread's seed keeps each render alike.
COHORT_SPREAD = 0.24
COHORT_BAR = 0.32
COHORT_SEED = 0


def progenitor_cohorts(ax, data: dict) -> None:
    """Each COMBAT donor's progenitors per 10,000 blood cells by cohort, with each cohort's pooled rate, under a bracket
    over healthy volunteers and the COVID-19 cohorts with the P value of the trend over them."""
    cohorts = data["cohorts"]["cohorts"]
    names = [cohort["cohort"] for cohort in cohorts]
    if names[: len(COHORT_STEPS)] != list(COHORT_STEPS) or set(names) - set(COHORT_COLOURS):
        raise ValueError(
            f"the cohorts {names} are not healthy volunteers and COVID-19 from mild to critical, then others "
            f"with a colour in COHORT_COLOURS"
        )
    rng = np.random.default_rng(COHORT_SEED)
    for x, cohort in enumerate(cohorts):
        rates = np.asarray(cohort["donorsPer10k"], dtype=float)
        if (rates < 0).any() or (rates >= COHORT_LIMITS[1]).any():
            raise ValueError(
                f"a {cohort['cohort']} donor's rate falls outside the axis {COHORT_LIMITS}: {rates.max():.3g}"
            )
        colour = COHORT_COLOURS[cohort["cohort"]]
        offsets = x + rng.uniform(-COHORT_SPREAD, COHORT_SPREAD, len(rates))
        seen = rates > 0
        fs.points(ax, offsets[seen], rates[seen], colour, size=fs.MARKER * 0.55)
        if not seen.all():
            fs.points(
                ax,
                offsets[~seen],
                np.full(int((~seen).sum()), COHORT_LIMITS[0]),
                colour,
                variant=True,
                size=fs.MARKER * 0.55,
                clip_on=False,
            )
        ax.plot(
            [x - COHORT_BAR, x + COHORT_BAR],
            [cohort["per10k"]] * 2,
            color=fs.INK,
            lw=fs.LW_PRIMARY,
            solid_capstyle="butt",
            zorder=5,
        )
    ax.set_yscale("log")
    ax.set_ylim(*COHORT_LIMITS)
    ax.set_yticks(list(COHORT_TICKS))
    fs.plain_log_axis(ax, "y")
    log_minor_ticks(ax.yaxis)
    ax.set_xlim(-0.55, len(cohorts) - 0.45)
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("progenitors\nper 10k cells", labelpad=1.0)
    # The bracket sits RATIO_CLEARANCE_MM over the highest donor of the four cohorts it spans.
    per_mm = np.log10(COHORT_LIMITS[1] / COHORT_LIMITS[0]) / height_mm(ax)
    highest = max(max(cohort["donorsPer10k"]) for cohort in cohorts[: len(COHORT_STEPS)])
    y = highest * 10 ** (RATIO_CLEARANCE_MM * per_mm)
    pair_bracket(
        ax,
        0,
        len(COHORT_STEPS) - 1,
        y,
        y - y / 10 ** (0.6 * per_mm),
        [(p_text(data["cohorts"]["trend"]["p"]), fs.INK)],
    )


# Each progenitor state's share of the progenitors from all blood and from the datasets of CD34+ cells only, one row per
# state from the commonest in all blood, named as on the map, on a log axis: all blood's share filled in its blue, the
# CD34+ datasets' share open in their grey, the two joined by a line; the legend sits over the plot.
SHARE_LIMITS = (0.1, 100.0)
SHARE_TICKS = (0.1, 1, 10, 100)
# The rows' names as on the map, the longest over two lines so the names' column stays narrow.
SHARE_ROW_NAMES = {**STATE_NAMES, "HSC/MPP": "HSC/\nMPP"}
SHARE_TOP_GAP = 0.35


def progenitor_state_shares(ax, data: dict) -> None:
    """The share of each progenitor state among the progenitors from all blood and from the CD34+ datasets."""
    shares = data["stateShares"]
    states = sorted(shares["blood"], key=lambda state: -shares["blood"][state])
    # The top row, whose name takes two lines, sits SHARE_TOP_GAP rows higher than the even step.
    ys = [len(states) - 1 + SHARE_TOP_GAP, *range(len(states) - 2, -1, -1)]
    for state, y in zip(states, ys):
        pair = [100 * shares[source][state] for source in ("blood", "enriched")]
        if not all(SHARE_LIMITS[0] < share < SHARE_LIMITS[1] for share in pair):
            raise ValueError(f"{state}'s shares {pair} fall outside the axis {SHARE_LIMITS}")
        ax.plot(pair, [y, y], color=fs.SLATE, lw=fs.LW_THIN, zorder=2)
    fs.points(
        ax,
        [100 * shares["enriched"][state] for state in states],
        ys,
        REFERENCE_SOURCE,
        variant=True,
        zorder=3,
    )
    fs.points(ax, [100 * shares["blood"][state] for state in states], ys, BLOOD_SOURCE, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(*SHARE_LIMITS)
    ax.set_xticks(list(SHARE_TICKS))
    fs.plain_log_axis(ax, "x")
    log_minor_ticks(ax.xaxis)
    ax.grid(True, axis="x", which="major")
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, ys[0] + 0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([SHARE_ROW_NAMES[state] for state in states])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("% of progenitors", labelpad=1.0)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=fs.MARKER_MS, mfc=BLOOD_SOURCE, mec=fs.WHITE),
        plt.Line2D(
            [], [], ls="none", marker="o", ms=fs.MARKER_MS, mfc=fs.WHITE, mec=REFERENCE_SOURCE
        ),
    ]
    legend_above(
        ax,
        handles,
        [label for _, _, label in SOURCES],
        ncol=2,
        left_mm=-6.0,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.5,
        fontsize=fs.NOTE,
    )


# ---- the same analysis from local disk and from object storage -----------------------------------------------

# The backends along the x axis, with the tools that read each, as bars this wide: BPCells reads local files only, which
# the caption says. The legend names the tools in two columns over the plot, Scanpy without its Dask, which the caption
# names, so the three fit beside the letter.
BACKENDS = (("localDisk", "local\ndisk"), ("objectStorage", "object\nstorage"))
BACKEND_TOOLS = ("cytearc", "bpcells", "scanpy")
BACKEND_LABELS = {"scanpy": "Scanpy"}
# The two backends' groups of bars, this far apart.
BACKEND_GROUP_GAP_MM = 3.5
# A backend's bracket sits this many minutes above the tallest of its error bars.
BACKEND_BRACKET_CLEARANCE = 4.0
BACKEND_LIMIT_MIN = 50.0
# A tool with no reader for a backend keeps its slot, marked at the axis by the open ink triangle the analysis-scaling
# figure gives a run that could not complete, with the reason written up the slot.
NOT_SUPPORTED = {
    "ls": "none",
    "marker": "v",
    "ms": 3.5,
    "mfc": fs.WHITE,
    "mec": fs.INK,
    "mew": 0.65,
}
NOT_SUPPORTED_MIN = 1.6


def range_bar(ax, x: float, low: float, high: float, cap: float = 0.08) -> None:
    """The range of the runs, lowest to highest, as an ink error bar with short caps."""
    ax.plot([x, x], [low, high], color=fs.INK, lw=fs.LW_AXIS, solid_capstyle="butt", zorder=5)
    for y in (low, high):
        ax.plot(
            [x - cap, x + cap], [y, y], color=fs.INK, lw=fs.LW_AXIS, solid_capstyle="butt", zorder=5
        )


def pair_bracket(ax, x0: float, x1: float, y: float, drop: float, lines) -> None:
    """A bracket over two bars, from x0 to x1 at height y with its ends dropping by drop, and lines of text over it,
    each (text, colour), from the bracket up."""
    ax.plot(
        [x0, x0, x1, x1],
        [y - drop, y, y, y - drop],
        color=fs.INK,
        lw=fs.LW_THIN,
        solid_capstyle="butt",
        zorder=5,
    )
    for i, (text, colour) in enumerate(lines):
        ax.annotate(
            text,
            ((x0 + x1) / 2, y),
            xytext=(0, 1.0 + 6.5 * i),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fs.NOTE,
            color=colour,
        )


def storage_backends(ax, data: dict) -> None:
    """Analysis minutes of the one-million-cell embryo workflow with its data on local disk and on object storage: on
    each backend, CyteArc, BPCells and Scanpy on Dask over a Zarr copy, bars the means of three runs and error bars
    their ranges, and an open triangle where a tool cannot read the backend; over each backend's bars, the Welch P value
    of CyteArc's geometric mean time against Scanpy's. CyteArc against BPCells is in the caption: its bracket's P
    value, over two bars by the axis, would cross the axis labels."""
    tools = {tool["key"]: tool for tool in data["tools"]}
    groups = bar_centres(ax, [len(BACKEND_TOOLS)] * len(BACKENDS), BACKEND_GROUP_GAP_MM)
    for (backend, _), centres in zip(BACKENDS, groups):
        x = dict(zip(BACKEND_TOOLS, centres))
        highs = []
        for key in BACKEND_TOOLS:
            summary = tools[key][backend]
            if summary is None:
                ax.plot(x[key], NOT_SUPPORTED_MIN, zorder=5, **NOT_SUPPORTED)
                ax.annotate(
                    "not supported",
                    (x[key], NOT_SUPPORTED_MIN),
                    xytext=(0, 3.5),
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=fs.NOTE,
                    color=fs.SLATE,
                )
                continue
            ax.bar(x[key], summary["mean"], width=BAR_MM, zorder=3, **tool_bar(key))
            range_bar(ax, x[key], summary["low"], summary["high"], cap=0.5)
            highs.append(summary["high"])
        pair_bracket(
            ax,
            x["cytearc"],
            x["scanpy"],
            max(highs) + BACKEND_BRACKET_CLEARANCE,
            1.0,
            [(p_text(data["scanpyOverCytearc"][backend]["p"]), fs.INK)],
        )
    ax.set_xticks([float(np.mean(centres)) for centres in groups])
    ax.set_xticklabels([name for _, name in BACKENDS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, BACKEND_LIMIT_MIN)
    ax.set_yticks([0, 20, 40])
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.set_ylabel("analysis time (min)")
    # With no header, the legend of the tools takes the header's place beside the letter.
    left, bottom, _, height = CELLS["storage_backends"].plot
    legend_above(
        ax,
        [Patch(**tool_bar(key)) for key in BACKEND_TOOLS],
        [BACKEND_LABELS.get(key, tools[key]["label"]) for key in BACKEND_TOOLS],
        ncol=2,
        left_mm=HEADER_INDENT_MM - left,
        top_mm=CELLS["storage_backends"].height - 1.0 - bottom - height,
        handlelength=1.0,
        columnspacing=0.6,
    )


PANELS = {
    draw.__name__: draw
    for draw in (
        mount_overlay,
        storage_backends,
        tabula_query,
        interferon_score,
        composite_umap,
        progenitor_funnel,
        progenitor_states,
        progenitor_sources,
        progenitor_state_shares,
        progenitor_diseases,
        progenitor_covid,
        progenitor_cohorts,
    )
}
# Panels that read another panel's data file.
DATA_FILE = {
    "composite_umap": "composite_footprint",
    "progenitor_funnel": "progenitor_effects",
    "progenitor_states": "progenitor_maps",
    "progenitor_sources": "progenitor_maps",
    "progenitor_state_shares": "progenitor_effects",
    "progenitor_diseases": "progenitor_effects",
    "progenitor_covid": "progenitor_effects",
    "progenitor_cohorts": "progenitor_effects",
}

# Short headers beside panel letters, for the panels whose subject their axes and legend do not name; the schematic
# carries its own titles. A header can be written from the panel's data.
HEADERS = {
    "tabula_query": "Tabula Sapiens query",
    "interferon_score": lambda data: (
        f"IFN response in {data['totals']['primaryBloodCellsRead'] / 1e6:.1f}M cells, "
        f"{data['totals']['donorsWithScore']:,} donors"
    ),
    "composite_umap": lambda data: (
        f"{sum(s['cells'] for s in data['umap']['studies']) / 1e6:.2f}M cells from "
        f"{len(data['umap']['studies'])} blood studies"
    ),
    "progenitor_funnel": lambda data: f"Progenitors in {data['discovery']['stores']} datasets",
    "progenitor_states": "Progenitor states",
    "progenitor_sources": "Progenitor sources",
    "progenitor_covid": lambda data: (
        f"{len(data['covid'])} studies, "
        f"{sum(s['healthyDonors'] + s['covidDonors'] for s in data['covid']):,} donors"
    ),
    "progenitor_state_shares": "States by source",
    "progenitor_diseases": lambda data: f"{len(data['diseases'])} diseases",
    "progenitor_cohorts": "COMBAT: COVID-19 and sepsis",
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
    "mount_overlay": Cell(160.0, 42.0, (0.0, 0.0, 160.0, 42.0)),
    "storage_backends": Cell(
        32.5, 39.5, (10.0, ROW_FRAME_MM[0], 21.0, ROW_FRAME_MM[1] - ROW_FRAME_MM[0])
    ),
    "tabula_query": Cell(
        72.5,
        39.5,
        (
            4.5,
            ROW_FRAME_MM[0],
            ROW_FRAME_MM[1] - ROW_FRAME_MM[0],
            ROW_FRAME_MM[1] - ROW_FRAME_MM[0],
        ),
    ),
    "interferon_score": Cell(55.0, 77.5, (36.5, 7.5, 18.0, 59.0)),
    "composite_umap": Cell(105.0, 38.0, (4.5, 8.0, 22.5, 22.5)),
    "progenitor_funnel": Cell(41.0, 55.0, (0.0, 1.0, 41.0, 48.5)),
    "progenitor_states": Cell(30.0, 26.0, (4.0, 2.5, 24.0, 19.0)),
    "progenitor_sources": Cell(30.0, 29.0, (4.0, 3.6, 24.0, 18.5)),
    "progenitor_state_shares": Cell(28.0, 55.0, (7.2, 9.0, 18.8, 35.0)),
    "progenitor_diseases": Cell(30.0, 32.0, (8.0, 6.9, 20.0, 20.4)),
    "progenitor_covid": Cell(31.0, 32.0, (11.5, 6.9, 17.5, 16.9)),
    "progenitor_cohorts": Cell(61.0, 23.0, (14.5, 4.2, 45.5, 13.9)),
}
# Where each panel's cell sits: its lower-left corner, in millimetres from the figure's lower-left corner. The schematic
# spans the top. Under it, a left column holds one row for reading a store from disk or object storage and querying one
# in place, and one for the ten-study composite, both as wide as each other; the screen's interferon response runs down
# the right of both rows. The bottom row pulls the circulating progenitors out of every blood store; at its right, the
# diseases and the COVID-19 studies sit side by side over COMBAT's cohorts, from studies to one study's donors. The
# panels take their letters in this order.
LAYOUT = {
    "mount_overlay": (0.0, 132.5),
    "storage_backends": (0.0, 93.0),
    "tabula_query": (32.5, 93.0),
    "interferon_score": (105.0, 55.0),
    "composite_umap": (0.0, 55.0),
    "progenitor_funnel": (0.0, 0.0),
    "progenitor_states": (41.0, 29.0),
    "progenitor_sources": (41.0, 0.0),
    "progenitor_state_shares": (71.0, 0.0),
    "progenitor_diseases": (99.0, 23.0),
    "progenitor_covid": (129.0, 23.0),
    "progenitor_cohorts": (99.0, 0.0),
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


def panel_data(name: str) -> dict:
    return json.loads((HERE / f"{DATA_FILE.get(name, name)}.json").read_text())


def header(name: str) -> str:
    text = HEADERS[name]
    return text(panel_data(name)) if callable(text) else text


def draw_panel(fig, name: str, left: float, bottom: float):
    """Draw one panel with its cell's lower-left corner at (left, bottom) millimetres in the figure."""
    x, y, w, h = CELLS[name].plot
    ax = fs.add_axes(fig, left + x, bottom + y, w, h)
    ax.set_gid(name)
    PANELS[name](ax, panel_data(name))
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
            fs.panel_title(fig, left + HEADER_INDENT_MM, top_y - 2.1, header(name))
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
