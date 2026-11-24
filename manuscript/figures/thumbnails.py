"""Thumbnails of single-cell data and results for the schematics, from one illustrative dataset.

The dataset is synthetic and fixed by its seed: 256 cells in six groups laid out in two
dimensions like an embedding, with counts of twelve genes (two markers for each of the first
five groups, one gene that splits the first group in two, and one expressed everywhere). Every
thumbnail is computed from it by the operation it depicts: the principal components from the
log counts, the graph from nearest neighbours, the marker dots from the expression of each
group. The thumbnails therefore agree with one another, as the results of one store do.

Thumbnails draw on a glyphs.mm_axes in millimetres; (x, y) is the lower-left corner.

  heatmap   the count matrix (cells by genes), optionally cut into blocks of cells or of genes
  table     a cell table (group column) or a gene table (mean and highly variable columns)
  pca       the first two principal components
  graph     the nearest-neighbour graph
  loadings  the principal-component loadings (genes by components)
  embedding the two-dimensional layout, coloured by the first or second clustering
  dotplot   marker genes by group
  record    the provenance record of a result, as a small document
"""

from functools import lru_cache

import numpy as np
from manuscript.figures import style as fs
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Rectangle

# (centre, (major and minor spread), angle in degrees, cells, first group, second group). The
# second clustering splits the first group in two.
PARTS = (
    ((-0.62, 0.22), (0.46, 0.30), 20, 48, 0, 0),
    ((0.62, -0.18), (0.44, 0.30), 20, 44, 0, 5),
    ((2.35, 1.95), (0.40, 0.28), 55, 42, 1, 1),
    ((-2.35, 1.85), (0.44, 0.30), -35, 42, 2, 2),
    ((2.2, -2.1), (0.38, 0.30), 15, 38, 3, 3),
    ((-2.05, -2.1), (0.46, 0.28), -55, 42, 4, 4),
)
GENES = 12
MARKERS = [0, 2, 4, 6, 8]  # the first marker of each of the five first groups


@lru_cache(maxsize=1)
def dataset() -> dict:
    rng = np.random.default_rng(11)
    xy, part, first, second = [], [], [], []
    for k, (centre, (major, minor), angle, n, g1, g2) in enumerate(PARTS):
        z = rng.normal(size=(n, 2)) * (major, minor)
        a = np.radians(angle)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        xy.append(z @ rot.T + centre)
        part += [k] * n
        first += [g1] * n
        second += [g2] * n
    xy = np.vstack(xy)
    part, first, second = np.array(part), np.array(first), np.array(second)
    rate = np.full((len(part), GENES), 0.35)
    for i, g in enumerate(first):
        rate[i, 2 * g] = 3.2
        rate[i, 2 * g + 1] = 2.2
    rate[part == 1, 10] = 2.5
    rate[:, 11] = 1.8
    size = rng.lognormal(0.0, 0.3, size=(len(part), 1))
    counts = rng.poisson(rate * size)
    logc = np.log1p(counts)
    centred = logc - logc.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    pcs = u[:, :2] * s[:2]
    nodes = np.arange(0, len(part), 5)
    loadings = np.linalg.svd(centred, full_matrices=False)[2][:4].T
    return {
        "loadings": loadings,
        "xy": xy,
        "part": part,
        "first": first,
        "second": second,
        "logc": logc,
        "pcs": pcs,
        "nodes": nodes,
    }


def _fit(
    points: np.ndarray, x: float, y: float, w: float, h: float, pad: float = 0.1
) -> np.ndarray:
    """Scale points into the box with equal units on both axes, centred."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = hi - lo
    scale = min(w * (1 - 2 * pad) / span[0], h * (1 - 2 * pad) / span[1])
    centre = (lo + hi) / 2
    return (points - centre) * scale + (x + w / 2, y + h / 2)


def frame(ax, x, y, w, h, *, z=3, edge=fs.LIGHT, fill="white"):
    """The frame of a thumbnail: 0.6 pt light, no axes."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=fs.LW_AXIS, zorder=z))


def _cells_for_heatmap(n: int) -> np.ndarray:
    """n cells, as evenly as possible from each part, in the order of the parts."""
    data = dataset()
    per = np.array_split(np.arange(n), len(PARTS))
    chosen = []
    for k, block in enumerate(per):
        members = np.flatnonzero(data["part"] == k)
        chosen += list(members[np.linspace(0, len(members) - 1, len(block)).astype(int)])
    return np.array(chosen)


def heatmap(
    ax,
    x,
    y,
    w,
    h,
    *,
    cells=24,
    blocks=None,
    n_blocks=4,
    faded=(),
    outlined=(),
    z=3,
    outline=fs.BLUE,
):
    """The count matrix, cells as rows and genes as columns, in the sequential ramp.

    blocks "rows" cuts it into n_blocks bands of cells (stored by cell), "cols" into bands of
    genes (stored by gene). Blocks in faded are drawn pale (already read), blocks in outlined
    get a blue edge (in memory now). Returns the extent of each block, top to bottom or left to
    right.
    """
    data = dataset()
    m = data["logc"][_cells_for_heatmap(cells)]
    ax.imshow(
        m,
        extent=(x, x + w, y, y + h),
        cmap=fs.SEQUENTIAL,
        vmin=0,
        vmax=np.quantile(data["logc"], 0.99),
        interpolation="nearest",
        aspect="auto",
        zorder=z,
    )
    ax.set_aspect("equal")
    extents = []
    if blocks == "rows":
        edges = np.linspace(y + h, y, n_blocks + 1)
        extents = [(x, edges[i + 1], w, edges[i] - edges[i + 1]) for i in range(n_blocks)]
        for e in edges[1:-1]:
            ax.plot(
                [x, x + w],
                [e, e],
                color=fs.SLATE,
                lw=fs.LW_THIN,
                zorder=z + 0.1,
                solid_capstyle="butt",
            )
    elif blocks == "cols":
        edges = np.linspace(x, x + w, n_blocks + 1)
        extents = [(edges[i], y, edges[i + 1] - edges[i], h) for i in range(n_blocks)]
        for e in edges[1:-1]:
            ax.plot(
                [e, e],
                [y, y + h],
                color=fs.SLATE,
                lw=fs.LW_THIN,
                zorder=z + 0.1,
                solid_capstyle="butt",
            )
    for i in faded:
        ax.add_patch(
            Rectangle(
                extents[i][:2],
                *extents[i][2:],
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                zorder=z + 0.2,
            )
        )
    for i in outlined:
        ax.add_patch(
            Rectangle(
                extents[i][:2],
                *extents[i][2:],
                facecolor="none",
                edgecolor=outline,
                lw=1.0,
                zorder=z + 0.3,
            )
        )
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="none",
            edgecolor=fs.SLATE if blocks else fs.LIGHT,
            lw=fs.LW_THIN,
            zorder=z + 0.25,
        )
    )
    return extents


def heatmap_slice(ax, x, y, w, h, block: int, *, cells=32, n_blocks=8, z=3, edge=fs.BLUE):
    """One band of cells of the matrix drawn by heatmap(blocks="rows"), at a new place."""
    data = dataset()
    rows = np.array_split(np.arange(cells), n_blocks)[block]
    m = data["logc"][_cells_for_heatmap(cells)][rows]
    ax.imshow(
        m,
        extent=(x, x + w, y, y + h),
        cmap=fs.SEQUENTIAL,
        vmin=0,
        vmax=np.quantile(data["logc"], 0.99),
        interpolation="nearest",
        aspect="auto",
        zorder=z,
    )
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=edge, lw=1.0, zorder=z + 0.3))


def table(ax, x, y, w, h, *, kind="cells", rows=5, z=3):
    """A cell table (identifier, group, genes detected) or a gene table (name, mean, variable)."""
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="white", edgecolor=fs.LIGHT, lw=fs.LW_AXIS, zorder=z)
    )
    header = h / (rows + 1)
    ax.add_patch(
        Rectangle(
            (x, y + h - header),
            w,
            header,
            facecolor=fs.PAPER,
            edgecolor=fs.LIGHT,
            lw=fs.LW_THIN,
            zorder=z + 0.1,
        )
    )
    rng = np.random.default_rng(5 if kind == "cells" else 6)
    for r in range(rows):
        cy = y + h - header * (r + 1.5)
        if r:
            ax.plot([x, x + w], [cy + header / 2] * 2, color=fs.GRID, lw=fs.LW_THIN, zorder=z + 0.1)
        ax.plot(
            [x + 0.08 * w, x + 0.3 * w],
            [cy, cy],
            color=fs.LIGHT,
            lw=1.0,
            zorder=z + 0.2,
            solid_capstyle="butt",
        )
        if kind == "cells":
            ax.add_patch(
                Rectangle(
                    (x + 0.4 * w, cy - header * 0.28),
                    0.12 * w,
                    header * 0.56,
                    facecolor=fs.GROUPS[r % 5],
                    edgecolor="none",
                    zorder=z + 0.2,
                )
            )
            ax.plot(
                [x + 0.62 * w, x + 0.62 * w + 0.3 * w * rng.uniform(0.4, 1)],
                [cy, cy],
                color=fs.SLATE,
                lw=1.0,
                zorder=z + 0.2,
                solid_capstyle="butt",
            )
        else:
            ax.plot(
                [x + 0.4 * w, x + 0.4 * w + 0.28 * w * rng.uniform(0.3, 1)],
                [cy, cy],
                color=fs.SLATE,
                lw=1.0,
                zorder=z + 0.2,
                solid_capstyle="butt",
            )
            if r in (0, 2, 3):
                cx = x + 0.85 * w
                ax.plot(
                    [cx - 0.05 * w, cx - 0.01 * w, cx + 0.06 * w],
                    [cy, cy - header * 0.25, cy + header * 0.3],
                    color=fs.INK,
                    lw=0.6,
                    zorder=z + 0.2,
                )


def _scatter(ax, pts, colours, *, size, z, alpha=1.0):
    ax.scatter(pts[:, 0], pts[:, 1], s=size, c=colours, linewidths=0, alpha=alpha, zorder=z)


def embedding(ax, x, y, w, h, *, clustering="first", size=1.2, z=3, alpha=1.0, framed=True):
    data = dataset()
    if framed:
        frame(ax, x, y, w, h, z=z)
    pts = _fit(data["xy"], x, y, w, h)
    colours = [fs.GROUPS[g] for g in data[clustering]]
    _scatter(ax, pts, colours, size=size, z=z + 0.1, alpha=alpha)
    return pts


def pca(ax, x, y, w, h, *, size=1.1, z=3):
    data = dataset()
    frame(ax, x, y, w, h, z=z)
    pts = _fit(data["pcs"], x, y, w, h)
    _scatter(ax, pts, [fs.GROUPS[g] for g in data["first"]], size=size, z=z + 0.1)


def graph(ax, x, y, w, h, *, k=3, size=1.5, z=3):
    """The nearest-neighbour graph with k neighbours: grey nodes and edges, not an embedding."""
    data = dataset()
    frame(ax, x, y, w, h, z=z)
    nodes = data["xy"][data["nodes"]]
    d = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    edges = sorted({tuple(sorted((a, b))) for a in range(len(nodes)) for b in np.argsort(d[a])[:k]})
    pts = _fit(nodes, x, y, w, h)
    ax.add_collection(
        LineCollection(
            [(pts[a], pts[b]) for a, b in edges], colors=fs.SLATE, linewidths=0.35, zorder=z + 0.1
        )
    )
    _scatter(ax, pts, [fs.INK] * len(pts), size=size, z=z + 0.2)


def loadings(ax, x, y, w, h, *, z=3):
    """The principal-component loadings: genes (rows) by components (columns), signed."""
    data = dataset()
    m = data["loadings"]
    v = np.abs(m).max()
    ax.imshow(
        m,
        extent=(x, x + w, y, y + h),
        cmap=fs.DIVERGING,
        vmin=-v,
        vmax=v,
        interpolation="nearest",
        aspect="auto",
        zorder=z,
    )
    ax.set_aspect("equal")
    ax.add_patch(
        Rectangle(
            (x, y), w, h, facecolor="none", edgecolor=fs.LIGHT, lw=fs.LW_THIN, zorder=z + 0.25
        )
    )


def dotplot(ax, x, y, w, h, *, z=3):
    """Marker genes (columns) by group (rows): dot size, the share of cells expressing; colour, the mean."""
    data = dataset()
    frame(ax, x, y, w, h, z=z)
    groups = range(5)
    xs, ys, sizes, means = [], [], [], []
    for r, g in enumerate(groups):
        members = data["first"] == g
        for c, gene in enumerate(MARKERS):
            values = data["logc"][members, gene]
            xs.append(x + w * (c + 0.5) / len(MARKERS))
            ys.append(y + h - h * (r + 0.5) / len(groups))
            sizes.append(0.3 + 13.0 * (values > 0).mean() ** 2)
            means.append(values.mean())
    ax.scatter(
        xs,
        ys,
        s=sizes,
        c=means,
        cmap=fs.SEQUENTIAL,
        vmin=0,
        vmax=max(means),
        linewidths=0,
        zorder=z + 0.1,
    )


# Real data -----------------------------------------------------------------------------
# The functions below draw supplied measured reference arrays with the
# same frames, marks and colours as the thumbnails of the illustrative dataset above.


def scatter_xy(
    ax, x, y, w, h, xy, colours, *, size=0.8, z=3, alpha=1.0, framed=True, pad=0.08, fit_to=None
):
    """Points in a framed box, scaled with equal units on both axes (fit_to sets the extent)."""
    if framed:
        frame(ax, x, y, w, h, z=z)
    reference = np.asarray(fit_to if fit_to is not None else xy)
    lo, hi = reference.min(axis=0), reference.max(axis=0)
    scale = min(w * (1 - 2 * pad) / (hi[0] - lo[0]), h * (1 - 2 * pad) / (hi[1] - lo[1]))
    pts = (np.asarray(xy) - (lo + hi) / 2) * scale + (x + w / 2, y + h / 2)
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=size,
        c=colours,
        linewidths=0,
        alpha=alpha,
        zorder=z + 0.1,
        rasterized=True,
    )
    return pts


def edges_xy(ax, x, y, w, h, xy, pairs, *, fit_to=None, z=3, pad=0.08, node_size=1.2, lw=0.3):
    """A neighbour graph drawn on given positions: grey edges and ink nodes."""
    frame(ax, x, y, w, h, z=z)
    reference = np.asarray(fit_to if fit_to is not None else xy)
    lo, hi = reference.min(axis=0), reference.max(axis=0)
    scale = min(w * (1 - 2 * pad) / (hi[0] - lo[0]), h * (1 - 2 * pad) / (hi[1] - lo[1]))
    pts = (np.asarray(xy) - (lo + hi) / 2) * scale + (x + w / 2, y + h / 2)
    ax.add_collection(
        LineCollection(
            [(pts[a], pts[b]) for a, b in pairs],
            colors=fs.SLATE,
            linewidths=lw,
            zorder=z + 0.1,
            rasterized=True,
        )
    )
    return pts


def matrix(
    ax,
    x,
    y,
    w,
    h,
    values,
    *,
    cmap=None,
    vmin=None,
    vmax=None,
    z=3,
    edge=fs.LIGHT,
    rows=None,
    cols=None,
    faded=(),
    outlined=(),
    outline=fs.BLUE,
):
    """A matrix as a heatmap; rows or cols cut it into blocks (slate lines); returns the row blocks."""
    cmap = cmap or fs.SEQUENTIAL
    values = np.asarray(values)
    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax
    ax.imshow(
        values,
        extent=(x, x + w, y, y + h),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
        zorder=z,
    )
    ax.set_aspect("equal")
    blocks = []
    if rows:
        edges = np.linspace(y + h, y, rows + 1)
        blocks = [(x, edges[i + 1], w, edges[i] - edges[i + 1]) for i in range(rows)]
        for e in edges[1:-1]:
            ax.plot(
                [x, x + w],
                [e, e],
                color=fs.SLATE,
                lw=fs.LW_THIN,
                zorder=z + 0.1,
                solid_capstyle="butt",
            )
    if cols:
        for e in np.linspace(x, x + w, cols + 1)[1:-1]:
            ax.plot(
                [e, e],
                [y, y + h],
                color=fs.SLATE,
                lw=fs.LW_THIN,
                zorder=z + 0.1,
                solid_capstyle="butt",
            )
    for i in faded:
        ax.add_patch(
            Rectangle(
                blocks[i][:2],
                *blocks[i][2:],
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                zorder=z + 0.2,
            )
        )
    for i in outlined:
        ax.add_patch(
            Rectangle(
                blocks[i][:2],
                *blocks[i][2:],
                facecolor="none",
                edgecolor=outline,
                lw=1.0,
                zorder=z + 0.3,
            )
        )
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="none",
            edgecolor=fs.SLATE if (rows or cols) else edge,
            lw=fs.LW_THIN,
            zorder=z + 0.25,
        )
    )
    return blocks


def strip(ax, x, y, w, h, colours, *, vertical=True, z=3):
    """A strip of coloured cells beside a matrix: one per row (vertical) or per column."""
    n = len(colours)
    for i, colour in enumerate(colours):
        if vertical:
            ax.add_patch(
                Rectangle(
                    (x, y + h - (i + 1) * h / n),
                    w,
                    h / n,
                    facecolor=colour,
                    edgecolor="none",
                    zorder=z,
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (x + i * w / n, y), w / n, h, facecolor=colour, edgecolor="none", zorder=z
                )
            )


def dots(ax, x, y, w, h, share, mean, *, z=3, max_size=10.0):
    """A dot plot: rows are groups, columns genes; size is the share expressing, colour the mean."""
    frame(ax, x, y, w, h, z=z)
    share, mean = np.asarray(share), np.asarray(mean)
    rows, cols = share.shape
    xs, ys = np.meshgrid(
        x + w * (np.arange(cols) + 0.5) / cols, y + h - h * (np.arange(rows) + 0.5) / rows
    )
    ax.scatter(
        xs.ravel(),
        ys.ravel(),
        s=0.2 + max_size * share.ravel(),
        c=mean.ravel(),
        cmap=fs.SEQUENTIAL,
        vmin=0,
        vmax=mean.max(),
        linewidths=0,
        zorder=z + 0.1,
    )


def record(ax, x, y, w=2.2, h=2.8, *, colour=None, z=6):
    """A provenance record: a small document with a folded corner, in the Record module's tone."""
    colour = colour or fs.MODULES["record"].strong
    f = min(w, h) * 0.32
    ax.add_patch(
        Polygon(
            [(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)],
            closed=True,
            facecolor="white",
            edgecolor=colour,
            lw=fs.LW_THIN,
            zorder=z,
            joinstyle="round",
        )
    )
    ax.add_patch(
        Polygon(
            [(x + w - f, y + h), (x + w - f, y + h - f), (x + w, y + h - f)],
            closed=True,
            facecolor=fs.tint(colour, 0.35),
            edgecolor=colour,
            lw=fs.LW_THIN,
            zorder=z + 0.1,
            joinstyle="round",
        )
    )
    for k in range(3):
        yy = y + h * (0.28 + 0.2 * k)
        ax.plot(
            [x + 0.18 * w, x + (0.82 if k else 0.6) * w - (f * 0.6 if k == 2 else 0)],
            [yy, yy],
            color=fs.tint(colour, 0.6),
            lw=0.4,
            zorder=z + 0.1,
            solid_capstyle="butt",
        )


def mapped(ax, x, y, w, h, *, stage="placed", z=3, seed=4):
    """Query cells on the reference embedding: placed (dark), or labelled (coloured, grey where abstained)."""
    data = dataset()
    frame(ax, x, y, w, h, z=z)
    ref = _fit(data["xy"], x, y, w, h)
    _scatter(
        ax,
        ref,
        [fs.GROUPS[g] for g in data["first"]],
        size=1.1,
        z=z + 0.1,
        alpha=0.25 if stage == "placed" else 0.14,
    )
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ref), 40, replace=False)
    query = ref[pick] + rng.normal(scale=0.2, size=(40, 2))
    if stage == "placed":
        _scatter(ax, query, [fs.INK] * len(query), size=1.6, z=z + 0.2)
    else:
        abstain = rng.random(len(query)) < 0.16
        colours = [fs.SLATE if a else fs.GROUPS[data["first"][i]] for i, a in zip(pick, abstain)]
        _scatter(ax, query, colours, size=2.0, z=z + 0.2)
