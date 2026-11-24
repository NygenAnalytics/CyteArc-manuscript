"""Schematic glyphs shared by every figure.

Glyphs draw on an axes whose data units are millimetres of the printed figure (mm_axes), so a
glyph has the same size wherever it appears. Every colour comes from style.

  folder          a store: a folder whose tab carries its name (its contents are thumbnails)
  cloud           object storage
  laptop          the user's own machine
  lock            opened read-only
  card            a module (tinted, with the module's strong edge) or a place (paper)
  chip            a named result where a thumbnail would not fit: blue edge new, light reused
  provenance      key and value rows of one provenance record, as a document
  badge           a numbered step
  arrow           solid for data read or written, dashed for a reference that is not copied
  ring            the coloured arcs around the hub of Fig. 1

Thumbnails of data and results (count matrix, tables, PCA, graph, embedding, markers, the
provenance record icon) are in thumbnails.py.
"""

import numpy as np
from manuscript.figures import style as fs
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Wedge
from matplotlib.path import Path


def mm_axes(fig, x: float, y: float, w: float, h: float):
    """An axes covering (x, y, w, h) millimetres of the figure, in millimetre data units."""
    fw, fh = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    ax = fig.add_axes([x / fw, y / fh, w / fw, h / fh])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def text(ax, x, y, s, *, size=fs.TICK, colour=fs.INK, **kwargs):
    kwargs.setdefault("va", "center")
    kwargs.setdefault("ha", "left")
    return ax.text(x, y, s, fontsize=size, color=colour, **kwargs)


def card(
    ax,
    x0,
    y0,
    x1,
    y1,
    *,
    module: str | None = None,
    title: str | None = None,
    letter: str | None = None,
    subtitle: str | None = None,
    radius=1.6,
    z=0,
    align="left",
):
    """A rounded card; with a module, its tint and strong edge, else paper with a light edge.

    The letter, title and subtitle sit in the top-left corner, or the top-right with align="right".
    """
    fill, edge = (
        (fs.MODULES[module].tint, fs.MODULES[module].strong) if module else (fs.PAPER, fs.LIGHT)
    )
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            lw=fs.LW_AXIS,
            zorder=z,
        )
    )
    top = y1 - 4.2
    if align == "left":
        x = x0 + 2.4
        if letter:
            ax.text(
                x,
                top,
                letter,
                fontsize=fs.LETTER,
                fontweight="bold",
                color=fs.INK,
                va="baseline",
                ha="left",
                gid="panel-letter",
                zorder=z + 1,
            )
            x += 3.4
        if title:
            ax.text(
                x,
                top,
                title,
                fontsize=fs.LABEL,
                fontweight="bold",
                color=fs.INK,
                va="baseline",
                ha="left",
                gid="card-title",
                zorder=z + 1,
            )
        if subtitle:
            ax.text(
                x,
                top - 3.4,
                subtitle,
                fontsize=fs.TICK,
                color=fs.SLATE,
                va="baseline",
                ha="left",
                zorder=z + 1,
            )
    else:
        x = x1 - 2.4
        left = x
        if title:
            t = ax.text(
                x,
                top,
                title,
                fontsize=fs.LABEL,
                fontweight="bold",
                color=fs.INK,
                va="baseline",
                ha="right",
                gid="card-title",
                zorder=z + 1,
            )
            box = t.get_window_extent(ax.figure.canvas.get_renderer())
            left = ax.transData.inverted().transform((box.x0, box.y0))[0]
        if letter:
            ax.text(
                left - 1.8,
                top,
                letter,
                fontsize=fs.LETTER,
                fontweight="bold",
                color=fs.INK,
                va="baseline",
                ha="right",
                gid="panel-letter",
                zorder=z + 1,
            )
        if subtitle:
            ax.text(
                x,
                top - 3.4,
                subtitle,
                fontsize=fs.TICK,
                color=fs.SLATE,
                va="baseline",
                ha="right",
                zorder=z + 1,
            )


def text_width(s: str, size: float = fs.TICK) -> float:
    """Approximate printed width in millimetres of one line of TeX Gyre Heros."""
    return len(s) * size * 0.19


def chip(ax, x, y, w, s, *, state="plain", h=3.4, z=3, ha="center"):
    """One stored result, centred at (x, y): new (blue edge), reused (light) or plain (ink).

    w None sizes the chip to its text.
    """
    if w is None:
        w = text_width(s) + 2.4
    edge, colour = {
        "new": (fs.BLUE, fs.INK),
        "reused": (fs.LIGHT, fs.SLATE),
        "plain": (fs.SLATE, fs.INK),
    }[state]
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor="white",
            edgecolor=edge,
            lw=fs.LW_AXIS if state != "new" else 0.9,
            zorder=z,
        )
    )
    ax.text(
        x, y, s, fontsize=fs.TICK, color=colour, va="center_baseline", ha="center", zorder=z + 1
    )


def badge(ax, x, y, s, *, colour=fs.INK, r=1.55, z=5):
    ax.add_patch(Circle((x, y), r, facecolor=colour, edgecolor="white", lw=fs.LW_THIN, zorder=z))
    ax.text(
        x, y, s, fontsize=fs.TICK, color="white", va="center_baseline", ha="center", zorder=z + 1
    )


def arrow(
    ax, start, end, *, colour=fs.INK, dashed=False, rad=0.0, lw=0.8, head=5.0, z=2, shrink=0.0
):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=head,
            color=colour,
            lw=lw,
            ls=(0, (3, 2)) if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=shrink,
            shrinkB=shrink,
            zorder=z,
        )
    )


def lock(ax, cx, cy, s=3.0, *, colour=fs.INK, z=6):
    """Read-only: a padlock whose body is s millimetres wide."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - s / 2, cy - s * 0.45),
            s,
            s * 0.8,
            boxstyle=f"round,pad=0,rounding_size={s * 0.12}",
            facecolor=colour,
            edgecolor="white",
            lw=fs.LW_THIN,
            zorder=z,
        )
    )
    t = np.linspace(0, np.pi, 30)
    r = s * 0.3
    ax.plot(
        cx + r * np.cos(t),
        cy + s * 0.35 + r * np.sin(t) * 1.1,
        color=colour,
        lw=1.0,
        zorder=z - 0.1,
        solid_capstyle="butt",
    )
    ax.plot(
        [cx - r, cx - r],
        [cy + s * 0.2, cy + s * 0.35],
        color=colour,
        lw=1.0,
        zorder=z - 0.1,
        solid_capstyle="butt",
    )
    ax.plot(
        [cx + r, cx + r],
        [cy + s * 0.2, cy + s * 0.35],
        color=colour,
        lw=1.0,
        zorder=z - 0.1,
        solid_capstyle="butt",
    )
    ax.add_patch(
        Circle((cx, cy - s * 0.08), s * 0.09, facecolor="white", edgecolor="none", zorder=z + 0.1)
    )


def provenance(ax, x, y, rows, *, w=40.0, key_w=13.0, line=3.1, z=2, fill="white", colour=None):
    """A provenance record: keys in slate, values in ink; (x, y) is the top-left corner.

    Drawn as a document with a folded corner in the Record module's tone, like the record icons
    of thumbnails, so a record looks the same at every size.
    """
    colour = colour or fs.MODULES["record"].strong
    h = line * len(rows) + 1.8
    f = 2.2
    ax.add_patch(
        Polygon(
            [(x, y - h), (x + w, y - h), (x + w, y - f), (x + w - f, y), (x, y)],
            closed=True,
            facecolor=fill,
            edgecolor=colour,
            lw=fs.LW_AXIS,
            zorder=z,
            joinstyle="round",
        )
    )
    ax.add_patch(
        Polygon(
            [(x + w - f, y), (x + w - f, y - f), (x + w, y - f)],
            closed=True,
            facecolor=fs.tint(colour, 0.35),
            edgecolor=colour,
            lw=fs.LW_THIN,
            zorder=z + 0.1,
            joinstyle="round",
        )
    )
    for i, (key, value) in enumerate(rows):
        yy = y - 1.1 - line * (i + 0.5)
        ax.text(
            x + 1.6,
            yy,
            key,
            fontsize=fs.NOTE,
            color=fs.SLATE,
            va="center_baseline",
            ha="left",
            zorder=z + 1,
        )
        ax.text(
            x + 1.6 + key_w,
            yy,
            value,
            fontsize=fs.TICK,
            color=fs.INK,
            va="center_baseline",
            ha="left",
            zorder=z + 1,
        )
    return h


def folder(
    ax,
    x0,
    y0,
    x1,
    y1,
    *,
    label=None,
    tab=None,
    tab_h=2.6,
    size=fs.NOTE,
    fill="white",
    edge=fs.INK,
    lw=fs.LW_AXIS,
    z=2,
    side="left",
):
    """A store: a folder with its name on the tab; (x0, y0, x1, y1) is the body. side puts the tab at the body's left
    end or, mirrored, at its right end, where arrows that enter the body from above at its left do not cross it."""
    if side not in ("left", "right"):
        raise ValueError(f"a folder's tab is on its left or right side, not {side!r}")
    label_width = 0.0
    if label:
        text = ax.text(
            x0 + 1.6 if side == "left" else x1 - 1.6,
            y1 + tab_h / 2,
            label,
            fontsize=size,
            color=fs.INK,
            va="center",
            ha=side,
            zorder=z + 0.1,
        )
        label_width = (
            text.get_window_extent(ax.figure.canvas.get_renderer())
            .transformed(ax.transData.inverted())
            .width
        )
    tab = tab or (label_width + 3.2)
    slope = tab_h * 0.7
    corners = np.asarray(
        [
            (x0, y0),
            (x1, y0),
            (x1, y1),
            (x0 + tab + slope, y1),
            (x0 + tab, y1 + tab_h),
            (x0, y1 + tab_h),
        ]
    )
    if side == "right":
        corners[:, 0] = x0 + x1 - corners[:, 0]
    vertices, codes = [], []
    for i, corner in enumerate(corners):
        before = corners[i - 1] - corner
        after = corners[(i + 1) % len(corners)] - corner
        before_length, after_length = np.linalg.norm(before), np.linalg.norm(after)
        radius = min(0.8, before_length / 2, after_length / 2)
        vertices.extend(
            (
                corner + before * radius / before_length,
                corner,
                corner + after * radius / after_length,
            )
        )
        codes.extend((Path.MOVETO if i == 0 else Path.LINETO, Path.CURVE3, Path.CURVE3))
    vertices.append(vertices[0])
    codes.append(Path.CLOSEPOLY)
    ax.add_patch(
        PathPatch(
            Path(vertices, codes),
            facecolor=fill,
            edgecolor=edge,
            lw=lw,
            zorder=z,
            joinstyle="round",
        )
    )


def cloud(ax, cx, cy, w, *, fill="white", edge=fs.INK, lw=fs.LW_AXIS, z=2):
    """Object storage: a cloud w millimetres wide centred at (cx, cy); returns its height."""
    h = 0.58 * w
    shapes = [(-0.25, -0.02, 0.17), (-0.03, 0.16, 0.24), (0.22, 0.04, 0.18)]
    base = (cx - 0.42 * w, cy - 0.34 * h, 0.84 * w, 0.40 * h)
    inset = lw * 0.3528 / 2
    for layer, (edgecolor, shrink, zz) in enumerate(((edge, 0.0, z), ("none", inset, z + 0.01))):
        for dx, dy, r in shapes:
            ax.add_patch(
                Circle(
                    (cx + dx * w, cy + dy * h),
                    r * w - shrink,
                    facecolor=fill,
                    edgecolor=edgecolor,
                    lw=lw if layer == 0 else 0,
                    zorder=zz,
                )
            )
        ax.add_patch(
            FancyBboxPatch(
                (base[0] + shrink, base[1] + shrink),
                base[2] - 2 * shrink,
                base[3] - 2 * shrink,
                boxstyle=f"round,pad=0,rounding_size={base[3] / 2 - shrink}",
                facecolor=fill,
                edgecolor=edgecolor,
                lw=lw if layer == 0 else 0,
                zorder=zz,
            )
        )
    return h


def laptop(ax, cx, y, w, *, z=2):
    """The user's machine, w millimetres wide with its base at y; returns the screen (x, y, w, h)."""
    sw, sh = 0.82 * w, 0.56 * w
    sx, sy = cx - sw / 2, y + 1.3
    ax.add_patch(
        FancyBboxPatch(
            (sx, sy),
            sw,
            sh,
            boxstyle="round,pad=0,rounding_size=0.9",
            facecolor="white",
            edgecolor=fs.INK,
            lw=fs.LW_AXIS,
            zorder=z,
        )
    )
    ax.add_patch(
        Polygon(
            [
                (cx - w / 2, y),
                (cx + w / 2, y),
                (cx + w / 2 - 1.2, y + 1.3),
                (cx - w / 2 + 1.2, y + 1.3),
            ],
            closed=True,
            facecolor=fs.PAPER,
            edgecolor=fs.INK,
            lw=fs.LW_AXIS,
            zorder=z,
            joinstyle="round",
        )
    )
    return sx + 1.2, sy + 1.2, sw - 2.4, sh - 2.4


def ring(ax, cx, cy, r, width, arcs, *, z=1):
    """Arcs of a ring, each (start degrees, end degrees, colour)."""
    for start, end, colour in arcs:
        ax.add_patch(
            Wedge(
                (cx, cy), r, start, end, width=width, facecolor=colour, edgecolor="none", zorder=z
            )
        )
