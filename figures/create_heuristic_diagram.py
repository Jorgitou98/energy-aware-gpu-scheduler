#!/usr/bin/env python3
"""Generate Figure 3 using only Python and Matplotlib (no LaTeX installation).

Run:
    python create_heuristic_diagram.py
    python create_heuristic_diagram.py --output-dir figures --dpi 300

Outputs: heuristic.pdf, .png, and .svg.
All coordinates, labels, colors, and schematic examples are in this file.
Coordinates are expressed in nominal centimeters; font sizes are points.
The figure illustrates an algorithm, not the results of a measured run.
It illustrates U/P ranking and the optional objective-bound pruning variant.
This script is a schematic renderer; it does not implement or benchmark the planner.
"""

from pathlib import Path
import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "figure2-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# ------------------------- EDITABLE SETTINGS ------------------------------
WIDTH_CM = 18.6
HEIGHT_SCALE = 0.92  # Compact vertical geometry; text retains its original size.
FONT_SCALE = 1.0
DEFAULT_DPI = 300
BEAM_WIDTH = 512
DIVERSITY_WIDTH = 256
MAX_PER_SIGNATURE = 2
REFINEMENT_PASSES = 4
NAME = "heuristic"
COLORS = {
    "blue": "#426EA1", "orange": "#BC843A",
    "green": "#2B8878", "purple": "#8A6FA1",
    "ink": "#151515", "gray": "#666666",
    "border": "#C4C4C4", "branch": "#B3B3B3",
}
# Each tuple describes a color and two (frequency MHz, occupied slices) pairs.
# Zero frequency denotes an unused GPU. Swapping identical GPUs does not
# change a signature. The color does not encode model quality or objective.
SIGNATURES = [
    ("blue",   (1200, 4), (0, 0)),
    ("orange", (1200, 2), (1200, 2)),
    ("green",  (900, 2),  (1200, 2)),
    ("purple", (900, 4),  (0, 0)),
]
# --------------------------------------------------------------------------


def mix(color, proportion, background="white"):
    """Mix a color with white, or with another background color."""
    return tuple(proportion * a + (1 - proportion) * b
                 for a, b in zip(to_rgb(color), to_rgb(background)))


def draw_figure(output_dir, dpi=DEFAULT_DPI):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8.5 * FONT_SCALE, "text.color": COLORS["ink"],
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "text.usetex": False,
    })
    # Preserve the width and font sizes; shorten vertical spacing and boxes.
    xmin, xmax, ymin, ymax = -.03, 18.57, -.01*HEIGHT_SCALE, 8.68*HEIGHT_SCALE
    scale = WIDTH_CM / (xmax - xmin)
    fig = plt.figure(figsize=(WIDTH_CM / 2.54, (ymax-ymin)*scale / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
    ax.set_axis_off()

    def text(x, y, label, size=8.5, ha="center", color=None,
             bold=False, background=None, rotation=0, **kwargs):
        return ax.text(x, y*HEIGHT_SCALE, label, ha=ha, va="center",
                       fontsize=size*FONT_SCALE,
                       fontweight="bold" if bold else "normal",
                       color=color or COLORS["ink"], linespacing=1.25,
                       rotation=rotation, zorder=7,
                       bbox=None if background is None else
                       dict(facecolor=background, edgecolor="none", pad=.8),
                       **kwargs)

    def line(points, color=None, lw=.55, dashed=False, zorder=3):
        ax.plot(*zip(*[(x, y*HEIGHT_SCALE) for x, y in points]),
                color=color or COLORS["branch"], lw=lw,
                linestyle=(0, (4, 4)) if dashed else "-",
                solid_capstyle="butt", zorder=zorder)

    def arrow(points, color=None, lw=.60, dashed=False):
        color = color or "#737373"
        if len(points) > 2:
            line(points[:-1], color, lw, dashed)
        ax.add_patch(FancyArrowPatch(
            (points[-2][0], points[-2][1]*HEIGHT_SCALE),
            (points[-1][0], points[-1][1]*HEIGHT_SCALE),
            arrowstyle="-|>", mutation_scale=5.5,
            shrinkA=0, shrinkB=0, color=color, linewidth=lw,
            linestyle=(0, (4, 4)) if dashed else "-", zorder=4))

    def box(x0, y0, x1, y1, fill="white", edge=None,
            radius=.075, lw=.5, dashed=False, zorder=1):
        style = (0, (4, 4)) if dashed else "-"
        y0, y1 = y0*HEIGHT_SCALE, y1*HEIGHT_SCALE
        if radius:
            p = FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge or COLORS["border"],
                              linewidth=lw, linestyle=style, zorder=zorder)
        else:
            p = Rectangle((x0, y0), x1-x0, y1-y0,
                          facecolor=fill, edgecolor=edge or COLORS["border"],
                          linewidth=lw, linestyle=style, zorder=zorder)
        ax.add_patch(p)

    def dot(x, y, color, radius=.15):
        ax.add_patch(Circle((x, y*HEIGHT_SCALE),
                            radius*HEIGHT_SCALE if radius < .1 else radius,
                            facecolor=color,
                            edgecolor="white", linewidth=.4, zorder=5))

    def cross(x, y, radius=.05, color="#999999"):
        line([(x-radius, y-radius), (x+radius, y+radius)], color, .5, zorder=5)
        line([(x-radius, y+radius), (x+radius, y-radius)], color, .5, zorder=5)

    def heading(x, y, tag, label):
        ax.add_patch(Circle((x, y*HEIGHT_SCALE), .21, facecolor="#F0F0F0",
                            edgecolor="#B8B8B8", linewidth=.4, zorder=6))
        text(x, y, tag, 7, bold=True, color="#404040")
        text(x+.33, y, label, 8.5, ha="left", bold=True)

    blue, orange, green = [COLORS[c] for c in ("blue", "orange", "green")]
    faint_blue, faint_green = mix(blue, .02), mix(green, .015)

    # Panel titles.
    text(0, 8.50, "(a) Stage 1: greedy deployment", 9.6, ha="left", bold=True)
    text(8.65, 8.50, "(b) Stage 2: beam search", 9.6, ha="left", bold=True)

    # (a1) Preparation.
    box(0, 7.40, 8, 8.20, faint_blue)
    heading(.32, 7.80, "a1", "Prepare")
    text(4.43, 7.80, "One replica per service\nFewest eligible options first", 7.3)
    text(7.83, 7.80, r"$q_0=0$", ha="right")
    arrow([(4, 7.40), (4, 7.21)])

    # (a2) Candidate insertion: reconfigure the occupied GPU or use an empty one.
    box(0, 3.72, 8, 7.21)
    box(.01, 6.66, 7.99, 7.20, mix(blue, .03), edge="none")
    line([(.01, 6.66), (7.99, 6.66)], "#D9D9D9", .5)
    heading(.32, 6.94, "a2", r"Generate insertion candidates for $s_2$")
    text(1.77, 6.30, "Current plan", bold=True)
    text(.18, 5.80, r"$g_1$", 7.3, ha="left")
    box(.58, 5.57, 2.98, 6.03, mix(blue, .15), mix(blue, .65), radius=0)
    text(1.78, 5.80, r"$s_1$: 4 slices")
    text(.18, 4.80, r"$g_2$", 7.3, ha="left")
    box(.58, 4.57, 2.98, 5.03, edge="#B3B3B3", radius=0, dashed=True)
    text(1.78, 4.80, "unused", color="#808080")

    box(4.02, 5.60, 7.77, 6.47, mix(green, .03), mix(green, .65))
    text(4.18, 6.27, r"Reconfigure $g_1$", 7.3, ha="left", color=mix(green, .65, "black"))
    text(4.22, 5.89, r"$g_1$", 7.3, ha="left")
    box(4.75, 5.71, 5.95, 6.07, mix(blue, .15), mix(blue, .65), radius=0)
    box(5.95, 5.71, 7.15, 6.07, mix(orange, .20), mix(orange, .80), radius=0)
    text(5.35, 5.89, r"$s_1$: 2")
    text(6.55, 5.89, r"$s_2$: 2")

    box(4.02, 4.06, 7.77, 5.42)
    text(4.18, 5.18, r"Use $g_2$", 7.3, ha="left", color=COLORS["gray"])
    text(4.22, 4.82, r"$g_1$", 7.3, ha="left")
    box(4.75, 4.67, 7.15, 4.97, mix(blue, .15), mix(blue, .65), radius=0)
    text(5.95, 4.82, r"$s_1$: 4 slices", 7.3)
    text(4.22, 4.40, r"$g_2$", 7.3, ha="left")
    box(4.75, 4.25, 5.95, 4.55, mix(orange, .20), mix(orange, .80), radius=0)
    box(5.95, 4.25, 7.15, 4.55, edge="#B3B3B3", radius=0, dashed=True)
    text(5.35, 4.40, r"$s_2$: 2", 7.3)
    arrow([(3.06, 5.49), (3.56, 5.49), (3.56, 6.01), (3.96, 6.01)])
    arrow([(3.56, 5.49), (3.56, 4.80), (3.96, 4.80)])
    text(.17, 3.90, "Jointly reselect variants, MIG sizes and a common GPU frequency",
         7.3, ha="left", color="#595959")

    # (a3) Commit one insertion, update q, and proceed to the next service.
    arrow([(4, 3.72), (4, 3.48)])
    box(0, 2.64, 8, 3.48, faint_blue)
    heading(.32, 3.06, "a3", "Select one")
    text(3.88, 3.06, r"$\max\,(\Delta U-q\Delta P)$")
    arrow([(5.36, 3.06), (5.84, 3.06)], mix(green, .75, "black"))
    text(6.89, 3.27, "Update", 7.3)
    text(6.89, 2.92, r"$q\leftarrow U/P$")
    arrow([(8, 3.06), (8.30, 3.06), (8.30, 6.94), (8.02, 6.94)])
    text(8.30, 4.92, "Next service", 7.3, color=COLORS["gray"],
         rotation=90, background="white")

    # (a4) Refinement keeps service-to-GPU membership fixed.
    arrow([(4, 2.64), (4, 2.30)])
    text(4.17, 2.47, "All services placed", 7.3, ha="left", color=COLORS["gray"])
    box(0, .26, 8, 2.30, faint_blue)
    heading(.32, 2.05, "a4", r"$N_{\rm ref}$ refinement passes")
    text(7.82, 2.05, "Fixed GPU membership", 7.3, ha="right", color=COLORS["gray"])
    for x, gpu in [(1.21, 1), (3.51, 2)]:
        box(x-.95, 1.25, x+.95, 1.69)
        text(x, 1.47, rf"$\max(U_{gpu}-qP_{gpu})$", 8)
    text(4.94, 1.47, r"$\cdots$")
    arrow([(5.25, 1.47), (5.85, 1.47)])
    text(6.67, 1.47, r"$q\leftarrow U/P$")
    arrow([(6.67, 1.21), (6.67, 1.07), (1.21, 1.07), (1.21, 1.22)])
    line([(.20, .91), (7.80, .91)], "#D9D9D9", .5)
    text(.23, .57, r"Keep best $U/P$ across the $N_{\rm ref}$ passes", 7.7,
         ha="left", bold=True)
    text(7.70, .57, r"$\eta_G$", 9, ha="right")

    # (b1) Arrangement examples. An arrangement can subsequently be placed
    # on different GPUs; the widths here denote MIG instance sizes.
    box(8.65, 6.45, 18.3, 8.20, faint_green)
    heading(8.97, 7.88, "b1", "Build arrangements")
    text(18.08, 7.84, "One variant per arrangement\nReplica rates sum to $\\lambda_s$",
         7.3, ha="right", color=COLORS["gray"])
    for x, letter in [(8.99, "A"), (11.65, "B"), (14.8, "C")]:
        text(x, 7.11, letter, bold=True)
    for x0, x1, label, size in [
        (9.32, 10.92, r"$\lambda_s$", "2 slices"),
        (12.0, 12.8, r"$\lambda_s/2$", "1 slice"),
        (13.15, 13.95, r"$\lambda_s/2$", "1 slice"),
        (15.15, 16.75, r"$r_1$", "2 slices"),
        (17.1, 17.9, r"$r_2$", "1 slice"),
    ]:
        box(x0, 6.88, x1, 7.34, "#F0F0F0", "#737373", radius=.035)
        text((x0+x1)/2, 7.11, label)
        text((x0+x1)/2, 6.66, size, 7.3, color=COLORS["gray"])
    arrow([(11.1, 6.45), (11.1, 6.20)])

    # (b2)-(b3) Representative expansion, pruning, and selection.
    for x0, x1, color, tag, label in [
        (8.65, 13.40, blue, "b2", "Expand placements"),
        (13.63, 18.3, green, "b3", "Prune and select"),
    ]:
        box(x0, .96, x1, 6.20, mix(color, .01))
        box(x0+.01, 5.64, x1-.01, 6.19, mix(color, .05), edge="none")
        line([(x0+.01, 5.64), (x1-.01, 5.64)], "#D9D9D9", .5)
        heading(x0+.32, 5.92, tag, label)
    for x, label in [
        (9.16, "Empty"), (10.79, f"After $s_1$\n$\\leq {BEAM_WIDTH}$"),
        (12.95, r"Place $s_2$"), (15.11, f"Retain\n$\\leq {BEAM_WIDTH}$"),
        (17.32, "Next service"),
    ]:
        text(x, 5.30, label, 7.3)

    dy = .29
    box(10.45, 2.53+dy, 11.13, 4.76+dy, "#FAFAFA", edge="none")
    box(14.77, 2.53+dy, 15.45, 4.76+dy, mix(green, .05), edge="none")
    root = (9.03, 3.67+dy)
    dot(*root, "#737373")
    parents = [(4.49, "A"), (3.93, "B"), (3.37, "B"), (2.81, "C")]
    for y, arrangement in parents:
        point = (10.79, y+dy)
        line([root, point])
        dot(*point, "#8C8C8C")
        text(9.81, 3.67+.44*(y-3.67)+dy, arrangement, 7.3,
             background=mix(blue, .01))
        for offset in [-.16, 0, .16]:
            line([point, (12.95, y+offset+dy)])
    candidates = [
        (4.65, "blue"), (4.49, "blue"), (4.33, "orange"),
        (4.09, "orange"), (3.93, "orange"), (3.77, "blue"),
        (3.53, "green"), (3.37, "green"), (3.21, "green"),
        (2.97, "purple"), (2.81, "purple"), (2.65, "purple"),
    ]
    kept = {1, 4, 6, 10}  # zero-based indices of the representative survivors
    for index, (y, color_name) in enumerate(candidates):
        y += dy
        color = COLORS[color_name]
        dot(12.95, y, color, radius=.08)
        if index not in kept:
            line([(12.95, y), (14.16, y)], "#C4C4C4", dashed=True)
            cross(14.145, y, .045, "#A6A6A6")
            continue
        line([(12.95, y), (15.11, y)], mix(color, .75, "black"), .8)
        dot(15.11, y, color)
        if color_name == "orange":
            # Both GPUs use 1200 MHz. A later replica requiring 900 MHz
            # cannot be added without changing an earlier placement.
            line([(15.11, y), (17.32, y)])
            cross(17.32, y, .07)
        else:
            for offset in [-.12, .12]:
                line([(15.11, y), (17.32, y+offset)])
                dot(17.32, y+offset, "#B3B3B3", radius=.08)
    text(17.96, 3.69+dy, r"$\cdots$")

    # Concrete signature legend, applying to candidates and survivors after s2.
    text(8.82, 2.60, "Signature examples (1 cell = 1 slice)", 7.3,
         ha="left", color="#404040")
    text(10.18, 2.29, "GPU 1", 6.8, color=COLORS["gray"])
    text(12.40, 2.29, "GPU 2", 6.8, color=COLORS["gray"])
    for y, (color_name, gpu1, gpu2) in zip([2.02, 1.73, 1.44, 1.15], SIGNATURES):
        color = COLORS[color_name]
        dot(8.99, y, color, radius=.09)
        for (frequency, occupied), label_x, cell_x in [
            (gpu1, 10.29, 10.39), (gpu2, 12.52, 12.62),
        ]:
            label = f"{frequency} MHz" if frequency else "idle"
            text(label_x, y, label, 6.5, ha="right")
            for j in range(4):
                x = cell_x+j*.17
                box(x, y-.065, x+.14, y+.065,
                    mix(color, .75) if j < occupied else "white",
                    mix(color, .70, "black") if j < occupied else "#B3B3B3",
                    radius=0, lw=.3)

    # Pruning compares an admissible upper bound on the final ratio
    # to the complete greedy seed. The threshold is shown directly.
    line([(13.82, 2.76), (18.10, 2.76)], "#D9D9D9", .5)
    text(15.965, 2.52, r"Prune: final $U/P$ upper bound $\leq\eta_G$", 7.1)
    # Highlight the two complementary quotas. If the diversity pass retains
    # fewer than 256 states, the unrestricted pass fills the remaining slots
    # to 512, or retains all available candidates if fewer than 512 exist.
    text(15.965, 2.18, f"{BEAM_WIDTH}-state beam", 7.7, bold=True)
    for x0, x1, heading_label, rule, tint in [
        (13.81, 15.88, "Best by $U/P$", r"$\leq 2$ per signature", .055),
        (16.06, 18.12, "Next best by $U/P$", "No signature cap", .015),
    ]:
        box(x0, .99, x1, 1.99, mix(green, tint), edge="#BCCAC5", lw=.45)
        xc=(x0+x1)/2
        text(xc, 1.77, str(DIVERSITY_WIDTH), 8.0, bold=True,
             color=mix(green, .65, "black"))
        text(xc, 1.44, heading_label, 7.0)
        text(xc, 1.17, rule, 6.6, color="#505050")

    # Greedy remains the fallback if the beam yields no better complete plan.
    arrow([(16.09, .96), (16.09, .80)])
    box(8.65, .26, 18.30, .80, mix(green, .03))
    heading(8.97, .53, "b4", r"Return best complete plan by $U/P$")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Tight export uses the painted artists, not the axes rectangle. Only a
    # 0.5-pt border is retained, including around the stage titles.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    artists = list(ax.texts) + list(ax.patches) + list(ax.lines)
    extents = [artist.get_window_extent(renderer) for artist in artists
               if artist.get_visible()]
    bbox = Bbox.union(extents).transformed(fig.dpi_scale_trans.inverted())
    bbox = Bbox.from_extents(bbox.x0-.5/72, bbox.y0-.5/72,
                            bbox.x1+.5/72, bbox.y1+.5/72)
    for extension in ("pdf", "png", "svg"):
        path = output_dir / f"{NAME}.{extension}"
        # Build timestamps are not experimental provenance. Omit automatic dates.
        metadata = ({"CreationDate": None, "ModDate": None} if extension == "pdf"
                    else {"Date": None} if extension == "svg" else {})
        with matplotlib.rc_context({"svg.hashsalt": "efficient-mig-heuristic"}):
            fig.savefig(path, dpi=dpi, bbox_inches=bbox, pad_inches=0, metadata=metadata)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--outdir", type=Path,
                        default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()
    draw_figure(args.output_dir, args.dpi)
