"""Vector method figure for CARA's family-wide matching-core compilation."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as PlotPath


NAVY = "#193249"
BLUE = "#3166a0"
TEAL = "#127b79"
ORANGE = "#c07437"
GREY = "#62717c"
PALE = "#b0bac2"


def label(ax, x, y, value, size=8, **kwargs):
    return ax.text(x, y, value, fontsize=size, ha=kwargs.pop("ha", "center"),
                   va="center", color=kwargs.pop("color", NAVY), **kwargs)


def box(ax, x, y, width, height, title, subtitle="", *, color=BLUE,
        fill="#f3f6fa", title_size=8, subtitle_size=7):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=.006,rounding_size=.017", linewidth=.8,
        edgecolor=color, facecolor=fill, zorder=0))
    label(ax, x + width/2, y + height*(.65 if subtitle else .5), title,
          title_size, weight="semibold")
    if subtitle:
        label(ax, x + width/2, y + height*.25, subtitle, subtitle_size, color=GREY)


def arrow(ax, start, end, color=GREY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7.2,
                               linewidth=.85, color=color))


def lock(ax, x, y, color=TEAL):
    ax.add_patch(Arc((x, y + .012), .026, .030, theta1=0, theta2=180,
                     linewidth=.9, color=color, zorder=4))
    ax.add_patch(FancyBboxPatch((x - .019, y - .016), .038, .029,
        boxstyle="round,pad=0,rounding_size=.003", linewidth=.5,
        facecolor=color, edgecolor=color, zorder=4))
    ax.plot([x], [y - .001], marker="o", markersize=.9, color="white", zorder=5)


def draw(output, preview=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.35),
        gridspec_kw={"width_ratios": [1.0, 1.16, 1.12]})
    fig.subplots_adjust(left=.012, right=.993, bottom=.035, top=.985, wspace=.115)
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    left, middle, right = axes

    # Resident identities remain distinct; two origin counts share a constraint.
    label(left, .0, .963, "(a) Two reuse relations", 8.8, ha="left", weight="bold")
    label(left, .5, .867, "Four mutable workloads", 7.6, color=GREY)
    for x, number, color in ((.20, 1, BLUE), (.38, 2, BLUE),
                             (.63, 3, ORANGE), (.81, 4, ORANGE)):
        left.scatter([x], [.775], s=205, facecolor=color, edgecolor="white", linewidth=.6)
        label(left, x, .775, str(number), 8.4, color="white", weight="bold")
    label(left, .29, .688, "from host A", 7.1, color=BLUE)
    label(left, .72, .688, "from host B", 7.1, color=ORANGE)
    arrow(left, (.5, .66), (.5, .62))
    box(left, .035, .465, .93, .145, "16 candidate repairs", "Fixed backgrounds stay nonempty",
        title_size=8.0, subtitle_size=6.6)
    label(left, .23, .365, r"$A_1=\{1,3\}$", 8.5)
    label(left, .23, .282, r"$A_2=\{1,4\}$", 8.5)
    arrow(left, (.46, .36), (.60, .33), TEAL)
    arrow(left, (.46, .28), (.60, .31), TEAL)
    label(left, .78, .326, r"$\mathbf{a}=(1,1)$", 8.9, color=TEAL)
    box(left, .015, .08, .465, .12, "Resident identity", "Distinct load views",
        title_size=6.8, subtitle_size=6.1)
    box(left, .52, .08, .465, .12, "Origin counts", "Shared matching",
        color=TEAL, fill="#eef7f5", title_size=6.8, subtitle_size=6.1)
    arrow(left, (.23, .245), (.23, .204), BLUE)
    arrow(left, (.78, .255), (.78, .204), TEAL)
    label(left, .5, .02, r"$16$ masks, at most $9$ count states", 6.9, color=GREY)

    # Five 8-resident hosts. The seed has 7 own + 1 incoming residents per
    # group, hence connected cyclic support. Two native A and two native B
    # residents vary across columns 1 and 2. C-3, D-4 and E-5 are persistent.
    label(middle, .0, .963, "(b) Compile the matching core", 8.8,
          ha="left", weight="bold")
    label(middle, .5, .885, "Connected family support", 7.6, color=GREY)
    label(middle, .19, .819, "Origins", 7.1)
    label(middle, .81, .819, "Groups", 7.1)
    ys = [.738, .641, .533, .425, .317]
    x0, x1 = .19, .81
    middle.add_patch(FancyBboxPatch((.108, .591), .784, .191,
        boxstyle="round,pad=.008,rounding_size=.014", linewidth=.7,
        edgecolor=BLUE, facecolor="#f3f6fa", zorder=0))

    # Weak possible overlaps show that the original graph is connected.
    for row, column in ((1, 2), (2, 3), (3, 4)):
        middle.plot([x0 + .027, x1 - .027], [ys[row], ys[column]],
                    color=PALE, linewidth=.85, linestyle=(0, (2.5, 2.0)), zorder=1)
        label(middle, .69, ys[row] + .76*(ys[column] - ys[row]) + .013,
              "1", 6.4, color=GREY,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": .45}, zorder=4)
    # The E-to-1 cycle edge goes around the left/top margin to keep the
    # persistent edges readable. Its endpoints are part of the same support.
    route = PlotPath([(x0-.027, ys[4]), (.047, ys[4]), (.047, .787),
                      (.81, .787), (.81, ys[0]+.027)],
                     [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.LINETO,
                      PlotPath.LINETO, PlotPath.LINETO])
    middle.add_patch(PathPatch(route, fill=False, color=PALE, linewidth=.85,
                              linestyle=(0, (2.5, 2.0)), joinstyle="round", zorder=1))
    label(middle, .047, .565, "1", 6.4, color=GREY,
          bbox={"facecolor": "white", "edgecolor": "none", "pad": .5}, zorder=4)

    # The top 2-by-2 region contains the varying matching coefficients.
    for row, color in ((0, BLUE), (1, ORANGE)):
        for column in (0, 1):
            middle.plot([x0+.027, x1-.027], [ys[row], ys[column]],
                        color=color, linewidth=1.0, alpha=.9, zorder=2)
    for index in (2, 3, 4):
        middle.plot([x0+.027, x1-.027], [ys[index], ys[index]],
                    color=TEAL, linewidth=1.9, zorder=2)
        label(middle, .46, ys[index], "7", 7.2, color=TEAL, weight="semibold",
              bbox={"facecolor": "white", "edgecolor": "none", "pad": .8}, zorder=4)
        lock(middle, .58, ys[index])
    for index, y in enumerate(ys):
        row_color = BLUE if index == 0 else ORANGE if index == 1 else TEAL
        column_color = BLUE if index < 2 else TEAL
        for x, value, color in ((x0, chr(65+index), row_color),
                                (x1, str(index+1), column_color)):
            middle.scatter([x], [y], s=108, facecolor="white", edgecolor=color,
                           linewidth=1, zorder=5)
            label(middle, x, y, value, 7.1, color=color, zorder=6)
    label(middle, .5, .244, r"$7\ \geq\ 1+1$", 8.7, color=TEAL, weight="semibold")
    label(middle, .5, .199, "Lock C–3, D–4 and E–5 for every repair", 6.5, color=GREY)
    label(middle, .5, .118, r"$v=21+v_{2\times2}$", 9.1)
    label(middle, .5, .062, "Here: a 2×2 residual core", 7.2, color=BLUE)
    label(middle, .5, .015, "Larger cores use exact two-column replay", 6.4, color=GREY)

    # The original repair-and-release consumers remain the same.
    label(right, .0, .963, "(c) Compute for the decision", 8.8,
          ha="left", weight="bold")
    stages = [(.755, "Capacity + movement", "Reject infeasible repairs", ORANGE, "#fcf5ed"),
              (.555, "Optimistic gain key", "Skip profiles below the cutoff", TEAL, "#eef7f5"),
              (.355, "Canonical shared profiles", "Resolve the remaining repairs", BLUE, "#f3f6fa"),
              (.155, "Final top-P actions", "Restore their physical host maps", BLUE, "#f3f6fa")]
    for y, title, subtitle, color, fill in stages:
        box(right, .025, y, .95, .143, title, subtitle, color=color, fill=fill,
            title_size=7.8, subtitle_size=6.8)
    for y in (.755, .555, .355):
        arrow(right, (.5, y-.008), (.5, y-.053))
    arrow(right, (.5, .145), (.5, .092))
    label(right, .5, .062, "Selection freezes one action", 7.5, weight="semibold")
    label(right, .5, .004, "Historical census  /  fresh certificate", 6.8, color=GREY)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    if preview is not None:
        preview = Path(preview)
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("figures") / "method.pdf")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    draw(args.output, args.preview)
