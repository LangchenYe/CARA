"""Vector overview of CARA's decision path and three query-reuse scopes.

The public ``draw(output, preview=None)`` entry point is also used by
``draw_results.py``.  The diagram illustrates query execution; its acquisition
plans select candidate profiles, each evaluated on the complete Build history.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

NAVY, BLUE, TEAL, ORANGE = '#22394d', '#426e99', '#217f7a', '#b97035'
GREY, LINE = '#5e6b76', '#ced8df'
LIGHT_BLUE, LIGHT_TEAL, LIGHT_ORANGE = '#edf3f9', '#edf6f3', '#fcf2e8'
PANEL = '#fbfcfd'


def label(ax, x, y, value, size=7.2, color=NAVY, **kw):
    return ax.text(x, y, value, ha=kw.pop('ha', 'center'),
                   va=kw.pop('va', 'center'), fontsize=size, color=color,
                   linespacing=1.26, **kw)


def arrow(ax, start, end, color=GREY, style='-|>', **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
        mutation_scale=7.1, color=color, linewidth=.82,
        shrinkA=1.2, shrinkB=1.2, **kw))


def path(ax, points, color=GREY, arrow_end=True, **kw):
    """Orthogonal connector whose last segment carries the arrowhead."""
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), color=color, linewidth=.82,
                solid_capstyle='round', **kw)
    if arrow_end:
        arrow(ax, points[-2], points[-1], color=color, **kw)
    else:
        ax.plot(*zip(*points[-2:]), color=color, linewidth=.82, **kw)


def box(ax, xy, width, height, title, detail=None, color=BLUE,
        fill=LIGHT_BLUE, title_size=7.4, detail_size=6.6,
        title_fraction=.67, detail_fraction=.27, **kw):
    x, y = xy
    ax.add_patch(FancyBboxPatch(xy, width, height,
        boxstyle='round,pad=0,rounding_size=0.005', facecolor=fill,
        edgecolor=color, linewidth=.8, **kw))
    label(ax, x+width/2, y+height*(title_fraction if detail else .5), title,
          title_size, color, weight='semibold')
    if detail:
        label(ax, x+width/2, y+height*detail_fraction, detail, detail_size, GREY)


def matrix(ax, x, y, width, height, seed=False):
    """Illustrative seven-native/one-cyclic-resident overlap, as in the text."""
    cw, ch = width/5, height/5
    for row in range(5):
        for col in range(5):
            edited = not seed and col < 2
            core = not seed and row < 2 and col < 2
            fixed = row == col and not edited
            face = TEAL if fixed else LIGHT_BLUE if edited else 'white'
            ax.add_patch(Rectangle((x+col*cw, y+(4-row)*ch), cw, ch,
                facecolor=face, edgecolor=LINE, linewidth=.48))
            if core:
                label(ax, x+(col+.5)*cw, y+(4-row+.5)*ch,
                      r'$\ast$', 8.0, BLUE)
            elif fixed:
                label(ax, x+(col+.5)*cw, y+(4-row+.5)*ch,
                      '7', 6.7, 'white', weight='bold')
            elif edited:
                # Edited columns may change; no fixed numerical overlap is
                # claimed outside the displayed 2-by-2 residual core.
                if col == (row+1) % 5:
                    label(ax, x+(col+.5)*cw, y+(4-row+.5)*ch,
                          r'$\cdot$', 7.3, BLUE)
            elif col == (row+1) % 5:
                label(ax, x+(col+.5)*cw, y+(4-row+.5)*ch, '1', 6.5, GREY)
    if not seed:
        ax.add_patch(Rectangle((x, y+3*ch), 2*cw, 2*ch, fill=False,
                              edgecolor=BLUE, linewidth=1.12))


def panel(ax, x, width, number, title, subtitle, color):
    ax.add_patch(FancyBboxPatch((x, .071), width, .610,
        boxstyle='round,pad=0,rounding_size=0.007', facecolor=PANEL,
        edgecolor=LINE, linewidth=.65))
    ax.plot([x+.013, x+width-.013], [.681, .681], color=color,
            linewidth=2.1, solid_capstyle='butt')
    label(ax, x+.016, .650, f'{number}  {title}', 8.45, color,
          ha='left', weight='bold')
    label(ax, x+.016, .612, subtitle, 6.8, GREY, ha='left')


def draw(output, preview=None):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'mathtext.fontset': 'dejavusans'})
    fig = plt.figure(figsize=(7.16, 3.65))
    ax = fig.add_axes([.006, .008, .988, .984])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')

    # (a) A decision contains many virtual proposals but at most one release.
    label(ax, .012, .974,
          '(a)  Decision path: many query alternatives, one released repair',
          8.5, ha='left', weight='bold')
    box(ax, (.012, .802), .145, .137,
        r'Incumbent $\pi^{\mathrm{I}}$',
        'Fixed deployment\nBuild: all windows',
        title_size=7.55, detail_size=6.5)
    box(ax, (.187, .802), .180, .137,
        'CARA queries', 'Virtual seeds\nCompile + acquire profiles',
        color=TEAL, fill=LIGHT_TEAL, title_size=7.7, detail_size=6.65)
    box(ax, (.397, .802), .123, .137,
        r'Top-$P$ records', 'Ranked + physical\n+ seen state',
        title_size=7.3, detail_size=6.55)
    box(ax, (.550, .802), .118, .137,
        'Selection', 'Held-out table\nFreeze one action',
        color=ORANGE, fill=LIGHT_ORANGE,
        title_size=7.6, detail_size=6.55)
    box(ax, (.699, .878), .159, .061,
        'Historical Audit', 'Complete table',
        color=ORANGE, fill=LIGHT_ORANGE, title_size=6.95, detail_size=6.5)
    box(ax, (.699, .802), .159, .061,
        'Fresh admission', 'Independent scenes',
        color=ORANGE, fill=LIGHT_ORANGE, title_size=6.95, detail_size=6.5)
    label(ax, .779, .870, 'OR', 6.5, ORANGE,
          bbox=dict(facecolor='white', edgecolor='none', pad=.2))
    box(ax, (.890, .813), .098, .115,
        'Release', 'One physical\nrepair', color=ORANGE, fill=LIGHT_ORANGE,
        title_size=7.4, detail_size=6.5)
    for a, b in ((.157, .187), (.367, .397), (.520, .550)):
        arrow(ax, (a, .870), (b, .870), TEAL if a < .4 else BLUE)
    path(ax, [(.668, .870), (.683, .870), (.683, .909), (.699, .909)], ORANGE)
    path(ax, [(.683, .870), (.683, .833), (.699, .833)], ORANGE)
    path(ax, [(.858, .909), (.875, .909), (.875, .870), (.890, .870)], ORANGE)
    path(ax, [(.858, .833), (.875, .833), (.875, .870)], ORANGE,
         arrow_end=False)
    # This feedback changes only a proposal context, never the deployment.
    path(ax, [(.459, .802), (.459, .757), (.277, .757), (.277, .802)],
         TEAL, linestyle=(0, (2.3, 2.1)))
    label(ax, .368, .757, 'Next virtual seed', 6.5, TEAL,
          bbox=dict(facecolor='white', edgecolor='none', pad=1.6))
    label(ax, .988, .757, r'Failure retains $\pi^{\mathrm{I}}$',
          6.65, ORANGE, ha='right')

    # (b) Scope boundaries are compilation / reuse boundaries, not data splits.
    label(ax, .012, .718, '(b)  Query compiler zoom', 8.5,
          ha='left', weight='bold')
    label(ax, .988, .718,
          'Acquire candidate profiles; each exact profile uses all Build windows',
          6.8, GREY, ha='right')
    panel(ax, .012, .273, '1', 'Seed reuse',
          'Shared across edited pairs', TEAL)
    panel(ax, .309, .283, '2', 'Family reuse',
          'Shared across candidate masks', BLUE)
    panel(ax, .616, .372, '3', 'Candidate acquisition',
          'Plan choice depends on query shape and selectivity', TEAL)

    # Seed-universal matching and incremental common-host evidence.
    matrix(ax, .035, .357, .118, .223, seed=True)
    label(ax, .219, .547, 'Seed\noverlap', 7.2, TEAL, weight='semibold')
    label(ax, .219, .461, 'One sparse scan\nFix safe edges', 6.7, GREY)
    label(ax, .219, .377, 'Resident keys\nshare load views', 6.5, GREY)
    label(ax, .149, .318,
          r'$3W^0_{ij}\geq s_i+t_j$', 8.0, TEAL, weight='semibold')
    label(ax, .149, .271, 'Incremental host summaries',
          7.05, BLUE, weight='semibold')
    box(ax, (.105, .208), .088, .036, 'Root',
        color=BLUE, fill=LIGHT_BLUE, title_size=6.6)
    for i, x in enumerate((.044, .086, .128, .170, .212, .254)):
        changed = i < 2
        ax.plot([.149, x], [.208, .161], color=BLUE if changed else LINE,
                linewidth=.65)
        ax.add_patch(Rectangle((x-.014, .132), .028, .028,
            edgecolor=BLUE if changed else LINE,
            facecolor=LIGHT_BLUE if changed else 'white', linewidth=.65))
    label(ax, .149, .098, 'Counts, sums and top-three peaks', 6.5, GREY)

    # Editing the first two columns restores at most their selected edges.
    matrix(ax, .331, .357, .118, .223, seed=False)
    label(ax, .520, .545, 'Restore two\nedited columns',
          6.75, BLUE, weight='semibold')
    label(ax, .520, .451, 'Small residual\nassignment core', 6.6, GREY)
    label(ax, .450, .321,
          'Origin counts share movement queries', 6.9, BLUE)
    box(ax, (.328, .223), .245, .070,
        r'Compile family radii $\rho_-,\rho_+$',
        'Cover every mutable subset', color=TEAL, fill=LIGHT_TEAL,
        title_size=7.15, detail_size=6.5)
    label(ax, .450, .202, r'$L\ \leq\ \mathrm{canonical\ load}\ \leq\ U$',
          7.0, TEAL)
    box(ax, (.328, .132), .107, .047, r'Half sums $S_0$',
        color=BLUE, fill=LIGHT_BLUE, title_size=6.85)
    box(ax, (.466, .132), .107, .047, r'Half sums $S_1$',
        color=BLUE, fill=LIGHT_BLUE, title_size=6.85)
    label(ax, .450, .155, '+', 8.8, BLUE)
    label(ax, .450, .098, 'Two reads + one addition per cell', 6.5, GREY)
    # Scope-specialization arrows are kept in the panel gutters.
    arrow(ax, (.286, .467), (.307, .467), GREY)
    arrow(ax, (.593, .467), (.614, .467), GREY)

    # Both acquisition plans feed the same exact consumer.
    label(ax, .708, .564, 'Direct refined-key', 7.05, BLUE,
          weight='semibold')
    label(ax, .887, .564, 'Optional staged', 7.05, BLUE,
          weight='semibold')
    box(ax, (.636, .408), .145, .109, r'Refined keys $U(a)$',
        'Optional J/V\npre-screen', title_size=7.05, detail_size=6.5,
        title_fraction=.76, detail_fraction=.31)
    label(ax, .887, .538, 'Exact pilot if needed', 6.5, TEAL)
    box(ax, (.808, .477), .160, .040, 'Coarse bounds', title_size=6.9)
    box(ax, (.808, .408), .160, .040, r'Refined keys $U(a)$', title_size=6.9)
    arrow(ax, (.888, .477), (.888, .448), BLUE)
    arrow(ax, (.654, .408), (.654, .350), TEAL)
    arrow(ax, (.951, .408), (.951, .350), TEAL)
    label(ax, .803, .379, r'Omit only strict losers: $U(a)<\theta$',
          6.7, ORANGE)
    box(ax, (.642, .270), .321, .080, 'Exact profile batches',
        'Complete Build windows; canonical loads', color=TEAL,
        fill=LIGHT_TEAL, title_size=7.65, detail_size=6.6)
    arrow(ax, (.803, .270), (.803, .220), TEAL)
    box(ax, (.642, .145), .321, .075,
        r'Top-$P$ retention + physical recovery',
        r'Exact cutoff $\theta$; preserve seen state', color=TEAL,
        fill=LIGHT_TEAL, title_size=7.1, detail_size=6.6)
    label(ax, .803, .098, r'$\theta$ comes only from a full exact pool',
          6.6, GREY)

    ax.plot([.012, .988], [.052, .052], color=LINE, linewidth=.7)
    label(ax, .012, .025,
          'Query contract: identical ordered records, canonical profiles, '
          'deterministic physical ties and continuing seen state',
          6.9, GREY, ha='left')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if preview:
        preview = Path(preview)
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=260)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('figures/method.pdf'))
    parser.add_argument('--preview', type=Path)
    args = parser.parse_args()
    draw(args.output, args.preview)
