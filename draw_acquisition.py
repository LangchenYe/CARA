"""Plot exact profile acquisition and its independent batch-size study."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(queries, parameters, output, selection_budget=None):
    records = json.loads((Path(queries) / 'records.json').read_text())
    azure = next(row for row in records if row['source'] == 'azure2019')
    development = json.loads((Path(parameters) / 'records.json').read_text())
    batches = [8, 16, 32, 64]
    final = 'indexed' if 'indexed' in azure['median_seconds'] else 'compiled'
    colors = {'no_bootstrap': '#b67a45', final: '#147d79'}
    labels = {'no_bootstrap': 'Delayed acquisition', final: 'CARA'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 7.5,
                         'ytick.labelsize': 7.5, 'pdf.fonttype': 42})
    columns = 3 if selection_budget is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(7.16, 2.05))
    fig.subplots_adjust(left=.067, right=.99, bottom=.24, top=.87, wspace=.48)
    ax = axes[0]
    for plan in ['no_bootstrap', final]:
        families = azure['trials'][0]['plans'][plan]['families']
        values = [row['counters']['profiled_masks'] for row in families]
        ax.plot(np.arange(1, len(values) + 1), np.cumsum(values), marker='o',
                markersize=3.5, linewidth=1.6, color=colors[plan], label=labels[plan])
    ax.set(xlabel='Completed repair families', ylabel='Cumulative exact profiles',
           xticks=range(1, 9))
    ax.set_title('(a) Acquisition on Azure', loc='left', fontweight='bold', fontsize=8.1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc='center right', fontsize=6.6)
    ax.grid(axis='y', color='#e4e8eb', linewidth=.65)
    ax.set_axisbelow(True)
    ax = axes[1]
    totals = [sum(row['medians'][str(b)]['cpu_seconds'] for row in development) for b in batches]
    trials = np.asarray([[sum(row['trials'][rep]['plans'][str(b)]['cpu_seconds']
                             for row in development) for rep in range(3)] for b in batches])
    # Min/max sums describe serial repetitions; they are not confidence intervals.
    lows = np.minimum(trials.min(axis=1), totals)
    highs = np.maximum(trials.max(axis=1), totals)
    ax.errorbar(range(4), totals, yerr=[np.array(totals) - lows, highs - np.array(totals)],
                fmt='o-', color='#3d678f', linewidth=1.6, markersize=4, capsize=3)
    selected = int(np.argmin(totals))
    ax.scatter([selected], [totals[selected]], s=75, facecolors='none',
               edgecolors='#147d79', linewidths=1.6, zorder=5)
    ax.set_xticks(range(4), [str(b) for b in batches])
    ax.set(xlabel='Profile batch limit', ylabel='Aggregate query CPU (s)')
    ax.set_title('(b) Independent batch study', loc='left', fontweight='bold', fontsize=8.1)
    ax.grid(axis='y', color='#e4e8eb', linewidth=.65)
    ax.set_axisbelow(True)
    if selection_budget is not None:
        study = json.loads((Path(selection_budget) / 'records.json').read_text())
        ax = axes[2]
        for plan, label, color in [('no_score', 'No profile filter', '#b67a45'),
                                   ('compiled', 'CARA', '#147d79')]:
            ax.plot(range(len(study)),
                    [row['medians'][plan]['profiled_masks'] for row in study],
                    marker='o', markersize=3.5, linewidth=1.6, label=label, color=color)
        ax.set_xticks(range(len(study)), [str(row['top_p']) for row in study])
        ax.set(xlabel=r'Retained result size $P$', ylabel='Exact profiles acquired', ylim=(0, None))
        ax.set_title('(c) Result size', loc='left', fontweight='bold', fontsize=8.1)
        ax.legend(frameon=False, fontsize=6.6, loc='center right')
        ax.grid(axis='y', color='#e4e8eb', linewidth=.65)
        ax.set_axisbelow(True)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color('#aeb8bf')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-budget', type=Path)
    args = parser.parse_args()
    draw(args.queries, args.parameters, args.output, args.selection_budget)
