"""Draw the compiled-query and full-source outcome figures from complete studies."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

SOURCES=('azure2019','borg2019_d_trace','borg2019_e_trace','huawei2025')
SHORT=('AZ','BD','BE','HW')
TEAL,BLUE,GREY,ORANGE,PURPLE='#187c79','#416b93','#697680','#bd793f','#8a6b96'
STYLE={
 'family_only':('Family-local',GREY,'s'),
 'family_cost_gate':('Gate-only',ORANGE,'x'),
 'sparse':('Sparse-JV',PURPLE,'^'),
 'components':('Component',GREY,'s'),
 'static':('Static-Reduction','#8e914c','v'),
 'replay':('Dynamic replay',BLUE,'D'),
 'indexed':('CARA',TEAL,'o'),
 'no_seed_index':('Without seed index',BLUE,'x')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.0,'axes.labelsize':7.1,
 'axes.titlesize':8.0,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.4,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'pdf.fonttype':42,'ps.fonttype':42})


def read(path):
    return json.loads(Path(path).read_text())


def complete(path):
    data=read(Path(path)/'summary.json')
    if not data.get('complete'):
        raise ValueError('Only complete studies can produce paper figures: '+str(path))
    return data


def save(fig,output,name):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/(name+'.pdf'))
    fig.savefig(output/(name+'.png'),dpi=240)
    plt.close(fig)


def grid(ax):
    ax.set_axisbelow(True); ax.grid(axis='y',color='#dfe5e8',linewidth=.45)
    ax.tick_params(length=2.5,pad=2)


def execution(packed,amortization,selection,output):
    summary=complete(packed); amort=complete(amortization); selected=complete(selection)
    rows=read(Path(packed)/'records.json')
    expected={(s,k,w) for s in SOURCES for k in ('packed','repaired') for w in (8,12,16)}
    if len(rows)!=24 or {(r['source'],r['kind'],r['width']) for r in rows}!=expected or not summary['all_exact']:
        raise ValueError('The complete four-source/two-seed/three-width study is required')
    if not all(r['all_family_pools_exact'] and r['all_materialized_profiles_exact'] and r['all_exclusion_leaves_checked'] for r in rows):
        raise ValueError('Incomplete query checks')
    fig,axes=plt.subplots(2,3,figsize=(7.16,3.94))
    fig.subplots_adjust(left=.105,right=.995,bottom=.160,top=.837,wspace=.36,hspace=.64)
    plans=('family_cost_gate','components','sparse','static','replay','indexed')
    offsets=np.linspace(-.22,.22,len(plans))
    for col,width in enumerate((8,12,16)):
        ax=axes[0,col]; grid(ax)
        cells=[next(r for r in rows if (r['source'],r['kind'],r['width'])==(source,kind,width)) for source in SOURCES for kind in ('packed','repaired')]
        values=[]
        for offset,plan in zip(offsets,plans):
            middle=[]; low=[]; high=[]
            for r in cells:
                ref=r['medians']['family_only']['cpu_seconds']
                trials=np.array([t['plans'][plan]['cpu_seconds']/ref for t in r['trials']])
                m=r['medians'][plan]['cpu_seconds']/ref
                middle.append(m);low.append(max(0,m-trials.min()));high.append(max(0,trials.max()-m)); values.extend(trials)
            title,color,marker=STYLE[plan]
            ax.errorbar(np.arange(8)+offset,middle,yerr=[low,high],marker=marker,markersize=3.1 if plan=='indexed' else 2.7,
                color=color,linestyle='none',elinewidth=.55,capsize=1.0,markeredgewidth=.7)
        ax.axhline(1,color=GREY,linewidth=.65,linestyle=(0,(3,3)))
        ax.set_yscale('log'); ax.yaxis.set_major_locator(FixedLocator([.05,.1,.2,.3,.5,1,2,3,5,10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}')); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(min(values+[1])*.80,max(values+[1])*1.20)
        # Explicit seed labels avoid implying eight independent data sources.
        ax.set_xticks(range(8),[f'{short}\n{kind}' for short in SHORT for kind in ('P','R')])
        ax.set_xlim(-.55,7.55); ax.set_title(f'({chr(97+col)}) Width {width}',loc='left',pad=5)
        if col==0:ax.set_ylabel('CPU / family-local')
    handles=[Line2D([],[],color=STYLE[p][1],marker=STYLE[p][2],linestyle='none',markersize=4,label=STYLE[p][0]) for p in plans]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.535,.995),ncol=3,frameon=False,
               columnspacing=1.4,handletextpad=.4)
    ax=axes[1,0];grid(ax)
    variants=('family_cost_gate','recomputed_bounds','full_uniform','no_compiled_bounds','no_seed_index','no_bound_views','no_subset_views','no_metric_cascade')
    names=('Gate-only','Recomputed','Full-first','−Radii','−Seed','−Summary','−Subset','−Cascade')
    for offset,width,color in ((-.22,8,BLUE),(0,12,TEAL),(.22,16,ORANGE)):
        subset=[r for r in rows if r['width']==width]
        denominator=sum(r['medians']['indexed']['cpu_seconds'] for r in subset)
        ratio=[sum(r['medians'][v]['cpu_seconds'] for r in subset)/denominator for v in variants]
        ax.barh(np.arange(len(variants))+offset,ratio,height=.21,color=color,label=f'w={width}')
    ax.axvline(1,color=GREY,linestyle=(0,(3,3)),linewidth=.65)
    ax.set_yticks(range(len(variants)),names);ax.invert_yaxis(); ax.set_xlabel('Total CPU / CARA'); ax.set_title('(d) Component ablation',loc='left',pad=5)
    ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.53),columnspacing=.7,handlelength=1.)
    ax=axes[1,1];grid(ax)
    if amort.get('sources')!=1 or not amort.get('all_exact'):
        raise ValueError('Expected the complete Azure seed-amortization study')
    prefixes=amort['rows'][0]['prefixes']
    for plan in ('family_only','family_cost_gate','sparse','no_seed_index','indexed'):
        name,color,marker=STYLE[plan]
        x=[r['families'] for r in prefixes]; y=[1000*r['medians'][plan]['cpu_seconds']/r['families'] for r in prefixes]
        ax.plot(x,y,color=color,marker=marker,markersize=3,linewidth=.85,linestyle='--' if plan=='no_seed_index' else '-')
    ax.set_xscale('log',base=2);ax.set_xticks([1,4,16,64,128],['1','4','16','64','128'])
    ax.set_xlabel('Families sharing one seed');ax.set_ylabel('CPU per family (ms)');ax.set_title('(e) Compilation amortization',loc='left',pad=5)
    # Only this dashed ablation is absent from the shared top legend.
    ax.text(.98,.95,'Grey: family-local\nDashed: without seed index',ha='right',va='top',transform=ax.transAxes,fontsize=5.9,color=BLUE)
    ax=axes[1,2];grid(ax)
    if not selected.get('all_profiles_and_exclusions_checked'):
        raise ValueError('Missing independent selection-budget checks')
    for plan,label,color,marker in (('no_score','All eligible',GREY,'s'),('compiled','CARA',TEAL,'o')):
        x=[r['top_p'] for r in selected['rows']]; y=[r['medians'][plan]['profiled_masks'] for r in selected['rows']]
        ax.plot(x,y,color=color,marker=marker,markersize=3.5,linewidth=1.,label=label)
    ax.set_xscale('log',base=2);ax.set_xticks([4,8,16,32,64],['4','8','16','32','64']);ax.set_xlabel('Requested result size P')
    ax.set_ylabel('Exact profiles');ax.set_title('(f) Result size',loc='left',pad=5);ax.legend(frameon=False,loc='center left')
    save(fig,output,'execution')


def source_quality(source,output):
    data=complete(source)
    if data['origin_contexts']!=13 or data['budget_stream_contexts']!=78 or not data['all_source_vms'] or not data['all_target_access_guards']:
        raise ValueError('Expected all 13 source-days, two budgets and three streams')
    rows={(r['method'],r['budget_cpu_seconds']):r['equal_source'] for r in data['policies']}
    budgets=sorted({budget for method,budget in rows})
    methods=(('Swap','Swap',ORANGE,'v'),('Joint-SA','Joint-SA',BLUE,'D'),('Family-only','Family-local',GREY,'s'),('CARA','CARA',TEAL,'o'))
    fig,axes=plt.subplots(1,4,figsize=(7.16,2.13))
    fig.subplots_adjust(left=.073,right=.995,bottom=.245,top=.735,wspace=.48)
    for ax in axes:
        grid(ax)
        ax.set_xlabel('CPU budget (s)'); ax.set_xscale('log',base=2)
        ax.set_xticks(budgets,[f'{b:g}' for b in budgets]); ax.xaxis.set_minor_formatter(NullFormatter())
    for method,label,color,marker in methods:
        values=[rows[method,b] for b in budgets]
        if method in ('Family-only','CARA'):
            series=(
                [r['complete_families']/r['total_decision_cpu_seconds'] for r in values],
                [r['evaluated_candidates'] for r in values],
                [r['overrun_p95'] for r in values])
            for ax,y in zip(axes[:3],series):
                ax.plot(budgets,y,color=color,marker=marker,markersize=4.,linewidth=1.,label=label)
        reduction=[100*(rows['FFD',b]['VR']-rows[method,b]['VR'])/max(abs(rows['FFD',b]['VR']),1e-8) for b in budgets]
        axes[3].plot(budgets,reduction,color=color,marker=marker,markersize=4.,linewidth=1.,label=label)
    for ax,title in zip(axes,('(a) Completed work','(b) Profile acquisition','(c) Budget overrun','(d) Released quality')):
        ax.set_title(title,loc='left',pad=5)
    axes[0].set_ylabel('Families / complete CPU s')
    axes[1].set_ylabel('Scored candidates'); axes[1].set_yscale('log')
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[2].set_ylabel('Mean source p95 overrun (s)')
    axes[3].set_ylabel('VR reduction from FFD (%)')
    fig.legend(*axes[3].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.99),ncol=4,frameon=False,columnspacing=1.7)
    save(fig,output,'quality')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packed',type=Path,required=True)
    parser.add_argument('--amortization',type=Path,required=True)
    parser.add_argument('--selection-budget',type=Path,required=True)
    parser.add_argument('--source-repair',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('figures'))
    args=parser.parse_args()
    execution(args.packed,args.amortization,args.selection_budget,args.output)
    source_quality(args.source_repair,args.output)
