"""Plot fixed-budget hanchan results with paired-wall bootstrap intervals."""
import argparse,json,os,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'placement-matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'score':'#909a96','rank':'#236b57'}
LABELS={'score':'Points reward','rank':'Placement reward'}


def get(summary,key):
    direct={'rank':'mean_rank','first_place':'first_place_rate','last_place':'last_place_rate'}
    value=summary[direct[key]] if key in direct else summary['style'].get(key)
    return value,summary['ci95'].get(key)


def bars(ax,groups,keys,labels,title):
    width=.34;x=np.arange(len(keys))
    for i,mode in enumerate(('score','rank')):
        for j,key in enumerate(keys):
            value,ci=get(groups[mode],key);position=x[j]+(i-.5)*width
            if value is None:
                ax.text(position,1,'N/A',ha='center',fontsize=8);continue
            value*=100;ax.bar(position,value,width,color=COLORS[mode],label=LABELS[mode] if j==0 else None)
            if ci is not None:
                ax.errorbar(position,value,yerr=np.array([[max(0,value-100*ci[0])],[max(0,100*ci[1]-value)]]),color='#1f2925',capsize=3,linewidth=1)
            ax.text(position,value+1.5,f'{value:.1f}',ha='center',fontsize=8)
    ax.set_xticks(x,labels);ax.set_title(title,loc='left',fontweight='bold',fontsize=11)
    ax.set_ylim(0,min(105,max(15,ax.get_ylim()[1]*1.12)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=json.loads(Path(a.evaluation).read_text());groups=data['pooled_final_models']
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,mode in enumerate(('score','rank')):
        value,ci=get(groups[mode],'rank')
        axes[0,0].errorbar(value,1-i,xerr=np.array([[max(0,value-ci[0])],[max(0,ci[1]-value)]]),fmt='o',color=COLORS[mode],capsize=5,markersize=9)
        axes[0,0].text(value,1-i+.15,f'{value:.3f}',ha='center',fontsize=10)
    axes[0,0].set_yticks([1,0],[LABELS['score'],LABELS['rank']]);axes[0,0].set_ylim(-.5,1.5)
    axes[0,0].axvline(2.5,color='#b8bdbb',linestyle='--',linewidth=1)
    axes[0,0].set_title('Mean final rank (lower is better)',loc='left',fontweight='bold',fontsize=11)
    bars(axes[0,1],groups,['first_place','last_place'],['First place','Fourth place'],'Final placement (%)')
    bars(axes[0,2],groups,['open','riichi'],['Open hand','Riichi declared'],'Hand style (%)')
    bars(axes[1,0],groups,['late_lead_riichi_choice','late_last_riichi_choice'],['Late / strict leader','Late / strict last'],'Riichi chosen when legal (%)')
    bars(axes[1,1],groups,['early_safe_choice','late_lead_safe_choice','late_last_safe_choice'],['Early','Late leader','Late last'],'Genbutsu chosen when available (%)')
    bars(axes[1,2],groups,['early_shanten_backoff','late_lead_shanten_backoff','late_last_shanten_backoff'],['Early','Late leader','Late last'],'Shanten sacrificed for genbutsu (%)')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=9)
    axes[0,1].legend(frameon=False,fontsize=9,loc='upper right')
    fig.suptitle('Points versus final placement — fixed-budget South matches',fontsize=17,fontweight='bold')
    games=groups['rank']['games'];seeds=len(data['experiment']['seeds']);step=data['experiment']['runs'][0]['target_steps']
    fig.supxlabel(f'{seeds} fitted training seeds / reward; {step:,} decisions / model. {games:,} test matches / reward. Bars and dots: 95% test-wall block bootstrap.\nLate = South 3 onward. Defensive choices require a common genbutsu, at least two discard kinds, and not already being in riichi. Conditional rates use encountered states.',fontsize=9)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170,facecolor='white');plt.close(fig);print(output)

if __name__=='__main__':main()
