"""Plot observed policy style against each variant's own greedy baseline."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--riichi',required=True);p.add_argument('--sichuan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    ri=json.loads(Path(a.riichi).read_text());sc=json.loads(Path(a.sichuan).read_text())
    r=ri['matchups']['greedy'];s=sc['summary'];fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    colors=['#236b57','#a0a6a3'];width=.34
    for ax,title,key,mult in [(axes[0],'Open-hand rate','open',100),(axes[1],'Kongs per 100 hands','kongs',100)]:
        for j,label in enumerate(['rl','greedy']):
            sv=s[label]['open_rate' if key=='open' else 'kongs_per_hand']*mult
            rv=r[label]['style'][key]*mult
            x=np.arange(2)+(j-.5)*width
            ax.bar(x,[sv,rv],width,label='Trained' if label=='rl' else 'Greedy',color=colors[j])
            for xx,v in zip(x,[sv,rv]):ax.text(xx,v+.8,f'{v:.1f}',ha='center',fontsize=10)
        ax.set_xticks([0,1],['Sichuan blood battle','Riichi East']);ax.set_title(title,loc='left',fontweight='bold');ax.set_ylim(0,100 if key=='open' else max(35,1.25*max(s['rl']['kongs_per_hand'],s['greedy']['kongs_per_hand'])*100));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    keys=['riichi','safe_choice','shanten_backoff'];labels=['Riichi\ndeclared','Genbutsu\nwhen available','Shanten sacrificed\nfor genbutsu']
    for j,policy in enumerate(['rl','greedy']):
        values=[r[policy]['style'][k] for k in keys]
        x=np.arange(3)+(j-.5)*width
        axes[2].bar(x,[100*v if v is not None else 0 for v in values],width,color=colors[j])
        for xx,v in zip(x,values):axes[2].text(xx,100*v+1 if v is not None else 1,'N/A' if v is None else f'{100*v:.1f}',ha='center',fontsize=9)
    axes[2].set_xticks(range(3),labels);axes[2].set_ylim(0,100);axes[2].set_title('Riichi: commitment and defense (%)',loc='left',fontweight='bold');axes[2].spines[['top','right']].set_visible(False);axes[2].grid(axis='y',alpha=.15);axes[2].set_axisbelow(True)
    axes[0].legend(frameon=False,loc='upper right');fig.suptitle('Observed playing style — each policy versus three greedy opponents',fontsize=15,fontweight='bold')
    fig.supxlabel(f"Sichuan: {s['rl']['games']} hands/policy. Riichi RL: {r['rl']['games']} East matches / {r['rl']['hands']} hands.\nDefense is conditional on a common safe discard, at least two discard kinds, and not already being in riichi. Point estimates; see report for intervals.",fontsize=9)
    destination=Path(a.output);destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=170,facecolor='white');plt.close(fig)
    print(destination)

if __name__=='__main__':main()
