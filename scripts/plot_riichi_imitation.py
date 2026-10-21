"""Plot held-out greedy fidelity, separately from match-strength estimates."""
import argparse
import json
from pathlib import Path


def plot(root,baseline_audit,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    root=Path(root);baseline=json.loads(Path(baseline_audit).read_text())['models']['published']['audit']['all']
    figure,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    colors={'v1':'#dd8a32','v2':'#3565a8'}
    for version in ['v1','v2']:
        initial=json.loads((root/f'bc_{version}_initial/imitation.json').read_text())
        refit=json.loads((root/f'bc_{version}/imitation.json').read_text())
        history=initial['history']+refit['history'];epochs=range(1,len(history)+1)
        for axis,key in zip(axes,['teacher_set_accuracy','teacher_set_probability']):
            axis.plot(epochs,[x['validation']['all'][key] for x in history],label=version.upper(),color=colors[version],linewidth=1.8)
            axis.axvline(len(initial['history'])+.5,color='#777777',linewidth=1,linestyle=':')
    for axis,key,title in zip(axes,['teacher_set_accuracy','teacher_set_probability'],
            ['Greedy-optimal action agreement','Policy probability on greedy-optimal moves']):
        axis.axhline(baseline[key],label='Published V1 (fixed)',color='#777777',linestyle='--',linewidth=1.2)
        axis.set_title(title,fontsize=11);axis.set_xlabel('Epoch (20 BC + 8 learner-state refits)')
        axis.set_ylim(.5 if key=='teacher_set_probability' else .65,1.005)
        axis.yaxis.set_major_formatter(PercentFormatter(1));axis.grid(alpha=.18)
        axis.legend(fontsize=8,loc='lower right')
    figure.suptitle('4,096 held-out decisions: behavior fidelity, not playing strength',fontsize=12)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);figure.savefig(output,dpi=180);plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--baseline-audit',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();plot(args.root,args.baseline_audit,args.output)

if __name__=='__main__':main()
