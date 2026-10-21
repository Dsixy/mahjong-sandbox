"""Summarize independent match tests, teacher fidelity and training curves."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def paired_candidate(rows,candidate,baseline,seed):
    import numpy as np
    from mahjong_sandbox.rl.riichi_evaluate import _totals,METRICS
    walls=sorted({row['seed'] for row in rows})
    samples=np.random.default_rng(seed).integers(0,len(walls),(2000,len(walls)))
    grouped=[[_totals([row for row in rows if row['seed']==wall and row['policy']==label])
        for wall in walls] for label in [candidate,baseline]]
    result={}
    for metric in ['rank','win','deal_in','net_score','last_place','first_place','rank_utility']:
        num,den=METRICS[metric]
        an,ad=np.array([x[num] for x in grouped[0]]),np.array([x[den] for x in grouped[0]])
        bn,bd=np.array([x[num] for x in grouped[1]]),np.array([x[den] for x in grouped[1]])
        values=an[samples].sum(1)/ad[samples].sum(1)-bn[samples].sum(1)/bd[samples].sum(1)
        result[metric]={'candidate_minus_baseline':float(an.sum()/ad.sum()-bn.sum()/bd.sum()),
            'ci95':np.quantile(values,[.025,.975]).tolist()}
    return result


def analyze(root,output):
    import numpy as np
    from mahjong_sandbox.rl.riichi_evaluate import summarize,_totals,METRICS
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    evaluation=json.loads((root/'evaluation.json').read_text());plan=evaluation['experiment']
    metrics=('mean_rank','last_place_rate','first_place_rate','win_rate','deal_in_rate','mean_net_score')
    summaries={label:{key:row[key] for key in ('games','hands',*metrics,'ci95','style')}
        for label,row in evaluation['matchups']['greedy'].items()}
    selected=evaluation['selected_label'];rows=evaluation['game_records']
    v2_labels=['rl' if label==selected else label for label in ['v2_'+str(s) for s in plan['train_seeds']]]
    pooled_rows=[x for x in rows if x['policy'] in v2_labels]
    pooled=summarize(pooled_rows,seed=plan['test_seed'])
    v1_label='v1_'+str(plan['train_seeds'][0])
    baseline_rows=[x for x in rows if x['policy']==v1_label]
    seeds=sorted({x['seed'] for x in rows})
    samples=np.random.default_rng(plan['test_seed']).integers(0,len(seeds),(2000,len(seeds)))
    grouped=[[_totals([x for x in group if x['seed']==seed]) for seed in seeds] for group in [pooled_rows,baseline_rows]]
    paired={}
    for metric in ('rank','win','deal_in','net_score','last_place'):
        num,den=METRICS[metric]
        an,ad=np.array([x[num] for x in grouped[0]]),np.array([x[den] for x in grouped[0]])
        bn,bd=np.array([x[num] for x in grouped[1]]),np.array([x[den] for x in grouped[1]])
        delta=an[samples].sum(1)/ad[samples].sum(1)-bn[samples].sum(1)/bd[samples].sum(1)
        paired[metric]={'pooled_v2_minus_v1_control':float(an.sum()/ad.sum()-bn.sum()/bd.sum()),
            'ci95':np.quantile(delta,[.025,.975]).tolist()}
    imitation={}
    for version in ['v1','v2']:
        for suffix in ['_initial','']:
            folder=root/('bc_'+version+suffix)
            report=json.loads((folder/'imitation.json').read_text())
            acceptance=json.loads((folder/'acceptance.json').read_text())
            metadata=json.loads((folder/'warmstart.json').read_text())
            imitation[folder.name]={'model_sha256':metadata['model_sha256'],'selected_epoch':metadata['training']['imitation']['selected_epoch'],
                'epochs':report['epochs'],'training_samples_per_epoch':report['training_samples_per_epoch'],
                'elapsed_seconds':report['elapsed_seconds'],'acceptance':acceptance}
    curves={}
    for label in [v1_label,*['v2_'+str(s) for s in plan['train_seeds']]]:
        events=[]
        for line in (root/label/'metrics.jsonl').read_text().splitlines():
            try:events.append(json.loads(line))
            except ValueError:pass
        curves[label]={'validation':[{key:x.get(key) for key in ('timesteps',*metrics)} for x in events if x.get('event')=='validation'],
            'updates':[{key:x.get(key) for key in ('timesteps','elapsed_seconds','train/rank','train/net_score','train/approx_kl','train/value_loss','train/explained_variance','train/entropy_loss','anchor/set_cross_entropy')} for x in events if x.get('event')=='ppo_update'],
            'report':json.loads((root/label/'training.json').read_text())}
    selection_path=root/'validation-selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else None
    if selection:
        for label,details in selection['runs'].items():
            curves[label]['live_validation']=curves[label]['validation']
            curves[label]['validation']=[{'timesteps':entry['step'],**{k:entry['statistics'][k] for k in metrics},
                'style':entry['statistics']['style']} for entry in details['candidates']]
    cold_path=root/'coldstart-match-validation.json'
    cold=json.loads(cold_path.read_text()) if cold_path.exists() else None
    if cold:
        for version,entry in cold['models'].items():
            if entry['model_sha256']!=imitation['bc_'+version]['model_sha256']:
                raise ValueError('cold match-validation model differs from the teacher-accepted warmstart')
    artifact_path=root/'artifact-model.json';artifact=json.loads(artifact_path.read_text()) if artifact_path.exists() else None
    if artifact:
        label='rl' if artifact['label']==selected else artifact['label']
        warmstart='v1_bc' if artifact['label'].startswith('v1_') else 'v2_bc'
        artifact={**artifact,'evaluation_label':label,'paired_comparisons':{
            baseline:paired_candidate(rows,label,baseline,plan['test_seed']) for baseline in ['greedy','v1_published',warmstart]}}
    summary={'format':'riichi_curriculum_analysis_v1','experiment':plan,'selected_label':selected,
        'evaluation_sha256':hashlib.sha256((root/'evaluation.json').read_bytes()).hexdigest(),
        'games':evaluation['games'],'hands':evaluation['hands'],'models':evaluation['models'],
        'matchups':summaries,'paired_comparisons':evaluation['paired_comparisons']['greedy'],
        'pooled_v2':pooled,'pooled_v2_vs_matched_v1':paired,'imitation':imitation,'training':curves,
        'release_gate':evaluation['release_gate'],'global_checkpoint_selection':selection,'coldstart_match_validation':cold,
        'validation_best_artifact':artifact,
        'interval_note':'Test-wall block bootstrap conditional on these fitted models; two V2 training seeds do not estimate uncertainty over arbitrary retraining.',
        'comparison_limits':'Published V1 used a different reward and training budget. Matched V1 control shares new teacher data, DAgger budgets, rank reward, self-play and anchor settings; V2 also fixes the declaration marker. Each run selects by validation before testing.'}
    (output/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    counts=[{k:v for k,v in row.items() if k!='hand_records'} for row in rows]
    with (output/'test-counts.json.gz').open('wb') as file:
        with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as archive:archive.write(json.dumps(counts,separators=(',',':')).encode())
    plot(summary,output/'comparison.png')
    plot_training(summary,output/'training.png')
    plot_style_training(summary,output/'training-style.png')
    return summary


def plot(summary,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=['greedy','v1_published','v1_bc','v1_'+str(summary['experiment']['train_seeds'][0]),'v2_bc','rl']
    names=['Greedy','Published V1','V1 BC','V1 RL control','V2 BC','V2 RL selected']
    figure,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for axis,(metric,title,interval) in zip(axes,[('mean_rank','Mean placement (lower is better)','rank'),
            ('last_place_rate','Fourth-place rate','last_place'),('mean_net_score','Net points per South match','net_score')]):
        values=np.array([summary['matchups'][key][metric] for key in labels])
        bounds=np.array([summary['matchups'][key]['ci95'][interval] for key in labels])
        axis.errorbar(values,np.arange(len(labels)),xerr=np.stack((values-bounds[:,0],bounds[:,1]-values)),
            fmt='o',capsize=3,color='#3b5b92')
        axis.set_yticks(range(len(labels)),names);axis.invert_yaxis();axis.set_title(title,fontsize=11)
        axis.grid(axis='x',alpha=.2)
        if interval=='last_place':axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        axis.axvline(2.5 if interval=='rank' else .25 if interval=='last_place' else 0,color='#999999',linestyle=':',linewidth=1)
    figure.suptitle('Held-out wall blocks, four seats; 95% wall-block bootstrap',fontsize=13)
    figure.savefig(path,dpi=180);plt.close(figure)


def plot_training(summary,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    colors=['#dd8a32','#3565a8','#32917d']
    for (label,run),color in zip(summary['training'].items(),colors):
        name='Matched V1' if label.startswith('v1_') else 'V2 seed '+label[-2:]
        validation=run['validation'];updates=run['updates']
        axes[0].plot([x['timesteps']/1000 for x in validation],[x['mean_rank'] for x in validation],
            marker='o',label=name,color=color)
        for axis,key in zip(axes[1:],['train/explained_variance','anchor/set_cross_entropy']):
            axis.plot([x['timesteps']/1000 for x in updates],[x.get(key) for x in updates],
                label=name,color=color,alpha=.85)
    axes[0].axhline(2.5,color='#999999',linestyle=':',linewidth=1)
    for axis,title in zip(axes,['Mean placement on validation walls','Value explained variance (training)',
            'Teacher anchor loss (training batches)']):
        axis.set_title(title,fontsize=10);axis.set_xlabel('Cumulative decisions (thousands)')
        axis.grid(alpha=.18);axis.legend(fontsize=8)
    figure.suptitle('Fixed training budget; validation curves do not establish test strength',fontsize=12)
    figure.savefig(path,dpi=180);plt.close(figure)


def plot_style_training(summary,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    figure,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for (label,run),color in zip(summary['training'].items(),['#dd8a32','#3565a8','#32917d']):
        name='Matched V1' if label.startswith('v1_') else 'V2 seed '+label[-2:]
        version=label.split('_')[0];cold=summary.get('coldstart_match_validation')
        points=([{'timesteps':0,**cold['models'][version]['statistics']}] if cold else [])+run['validation']
        x=[point['timesteps']/1000 for point in points]
        values=[[point.get('style',{}).get('win_income') for point in points],
            [point['deal_in_rate'] for point in points],[point['mean_rank'] for point in points]]
        for axis,y in zip(axes,values):axis.plot(x,y,marker='o',color=color,label=name)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[2].axhline(2.5,color='#999999',linestyle=':',linewidth=1)
    for axis,title in zip(axes,['Mean income per win (includes deposits)','Deal-in rate per hand','Mean placement (lower is better)']):
        axis.set_title(title,fontsize=10);axis.set_xlabel('Cumulative RL decisions (thousands)')
        axis.grid(alpha=.18);axis.legend(fontsize=8)
    figure.suptitle('Validation point estimates: 128 South matches per checkpoint; step 0 is BC',fontsize=12)
    figure.savefig(path,dpi=180);plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=analyze(args.root,args.output)
    print(json.dumps({k:result[k] for k in ('games','hands','selected_label','release_gate')},indent=2))

if __name__=='__main__':main()
