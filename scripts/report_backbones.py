"""Consolidate frozen backbone results without fitting or selecting models."""
import json
import os
from prepare_data import ROOT


def main():
    out=ROOT/'reports/backbone_evaluation'
    state=json.loads((ROOT/'artifacts/backbone_evaluation/progress.json').read_text())
    if not state.get('complete'):raise ValueError('Evaluation incomplete')
    rows=[]
    for slug in ['ViT-B-32','ViT-B-16','ViT-L-14']:
        n=json.loads((out/f'{slug}_newsclippings.json').read_text())['results']['test']['interaction_lr']
        v=json.loads((out/f'{slug}_verite.json').read_text())['results']
        rows.append((slug,n,v['primary_true_vs_ooc']['interaction_lr'],v['secondary_all']['interaction_lr']))
    lines=['# Frozen backbone comparison — completed','','All three candidates use freshly extracted standard-CLIP features and validation-selected interaction logistic regression. This is follow-up evaluation after prior benchmark inspection; no test-based tuning was performed.','','| Backbone | NewsCLIPpings accuracy | NewsCLIPpings macro-F1 | VERITE primary macro-F1 | VERITE OOC recall | VERITE all-misleading macro-F1 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,n,v,a in rows:lines.append(f"| {name} | {n['accuracy']:.2%} | {n['macro_f1']:.2%} | {v['macro_f1']:.2%} | {v['misleading_recall']:.2%} | {a['macro_f1']:.2%} |")
    lines+=['','![Three-backbone comparison](comparison.png)','','L/14 had the highest validation macro-F1 before test scoring. It also has the highest reported point estimates here, but the broader VERITE task remains weak. Test scores do not establish factual verification of arbitrary uploads. All-misleading VERITE is imbalanced; accuracy alone is insufficient.','','## Paired differences relative to fresh B/32','','| Task | Backbone | 95% interval of macro-F1 change (percentage points) |','| --- | --- | --- |']
    intervals=json.loads((out/'paired_macro_f1_intervals.json').read_text())
    for task,models in intervals.items():
        for name,(lo,hi) in models.items():lines.append(f'| {task} | {name} | {100*lo:+.2f} to {100*hi:+.2f} |')
    lines+=['','Intervals use 2,000 paired grouped-bootstrap draws with seed 42. They are exploratory and not multiplicity-adjusted. L/14 has a positive interval versus fresh B/32 on NewsCLIPpings and primary VERITE, but not on all-misleading VERITE. No interval here compares L/14 directly against B/16.','','Full task JSON files include confusion matrices, ranking metrics, category counts and caption truncation counts. VERITE truncation matches the recorded evaluation policy; the app rejects overly long captions instead. Protocol and frozen manifest remain unchanged. Raw predictions are stored in ignored artifacts/backbone_evaluation.','','These results address backbone effectiveness and external generalisation. They do not replace the separate representation-ablation evidence or constitute reproduction of the main reference paper. App integration uses the validation-leading L/14 candidate; its costs and weaker secondary result must be disclosed.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    x=np.arange(3)
    for j,title in enumerate(['NewsCLIPpings test','VERITE true vs OOC','VERITE all misleading']):
        values=[r[j+1]['macro_f1']*100 for r in rows]
        bars=ax.bar(x+(j-1)*.24,values,.24,label=title);ax.bar_label(bars,fmt='%.2f',fontsize=8)
    ax.set(xticks=x,xticklabels=[r[0] for r in rows],ylim=(0,100),ylabel='Macro-F1 (%)',title='Frozen CLIP backbones with interaction logistic regression')
    ax.legend();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
