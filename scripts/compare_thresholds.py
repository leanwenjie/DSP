"""Validation-only screening analysis. Existing model artifacts are not changed."""
import csv
import json
import math
import os
import numpy as np
from sklearn.metrics import precision_recall_curve
from analyse_similarity import read_predictions, summarise
from prepare_data import ROOT, digest


def screening_threshold(scores, labels, target=.8):
    scores=np.asarray(scores,dtype=np.float64)
    labels=np.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Aligned finite one-dimensional scores required')
    if not 0 < target <= 1 or not np.isin(labels,[0,1]).all() or not np.any(labels==1):
        raise ValueError('Binary labels with positives and target in (0,1] required')
    positives=np.sort(scores[labels==1])
    # Largest inclusive threshold meeting the target: fewest flags, ties intact.
    return float(positives[-math.ceil(target*len(positives))])


def metrics(y,p):
    r=summarise(y,p)
    flags=r['tp']+r['fp']
    r.update(precision=r['tp']/flags if flags else None, flagged=flags, flagged_fraction=flags/len(y))
    return r


def main():
    out=ROOT/'reports/thresholds';out.mkdir(parents=True,exist_ok=True)
    base=ROOT/'artifacts/baseline/val_predictions.csv'
    rows=read_predictions(base)
    y=np.array([int(r['label']) for r in rows])
    ids=np.array([int(r['caption_id']) for r in rows])
    images=np.array([int(r['image_id']) for r in rows])
    cosine=np.array([float(r['cosine_similarity']) for r in rows])
    annotation=json.loads((ROOT/'data/newsclippings/val.json').read_text())['annotations']
    np.testing.assert_array_equal(ids,[r['id'] for r in annotation])
    np.testing.assert_array_equal(images,[r['image_id'] for r in annotation])
    np.testing.assert_array_equal(y,[int(r['falsified']) for r in annotation])
    models={'Similarity':(-cosine,np.array([int(r['prediction']) for r in rows]))}
    paths=[base]
    for name,path in [('Logistic regression',ROOT/'artifacts/logistic/validation_predictions.npz'),('MLP seed 42',ROOT/'artifacts/mlp/validation_seed42.npz')]:
        with np.load(path,allow_pickle=False) as d:
            np.testing.assert_array_equal(d['caption_id'],ids)
            np.testing.assert_array_equal(d['labels'],y)
            if 'image_id' in d:np.testing.assert_array_equal(d['image_id'],images)
            elif len(set(zip(ids,y))) != len(y):raise ValueError('Ambiguous caption/label IDs')
            score=d['decision' if name=='Logistic regression' else 'probability'].astype(np.float64)
            models[name]=(score,d['predictions'].copy())
        paths.append(path)
    bin_path=ROOT/'reports/similarity_analysis/metrics.json'
    cutoff=json.loads(bin_path.read_text())['bin_cutoffs'][-1]
    high=cosine>=cutoff
    results=[]
    for name,(scores,current) in models.items():
        threshold=screening_threshold(scores,y)
        for mode,prediction in [('macro_f1',current),('screening_80',scores>=threshold)]:
            results.append(dict(model=name,mode=mode,screening_threshold=threshold if mode=='screening_80' else None,
                                overall=metrics(y,prediction),high_similarity=metrics(y[high],prediction[high])))
    report=dict(split='val',test_evaluated=False,recall_target=.8,
                selection_rule='Highest inclusive score threshold reaching >=80% recall; minimises flags, not necessarily maximises precision.',
                score_definitions={'Similarity':'negative cosine similarity','Logistic regression':'decision function','MLP seed 42':'uncalibrated sigmoid score'},
                high_similarity_cutoff=cutoff,high_similarity_source='training 80th percentile; unchanged',
                warning='Thresholds selected and measured on same validation set. 80% recall is not guaranteed on unseen inputs. No production threshold or original model artifact changed.',
                provenance_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths+[bin_path]},results=results)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for name,(scores,current) in models.items():
        precision,recall,_=precision_recall_curve(y,scores)
        axes[0].plot(recall,precision,label=name)
        s=next(r for r in results if r['model']==name and r['mode']=='screening_80')['overall']
        axes[0].scatter(s['misleading_recall'],s['precision'])
    axes[0].axvline(.8,color='grey',linestyle='--')
    axes[0].set(xlabel='Misleading-pair recall',ylabel='Precision',title='Validation precision–recall trade-off',xlim=(0,1),ylim=(0,1))
    axes[0].legend(fontsize=8)
    x=np.arange(3);names=list(models)
    for offset,mode,label in [(-.18,'macro_f1','Macro-F1 operating point'),(.18,'screening_80','80% recall screening point')]:
        counts=[next(r for r in results if r['model']==name and r['mode']==mode)['overall']['fp'] for name in names]
        bars=axes[1].bar(x+offset,counts,width=.36,label=label)
        axes[1].bar_label(bars,padding=3)
    axes[1].set(xticks=x,xticklabels=names,ylim=(0,2200),ylabel='Original pairs incorrectly flagged',title='Cost in false alarms (3,512 original pairs)')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'screening_tradeoff.png',dpi=180);plt.close(fig)
    for r in results:
        o=r['overall'];h=r['high_similarity']
        print(r['model'],r['mode'],{k:round(o[k],4) for k in ['precision','misleading_recall','false_positive_rate','macro_f1','fp','tp','flagged']},'high recall',round(h['misleading_recall'],4))

if __name__=='__main__':main()
