"""Descriptive validation error analysis using training-defined similarity bins.

No models are refitted and no thresholds are selected. Test sets are not read.
"""
import csv
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from prepare_data import ROOT, digest


def read_predictions(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def summarise(y, prediction):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    recall = float(tp / (tp + fn)) if tp + fn else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    return dict(n=len(y), original=int(tn+fp), misleading=int(tp+fn),
                accuracy=float(np.mean(y == prediction)),
                macro_f1=float(f1_score(y, prediction, labels=[0,1], average='macro', zero_division=0)),
                misleading_recall=recall, false_positive_rate=1-specificity if specificity is not None else None,
                balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def main():
    out = ROOT / 'reports/similarity_analysis'
    out.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / 'artifacts/baseline/train_predictions.csv'
    val_path = ROOT / 'artifacts/baseline/val_predictions.csv'
    train = read_predictions(train_path)
    rows = read_predictions(val_path)
    scores = np.array([float(r['cosine_similarity']) for r in rows])
    labels = np.array([int(r['label']) for r in rows])
    caption_ids = np.array([int(r['caption_id']) for r in rows])
    image_ids = np.array([int(r['image_id']) for r in rows])
    # Check alignment against raw validation annotations, not just repeated caption IDs.
    annotations = json.loads((ROOT / 'data/newsclippings/val.json').read_text())['annotations']
    np.testing.assert_array_equal(caption_ids, [r['id'] for r in annotations])
    np.testing.assert_array_equal(image_ids, [r['image_id'] for r in annotations])
    np.testing.assert_array_equal(labels, [int(r['falsified']) for r in annotations])
    predictions = {'Similarity': np.array([int(r['prediction']) for r in rows])}
    lr_path = ROOT / 'artifacts/logistic/validation_predictions.npz'
    mlp_path = ROOT / 'artifacts/mlp/validation_seed42.npz'
    with np.load(lr_path, allow_pickle=False) as lr:
        np.testing.assert_array_equal(lr['caption_id'], caption_ids)
        np.testing.assert_array_equal(lr['image_id'], image_ids)
        np.testing.assert_array_equal(lr['labels'], labels)
        predictions['Logistic regression'] = lr['predictions'].copy()
    with np.load(mlp_path, allow_pickle=False) as mlp:
        # Caption ID plus label uniquely identifies each pair in this benchmark.
        if len(set(zip(caption_ids, labels))) != len(rows):
            raise ValueError('Caption/label composite key is not unique')
        np.testing.assert_array_equal(mlp['caption_id'], caption_ids)
        np.testing.assert_array_equal(mlp['labels'], labels)
        predictions['MLP seed 42'] = mlp['predictions'].copy()
    cutoffs = np.quantile([float(r['cosine_similarity']) for r in train], [.2,.4,.6,.8])
    bins = np.searchsorted(cutoffs, scores, side='right')
    summaries = []
    for b in range(5):
        selected = bins == b
        for model, prediction in predictions.items():
            summaries.append(dict(bin=b+1, model=model, **summarise(labels[selected], prediction[selected])))
    for model, prediction in predictions.items():
        overall = summarise(labels, prediction)
        for key in ('n', 'original', 'misleading', 'tn', 'fp', 'fn', 'tp'):
            if sum(r[key] for r in summaries if r['model'] == model) != overall[key]:
                raise ValueError('Binned counts do not reconcile with overall predictions')
    # Within-caption comparison retains the original/generated mismatch relationship.
    by_caption = {}
    for i, cid in enumerate(caption_ids):
        by_caption.setdefault(int(cid), {})[int(labels[i])] = float(scores[i])
    if any(set(group) != {0,1} for group in by_caption.values()):
        raise ValueError('Expected one original and one misleading pair per caption')
    margins = np.array([group[0]-group[1] for group in by_caption.values()])
    result = dict(split='val', test_evaluated=False, bin_source='training similarity quintiles',
                  bin_cutoffs=cutoffs.tolist(), rows=summaries,
                  paired_captions=len(margins),
                  original_higher_similarity_fraction=float(np.mean(margins > 0)),
                  misleading_equal_or_higher_similarity_fraction=float(np.mean(margins <= 0)),
                  provenance_sha256={str(p.relative_to(ROOT)):digest(p) for p in [train_path,val_path,lr_path,mlp_path]},
                  interpretation='CLIP similarity is a model-derived proxy, not human-labelled semantic compatibility. Descriptive post-selection validation analysis; no causal inference.')
    (out / 'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out / 'metrics_by_bin.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT / 'artifacts/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    x=np.arange(1,6)
    original=np.array([np.sum((bins==b)&(labels==0)) for b in range(5)])
    misleading=np.array([np.sum((bins==b)&(labels==1)) for b in range(5)])
    axes[0].bar(x,original,label='Original',color='#267aab')
    axes[0].bar(x,misleading,bottom=original,label='Misleading',color='#d66a36')
    axes[0].set(ylabel='Validation pairs',title='Class mix changes with similarity')
    axes[0].legend()
    for model in predictions:
        group=[r for r in summaries if r['model']==model]
        axes[1].plot(x,[r['misleading_recall'] for r in group],marker='o',label=model)
        axes[2].plot(x,[r['false_positive_rate'] for r in group],marker='o',label=model)
    axes[1].set(ylabel='Recall',ylim=(-.03,1.03),title='How many misleading pairs are caught?')
    axes[2].set(ylabel='False-positive rate',ylim=(-.03,1.03),title='How many original pairs are flagged?')
    for ax in axes:
        ax.set(xticks=x,xlabel='CLIP similarity bin (1 low → 5 high)')
    axes[2].legend(fontsize=8)
    fig.savefig(out / 'similarity_error_analysis.png',dpi=180)
    plt.close(fig)
    print(json.dumps({'cutoffs':cutoffs.tolist(),'paired_original_higher':result['original_higher_similarity_fraction'],
                      'bins':summaries},indent=2))

if __name__=='__main__':
    main()
