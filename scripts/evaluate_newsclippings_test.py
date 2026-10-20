"""Evaluate frozen models once selected; never train or tune on test labels."""
import csv
import json
import os
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from prepare_data import ROOT, DATA, digest
from train_logistic import features, predict_saved
from train_mlp import saved_probability


def connected_groups(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in rows:
        a, b = find(row['id']), find(row['image_id'])
        parent[b] = a
    return np.array([find(row['id']) for row in rows])


def bootstrap_scores(y, pred, groups, repetitions=2000):
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(unique), 4), dtype=np.int64)
    np.add.at(counts, (inverse, y*2+pred), 1)
    rng = np.random.default_rng(42)
    values = []
    for start in range(0, repetitions, 100):
        sampled = counts[rng.integers(0, len(unique), size=(min(100,repetitions-start),len(unique)))].sum(axis=1)
        tn,fp,fn,tp = sampled.T
        values.extend(.5*(2*tn/np.maximum(2*tn+fp+fn,1)+2*tp/np.maximum(2*tp+fp+fn,1)))
    return np.array(values)


def main():
    out=ROOT/'reports/newsclippings_test'; art=ROOT/'artifacts/newsclippings_test'
    out.mkdir(exist_ok=True); art.mkdir(exist_ok=True)
    annotation=DATA/'newsclippings/test.json'
    if digest(annotation)!='616e06af39fd8afa88d1920e455e4d6ca69e013c9b2fa4dadda593fb0e813aea':
        raise ValueError('Unexpected test annotations')
    rows=json.loads(annotation.read_text())['annotations']
    y=np.array([int(r['falsified']) for r in rows]); groups=connected_groups(rows)
    if len(y)!=7264 or np.bincount(y).tolist()!=[3632,3632]: raise ValueError('Unexpected split counts')
    models=[ROOT/'artifacts/logistic/best_model.npz']+[ROOT/f'artifacts/mlp/model_seed{s}.npz' for s in (42,43,44)]
    paths=models+[ROOT/'reports/baseline/metrics.json',out/'PROTOCOL.md',annotation,ROOT/'scripts/evaluate_newsclippings_test.py']
    provenance={str(p.relative_to(ROOT)):digest(p) for p in paths}
    vectors=[]
    for modality,key in [('image','image_id'),('text','id')]:
        name=f'clip_{modality}_embeddings'
        path=DATA/f'newsclippings/embeddings/{name}/{name}_test.pkl'
        receipt=json.loads(path.with_name(path.name+'.receipt.json').read_text())
        sha=digest(path)
        if not receipt.get('upstream_sha256') or sha!=receipt['upstream_sha256']: raise ValueError('Unverified embeddings')
        provenance[str(path.relative_to(ROOT))]=sha
        with path.open('rb') as f: mapping=pickle.load(f)  # Only verified official artifacts.
        a=np.stack([mapping[r[key]] for r in rows]).astype(np.float32); del mapping
        norms=np.linalg.norm(a,axis=1,keepdims=True)
        if a.shape!=(len(rows),512) or not np.isfinite(a).all() or np.any(norms==0): raise ValueError('Invalid embeddings')
        vectors.append(a/norms)
    threshold=json.loads(paths[4].read_text())['threshold']
    thresholds={'Similarity':{'cosine_less_than':threshold}}
    names=['Logistic regression','MLP seed 42','MLP seed 43','MLP seed 44']
    for name,path in zip(names,models):
        with np.load(path,allow_pickle=False) as d: thresholds[name]={'negative_score_less_than':float(d['threshold'])}
    (out/'frozen_manifest.json').write_text(json.dumps(dict(sha256=provenance,thresholds=thresholds,normalisation='float32 L2; same as training'),indent=2)+'\n')
    cosine=np.einsum('ij,ij->i',*vectors,dtype=np.float64)
    predictions={'Similarity':(-cosine,(cosine<threshold).astype(int))}
    with threadpool_limits(limits=2):
        predictions[names[0]]=predict_saved(vectors,models[0])
        for name,path in zip(names[1:],models[1:]):
            with np.load(path,allow_pickle=False) as d:
                x=features(vectors,'interaction').astype(np.float32)
                x-=d['mean'].astype(x.dtype); x/=d['scale'].astype(x.dtype)
                score=saved_probability(x,d)
                predictions[name]=(score,(-score<float(d['threshold'])).astype(int))
    results={}; reference=None
    for name,(score,pred) in predictions.items():
        boot=bootstrap_scores(y,pred,groups)
        if reference is None: reference=boot
        r=classification_report(y,pred,labels=[0,1],target_names=['original','out_of_context'],output_dict=True,zero_division=0)
        results[name]=dict(accuracy=float(np.mean(y==pred)),macro_f1=r['macro avg']['f1-score'],classification_report=r,
                          confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),roc_auc=float(roc_auc_score(y,score)),
                          average_precision=float(average_precision_score(y,score)),macro_f1_95_interval=np.quantile(boot,[.025,.975]).tolist(),
                          paired_delta_vs_similarity_95_interval=np.quantile(boot-reference,[.025,.975]).tolist())
    with (art/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['row_index','caption_id','image_id','group','label','model','score','prediction'])
        for name,(score,pred) in predictions.items():
            w.writerows((i,r['id'],r['image_id'],int(groups[i]),int(y[i]),name,float(score[i]),int(pred[i])) for i,r in enumerate(rows))
    _,sizes=np.unique(groups,return_counts=True)
    report=dict(split='test',count=len(rows),class_counts=np.bincount(y).tolist(),models=results,
                bootstrap=dict(unit='connected shared VisualNews IDs',groups=len(sizes),largest_group=int(sizes.max()),repetitions=2000,seed=42),
                predictions_sha256=digest(art/'predictions.csv'),seed_policy='42 predesignated; no test-based selection')
    for p,sha in provenance.items():
        if digest(ROOT/p)!=sha: raise RuntimeError(f'Frozen input changed: {p}')
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Frozen NewsCLIPpings test results','','All 7,264 test pairs (3,632 per class) were scored with unchanged training-fitted models and validation-selected thresholds. No VERITE result was used to alter these models.','',
           '| Model | Accuracy | Macro-F1 | 95% macro-F1 interval | OOC precision | OOC recall |', '| --- | ---: | ---: | --- | ---: | ---: |']
    for name,r in results.items():
        c=r['classification_report']['out_of_context']; lo,hi=r['macro_f1_95_interval']
        lines.append(f"| {name} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} | {lo:.2%}–{hi:.2%} | {c['precision']:.2%} | {c['recall']:.2%} |")
    lines+=['',f"Intervals use 2,000 grouped bootstrap draws: {len(sizes)} connected groups of shared VisualNews IDs; largest group {sizes.max()} pairs. This preserves known content reuse, but cannot identify unrecorded content duplication.",'','| Model | Macro-F1 change vs similarity (percentage points) | Paired 95% interval (percentage points) |','| --- | ---: | --- |']
    for name,r in results.items():
        if name=='Similarity': continue
        lo,hi=r['paired_delta_vs_similarity_95_interval'];delta=r['macro_f1']-results['Similarity']['macro_f1']
        lines.append(f'| {name} | {100*delta:+.2f} | {100*lo:+.2f} to {100*hi:+.2f} |')
    lines+=['','MLP seed 42 remains the predesignated model. Other seeds describe variability and must not be selected by these scores. These intervals concern this benchmark and frozen models; they do not establish factual verification of arbitrary uploads.','',
            'The test split is now evaluated. Do not tune on these results and then describe it as untouched. VERITE evaluates a different distribution and label construction; cross-dataset comparisons are descriptive. 5Pils-OOC remains unscored.','',
            '![Held-out comparison](comparison.png)','','Reproduce with `.venv/bin/python scripts/evaluate_newsclippings_test.py`. Full per-class metrics, confusion matrices and ranking metrics are in `metrics.json`; hashes and thresholds are in `frozen_manifest.json`. Row-level predictions remain in ignored `artifacts/newsclippings_test/`.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verite=json.loads((ROOT/'reports/verite/evaluation/metrics.json').read_text())['models']
    names=['Similarity','Logistic regression','MLP seed 42']; x=np.arange(3)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for offset,vals,label in [(-.18,[results[n]['macro_f1'] for n in names],'NewsCLIPpings test'),(.18,[verite[n]['primary_true_vs_ooc']['macro_f1'] for n in names],'VERITE true vs out-of-context')]:
        bars=ax.bar(x+offset,vals,width=.36,label=label);ax.bar_label(bars,fmt='%.3f',padding=3)
    ax.set(xticks=x,xticklabels=names,ylim=(0,1),ylabel='Macro-F1',title='Frozen models on two held-out benchmarks');ax.legend()
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
