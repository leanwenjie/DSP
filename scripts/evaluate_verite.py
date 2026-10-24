"""Frozen external evaluation; no fitting or threshold selection on VERITE."""
import csv
import json
import os
import re
from pathlib import Path
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from prepare_data import ROOT, digest
from model_inference import predict_saved


def story_id(path):
    match = re.fullmatch(r'(?:true|false)_(\d+)\.[^.]+', Path(path).name)
    if not match:
        raise ValueError(f'Cannot establish story ID: {path}')
    return int(match.group(1))


def bootstrap_macro_f1(y, pred, groups, repetitions=2000):
    """Resample whole stories; derive macro-F1 from accumulated confusion counts."""
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(unique), 4), dtype=np.int64)
    np.add.at(counts, (inverse, y * 2 + pred), 1)
    rng = np.random.default_rng(42)
    sampled = counts[rng.integers(0, len(unique), size=(repetitions, len(unique)))].sum(axis=1)
    tn, fp, fn, tp = sampled.T
    f1 = .5 * (2 * tn / np.maximum(2 * tn + fp + fn, 1) + 2 * tp / np.maximum(2 * tp + fp + fn, 1))
    return np.quantile(f1, [.025, .975]).tolist()


def measure(y, pred, score, groups):
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    report = classification_report(y, pred, labels=[0, 1], target_names=['true', 'misleading'], output_dict=True, zero_division=0)
    return dict(count=len(y), accuracy=float(np.mean(y == pred)), macro_f1=report['macro avg']['f1-score'],
                macro_f1_95_cluster_bootstrap=bootstrap_macro_f1(y, pred, groups),
                classification_report=report, confusion_matrix=matrix.tolist(),
                false_positive_rate=float(fp / (tn + fp)), misleading_recall=float(tp / (tp + fn)),
                roc_auc=float(roc_auc_score(y, score)), average_precision=float(average_precision_score(y, score)))


def main():
    torch.set_num_threads(2)
    out = ROOT / 'reports/verite/evaluation'
    artifacts = ROOT / 'artifacts/verite'
    out.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
    annotation = ROOT / 'data/verite/VERITE.csv'
    if digest(annotation) != '7f5661860f2e34624c1721710ed37fac5e1e78a4307400530dd587cb0d15a040':
        raise ValueError('Unexpected annotation revision')
    with annotation.open() as f:
        rows = list(csv.DictReader(f))
    labels = np.array([r['label'] for r in rows])
    if set(labels) != {'true', 'miscaptioned', 'out-of-context'}:
        raise ValueError('Unexpected labels')
    groups = np.array([story_id(r['image_path']) for r in rows])
    paths = [ROOT/'reports/baseline/metrics.json', ROOT/'artifacts/logistic/best_model.npz']
    paths += [ROOT/f'artifacts/mlp/model_seed{s}.npz' for s in (42,43,44)]
    paths += [ROOT/'reports/verite/EVALUATION_PROTOCOL.md', ROOT/'artifacts/clip/ViT-B-32.pt', annotation]
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    baseline_threshold = json.loads(paths[0].read_text())['threshold']
    thresholds = {'Similarity': {'cosine_less_than': baseline_threshold}}
    for name, p in zip(['Logistic regression', 'MLP seed 42', 'MLP seed 43', 'MLP seed 44'], paths[1:5]):
        with np.load(p, allow_pickle=False) as d:
            thresholds[name] = {'negative_score_less_than': float(d['threshold'])}
    model, preprocess = clip.load('ViT-B/32', device='cpu', jit=False, download_root=str(ROOT/'artifacts/clip'))
    model.eval()
    manifest = dict(artifact_sha256=frozen, thresholds=thresholds, model='OpenAI CLIP ViT-B/32',
                    preprocess=str(preprocess), torch_version=torch.__version__, device='cpu',
                    clip_commit='d05afc436d78f1c48dc0dbf8e5980a9d471f35f6',
                    image_sha256={p: digest(ROOT/'data/verite'/p) for p in sorted(set(r['image_path'] for r in rows))},
                    policy='Frozen NewsCLIPpings models and thresholds. Standard OpenAI preprocessing. No RED-DOT pre-resize. No fitting on VERITE.')
    # Written before generating classifier predictions.
    (out/'frozen_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    encoded = {}
    with torch.inference_mode():
        for modality, key in [('image','image_path'), ('text','caption')]:
            unique = list(dict.fromkeys(r[key] for r in rows))
            chunks = []
            for start in range(0, len(unique), 16):
                batch = unique[start:start+16]
                if modality == 'image':
                    tensors = []
                    for p in batch:
                        with Image.open(ROOT/'data/verite'/p) as im:
                            tensors.append(preprocess(im))
                    values = model.encode_image(torch.stack(tensors))
                else:
                    values = model.encode_text(clip.tokenize(batch, truncate=True))
                chunks.append(values.float().numpy())
                if start % 160 == 0:
                    print(f'{modality}: {min(start+16,len(unique))}/{len(unique)}', flush=True)
            matrix = np.concatenate(chunks)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if not np.isfinite(matrix).all() or np.any(norms == 0):
                raise ValueError('Invalid embeddings')
            matrix /= norms
            lookup = {value: i for i, value in enumerate(unique)}
            encoded[modality] = matrix[[lookup[r[key]] for r in rows]]
    vectors = [encoded['image'], encoded['text']]
    np.savez_compressed(artifacts/'features.npz', **encoded)
    cosine = np.einsum('ij,ij->i', *vectors, dtype=np.float64)
    predictions = {'Similarity': (-cosine, (cosine < baseline_threshold).astype(int))}
    with threadpool_limits(limits=2):
        predictions['Logistic regression'] = predict_saved(vectors, paths[1])
        for seed, path in zip((42,43,44), paths[2:5]):
            predictions[f'MLP seed {seed}'] = predict_saved(vectors, path)
    y = (labels != 'true').astype(int)
    results = {}
    for name, (score, pred) in predictions.items():
        results[name] = {}
        for task, mask in [('primary_true_vs_ooc', labels != 'miscaptioned'), ('secondary_true_vs_all_misleading', np.ones(len(rows), dtype=bool))]:
            results[name][task] = measure(y[mask], pred[mask], score[mask], groups[mask])
        results[name]['category_detection'] = {label: dict(count=int(np.sum(labels==label)), flagged=int(pred[labels==label].sum()), flag_rate=float(pred[labels==label].mean())) for label in sorted(set(labels))}
    with (artifacts/'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_index','story_id','label','model','score','prediction'])
        for name, (score,pred) in predictions.items():
            writer.writerows((i,int(groups[i]),labels[i],name,float(score[i]),int(pred[i])) for i in range(len(rows)))
    report = dict(dataset='VERITE released 1001-row file', models=results, bootstrap=dict(repetitions=2000,seed=42,unit='story_id',interval='95% percentile'),
                  mlp_seed_policy='42 predesignated; 43 and 44 variability only',
                  limitations=['No external evidence retrieval; contextual verification limited.', 'Image parity with NewsCLIPpings training features unverified; RED-DOT image feature agreement partial.', 'Documentation says 1000 rows; original released file contains 1001.'],
                  feature_sha256=digest(artifacts/'features.npz'), predictions_sha256=digest(artifacts/'predictions.csv'))
    for p in paths:
        if digest(p) != frozen[str(p.relative_to(ROOT))]:
            raise RuntimeError(f'Frozen input changed: {p}')
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['Similarity','Logistic regression','MLP seed 42']
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x = np.arange(len(names))
    for offset, task, label in [(-.18,'primary_true_vs_ooc','True vs out-of-context'),(.18,'secondary_true_vs_all_misleading','True vs all misleading')]:
        vals = np.array([results[n][task]['macro_f1'] for n in names])
        ci = np.array([results[n][task]['macro_f1_95_cluster_bootstrap'] for n in names])
        axes[0].bar(x+offset, vals, width=.36,label=label,yerr=np.maximum(0,np.stack([vals-ci[:,0],ci[:,1]-vals])),capsize=3)
    axes[0].set(xticks=x,xticklabels=names,ylim=(0,1),ylabel='Macro-F1',title='Frozen external evaluation (95% story bootstrap)')
    axes[0].legend(fontsize=8)
    for offset, label, title in [(-.24,'true','False alarms on true pairs'),(0,'out-of-context','Out-of-context recall'),(.24,'miscaptioned','Miscaptioned recall')]:
        axes[1].bar(x+offset,[results[n]['category_detection'][label]['flag_rate'] for n in names],width=.24,label=title)
    axes[1].set(xticks=x,xticklabels=names,ylim=(0,1),ylabel='Fraction flagged as misleading',title='Detection and false alarms by category')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'results.png',dpi=180); plt.close(fig)
    print(json.dumps({n:{t:results[n][t]['macro_f1'] for t in ['primary_true_vs_ooc','secondary_true_vs_all_misleading']} for n in results},indent=2))

if __name__ == '__main__':
    main()
