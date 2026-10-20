"""CLIP cosine-similarity baseline. Train/validation only; no test-set access."""
import csv
import json
import os
import pickle
from pathlib import Path
import numpy as np
from prepare_data import ROOT, DATA, digest


def select_threshold(scores, labels):
    """Maximise macro-F1 for prediction = score < threshold; ties choose lowest threshold."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if len(scores) == 0 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Expected aligned nonempty finite scores and labels')
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError('Both binary classes are required')
    order = np.argsort(scores, kind='stable')
    s, y = scores[order], labels[order]
    # Never split identical scores between predicted classes.
    ends = np.r_[np.flatnonzero(np.diff(s)) + 1, len(s)]
    k = np.r_[0, ends]
    positives = np.r_[0, np.cumsum(y)][k]
    tp, fp = positives, k - positives
    fn, tn = y.sum() - tp, len(y) - y.sum() - fp
    f1_pos = 2 * tp / (2 * tp + fp + fn)
    f1_neg = 2 * tn / (2 * tn + fp + fn)
    macro = (f1_pos + f1_neg) / 2
    thresholds = np.r_[s[0], np.nextafter(s[ends - 1], np.inf)]
    best = int(np.argmax(macro))
    return float(thresholds[best]), float(macro[best]), thresholds, macro


def load_vectors(split):
    if split not in ('train', 'val'):
        raise ValueError('This development script does not access test data')
    rows = json.loads((DATA / f'newsclippings/{split}.json').read_text())['annotations']
    vectors = []
    provenance = {}
    for modality, key in [('image', 'image_id'), ('text', 'id')]:
        name = f'clip_{modality}_embeddings'
        path = DATA / f'newsclippings/embeddings/{name}/{name}_{split}.pkl'
        receipt = json.loads(path.with_name(path.name + '.receipt.json').read_text())
        checksum = digest(path)
        if not receipt.get('upstream_sha256') or checksum != receipt['upstream_sha256']:
            raise ValueError(f'Official embedding hash mismatch: {path}')
        provenance[modality] = checksum
        with path.open('rb') as f:
            mapping = pickle.load(f)  # Only pinned official files verified above.
        a = np.stack([mapping[r[key]] for r in rows]).astype(np.float32)
        del mapping
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        if a.shape != (len(rows), 512) or not np.isfinite(a).all() or (norms == 0).any():
            raise ValueError('Invalid embeddings')
        vectors.append(a / norms)
    labels = np.array([int(r['falsified']) for r in rows])
    provenance['annotations'] = digest(DATA / f'newsclippings/{split}.json')
    return rows, vectors, labels, provenance


def load_split(split):
    rows, vectors, labels, provenance = load_vectors(split)
    # Float64 dot product ensures threshold comparisons round-trip consistently.
    scores = np.einsum('ij,ij->i', *vectors, dtype=np.float64)
    return rows, scores, labels, provenance


def main():
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    out = ROOT / 'reports/baseline'
    out.mkdir(parents=True, exist_ok=True)
    predictions_dir = ROOT / 'artifacts/baseline'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'artifacts/matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {s: load_split(s) for s in ('train', 'val')}
    threshold, best, candidates, objective = select_threshold(data['val'][1], data['val'][2])
    result = {'model': 'Frozen official CLIP ViT-B/32 cosine similarity',
              'positive_class': 'out_of_context', 'prediction_rule': 'cosine_similarity < threshold',
              'threshold': threshold, 'selection_split': 'val', 'selection_metric': 'macro-F1',
              'warning': 'Validation performance is optimistically selected; not a final test result.',
              'test_evaluated': False, 'splits': {}}
    for split, (rows, scores, labels, provenance) in data.items():
        predictions = (scores < threshold).astype(int)
        result['splits'][split] = {'count': len(labels), 'provenance_sha256': provenance,
            'classification_report': classification_report(labels, predictions, target_names=['original', 'out_of_context'], output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
            'roc_auc': float(roc_auc_score(labels, -scores))}
        with (predictions_dir / f'{split}_predictions.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['caption_id', 'image_id', 'label', 'cosine_similarity', 'prediction'])
            writer.writerows((r['id'], r['image_id'], int(y), float(s), int(p)) for r, y, s, p in zip(rows, labels, scores, predictions))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    bins = np.linspace(min(x[1].min() for x in data.values()), max(x[1].max() for x in data.values()), 45)
    for ax, (split, (_, scores, labels, _)) in zip(axes, data.items()):
        for label, title, color in [(0, 'Original', '#267aab'), (1, 'Out of context', '#d66a36')]:
            ax.hist(scores[labels == label], bins=bins, density=True, alpha=.55, label=title, color=color)
        ax.axvline(threshold, color='black', linestyle='--', label='Validation-selected threshold')
        ax.set(title=f'{split}: CLIP similarity', xlabel='Cosine similarity', ylabel='Density')
        ax.legend(fontsize=8)
    fig.savefig(out / 'similarity_distributions.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(candidates, objective, color='#267aab')
    axes[0].axvline(threshold, color='black', linestyle='--')
    axes[0].set(title='Validation threshold selection', xlabel='Threshold', ylabel='Macro-F1')
    cm = np.array(result['splits']['val']['confusion_matrix'])
    axes[1].imshow(cm, cmap='Blues')
    for (i,j), value in np.ndenumerate(cm):
        axes[1].text(j, i, str(value), ha='center', va='center')
    axes[1].set(xticks=[0,1], yticks=[0,1], xticklabels=['Original','Out of context'], yticklabels=['Original','Out of context'], xlabel='Predicted', ylabel='Actual', title='Validation confusion matrix')
    fig.savefig(out / 'validation_selection.png', dpi=180)
    plt.close(fig)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'threshold': threshold, 'validation_macro_f1': best,
                     'validation_accuracy': result['splits']['val']['classification_report']['accuracy'],
                     'test_evaluated': False}, indent=2))

if __name__ == '__main__':
    main()
