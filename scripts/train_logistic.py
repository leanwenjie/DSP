"""Train-only scaling/fitting; validation-only LR feature/C/threshold selection."""
import json
import os
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from similarity_baseline import load_vectors, select_threshold
from prepare_data import ROOT


def features(vectors, mode):
    image, text = vectors
    if image.shape != text.shape or image.ndim != 2:
        raise ValueError('Aligned image/text matrices required')
    if mode == 'concat':
        return np.concatenate([image, text], axis=1)
    if mode == 'interaction':
        cosine = np.einsum('ij,ij->i', image, text)[:, None]
        return np.concatenate([image, text, np.abs(image-text), image*text, cosine], axis=1)
    raise ValueError('Unknown feature mode')


def predict_saved(vectors, path):
    """Apply saved numeric parameters to L2-normalised CLIP vectors."""
    with np.load(path, allow_pickle=False) as saved:
        x = features(vectors, str(saved['feature_mode'])).astype(np.float32)
        # Match sklearn's float32, in-place StandardScaler transform.
        x -= saved['mean'].astype(x.dtype)
        x /= saved['scale'].astype(x.dtype)
        decision = (x @ saved['coef'].T).ravel() + saved['intercept'][0]
        return decision, (-decision < float(saved['threshold'])).astype(int)


def main():
    out = ROOT / 'reports/logistic'
    artifacts = ROOT / 'artifacts/logistic'
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    data = {split: load_vectors(split) for split in ('train', 'val')}
    y_train, y_val = data['train'][2], data['val'][2]
    results = []
    best = None
    with threadpool_limits(limits=2):
        for mode in ('concat', 'interaction'):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features(data['train'][1], mode))
            x_val = scaler.transform(features(data['val'][1], mode))
            for c in (0.001, 0.01, 0.1):
                start = time.monotonic()
                model = LogisticRegression(C=c, solver='lbfgs', max_iter=1000, tol=1e-5, random_state=42)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    model.fit(x_train, y_train)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError(f'Failed to converge: {mode}, C={c}')
                decision = model.decision_function(x_val)
                # Reuse verified less-than threshold logic by negating the decision.
                threshold, macro, _, _ = select_threshold(-decision, y_val)
                prediction = (-decision < threshold).astype(int)
                record = {'features': mode, 'dimensions': x_train.shape[1], 'C': c,
                    'iterations': int(model.n_iter_[0]), 'seconds': time.monotonic()-start,
                    'threshold_on_negative_decision': threshold, 'validation_macro_f1': macro,
                    'validation_macro_f1_default_threshold': float(f1_score(y_val, decision > 0, average='macro')),
                    'validation_report': classification_report(y_val, prediction, target_names=['original','out_of_context'], output_dict=True),
                    'validation_confusion_matrix': confusion_matrix(y_val, prediction).tolist(),
                    'validation_roc_auc': float(roc_auc_score(y_val, decision))}
                results.append(record)
                print(json.dumps({k:record[k] for k in ('features','C','iterations','seconds','validation_macro_f1')}), flush=True)
                if best is None or macro > best['validation_macro_f1']:
                    best = record.copy()
                    # Save portable numeric parameters, avoiding executable model pickles.
                    np.savez(artifacts / 'best_model.npz', mean=scaler.mean_, scale=scaler.scale_,
                             coef=model.coef_, intercept=model.intercept_, threshold=threshold,
                             feature_mode=mode, classes=model.classes_)
                    np.savez(artifacts / 'validation_predictions.npz',
                             caption_id=np.array([r['id'] for r in data['val'][0]]),
                             image_id=np.array([r['image_id'] for r in data['val'][0]]),
                             labels=y_val, decision=decision, predictions=prediction)
                    # Verify saved numeric inference reproduces the sklearn path.
                    saved = np.load(artifacts / 'best_model.npz', allow_pickle=False)
                    restored = (x_val @ saved['coef'].T).ravel() + saved['intercept'][0]
                    if not np.allclose(restored, decision, rtol=1e-7, atol=1e-7):
                        raise RuntimeError('Saved model inference mismatch')
            del x_train, x_val
    report = {'selection_split': 'val', 'test_evaluated': False,
              'warning': 'Feature set, C, and threshold selected on validation; these are optimistic development scores.',
              'scaler_fit_split': 'train', 'training_pairs': len(y_train), 'validation_pairs': len(y_val),
              'provenance_sha256': {s:d[3] for s,d in data.items()}, 'candidates': results, 'selected': best}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    cache = ROOT / 'artifacts/matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline = json.loads((ROOT / 'reports/baseline/metrics.json').read_text())
    labels = ['Similarity'] + [f'{r["features"]}\nC={r["C"]}' for r in results]
    values = [baseline['splits']['val']['classification_report']['macro avg']['f1-score']] + [r['validation_macro_f1'] for r in results]
    fig, ax = plt.subplots(figsize=(11,5), constrained_layout=True)
    bars = ax.bar(labels, values, color=['#888888']+['#267aab']*3+['#d66a36']*3)
    ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.set(ylim=(0,1), ylabel='Validation macro-F1', title='Development comparison — validation-selected thresholds; test sets reserved')
    fig.savefig(out / 'validation_comparison.png', dpi=180)
    plt.close(fig)
    print('Selected:', best['features'], 'C=', best['C'], 'macro-F1=',best['validation_macro_f1'])

if __name__ == '__main__':
    main()
