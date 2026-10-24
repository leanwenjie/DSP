"""Three-seed, single-architecture MLP study. No test data access."""
import argparse
import json
import os
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits
from similarity_baseline import load_vectors, select_threshold
from model_inference import features, saved_probability, predict_saved
from prepare_data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-mode', choices=('interaction', 'concat'), default='interaction')
    args = parser.parse_args()
    mode = args.feature_mode
    study = 'mlp' if mode == 'interaction' else 'mlp_concat'
    out, artifacts = ROOT / f'reports/{study}', ROOT / f'artifacts/{study}'
    if mode == 'concat' and artifacts.exists() and any(artifacts.iterdir()):
        raise FileExistsError('Preserve existing concatenation study; use its saved results')
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    data = {s: load_vectors(s) for s in ('train', 'val')}
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features(data['train'][1], mode))
    x_val = scaler.transform(features(data['val'][1], mode))
    y_train, y_val = data['train'][2], data['val'][2]
    config = dict(hidden_layer_sizes=(32,), activation='relu', solver='adam', alpha=.001,
                  batch_size=512, learning_rate_init=.0003, shuffle=True, early_stopping=False)
    runs = []
    with threadpool_limits(limits=2):
        for seed in (42, 43, 44):
            model = MLPClassifier(**config, random_state=seed)
            best_score, stale, history, best, best_weights = -1, 0, [], None, None
            started = time.monotonic()
            for epoch in range(1, 31):
                model.partial_fit(x_train, y_train, classes=np.array([0, 1]))
                probability = model.predict_proba(x_val)[:, 1].astype(np.float64)
                threshold, score, _, _ = select_threshold(-probability, y_val)
                history.append({'epoch': epoch, 'train_loss': float(model.loss_), 'val_macro_f1': score})
                print(f'seed={seed} epoch={epoch} loss={model.loss_:.4f} val_macro_f1={score:.4f}', flush=True)
                if score > best_score:
                    best_score, stale = score, 0
                    prediction = (-probability < threshold).astype(int)
                    best = dict(seed=seed, selected_epoch=epoch, validation_macro_f1=score,
                                threshold_on_negative_probability=threshold,
                                report=classification_report(y_val, prediction, target_names=['original','out_of_context'], output_dict=True),
                                confusion_matrix=confusion_matrix(y_val, prediction).tolist(),
                                roc_auc=float(roc_auc_score(y_val, probability)))
                    if not np.isclose(score, best['report']['macro avg']['f1-score'], atol=1e-12, rtol=0):
                        raise RuntimeError('Threshold objective disagrees with prediction metrics')
                    best_weights = dict(w0=model.coefs_[0].copy(), w1=model.coefs_[1].copy(),
                                        b0=model.intercepts_[0].copy(), b1=model.intercepts_[1].copy())
                    np.savez(artifacts / f'model_seed{seed}.npz', **best_weights,
                             mean=scaler.mean_, scale=scaler.scale_, threshold=threshold, feature_mode=mode)
                    np.savez(artifacts / f'validation_seed{seed}.npz', labels=y_val, probability=probability,
                             predictions=prediction, caption_id=[r['id'] for r in data['val'][0]])
                else:
                    stale += 1
                if stale >= 5:
                    break
            best.update(history=history, seconds=time.monotonic()-started, epochs_run=epoch,
                        stop_reason='patience_5' if stale >= 5 else 'max_epochs_30')
            # Validate complete saved numeric inference, including preprocessing.
            restored, saved_predictions = predict_saved(data['val'][1], artifacts / f'model_seed{seed}.npz')
            with np.load(artifacts / f'validation_seed{seed}.npz', allow_pickle=False) as reference:
                np.testing.assert_allclose(restored, reference['probability'], rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(saved_predictions, reference['predictions'])
            best['saved_prediction_parity'] = True
            runs.append(best)
            (out / 'runs.json').write_text(json.dumps(runs, indent=2)+'\n')
    scores = [r['validation_macro_f1'] for r in runs]
    report = dict(feature_mode=mode, config=config, seeds=[42,43,44], max_epochs=30, patience=5,
                  provisional_model_seed=42, seed_policy='42 preselected; report all seeds, do not pick the best seed',
                  scaler_fit_split='train', selection_split='val', test_evaluated=False,
                  provenance_sha256={s:d[3] for s,d in data.items()}, runs=runs,
                  macro_f1_mean=float(np.mean(scores)), macro_f1_sample_std=float(np.std(scores, ddof=1)),
                  warning='Validation selects epochs and thresholds. Optimistic development scores; seed SD is not a confidence interval.')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5), constrained_layout=True)
    for run in runs:
        ax.plot([r['epoch'] for r in run['history']], [r['val_macro_f1'] for r in run['history']], marker='.', label=f"Seed {run['seed']}")
    logistic = json.loads((ROOT / 'reports/logistic/metrics.json').read_text())['selected']['validation_macro_f1']
    ax.axhline(logistic, linestyle='--', color='black', label='Selected logistic regression')
    ax.set(xlabel='Epoch', ylabel='Validation macro-F1', title=f'MLP {mode} development curves — validation selection')
    ax.legend()
    fig.savefig(out / 'learning_curves.png', dpi=180)
    plt.close(fig)
    print('MLP mean macro-F1:', report['macro_f1_mean'], 'sample SD:', report['macro_f1_sample_std'])

if __name__ == '__main__':
    main()
