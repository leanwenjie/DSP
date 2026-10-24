"""Fixed follow-up: compare concatenation and frozen interaction MLPs on test."""
import json
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from prepare_data import ROOT, DATA, digest
from model_inference import predict_saved
from evaluate_newsclippings_test import connected_groups, bootstrap_scores


def main():
    out = ROOT / 'reports/mlp_concat'
    art = ROOT / 'artifacts/mlp_concat'
    manifest_path = out / 'test_frozen_manifest.json'
    paths = [ROOT / 'docs/REFERENCE_ALIGNMENT.md', ROOT / 'scripts/train_mlp.py',
             ROOT / 'scripts/evaluate_mlp_concat.py', ROOT / 'scripts/model_inference.py',
             out / 'metrics.json']
    for mode in ('mlp_concat', 'mlp'):
        paths += [ROOT / f'artifacts/{mode}/model_seed{s}.npz' for s in (42,43,44)]
    original = json.loads((ROOT / 'reports/newsclippings_test/frozen_manifest.json').read_text())['sha256']
    for p, sha in original.items():
        if digest(ROOT / p) != sha:
            raise ValueError(f'Original frozen input changed: {p}')
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    hashes.update(original)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['sha256'] != hashes:
            raise ValueError('Follow-up frozen inputs changed')
    else:
        manifest_path.write_text(json.dumps({'sha256': hashes, 'policy': 'Follow-up after prior test inspection; validation-selected thresholds; no test tuning'}, indent=2)+'\n')
    rows = json.loads((DATA / 'newsclippings/test.json').read_text())['annotations']
    y = np.array([int(r['falsified']) for r in rows])
    assert len(y) == 7264 and np.bincount(y).tolist() == [3632,3632]
    vectors = []
    for modality, key in [('image','image_id'), ('text','id')]:
        name = f'clip_{modality}_embeddings'
        path = DATA / f'newsclippings/embeddings/{name}/{name}_test.pkl'
        # Hash checked against original frozen manifest before deserialization.
        with path.open('rb') as f:
            mapping = pickle.load(f)
        a = np.stack([mapping[r[key]] for r in rows]).astype(np.float32)
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        if a.shape != (len(y),512) or not np.isfinite(a).all() or (norms == 0).any():
            raise ValueError('Invalid embeddings')
        vectors.append(a / norms)
    groups = connected_groups(rows)
    results, deltas = {}, {}
    with threadpool_limits(limits=2):
        for seed in (42,43,44):
            boots = {}
            for mode in ('mlp_concat','mlp'):
                score, pred = predict_saved(vectors, ROOT / f'artifacts/{mode}/model_seed{seed}.npz')
                r = classification_report(y, pred, output_dict=True, zero_division=0)
                boots[mode] = bootstrap_scores(y, pred, groups)
                results[f'{mode}_{seed}'] = dict(accuracy=r['accuracy'], macro_f1=r['macro avg']['f1-score'],
                    report=r, confusion_matrix=confusion_matrix(y,pred).tolist(),
                    macro_f1_95_interval=np.quantile(boots[mode],[.025,.975]).tolist())
                np.savez(art / f'test_{mode}_{seed}.npz', labels=y, score=score, predictions=pred)
            deltas[str(seed)] = dict(interaction_minus_concat=results[f'mlp_{seed}']['macro_f1']-results[f'mlp_concat_{seed}']['macro_f1'],
                paired_95_interval=np.quantile(boots['mlp']-boots['mlp_concat'],[.025,.975]).tolist())
    for p, sha in hashes.items():
        if digest(ROOT/p) != sha:
            raise ValueError(f'Frozen input changed: {p}')
    (out / 'test_metrics.json').write_text(json.dumps(dict(count=len(y),models=results,deltas=deltas,
        bootstrap={'repetitions':2000,'seed':42,'groups':len(np.unique(groups))},
        seed_policy='42 designated; all seeds reported; no test selection'),indent=2)+'\n')
    lines = ['# Paper-inspired MLP feature ablation', '',
        'Follow-up experiment after inspecting earlier test results. This is not an exact reproduction of Shalabi et al. Table IV. See [fixed protocol](../../docs/REFERENCE_ALIGNMENT.md).', '',
        '| Features | Seed | Test accuracy | Test macro-F1 | OOC recall |', '| --- | ---: | ---: | ---: | ---: |']
    for name,r in results.items():
        mode,seed=name.rsplit('_',1)
        lines.append(f"| {mode} | {seed} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} | {r['report']['1']['recall']:.2%} |")
    lines += ['', '| Seed | Interaction minus concat macro-F1 (pp) | Paired 95% interval (pp) |', '| --- | ---: | --- |']
    for seed,r in deltas.items():
        lo,hi=r['paired_95_interval']
        lines.append(f"| {seed} | {100*r['interaction_minus_concat']:+.2f} | {100*lo:+.2f} to {100*hi:+.2f} |")
    lines += ['', 'Intervals resample connected shared VisualNews-ID groups; same draws for both models. They do not include training variability or correct for prior test inspection. Seeds are reported separately. Feature dimensionality and parameter counts differ; all training settings match.', '',
        'Published context: Shalabi et al. report 53% CLIP-only MLP accuracy, 56% CLIP-only Transformer accuracy and 68% best generation-assisted accuracy. Our baseline is an adaptation with explicit settings, not a replication. These published scores have no paired predictions available here.', '',
        'Original model files, original evaluation reports and app default are unchanged. Training metrics and curves are in metrics.json and learning_curves.png; full test metrics are in test_metrics.json.']
    (out / 'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
