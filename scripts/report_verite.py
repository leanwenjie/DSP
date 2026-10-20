"""Render the frozen VERITE evaluation report without rerunning inference."""
import json
import numpy as np
from prepare_data import ROOT


def main():
    folder = ROOT/'reports/verite/evaluation'
    report = json.loads((folder/'metrics.json').read_text())
    models = report['models']
    lines = ['# Frozen VERITE external evaluation', '',
             'Models, scalers, and thresholds were selected on NewsCLIPpings train/validation and frozen before VERITE scoring. Fresh features use standard OpenAI CLIP ViT-B/32 preprocessing. No training, calibration, or threshold optimisation used VERITE.', '',
             '## Primary task: true versus out-of-context', '',
             '338 true pairs and 325 out-of-context pairs (663 total). Positive means out-of-context. MLP seed 42 was designated before evaluation; the other seeds describe training variability.', '',
             '| Model | Accuracy | Macro-F1 | 95% macro-F1 interval | OOC recall | False-alarm rate | ROC-AUC |',
             '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for name, result in models.items():
        r = result['primary_true_vs_ooc']; lo,hi = r['macro_f1_95_cluster_bootstrap']
        lines.append(f"| {name} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} | {lo:.2%}–{hi:.2%} | {r['misleading_recall']:.2%} | {r['false_positive_rate']:.2%} | {r['roc_auc']:.4f} |")
    lines += ['', '## Secondary task: true versus all misleading', '',
              '338 true, 338 miscaptioned and 325 out-of-context pairs (1,001 total). Both misleading categories map to the positive class; this is not three-class classification.', '',
              '| Model | Accuracy | Macro-F1 | OOC detected / 325 | Miscaptioned detected / 338 | False alarms / 338 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, result in models.items():
        r=result['secondary_true_vs_all_misleading']; c=result['category_detection']
        lines.append(f"| {name} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} | {c['out-of-context']['flagged']} | {c['miscaptioned']['flagged']} | {c['true']['flagged']} |")
    lines += ['', '![Frozen-model comparison and category detection](results.png)', '', '## Variability and interpretation', '']
    for task in ['primary_true_vs_ooc','secondary_true_vs_all_misleading']:
        scores=np.array([models[f'MLP seed {s}'][task]['macro_f1'] for s in (42,43,44)])
        lines.append(f"- {task}: three-seed MLP macro-F1 mean {scores.mean():.2%}, sample SD {scores.std(ddof=1)*100:.2f} percentage points. This SD is not a confidence interval.")
    lines += ['', 'Intervals use 2,000 bootstrap draws of entire stories (seed 42), keeping related image-caption examples together. They quantify sampling variability conditional on each frozen model; they do not include preprocessing uncertainty. Overlapping intervals or point estimates alone do not establish statistical superiority.', '',
              'The all-misleading task has 66.23% positive prevalence. An always-misleading classifier therefore obtains 66.23% accuracy, but only about 39.84% macro-F1; accuracy alone is insufficient. The primary task is approximately balanced.', '',
              'These are external-transfer results, not a reproduction of RED-DOT. The original released annotation file contains 1,001 rows despite the documentation headline of 1,000. Text feature parity is supported; image parity with NewsCLIPpings is unverified and RED-DOT image feature agreement is partial. The models do not retrieve evidence about dates, locations, or events.', '',
              'VERITE has now been evaluated. Further development must not tune against these results and then describe the same set as untouched. NewsCLIPpings test and other reserved datasets were not evaluated by this script.', '',
              '## Reproduce and inspect', '',
              '```bash', '.venv/bin/python scripts/evaluate_verite.py', '.venv/bin/python scripts/report_verite.py', '```', '',
              'See `frozen_manifest.json` for model, checkpoint, image and annotation hashes and frozen thresholds. `metrics.json` includes per-class precision/recall/F1, confusion matrices (rows=true labels, columns=predictions; order true, misleading), ranking metrics and intervals. Local row-level predictions and normalised features are in ignored `artifacts/verite/`. Original model artifacts are unchanged.']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
