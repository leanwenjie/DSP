# Evaluation completed

See [consolidated results and chart](RESULTS.md). All three tasks/backbones and paired intervals finished; the following describes the reproducible run.

# Frozen three-backbone follow-up evaluation

Started 2026-10-29 after all development audits passed. See [protocol](PROTOCOL.md) and `frozen_manifest.json`. Completion must be read from `artifacts/backbone_evaluation/progress.json`; this document is not a live status indicator.

Audits verified complete finite unit-normalised training/validation features, saved validation prediction parity, reconciliation of selected metrics, unchanged model hashes and source provenance. The B/16 historical trainer is checked against its preserved pre-resume-fix source. No training settings changed for evaluation.

Run/resume: `.venv/bin/python scripts/evaluate_clip_backbones.py`

Persistent log: `artifacts/backbone_evaluation_run.log`. Feature extraction resumes at completed batch boundaries. A process lock prevents simultaneous runs. If the environment stops, launch the same command; do not change frozen inputs or edit the evaluator during the run.

Sequence: B/32 NewsCLIPpings test + VERITE, then B/16, then L/14. Each task extracts unique images/texts with standard backbone preprocessing and scores the frozen interaction LR and its validation-selected cosine baseline. JSON results appear only after each task finishes. Raw numeric features/predictions stay in ignored artifacts; published reports contain metrics only. Model checkpoint hashes, image hashes and dataset checks guard provenance.

Expected results: `ViT-B-32_newsclippings.json`, `ViT-B-32_verite.json`, equivalent files for B/16 and L/14, and `paired_macro_f1_intervals.json` after every run finishes. Evaluation includes accuracy, macro-F1, class precision/recall, false alarms, confusion matrices, ROC-AUC/AP and 2,000 grouped-bootstrap intervals. VERITE true-vs-OOC and all-misleading tasks remain separate. Truncated VERITE caption counts are disclosed; test captions are not silently truncated.

These are follow-up evaluations because original test outcomes were previously known. Pairwise intervals are exploratory and not multiplicity-adjusted. No new model or threshold selection and no app replacement occurs. Cross-backbone representation ablations beyond the selected interaction heads are not scored by this evaluator; their validation candidate tables remain in the development reports.
