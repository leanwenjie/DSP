# Self-consistent CLIP B/32 baseline — 2026-10-27

Follow-up experiment after previously observing cached-feature test results. Do not describe subsequent evaluation as untouched-test research.

Regenerate train/validation images and captions with the standard OpenAI CLIP ViT-B/32 preprocessing used by the current app: RGB conversion, aspect-preserving bicubic resize, centre crop, CLIP normalisation; tokenize without silent truncation. Frozen pretrained checkpoint; float32 unit-normalised vectors. Extract unique IDs and align to original pair annotations. No test or VERITE data is used in this development script.

Reuse the existing LR comparison: concatenation and interaction features, C in {0.001,0.01,0.1}, train-only StandardScaler, lbfgs max_iter=1000/tol=1e-5. Select feature mode, C and threshold by validation macro-F1, existing deterministic threshold tie rule. Report every candidate, confusion matrix, precision/recall/F1 and accuracy. Include the freshly extracted cosine baseline with validation-selected threshold. These are development scores, not evidence of improved held-out accuracy.

After selection, save portable numeric parameters and input/protocol hashes under artifacts/fresh_clip and reports/fresh_clip. Require saved inference parity with sklearn. Do not replace app weights or original frozen artifacts. Separate frozen test/external evaluation is a subsequent step after confirming complete extraction and validation results. Do not change this protocol in response to results.

Extraction resumes by modality/split at completed batch boundaries; float32 .npy buffers and completed counts are written before proceeding. A process lock prevents concurrent writers. Local per-image acquisition checksums are verified when each batch is read. Immutable provenance guards resume. Run `.venv/bin/python scripts/train_fresh_clip.py`; monitor artifacts/fresh_clip/progress.json and /tmp/fresh_clip.log.
