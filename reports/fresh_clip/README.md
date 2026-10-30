# Fresh standard-CLIP baseline

Started 2026-10-27. See [fixed protocol](PROTOCOL.md). Current completion status is recorded in local `artifacts/fresh_clip/progress.json`; this README is not a live status display.

The pipeline regenerates 65,819 unique image embeddings and 39,048 unique caption embeddings across NewsCLIPpings train/validation using the app's standard OpenAI CLIP B/32 preprocessing. It then automatically trains concatenation/interaction LR with three C values each, selecting by validation macro-F1. Extraction is CPU-based, float32, batched and resumable, with per-image hash checks and provenance guards. It writes separate artifacts and never overwrites the original models or modifies the app.

Run or resume: `.venv/bin/python scripts/train_fresh_clip.py`

Monitor: `tail -f /tmp/fresh_clip.log` or read `artifacts/fresh_clip/progress.json`. Do not launch a second copy while running; a lock prevents concurrent writers. The saved feature arrays are only valid through their checkpointed rows until extraction completes.

At completion, `metrics.json` records all six candidates, fresh cosine baseline, validation selection, provenance and the selected model hash. `artifacts/fresh_clip/best_model.npz` is the candidate model, not an approved app replacement. Validation metrics alone do not establish better generalisation. No test/VERITE evaluation occurs in this script. Frozen evaluation and possible app integration follow a review of extraction and training results.

## Completed development and audit

All training/validation extraction and six LR candidates completed. Feature coverage, unit norms, saved-inference parity and metric reconciliation passed; `audit.json` records checks and `frozen_manifest.json` freezes the selected candidate before future evaluation.

| Features | C | Validation accuracy | Validation macro-F1 | Misleading recall |
| --- | ---: | ---: | ---: | ---: |
| concat | 0.001 | 53.50% | 53.37% | 48.09% |
| concat | 0.01 | 53.45% | 53.32% | 48.18% |
| concat | 0.1 | 53.42% | 53.25% | 47.41% |
| interaction | 0.001 | 68.76% | 68.76% | 67.14% |
| interaction | 0.01 | 68.51% | 68.51% | 68.31% |
| interaction | 0.1 | 68.49% | 68.49% | 68.48% |

Selected: interaction LR, C=0.001. Fresh cosine baseline validation macro-F1: 65.45%. All numbers above are validation-selected development scores. Fresh test/VERITE evaluation and app integration remain pending.
