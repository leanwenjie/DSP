# Current project file guide

Updated 2026-11-14. Start with the current app and research sections below. Earlier experiments remain reproducible; they are not the current app specification. No model or decision rule changed during cleanup.

## Read in this order

1. `README.md`: project scope, results and navigation.
2. `docs/OBJECTIVES_AND_PLAN.md`: approved objectives and experiment history.
3. `app.py` → `scripts/app_inference.py` → `scripts/model_inference.py`: one upload through the app.
4. `scripts/train_clip_backbones.py`: fresh B/16 and L/14 feature extraction and classifier selection.
5. `reports/backbone_evaluation/RESULTS.md`: original three-backbone evaluation.
6. `reports/unified_screening/README.md`: current default rule and its false-alarm trade-off.

## Current application

| File | Purpose |
|---|---|
| `app.py` | Streamlit upload/caption form and one screening result. |
| `scripts/app_inference.py` | Input validation, hash checks, CPU CLIP encoding, classifier and unified decision. |
| `scripts/model_inference.py` | Shared concatenation/interaction features, training-fitted scaling and numeric classifier inference. |
| `scripts/prepare_data.py` | Shared ROOT/digest helpers imported by inference; its command-line preparation also covers historical datasets. Importing does not run downloads. |
| `.streamlit/config.toml` | UI/server settings and upload limit. |
| `requirements-data.txt` | NumPy and Pillow. |
| `requirements-models.txt` | Data requirements plus scikit-learn and Matplotlib. |
| `requirements-encoder.txt` | Model requirements plus pinned CLIP, PyTorch and tokenizer dependencies. |
| `requirements-app.txt` | Encoder requirements plus Streamlit. These layered files are not redundant. |
| `exports/DSP-app-Windows.zip` | App-only distribution with small trained heads and provenance files. Encoder checkpoints download on first use; no benchmark data. |
| `reports/app/README.md` | Current runtime requirements and limitations. |
| `reports/app/HISTORY.md` | Previous app notes and timings; not current instructions. |

Default: L/14 logistic classifier OR its existing low-cosine-similarity baseline. Other menu options are historical B/32 classifiers/baseline. B/16 is a research comparison, not an app option. The proposed trainable fusion network is not implemented.

## Current research pipeline

| Programs | Purpose |
|---|---|
| `download_visualnews.py`, `audit_visualnews_download.py` | Resumable required-image acquisition and coverage/integrity audit. |
| `prepare_verite_upload.py` | Assemble uploaded archives, safely extract and check VERITE coverage. |
| `visualnews_eda.py` | Dataset summaries, plots, local example gallery and encoder diagnostic. |
| `train_fresh_clip.py`, `audit_fresh_clip.py` | Fresh B/32 features, six LR candidates and saved-model audit. |
| `train_clip_backbones.py` | Fresh B/16 and L/14 extraction, training-only scaling and validation selection. |
| `profile_clip_backbone.py` | CPU inference/memory profile; not a laptop GPU benchmark. |
| `evaluate_clip_backbones.py` | Frozen three-backbone NewsCLIPpings and VERITE evaluation. |
| `report_backbones.py` | Result table/figure from saved evaluations. |
| `evaluate_unified_screening.py` | Fixed OR-rule follow-up, original-score parity, raw-input checks and optional banana diagnostic. |
| `check_l14_app.py` | Original classifier parity plus current unified raw-input decisions; writes under reports/unified_screening. |

Program names in the table are under `scripts/`. Training programs may create or overwrite artifacts; do not rerun them merely to launch the app.

## Historical experiments worth keeping

| Programs | Research evidence |
|---|---|
| `similarity_baseline.py`, `train_logistic.py`, `train_mlp.py` | Initial cached-B/32 baseline, LR and MLP studies. Training modules still re-export helpers imported by tests/evaluators. |
| `audit_embeddings.py` | Official cached-feature integrity and coverage. |
| `evaluate_newsclippings_test.py`, `evaluate_verite.py`, `report_verite.py` | Original frozen test and external evaluation. |
| `evaluate_mlp_concat.py` | Matched MLP representation follow-up. |
| `analyse_similarity.py`, `compare_thresholds.py` | High-similarity failures and recall/false-alarm trade-off. |
| `verify_clip_encoder.py`, `verify_verite_features.py`, `diagnose_clip_preprocessing.py` | Feature compatibility evidence explaining the fresh pipeline. |
| `check_app.py` | Legacy B/32 prediction parity and exploratory timing. |

Related reports remain in `reports/baseline`, `logistic`, `mlp`, `mlp_concat`, `newsclippings_test`, `verite`, `encoder`, `similarity_analysis` and `thresholds`. Historical figures/metrics are evidence, not duplicate files to delete. Preserve hashed scripts and protocols in their existing locations.

## Optional work, retained

`collect_5pils_images.py`, `process_5pils_images.py`, their third-party licence, `data/5pils_ooc/` and two 5Pils audit reports support an incomplete, unscored extension. `prepare_data.py` still expects some 5Pils annotations in its command-line audit. No optional dataset was removed in this cleanup.

`data/research_options/` and `reports/scope_options/` record an unadopted ARO investigation. They are not inputs to the app or completed primary evaluations.

## Data and trained artifacts

- `data/newsclippings/`: annotations and historical official cached embeddings.
- `data/visualnews/`: metadata and 71,966 required images. Full image ledger: `downloaded_images.jsonl`.
- `data/verite/`: 1,001 annotation pairs, images and provenance; archive retained as recovery backup.
- `artifacts/clip/`: official pretrained B/32, B/16 and L/14 checkpoints.
- `artifacts/logistic/`, `artifacts/mlp/`, `artifacts/mlp_concat/`: historical classifier weights and predictions.
- `artifacts/fresh_clip/`, `artifacts/backbones/`: fresh features and selected trained heads.
- `artifacts/backbone_evaluation/`: cached test/external features and predictions.

These are Git-ignored but important. Source-only clones do not contain them. Benchmark data is unnecessary for a new user upload but necessary for reproducing evaluation. All app heads are verified during initialisation; deleting legacy heads can break default-model startup.

## Tests, reports and documentation

`tests/` checks feature arithmetic, threshold behaviour, grouping/bootstrap calculations, input validation, app flow and downloads. Passing software tests does not establish arbitrary-upload accuracy.

`reports/` contains readable Markdown, numeric JSON, figures, and frozen manifests. Each has a different role. `docs/LITERATURE_REVIEW.md` records literature; `docs/REFERENCE_ALIGNMENT.md` includes a frozen historical protocol. Do not silently update old experiment claims to current results.

`docs/FILE_INVENTORY.csv` is a dated source/report/configuration, local-model and export inventory, not a recursive listing of all images, features or installed libraries. Its scope is recorded in `docs/INVENTORY_SCOPE.md`.

## Paused deliverables and cleanup

The outdated report/slides sources, generator, optional requirements and generated submission files are preserved at `exports/paused-deliverables-2026-11-14.zip` (local, Git-ignored). Original relative paths and SHA256 hashes are inside. Extract to a separate directory for review before restoring; the old generator can overwrite reports with older results.

Cleanup removed only unused Playwright browser downloads and regenerable caches. It did not change code, models, datasets, evaluation outputs or app ZIP. See `docs/CLEANUP_2026-11-14.json`; the previous cleanup log is retained. Old guide: `docs/archive/FILE_GUIDE_2026-11-08.md`.

Keep `.git/` and `.venv/`. Python caches may return during tests. The browser cache can be downloaded again if browser automation is needed. `exports/DSP-demo-examples.zip` is a local user-requested demonstration pack, not a benchmark or public source distribution.
