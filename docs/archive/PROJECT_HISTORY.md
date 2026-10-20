> Historical project overview before the file-organisation pass. Current status is in [the root README](../../README.md). Relative links have been adjusted for this location.

# Multimodal Out-of-Context Misinformation Detection

Bachelor's project investigating image–caption mismatch detection with vision-language models.

## Project constraints

- Main references published in 2023–2026; no originality requirement.
- Free software and no required paid APIs.
- Development hardware: user's NVIDIA GeForce RTX 4050 Laptop GPU, confirmed 6141 MiB (approximately 6 GB) VRAM. This is separate from the cloud machine.
- Streamlit app must accept uploaded images and user-entered captions.
- Project duration: 13 weeks; calendar start/submission dates not specified.

## Research reference

Papadopoulos et al. (2025). *RED-DOT: Multimodal Fact-checking via Relevant Evidence Detection*. IEEE Transactions on Computational Social Systems.

- Paper: https://ieeexplore.ieee.org/abstract/document/10948326
- Official implementation: https://github.com/stevejpapad/relevant-evidence-detection
- Shared datasets: NewsCLIPpings and VERITE.

This is a smaller comparative study of evidence-free CLIP similarity, logistic regression and MLP classifiers. RED-DOT provides the methodological reference for multimodal feature combination and the distinction between evidence-free and evidence-assisted detection. We do not reproduce its transformer architecture, evidence retrieval, or published scores. The title remains **Multimodal Out-of-Context Misinformation Detection using Vision-Language Models**.

Supporting references: [COVE (NAACL 2025)](https://aclanthology.org/2025.naacl-long.102/) motivates contextual verification and the optional 5Pils-OOC benchmark; [VERITE (2024)](https://link.springer.com/article/10.1007/s13735-023-00312-6) defines the external dataset. This supersedes the earlier provisional selection of COVE as the main reference and the unadopted title-change proposals in the historical scope review.

## Required models

1. Frozen CLIP ViT-B/32: cosine similarity with a threshold selected on validation data.
2. Frozen CLIP ViT-B/32 plus logistic regression on combined image/text features.
3. Frozen CLIP ViT-B/32 plus a small MLP on combined features and interactions.

Compare concatenation against concatenation plus absolute differences, elementwise products, and cosine similarity. Fit transformations on training data only. No large-language-model training is required. Extract embeddings in batches appropriate to available VRAM and train small classifiers on cached features.

An evidence-assisted extension is conditional on time, evidence coverage, and hardware. Results using supplied benchmark evidence must not be presented as arbitrary-upload retrieval capability.

## Datasets

- NewsCLIPpings `merged_balanced`: official train, validation, and test partitions; original images/captions from VisualNews.
- VERITE: core external evaluation; 1,001 released pairs covered by 663 images. Primary true-versus-out-of-context task; supplementary true-versus-all-misleading task. No tuning on VERITE.
- 5Pils-OOC: optional additional external benchmark from COVE; 624 images with one accurate and one misleading caption each. Available processed coverage is 618 images; evaluate only with disclosed exclusions and image review. Never tune on it.

Validate actual counts, image availability, feature mappings, label definitions, duplicates, and split overlap before experiments. Report exclusions. Do not use IDs, row order, generation scores, or labels as predictive inputs. Preserve image-associated pairs during uncertainty estimation.

## Measurable objectives

1. Compare models using test macro-F1 (primary), accuracy, out-of-context precision/recall/F1, and confusion matrices.
2. Measure feature-fusion effects and the performance change between NewsCLIPpings and VERITE (with 5Pils-OOC as an optional extension); report paired/grouped bootstrap confidence intervals and multiple training seeds for stochastic models.
3. Build and evaluate upload-and-caption inference, including invalid-input handling, median/p95 latency, and a documented set of correct and incorrect cases.

Do not promise a particular accuracy or improvement before experiments. Keep final test sets untouched until model and threshold choices are fixed.

## App behaviour

Accept an image and caption, show the submitted inputs, a mismatch-risk prediction, and a clearly labelled model score. Evaluate calibration before presenting scores as probabilities. Support CPU inference if necessary.

The system estimates image–caption compatibility; it cannot establish factual truth, identity, location, or date from similarity alone. New uploads can differ from benchmark data. Do not fabricate factual explanations. No paid reverse-image search or external LLM API is required.

## Report and visual story

Use CRISP-DM to organise introduction/problem statement, objectives, methodology, modelling, tools, app demonstration, and visual analysis. Plot label/data availability, similarity distributions, model performance with uncertainty, confusion matrices, synthetic versus real-world performance, and representative mistakes.

## Milestones

| Weeks | Work | Deliverable |
| --- | --- | --- |
| 1–2 | Literature review, proposal, dataset/model access and storage checks | Approved scope and data inventory |
| 3–4 | Data preparation, split audit, exploratory analysis | Reproducible loaders and initial figures |
| 5–6 | CLIP similarity and logistic regression baselines | Validation results and reproducible experiments |
| 7–8 | MLP training, ablations, validation-only model selection | Frozen model configurations and thresholds |
| 9–10 | Final evaluation on both test datasets, uncertainty and error analysis | Final metrics, figures, and limitations |
| 11 | Streamlit upload app and inference integration | Working local demonstration |
| 12 | App tests, latency measurements, report completion | Evaluated app and report draft |
| 13 | Final review, presentation, demo rehearsal and contingency | Submission package |

Write report sections throughout the project. Evidence-assisted reasoning is optional and must not displace required evaluation or app delivery. Confirm image access early; a dataset download problem must be resolved before committing to full evaluation. Local GPU memory is confirmed as 6141 MiB. Start CLIP embedding extraction with batch size 8 in evaluation/inference mode, cache embeddings to disk, and increase only after measuring memory usage. Reduce the batch size if other processes consume GPU memory. Train classifiers from cached embeddings; full CLIP fine-tuning and an 8B language model are not required. Actual runtime and peak memory still need measurement on the user's laptop.

## Initial verification

Data preparation is implemented; see [the data status and reproduction instructions](../../reports/DATA_STATUS.md). NewsCLIPpings annotations and all six CLIP embedding files have been downloaded and verified. Every required vector passed coverage, dimension, and finite-value checks. No cross-split VisualNews ID overlap was found.

5Pils-OOC has 1,248 verified labelled pairs. Of 624 images, 620 downloaded and 618 passed published preprocessing plus dimension checks. Four raw URLs failed and two crop coordinates exceeded current image bounds. Keep both captions of each excluded image out of any reduced-coverage evaluation, and report that evaluation separately from the full benchmark. Original VisualNews images remain unavailable from the documented homepage (404); official NewsCLIPpings validation captions were subsequently recovered for encoder checks.

This cloud machine has no `nvidia-smi`; the user's RTX 4050 is a separate resource. The [CLIP similarity baseline](../../reports/baseline/README.md) achieves validation macro-F1 0.6458. The [logistic-regression comparison](../../reports/logistic/README.md) selects interaction features with validation macro-F1 0.6823 and accuracy 0.6825. These are development results, not final test scores; the subsequent NewsCLIPpings test evaluation is linked below; 5Pils-OOC remains unscored. Reloaded numeric classifier parameters reproduce all validation predictions.

The [three-seed MLP study](../../reports/mlp/README.md) achieves mean validation macro-F1 0.6801 (sample SD 0.0030). Preselected seed 42 achieves 0.6826, only about 0.03 percentage points above logistic regression. This study does not establish a reliable advantage for the neural classifier. All three saved models reproduce their validation predictions; the subsequent NewsCLIPpings test evaluation is linked below; 5Pils-OOC remains unscored.

The [similarity-based error analysis](../../reports/similarity_analysis/README.md) uses training-defined bins and existing validation predictions. Logistic regression catches only 35 of 300 misleading pairs in the highest-similarity group, despite 78.23% subgroup accuracy. Low-similarity original pairs also suffer high false-alarm rates. These descriptive results do not establish semantic-compatibility accuracy or the causes of individual errors.

The [80% recall screening analysis](../../reports/thresholds/README.md) finds that global validation recall can rise to 80.01%, but original-pair false-positive rates rise to 46.87% for logistic regression and 46.36% for the MLP. High-similarity misleading recall improves to 28.67% and 33.33%, respectively. These are exploratory operating points; the original model thresholds remain unchanged and the subsequent NewsCLIPpings test evaluation is linked below; 5Pils-OOC remains unscored.

[VERITE images are now available locally](../../reports/verite/README.md) from the user's split ZIP upload. All 663 referenced images passed validation and cover all 1,001 rows in the current public annotations. The original released CSV and RED-DOT tables also contain these 1,001 pairs; retain them and disclose the documentation’s 1,000-pair discrepancy. The [evaluation protocol](../../reports/verite/EVALUATION_PROTOCOL.md) defines the binary tasks before scoring. Frozen VERITE scoring is now complete, without VERITE tuning; the uploaded images do not require an HF token for local use.

[CLIP encoder compatibility checks](../../reports/encoder/README.md) validate fresh text features against cached NewsCLIPpings features and compare uploaded VERITE images/text against RED-DOT reference vectors. These checks do not measure misinformation-detection accuracy.

The [frozen VERITE evaluation](../../reports/verite/evaluation/README.md) uses all 1,001 released rows. On the prespecified 663-pair true-versus-out-of-context task, macro-F1 is 67.86% for CLIP similarity, 69.00% for logistic regression, and 67.23% for the predesignated MLP seed 42. Logistic regression detects 177/325 out-of-context and 74/338 miscaptioned pairs, with 52/338 false alarms on true pairs. Its all-misleading macro-F1 is 53.59%. These point estimates do not establish statistical superiority. Model weights and thresholds stayed frozen; VERITE is now an evaluated benchmark, not an untouched set for future tuning. All 17 tests passed at that stage.

## Completed core evaluation

The [frozen NewsCLIPpings test evaluation](../../reports/newsclippings_test/README.md) covers all 7,264 test pairs. Macro-F1 is 65.19% for similarity, 69.63% for logistic regression and 68.86% for the predesignated MLP seed 42. Logistic regression improves over similarity by 4.45 percentage points (paired grouped-bootstrap 95% interval: +3.47 to +5.41), with lower OOC recall (66.82% versus 71.23%). All model artifacts and thresholds remained unchanged. This is benchmark-specific evidence, not a claim of arbitrary-upload factual verification.

NewsCLIPpings test and VERITE are now evaluated. 5Pils-OOC is an optional additional evaluation and remains unscored. The core methodological reference is RED-DOT (2025); the current implementation remains an evidence-free adaptation. Nineteen automated tests pass, including shared-ID grouping and bootstrap checks. The upload app and initial cloud CPU latency measurements are now implemented (see below); the final report and presentation remain unfinished. Files are local; no GitHub commit or push has been made.

## Upload application

The [Streamlit app](../../reports/app/README.md) supports image uploads and caption entry, with frozen logistic regression (default), CLIP similarity and MLP seed 42. It shows screening results and uncalibrated model scores, not factual verdicts. Input validation, model-hash checks and CPU inference are implemented. Run from the project directory:

```bash
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Install `requirements-app.txt` in the prepared CPU environment if needed. The retained numeric model artifacts are required; a source-only checkout does not contain them. Twenty-six tests pass. Live inference checks matched saved benchmark predictions, and a small 12-example CPU sample measured approximately 0.11 s median and 0.13 s p95 warm inference. Laptop/GPU timing, browser usability, public deployment and calibration remain unverified. See the app report for instructions and limitations.
