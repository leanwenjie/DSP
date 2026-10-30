# Current objectives and methodology plan

User-approved wording, 2026-10-27. This document is the current source of truth for project objectives. It supersedes objective wording in older planning documents. Historical experiment protocols and their hashes remain unchanged. Report/slides drafts remain on hold and may contain superseded wording.

Project title: **Multimodal Out-of-Context Misinformation Detection Using Vision-Language Models**.

## Objectives

1. To compare the effectiveness of selected pretrained vision-language models for out-of-context misinformation detection.
2. To investigate the influence of image–text feature representations on detection performance using machine-learning classifiers.
3. To assess the generalisation and limitations of the developed detection approaches through cross-dataset evaluation and error analysis.
4. To demonstrate an application for screening image–text pairs for potential out-of-context misinformation.

## Objective 1 — Vision-language model comparison

Planned candidates: pretrained CLIP ViT-B/32, ViT-B/16 and ViT-L/14, frozen during feature extraction. These are three backbones within the CLIP family, not three independent model families. B/16 and L/14 remain subject to memory/runtime profiling; neither has been evaluated here. If a candidate is infeasible, document a replacement and revise the experimental protocol before evaluation. A single-backbone experiment cannot fulfil the plural-model objective.

Use identical NewsCLIPpings pair splits, each backbone's documented preprocessing consistently across development and inference, and a separate fitted classifier per backbone. Fix a common downstream logistic-regression procedure and regularisation-search budget before new runs. For the primary backbone comparison hold feature construction fixed (interaction representation); assess alternative representations separately under Objective 2. The backbone size comparison necessarily also changes embedding/input dimensions and classifier parameter counts; report them rather than claiming pure architecture causality.

Primary selection metric: validation macro-F1. Report test accuracy, macro-F1, misleading-pair precision/recall and false-alarm rate after freezing candidates. Also report extraction throughput and inference latency with hardware, batch size and load-time scope. Larger models are not assumed to be better. No performance target is retrospectively set from observed scores.

Current evidence: original cached-feature B/32 experiments completed. Fresh standard-preprocessing B/32 train/validation extraction and six-candidate LR study completed; selected interaction LR C=0.001 has validation macro-F1 68.7560%, accuracy 68.7642%, misleading recall 67.1412%. These are development scores, not held-out improvements. Fresh B/32 feature/inference audit passed and the selected candidate is frozen. Fresh B/32 test and VERITE evaluation are pending. CPU feasibility profiling passed for all three backbones; see reports/backbone_comparison/README.md. B/16 full train/validation extraction is launched, with L/14 queued after successful completion. No new backbone performance results are available yet. No laptop GPU profiling has been performed.

## Objective 2 — Image–text representations

Compare concatenation [v,t] with [v,t,abs(v-t),v*t,cos(v,t)], keeping the backbone, data and classifier training settings fixed. For a d-dimensional backbone the representations have 2d and 4d+1 dimensions. Do not reuse B/32 heads with another backbone. An interaction-only ablation is optional and must be predefined before execution.

Report paired differences in macro-F1 and corresponding accuracy/recall/false alarms; quantify seed variability when training stochastic MLPs. Feature dimensionality changes parameter count, so identify comparisons as matched-training/fixed-budget rather than parameter-count controlled.

Current evidence: cached B/32 MLP comparison completed for seeds 42/43/44. Seed 42 test macro-F1 is 60.35% for concatenation and 68.86% for interactions (+8.51 percentage points; paired grouped-bootstrap 95% interval +7.18 to +9.78). All concatenation runs reached the 30-epoch budget; best achievable performance is not established. This was a follow-up after prior test inspection. Fresh B/32 LR has both feature modes in its validation candidate table. Multi-backbone feature comparisons are pending.

## Objective 3 — Generalisation and limitations

NewsCLIPpings supports development and within-benchmark testing. VERITE supports frozen external evaluation. Retain separate results for 663 true-versus-OOC pairs and all 1,001 pairs; neither is an independently labelled visual-consistency dataset. No VERITE-based model or threshold selection.

Report confusion matrices, class precision/recall, macro-F1, accuracy, false-alarm rate and uncertainty grouped by related content. Analyse high-similarity misleading pairs and low-similarity genuine pairs. Source/topic breakdowns can use the newly available VisualNews metadata, but are descriptive unless a new domain-held-out protocol is predefined. EDA and feature-parity checks are evidence about data quality, not model effectiveness.

Current evidence: cached-model NewsCLIPpings test evaluation, frozen VERITE evaluation and validation error analyses completed. Fresh-image B/32 features differ from released cached features; six tested preprocessing variants did not reproduce them. The new fresh-feature branch addresses consistent extraction but has not yet established external generalisation. New backbone evaluation remains pending.

Both original test datasets have already been examined. Later experiments are follow-ups; a new untouched confirmatory claim requires a separately reserved, suitable evaluation set. Never merge validation-selected scores with test scores in a comparison table without identifying the split.

## Objective 4 — Application demonstration

Demonstrate an image upload, caption entry, a screening prediction, handling of invalid inputs, and measured inference latency. Include a known limitation case. The app's scores are not calibrated factual probabilities; it does not independently establish dates, locations or provenance. Dataset demonstration examples must not be presented as unseen effectiveness evidence.

Current evidence: Streamlit app and CPU checks completed using original frozen B/32 heads. Laptop/Windows usability verification remains pending. The app has not been switched to the fresh-feature candidate. Integrate a new model only after its frozen evaluation, numerical inference checks, packaging and input checks. Supporting multiple backbones in the final app is optional; comparison in the research does not require loading all three in the application.

## Panel evidence map

| Objective | Main question | Evidence |
| --- | --- | --- |
| 1 | Which pretrained backbone is effective under common conditions? | Backbone/data/compute table and frozen performance comparison |
| 2 | Does feature construction influence detection? | Feature diagram and matched representation ablation |
| 3 | Does performance transfer, and where does it fail? | External results, confusion matrices and error analyses |
| 4 | Can the implemented approach be demonstrated? | Working app, valid/invalid inputs, limitation case and latency |

## Sequence of remaining work

1. Audit completed fresh B/32 extraction and training outputs; freeze its selected model for follow-up evaluation.
2. Profile B/16 and L/14 memory and throughput on available hardware before full extraction. Document a fixed comparison protocol and common tuning budget.
3. Extract/train feasible candidates with consistent preprocessing and run predefined feature comparisons using development data only.
4. Freeze candidates; run follow-up benchmark/external evaluation, reporting all planned candidates rather than choosing them on test performance.
5. Integrate the validated application candidate and complete laptop demonstration checks. Write report/slides when the user resumes those deliverables.

Main reference remains Shalabi et al. (2023); WACV 2025 Similarity over Factuality informs the broader backbone comparison and shortcut analysis. See LITERATURE_REVIEW.md for limitations of cross-paper comparisons. No reference-paper reproduction or state-of-the-art improvement is claimed.

## Execution update — 2026-10-29

All three development runs completed: selected validation macro-F1 B/32 68.76%, B/16 77.70%, L/14 82.92%. These are development scores. All three candidates passed full feature/inference/metric audits and were frozen in `reports/backbone_evaluation/frozen_manifest.json`. Follow-up NewsCLIPpings test and VERITE extraction/evaluation has started; consult its progress file for live state. App integration remains pending. This update supersedes earlier pending/running statements above.

## Completion update — 2026-10-30

All three frozen fresh-backbone evaluations and paired intervals completed. Consolidated results: reports/backbone_evaluation/RESULTS.md. L/14, the validation-leading candidate, is integrated as the app default; 1,001 numeric and six raw-input parity checks passed. These updates supersede earlier pending/running statements. Laptop demonstration remains pending; historical report/slides remain on hold.
