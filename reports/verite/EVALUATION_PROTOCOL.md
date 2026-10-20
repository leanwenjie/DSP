# VERITE evaluation protocol — before classifier scoring

## Dataset version

Use the public 1,001-row VERITE.csv with SHA256 `7f5661860f2e34624c1721710ed37fac5e1e78a4307400530dd587cb0d15a040`, covered by all 663 referenced uploaded images. The original 2023 release and RED-DOT's released annotation/evidence tables contain the same image-caption-label set. RED-DOT's ViT-B/32 feature arrays each have shape (1001, 512). Do not delete a row to match the documentation's 1,000-row headline. Report the exact released-file count and this discrepancy when comparing literature.

## Task definitions

- Primary: distinguish truthful from out-of-context pairs: 338 true + 325 out-of-context = 663 pairs. Label true as 0 and out-of-context as 1.
- Secondary: distinguish truthful from all misleading pairs: 338 true + 338 miscaptioned + 325 out-of-context = 1,001 pairs. Both misleading categories map to 1, but report recall separately for each original category.
- The models are binary. Do not present these results as a three-class classifier or silently equate the primary and secondary tasks.

## Model and threshold policy

Evaluate existing frozen CLIP similarity, selected interaction logistic regression, and the three saved MLP seeds. Retain seed 42 as the predesignated MLP artifact and report the other seeds as variability, not opportunities for test-set selection. Keep all NewsCLIPpings-validation-selected thresholds fixed. Report the exploratory 80%-recall screening operating points separately, if evaluated; do not retune them to reach 80% on VERITE.

Before scoring, record model-artifact hashes and the encoder/preprocessing configuration. Feature compatibility checks may compare regenerated vectors with public reference vectors, but must not use label-based performance to choose preprocessing. Do not fit a scaler, calibration layer, threshold, or model to VERITE.

## Reporting

Report macro-F1, accuracy, per-class precision/recall/F1, confusion counts, and ranking metrics where appropriate. Group confidence-interval resampling by the original story identifier shared by true/miscaptioned/out-of-context examples; row-wise independent resampling would ignore related examples. Keep these intervals distinct from variability across training seeds.

Use actual binary task counts and label prevalence when interpreting accuracy and precision. No statistical superiority or published-paper reproduction claim follows merely from comparing point estimates. Report encoder-parity limitations, the missing original VisualNews images, and the fact that these models do not retrieve contextual evidence.

## Status

No VERITE classifier predictions or label-based model evaluations have been generated at this stage. This file specifies the intended evaluation; it is not a completed results report.
