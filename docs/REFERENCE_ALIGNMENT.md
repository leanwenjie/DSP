# Main reference and controlled follow-up protocol

Recorded 2026-10-24, before training the new concatenation MLP. Existing test outcomes were already known; this is a follow-up study, not a preregistered untouched-test experiment.

## Reference and scope

Fatma Shalabi, Huy H. Nguyen, Hichem Felouat, Ching-Chun Chang and Isao Echizen (2023). *Image-Text Out-Of-Context Detection Using Synthetic Multimodal Misinformation*. APSIPA ASC. https://doi.org/10.1109/APSIPAASC58517.2023.10317336 . Accessible manuscript: https://arxiv.org/abs/2403.08783 (uploaded in 2024).

Section IV specifies NewsCLIPpings Merged/Balanced: 71,072 training, 7,024 validation and 7,264 test pairs, with CLIP ViT-B/32 and 512-dimensional modality embeddings. These match our configuration and counts; identical counts do not establish byte-for-byte preprocessing parity with the authors.

Table IV reports CLIP-only feature-map MLP accuracy 53% and Transformer accuracy 56%. Tables III/V report best generation-assisted accuracy 68%. These are accuracy, not macro-F1. The full method uses generated images/captions, Sentence-BERT and an additional vision Transformer. Our system uses the original pair only. Published rounded scores are contextual comparisons; do not infer statistical superiority from the 69.66% versus 68% difference. No official implementation was located in our search. Exact classifier/feature-preparation reproduction has not been established.

The project asks whether lightweight classifiers on frozen CLIP interaction features provide competitive OOC screening. It does not reproduce synthetic generation or verify arbitrary dates, locations or events. RED-DOT is optional related work, not the methodological anchor. Historical reports and frozen protocols retain their original wording. Academic report/slides are on hold and may still describe the previous reference.

## Measurable objectives

1. Compare cosine similarity, logistic regression and MLP on all 7,264 NewsCLIPpings test pairs using accuracy, macro-F1, precision, recall and false-alarm rate.
2. Measure the contribution of interaction features against concatenated CLIP features under matched MLP settings; report all three seeds and paired grouped-bootstrap macro-F1 differences. Compare published accuracy separately, disclosing implementation differences.
3. Assess frozen-model external generalisation on VERITE, reporting both the 663-pair true-versus-OOC task and the 1,001-pair all-misleading task with category-level errors. VERITE is not used for selection.
4. Demonstrate upload/caption screening and invalid-input handling; report median and p95 inference latency with hardware and measurement scope. Laptop verification remains pending.

## Fixed follow-up experiment

Reuse the existing trainer with `--feature-mode concat`. Concatenate normalized image/text vectors (1,024 dimensions), rather than the existing interaction representation (2,049). Train-only StandardScaler; 32 ReLU hidden units; Adam; alpha .001; learning rate .0003; batch 512; seeds 42/43/44; maximum 30 epochs; patience 5. Epoch and threshold selection maximize validation macro-F1 using the existing tie rule. Seed 42 remains designated; never select seeds by test scores. All settings match the original interaction MLP. Input dimensionality changes parameter count, so this is a matched-training feature ablation, not a parameter-count-controlled architecture experiment.

Do not retrain or overwrite existing interaction models. Save new artifacts under `artifacts/mlp_concat/` and reports under `reports/mlp_concat/`. Hash the selected models and this protocol before test scoring. Evaluate once with frozen validation thresholds on the existing NewsCLIPpings test set; reuse connected VisualNews-ID groups for 2,000 paired bootstrap draws, seed 42. No test-guided adjustments. No app model replacement.

This is a **paper-inspired concatenation baseline**, not reproduction of Table IV's 53% MLP. Existing concatenation logistic regression is useful development evidence, but its historical score alone does not isolate feature effects across classifiers. New VERITE scoring is not necessary for this narrowly defined NewsCLIPpings feature ablation; retain existing external results.

## Reproduction

```bash
.venv/bin/python scripts/train_mlp.py --feature-mode concat
.venv/bin/python scripts/evaluate_mlp_concat.py
```

The new training mode refuses to overwrite existing concatenation artifacts. Original default training remains unchanged in output locations; do not rerun it against frozen models.
