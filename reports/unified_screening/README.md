# Unified image–caption screening — 2026-11-08

The default L/14 app now returns one screening decision: flag if the original classifier flags **OR** CLIP cosine similarity is below its unchanged news-validation threshold (0.2376762482060819). This reuses the ready model and its existing similarity baseline; no fine-tuning, new encoder, or threshold fitting. Low compatibility is included within potential-misinformation screening, not shown as a second verdict. Legacy B/32 comparison options retain their original rules.

This is a deliberate recall/false-alarm trade-off, not an across-the-board improvement. The fixed rule was recorded in PROTOCOL.md before follow-up scoring. All test sets were previously examined; results are exploratory, not a newly untouched evaluation. Original classifier metrics, weights and thresholds are preserved.

| Evaluation | Classifier macro-F1 | Unified macro-F1 | Classifier recall | Unified recall | Added detections | Added false alarms |
|---|---:|---:|---:|---:|---:|---:|
| NewsCLIPpings (7,264) | 83.62% | 81.86% | 81.58% | 86.23% | 169 | 295 |
| VERITE true vs OOC (663) | 73.88% | 74.98% | 55.69% | 59.69% | 13 | 8 |
| VERITE all misleading (1,001) | 53.77% | 55.60% | 34.09% | 37.71% | 24 | 8 |

## Banana diagnostic

Used [a public banana image](https://commons.wikimedia.org/wiki/File:Banana-Single.jpg), downloaded from https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg, **not the user's original upload**. Image hash is recorded in metrics.json; the image is not bundled. Four prompts are illustrations, not an independent everyday challenge set or accuracy estimate. A first-person caption can refer to the poster, so flagging it is a request for review, not proof of misinformation.

| Caption | Original classifier | Unified screening |
|---|---|---|
| A yellow banana on a white background. | Not flagged | Not flagged |
| A photograph of a piece of fruit. | Not flagged | Not flagged |
| A person standing in a room. | Flagged | Flagged |
| i am human | Not flagged | Flagged |

## Verification and reproduction

- 32 automated tests passed, including both triggers, disagreements, no-trigger and exact-threshold cases, legacy behaviour, and one-result interface rendering.
- All 8,265 cached NewsCLIPpings/VERITE rows preserved original classifier scores/predictions and reproduced the fixed OR rule.
- Six raw VERITE image/caption checks matched cached classifier scores, cosine values and combined decisions.
- All 43 historical frozen input hashes passed. New results do not overwrite historical reports.
- Updated ZIP includes hash-verified L/14 development metrics needed for the cosine threshold; CRC and internal checksum manifest passed. Package smoke evidence is recorded separately in package_check.json. Windows/laptop execution remains untested.

Run `.venv/bin/python scripts/evaluate_unified_screening.py`; optionally supply `--banana-image /path/to/image.jpg` for the four illustrative prompts. The optional image can differ; its hash must be reported. This command uses retained benchmark data/features, not the source-only ZIP. Run `.venv/bin/python -m unittest discover -s tests -q` for software checks.

The app still cannot establish dates, locations, event identity or factual truth. Similarity can fail on unfamiliar, figurative or indirect captions. A positive result requests human review; a negative result does not confirm truth. Broader real-world reliability remains unestablished. No new everyday-image challenge dataset was created.
