# Image preprocessing diagnosis — 2026-10-27

Six configurations were fixed before scoring on the same 64 validation images (seed 42). No test predictions or classifier labels were used. The numerical-agreement criterion was cosine >0.999 for every sampled image. This is not a detection-accuracy experiment.

| Preprocessing | Mean cosine agreement | Minimum | Criterion passed |
| --- | ---: | ---: | --- |
| standard_bicubic | 0.982078 | 0.856282 | False |
| short_edge_224_bilinear | 0.977732 | 0.848521 | False |
| square_256_crop_224_unnormalised | 0.823321 | 0.650170 | False |
| square_256_crop_224_normalised | 0.914440 | 0.740289 | False |
| square_224_normalised | 0.909767 | 0.776885 | False |
| short_edge_256_crop_224_normalised | 0.936446 | 0.790378 | False |

## Finding

None of the six variants reproduces the cached image features to the criterion. Standard OpenAI preprocessing remains the closest by both mean and minimum agreement among tested variants. The square 256 resize/224 crop settings from the official training repository worsen agreement, both with and without CLIP normalisation. Thus this training config is not established as the provenance of the released cached embeddings. The official dataset README identifies B/32 but does not resolve the extraction details inspected here.

Possible causes include a different original image rendition, extraction-time transforms, encoder/runtime differences or other undocumented preprocessing. This experiment does not determine which cause applies. Do not claim that the downloaded files are corrupt or choose new app preprocessing solely because it improves a classifier score.

## Consequences and next step

Existing cached-feature NewsCLIPpings results remain valid for those recorded inputs. Fresh-image app and VERITE inference has an unresolved feature-distribution difference from training. No app, model, threshold, checkpoint or frozen evaluation file was changed.

The reproducible way forward is a separately versioned experiment: extract train and validation image/text features with one documented standard OpenAI pipeline, retrain the selected lightweight baseline, choose thresholds on validation, and then freeze it before evaluating fresh test/VERITE features. Preserve existing results as the cached-feature baseline. This removes dependence on reconstructing undocumented extraction, but does not guarantee higher accuracy. Any new evaluation is a follow-up because test outcomes are already known.

First establish a self-consistent B/32 baseline. A later L/14 experiment must use the same complete extraction and evaluation discipline. The downloaded images now make that possible, subject to CPU runtime/GPU memory profiling. No full feature regeneration or new classifier training has been started by this diagnostic.

## Reproduce

`.venv/bin/python scripts/diagnose_clip_preprocessing.py`

Inputs, per-image results and full transforms are recorded in `preprocessing_diagnostic.json`. Source inspected: https://github.com/g-luo/news_clippings_training/blob/master/configs/models/clip.yaml and https://github.com/g-luo/news_clippings/blob/master/README.md .
