# CLIP inference and compatibility checks

This setup uses OpenAI's CLIP ViT-B/32 implementation pinned to commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`, torch 2.7.1 and torchvision 0.22.1. The cloud instance uses CPU builds. This does not install or configure the user's separate RTX 4050 laptop.

## Reproduce

From `/workspace/DSP`:

```bash
.venv/bin/python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-cache-dir -r requirements-encoder.txt
.venv/bin/python scripts/verify_clip_encoder.py
.venv/bin/python scripts/verify_verite_features.py
```

The official CLIP downloader verifies the checkpoint SHA256 from its published URL. Cache location is ignored `artifacts/clip/`. The scripts use evaluation mode and inference mode, two CPU threads, official preprocessing and tokenisation, and small batches. Model features are L2-normalised before comparison.

`verify_clip_encoder.py` checks 64 deterministically sampled validation captions against the official NewsCLIPpings cached vectors, using the authors' pinned metadata. This checks feature compatibility without looking at classifier scores or labels. It also runs one synthetic image through preprocessing and the image encoder. A successful synthetic forward pass does not establish historical image-feature parity.

`verify_verite_features.py` compares 32 deterministic examples against RED-DOT's released ViT-B/32 image/text vectors, after confirming annotation order. The reference revision is `4dd44fdc8ff8bfec2acb265019b24de77cc3a09a`. No misinformation classifier is run and no labels are used to choose preprocessing. Agreement with RED-DOT is useful evidence but not independent proof of image parity with the original NewsCLIPpings pipeline.

Read the generated `compatibility.json` and `verite_reference_parity.json` for measured results. Runtime checks, feature agreement, and final label-based evaluation are separate claims. Original VisualNews image-level parity remains unavailable unless actual source images are recovered.

## Measured results

| Feature check | Examples | Minimum cosine | Mean cosine | All > 0.999 |
| --- | ---: | ---: | ---: | --- |
| NewsCLIPpings validation text | 64 | 0.999997 | 0.999999 | Yes |
| VERITE text versus RED-DOT | 32 | 1.000000 (rounded) | 1.000000 (rounded) | Yes |
| VERITE image, standard OpenAI processing | 32 | 0.968000 | 0.993318 | No |
| VERITE image, RED-DOT 400-pixel pre-resize | 32 | 0.975649 | 0.996785 | No |

These are feature-vector similarities, **not detection accuracy**. The sample is deterministic and does not establish equivalence for every example.

The pinned RED-DOT `src/extract_features.py` converts to RGB and resizes the long edge to 400 pixels before its evaluation processor. Reproducing that documented step improves agreement, but residual image differences remain unexplained. Historical image bytes, library versions, or other implementation details could contribute; none has been confirmed as the remaining cause. The check records both paths and uses no class labels or detection scores to choose either.

Keep standard OpenAI preprocessing for the current model inference pipeline; the RED-DOT resize variant is a reference diagnostic, not a change selected on benchmark performance. Text compatibility is supported. Image compatibility with the released RED-DOT vectors is partial, and image parity with NewsCLIPpings training features is unverified. Future scores must disclose these limits and must not be described as an exact RED-DOT reproduction. No VERITE classifier scoring was performed.

The feature checks above preceded the [completed frozen classifier evaluation](../verite/evaluation/README.md); their no-scoring statements describe the checks themselves.

## Update 2026-10-27

Original images are now available. The 64-image validation parity check failed the >0.999 agreement criterion (minimum cosine 0.856282, mean 0.982078). See `visualnews_image_parity.json` and [EDA interpretation](../eda/README.md). Earlier unavailable-image statements describe historical checks. App preprocessing remains unchanged pending diagnosis.

Follow-up: [six-variant preprocessing diagnosis](PREPROCESSING_DIAGNOSIS.md) found no matching variant. Upstream square resizing worsens agreement; it is not a confirmed correction.
