# Image and Caption Review — current app

Current instructions checked 2026-11-14. Historical changes and timing samples are in [HISTORY.md](HISTORY.md); older model results there do not describe the current default rule.

## Behaviour and limits

The default **CLIP L/14 + logistic regression** returns one result: flag if its unchanged classifier flags OR its existing CLIP cosine score is below 0.2376762482060819. Weights and thresholds remain frozen. Other menu options retain the historical B/32 logistic, similarity and seed-42 MLP rules.

| Evaluation | Unified macro-F1 | Unified misleading recall |
|---|---:|---:|
| NewsCLIPpings test | 81.86% | 86.23% |
| VERITE true vs OOC | 74.98% | 59.69% |
| VERITE all misleading | 55.60% | 37.71% |

These are exploratory follow-up results on previously examined benchmarks. The original L/14 classifier had 83.62% NewsCLIPpings macro-F1 and 81.58% recall. The OR rule adds both detections and false alarms; it is not an across-the-board improvement. See [full comparison](../unified_screening/README.md).

The app screens possible misleading pairings, not factual truth. It does not retrieve evidence or verify dates, places or events. Arbitrary everyday-image reliability is unestablished; a matching banana upload was reported falsely flagged. A successful substitute-image test does not resolve that exact failure. Scores are not calibrated truth probabilities.

## Start

From `/workspace/DSP` in the prepared CPU environment:

```bash
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

A fresh installation needs the layered `requirements-app.txt` dependencies and restored trained heads. The app ZIP provides the small heads and runtime files; a source-only checkout does not. See the root README and ZIP's START_HERE.txt for installation. The ZIP includes Windows launchers, but actual Windows and RTX 4050 execution remain unverified. Current inference explicitly uses CPU.

## Runtime files to preserve

- `app.py`, `.streamlit/config.toml`, `scripts/app_inference.py`, `scripts/model_inference.py`, `scripts/prepare_data.py`.
- `artifacts/backbones/ViT-L-14/best_model.npz`.
- `artifacts/logistic/best_model.npz`, `artifacts/mlp/model_seed42.npz`.
- `reports/baseline/metrics.json`, `reports/backbone_comparison/ViT-L-14/metrics.json`.
- `reports/newsclippings_test/frozen_manifest.json`, `reports/backbone_evaluation/frozen_manifest.json`.
- CLIP checkpoints under `artifacts/clip/`, or network access to download official checksum-verified checkpoints. L/14 is about 890 MiB, B/32 about 338 MiB.

All classifier heads are checked at initialisation, even when using the default. One encoder is retained at a time; changing backbone loads another. Benchmark images are not required for new uploads. Processes must be restarted after environment restoration.

## Inputs and privacy

Accepts still JPEG/PNG/WebP files up to 10 MB and 20 megapixels. Rejects invalid images, animations, empty captions and captions beyond CLIP's 75-text-token budget. Long benchmark captions use the recorded evaluation truncation policy, whereas the app rejects them. Uploads are processed in server memory without application-level saving or an external prediction API. Users must trust the hosting server.

The UI shows one flag/no-flag result, analysis time, and an explanation expander. Timing excludes upload, input validation, initial encoder loading and inference-lock waiting.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m pip check
```

For retained research data, `scripts/check_l14_app.py` checks original classifier parity and current unified raw-input decisions. `scripts/evaluate_unified_screening.py` reproduces the full follow-up. Neither is necessary for routine app startup. `scripts/check_app.py` checks legacy B/32 options and can rewrite its historical timing report; do not use it as the default-model benchmark.

Recorded unified verification includes 32 software tests, 8,265 cached prediction checks, six raw VERITE inputs, and an extracted-package real-model form submission. See [metrics](../unified_screening/metrics.json) and [package check](../unified_screening/package_check.json). Interface unit tests use controlled detector outputs; package integration separately used the real model.

Software correctness is distinct from predictive reliability. Browser usability, concurrent-user load, and actual laptop performance remain outstanding. Demonstrations should include a known failure and use complete benchmark results for accuracy claims.
