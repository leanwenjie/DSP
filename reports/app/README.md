# Image and Caption Review app

The local Streamlit app accepts a still JPEG, PNG or WebP image and a caption, then runs a frozen CLIP-based classifier. Logistic regression is the default; similarity and the predesignated seed-42 MLP are available for comparison. Models and thresholds are unchanged from the recorded benchmark evaluations.

## Run

From the project directory, using the prepared CPU environment:

```bash
.venv/bin/python -m pip install -r requirements-app.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

For a new CPU environment, create `.venv` and install torch 2.7.1 and torchvision 0.22.1 from the official CPU index first, as documented in `reports/encoder/README.md`. On Windows use the corresponding `.venv\Scripts\python.exe` executable. Windows and RTX 4050 execution have not been tested here. The app currently uses CPU intentionally to match the verified inference path.

Required retained files: `artifacts/logistic/best_model.npz`, `artifacts/mlp/model_seed42.npz`, `reports/baseline/metrics.json`, and `reports/newsclippings_test/frozen_manifest.json`. The official CLIP checkpoint is cached under `artifacts/clip/ViT-B-32.pt`; CLIP downloads and verifies it if absent. Numeric artifacts are ignored by Git, so a source-only checkout is not sufficient. Restore the evaluated artifacts from the prepared workspace; do not retrain silently to replace them. The app checks their frozen hashes before inference. Original benchmark images are not required to assess a new upload.

## Functionality

- Upload an image and enter a caption; choose one of three evaluated models.
- Show the submitted image/caption and either “Flagged for review” or “No mismatch flagged.” Neither is a factual verdict.
- Show the model score, fixed threshold and measured inference time. Scores are not calibrated probabilities.
- Reject missing/corrupt files, unsupported formats, animated images, uploads over 10 MB and images over 20 megapixels.
- Reject captions exceeding CLIP's 75 text-token budget rather than silently truncating them. Character count is not token count.
- Report missing/changed model files or runtime failures without exposing a traceback in the interface.
- Process uploads in server memory without application-level disk saving or external prediction API calls. Streamlit runs on the host machine; users must trust that host. No dataset images are bundled into the interface.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python scripts/check_app.py
.venv/bin/python -m pip check
```

Twenty-six automated tests passed, including invalid-input cases, animation/size/token limits, interface submission, both result states and runtime-error handling. Interface unit tests use a controlled detector; a separate Streamlit AppTest submission exercised the real CLIP and logistic model with an image and caption. The real server returned successful local health and page responses. This is functional verification, not a manual browser usability study.

`check_app.py` checks 36 cached-vector model outputs and 15 raw-input predictions against the frozen VERITE outputs. Twelve predetermined examples supply a small warmed-up CPU timing sample. It also records correct and incorrect cases without using them to tune anything. See `smoke_and_latency.json` for measurements and limitations. These checks do not constitute a new accuracy estimate. Initialisation, upload and validation are excluded from warm latency; concurrency, browser-to-server latency and laptop GPU performance remain unmeasured.

## Demonstration sequence

1. Start the app and explain that it screens image-caption pairings rather than establishing factual truth.
2. Upload an image and submit a short caption with logistic regression.
3. Explain the result and the fixed threshold; do not call the score a percentage confidence.
4. Repeat with another model to illustrate different decision rules; do not choose whichever output agrees with a desired answer.
5. Show a missing input or excessively long caption to demonstrate validation.
6. Discuss one correct and one incorrect benchmark case from the local smoke report. Label these as selected illustrations, not representative accuracy evidence; use the complete benchmark reports for performance claims.

## Remaining limitations

There is no evidence retrieval or independent verification of date, location, identity or event. Negative results can miss misinformation. Exact image-feature parity with the original NewsCLIPpings pipeline remains unverified. The app is a local research demonstration; public hosting, authentication, load testing, score calibration, browser usability testing and the user's laptop benchmark have not been completed.

Measured in this cloud CPU run: initialisation 1.84 s; warm median 0.109 s and p95 0.133 s over 12 single-request logistic-regression inputs. These are exploratory measurements, not performance guarantees.

Inference simplification: shared numeric heads now live in `scripts/model_inference.py`. The app no longer imports training scripts; the laptop ZIP omits train_logistic.py, train_mlp.py and similarity_baseline.py. All 32,100 saved learned-model predictions checked across validation and VERITE are unchanged; 27 tests pass. The extracted bundle also passes real-image checks for all three app models. Frozen hashes and input validation remain intact.
