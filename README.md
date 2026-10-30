# Multimodal Out-of-Context Misinformation Detection using Vision-Language Models

Bachelor’s research project comparing selected pretrained vision-language backbones and image–text representations for out-of-context misinformation screening. Completed experiments compare CLIP ViT-B/32, ViT-B/16 and ViT-L/14. Logistic regression and MLP support the representation comparisons. The main reference is [Shalabi et al. (2023), Image-Text Out-Of-Context Detection Using Synthetic Multimodal Misinformation](https://doi.org/10.1109/APSIPAASC58517.2023.10317336). We share its NewsCLIPpings Merged/Balanced split sizes and CLIP ViT-B/32 backbone, but do not reproduce its generation-assisted pipeline. VERITE provides external evaluation; RED-DOT is optional related work. The title is fixed; the planned duration is 13 weeks with free tools and a separate 6 GB RTX 4050 laptop.

## Start here

- [Frozen three-backbone test/external evaluation](reports/backbone_evaluation/README.md)

- [Three-backbone CPU feasibility results](reports/backbone_comparison/README.md)

- [Current approved objectives, methodology and completion status](docs/OBJECTIVES_AND_PLAN.md)

- [Self-consistent CLIP feature regeneration and new baseline](reports/fresh_clip/README.md)

- [Consolidated EDA and image-feature compatibility finding](reports/eda/README.md)

- [Focused literature review and improvement priorities](docs/LITERATURE_REVIEW.md)

- [Main-reference alignment and historical follow-up protocol](docs/REFERENCE_ALIGNMENT.md)

- [Every file explained, storage and cleanup guidance](docs/FILE_GUIDE.md)
- [App installation, demonstration and limitations](reports/app/README.md)
- [Follow-up: matched MLP feature comparison](reports/mlp_concat/README.md)
- [NewsCLIPpings held-out test results](reports/newsclippings_test/README.md)
- [VERITE external evaluation](reports/verite/evaluation/README.md)
- [Earlier planning and experiment history](docs/archive/PROJECT_HISTORY.md) — historical, not the current task list

## Current status

**2026-10-30 update:** all three fresh-backbone evaluations are complete. The app now defaults to the validation-leading L/14 interaction LR. Its numeric route matched all 1,001 VERITE predictions, and six raw-image checks passed. See [final comparison](reports/backbone_evaluation/RESULTS.md).

All 71,966 required images were downloaded and verified. Three fresh-feature extraction/training runs, frozen NewsCLIPpings test evaluation and VERITE evaluation are complete. The current four-objective plan is in [OBJECTIVES_AND_PLAN.md](docs/OBJECTIVES_AND_PLAN.md).

Thirty-one automated tests passed at the latest recorded pipeline check. Laptop/GPU performance is untested. The table below contains completed fresh-feature follow-up results. Historical cached-feature results remain in their original reports.

| Backbone + interaction LR | NewsCLIPpings test macro-F1 | VERITE true-vs-OOC macro-F1 | VERITE all-misleading macro-F1 |
| --- | ---: | ---: | ---: |
| Fresh CLIP B/32 | 69.67% | 67.65% | 52.77% |
| CLIP B/16 | 78.21% | 69.00% | 53.19% |
| CLIP L/14 (app default) | 83.62% | 73.88% | 53.77% |

These tasks differ in data distribution and construction. Read the full reports for recall, false alarms, confidence intervals and the weaker all-misleading VERITE results. Both core test sets are now evaluated: do not tune on them and later describe them as untouched. Model weights, scalers and thresholds remain frozen. 5Pils-OOC is optional and unscored; ARO was an exploratory route that was not adopted.

## Run on your Windows laptop

1. Install **Python 3.12 (64-bit)** from [python.org](https://www.python.org/downloads/) and **Git for Windows** from [git-scm.com](https://git-scm.com/downloads/win).
2. Download [DSP-app-Windows.zip](https://github.com/leanwenjie/DSP/raw/refs/heads/main/exports/DSP-app-Windows.zip) and extract the entire folder.
3. Open `DSP-app`, then double-click `setup_windows.bat`. Wait for installation to complete.
4. Double-click `start_app.bat`. Keep its window open while using the app. If no browser opens, enter `localhost:8501` in your browser.
5. Upload an image, enter a caption, and click **Assess pairing**. The first assessment downloads the official CLIP checkpoint (~890 MiB for default L/14; ~338 MiB for legacy B/32); later runs reuse it.

The app-only ZIP includes the evaluated small classifiers and required code/configuration, but no research datasets or virtual environment. It uses CPU, not CUDA. The extracted package passed a real cloud CPU inference check; the Windows launcher scripts have not been executed on Windows here. See `START_HERE.txt` inside the ZIP for troubleshooting and limitations. For subsequent use, only run `start_app.bat`.

The GitHub **Code → Download ZIP** option downloads the full source repository; it is different from the ready-to-install app-only ZIP above. A plain source checkout needs the ignored model files restored (they are included inside the app-only ZIP).

## Run the app in the prepared research environment

From the prepared project directory:

```bash
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Install `requirements-app.txt` in the prepared CPU environment if needed; see the app guide for fresh installation. Retained frozen model files are required: a source-only checkout is insufficient. The app does not need benchmark images to assess a new upload. It presents screening results and uncalibrated scores, not factual verdicts or independent date/location verification.

## Layout

```text
app.py                  Streamlit user interface
.streamlit/             Interface/server configuration
requirements-*.txt      Layered installation requirements
scripts/                Data, training, inference and evaluation programs
tests/                  Automated correctness checks
reports/                Research results, figures, audits and protocols
docs/                   File guide, inventory, cleanup record and history
exports/                Small laptop-ready app ZIP (no benchmark data)
data/                   Local datasets and download receipts (Git-ignored)
artifacts/              Local weights, features and predictions (Git-ignored)
.venv/                  Installed Python environment (Git-ignored)
```

The paper-inspired follow-up compares concatenation against interaction features under the same MLP training settings. Seed 42 test macro-F1 is 60.35% versus 68.86% (+8.51 percentage points); all seeds and limitations are in the linked follow-up report. This is not an exact reproduction of the reference paper.

## Remaining deliverables

- Final academic report and presentation, using the recorded experimental evidence.
- Browser usability review and the user’s laptop benchmark.
- Optional 5Pils-OOC evaluation after processed-image review and explicit coverage exclusions.

Public hosting, evidence retrieval and calibrated factual probabilities are not implemented. Research datasets and the cloud environment stay local. Source code, reports and a small app-only ZIP can be shared through this repository.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python scripts/check_app.py
```

Do not automatically rerun training scripts: they overwrite model artifacts. Frozen hashes protect app/evaluation compatibility. Do not publish dataset images, credentials, `.venv/`, or ignored artifacts as part of a source-only submission. Preserve a separate local backup of the evaluated model files.
