# Project routes: reference-paper and scope review

Historical feasibility review. The project retains its original title. This historical review preceded the adoption of Shalabi et al. (2023) as the main reference; see [current alignment](../../docs/REFERENCE_ALIGNMENT.md). VERITE images are now available and its evaluation is complete; routes proposing a different title were not adopted. The options and availability notes below record the earlier investigation, not current status. Existing results and reserved test sets are preserved. Author-maintained repositories, citation blocks, dataset loaders and selected implementation files were inspected. This is not a complete independent review of each paper's experiments.

## 1. Keep contextual misinformation detection; add evidence

Reference: Papadopoulos et al. (2025), **RED-DOT: Multimodal Fact-checking via Relevant Evidence Detection**, IEEE Transactions on Computational Social Systems.

- Paper: https://ieeexplore.ieee.org/abstract/document/10948326
- Official repository: https://github.com/stevejpapad/relevant-evidence-detection
- Data: NewsCLIPpings plus retrieved evidence, and VERITE.
- The official main script includes baseline runs with 0, 1, 2, and 4 pieces of evidence, modality-fusion ablations, and several transformer variants. This is a closer methodological reference for our fusion experiments than COVE's large-model pipeline, but our LR/MLP are still adaptations, not reproductions.
- The VERITE evidence CSV is accessible. NewsCLIPpings train/validation evidence coverage and feature alignment remain unverified. The workflow still depends on VisualNews for feature extraction.
- Free cached evidence could support a bounded benchmark experiment. It does not provide retrieval for arbitrary new uploads. Use genuine retrieved evidence only, never true captions or verified context as predictor input. Do not tune on VERITE or 5Pils-OOC.
- A stronger method is plausible, not demonstrated. Do not copy published scores into our comparison or assume 6 GB feasibility without profiling the selected implementation.

## 2. Keep current data; narrow the claim to original-pair consistency

Possible title: **News Image–Caption Consistency Assessment Using CLIP-Based Multimodal Learning**.

- Define consistency operationally as original versus generated mismatched pairs on NewsCLIPpings.
- Keep the existing embeddings, labels, classifiers, analyses and app plan. RED-DOT (2025) could be the main reference for a carefully described evidence-free adaptation; COVE remains related work.
- Objectives: compare validation-selected models on a held-out test; quantify precision/recall trade-offs and hard-case errors; evaluate upload inference and latency after encoder compatibility is checked.
- This is the smallest project change. It does not raise accuracy, solve date/location verification, or transform pairing labels into human visual-compatibility labels. Original VisualNews image availability remains an issue.

## 3. Genuine visual consistency pivot: ARO and NegCLIP

Reference: Yuksekgonul, Bianchi, Kalluri, Jurafsky, and Zou (2023), **When and why Vision-Language Models behave like Bags-of-Words, and what to do about it?**, ICLR 2023.

- Paper: https://openreview.net/forum?id=KRLUvxh8uaX
- Official repository: https://github.com/mertyg/vision-language-models-are-bows
- Main evaluation data: ARO's Visual Genome Relation and Attribution subsets, with cropped images and correct/perturbed captions. This meets the requirement to use data from the main reference paper.
- Official annotations downloaded and parsed: 23,937 relation records (5,316 distinct images) and 28,748 attribution records (4,517 distinct images). Counts describe the downloaded files, not claims about every paper reporting/filtering convention. No models were scored on them. SHA256 hashes and source URLs are in aro_annotation_audit.json.
- The official loader applies bounding-box crops. Those crops must be preserved. After the normal Google Drive large-file confirmation flow, both downloads returned HTTP 206 with ZIP-format leading bytes: the image archive is 908,654,022 bytes and the author-linked NegCLIP checkpoint is 1,815,619,309 bytes. Only 32-byte range probes were read. Full download, archive safety/integrity checks, checkpoint loading and memory profiling remain pending.
- Compare frozen CLIP ViT-B/32 with the authors' released NegCLIP ViT-B/32 checkpoint. Optionally add a BLIP matching model after memory/compatibility checks. Do not retrain NegCLIP with the authors' batch size 256 on a 6 GB laptop.
- Primary metric: rank the correct caption above the perturbed caption; report pairwise accuracy, relation/attribute breakdowns, uncertainty grouped by source image, and latency. The official two-caption implementation counts tied scores as incorrect. Do not train or tune on these evaluation sets; use separate development data for any new classifier or threshold.
- Existing NewsCLIPpings classification results become preliminary/background experiments. They are not comparable numerically with ARO pairwise accuracy. Existing trained heads cannot be assumed suitable for the new task or NegCLIP's feature space.
- App: accept an uploaded image and caption and display an alignment score; support an optional alternative caption for comparative ranking. ARO's forced-choice results do not establish single-pair binary accuracy or calibrated confidence. A binary consistent/inconsistent claim needs separate labelled development/evaluation examples.
- This is the best conceptual match if the user wants visible object/attribute/relation consistency rather than news truth verification. It is not a promise of higher accuracy.

Possible title: **Assessing Image–Caption Semantic Consistency Using Vision-Language Models**.

Measurable objectives: (1) compare CLIP and NegCLIP on ARO relation/attribute pairwise accuracy; (2) quantify performance by relation/attribute and analyse failure modes; (3) build an upload app and measure functional correctness and inference latency. External date/location verification is explicitly outside scope.

## 4. VERITE as a recent reference/benchmark

Reference: Papadopoulos et al. (2024), **VERITE: a Robust benchmark for multimodal misinformation detection accounting for unimodal bias**, International Journal of Multimedia Information Retrieval.

- Paper: https://link.springer.com/article/10.1007/s13735-023-00312-6
- Official repository: https://github.com/stevejpapad/image-text-verification
- The repository describes 1,000 pairs: 338 truthful, 338 miscaptioned and 324 out-of-context. It provides CLIP ViT-L/14 features and supports binary or three-class studies.
- Its July 2026 update states that raw images are on Hugging Face and require academic/research institutional-email verification: https://huggingface.co/datasets/stefpapad/VERITE . No access request has been submitted.
- Useful for separate error categories and unimodal-bias analysis, but changing benchmark does not remove contextual difficulty. ViT-L/14 features are not interchangeable with our ViT-B/32 features. Do not use this external benchmark to tune the present model.

## Recommendation

If retaining the original misinformation title is essential, investigate a limited RED-DOT-inspired evidence experiment, contingent on training/validation evidence access. If minimal rework is the priority, choose route 2 and make the pairing-label definition explicit. If the priority is genuinely visible semantic consistency, route 3 is the best fit: first validate images/checkpoint access and run a frozen-model feasibility check, then finalise the title and app claims. No project files or experimental labels have been redefined to make results look better.
