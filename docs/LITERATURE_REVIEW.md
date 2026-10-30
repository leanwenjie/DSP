# Literature review and improvement priorities

Current objectives and experiment status: [OBJECTIVES_AND_PLAN.md](OBJECTIVES_AND_PLAN.md). The review below records its review-time findings; acquisition and fresh B/32 development have since completed.

Reviewed 2026-10-24. Focused primary-source review, not an exhaustive systematic review. Main reference remains Shalabi et al. (2023). Published numbers are not interchangeable across encoders, inputs, splits, metrics or model-selection protocols. No new models were trained for this review.

## 1. Shalabi et al. — Image-Text Out-Of-Context Detection Using Synthetic Multimodal Misinformation

APSIPA ASC 2023. [Publication](https://doi.org/10.1109/APSIPAASC58517.2023.10317336), [accessible manuscript](https://arxiv.org/abs/2403.08783), Sections III–IV, Tables III–V.

Uses NewsCLIPpings Merged/Balanced (71,072/7,024/7,264), CLIP ViT-B/32, generated captions/images, Sentence-BERT and a separate vision Transformer. Best generation-assisted accuracy is 68%; CLIP-only feature-map baselines are also reported. It motivates our lightweight alternative, but a numerical advantage over its rounded published score is not a controlled superiority result. The paper acknowledges temporal/contextual ambiguity; our method does not solve it either.

Action: retain our explicit interaction-feature ablation. Our concatenation MLP reached its 30-epoch budget in every seed. Its lower score supports a fixed-budget result, not an optimized-model comparison. If extending training, predefine the extension for both feature modes using development data, never extend because of test performance.

## 2. Papadopoulos et al. — Similarity over Factuality: Are we Making Progress on Multimodal Out-of-Context Misinformation Detection?

WACV 2025. [Final published paper](https://openaccess.thecvf.com/content/WACV2025/html/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.html), [code](https://github.com/stevejpapad/outcontext-misinfo-progress). Reviewed final paper Sections 4–6, Tables 2–4 (table numbering differs from the earlier arXiv version).

MUSE uses similarity measures with simple classifiers; AITR adds a Transformer. Full models use external evidence. Crucially, Table 3 also reports a single image–caption-similarity MLP without evidence: 80.69% NewsCLIPpings accuracy and 70.89% VERITE true-vs-OOC accuracy. The default encoder is CLIP ViT-L/14, not our B/32; the published high score must not be attributed to classifier design alone. Table 2 uses a special VERITE OOD-cross-validation selection protocol, unlike our frozen transfer evaluation. Do not merge these tables into a single comparable ranking.

The authors warn that similarity and retrieval shortcuts can explain strong scores without factual reasoning. This supports our hard-pair analysis and argues against claiming state of the art based only on comparison with the 2023 paper.

Actions: (a) test a small nonlinear classifier on the existing scalar cosine feature as a cheap diagnostic, retaining the threshold baseline; (b) compare interaction-only versus raw-plus-interaction features to assess reliance on unimodal content; (c) investigate official L/14 feature availability before proposing an encoder upgrade. Raw NewsCLIPpings images are unavailable locally, so L/14 extraction for VERITE alone would not produce a valid matched trained system. Do not promise a jump to 80% from a backbone change.

## 3. Papadopoulos et al. — VERITE: a Robust benchmark for multimodal misinformation detection accounting for unimodal bias

International Journal of Multimedia Information Retrieval, 2024. [Paper](https://doi.org/10.1007/s13735-023-00312-6), [code](https://github.com/stevejpapad/image-text-verification). Sections 3–5 and Tables 5–7.

Examines unimodal bias and introduces modality-balanced evaluation, where images/captions recur in truthful and misleading contexts. Uses CLIP L/14 features, Transformer variants and multiple synthetic training sources including CHASMA and NC-t2t. NC-t2t training differs substantially from our Merged/Balanced set. Its real-world benchmark is small and story-dependent; released counts differ from the paper headline.

Actions: add image-only and text-only baselines under a predefined development protocol to diagnose shortcuts; retain grouped uncertainty estimates and both VERITE label views. Weak unimodal baselines would support the need for paired information but would not prove factual understanding. Do not train/tune on VERITE to improve its already-observed score.

## 4. Gu et al. — Learning Domain-Invariant Features for Out-of-Context News Detection

2024 manuscript, [arXiv 2406.07430](https://arxiv.org/abs/2406.07430), reviewed v2. Sections 3–4 and Table 1.

ConDA-TTA studies domain adaptation using labeled source data, unlabeled target data, contrastive alignment and test-time adaptation. It evaluates Twitter-COMMs and NewsCLIPpings with domain-based protocols. These results are not directly comparable to ordinary frozen Merged/Balanced classification; access to unlabeled target examples changes the setting.

Action: begin with descriptive per-source/topic error analysis only if trustworthy metadata can be recovered. A later source-held-out experiment could test robustness, but it needs a separately defined split and adequate development data. Do not silently apply test-time adaptation to the frozen benchmark and describe it as the original protocol.

## 5. Tonglet, Thiem and Gurevych — COVE: COntext and VEracity prediction for out-of-context images

NAACL 2025. [Paper and publication record](https://aclanthology.org/2025.naacl-long.102/). Reviewed main method, limitations and implementation appendix.

Separates image context recovery from caption veracity prediction. Uses retrieved web evidence and components including Llama 3, LlavaNext and DeBERTa; the appendix describes Google Vision API use. It addresses the actual missing-context problem more directly than embedding compatibility. Its limitations include unreliable retrieved captions and context errors. Full reproduction does not fit our current evidence-free data and assured-free resource setup.

Action: treat evidence-assisted context verification as a separate future extension. A feasible first step would require an accessible evidence collection and evidence-source audit. Adding an LLM without evidence is not a substitute for recovering the true date/location, and generated explanations must not be presented as factual evidence.

## Proposed bounded improvement phase

These are proposals, not completed experiments or new claimed contributions.

1. **Low-cost development diagnostics:** scalar-cosine nonlinear baseline; image-only and text-only logistic regression; interaction-only logistic regression using abs difference, product and cosine. Reuse cached B/32 embeddings and existing dependencies. This asks which signals help, before adding model size.
2. **Training-budget sensitivity:** assess longer fixed budgets for both MLP feature modes with validation-only stopping. Preserve the original fixed-budget result and weights. Increased parameters and optimization affect comparisons; do not claim pure feature causality.
3. **Potentially larger improvement:** investigate licensed/access-authorized L/14 embeddings for all matching splits. Confirm IDs, normalization, preprocessing and provenance before training. A 6 GB GPU alone does not solve missing-image access.
4. **Evidence extension:** only after evidence access and costs are resolved; maintain a distinct evidence-assisted protocol and app claim.

Freeze candidate settings, development selection criteria and final evaluation plan before executing a new phase. Rank candidates on validation macro-F1, report recall/FPR trade-offs and select one candidate without inspecting additional test outcomes. Existing NewsCLIPpings test and VERITE results are already known: subsequent results remain exploratory follow-ups. A genuinely confirmatory claim requires a separately reserved evaluation set. The partially available, unscored 5Pils collection may be investigated for this purpose, but its task compatibility and missing-image bias must be checked before designation.

## Contribution wording after review

Our current evidence supports a reproducible lightweight implementation, a matched-budget feature comparison, and transparent transfer/error analysis. It does not support novel invention of interaction features, general state-of-the-art performance, comprehensive fact-checking, or verified superiority over all published systems. The main reference is a methodological starting point, not a ceiling on the literature comparison.
