# Paper-inspired MLP feature ablation

Follow-up experiment after inspecting earlier test results. This is not an exact reproduction of Shalabi et al. Table IV. See [fixed protocol](../../docs/REFERENCE_ALIGNMENT.md).

| Features | Seed | Test accuracy | Test macro-F1 | OOC recall |
| --- | ---: | ---: | ---: | ---: |
| mlp_concat | 42 | 60.35% | 60.35% | 59.33% |
| mlp | 42 | 68.86% | 68.86% | 68.56% |
| mlp_concat | 43 | 60.57% | 60.48% | 55.78% |
| mlp | 43 | 68.75% | 68.75% | 69.33% |
| mlp_concat | 44 | 60.28% | 60.28% | 59.75% |
| mlp | 44 | 68.53% | 68.53% | 68.39% |

| Seed | Interaction minus concat macro-F1 (pp) | Paired 95% interval (pp) |
| --- | ---: | --- |
| 42 | +8.51 | +7.18 to +9.78 |
| 43 | +8.27 | +7.04 to +9.55 |
| 44 | +8.25 | +6.95 to +9.57 |

Intervals resample connected shared VisualNews-ID groups; same draws for both models. They do not include training variability or correct for prior test inspection. Seeds are reported separately. Feature dimensionality and parameter counts differ; all training settings match.

Published context: Shalabi et al. report 53% CLIP-only MLP accuracy, 56% CLIP-only Transformer accuracy and 68% best generation-assisted accuracy. Our baseline is an adaptation with explicit settings, not a replication. These published scores have no paired predictions available here.

Original model files, original evaluation reports and app default are unchanged. Training metrics and curves are in metrics.json and learning_curves.png; full test metrics are in test_metrics.json.

Training-budget limitation: inspect the learning curves and selected epochs in metrics.json. Reaching the 30-epoch limit is not evidence of convergence. These results compare the fixed training budget, not the best achievable performance of each feature representation; do not extend training in response to test outcomes.
