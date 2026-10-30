# Backbone feasibility results

All three pretrained OpenAI CLIP checkpoints completed eight-sample CPU inference without nonfinite outputs. This is a feasibility profile, not a classifier comparison or a laptop-GPU test. See [fixed protocol](PROTOCOL.md).

| Backbone | Embedding dimension | Image forward median | Text forward median | Peak process RAM |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/32 | 512 | 0.091 s | 0.075 s | 1.61 GiB |
| ViT-B/16 | 512 | 0.228 s | 0.080 s | 1.60 GiB |
| ViT-L/14 | 768 | 0.959 s | 0.106 s | 3.78 GiB |

CPU float32, two Torch threads, batch size 1, one warm-up per modality. Peak process memory includes checkpoint loading and Python/metadata overhead; it is not VRAM. Median times exclude preprocessing, tokenization, download and model loading. Full timing samples and p95 are in the JSON files. These short runs do not establish sustained batch throughput or predict exact full-extraction duration.

All candidates are feasible for small-batch inference in this cloud environment. RTX4050 6GB feasibility remains untested. Larger models are not assumed more accurate. B/16 full train/validation extraction has been launched; L/14 is queued to start only after B/16 completes successfully. The next step is separately versioned train/validation extraction with model-specific dimensions, then the predefined LR comparison; preserve the completed/frozen B/32 pipeline.

## Full development runs

Use `scripts/train_clip_backbones.py ViT-B/16` or `ViT-L/14`. Runs use CPU float32 batches of eight, two threads, checkpointed features and process locks. Each runs the six predefined LR candidates automatically after extraction. Both concat and interaction winners are retained, as well as the overall validation winner. Progress lives in `artifacts/backbones/ViT-B-16/progress.json` and `artifacts/backbones/ViT-L-14/progress.json`. Logs are `/tmp/clip_b16_training.log` and `/tmp/clip_l14_training.log`. A sequential shell starts L/14 only if B/16 succeeds; if B/16 fails, inspect and resume it before launching L/14. A queued model does not yet have a progress file.

No test/VERITE scoring or app replacement occurs during these runs. Larger backbones may take many CPU hours. Do not edit the hashed protocol, trainer or inference helpers during an active run; resume verifies provenance. Historical B/32 trainer is deliberately preserved rather than refactored because its frozen extraction manifest hashes that source.
