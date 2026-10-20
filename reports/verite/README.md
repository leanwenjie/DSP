# VERITE uploaded-image audit

The three user-supplied parts were reassembled in numeric order into ignored `data/verite/images.zip`, then extracted under `data/verite/images/`. The reconstructed archive is 65,287,817 bytes, matching the size advertised by the Hugging Face repository. Its SHA256 is `e9e3c7237d19ce42e288408461b4d480717bf818b1488c28052d32d60cbc05e5`. Matching size is not an independent publisher-hash verification; no authoritative archive digest was available from the gated metadata.

ZIP CRC validation passed. Extraction checks reject unsafe paths, symlinks, duplicate paths, and conflicting existing files. Every referenced image passed Pillow verification and full decoding.

| Item | Count |
| --- | ---: |
| Annotation pairs | 1,001 |
| True pairs | 338 |
| Miscaptioned pairs | 338 |
| Out-of-context pairs | 325 |
| Distinct referenced images | 663 |
| Valid referenced images | 663 |
| Missing or invalid referenced images | 0 |

One additional archive file, `images/false_179.jpeg`, is not referenced by the downloaded annotations. It is retained but unused. No examples were silently dropped, relabelled, or added.

## Evaluation status

The current public annotation CSV contains 1,001 pairs, while repository documentation describes 1,000 (324 rather than 325 out-of-context pairs). Image availability does not explain this discrepancy: all current rows are covered. The original July 2023 released CSV and RED-DOT tables contain the same 1,001 pairs; see `version_audit.json`. We retain that released file and disclose the documentation discrepancy when comparing published results. No authoritative reason for the differing headline count was established.

At the initial image audit, no misinformation-classifier scoring had occurred. The subsequent [frozen evaluation](evaluation/README.md) is now complete; no model selection or threshold tuning used VERITE. Feature-only encoder checks are documented in [the encoder report](../encoder/README.md). The [prespecified evaluation protocol](EVALUATION_PROTOCOL.md) preserves the original labels and defines primary true-versus-out-of-context and secondary true-versus-all-misleading tasks. The existing classifiers were trained on NewsCLIPpings, not VERITE. These uploads provide images, not a retrieval system or additional training evidence.

The archive and image files remain ignored by Git. Respect the dataset's research-use and no-redistribution terms; do not package them into a public app or repository. A Hugging Face token is no longer needed for this local copy. The previous `access_audit.json` records the earlier gated-download status, not current local availability.

## Reproduce

```bash
.venv/bin/python scripts/prepare_verite_upload.py /path/to/images.zip.001 /path/to/images.zip.002 /path/to/images.zip.003
```

The script sorts numeric part suffixes, checks completeness, preserves conflicting existing files by failing rather than overwriting, verifies archive/image integrity, and writes `image_audit.json`. Detailed image hashes are in ignored `data/verite/image_manifest.json`. The public annotation CSV must already exist at `data/verite/VERITE.csv`.
