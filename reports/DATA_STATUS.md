# Data preparation status

## Reproduce

Use Python 3.11+ from the repository root:

```bash
python -m pip install -r requirements-data.txt
python scripts/prepare_data.py --embeddings
python scripts/audit_embeddings.py
python scripts/collect_5pils_images.py
python scripts/process_5pils_images.py
```

Use a virtual environment for a new installation. Downloaded data is ignored by Git. No credentials or paid APIs are required. Downloads use HTTPS verification, pinned upstream revisions, atomic file replacement, and SHA256 receipts. Official CLIP pickle files additionally match the publisher's LFS SHA256 before loading. Never load arbitrary pickle files.

## NewsCLIPpings

The selected benchmark is `merged_balanced`: 71,072 train, 7,024 validation, and 7,264 test pairs. Each split is balanced. No duplicate image/caption pairs within a split or shared VisualNews IDs between these splits were found. All six official CLIP ViT-B/32 embedding files are downloaded (approximately 2.2 GB). Every embedding needed for the selected pairs is present, finite, nonzero, and 512-dimensional.

See `data_audit.json` and `embedding_audit.json` for machine-readable results. This is a structural audit, not a model evaluation. Test examples have not been used for model selection.

The VisualNews homepage linked by its official repository currently returns HTTP 404. A community metadata mirror exists at https://huggingface.co/datasets/twelcone/VisualNews but has not been adopted as authoritative. Original VisualNews images remain unavailable; official validation captions were subsequently recovered for encoder checks. Cached official embeddings permit classifier experiments while image acquisition is investigated. End-to-end CLIP preprocessing parity still needs verification before the app is declared ready.

## 5Pils-OOC

The released annotations contain 1,248 pairs across 624 images. Labels are verified by matching `(image_id, caption_id)` to `true_veracity` in COVE's released intermediate evidence table and checking agreement with caption/true-caption equality. There are 624 pairs per class. Never pass `true_caption`, `true_veracity`, or other ground-truth context into a predictor.

The collection script retrieves one raw image per ID, validates image decoding, and records failures. The processing script applies the upstream crop coordinates and standard sidebar-removal algorithm with explicit dimension checks. Raw images must not be substituted for processed benchmark images silently. Live web images are not guaranteed to be identical to historical benchmark files; visual inspection remains necessary. Exclude both captions for any unavailable image and report reduced coverage.

See `5pils_image_audit.json` and `5pils_processing_audit.json` for current availability and failures. These files are overwritten by subsequent runs. Detailed hashes and paths remain under ignored `data/5pils_ooc/`.

The standard-processing algorithm follows UKPLab/5pils at revision `acf672a20900f80e99f2d013aa2f30b7469fa180` (Apache-2.0). Dataset annotations are CC-BY-SA-4.0; preserve attribution and consult source terms before redistributing images. The downloaded dataset license is retained in `data/5pils_ooc/LICENSE`.

## Outstanding work

- Recover the unavailable original images or transparently evaluate the available paired subset.
- Review processed images before external model evaluation.
- Locate a trustworthy VisualNews source and verify inference embeddings against the cached training embeddings.
- Baselines are trained. Frozen NewsCLIPpings test and VERITE evaluations are complete; see [current test results](newsclippings_test/README.md). Keep 5Pils-OOC unscored until its processed images have been reviewed.
