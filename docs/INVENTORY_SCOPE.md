# Inventory scope — 2026-11-14

FILE_INVENTORY.csv lists existing Git-tracked or non-ignored source, tests, documentation, reports and configuration, plus local .npz/.pt model bundles and export files. Paths are relative to the project root and sizes are logical bytes at inventory generation. It excludes itself, .git, .venv, caches, dataset images/annotations, bulk .npy feature arrays and other generated outputs. This is a dated snapshot, not a live complete filesystem listing.

The VisualNews per-image ledger is data/visualnews/downloaded_images.jsonl. Dataset coverage is documented in reports/data_audit.json, reports/visualnews_download_audit.json and reports/verite/image_audit.json. Local data/artifacts/exports may not be available in a Git clone.
