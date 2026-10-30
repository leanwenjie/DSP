# Official VisualNews access check — 2026-10-25

The [official notebook](https://github.com/FuxiaoLiu/VisualNews-Repository/blob/main/VisualNews-Dataset.ipynb) assumes the archive has already been extracted. Its `/p/newscaptioning/...` paths belong to the authors and must be replaced with local paths. It loads `origin/data.json`, then resolves each record's `image_path` and `article_path` relative to the origin directory. No legacy model dependencies are required simply to read JSON, JPEGs and text articles.

The [official download directory](https://www.cs.rice.edu/~vo9/visualnews/) is reachable now. It lists `origin.tar` (~97.9 GB, 91.2 GiB) and `articles.tar.gz` (listed as 4.2G). We verified a 206 response for bytes 0–1048575 of origin.tar, and decoded the first 10 complete JPEG entries. Samples are ignored local files in `data/visualnews/access_sample/`; checksums and dimensions are in `reports/visualnews_access_audit.json`. This establishes actual image access, not complete dataset coverage. No full archive download was attempted.

Notebook example fields: id, caption, topic, source, image_path, article_path. There are no dedicated capture-date or verified-location fields in those examples. Captions/articles may mention dates and places; this is not automatically reliable ground truth for image context.

Next acquisition work: obtain the official data.json; map NewsCLIPpings caption `id` and `image_id` to VisualNews records; locate archive entries and fetch the required subset. HTTP range support permits partial retrieval, but without an archive index locating arbitrary files requires header scanning or an archive download. Do not assume there is a ready random-access image index. Article retrieval needs separate verification.

Recovering the correct image subset would enable image EDA, encoder parity checks and extraction of stronger CLIP features. Re-extraction must cover matching training/validation/test data with fixed preprocessing. Test outcomes already observed remain follow-up research. Raw images are not committed to GitHub.

## Community metadata mirror verified (2026-10-25)

Downloaded `twelcone/VisualNews/data.json` at revision `8a68a1f31d3ed8ac30a86e8e78478f45dd80bc24` (356,272,348 bytes) into ignored `data/visualnews/data.json`. It contains 1,259,732 unique IDs and covers every required NewsCLIPpings train/validation/test ID. All 40,830 records in our existing official validation metadata file match on caption and normalized image path. This is strong consistency evidence, not proof the whole mirror is byte-identical to official metadata.

The selected NewsCLIPpings splits require 59,876 / 5,943 / 6,147 distinct images respectively (71,966 total; existing split-ID audit finds no overlap). Only 4 of the 10 arbitrary archive sample paths appear in this metadata; the reason for the other six is not established. Do not infer complete archive/metadata equivalence from the sample. Retrieve by the required metadata paths and audit actual coverage before feature extraction. No dedicated date/location fields exist in this JSON. Full audit: `reports/visualnews_metadata_audit.json`.

## Resumable subset downloader

`scripts/download_visualnews.py` scans verified tar headers using bounded HTTP ranges and saves only the 71,966 required image paths. It verifies image decoding and records image hashes. Metadata is pinned by SHA256. It refuses unsupported tar entry types and stops with a checkpoint on errors or when less than 3 GiB is free. No full tar is stored. Do not run concurrent copies against the same state.

```bash
.venv/bin/python scripts/download_visualnews.py --max-gib 1
```

The argument limits transferred GiB per invocation (with at most one requested range of overshoot); it is not a retained-image storage limit. Repeating resumes from `data/visualnews/download_state.json`. `downloaded_images.jsonl` lists saved images. Archive scanning may still transfer a substantial fraction of the original 98 GB: HTTP range support is not a pre-existing file index. A finished scan must be checked against required metadata paths before claiming complete coverage. Until then, do not retrain using a source-biased partial image subset.

## Completion 2026-10-27

All 71,966 required images were downloaded and independently verified, with no missing IDs, duplicate IDs, checksum failures or invalid images. This completes the selected NewsCLIPpings subset, not the full VisualNews dataset. See reports/visualnews_download_audit.json.
