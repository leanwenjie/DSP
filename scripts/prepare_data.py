"""Download pinned benchmark data; verify hashes and audit labels/split overlap.

Run from any directory. Uses only Python's standard library.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
HF_REV = 'a2bbe9a8125f47a5c26338125c71e05be0266cf7'
COVE_REV = '4f76e18d1c297c7aac9b8ed835b5512b7a0d89e5'
HF = f'https://huggingface.co/g-luo/news-clippings/resolve/{HF_REV}/'
COVE = f'https://raw.githubusercontent.com/UKPLab/naacl2025-cove/{COVE_REV}/'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, relative, expected=None):
    path = DATA / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + '.receipt.json')
    if path.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        actual = digest(path)
        if saved['url'] == url and actual == saved['sha256'] and (not expected or actual == expected):
            return saved
    temporary = path.with_name(path.name + '.part')
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as out:
            if response.status != 200:
                raise ValueError(f'Expected complete HTTP 200 response: {response.status}')
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                out.write(chunk)
        actual = digest(temporary)
        if expected and actual != expected:
            raise ValueError(f'Upstream SHA256 mismatch: {relative}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    saved = dict(url=url, path=relative, bytes=path.stat().st_size, sha256=actual,
                 upstream_sha256=expected)
    receipt.write_text(json.dumps(saved, indent=2) + '\n')
    print(f'Downloaded {relative}: {saved["bytes"]:,} bytes', flush=True)
    return saved


def audit():
    stats, ids = {}, {}
    for split in ['train', 'val', 'test']:
        records = json.loads((DATA / f'newsclippings/{split}.json').read_text())['annotations']
        for row in records:
            if not {'id', 'image_id', 'falsified'} <= row.keys() or type(row['falsified']) is not bool:
                raise ValueError(f'Invalid annotation in {split}')
        pairs = [(r['id'], r['image_id']) for r in records]
        stats[split] = {'records': len(records), 'labels': dict(Counter(str(r['falsified']) for r in records)),
                        'duplicate_pairs': len(pairs) - len(set(pairs))}
        ids[split] = {r[k] for r in records for k in ('id', 'image_id')}
    overlap = {f'{a}/{b}': len(ids[a] & ids[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    ooc = json.loads((DATA / '5pils_ooc/test.json').read_text())
    groups = Counter(r['image_id'] for r in ooc)
    with (DATA / '5pils_ooc/context_input.csv').open() as f:
        evidence = {(int(r['image_id']), int(r['caption_id'])): r for r in csv.DictReader(f)}
    labels = []
    for row in ooc:
        label = 'pristine' if row['caption'] == row['true_caption'] else 'falsified'
        reference = evidence[(row['image_id'], row['id'])]
        if reference['caption'] != row['caption'] or reference['true_veracity'] != label:
            raise ValueError('5Pils label/caption mismatch against released evidence table')
        labels.append(label)
    result = {'newsclippings': stats, 'cross_split_visualnews_id_overlap': overlap,
              '5pils_ooc': {'records': len(ooc), 'distinct_images': len(groups),
                            'records_per_image': dict(Counter(groups.values())),
                            'labels': dict(Counter(labels)),
                            'label_verification': 'Caption equality cross-checked against released true_veracity by image_id/caption_id.'},
              'note': 'Annotation audit only; no test predictions or model selection performed.'}
    target = ROOT / 'reports/data_audit.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', action='store_true', help='Also download ~2.2 GB of official CLIP embeddings')
    args = parser.parse_args()
    jobs = [(HF + f'data/merged_balanced/{s}.json', f'newsclippings/{s}.json', None) for s in ['train', 'val', 'test']]
    jobs += [(COVE + 'data/5pils_ooc/test.json', '5pils_ooc/test.json', None),
             (COVE + 'data/5pils_ooc/LICENSE', '5pils_ooc/LICENSE', None),
             (COVE + 'results/intermediate/context_input_5pils_ooc_test.csv', '5pils_ooc/context_input.csv', None)]
    if args.embeddings:
        api = f'https://huggingface.co/api/models/g-luo/news-clippings/tree/{HF_REV}/embeddings?recursive=true'
        with urllib.request.urlopen(api, timeout=30) as response:
            entries = json.load(response)
        for entry in entries:
            if entry['type'] == 'file' and entry['path'].startswith(('embeddings/clip_image_embeddings/', 'embeddings/clip_text_embeddings/')):
                jobs.append((HF + entry['path'], 'newsclippings/' + entry['path'], entry['lfs']['oid']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(lambda job: download(*job), jobs))
    # Preserve provenance of already-downloaded optional artifacts on later runs.
    receipts = [json.loads(p.read_text()) for p in sorted(DATA.rglob('*.receipt.json'))]
    (DATA / 'manifest.json').write_text(json.dumps(receipts, indent=2) + '\n')
    audit()

if __name__ == '__main__':
    main()
