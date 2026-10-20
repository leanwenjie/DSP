"""Collect raw 5Pils-OOC images. These are NOT the processed benchmark images."""
import argparse
import concurrent.futures
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import urllib.request
from PIL import Image
from prepare_data import DATA, ROOT, download

REV = 'acf672a20900f80e99f2d013aa2f30b7469fa180'


def collect(row):
    image_id = str(row['image_id'])
    if not image_id.isdecimal():
        raise ValueError('Expected numeric image ID')
    url = row['image_URL']
    path = DATA / '5pils_ooc/raw_images' / (image_id + '.img')
    result = {'image_id': row['image_id'], 'url': url, 'path': str(path.relative_to(ROOT)),
              'benchmark_image_path': row['image_path']}
    try:
        if path.exists():
            payload = path.read_bytes()
        else:
            request = urllib.request.Request(url, headers={'User-Agent': 'DSP-academic-dataset-preparation/1.0'})
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read(25 * 1024 * 1024 + 1)
            if len(payload) > 25 * 1024 * 1024:
                raise ValueError('Image exceeds 25 MiB limit')
        with Image.open(io.BytesIO(payload)) as im:
            result.update(width=im.width, height=im.height, format=im.format)
            im.verify()
        path.write_bytes(payload)
        result.update(status='downloaded_raw', sha256=hashlib.sha256(payload).hexdigest())
    except Exception as exc:
        result.update(status='failed', error=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    download(f'https://raw.githubusercontent.com/UKPLab/5pils/{REV}/dataset/image_processing_instructions.txt',
             '5pils_ooc/image_processing_instructions.txt')
    rows = json.loads((DATA / '5pils_ooc/test.json').read_text())
    unique = {}
    for row in rows:
        if row['image_id'] in unique and unique[row['image_id']]['image_URL'] != row['image_URL']:
            raise ValueError('Conflicting URL for image ID')
        unique[row['image_id']] = row
    (DATA / '5pils_ooc/raw_images').mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(collect, unique.values()):
            results.append(result)
            if len(results) % 50 == 0:
                print(f'{len(results)}/{len(unique)} checked', flush=True)
    (DATA / '5pils_ooc/image_download_manifest.json').write_text(json.dumps(results, indent=2) + '\n')
    summary = {'total_unique_images': len(unique), 'outcomes': dict(Counter(r['status'] for r in results)),
               'processing_status': 'RAW ONLY: upstream cropping/standard processing must be applied before benchmark evaluation.',
               'failures': [r for r in results if r['status'] == 'failed']}
    (ROOT / 'reports/5pils_image_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'failures'}, indent=2))

if __name__ == '__main__':
    main()
