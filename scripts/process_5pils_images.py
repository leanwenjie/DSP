"""Apply published 5Pils crops and reproduce its standard sidebar-removal algorithm.

Reference: UKPLab/5pils dataset_collection/scrape_utils.py at
acf672a20900f80e99f2d013aa2f30b7469fa180 (Apache-2.0).
Reimplemented with explicit input checks; not a guarantee that live web images
are byte-identical to those used by the authors.
"""
import ast
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_data import DATA, ROOT, digest


def longest_rows(array):
    colorful = (np.ptp(array, axis=1) > 30).any(axis=1)
    transitions = np.diff(colorful.astype(int))
    starts = np.flatnonzero(transitions == 1) + 1
    ends = np.flatnonzero(transitions == -1)
    if len(starts) == 0 or (len(ends) and starts[0] > ends[0]):
        starts = np.insert(starts, 0, 0)
    if len(ends) == 0 or (len(starts) and ends[-1] < starts[-1]):
        ends = np.append(ends, len(array) - 1)
    start, end = max(zip(starts, ends), key=lambda pair: pair[1] - pair[0])
    if end <= start:
        raise ValueError('No nonempty region under published standard-processing rule')
    # Preserve upstream end-exclusive slicing, even though end marks a colorful row.
    return array[start:end]


def main():
    instructions = dict(line.split(': ', 1) for line in
        (DATA / '5pils_ooc/image_processing_instructions.txt').read_text().splitlines() if line.strip())
    rows = json.loads((DATA / '5pils_ooc/image_download_manifest.json').read_text())
    output = DATA / '5pils_ooc/processed_img'
    output.mkdir(exist_ok=True)
    results = []
    for row in rows:
        result = {'image_id': row['image_id']}
        try:
            if row['status'] != 'downloaded_raw':
                raise ValueError('Raw image unavailable')
            source = ROOT / row['path']
            if digest(source) != row['sha256']:
                raise ValueError('Raw image hash changed')
            name = Path(row['benchmark_image_path']).name
            instruction = instructions[name]
            with Image.open(source) as im:
                im.load()
                im = im.convert('RGB')
                if instruction == 'Standard processing':
                    a = longest_rows(np.asarray(im))
                    a = longest_rows(a.transpose(1, 0, 2)).transpose(1, 0, 2)
                    processed = Image.fromarray(a)
                elif instruction.startswith('Cropped: '):
                    box = ast.literal_eval(instruction.split(': ', 1)[1])
                    if len(box) != 4 or not (0 <= box[0] < box[2] <= im.width and 0 <= box[1] < box[3] <= im.height):
                        raise ValueError(f'Published crop outside current image dimensions: {box}, {im.size}')
                    processed = im.crop(box)
                else:
                    raise ValueError('Unsupported processing instruction')
                target = output / name
                processed.save(target)
                result.update(status='processed', path=str(target.relative_to(ROOT)),
                              width=processed.width, height=processed.height, sha256=digest(target))
        except Exception as exc:
            result.update(status='unavailable', error=str(exc))
        results.append(result)
    (DATA / '5pils_ooc/processed_manifest.json').write_text(json.dumps(results, indent=2) + '\n')
    summary = {'outcomes': dict(Counter(r['status'] for r in results)),
               'unavailable': [r for r in results if r['status'] != 'processed'],
               'note': 'Published crops/standard algorithm applied to current web images; visual spot-check remains required. Exclude both captions for any unavailable image.'}
    (ROOT / 'reports/5pils_processing_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
