"""Audit a snapshot of the downloaded subset without interrupting acquisition."""
import json
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from prepare_data import ROOT, digest


def main():
    base = ROOT / 'data/visualnews'
    # Ignore an incomplete last line while a downloader is appending.
    lines = (base/'downloaded_images.jsonl').read_text().splitlines(keepends=True)
    entries = [json.loads(line) for line in lines if line.endswith('\n')]
    metadata = {r['id']: r for r in json.loads((base/'data.json').read_text())}
    valid, failures = set(), []
    for r in entries:
        path = base/'origin'/r['path']
        try:
            if not path.resolve().is_relative_to((base/'origin').resolve()): raise ValueError('Unsafe path')
            if metadata[r['id']]['image_path'].removeprefix('./') != r['path']: raise ValueError('Metadata path mismatch')
            if path.stat().st_size != r['bytes'] or digest(path) != r['sha256']: raise ValueError('File integrity mismatch')
            with Image.open(path) as im: im.verify()
            valid.add(r['id'])
        except (OSError, ValueError, KeyError) as exc:
            failures.append({'id':r['id'],'error':str(exc)})
    splits = {}
    required = set()
    for split in ('train','val','test'):
        rows = json.loads((ROOT/f'data/newsclippings/{split}.json').read_text())['annotations']
        ids = {r['image_id'] for r in rows}; required.update(ids)
        splits[split] = dict(required_unique_images=len(ids),verified_unique_images=len(ids & valid),
            missing_unique_images=len(ids-valid),pairs_with_verified_image=sum(r['image_id'] in valid for r in rows))
    state = json.loads((base/'download_state.json').read_text())
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),snapshot_entries=len(entries),
        duplicate_log_ids=len(entries)-len({r['id'] for r in entries}),verified_unique_images=len(valid),
        required_unique_images=len(required),unexpected_ids=sorted(valid-required),failures=failures,splits=splits,
        archive_scan_complete=state['complete'],archive_offset=state['offset'],
        coverage_complete=not failures and required <= valid,
        warning='Snapshot while acquisition may be running. Partial source-ordered subset is not a representative training/evaluation sample.')
    (ROOT/'reports/visualnews_download_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if failures: raise RuntimeError('Downloaded image verification failed')


if __name__ == '__main__':
    main()
