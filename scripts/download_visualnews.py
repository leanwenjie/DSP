"""Resume a bounded-memory scan of the official tar, retaining required images only."""
import argparse
import io
import json
import shutil
import tarfile
import time
from pathlib import Path
import requests
from PIL import Image
from prepare_data import ROOT, digest

URL = 'https://www.cs.rice.edu/~vo9/visualnews/origin.tar'
TOTAL = 97916641280
BLOCK = 8 * 1024 * 1024
BASE = ROOT / 'data/visualnews'
STATE = BASE / 'download_state.json'


class RemoteArchive:
    def __init__(self, session):
        self.session, self.start, self.cache, self.transferred = session, -1, b'', 0

    def read(self, offset, size):
        chunks = []
        while size:
            if not self.start <= offset < self.start + len(self.cache):
                end = min(TOTAL - 1, offset + max(BLOCK, size) - 1)
                for attempt in range(4):
                    try:
                        response = self.session.get(URL, headers={'Range': f'bytes={offset}-{end}'}, timeout=(15,90))
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{TOTAL}':
                            raise ValueError('Server did not honor exact range')
                        if len(response.content) != end-offset+1:
                            raise ValueError('Truncated range')
                        break
                    except requests.RequestException:
                        if attempt == 3: raise
                        time.sleep(2)
                self.start, self.cache = offset, response.content
                self.transferred += len(self.cache)
            n = min(size, self.start + len(self.cache) - offset)
            chunks.append(self.cache[offset-self.start:offset-self.start+n])
            offset += n
            size -= n
        return b''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-gib', type=float, default=1, help='Network budget per run; resume by rerunning')
    args = parser.parse_args()
    metadata = BASE / 'data.json'
    if digest(metadata) != '642a28a0065e3a14d6ba74c23ec2b516710086a4df8e6b204d44095ff94f5caf':
        raise ValueError('Unexpected metadata')
    ids = set()
    for split in ('train','val','test'):
        rows = json.loads((ROOT/f'data/newsclippings/{split}.json').read_text())['annotations']
        ids.update(r['image_id'] for r in rows)
    wanted = {r['image_path'].removeprefix('./'):r['id'] for r in json.loads(metadata.read_text()) if r['id'] in ids}
    assert len(wanted) == len(ids) == 71966
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(offset=0, scanned_members=0, saved=0, network_bytes=0, complete=False)
    archive = RemoteArchive(requests.Session())
    previous = state['network_bytes']
    def checkpoint():
        state.update(network_bytes=previous+archive.transferred, required_images=len(wanted), source=URL)
        temp = STATE.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2)+'\n'); temp.replace(STATE)
    try:
        while state['offset'] < TOTAL and not state['complete']:
            if archive.transferred >= args.max_gib*1024**3: break
            if shutil.disk_usage(BASE).free < 3*1024**3: raise RuntimeError('Stopped: preserve 3 GiB free disk')
            offset = state['offset']
            header = archive.read(offset,512)
            if header == bytes(512):
                state['complete'] = True
                break
            member = tarfile.TarInfo.frombuf(header,'utf-8','surrogateescape')
            if member.type not in (tarfile.REGTYPE,tarfile.AREGTYPE,tarfile.DIRTYPE):
                raise ValueError(f'Unsupported tar entry at {offset}: {member.type!r}; preserve checkpoint for inspection')
            name = member.name.removeprefix('./').removeprefix('origin/')
            if member.isfile() and name in wanted:
                if member.size > 50*1024**2: raise ValueError('Unexpected image size')
                path = BASE/'origin'/name
                if not path.resolve().is_relative_to((BASE/'origin').resolve()): raise ValueError('Unsafe path')
                raw = archive.read(offset+512,member.size)
                with Image.open(io.BytesIO(raw)) as im: im.verify()
                path.parent.mkdir(parents=True,exist_ok=True)
                temp = path.with_suffix(path.suffix+'.part');temp.write_bytes(raw);temp.replace(path)
                with (BASE/'downloaded_images.jsonl').open('a') as f:
                    f.write(json.dumps(dict(id=wanted[name],path=name,bytes=member.size,sha256=digest(path),archive_offset=offset))+'\n')
                state['saved'] += 1
            state['offset'] = offset+512+((member.size+511)//512)*512
            state['scanned_members'] += 1
            if state['scanned_members'] % 1000 == 0:
                checkpoint(); print(json.dumps(state),flush=True)
    finally:
        checkpoint()
    print(json.dumps(state),flush=True)


if __name__ == '__main__':
    main()
