"""Reassemble numeric ZIP parts, safely extract, and audit VERITE image coverage."""
import argparse
import csv
import hashlib
import json
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from PIL import Image
from prepare_data import ROOT, DATA, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parts',nargs='+',type=Path)
    args=parser.parse_args()
    parts=sorted(args.parts,key=lambda p:int(p.suffix[1:]))
    if [int(p.suffix[1:]) for p in parts] != list(range(1,len(parts)+1)):
        raise ValueError('Parts must start at 001 and be consecutive')
    directory=DATA/'verite';directory.mkdir(exist_ok=True)
    archive=directory/'images.zip';temp=directory/'images.zip.part'
    with temp.open('wb') as out:
        for part in parts:
            with part.open('rb') as source:shutil.copyfileobj(source,out)
    if archive.exists() and digest(archive)!=digest(temp):
        temp.unlink();raise ValueError('Different images.zip already exists; not overwriting')
    temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist()
        if sum(e.file_size for e in entries)>1024**3:
            raise ValueError('Archive expanded size exceeds 1 GiB')
        names=set()
        for entry in entries:
            p=PurePosixPath(entry.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in entry.filename or ':' in entry.filename:
                raise ValueError('Unsafe archive path')
            if stat.S_ISLNK(entry.external_attr>>16):raise ValueError('Symlink in archive')
            if str(p) in names:raise ValueError('Duplicate archive path')
            names.add(str(p))
            target=directory.joinpath(*p.parts)
            if not target.resolve().is_relative_to(directory.resolve()):raise ValueError('Path escapes target')
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failed: '+bad)
        for entry in entries:
            target=directory/entry.filename
            if entry.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            payload=z.read(entry)
            if target.exists() and hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(payload).digest():
                raise ValueError('Refusing to overwrite different existing file: '+entry.filename)
            target.write_bytes(payload)
    with (directory/'VERITE.csv').open() as f:rows=list(csv.DictReader(f))
    paths=sorted({r['image_path'] for r in rows})
    images=[];missing=[];invalid=[]
    for relative in paths:
        path=directory/relative
        if not path.resolve().is_relative_to(directory.resolve()):raise ValueError('Unsafe annotation image path')
        if not path.is_file():missing.append(relative);continue
        try:
            with Image.open(path) as im:im.verify()
            with Image.open(path) as im:
                im.load();images.append(dict(path=relative,width=im.width,height=im.height,format=im.format,sha256=digest(path)))
        except Exception as exc:invalid.append(dict(path=relative,error=str(exc)))
    unavailable=set(missing)|{r['path'] for r in invalid}
    usable=[r for r in rows if r['image_path'] not in unavailable]
    report=dict(archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),zip_crc='passed',
                parts=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in parts],
                annotations_sha256=digest(directory/'VERITE.csv'),annotation_rows=len(rows),
                label_counts=dict(Counter(r['label'] for r in rows)),referenced_images=len(paths),
                valid_referenced_images=len(images),missing_images=missing,invalid_images=invalid,
                usable_pairs=len(usable),usable_label_counts=dict(Counter(r['label'] for r in usable)),
                unavailable_rows=[r for r in rows if r['image_path'] in unavailable],
                unreferenced_archive_files=[e.filename for e in entries if not e.is_dir() and e.filename not in paths],
                note='Data integrity/coverage audit only. No model inference, tuning, or test scoring performed. Counts follow the downloaded annotation file; do not assume the paper benchmark is identical.')
    (directory/'image_manifest.json').write_text(json.dumps(images,indent=2)+'\n')
    out=ROOT/'reports/verite';out.mkdir(exist_ok=True)
    (out/'image_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
