"""Profile one frozen CLIP backbone in its own CPU process; no label scoring."""
import argparse
import json
import resource
import time
import numpy as np
import torch
import clip
from PIL import Image
from prepare_data import ROOT,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=['ViT-B/32','ViT-B/16','ViT-L/14'])
    args=parser.parse_args();name=args.model;torch.set_num_threads(2)
    ids=json.loads((ROOT/'reports/encoder/visualnews_image_parity.json').read_text())['sample_ids'][:8]
    path=ROOT/'data/visualnews/data.json'
    assert digest(path)=='642a28a0065e3a14d6ba74c23ec2b516710086a4df8e6b204d44095ff94f5caf'
    meta={r['id']:r for r in json.loads(path.read_text()) if r['id'] in ids}
    # OpenAI downloader checks the upstream SHA256 embedded in the model URL.
    started=time.monotonic()
    checkpoint=clip.clip._download(clip.clip._MODELS[name],str(ROOT/'artifacts/clip'))
    download_seconds=time.monotonic()-started
    started=time.monotonic();model,preprocess=clip.load(checkpoint,device='cpu',jit=False);model.eval()
    load_seconds=time.monotonic()-started
    images=[]
    for k in ids:
        with Image.open(ROOT/'data/visualnews/origin'/meta[k]['image_path'].removeprefix('./')) as im:images.append(preprocess(im.convert('RGB')).unsqueeze(0))
    tokens=clip.tokenize([meta[k]['caption'] for k in ids],truncate=False)
    times={'image':[],'text':[]}
    with torch.inference_mode():
        model.encode_image(images[0]);model.encode_text(tokens[:1])
        for modality,inputs in [('image',images),('text',[tokens[i:i+1] for i in range(len(ids))])]:
            encode=model.encode_image if modality=='image' else model.encode_text
            for x in inputs:
                start=time.monotonic();y=encode(x);times[modality].append(time.monotonic()-start)
                if not torch.isfinite(y).all():raise ValueError('Invalid encoder output')
    result=dict(model=name,device='cpu',threads=2,batch_size=1,sample_ids=ids,embedding_dimension=int(y.shape[1]),
        parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=digest(__import__('pathlib').Path(checkpoint)),
        download_or_cache_check_seconds=download_seconds,load_seconds=load_seconds,
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        torch=torch.__version__,preprocess=str(preprocess),
        timings={k:dict(seconds=v,median=float(np.median(v)),p95=float(np.quantile(v,.95))) for k,v in times.items()},
        limitation='Eight samples, CPU only. RSS includes loading/metadata overhead, not GPU VRAM. Forward latency excludes preprocessing/tokenization. No classification evaluation.')
    path=ROOT/'reports/backbone_comparison'/f"{name.replace('/','-')}_profile.json"
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
