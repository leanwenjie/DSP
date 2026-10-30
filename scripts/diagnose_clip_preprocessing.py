"""Fixed validation-only preprocessing diagnostics; never alter app or models."""
import json
import pickle
import numpy as np
import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from prepare_data import ROOT, digest


def main():
    previous=json.loads((ROOT/'reports/encoder/visualnews_image_parity.json').read_text())
    ids=previous['sample_ids']
    metadata_path=ROOT/'data/visualnews/data.json'
    assert digest(metadata_path)=='642a28a0065e3a14d6ba74c23ec2b516710086a4df8e6b204d44095ff94f5caf'
    meta={r['id']:r for r in json.loads(metadata_path.read_text())}
    path=ROOT/'data/newsclippings/embeddings/clip_image_embeddings/clip_image_embeddings_val.pkl'
    assert digest(path)==previous['reference_sha256']
    with path.open('rb') as f: cached=pickle.load(f)
    reference=np.stack([cached[k] for k in ids]).astype(np.float32)
    reference/=np.linalg.norm(reference,axis=1,keepdims=True)
    torch.set_num_threads(2)
    model,standard=clip.load('ViT-B/32',device='cpu',jit=False,download_root=str(ROOT/'artifacts/clip'));model.eval()
    normalise=Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))
    bicubic=InterpolationMode.BICUBIC
    candidates={
        'standard_bicubic':standard,
        'short_edge_224_bilinear':Compose([Resize(224,interpolation=InterpolationMode.BILINEAR),CenterCrop(224),ToTensor(),normalise]),
        'square_256_crop_224_unnormalised':Compose([Resize((256,256),interpolation=bicubic),CenterCrop(224),ToTensor()]),
        'square_256_crop_224_normalised':Compose([Resize((256,256),interpolation=bicubic),CenterCrop(224),ToTensor(),normalise]),
        'square_224_normalised':Compose([Resize((224,224),interpolation=bicubic),ToTensor(),normalise]),
        'short_edge_256_crop_224_normalised':Compose([Resize(256,interpolation=bicubic),CenterCrop(224),ToTensor(),normalise]),
    }
    out=ROOT/'reports/encoder/preprocessing_diagnostic.json'
    report=dict(sample_ids=ids,split='val',criterion='Every sampled cosine > .999; numerical agreement, not accuracy',
        candidate_policy='Fixed six variants before scoring; based on standard CLIP, older interpolation possibility and upstream square-resize config. No classifier labels or test predictions.',
        upstream_config='https://github.com/g-luo/news_clippings_training/blob/master/configs/models/clip.yaml',
        reference_sha256=digest(path),checkpoint_sha256=digest(ROOT/'artifacts/clip/ViT-B-32.pt'),results={})
    images=[]
    for k in ids:
        p=ROOT/'data/visualnews/origin'/meta[k]['image_path'].removeprefix('./')
        assert digest(p)==previous['image_sha256'][str(k)]
        with Image.open(p) as im:images.append(im.convert('RGB'))
    with torch.inference_mode():
        for name,transform in candidates.items():
            fresh=np.concatenate([model.encode_image(torch.stack([transform(im) for im in images[i:i+8]])).float().numpy() for i in range(0,len(ids),8)])
            fresh/=np.linalg.norm(fresh,axis=1,keepdims=True)
            similarity=np.einsum('ij,ij->i',fresh,reference,dtype=np.float64)
            result=dict(cosine_min=float(similarity.min()),cosine_mean=float(similarity.mean()),all_above_0999=bool((similarity>.999).all()),per_image=similarity.tolist(),transform=str(transform))
            report['results'][name]=result
            out.write_text(json.dumps(report,indent=2)+'\n')
            print(name,result['cosine_min'],result['cosine_mean'],result['all_above_0999'],flush=True)
    report['completed']=True
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
