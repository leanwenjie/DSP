"""Feature-parity check against RED-DOT artifacts; no classifier or label scoring."""
import csv
import json
import urllib.request
import numpy as np
import torch
import clip
from PIL import Image
from prepare_data import ROOT, DATA, digest


def main():
    torch.set_num_threads(2)
    root=DATA/'verite'
    revision='4dd44fdc8ff8bfec2acb265019b24de77cc3a09a'
    base=f'https://raw.githubusercontent.com/stevejpapad/relevant-evidence-detection/{revision}/data/VERITE/'
    files=['VERITE.csv','VERITE_clip_image_embeddings_ViTB32.npy','VERITE_clip_text_embeddings_ViTB32.npy']
    target=root/'red_dot_reference';target.mkdir(exist_ok=True)
    for name in files:
        path=target/name
        if not path.exists():path.write_bytes(urllib.request.urlopen(base+name,timeout=30).read())
    with (target/'VERITE.csv').open() as f:reference_rows=list(csv.DictReader(f))
    with (root/'VERITE.csv').open() as f:rows=list(csv.DictReader(f))
    if [(r['image_path'],r['caption']) for r in rows]!=[(r['image_path'],r['caption']) for r in reference_rows]:
        raise ValueError('Reference row order mismatch')
    image_reference=np.load(target/files[1],allow_pickle=False)
    text_reference=np.load(target/files[2],allow_pickle=False)
    indices=np.random.default_rng(42).choice(len(rows),size=32,replace=False)
    model,preprocess=clip.load('ViT-B/32',device='cpu',jit=False,download_root=str(ROOT/'artifacts/clip'))
    model.eval();outputs={'image':[],'image_red_dot_resize400':[],'text':[]}
    with torch.inference_mode():
        for start in range(0,len(indices),8):
            batch=[rows[int(i)] for i in indices[start:start+8]]
            images=[]; resized_images=[]
            for r in batch:
                with Image.open(root/r['image_path']) as image:
                    images.append(preprocess(image))
                    # RED-DOT src/extract_features.py explicitly resizes the long side
                    # to 400 (including upscaling) before the CLIP eval transform.
                    rgb = image.convert('RGB')
                    width, height = rgb.size
                    size = (400, int(height * (400 / width))) if width > height else (int(width * (400 / height)), 400)
                    resized_images.append(preprocess(rgb.resize(size, resample=Image.Resampling.BICUBIC)))
            outputs['image'].append(model.encode_image(torch.stack(images)).float().numpy())
            outputs['image_red_dot_resize400'].append(model.encode_image(torch.stack(resized_images)).float().numpy())
            outputs['text'].append(model.encode_text(clip.tokenize([r['caption'] for r in batch],truncate=True)).float().numpy())
    result={'reference_revision':revision,'reference_base_url':base,'sample_size':32,'sample_seed':42,
            'row_indices':indices.tolist(),'classifier_predictions_generated':False,'reference_files_sha256':{f:digest(target/f) for f in files},
            'red_dot_resize_source':f'https://github.com/stevejpapad/relevant-evidence-detection/blob/{revision}/src/extract_features.py',
            'preprocess':str(preprocess),'note':'Feature agreement only. Does not independently establish image parity with NewsCLIPpings training embeddings.'}
    for modality,ref in [('image',image_reference),('image_red_dot_resize400',image_reference),('text',text_reference)]:
        a=np.concatenate(outputs[modality]);b=ref[indices].astype(np.float32)
        a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
        similarity=np.einsum('ij,ij->i',a,b,dtype=np.float64)
        result[modality]={'cosine_min':float(similarity.min()),'cosine_mean':float(similarity.mean()),'cosine_median':float(np.median(similarity)),
                          'all_above_0_999':bool(np.all(similarity>.999))}
    out=ROOT/'reports/encoder';out.mkdir(exist_ok=True)
    (out/'verite_reference_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['row_indices','preprocess']},indent=2))

if __name__=='__main__':main()
