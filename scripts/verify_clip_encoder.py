"""Check CLIP text-feature compatibility without evaluating held-out labels."""
import json
import pickle
import time
import numpy as np
import torch
import clip
from PIL import Image
from prepare_data import ROOT, DATA, digest, download


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    metadata_url='https://huggingface.co/g-luo/news-clippings/resolve/a2bbe9a8125f47a5c26338125c71e05be0266cf7/metadata/val.json'
    metadata_path=DATA/'newsclippings/metadata_val.json'
    if not metadata_path.exists():download(metadata_url,'newsclippings/metadata_val.json')
    metadata=json.loads(metadata_path.read_text())
    path=DATA/'newsclippings/embeddings/clip_text_embeddings/clip_text_embeddings_val.pkl'
    receipt=json.loads(path.with_name(path.name+'.receipt.json').read_text())
    if digest(path)!=receipt['upstream_sha256']:raise ValueError('Embedding checksum mismatch')
    with path.open('rb') as f:cached=pickle.load(f)
    eligible=sorted(k for k in cached if str(k) in metadata and metadata[str(k)].get('caption'))
    ids=np.random.default_rng(42).choice(eligible,size=64,replace=False).tolist()
    captions=[metadata[str(k)]['caption'] for k in ids]
    cache=ROOT/'artifacts/clip';cache.mkdir(parents=True,exist_ok=True)
    # OpenAI downloader verifies the checkpoint against the SHA256 in its URL.
    started=time.monotonic()
    model,preprocess=clip.load('ViT-B/32',device='cpu',jit=False,download_root=str(cache))
    model.eval()
    generated=[]
    with torch.inference_mode():
        for start in range(0,len(captions),16):
            vectors=model.encode_text(clip.tokenize(captions[start:start+16],truncate=True)).float().numpy()
            generated.append(vectors)
        # Only a synthetic constant-colour input: no test-dataset predictions.
        synthetic=preprocess(Image.new('RGB',(320,240),(100,150,200))).unsqueeze(0)
        image_vector=model.encode_image(synthetic).float().numpy()
    generated=np.concatenate(generated)
    reference=np.stack([cached[k] for k in ids]).astype(np.float32)
    generated/=np.linalg.norm(generated,axis=1,keepdims=True)
    reference/=np.linalg.norm(reference,axis=1,keepdims=True)
    cosine=np.einsum('ij,ij->i',generated,reference,dtype=np.float64)
    long_count=sum(len(clip.clip._tokenizer.encode(caption))+2>77 for caption in captions)
    if image_vector.shape!=(1,512) or not np.isfinite(image_vector).all():raise ValueError('Image forward pass failed')
    result={'model':'OpenAI CLIP ViT-B/32','clip_commit':'d05afc436d78f1c48dc0dbf8e5980a9d471f35f6',
            'torch':torch.__version__,'device':'cpu','sample_size':64,'sample_seed':42,
            'sample_visualnews_ids':ids,'sample_truncated_captions':long_count,
            'metadata_url':metadata_url,'metadata_sha256':digest(metadata_path),
            'reference_embeddings_sha256':digest(path),'checkpoint_sha256':digest(cache/'ViT-B-32.pt'),
            'normalised_text_cosine_min':float(cosine.min()),'normalised_text_cosine_mean':float(cosine.mean()),
            'normalised_text_max_absolute_difference':float(np.max(np.abs(generated-reference))),
            'text_compatibility_pass':bool(np.min(cosine)>.999),
            'image_preprocess':str(preprocess),'synthetic_image_embedding_shape':list(image_vector.shape),
            'image_embedding_parity':'Not verified: original VisualNews images unavailable.',
            'elapsed_seconds_including_model_load':time.monotonic()-started,
            'test_evaluated':False,'note':'Text compatibility sample and image runtime smoke check only; not factual verification or image-feature parity.'}
    out=ROOT/'reports/encoder';out.mkdir(exist_ok=True)
    (out/'compatibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sample_visualnews_ids','image_preprocess']},indent=2))
    if not result['text_compatibility_pass']:raise SystemExit('Text compatibility failed; inspect before inference.')

if __name__=='__main__':main()
