"""Dataset EDA and deterministic validation-image encoder check; no model fitting."""
import collections
import json
import os
import pickle
import numpy as np
from PIL import Image
from prepare_data import ROOT, digest


def main():
    out = ROOT/'reports/eda'; out.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import clip
    import torch
    metadata_path = ROOT/'data/visualnews/data.json'
    assert digest(metadata_path) == '642a28a0065e3a14d6ba74c23ec2b516710086a4df8e6b204d44095ff94f5caf'
    metadata = {r['id']:r for r in json.loads(metadata_path.read_text())}
    splits = {s:json.loads((ROOT/f'data/newsclippings/{s}.json').read_text())['annotations'] for s in ('train','val','test')}
    summary = {}; lengths = {}; all_images = set(); dimensions = []
    for split,rows in splits.items():
        caption_ids = sorted({r['id'] for r in rows}); image_ids = {r['image_id'] for r in rows}; all_images.update(image_ids)
        words = [len(metadata[k]['caption'].split()) for k in caption_ids]
        tokens = [len(clip.clip._tokenizer.encode(metadata[k]['caption'])) for k in caption_ids]
        lengths[split] = words
        summary[split] = dict(pairs=len(rows),labels=dict(collections.Counter('misleading' if r['falsified'] else 'original' for r in rows)),
            unique_captions=len(caption_ids),unique_images=len(image_ids),
            caption_sources=dict(collections.Counter(metadata[k]['source'] for k in caption_ids)),
            caption_topics=dict(collections.Counter(metadata[k]['topic'] for k in caption_ids)),
            image_sources=dict(collections.Counter(metadata[k]['source'] for k in image_ids)),
            caption_words_percentiles=dict(zip(['min','p25','median','p75','p95','max'],np.percentile(words,[0,25,50,75,95,100]).tolist())),
            captions_over_75_clip_tokens=sum(n>75 for n in tokens))
    for k in sorted(all_images):
        with Image.open(ROOT/'data/visualnews/origin'/metadata[k]['image_path'].removeprefix('./')) as im:
            dimensions.append(im.size)
    dimensions = np.array(dimensions)
    result = dict(splits=summary,metadata_sha256=digest(metadata_path),
        image_count=len(dimensions),width_percentiles=np.percentile(dimensions[:,0],[0,50,95,100]).tolist(),
        height_percentiles=np.percentile(dimensions[:,1],[0,50,95,100]).tolist(),
        note='Counts by source/topic use unique caption IDs, not pair counts. Dimensions cover unique images across all splits. Test metadata exploration is descriptive; no model selection.')
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=np.arange(3);names=list(splits)
    for j,label in enumerate(['original','misleading']):
        axes[0,0].bar(x+(j-.5)*.35,[summary[s]['labels'][label] for s in names],width=.35,label=label)
    axes[0,0].set(xticks=x,xticklabels=names,title='Pair counts and class balance',ylabel='Pairs');axes[0,0].legend()
    sources=summary['train']['caption_sources']
    axes[0,1].bar(list(sources),list(sources.values()));axes[0,1].tick_params(axis='x',rotation=15)
    axes[0,1].set(title='Training caption sources',ylabel='Unique captions')
    for s in names:axes[1,0].hist(lengths[s],bins=np.arange(0,36,2),density=True,histtype='step',label=s)
    axes[1,0].set(title='Caption lengths (all selected captions)',xlabel='Whitespace-separated words',ylabel='Density');axes[1,0].legend()
    axes[1,1].hist(dimensions[:,0]/dimensions[:,1],bins=60,range=(0,4))
    axes[1,1].set(title='Image aspect ratios (display limited to 0–4)',xlabel='Width / height',ylabel='Unique images')
    fig.savefig(out/'dataset_overview.png',dpi=160);plt.close(fig)
    # Validation-only, seeded sample; no hand-picking based on parity outcomes.
    ids=np.random.default_rng(42).choice(sorted({r['image_id'] for r in splits['val']}),64,replace=False).tolist()
    path=ROOT/'data/newsclippings/embeddings/clip_image_embeddings/clip_image_embeddings_val.pkl'
    receipt=json.loads(path.with_name(path.name+'.receipt.json').read_text())
    assert digest(path)==receipt['upstream_sha256']
    with path.open('rb') as f:cached=pickle.load(f)
    torch.set_num_threads(2)
    model,preprocess=clip.load('ViT-B/32',device='cpu',jit=False,download_root=str(ROOT/'artifacts/clip'));model.eval()
    generated=[];hashes={}
    with torch.inference_mode():
        for start in range(0,len(ids),8):
            batch=[]
            for k in ids[start:start+8]:
                p=ROOT/'data/visualnews/origin'/metadata[k]['image_path'].removeprefix('./')
                hashes[str(k)]=digest(p)
                with Image.open(p) as im:batch.append(preprocess(im.convert('RGB')))
            generated.append(model.encode_image(torch.stack(batch)).float().numpy())
    a=np.concatenate(generated);b=np.stack([cached[k] for k in ids]).astype(np.float32)
    a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
    cosine=np.einsum('ij,ij->i',a,b,dtype=np.float64)
    parity=dict(sample_ids=ids,seed=42,count=64,split='val',model='OpenAI CLIP ViT-B/32',device='cpu',
        preprocessing=str(preprocess),checkpoint_sha256=digest(ROOT/'artifacts/clip/ViT-B-32.pt'),
        reference_sha256=digest(path),image_sha256=hashes,cosine_per_image=cosine.tolist(),
        cosine_min=float(cosine.min()),cosine_mean=float(cosine.mean()),max_absolute_difference=float(np.max(np.abs(a-b))),
        all_cosines_above_0999=bool((cosine>.999).all()),
        limitation='Sampled numerical agreement only, not full-dataset parity or detection accuracy. Standard app preprocessing; no preprocessing selection on test outcomes.')
    (ROOT/'reports/encoder/visualnews_image_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    # Local-only visual examples: captions and labels describe dataset pairing, not independent fact verdicts.
    local=ROOT/'outputs/eda';local.mkdir(parents=True,exist_ok=True)
    import textwrap
    sample=np.random.default_rng(42).choice(len(splits['train']),6,replace=False)
    fig,axes=plt.subplots(3,2,figsize=(13,13),constrained_layout=True)
    for ax,index in zip(axes.flat,sample):
        r=splits['train'][index];p=ROOT/'data/visualnews/origin'/metadata[r['image_id']]['image_path'].removeprefix('./')
        with Image.open(p) as im:ax.imshow(im.convert('RGB'))
        ax.axis('off');ax.set_title(('MISLEADING' if r['falsified'] else 'ORIGINAL')+f" | image {r['image_id']} / caption {r['id']}\n"+'\n'.join(textwrap.wrap(metadata[r['id']]['caption'],65)),fontsize=9)
    fig.savefig(local/'training_examples.png',dpi=120);plt.close(fig)
    print(json.dumps({'image_count':len(dimensions),'parity_min':parity['cosine_min'],'parity_mean':parity['cosine_mean'],'sample_agreement_above_0999':parity['all_cosines_above_0999'],'splits':summary},indent=2))


if __name__=='__main__':main()
