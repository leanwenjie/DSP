"""Resumable B/16 or L/14 development study; preserves the frozen B/32 script."""
import argparse
import re
import fcntl
import json
import time
import warnings
import numpy as np
import torch
import clip
import sklearn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from prepare_data import ROOT, digest
from model_inference import features, predict_saved
from similarity_baseline import select_threshold

# B/32 source is intentionally preserved because its frozen manifest hashes it.


def write_json(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def unit_vectors(vectors, dimension):
    a=np.asarray(vectors,dtype=np.float32)
    norms=np.linalg.norm(a,axis=1,keepdims=True)
    if a.ndim!=2 or a.shape[1]!=dimension or not np.isfinite(a).all() or (norms==0).any():
        raise ValueError('Invalid extracted features')
    return a/norms


def stable_preprocessing(value):
    """Remove only process-specific addresses from callable representations."""
    return re.sub(r"(<function [^ >]+) at 0x[0-9a-fA-F]+>", r"\1>", value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backbone',choices=['ViT-B/16','ViT-L/14'])
    args=parser.parse_args()
    backbone=args.backbone
    dimension=512 if backbone=='ViT-B/16' else 768
    batch_size=8
    slug=backbone.replace('/','-')
    ART=ROOT/'artifacts/backbones'/slug
    OUT=ROOT/'reports/backbone_comparison'/slug
    ART.parent.mkdir(exist_ok=True)
    ART.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    lock=(ART/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    model,preprocess=clip.load(backbone,device='cpu',jit=False,download_root=str(ROOT/'artifacts/clip'));model.eval()
    paths=[ROOT/'data/visualnews/data.json',ROOT/'data/visualnews/downloaded_images.jsonl',
           ROOT/f'artifacts/clip/{slug}.pt',ROOT/'reports/backbone_comparison/PROTOCOL.md',ROOT/'scripts/train_clip_backbones.py',
           ROOT/'scripts/model_inference.py',ROOT/'scripts/similarity_baseline.py']
    paths += [ROOT/f'data/newsclippings/{s}.json' for s in ('train','val')]
    provenance=dict(sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        torch=torch.__version__,sklearn=sklearn.__version__,numpy=np.__version__,preprocessing=str(preprocess),device='cpu',backbone=backbone,dimension=dimension,batch_size=batch_size)
    assert provenance['sha256']['data/visualnews/data.json']=='642a28a0065e3a14d6ba74c23ec2b516710086a4df8e6b204d44095ff94f5caf'
    manifest=ART/'extraction_manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        comparable=dict(provenance)
        comparable['preprocessing']=stable_preprocessing(comparable['preprocessing'])
        previous['preprocessing']=stable_preprocessing(previous['preprocessing'])
        if previous!=comparable:raise ValueError('Extraction provenance changed; do not mix runs')
    else:write_json(manifest,provenance)
    if (OUT/'metrics.json').exists():
        print('Study already complete; preserving saved results.',flush=True);return
    meta={r['id']:r for r in json.loads(paths[0].read_text())}
    hashes={r['id']:r['sha256'] for r in map(json.loads,paths[1].read_text().splitlines())}
    state_path=ART/'progress.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'completed':{},'stage':'extraction'}
    data={};started=time.monotonic()
    with torch.inference_mode():
        for split in ('train','val'):
            rows=json.loads((ROOT/f'data/newsclippings/{split}.json').read_text())['annotations'];vectors=[]
            for modality,key in [('image','image_id'),('text','id')]:
                ids=sorted({r[key] for r in rows});name=f'{split}_{modality}'
                path=ART/f'{name}.npy';n=state['completed'].get(name,0)
                if not 0<=n<=len(ids) or (n and not path.exists()):raise ValueError('Invalid resume state')
                a=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(ids),dimension))
                if a.shape!=(len(ids),dimension) or a.dtype!=np.float32:raise ValueError('Incompatible buffer')
                for i in range(n,len(ids),batch_size):
                    selected=ids[i:i+batch_size]
                    if modality=='image':
                        batch=[]
                        for k in selected:
                            p=ROOT/'data/visualnews/origin'/meta[k]['image_path'].removeprefix('./')
                            if digest(p)!=hashes[k]:raise ValueError(f'Image changed: {k}')
                            with Image.open(p) as im:batch.append(preprocess(im.convert('RGB')))
                        raw=model.encode_image(torch.stack(batch)).float().numpy()
                    else:raw=model.encode_text(clip.tokenize([meta[k]['caption'] for k in selected],truncate=False)).float().numpy()
                    a[i:i+len(selected)]=unit_vectors(raw,dimension);a.flush()
                    state['completed'][name]=i+len(selected)
                    state.update(active=name,active_total=len(ids),stage='extraction')
                    write_json(state_path,state)
                    if i==n or (i//batch_size)%100==0:print(name,i+len(selected),'/',len(ids),'run_seconds',round(time.monotonic()-started),flush=True)
                index={k:i for i,k in enumerate(ids)}
                vectors.append(np.asarray(a[[index[r[key]] for r in rows]]))
            data[split]=(vectors,np.array([int(r['falsified']) for r in rows]))
    state['stage']='training';write_json(state_path,state)
    y=data['train'][1];yv=data['val'][1];candidates=[];best=None;winners={}
    cosine=np.einsum('ij,ij->i',*data['val'][0],dtype=np.float64)
    threshold,_,_,_=select_threshold(cosine,yv)
    baseline=dict(threshold=threshold,report=classification_report(yv,cosine<threshold,output_dict=True))
    with threadpool_limits(limits=2):
        for mode in ('concat','interaction'):
            scaler=StandardScaler();xt=scaler.fit_transform(features(data['train'][0],mode));xv=scaler.transform(features(data['val'][0],mode))
            for c in (.001,.01,.1):
                model=LogisticRegression(C=c,solver='lbfgs',max_iter=1000,tol=1e-5,random_state=42)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);model.fit(xt,y)
                score=model.decision_function(xv);threshold,macro,_,_=select_threshold(-score,yv);pred=(-score<threshold).astype(int)
                r=dict(mode=mode,C=c,validation_macro_f1=macro,threshold=threshold,report=classification_report(yv,pred,output_dict=True),confusion_matrix=confusion_matrix(yv,pred).tolist())
                candidates.append(r);print(json.dumps(r),flush=True)
                if mode not in winners or macro>winners[mode]['validation_macro_f1']:
                    winners[mode]=r
                    np.savez(ART/f'{mode}_model.npz',mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,intercept=model.intercept_,threshold=threshold,feature_mode=mode,classes=model.classes_)
                    restored,labels=predict_saved(data['val'][0],ART/f'{mode}_model.npz')
                    np.testing.assert_allclose(restored,score,atol=1e-6,rtol=1e-6);np.testing.assert_array_equal(labels,pred)
                if best is None or macro>best['validation_macro_f1']:
                    best=r
                    np.savez(ART/'best_model.npz',mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,intercept=model.intercept_,threshold=threshold,feature_mode=mode,classes=model.classes_)
                    restored,labels=predict_saved(data['val'][0],ART/'best_model.npz')
                    np.testing.assert_allclose(restored,score,atol=1e-6,rtol=1e-6);np.testing.assert_array_equal(labels,pred)
                    np.savez(ART/'validation_predictions.npz',labels=yv,score=score,predictions=pred)
    write_json(OUT/'metrics.json',dict(backbone=backbone,dimension=dimension,representation_winners=winners,selected=best,candidates=candidates,similarity=baseline,selection_split='val',test_evaluated=False,warning='Validation-selected development scores; not held-out improvements',provenance=provenance,model_sha256=digest(ART/'best_model.npz')))
    state.update(stage='complete',complete=True);write_json(state_path,state)
    print('Development study complete. No app replacement or test evaluation.',flush=True)


if __name__=='__main__':main()
