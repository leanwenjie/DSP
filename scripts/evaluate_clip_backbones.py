"""Audit/freeze all three candidates, then resumable test and VERITE scoring."""
import csv
import fcntl
import json
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.metrics import classification_report
from threadpoolctl import threadpool_limits
from prepare_data import ROOT, digest
from model_inference import predict_saved
from train_clip_backbones import write_json, unit_vectors
from evaluate_newsclippings_test import connected_groups, bootstrap_scores
from evaluate_verite import story_id, measure

ART=ROOT/'artifacts/backbone_evaluation'
OUT=ROOT/'reports/backbone_evaluation'
MODELS={'ViT-B/32':ROOT/'artifacts/fresh_clip','ViT-B/16':ROOT/'artifacts/backbones/ViT-B-16','ViT-L/14':ROOT/'artifacts/backbones/ViT-L-14'}


def metric_path(name):
    return ROOT/'reports/fresh_clip/metrics.json' if name=='ViT-B/32' else ROOT/'reports/backbone_comparison'/name.replace('/','-')/'metrics.json'


def freeze():
    provenance={};audits={};thresholds={}
    for name,folder in MODELS.items():
        progress=json.loads((folder/'progress.json').read_text())
        if not progress.get('complete'):raise ValueError(f'Incomplete development: {name}')
        manifest=json.loads((folder/'extraction_manifest.json').read_text())
        source_aliases={}
        for relative,sha in manifest['sha256'].items():
            path=ROOT/relative
            if digest(path)!=sha:
                backup=ROOT/'artifacts/backbones/trainer_before_resume_fix.py'
                if relative!='scripts/train_clip_backbones.py' or digest(backup)!=sha:raise ValueError(f'Changed development input: {relative}')
                source_aliases[relative]=str(backup.relative_to(ROOT));path=backup
            provenance[str(path.relative_to(ROOT))]=sha
        metrics=json.loads(metric_path(name).read_text());dim=768 if name=='ViT-L/14' else 512
        if metrics['selected']['mode']!='interaction':raise ValueError('Primary comparison requires interaction winner')
        model=folder/'best_model.npz'
        if digest(model)!=metrics['model_sha256']:raise ValueError('Changed selected model')
        for p in [model,folder/'extraction_manifest.json',metric_path(name)]:provenance[str(p.relative_to(ROOT))]=digest(p)
        checks={};val=None
        for split in ('train','val'):
            rows=json.loads((ROOT/f'data/newsclippings/{split}.json').read_text())['annotations'];vectors=[]
            for modality,key in [('image','image_id'),('text','id')]:
                ids=sorted({r[key] for r in rows});p=folder/f'{split}_{modality}.npy'
                a=np.load(p,mmap_mode='r',allow_pickle=False)
                if a.shape!=(len(ids),dim) or progress['completed'][f'{split}_{modality}']!=len(ids):raise ValueError('Incomplete features')
                if not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5):raise ValueError('Invalid features')
                checks[f'{split}_{modality}']=digest(p);provenance[str(p.relative_to(ROOT))]=checks[f'{split}_{modality}']
                if split=='val':
                    index={k:i for i,k in enumerate(ids)};vectors.append(a[[index[r[key]] for r in rows]])
            if split=='val':val=(vectors,np.array([int(r['falsified']) for r in rows]))
        with threadpool_limits(limits=2):score,pred=predict_saved(val[0],model)
        with np.load(folder/'validation_predictions.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(val[1],saved['labels']);np.testing.assert_array_equal(pred,saved['predictions']);np.testing.assert_allclose(score,saved['score'],atol=1e-6,rtol=1e-6)
        actual=classification_report(val[1],pred,output_dict=True)['macro avg']['f1-score']
        if abs(actual-metrics['selected']['validation_macro_f1'])>1e-12:raise ValueError('Metric mismatch')
        audits[name]=dict(feature_hashes=checks,validation_macro_f1=actual,inference_parity=True,historical_source_aliases=source_aliases)
        with np.load(model,allow_pickle=False) as m:thresholds[name]=dict(negative_decision_threshold=float(m['threshold']),cosine_threshold=metrics['similarity']['threshold'])
    for p in [OUT/'PROTOCOL.md',ROOT/'scripts/evaluate_clip_backbones.py',ROOT/'scripts/evaluate_newsclippings_test.py',ROOT/'scripts/evaluate_verite.py',ROOT/'scripts/train_clip_backbones.py',ROOT/'data/newsclippings/test.json',ROOT/'data/verite/VERITE.csv',ROOT/'reports/verite/evaluation/frozen_manifest.json',ROOT/'reports/backbone_comparison/resume_fix.json']:
        provenance[str(p.relative_to(ROOT))]=digest(p)
    record=dict(sha256=provenance,audits=audits,thresholds=thresholds)
    p=OUT/'frozen_manifest.json'
    if p.exists() and json.loads(p.read_text())!=record:raise ValueError('Frozen evaluation manifest changed')
    write_json(p,record)
    print('All three development audits passed; frozen before evaluation.',flush=True)
    return record


def main():
    ART.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    lock=(ART/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=freeze();torch.set_num_threads(2)
    meta={r['id']:r for r in json.loads((ROOT/'data/visualnews/data.json').read_text())}
    hashes={r['id']:r['sha256'] for r in map(json.loads,(ROOT/'data/visualnews/downloaded_images.jsonl').read_text().splitlines())}
    news=json.loads((ROOT/'data/newsclippings/test.json').read_text())['annotations']
    with (ROOT/'data/verite/VERITE.csv').open() as f:verite=list(csv.DictReader(f))
    assert len(news)==7264 and len(verite)==1001
    verite_hashes=json.loads((ROOT/'reports/verite/evaluation/frozen_manifest.json').read_text())['image_sha256']
    state_path=ART/'progress.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'completed':{},'stage':'extraction'}
    for name,folder in MODELS.items():
        slug=name.replace('/','-');dim=768 if name=='ViT-L/14' else 512
        model,preprocess=clip.load(name,device='cpu',jit=False,download_root=str(ROOT/'artifacts/clip'));model.eval()
        for task,rows in [('newsclippings',news),('verite',verite)]:
            output=OUT/f'{slug}_{task}.json'
            if output.exists():continue
            keys=([r['image_id'] for r in rows],[r['id'] for r in rows]) if task=='newsclippings' else ([r['image_path'] for r in rows],[r['caption'] for r in rows])
            vectors=[];truncated=0
            with torch.inference_mode():
                for modality,row_keys in zip(['image','text'],keys):
                    unique=sorted(set(row_keys));key=f'{slug}_{task}_{modality}';p=ART/f'{key}.npy';n=state['completed'].get(key,0)
                    if n and not p.exists():raise ValueError('Missing resumable feature buffer')
                    a=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype=np.float32,shape=(len(unique),dim))
                    if a.shape!=(len(unique),dim) or not 0<=n<=len(unique):raise ValueError('Invalid buffer')
                    if task=='verite' and modality=='text':truncated=sum(len(clip.clip._tokenizer.encode(k))>75 for k in unique)
                    for start in range(n,len(unique),8):
                        batch=unique[start:start+8]
                        if modality=='image':
                            tensors=[]
                            for k in batch:
                                path=ROOT/'data/visualnews/origin'/meta[k]['image_path'].removeprefix('./') if task=='newsclippings' else ROOT/'data/verite'/k
                                expected=hashes[k] if task=='newsclippings' else verite_hashes[k]
                                if digest(path)!=expected:raise ValueError('Evaluation image changed')
                                with Image.open(path) as im:tensors.append(preprocess(im.convert('RGB')))
                            raw=model.encode_image(torch.stack(tensors)).float().numpy()
                        else:
                            captions=[meta[k]['caption'] for k in batch] if task=='newsclippings' else batch
                            raw=model.encode_text(clip.tokenize(captions,truncate=task=='verite')).float().numpy()
                        a[start:start+len(batch)]=unit_vectors(raw,dim);a.flush()
                        state['completed'][key]=start+len(batch);state.update(stage='extraction',active=key,active_total=len(unique));write_json(state_path,state)
                        if start==n or start%400==0:print(key,start+len(batch),'/',len(unique),flush=True)
                    if not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5):raise ValueError('Invalid finished buffer')
                    index={k:i for i,k in enumerate(unique)};vectors.append(a[[index[k] for k in row_keys]])
            with threadpool_limits(limits=2):score,pred=predict_saved(vectors,folder/'best_model.npz')
            cos=np.einsum('ij,ij->i',*vectors,dtype=np.float64);baseline=(cos<frozen['thresholds'][name]['cosine_threshold']).astype(int)
            if task=='newsclippings':
                labels=np.array([int(r['falsified']) for r in rows]);groups=connected_groups(rows);masks={'test':np.ones(len(rows),dtype=bool)}
            else:
                labels=np.array([int(r['label']!='true') for r in rows]);groups=np.array([story_id(r['image_path']) for r in rows]);masks={'primary_true_vs_ooc':np.array([r['label']!='miscaptioned' for r in rows]),'secondary_all':np.ones(len(rows),dtype=bool)}
            results={}
            for title,mask in masks.items():
                results[title]={}
                for method,s,prediction in [('interaction_lr',score,pred),('cosine',-cos,baseline)]:
                    results[title][method]=measure(labels[mask],prediction[mask],s[mask],groups[mask])
                    if task=='verite':results[title][method]['detected_by_category']={label:int(prediction[np.array([r['label']==label for r in rows]) & mask].sum()) for label in ['true','miscaptioned','out-of-context']}
            np.savez(ART/f'{slug}_{task}_predictions.npz',labels=labels,score=score,predictions=pred,cosine=cos,baseline=baseline,groups=groups)
            write_json(output,dict(backbone=name,task=task,results=results,unique_truncated_verite_captions=truncated,model_sha256=digest(folder/'best_model.npz'),test_tuned=False))
            print('Finished',slug,task,flush=True)
        del model
    # Paired bootstrap differences against fresh B/32, using identical group draws.
    comparisons={}
    for task in ['newsclippings','verite']:
        masks={'test':np.ones(len(news),bool)} if task=='newsclippings' else {'primary_true_vs_ooc':np.array([r['label']!='miscaptioned' for r in verite]),'secondary_all':np.ones(len(verite),bool)}
        for title,mask in masks.items():
            boots={}
            for name in MODELS:
                with np.load(ART/f"{name.replace('/','-')}_{task}_predictions.npz",allow_pickle=False) as r:boots[name]=bootstrap_scores(r['labels'][mask],r['predictions'][mask],r['groups'][mask])
            comparisons[f'{task}_{title}']={name:np.quantile(b-boots['ViT-B/32'],[.025,.975]).tolist() for name,b in boots.items() if name!='ViT-B/32'}
    for p,h in frozen['sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError(f'Frozen file changed: {p}')
    write_json(OUT/'paired_macro_f1_intervals.json',comparisons)
    state.update(stage='complete',complete=True);write_json(state_path,state)
    print('All frozen follow-up evaluations complete; app unchanged.',flush=True)


if __name__=='__main__':main()
