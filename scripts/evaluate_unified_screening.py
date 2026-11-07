"""Evaluate the fixed L/14 OR rule without fitting or modifying historical reports."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from app_inference import Detector, MODELS, validate_input
from prepare_data import ROOT, digest


def metrics(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return dict(count=len(labels), accuracy=accuracy_score(labels, predictions),
                macro_f1=f1_score(labels, predictions, average='macro', labels=[0,1], zero_division=0),
                misleading_precision=precision_score(labels, predictions, zero_division=0),
                misleading_recall=recall_score(labels, predictions, zero_division=0),
                false_positive_rate=float(fp/(tn+fp)), confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--banana-image',type=Path);args=parser.parse_args()
    folder=ROOT/'reports/unified_screening';folder.mkdir(exist_ok=True)
    frozen=json.loads((ROOT/'reports/backbone_evaluation/frozen_manifest.json').read_text())
    for path, expected in frozen['sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError(f'Changed frozen input: {path}')
    detector=Detector();art=ROOT/'artifacts/backbone_evaluation';report={};hashes={}
    verite=list(csv.DictReader((ROOT/'data/verite/VERITE.csv').open()))
    for task in ['newsclippings','verite']:
        rows=json.loads((ROOT/'data/newsclippings/test.json').read_text())['annotations'] if task=='newsclippings' else verite
        keys=['image_id','id'] if task=='newsclippings' else ['image_path','caption']
        vectors=[]
        for modality,key in zip(['image','text'],keys):
            p=art/f'ViT-L-14_{task}_{modality}.npy';hashes[str(p.relative_to(ROOT))]=digest(p)
            unique=sorted({r[key] for r in rows});index={value:i for i,value in enumerate(unique)}
            vectors.append(np.load(p,allow_pickle=False)[[index[r[key]] for r in rows]])
        p=art/f'ViT-L-14_{task}_predictions.npz';hashes[str(p.relative_to(ROOT))]=digest(p)
        with np.load(p,allow_pickle=False) as saved:
            labels=saved['labels'];old=saved['predictions'].astype(bool)
            predictions=[]
            for i in range(len(rows)):
                result=detector.screen_vectors([a[i:i+1] for a in vectors],MODELS[0])
                np.testing.assert_allclose(result['score'],saved['score'][i],atol=1e-5,rtol=1e-5)
                assert result['classifier_flagged']==bool(old[i])
                assert result['flagged']==bool(old[i] or saved['baseline'][i])
                predictions.append(result['flagged'])
            new=np.array(predictions)
            masks={'test':np.ones(len(rows),bool)} if task=='newsclippings' else {'primary':np.array([r['label']!='miscaptioned' for r in rows]),'all_misleading':np.ones(len(rows),bool)}
            for name,mask in masks.items():
                y=labels[mask];before=old[mask];after=new[mask]
                report[f'{task}_{name}']=dict(classifier_only=metrics(y,before),unified=metrics(y,after),
                    added_detections=int(np.sum((y==1)&~before&after)),added_false_alarms=int(np.sum((y==0)&~before&after)))
            if task=='verite':
                raw=[]
                for i in range(6):
                    image,text=validate_input((ROOT/'data/verite'/rows[i]['image_path']).read_bytes(),rows[i]['caption'])
                    result=detector.predict(image,text,MODELS[0])
                    np.testing.assert_allclose(result['score'],saved['score'][i],atol=1e-4,rtol=1e-4)
                    np.testing.assert_allclose(result['cosine'],saved['cosine'][i],atol=1e-4,rtol=1e-4)
                    assert result['flagged']==bool(new[i])
                    raw.append(dict(row=i,**result))
                report['raw_verite_checks']=raw
        print(f'Passed numeric checks: {task}, {len(rows)} rows',flush=True)
    if args.banana_image:
        captions=['A yellow banana on a white background.','A photograph of a piece of fruit.','A person standing in a room.','i am human']
        report['banana_diagnostic']=dict(image_sha256=digest(args.banana_image),
            note='Substitute public image, not the user upload. Four illustrative prompts; not a challenge-set accuracy estimate. First-person caption is ambiguous.',cases=[])
        for caption in captions:
            image,text=validate_input(args.banana_image.read_bytes(),caption)
            report['banana_diagnostic']['cases'].append(dict(caption=text,**detector.predict(image,text,MODELS[0])))
    report['provenance']=dict(cached_input_sha256=hashes,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),ROOT/'scripts/app_inference.py',folder/'PROTOCOL.md']},
        frozen_inputs_verified=len(frozen['sha256']),numeric_rows_checked=8265,threshold=detector.cosine_threshold,
        note='Exploratory fixed-rule follow-up on previously examined benchmarks. No fitting. Four-sentence banana diagnostic is not a generalisation estimate.')
    (folder/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved reports/unified_screening/metrics.json',flush=True)


if __name__=='__main__':main()
