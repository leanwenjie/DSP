"""Verify new app route against all cached VERITE predictions and raw inputs."""
import csv
import json
import time
import numpy as np
from app_inference import Detector,MODELS,validate_input
from prepare_data import ROOT,digest


def main():
    rows=list(csv.DictReader((ROOT/'data/verite/VERITE.csv').open()))
    folder=ROOT/'artifacts/backbone_evaluation';slug='ViT-L-14_verite'
    arrays=[]
    for modality,key in [('image','image_path'),('text','caption')]:
        unique=sorted({r[key] for r in rows});index={v:i for i,v in enumerate(unique)}
        a=np.load(folder/f'{slug}_{modality}.npy',allow_pickle=False)
        arrays.append(a[[index[r[key]] for r in rows]])
    detector=Detector();name=MODELS[0]
    with np.load(folder/f'{slug}_predictions.npz',allow_pickle=False) as expected:
        for i in range(len(rows)):
            score,_,flag,_,_=detector.classify_vectors([a[i:i+1] for a in arrays],name)
            np.testing.assert_allclose(score,expected['score'][i],atol=1e-5,rtol=1e-5)
            assert flag==bool(expected['predictions'][i])
        cases=[]
        for i in [0,1,2,3,4,5]:
            image,text=validate_input((ROOT/'data/verite'/rows[i]['image_path']).read_bytes(),rows[i]['caption'])
            started=time.monotonic();r=detector.predict(image,text,name)
            np.testing.assert_allclose(r['score'],expected['score'][i],atol=1e-4,rtol=1e-4)
            assert r['flagged']==bool(expected['predictions'][i])
            cases.append(dict(row=i,total_seconds=time.monotonic()-started,**r))
    report=dict(backbone='ViT-L/14',cached_prediction_checks=len(rows),raw_input_checks=len(cases),cases=cases,
        source_hashes={p:digest(ROOT/p) for p in ['scripts/app_inference.py','app.py','artifacts/backbones/ViT-L-14/best_model.npz']},
        timing_note='Small six-case CPU sample; first total includes encoder loading. Analysis seconds exclude loading. Not laptop measurements.')
    (ROOT/'reports/app/l14_integration.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed: 1001 numeric predictions and six real-image app predictions.',flush=True)


if __name__=='__main__':main()
