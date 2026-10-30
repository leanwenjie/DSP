"""Live app inference smoke checks and small CPU latency sample, without tuning."""
import csv
import json
import platform
import time
import numpy as np
from app_inference import Detector, MODELS, validate_input
from prepare_data import ROOT, digest


def main():
    rows=list(csv.DictReader((ROOT/'data/verite/VERITE.csv').open()))
    reference=list(csv.DictReader((ROOT/'artifacts/verite/predictions.csv').open()))
    evaluation=json.loads((ROOT/'reports/verite/evaluation/metrics.json').read_text())
    assert digest(ROOT/'artifacts/verite/predictions.csv')==evaluation['predictions_sha256']
    assert digest(ROOT/'artifacts/verite/features.npz')==evaluation['feature_sha256']
    names={'CLIP similarity':'Similarity','Logistic regression':'Logistic regression','MLP seed 42':'MLP seed 42'}
    expected={(int(r['row_index']),r['model']):r for r in reference}
    started=time.perf_counter(); detector=Detector(); cold=time.perf_counter()-started
    indices=np.random.default_rng(42).choice(len(rows),12,replace=False)
    parity=0
    with np.load(ROOT/'artifacts/verite/features.npz',allow_pickle=False) as d:
        for i in indices:
            for name in ('Logistic regression', 'CLIP similarity', 'MLP seed 42'):
                score,_,flagged,_,_=detector.classify_vectors([d['image'][i:i+1],d['text'][i:i+1]],name)
                ref=expected[(int(i),names[name])]
                expected_score=-float(ref['score']) if name=='CLIP similarity' else float(ref['score'])
                np.testing.assert_allclose(score,expected_score,atol=1e-5,rtol=1e-5)
                assert flagged==bool(int(ref['prediction']))
                parity+=1
    # Warm the encoder once; exclude this inference and initialization from warm timing.
    image,caption=validate_input((ROOT/'data/verite'/rows[0]['image_path']).read_bytes(),rows[0]['caption'])
    detector.predict(image,caption,'Logistic regression')
    cases=[]
    for i in indices:
        r=rows[int(i)]
        image,caption=validate_input((ROOT/'data/verite'/r['image_path']).read_bytes(),r['caption'])
        result=detector.predict(image,caption,'Logistic regression')
        ref=expected[(int(i),'Logistic regression')]
        assert result['flagged']==bool(int(ref['prediction']))
        np.testing.assert_allclose(result['score'],float(ref['score']),atol=1e-4,rtol=1e-4)
        cases.append(dict(row_index=int(i),label=r['label'],correct=result['flagged']==(r['label']!='true'),**result))
    # Exercise all three raw-input routes, not just their numeric heads.
    for name in ('Logistic regression', 'CLIP similarity', 'MLP seed 42'):
        result=detector.predict(image,caption,name)
        assert result['flagged']==bool(int(expected[(int(indices[-1]),names[name])]['prediction']))
    times=[c['seconds'] for c in cases]
    report=dict(device='CPU',python=platform.python_version(),sample_size=len(cases),seed=42,
                initialisation_seconds=cold,warm_median_seconds=float(np.median(times)),warm_p95_seconds=float(np.quantile(times,.95)),
                cached_vector_parity_checks=parity,raw_input_parity_checks=len(cases)+len(MODELS),cases=cases,
                limitations='Small reproducible cloud-CPU sample, not a load test or RTX 4050 benchmark. Timings exclude upload, validation and model initialization. Cases illustrate known benchmark behavior, not new accuracy estimates.')
    out=ROOT/'reports/app';out.mkdir(exist_ok=True)
    (out/'smoke_and_latency.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
