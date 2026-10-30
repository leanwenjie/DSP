"""Audit completed features, saved inference and freeze the fresh B/32 candidate."""
import json
import numpy as np
from sklearn.metrics import classification_report
from threadpoolctl import threadpool_limits
from prepare_data import ROOT, digest
from model_inference import predict_saved


def main():
    art=ROOT/'artifacts/fresh_clip';out=ROOT/'reports/fresh_clip'
    progress=json.loads((art/'progress.json').read_text())
    if not progress.get('complete'):raise ValueError('Extraction/training incomplete')
    provenance=json.loads((art/'extraction_manifest.json').read_text())
    for p,h in provenance['sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError(f'Changed input: {p}')
    result={};val=None
    for split in ('train','val'):
        rows=json.loads((ROOT/f'data/newsclippings/{split}.json').read_text())['annotations']
        vectors=[]
        for modality,key in [('image','image_id'),('text','id')]:
            ids=sorted({r[key] for r in rows});name=f'{split}_{modality}'
            a=np.load(art/f'{name}.npy',mmap_mode='r',allow_pickle=False)
            if a.shape!=(len(ids),512) or progress['completed'][name]!=len(ids):raise ValueError('Invalid feature coverage')
            norms=np.linalg.norm(a,axis=1)
            if not np.isfinite(a).all() or not np.allclose(norms,1,atol=1e-5):raise ValueError('Invalid feature values')
            result[name]=dict(shape=list(a.shape),norm_min=float(norms.min()),norm_max=float(norms.max()),sha256=digest(art/f'{name}.npy'))
            if split=='val':
                index={k:i for i,k in enumerate(ids)};vectors.append(a[[index[r[key]] for r in rows]])
        if split=='val':val=(vectors,np.array([int(r['falsified']) for r in rows]))
    with threadpool_limits(limits=2):score,pred=predict_saved(val[0],art/'best_model.npz')
    with np.load(art/'validation_predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['labels'],val[1]);np.testing.assert_array_equal(saved['predictions'],pred)
        np.testing.assert_allclose(saved['score'],score,rtol=1e-6,atol=1e-6)
    metrics=json.loads((out/'metrics.json').read_text())
    report=classification_report(val[1],pred,output_dict=True)
    assert abs(report['macro avg']['f1-score']-metrics['selected']['validation_macro_f1'])<1e-12
    assert digest(art/'best_model.npz')==metrics['model_sha256']
    paths=[art/'best_model.npz',art/'extraction_manifest.json',out/'PROTOCOL.md',out/'metrics.json',ROOT/'scripts/model_inference.py']
    frozen={'sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths},'feature_files':result,'selection_split':'val','test_evaluated':False,'policy':'Freeze before follow-up test/external evaluation; original test results already known.'}
    path=out/'frozen_manifest.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Existing frozen manifest differs')
    path.write_text(json.dumps(frozen,indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps(dict(feature_checks=result,saved_inference_parity=True,metric_reconciliation=True,validation_report=report),indent=2)+'\n')
    print('Fresh B/32 features, inference and metrics verified; candidate frozen.')


if __name__=='__main__':main()
