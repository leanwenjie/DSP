"""Validate official, SHA256-verified embeddings against each benchmark split."""
import json
import pickle
import numpy as np
from prepare_data import DATA, ROOT, digest

results = {}
for split in ['train', 'val', 'test']:
    annotations = json.loads((DATA / f'newsclippings/{split}.json').read_text())['annotations']
    for modality, field in [('image', 'image_id'), ('text', 'id')]:
        name = f'clip_{modality}_embeddings'
        path = DATA / f'newsclippings/embeddings/{name}/{name}_{split}.pkl'
        receipt = json.loads(path.with_name(path.name + '.receipt.json').read_text())
        if not receipt.get('upstream_sha256') or digest(path) != receipt['upstream_sha256']:
            raise ValueError('Refusing to load embeddings without a matching upstream SHA256')
        # Pickle is executable: use only these pinned, hash-verified official files.
        with path.open('rb') as f:
            embeddings = pickle.load(f)
        required = {r[field] for r in annotations}
        missing = required - embeddings.keys()
        bad = 0
        for key in required - missing:
            v = np.asarray(embeddings[key])
            if v.shape != (512,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
                bad += 1
        results[f'{split}/{modality}'] = dict(available=len(embeddings), required=len(required), missing=len(missing), invalid_vectors=bad)
        del embeddings
        print(split, modality, results[f'{split}/{modality}'], flush=True)
(ROOT / 'reports/embedding_audit.json').write_text(json.dumps(results, indent=2) + '\n')
if any(r['missing'] or r['invalid_vectors'] for r in results.values()):
    raise SystemExit('Embedding validation failed; see report')
