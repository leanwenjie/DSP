"""Local upload inference with frozen benchmark models; no factual verification."""
import io
import json
import threading
import time
import warnings
import numpy as np
import torch
import clip
from PIL import Image, UnidentifiedImageError
from prepare_data import ROOT, digest
from train_logistic import predict_saved, features
from train_mlp import saved_probability

MODELS = ('Logistic regression', 'CLIP similarity', 'MLP seed 42')
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def validate_input(content, caption):
    if not content:
        raise ValueError('Upload an image first.')
    if len(content) > MAX_BYTES:
        raise ValueError('Use an image smaller than 10 MB.')
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError('Enter a caption to assess.')
    caption = caption.strip()
    if len(caption) > 4000:
        raise ValueError('Shorten the caption; CLIP accepts at most 75 text tokens.')
    try:
        clip.tokenize([caption], truncate=False)
    except RuntimeError as exc:
        raise ValueError('The caption exceeds CLIP’s 75-text-token limit. Shorten it; it will not be silently truncated.') from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in ('JPEG', 'PNG', 'WEBP'):
                    raise ValueError('Use a JPEG, PNG or WebP image.')
                if getattr(source, 'n_frames', 1) != 1:
                    raise ValueError('Use a single still image, not an animation.')
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError('Resize the image to at most 20 megapixels.')
                source.load()
                image = source.convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('The image could not be decoded safely. Try another JPEG, PNG or WebP file.') from exc
    return image, caption


class Detector:
    def __init__(self):
        self.lock = threading.Lock()
        self.paths = {
            'Logistic regression': ROOT/'artifacts/logistic/best_model.npz',
            'MLP seed 42': ROOT/'artifacts/mlp/model_seed42.npz',
            'CLIP similarity': ROOT/'reports/baseline/metrics.json',
        }
        manifest = ROOT/'reports/newsclippings_test/frozen_manifest.json'
        try:
            self.hashes = json.loads(manifest.read_text())['sha256']
            for path in self.paths.values():
                self.verify(path)
        except (OSError, KeyError) as exc:
            raise RuntimeError('Frozen model files are missing. Restore the project model artifacts before starting inference.') from exc
        torch.set_num_threads(2)
        self.model, self.preprocess = clip.load('ViT-B/32', device='cpu', jit=False, download_root=str(ROOT/'artifacts/clip'))
        self.model.eval()

    def verify(self, path):
        if digest(path) != self.hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError('A frozen model file has changed. Restore the evaluated artifact before inference.')

    def classify_vectors(self, vectors, model_name):
        if model_name not in MODELS:
            raise ValueError('Unknown model.')
        path = self.paths[model_name]
        self.verify(path)
        if model_name == 'CLIP similarity':
            cosine = float(np.einsum('ij,ij->i', *vectors, dtype=np.float64)[0])
            threshold = float(json.loads(path.read_text())['threshold'])
            return cosine, threshold, cosine < threshold, 'cosine similarity', 'below'
        if model_name == 'Logistic regression':
            score, prediction = predict_saved(vectors, path)
            with np.load(path, allow_pickle=False) as d:
                threshold = -float(d['threshold'])
            return float(score[0]), threshold, bool(prediction[0]), 'decision score', 'above'
        with np.load(path, allow_pickle=False) as d:
            x = features(vectors, 'interaction').astype(np.float32)
            x -= d['mean'].astype(x.dtype); x /= d['scale'].astype(x.dtype)
            score = saved_probability(x, d)
            threshold = -float(d['threshold'])
            return float(score[0]), threshold, bool(-score[0] < float(d['threshold'])), 'sigmoid score (uncalibrated)', 'above'

    def predict(self, image, caption, model_name):
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            v = self.model.encode_image(self.preprocess(image).unsqueeze(0)).float().numpy()
            t = self.model.encode_text(clip.tokenize([caption], truncate=False)).float().numpy()
            vectors = []
            for a in (v,t):
                norm = np.linalg.norm(a,axis=1,keepdims=True)
                if not np.isfinite(a).all() or np.any(norm==0):
                    raise RuntimeError('The encoder returned invalid features.')
                vectors.append(a/norm)
            score, threshold, flagged, score_type, direction = self.classify_vectors(vectors, model_name)
        return dict(model=model_name, flagged=flagged, score=score, threshold=threshold,
                    score_type=score_type, flag_when=direction, seconds=time.perf_counter()-started)
