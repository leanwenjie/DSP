"""Shared NumPy inference for the frozen classifiers; no training imports."""
import numpy as np
from scipy.special import expit


def features(vectors, mode):
    image, text = vectors
    if image.shape != text.shape or image.ndim != 2:
        raise ValueError('Aligned image/text matrices required')
    if mode == 'concat':
        return np.concatenate([image, text], axis=1)
    if mode == 'interaction':
        cosine = np.einsum('ij,ij->i', image, text)[:, None]
        return np.concatenate([image, text, np.abs(image-text), image*text, cosine], axis=1)
    raise ValueError('Unknown feature mode')


def saved_probability(x, parameters):
    hidden = np.maximum(x @ parameters['w0'] + parameters['b0'], 0)
    return expit(hidden @ parameters['w1'] + parameters['b1']).ravel().astype(np.float64)


def score_saved(vectors, parameters):
    """Apply training-fitted float32 scaling and the stored linear/neural head."""
    x = features(vectors, str(parameters['feature_mode'])).astype(np.float32)
    # Preserve sklearn's in-place float32 transform and MLP float64 thresholds.
    x -= parameters['mean'].astype(x.dtype)
    x /= parameters['scale'].astype(x.dtype)
    if 'coef' in parameters:
        return (x @ parameters['coef'].T).ravel() + parameters['intercept'][0]
    return saved_probability(x, parameters)


def predict_saved(vectors, path):
    """Return scores and predictions using the stored validation threshold."""
    with np.load(path, allow_pickle=False) as saved:
        scores = score_saved(vectors, saved)
        return scores, (-scores < float(saved['threshold'])).astype(int)
