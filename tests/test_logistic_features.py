import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_logistic import features, predict_saved
from sklearn.preprocessing import StandardScaler

class FeatureTests(unittest.TestCase):
    def test_saved_scaling_matches_sklearn(self):
        rng = np.random.default_rng(5)
        vectors = [rng.normal(size=(30, 8)).astype(np.float32) for _ in range(2)]
        x = features(vectors, 'interaction')
        scaler = StandardScaler().fit(x)
        coef = rng.normal(size=(1, x.shape[1]))
        expected = (scaler.transform(x) @ coef.T).ravel() + .3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.npz'
            np.savez(path, mean=scaler.mean_, scale=scaler.scale_, coef=coef,
                     intercept=[.3], threshold=.1, feature_mode='interaction')
            decision, prediction = predict_saved(vectors, path)
        np.testing.assert_allclose(decision, expected, atol=1e-12)
        np.testing.assert_array_equal(prediction, -expected < .1)

    def test_interactions_distinguish_aligned_from_opposite(self):
        image = np.array([[1., 0.], [1., 0.]], dtype=np.float32)
        text = np.array([[1., 0.], [-1., 0.]], dtype=np.float32)
        x = features([image, text], 'interaction')
        np.testing.assert_array_equal(x[:, -1], [1., -1.])
        np.testing.assert_array_equal(x[:, 4:6], [[0., 0.], [2., 0.]])
        np.testing.assert_array_equal(x[:, 6:8], [[1., 0.], [-1., 0.]])

    def test_row_permutation_is_preserved(self):
        rng = np.random.default_rng(4)
        vectors = [rng.normal(size=(8, 512)).astype(np.float32) for _ in range(2)]
        order = rng.permutation(8)
        for mode, width in [('concat', 1024), ('interaction', 2049)]:
            original = features(vectors, mode)
            self.assertEqual(original.shape, (8, width))
            np.testing.assert_allclose(features([v[order] for v in vectors], mode), original[order])
