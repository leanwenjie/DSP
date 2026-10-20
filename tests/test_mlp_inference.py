import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_mlp import saved_probability

class MLPInferenceTests(unittest.TestCase):
    def test_float32_model_returns_float64_scores_for_thresholds(self):
        params = dict(w0=np.ones((2, 1), dtype=np.float32), b0=np.zeros(1, dtype=np.float32),
                      w1=np.ones((1, 1), dtype=np.float32), b1=np.zeros(1, dtype=np.float32))
        probability = saved_probability(np.zeros((1, 2), dtype=np.float32), params)
        self.assertEqual(probability.dtype, np.float64)
        threshold = np.nextafter(-probability[0], np.inf)
        self.assertTrue((-probability < threshold)[0])

    def test_relu_and_sigmoid(self):
        params = dict(w0=np.eye(2), b0=np.zeros(2), w1=np.array([[1.],[-1.]]), b1=np.zeros(1))
        actual = saved_probability(np.array([[-2.,-1.],[2.,1.],[1.,3.]]), params)
        np.testing.assert_allclose(actual, expit([0.,1.,-2.]))

    def test_batch_and_single_inference_agree(self):
        rng = np.random.default_rng(5)
        params = dict(w0=rng.normal(size=(4,3)), b0=rng.normal(size=3),
                      w1=rng.normal(size=(3,1)), b1=rng.normal(size=1))
        x = rng.normal(size=(7,4))
        np.testing.assert_allclose(saved_probability(x, params),
                                   np.concatenate([saved_probability(row[None,:], params) for row in x]))
