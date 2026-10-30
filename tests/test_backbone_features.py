"""Ensure larger-backbone features retain dimension and shared-head compatibility."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_clip_backbones import unit_vectors, stable_preprocessing
from model_inference import features, score_saved


class BackboneFeaturesTest(unittest.TestCase):
    def test_process_address_only(self):
        self.assertEqual(stable_preprocessing("<function rgb at 0xabc>"), stable_preprocessing("<function rgb at 0x123>"))
        self.assertNotEqual(stable_preprocessing("Resize(224)"), stable_preprocessing("Resize(256)"))

    def test_dimensions_and_linear_head(self):
        for d in (512,768):
            rng=np.random.default_rng(42)
            v=unit_vectors(rng.normal(size=(3,d)),d)
            t=unit_vectors(rng.normal(size=(3,d)),d)
            np.testing.assert_allclose(np.linalg.norm(v,axis=1),1,atol=1e-6)
            for mode,n in [('concat',2*d),('interaction',4*d+1)]:
                x=features((v,t),mode)
                self.assertEqual(x.shape,(3,n))
                p=dict(feature_mode=mode,mean=np.zeros(n),scale=np.ones(n),coef=np.ones((1,n)),intercept=np.array([.2]))
                np.testing.assert_allclose(score_saved((v,t),p),x.astype(np.float64).sum(axis=1)+.2,atol=1e-6)
            with self.assertRaises(ValueError):unit_vectors(np.ones((3,d+1)),d)
            with self.assertRaises(ValueError):unit_vectors(np.zeros((3,d)),d)
