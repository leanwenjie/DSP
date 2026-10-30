"""Feature normalisation and atomic resume checkpoint roundtrip."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_fresh_clip import unit_vectors, write_json


class FreshFeatureTest(unittest.TestCase):
    def test_normalisation_and_checkpoint(self):
        x=np.arange(1024,dtype=np.float32).reshape(2,512)+1
        y=unit_vectors(x)
        np.testing.assert_allclose(np.linalg.norm(y,axis=1),1,atol=1e-6)
        np.testing.assert_allclose(y[0],x[0]/np.linalg.norm(x[0]))
        for bad in [np.zeros((2,512)),np.full((2,512),np.nan),np.ones((2,511))]:
            with self.assertRaises(ValueError):unit_vectors(bad)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'progress.json'
            write_json(p,{'completed':{'train_image':32}})
            write_json(p,{'completed':{'train_image':64}})
            self.assertEqual(json.loads(p.read_text())['completed']['train_image'],64)
            self.assertFalse(p.with_suffix('.tmp').exists())
