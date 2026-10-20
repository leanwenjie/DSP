import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_newsclippings_test import connected_groups, bootstrap_scores

class TestEvaluation(unittest.TestCase):
    def test_transitive_shared_ids_are_grouped(self):
        rows=[{'id':1,'image_id':2},{'id':3,'image_id':2},{'id':3,'image_id':4},{'id':8,'image_id':8}]
        g=connected_groups(rows)
        self.assertEqual(len(set(g[:3])),1)
        self.assertNotEqual(g[0],g[3])

    def test_group_bootstrap_matches_explicit_resampling(self):
        y=np.array([0,1,0,1,0,1]); p=np.array([0,0,0,1,1,1]); g=np.array([1,1,2,2,3,3])
        rng=np.random.default_rng(42); expected=[]
        for selected in rng.integers(0,3,size=(111,3)):
            idx=np.concatenate([np.flatnonzero(g==s+1) for s in selected])
            expected.append(f1_score(y[idx],p[idx],average='macro',labels=[0,1],zero_division=0))
        np.testing.assert_allclose(bootstrap_scores(y,p,g,111),expected)
