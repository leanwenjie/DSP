import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_verite import bootstrap_macro_f1, story_id


class VeriteEvaluationTests(unittest.TestCase):
    def test_related_images_share_story(self):
        self.assertEqual(story_id('images/true_12.jpg'), story_id('images/false_12.jpeg'))
        with self.assertRaises(ValueError):
            story_id('images/unrecognised.jpg')

    def test_cluster_interval_matches_explicit_story_resampling(self):
        y = np.array([0,1,1,0,1,0,1,1])
        pred = np.array([0,1,0,1,1,0,0,0])
        groups = np.array([5,5,5,9,9,12,12,12])
        rng = np.random.default_rng(42)
        unique = np.unique(groups)
        scores = []
        for selected in rng.integers(0,len(unique),size=(100,len(unique))):
            indices = np.concatenate([np.flatnonzero(groups==unique[i]) for i in selected])
            scores.append(f1_score(y[indices],pred[indices],average='macro',labels=[0,1],zero_division=0))
        np.testing.assert_allclose(bootstrap_macro_f1(y,pred,groups,100),np.quantile(scores,[.025,.975]))
