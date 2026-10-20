import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_thresholds import screening_threshold

class ScreeningTests(unittest.TestCase):
    def test_target_and_minimum_flags_with_ties(self):
        rng=np.random.default_rng(7)
        for _ in range(30):
            scores=np.round(rng.random(60),1)
            y=np.r_[1,0,rng.integers(0,2,58)]
            threshold=screening_threshold(scores,y)
            flagged=scores>=threshold
            self.assertGreaterEqual(flagged[y==1].mean(),.8)
            eligible=[scores>=t for t in np.unique(scores) if (scores[y==1]>=t).mean()>=.8]
            self.assertEqual(flagged.sum(),min(p.sum() for p in eligible))

    def test_full_recall(self):
        self.assertEqual(screening_threshold([.1,.5,.9],[1,0,1],1),.1)

    def test_no_positives_rejected(self):
        with self.assertRaises(ValueError):screening_threshold([.2,.4],[0,0])
