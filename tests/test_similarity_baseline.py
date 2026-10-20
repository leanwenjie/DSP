import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from similarity_baseline import select_threshold, load_split

class ThresholdTests(unittest.TestCase):
    def test_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            scores = np.round(rng.uniform(-1, 1, 40), 1)
            labels = np.r_[0, 1, rng.integers(0, 2, 38)]
            threshold, best, candidates, _ = select_threshold(scores, labels)
            brute = [f1_score(labels, scores < t, average='macro') for t in candidates]
            self.assertAlmostEqual(best, max(brute))
            self.assertAlmostEqual(f1_score(labels, scores < threshold, average='macro'), best)

    def test_perfect_separation(self):
        threshold, best, _, _ = select_threshold([.1, .2, .7, .8], [1, 1, 0, 0])
        self.assertEqual(best, 1)
        self.assertTrue(.2 < threshold <= .7)

    def test_test_split_blocked(self):
        with self.assertRaises(ValueError):
            load_split('test')

    def test_invalid_input(self):
        for scores, labels in [([], []), ([float('nan'), 0], [0, 1]), ([0, 1], [1, 1])]:
            with self.assertRaises(ValueError):
                select_threshold(scores, labels)

if __name__ == '__main__':
    unittest.main()
