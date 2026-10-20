import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyse_similarity import summarise

class AnalysisTests(unittest.TestCase):
    def test_majority_accuracy_can_hide_zero_recall(self):
        y=np.array([0]*8+[1]*2)
        r=summarise(y,np.zeros(10,dtype=int))
        self.assertEqual(r['accuracy'],.8)
        self.assertEqual(r['balanced_accuracy'],.5)
        self.assertEqual(r['misleading_recall'],0)
        self.assertEqual(r['fn'],2)

    def test_missing_class_rate_is_undefined(self):
        r=summarise(np.zeros(3,dtype=int),np.zeros(3,dtype=int))
        self.assertIsNone(r['misleading_recall'])
        self.assertIsNone(r['balanced_accuracy'])
