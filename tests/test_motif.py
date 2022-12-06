import unittest

import numpy as np

from motifminer.motif import Motif

class TestMotif(unittest.TestCase):
    def test_init(self):
        a = Motif(pattern='abc')
        self.assertEqual(a.pattern, 'abc')
        self.assertEqual(a.indexes, {})
        self.assertAlmostEqual(a.occurences.size, 0)
        self.assertEqual(a.representative.size, 0)
    
    def test_record_index(self):
        a = Motif(pattern='abc')
        a.record_index(0, 1)
        self.assertEqual(list(a.indexes.keys()), [0])
        self.assertEqual(a.indexes[0], [1])
