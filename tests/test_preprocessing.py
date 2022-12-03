import unittest

import numpy as np

from motifminer import preprocessing
from .test_data import ts, seq

class TestPreprocessing(unittest.TestCase):
    def test_standardize(self):
        std = np.std(ts)
        
        a = -0.7/std
        b = 0.3/std
        c = 1.3/std

        got = preprocessing._standardize(ts)
        
        expected = np.array([
            [a, b, c, c, b, a],
            [a, a, b, b, a, a],
            [c, b, b, a, a, a],
            [c, b, a, a, b, c],
            [a, b, a, b, a, b]
        ])

        self.assertTrue((np.round(got, 15) == np.round(expected, 15)).all())

    def test_paa_w_1(self):
        paa = preprocessing._paa(ts, 1)

        self.assertTrue((paa == ts).all())
    
    def test_paa_w_2(self):
        paa = preprocessing._paa(ts, 2)

        expected = np.array([
            [0.5, 2, 0.5],
            [0, 1, 0],
            [1.5, 0.5, 0],
            [1.5, 0, 1.5],
            [0.5, 0.5, 0.5]
        ])

        self.assertTrue((paa == expected).all())
    
    def test_paa_w_3(self):
        paa = preprocessing._paa(ts, 3)

        expected = np.array([
            [1, 1],
            [1/3, 1/3],
            [4/3, 0],
            [1, 1],
            [1/3, 2/3]
        ])

        self.assertTrue((paa == expected).all())
    
    def test_fast_sax(self):
        sax = preprocessing._sax(ts, 1, 3)

        self.assertTrue((sax == ts).all())
    
    def test_sax(self):
        sax = preprocessing.sax(ts, 1, 3)

        self.assertEqual(sax, seq)

