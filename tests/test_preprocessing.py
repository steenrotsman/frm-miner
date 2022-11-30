import unittest

import numpy as np

from motifminer import preprocessing


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.ts = np.array([
            [0, 1, 2, 2, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [2, 1, 1, 0, 0, 0],
            [2, 1, 0, 0, 1, 2],
            [0, 1, 0, 1, 0, 1]
        ])
    
    def test_standardize(self):
        avg = np.mean(self.ts)
        std = np.std(self.ts)
        
        a = -0.7/std
        b = 0.3/std
        c = 1.3/std

        got = preprocessing._standardize(self.ts)
        
        expected = np.array([
            [a, b, c, c, b, a],
            [a, a, b, b, a, a],
            [c, b, b, a, a, a],
            [c, b, a, a, b, c],
            [a, b, a, b, a, b]
        ])

        self.assertTrue((np.round(got, 5) == np.round(expected, 5)).all())

    def test_paa_w_1(self):
        paa = preprocessing._paa(self.ts, 1)

        self.assertTrue((paa == self.ts).all())
    
    def test_paa_w_2(self):
        paa = preprocessing._paa(self.ts, 2)

        expected = np.array([
            [0.5, 2, 0.5],
            [0, 1, 0],
            [1.5, 0.5, 0],
            [1.5, 0, 1.5],
            [0.5, 0.5, 0.5]
        ])

        self.assertTrue((paa == expected).all())
    
    def test_paa_w_3(self):
        paa = preprocessing._paa(self.ts, 3)

        expected = np.array([
            [1, 1],
            [1/3, 1/3],
            [4/3, 0],
            [1, 1],
            [1/3, 2/3]
        ])

        self.assertTrue((paa == expected).all())
    
    def test_fast_sax(self):
        sax = preprocessing._sax(self.ts, 1, 3)

        self.assertTrue((sax == self.ts).all())
    
    def test_sax(self):
        sax = preprocessing.sax(self.ts, 1, 3)

        expected = [
            'abccba',
            'aabbaa',
            'cbbaaa',
            'cbaabc',
            'ababab'
        ]

        self.assertEqual(sax, expected)

