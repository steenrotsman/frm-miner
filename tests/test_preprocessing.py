import unittest

import tensorflow as tf

from motifminer import preprocessing
from .test_data import ts, rag

class TestPreprocessing(unittest.TestCase):
    def test_standardize(self):
        std = tf.math.reduce_std(ts)
        
        a = -0.7/std
        b = 0.3/std
        c = 1.3/std

        got = preprocessing.standardize(ts)
        
        expected = [
            [a, b, c, c, b, a],
            [a, a, b, b, a, a],
            [c, b, b, a, a, a],
            [c, b, a, a, b, c],
            [a, b, a, b, a, b]
        ]

        self.assertTrue(tf.math.reduce_all(expected==got))

    def test_paa_w_1(self):
        paa = preprocessing.paa(ts, 1)
        expected = ts
        
        self.assertTrue(tf.math.reduce_all(expected==paa))
    
    def test_paa_w_2(self):
        paa = preprocessing.paa(ts, 2)

        expected = [
            [0.5, 2, 0.5],
            [0, 1, 0],
            [1.5, 0.5, 0],
            [1.5, 0, 1.5],
            [0.5, 0.5, 0.5]
        ]

        self.assertTrue(tf.math.reduce_all(expected==paa))
    
    def test_paa_w_3(self):
        paa = preprocessing.paa(ts, 3)

        expected = [
            [1, 1],
            [1/3, 1/3],
            [4/3, 0],
            [1, 1],
            [1/3, 2/3]
        ]

        self.assertTrue(tf.math.reduce_all(expected==paa))

    def test_paa_rag_1(self):
        paa = preprocessing.paa(rag, 1)

        self.assertTrue(tf.math.reduce_all(rag==paa))
        
    def test_paa_rag_2(self):
        paa = preprocessing.paa(rag, 2)

        expected = tf.ragged.constant([
            [1.5],
            [3.5, 5],
            [6.5, 8.5]
        ])

        self.assertTrue(tf.math.reduce_all(expected==paa))

    def test_paa_rag_3(self):
        paa = preprocessing.paa(rag, 3)

        expected = tf.ragged.constant([
            [1.5],
            [4],
            [7, 9]
        ])

        self.assertTrue(tf.math.reduce_all(expected==paa))
    
    def test_sax(self):
        sax = preprocessing.sax(ts, 1, 3)

        expected = [
            'abccba',
            'aabbaa',
            'cbbaaa',
            'cbaabc',
            'ababab'
        ]

        self.assertEqual(sax, expected)
    
    def test_rag_sax(self):
        sax = preprocessing.sax(rag, 1, 3)

        expected = [
            'aa',
            'abb',
            'bccc'
        ]

        self.assertEqual(sax, expected)
