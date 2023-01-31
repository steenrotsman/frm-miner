import unittest

from motifminer.PatternMiner import PatternMiner
from motifminer.preprocessing import sax
from test_data import ts, rag


class TestMotif(unittest.TestCase):
    def test_lcs(self):
        pm = PatternMiner([])

        self.assertEqual(pm.lcs('bbbbbbbbbb', 'bbbcbbb', 10, 7), 6)

    def test_mine(self):
        seq = sax(ts, 1, 3)
        pm = PatternMiner(seq)
        pm.mine()

        expected = [
            'a', 'aa', 'ab',
            'b', 'ba', 'baa',
            'c', 'cb'
        ]
        
        self.assertListEqual(sorted(pm.frequent.keys()), expected)
    
    def test_rag_mine(self):
        seq = sax(rag, 1, 3)
        pm = PatternMiner(seq)
        pm.mine()

        expected = ['a', 'b']
        
        self.assertListEqual(sorted(pm.frequent.keys()), expected)
