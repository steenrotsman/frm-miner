import unittest

from frm.patterns import PatternMiner
from test_data import seq, rseq


class TestPatternMiner(unittest.TestCase):
    def test_lcs(self):
        pm = PatternMiner()

        self.assertEqual(pm.lcs('bbbbbbbbbb', 'bbbcbbb', 10, 7), 6)

    def test_mine(self):
        pm = PatternMiner(min_len=1, max_overlap=1.1)
        pm.mine(seq)

        expected = [
            'a', 'aa', 'ab',
            'b', 'ba', 'baa',
            'c', 'cb'
        ]
        
        self.assertListEqual(sorted(pm.frequent.keys()), expected)
    
    def test_rag_mine(self):
        pm = PatternMiner(min_len=1, max_overlap=1.1)
        pm.mine(rseq)

        expected = ['a', 'b']
        
        self.assertListEqual(sorted(pm.frequent.keys()), expected)
