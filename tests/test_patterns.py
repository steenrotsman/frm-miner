import unittest

from frm._frm_py.patterns import PatternMiner
from test_data import seq, rseq


class TestPatternMiner(unittest.TestCase):
    def test_lcs(self):
        pm = PatternMiner()

        self.assertEqual(pm.lcs('bbbbbbbbbb', 'bbbcbbb', 10, 7), 6)

    def test_mine(self):
        pm = PatternMiner(0.5, 1, 1)
        pm.mine(seq)

        expected = ['a', 'aa', 'c', 'ca', 'cc']
        
        self.assertListEqual(expected, sorted(pm.frequent.keys()))
    
    def test_rag_mine(self):
        pm = PatternMiner(min_len=1, max_overlap=1.1)
        pm.mine(rseq)

        expected = ['a', 'ac', 'c']
        
        self.assertListEqual(expected, sorted(pm.frequent.keys()))
