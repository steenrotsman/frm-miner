import unittest

from frm._frm_py.patterns import PatternMiner

from .data import seq_1, rseq_1


class TestPatternMiner(unittest.TestCase):
    def test_lcs(self):
        pm = PatternMiner()

        self.assertEqual(pm.lcs('bbbbbbbbbb', 'bbbcbbb', 10, 7), 6)

    def test_mine(self):
        pm = PatternMiner(min_len=1, max_len=0, max_overlap=1)
        pm.mine(seq_1)

        expected = ['a', 'aa', 'c', 'ca', 'cc']
        
        self.assertListEqual(expected, sorted(pm.frequent.keys()))
    
    def test_rag_mine(self):
        pm = PatternMiner(min_len=1, max_overlap=1.1)
        pm.mine(rseq_1)

        expected = ['a', 'ab', 'abc', 'b', 'bc', 'c']
        
        self.assertListEqual(expected, sorted(pm.frequent.keys()))
