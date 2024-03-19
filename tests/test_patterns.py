import unittest

from test_data import rseq_1, seq_1

from frm.patterns import PatternMiner


class TestPatternMiner(unittest.TestCase):
    def test_lcs(self):
        pm = PatternMiner()

        self.assertEqual(pm.lcs('bbbbbbbbbb', 'bbbcbbb', 10, 7), 6)

    def test_mine(self):
        pm = PatternMiner(min_len=1, max_overlap=1)
        pm.mine(seq_1)

        expected = ['a', 'aa', 'c', 'ca', 'cc']

        self.assertListEqual(expected, sorted(pm.frequent))

    def test_rag_mine(self):
        pm = PatternMiner(min_len=1, max_overlap=1)
        pm.mine(rseq_1)

        expected = ['a', 'ab', 'abc', 'b', 'bc', 'c']

        self.assertListEqual(expected, sorted(pm.frequent))
