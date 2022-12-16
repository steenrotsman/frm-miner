import unittest

from motifminer.GSP import GSP
from motifminer.preprocessing import sax
from .test_data import ts, rag

class TestMotif(unittest.TestCase):
    def test_lcs(self):
        gsp = GSP([])
        commons = ['a' * i for i in range(0, 100, 5)]
        for i, common in enumerate(commons):
            p1 = 'abc' + common + 'abc'
            p2 = 'def' + common + 'def'

            self.assertEqual(len(common), gsp.lcs(p1, p2, len(p1), len(p2)))

    def test_mine(self):
        seq = sax(ts, 1, 3)
        gsp = GSP(seq)
        gsp.mine()

        expected = [
            'a', 'aa', 'ab',
            'b', 'ba', 'baa',
            'c', 'cb'
        ]
        
        self.assertListEqual(sorted(gsp.frequent.keys()), expected)
    
    def test_rag_mine(self):
        seq = sax(rag, 1, 3)
        gsp = GSP(seq)
        gsp.mine()

        expected = ['a', 'b']
        
        self.assertListEqual(sorted(gsp.frequent.keys()), expected)
