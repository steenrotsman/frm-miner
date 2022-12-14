import unittest

from motifminer.GSP import GSP
from motifminer.preprocessing import sax
from .test_data import ts, rag

class TestMotif(unittest.TestCase):
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
