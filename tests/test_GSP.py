import unittest

from motifminer.GSP import GSP
from .test_data import seq, patterns

class TestMotif(unittest.TestCase):
    def test_mine(self):
        gsp = GSP(seq)
        gsp.mine()
        
        self.assertListEqual(sorted(gsp.frequent.keys()), patterns)
