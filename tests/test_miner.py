import unittest

from test_data import rag, ts

from frm import Miner


class TestMiner(unittest.TestCase):
    def test_frm_miner(self):
        miner = Miner(0.5, 1, 3)
        motifs = miner.mine(ts)
        patterns = sorted([m.pattern for m in motifs])
        self.assertListEqual(patterns, ['aa', 'ca', 'cc'])

    def test_rag_miner(self):
        miner = Miner(0.5, 1, 3)
        motifs = miner.mine(rag)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['abc'])
