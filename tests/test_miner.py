import unittest

from frm.miner import Miner
from test_data import ts, rag


class TestMiner(unittest.TestCase):
    def test_frm_miner(self):
        miner = Miner(ts, 0.5, 1, 3, 1, 1.1)
        motifs = miner.mine_motifs()
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['a', 'aa', 'c', 'ca', 'cc'])

    def test_rag_miner(self):
        miner = Miner(rag, 0.5, 1, 3, 1, 1.1)
        motifs = miner.mine_motifs()
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['a', 'c', 'ac'])
