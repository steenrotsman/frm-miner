import unittest

from frm import Miner as CppMiner
from frm._frm_py.miner import Miner as PyMiner
from test_data import ts, rag


class TestMiner(unittest.TestCase):
    def test_frm_py_miner(self):
        miner = PyMiner(0.5, 1, 3, 1, 1)
        motifs = miner.mine_motifs(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['a', 'aa', 'c', 'ca', 'cc'])

    def test_rag_py_miner(self):
        miner = PyMiner(0.5, 1, 3, 1, 1)
        motifs = miner.mine_motifs(rag)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['a', 'c', 'ac'])

    def test_frm_cpp_miner(self):
        miner = CppMiner(0.5, 1, 3, 1, 1)
        motifs = miner.mine(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, [[0], [2], [2, 0], [0, 0], [2, 2], [2, 0, 0]])

    def test_rag_cpp_miner(self):
        miner = CppMiner(0.5, 1, 3, 1, 1)
        motifs = miner.mine(rag)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, [[0], [0, 2], [2]])