import unittest

from frm import Miner as CppMiner
from frm._frm_py.miner import Miner as PyMiner
from test_data import ts, rag


class TestMiner(unittest.TestCase):
    def test_frm_py_miner(self):
        miner = PyMiner(0.5, 1, 3, 1, 0, 1)
        motifs = miner.mine_motifs(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['aa', 'a', 'ca', 'cc', 'c'])

    def test_rag_py_miner(self):
        miner = PyMiner(0.5, 1, 3, 1, 0, 1)
        motifs = miner.mine_motifs(rag)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['a', 'c', 'ac'])

    def test_frm_cpp_miner(self):
        miner = CppMiner(0.5, 1, 3, 1, 0, 1)
        motifs = miner.mine(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, [[0, 0], [0], [2, 0], [2, 2], [2]])

    def test_rag_cpp_miner(self):
        miner = CppMiner(0.5, 1, 3, 1, 0, 1)
        motifs = miner.mine(rag)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, [[0], [2], [0, 2]])

    def test_equal_miners(self):
        py_miner = PyMiner(0.5, 1, 3, 1, 0, 1)
        py_motifs = py_miner.mine_motifs(ts)

        cpp_miner = CppMiner(0.5, 1, 3, 1, 0, 1)
        cpp_motifs = cpp_miner.mine(ts)

        for py_motif, cpp_motif in zip(py_motifs, cpp_motifs):
            self.assertAlmostEqual(py_motif.naed, cpp_motif.naed)

    def test_equal_rag_miners(self):
        py_miner = PyMiner(0.5, 1, 3, 1, 0, 1)
        py_motifs = py_miner.mine_motifs(rag)

        cpp_miner = CppMiner(0.5, 1, 3, 1, 0, 1)
        cpp_motifs = cpp_miner.mine(rag)

        for py_motif, cpp_motif in zip(py_motifs, cpp_motifs):
            self.assertAlmostEqual(py_motif.naed, cpp_motif.naed)

    def test_max_len_py_miner(self):
        miner = PyMiner(0.5, 1, 3, 1, 1, 1)
        motifs = miner.mine_motifs(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['a', 'c'])

    def test_max_len_cpp_miner(self):
        miner = CppMiner(0.5, 1, 3, 1, 1, 1)
        motifs = miner.mine(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, [[0], [2]])
