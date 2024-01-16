import unittest

from frm import Miner as CppMiner
from frm._frm_py.miner import Miner as PyMiner

from .data import ts, rag, data


class TestMiner(unittest.TestCase):
    def test_frm_py_miner(self):
        miner = PyMiner(0.5, 1, 3, 1, 0, 1)
        motifs = miner.mine(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['aa', 'a', 'ca', 'cc', 'c'])

    def test_rag_py_miner(self):
        miner = PyMiner(0.5, 1, 3, 1, 0, 1)
        motifs = miner.mine(rag)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['ab', 'abc', 'b', 'bc', 'a', 'c'])

    def test_frm_cpp_miner(self):
        miner = CppMiner(0.5, 1, 3, 1, 0, 1)
        motifs = miner.mine(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['aa', 'a', 'ca', 'cc', 'c'])

    def test_rag_cpp_miner(self):
        miner = CppMiner(0.5, 1, 3, 1, 0, 1)
        motifs = miner.mine(rag)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['ab', 'abc', 'b', 'bc', 'a', 'c'])

    def test_equal_miners(self):
        py_miner = PyMiner(0.5, 1, 3, 1, 0, 1)
        py_motifs = py_miner.mine(ts)

        cpp_miner = CppMiner(0.5, 1, 3, 1, 0, 1)
        cpp_motifs = cpp_miner.mine(ts)

        for py_motif, cpp_motif in zip(py_motifs, cpp_motifs):
            self.assertAlmostEqual(py_motif.naed, cpp_motif.naed)

    def test_equal_rag_miners(self):
        py_miner = PyMiner(0.5, 1, 3, 1, 0, 1)
        py_motifs = py_miner.mine(rag)

        cpp_miner = CppMiner(0.5, 1, 3, 1, 0, 1)
        cpp_motifs = cpp_miner.mine(rag)

        for py_motif, cpp_motif in zip(py_motifs, cpp_motifs):
            self.assertAlmostEqual(py_motif.naed, cpp_motif.naed)

    def test_max_len_py_miner(self):
        miner = PyMiner(0.5, 1, 3, 1, 1, 1)
        motifs = miner.mine(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['a', 'c'])

    def test_max_len_cpp_miner(self):
        miner = CppMiner(0.5, 1, 3, 1, 1, 1)
        motifs = miner.mine(ts)
        patterns = [m.pattern for m in motifs]
        self.assertListEqual(patterns, ['a', 'c'])

    def test_equal_long(self):
        seglen, alphabet = 2, 4
        py = PyMiner(0.5, seglen, alphabet)
        cpp = CppMiner(0.5, seglen, alphabet)

        py.mine(data)
        cpp.mine(data)

        for p, c in zip(py.motifs, cpp.motifs):
            self.assertEqual(p.pattern, c.pattern)
            self.assertAlmostEqual(p.naed, c.naed)
