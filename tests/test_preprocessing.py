import unittest

from frm._frm_py.preprocessing import sax, standardise
from test_data import ts, norm, rag, seq, rseq


class TestPreprocessing(unittest.TestCase):
    def test_standardise(self):
        for expected, got in zip(norm, standardise(ts)):
            for a, b in zip(expected, got):
                self.assertAlmostEqual(a, b, 5)
    
    def test_sax(self):
        got = sax(standardise(ts), 1, 3)

        self.assertEqual(seq, got)

    def test_rag_sax(self):
        got = sax(standardise(rag), 1, 3)

        self.assertEqual(rseq, got)
