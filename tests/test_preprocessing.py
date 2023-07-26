import unittest

from frm._frm_py.preprocessing import sax, standardise

from test_data import ts, norm, rag, seq_1, seq_2, rseq_1, rseq_2


class TestPreprocessing(unittest.TestCase):
    def test_standardise(self):
        for expected, got in zip(norm, standardise(ts)):
            for a, b in zip(expected, got):
                self.assertAlmostEqual(a, b, 5)
    
    def test_sax_seglen_1(self):
        got = sax(standardise(ts), 1, 3)
        self.assertEqual(seq_1, got)

    def test_rag_sax_seglen_1(self):
        got = sax(standardise(rag), 1, 3)
        self.assertEqual(rseq_1, got)

    def test_sax_seglen_2(self):
        got = sax(standardise(ts), 2, 3)
        self.assertEqual(seq_2, got)

    def test_rag_sax_seglen_2(self):
        got = sax(standardise(rag), 2, 3)
        self.assertEqual(rseq_2, got)
