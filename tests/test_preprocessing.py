import unittest

from frm._frm_py.preprocessing import sax, standardise
from test_data import ts, rag, seq, rseq

# Round precision for float comparisons
ROUND = 7


class TestPreprocessing(unittest.TestCase):
    def test_standardise(self):
        std = 0.7371114795831993

        a = round(-0.7/std, ROUND)
        b = round(0.3/std, ROUND)
        c = round(1.3/std, ROUND)

        expected = [
            [a, b, c, c, b, a],
            [a, a, b, b, a, a],
            [c, b, b, a, a, a],
            [c, b, a, a, b, c],
            [a, b, a, b, a, b]
        ]

        got = standardise(ts, False)
        got = [[round(x, ROUND) for x in row] for row in got]

        self.assertEqual(got, expected)
    
    def test_sax(self):
        got = sax(standardise(ts, False), 1, 3)

        self.assertEqual(got, seq)

    def test_rag_sax(self):
        got = sax(standardise(rag, False), 1, 3)

        self.assertEqual(got, rseq)
