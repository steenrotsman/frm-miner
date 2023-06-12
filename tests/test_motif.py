import unittest

from frm._frm_py.motif import Motif


class TestMotif(unittest.TestCase):
    def test_init(self):
        a = Motif(pattern='abc')
        self.assertEqual(a.pattern, 'abc')
        self.assertEqual(a.indexes, {})
        self.assertEqual(len(a.matches), 0)
        self.assertIsNone(a.representative)
    
    def test_record_index(self):
        a = Motif(pattern='abc')
        a.record_index(0, 1)
        self.assertEqual(list(a.indexes.keys()), [0])
        self.assertEqual(a.indexes[0], [1])
