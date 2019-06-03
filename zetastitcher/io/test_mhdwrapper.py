import unittest

from ddt import ddt, data

from .mhdwrapper import MHDWrapper

test_vectors_element_type = [
    ['MET_CHAR', 'i1'],
    ['MET_SHORT', 'i2'],
    ['MET_INT', 'i4'],
    ['MET_LONG', 'i8'],
    ['MET_UCHAR', 'u1'],
    ['MET_USHORT', 'u2'],
    ['MET_UINT', 'u4'],
    ['MET_ULONG', 'u8'],
    ['MET_FLOAT', 'f4'],
    ['MET_DOUBLE', 'f8'],
]


@ddt
class TestMHDWrapper(unittest.TestCase):
    @data(*test_vectors_element_type)
    def test_element_type(self, value):
        a, b = value
        self.assertEqual(MHDWrapper.element_type_to_type_fmt(a), b)
