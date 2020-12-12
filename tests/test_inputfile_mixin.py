import unittest

from ddt import ddt, data

import numpy as np

from zetastitcher.io.inputfile_mixin import InputFileMixin

test_vectors = [
    [np.index_exp[1:100:1], np.index_exp[1:100]],
    [np.index_exp[0:42:1], np.index_exp[:42]],
    [np.index_exp[20:80:2], np.index_exp[20:80:2]],
    [np.index_exp[20:80:3], np.index_exp[20:80:3]],
    [np.index_exp[20:80:4], np.index_exp[20:80:4]],
    [np.index_exp[20:80:5], np.index_exp[20:80:5]],
    [np.index_exp[0:90:1], np.index_exp[:-10]],
    [np.index_exp[10:90:1], np.index_exp[10:-10]],
    [np.index_exp[10:90:2], np.index_exp[10:-10:2]],
    [np.index_exp[0:10:1], np.index_exp[:-90]],
    [np.index_exp[10:90:1], np.index_exp[-90:-10]],
    [np.index_exp[10:90:2], np.index_exp[-90:-10:2]],
    [np.index_exp[11:91:1], np.index_exp[-10:-90:-1]],
    [np.index_exp[11:91:2], np.index_exp[-10:-90:-2]],
    [np.index_exp[91:96:2], np.index_exp[-5:-10:-2]],
    [np.index_exp[91:91:2], np.index_exp[-10:-10:-2]],
    [np.index_exp[90:90:2], np.index_exp[-10:-10:2]],
    [np.index_exp[93:98:2], np.index_exp[-3:-8:-2]],
    [np.index_exp[93:98:1], np.index_exp[-3:-8:-1]],
    [np.index_exp[72:91:3], np.index_exp[-10:-30:-3]],
    [np.index_exp[71:91:5], np.index_exp[-10:-30:-5]],
    [np.index_exp[91:100:1], np.index_exp[:-10:-1]],
    [np.index_exp[0:100:1], np.index_exp[::-1]],
    [np.index_exp[0:100:1], np.index_exp[:]],
    [np.index_exp[0:100:1], np.index_exp[...]],
    [np.index_exp[70:100:1], np.index_exp[-30:]],
    [np.index_exp[70:100:1], np.index_exp[-30::]],
    [np.index_exp[70:100:2], np.index_exp[-30::2]],
    [np.index_exp[70:100:3], np.index_exp[-30::3]],
    [np.index_exp[70:100:4], np.index_exp[-30::4]],
    [np.index_exp[70:100:5], np.index_exp[-30::5]],
    [np.index_exp[0:100:5], np.index_exp[::5]],
    [np.index_exp[1:71:3], np.index_exp[-30::-3]],
]


@ddt
class TestInputFileMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.f = InputFileMixin()
        cls.f.nfrms = 100
        cls.a = np.linspace(1, 100, num=100)

    @data(*test_vectors)
    def testNormalizeSlice(self, value):
        a, b = value
        a = a[0]
        b = b[0]
        self.assertEqual(a, self.f._normalize_slice(b))
        np.testing.assert_equal(self.a[a], self.a[self.f._normalize_slice(b)])
