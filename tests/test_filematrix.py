import unittest

import numpy as np

from ddt import ddt, data

from zetastitcher.align.filematrix import parse_file_name


test_vectors = [
    ['151.00000_094.00000_014.30000_cam_0.tiff', [151.0, 94.0, 14.3]],
    ['9x_5326y_24819z_19140r_1024zf_1depth_40zstep_1pw_20800nm_0xl', [5326, 24819, 19140]],
    ['9x_-5326y_24819z_19140r_1024zf_1depth_40zstep_1pw_20800nm_0xl', [-5326, 24819, 19140]],
    ['x_170000_y_160000_z_073000_zstep_2_.dcimg', [170000, 160000, 73000]],
    ['170000_160000.raw', [170000, 160000, 0]],
    ['1536_0.tif', [1536, 0, 0]],
]


@ddt
class TestFileMatrix(unittest.TestCase):
    @data(*test_vectors)
    def test_parse_file_name(self, value):
        fields = parse_file_name(value[0])
        np.testing.assert_equal(fields, value[1])


if __name__ == '__main__':
    unittest.main()
