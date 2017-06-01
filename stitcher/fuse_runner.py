import re
import sys
import os.path
import argparse
import threading

from queue import Queue
from functools import lru_cache

import numpy as np

import skimage.external.tifffile as tiff

from .filematrix import FileMatrix
from .inputfile import InputFile
from .fuse import fuse_queue
from .numbers import numbers


def to_dtype(x, dtype):
    x = np.rint(x) if np.issubdtype(dtype, np.integer) else x
    return x.astype(dtype, copy=False)


class FuseRunner(object):
    def __init__(self, input_file=None):
        self.input_file = input_file  #: input file or folder
        self.fm = None  #: :class:`FileMatrix`
        self.path = None

        self.zmin = 0
        self.zmax = None

        self.output_filename = None

        self._is_multichannel = None

        self._load_df()

    def _load_df(self):
        if os.path.isdir(self.input_file):
            input_file = os.path.join(self.input_file, 'stitch.json')
        else:
            input_file = self.input_file

        self.path, file_name = os.path.split(input_file)
        self.fm = FileMatrix(input_file)

    @property
    @lru_cache()
    def is_multichannel(self):
        with InputFile(self.fm.data_frame.iloc[0].name) as f:
            if f.nchannels > 1:
                multichannel = True
            else:
                multichannel = False
        return multichannel

    @property
    @lru_cache()
    def output_shape(self):
        thickness = self.fm.full_thickness
        if self.zmax is not None:
            thickness -= (thickness - self.zmax)
        thickness -= self.zmin

        with InputFile(self.fm.data_frame.iloc[0].name) as f:
            output_shape = list(f.shape)

        output_shape[0] = thickness
        output_shape[-2] = self.fm.full_height
        output_shape[-1] = self.fm.full_width

        return output_shape

    def run(self):
        df = self.fm.data_frame
        for key in ['Xs', 'Ys', 'Zs']:
            df[key] -= df[key].min()

        fused = np.zeros(self.output_shape, dtype=np.float32)
        q = Queue()

        t = threading.Thread(target=fuse_queue, args=(q, fused))
        t.start()

        for index, row in self.fm.data_frame.iterrows():
            z_from = self.zmin - row.Zs
            if z_from < 0:
                z_from = 0

            if self.zmax is None:
                z_to = row.nfrms
            else:
                z_to = z_from + self.zmax - self.zmin

            if z_to - z_from <= 0:
                continue

            with InputFile(os.path.join(self.path, index)) as f:
                layer = np.copy(f.layer(z_from, z_to))
                dtype = layer.dtype
                layer = layer.astype(np.float32, copy=False)

            cx = layer.shape[-1] // 2
            cy = layer.shape[-2] // 2 + 10
            x = cx - 100
            xstr = re.search(r'\d+', index).group()
            for l in xstr:
                x_end = x + 30
                layer[..., cy:cy + 50, x:x_end] = numbers[int(l)]
                x = x_end + 5

            for f in range(0, layer.shape[0]):
                x = cx - 120
                xstr = str(z_from + f)
                for l in xstr:
                    x_end = x + 30
                    layer[f, ..., cy + 55:cy + 105, x:x_end] = \
                        numbers[int(l)]
                    x = x_end + 5

            top_left = [row.Zs + z_from, row.Ys, row.Xs]
            overlaps = self.fm.overlaps(index)

            q.put([layer, top_left, overlaps])

        q.put([None, None, None])  # close queue

        t.join()  # wait for fuse thread to finish

        if self.is_multichannel:
            fused = np.moveaxis(fused, -3, -1)

        tiff.imsave(self.output_filename, to_dtype(fused, dtype))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fuse stitched tiles in a folder.',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('input_file', help='input file (.json) or folder')

    parser.add_argument('-o', type=str, default='fused.tif',
                        dest='output_filename', help='output file name')

    parser.add_argument('--zmin', type=int, default=0)
    parser.add_argument('--zmax', type=int, default=None, help='noninclusive')

    return parser.parse_args(sys.argv[1:])


def main():
    arg = parse_args()
    fr = FuseRunner(arg.input_file)

    keys = ['zmin', 'zmax', 'output_filename']
    for k in keys:
        setattr(fr, k, getattr(arg, k))

    fr.run()


if __name__ == '__main__':
    main()
