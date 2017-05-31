import re
import sys
import os.path
import argparse

from queue import Queue

import numpy as np

import skimage.external.tifffile as tiff

from .filematrix import FileMatrix
from .inputfile import InputFile
from .fuse import fuse_queue
from .numbers import numbers


class FuseRunner(object):
    def __init__(self, input_file=None):
        self.input_file = input_file  #: input file or folder
        self.fm = None  #: :class:`FileMatrix`
        self.path = None

        self._load_df()

    def _load_df(self):
        if os.path.isdir(self.input_file):
            input_file = os.path.join(self.input_file, 'stitch.json')
        else:
            input_file = self.input_file

        self.path, file_name = os.path.split(input_file)
        self.fm = FileMatrix(input_file)

    def run(self):
        def rint(x):
            return int(round(x))

        df = self.fm.data_frame
        for key in ['Xs', 'Ys', 'Zs']:
            df[key] -= df[key].min()

        q = Queue()
        for index, row in self.fm.data_frame.iterrows():
            with InputFile(os.path.join(self.path, index)) as f:
                layer = np.copy(f.whole())

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
                xstr = str(f)
                for l in xstr:
                    x_end = x + 30
                    layer[f, ..., cy + 55:cy + 105, x:x_end] = \
                        numbers[int(l)]
                    x = x_end + 5

            top_left = [row.Zs, row.Ys, row.Xs]
            overlaps = self.fm.overlaps(index)

            q.put([layer, top_left, overlaps])

        q.put([None, None, None])  # close queue

        output_shape = list(layer.shape)
        output_shape[0] = self.fm.full_thickness
        output_shape[-2] = self.fm.full_height
        output_shape[-1] = self.fm.full_width

        fused_xy = fuse_queue(q, output_shape)

        with InputFile(index) as f:
            if f.nchannels > 1:
                multi_channel = True
            else:
                multi_channel = False

        if multi_channel:
            fused_xy = np.moveaxis(fused_xy, -3, -1)

        tiff.imsave('fused_xy.tiff', fused_xy.astype(np.int8))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fuse stitched tiles in a folder.',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('input_file', help='input file (.json) or folder')

    return parser.parse_args(sys.argv[1:])


def main():
    arg = parse_args()
    fr = FuseRunner(arg.input_file)
    fr.run()


if __name__ == '__main__':
    main()
