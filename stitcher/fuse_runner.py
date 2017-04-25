import sys
import os.path
import argparse

from queue import Queue

import numpy as np

import skimage.external.tifffile as tiff

from .filematrix import FileMatrix
from .inputfile import InputFile
from .fuse import fuse_queue


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

        stripe_q = Queue()

        df = self.fm.data_frame
        for key in ['Xs', 'Ys', 'Zs']:
            df[key] -= df[key].min()

        height = self.fm.full_height
        thickness = self.fm.full_thickness

        for group in self.fm.tiles_along_Y:
            group = group.copy()

            m = group.min()
            M = group.max()

            stripe_left_edge = M['Xs']
            stripe_width = rint(m['Xs'] + group.iloc[-1]['xsize'] - M['Xs'])

            tile_generator = group.itertuples()

            q = Queue()
            for tile in tile_generator:
                with InputFile(os.path.join(self.path, tile.Index)) as f:
                    layer = np.copy(f.whole())

                x_from_i = rint(stripe_left_edge - tile.Xs)
                x_to_i = x_from_i + stripe_width

                top_left = [tile.Zs, tile.Ys, 0]
                q.put([layer[..., x_from_i:x_to_i], top_left])

            q.put([None, None])  # close queue

            output_stripe = fuse_queue(q, stripe_thickness=thickness)
            output_stripe = np.rot90(output_stripe, axes=(-2, -1))

            stripe_pos = [0, M['Xs'] - stripe_width, m['Ys']]
            stripe_q.put([output_stripe, stripe_pos])

        stripe_q.put([None, None])
        fused_xy = fuse_queue(stripe_q, stripe_width=height)
        fused_xy = np.rot90(fused_xy, k=3, axes=(-2, -1))

        with InputFile(tile.Index) as f:
            if f.nchannels > 1:
                multi_channel = True
            else:
                multi_channel = False

        if multi_channel:
            fused_xy = np.moveaxis(fused_xy, -3, -1)

        tiff.imsave('fused_xy.tiff', fused_xy)


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
