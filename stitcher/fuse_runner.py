import sys
import os.path
import argparse

from queue import Queue

import numpy as np
import pandas as pd
import networkx as nx

import skimage.external.tifffile as tiff

from .filematrix import FileMatrix
from .inputfile import InputFile
from .fuse import fuse_queue


class FuseRunner(object):
    def __init__(self, input_file=None):
        self.input_file = input_file  #: input file or folder
        self.fm = None  #: :class:`FileMatrix` with filename as Index
        self.df = None  #: :class:`pandas.DataFrame` of optimal shifts

        self._load_df()
        self._compute_absolute_positions()

    def _load_df(self):
        def normalize_path(x):
            return os.path.normpath(os.path.join(path, x))

        if os.path.isdir(self.input_file):
            input_file = os.path.join(self.input_file, 'stitch.json')
        else:
            input_file = self.input_file

        path, file_name = os.path.split(input_file)

        with open(input_file, 'r') as f:
            df = pd.read_json(f.read(), orient='records')
        df['aname'] = df['aname'].apply(normalize_path)
        df['bname'] = df['bname'].apply(normalize_path)

        self.df = df
        fm = FileMatrix(input_file)
        fm.data_frame = fm.data_frame.set_index('filename')
        self.fm = fm

    def _compute_absolute_positions(self):
        fm_df = self.fm.data_frame
        df = self.df
        T = self.minimum_spanning_tree

        df_ab = df.set_index(['aname', 'bname'])

        fm_df['Xs'] = 0
        fm_df['Ys'] = 0
        fm_df['Zs'] = 0
        fm_df['weight'] = 0
        for edge in nx.dfs_edges(T, source=fm_df.iloc[0].name):
            try:
                row = df_ab.loc[edge]
                reversed_edge = False
            except KeyError:
                row = df_ab.loc[edge[::-1]]
                reversed_edge = True

            axis = row['axis']
            if axis == 2:
                stride_y = fm_df.ix[edge[0], 'xsize']
                key_Xs = 'Ys'
                key_Ys = 'Xs'
            else:
                stride_y = fm_df.ix[edge[0], 'ysize']
                key_Xs = 'Xs'
                key_Ys = 'Ys'

            sign = (-1 if reversed_edge else 1)

            # absolute stitched positions
            Xs = (fm_df.ix[edge[0], key_Xs] + sign * row['dx'])
            Ys = (fm_df.ix[edge[0], key_Ys] + sign * (stride_y - row['dy']))
            Zs = (fm_df.ix[edge[0], 'Zs'] + row['dz'])

            fm_df.ix[edge[1], key_Xs] = Xs
            fm_df.ix[edge[1], key_Ys] = Ys
            fm_df.ix[edge[1], 'Zs'] = Zs
            fm_df.ix[edge[1], 'weight'] = 1 - row['score']

        for key in ['Xs', 'Ys', 'Zs']:
            fm_df[key] -= fm_df[key].min()

    def _add_edges_from_df(self, G):
        df = self.df
        G.add_edges_from(((
            u, v, {'weight': w}) for u, v, w in
            np.c_[df['aname'], df['bname'], 1 - df['score']]))

    @property
    def minimum_spanning_tree(self):
        G = nx.Graph()
        self._add_edges_from_df(G)
        T = nx.minimum_spanning_tree(G)
        return T

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
                with InputFile(tile.Index) as f:
                    layer = np.copy(f.whole())

                x_from_i = rint(stripe_left_edge - tile.Xs)
                x_to_i = x_from_i + stripe_width

                top_left = [tile.Zs, tile.Ys, 0]
                q.put([layer[..., x_from_i:x_to_i], top_left])

            q.put([None, None])  # close queue

            output_stripe = fuse_queue(q, stripe_thickness=thickness)

            stripe_pos = [0, M['Xs'] - stripe_width, m['Ys']]
            stripe_q.put([np.rot90(output_stripe, axes=(-2, -1)), stripe_pos])

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
