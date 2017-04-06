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
        fm = FileMatrix(path)
        fm.data_frame = fm.data_frame.set_index('filename')
        fm.ascending_tiles_X = True
        fm.ascending_tiles_Y = False
        self.fm = fm

    def _compute_absolute_positions(self):
        fm_df = self.fm.data_frame
        df = self.df
        T = self.minimum_spanning_tree

        fm_df['Xs'] = 0
        fm_df['Ys'] = 0
        fm_df['Zs'] = 0
        fm_df['weight'] = 0
        for edge in nx.dfs_edges(T, source=fm_df.iloc[0].name):
            edge_data = T.get_edge_data(*edge)
            if df.loc[edge_data['label']]['axis'] == 2:
                stride_y = fm_df.ix[edge[1], 'xsize']
                key_Y = 'X'

                key_Xs = 'Ys'
                key_Ys = 'Xs'

                ascending_sign = 1 if self.fm.ascending_tiles_X else -1
            else:
                stride_y = fm_df.ix[edge[1], 'ysize']
                key_Y = 'Y'

                key_Xs = 'Xs'
                key_Ys = 'Ys'

                ascending_sign = 1 if self.fm.ascending_tiles_Y else -1

            sign_y = (1 if fm_df.ix[edge[1], key_Y] >= fm_df.ix[edge[0], key_Y]
                      else -1) * ascending_sign
            sign_z = (1 if fm_df.ix[edge[1], 'Z'] >= fm_df.ix[edge[0], 'Z']
                      else -1)

            # absolute stitched positions
            Xs = (fm_df.ix[edge[0], key_Xs] + df.loc[edge_data['label']]['dx'])
            Ys = (fm_df.ix[edge[0], key_Ys]
                  + sign_y * (stride_y - df.loc[edge_data['label']]['dy']))
            Zs = (fm_df.ix[edge[0], 'Zs']
                  + sign_z * df.loc[edge_data['label']]['dz'])

            fm_df.ix[edge[1], key_Xs] = Xs
            fm_df.ix[edge[1], key_Ys] = Ys
            fm_df.ix[edge[1], 'Zs'] = Zs
            fm_df.ix[edge[1], 'weight'] = edge_data['weight']

        for key in ['Xs', 'Ys', 'Zs']:
            fm_df[key] -= fm_df[key].min()

    @property
    def minimum_spanning_tree(self):
        df = self.df
        G = nx.Graph()
        for index, row in df.iterrows():
            G.add_node(row['aname'])
            G.add_node(row['bname'])

        G.add_edges_from(((
            u, v, {'weight': d, 'label': i}) for u, v, d, i in
            np.c_[df['aname'], df['bname'], 1 - df['score'], df.index]))

        T = nx.minimum_spanning_tree(G)
        return T

    def run(self):
        stripe_q = Queue()

        for group in self.fm.tiles_along_Y:
            if group.iloc[0]['X'] < 180000 or group.iloc[0]['X'] > 210000:
                continue

            # skip unstitchable tiles
            group = group[group.weight < 0.6]

            m = group.min()
            M = group.max()

            stripe_left_edge = int(np.rint(M['Xs'] - m['Xs']))

            stripe_width = int(
                np.rint(group.iloc[-1]['xsize'] - stripe_left_edge))
            stripe_height = int(
                np.rint(M['Ys'] - m['Ys'] + group.iloc[-1]['ysize'])) + 1

            print(stripe_height, stripe_width)

            tile_generator = group.itertuples()

            z_frame = 1500
            prev_Zs = 0
            q = Queue()
            for tile in tile_generator:
                with InputFile(tile.Index) as f:
                    layer = np.copy(f.layer(int(np.rint(z_frame))))

                dz = int(np.rint(tile.Zs - prev_Zs))

                ax_from_i = int(
                    np.rint(stripe_left_edge - (tile.Xs - m['Xs'])))
                ax_to_i = ax_from_i + stripe_width

                q.put([layer[..., ax_from_i:ax_to_i], np.asscalar(tile.Ys)])

                z_frame = z_frame - dz
                prev_Zs = tile.Zs

            q.put([None, None])
            fuse_queue(q, stripe_shape=(stripe_height, stripe_width),
                       dest_queue=stripe_q)
        stripe_q.put(None)
        i = 0
        for s in iter(stripe_q.get, None):
            tiff.imsave('/mnt/data/temp/stitch/output{}.tiff'.format(i), s)
            i += 1


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
