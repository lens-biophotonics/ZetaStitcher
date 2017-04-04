import sys
import os.path
import argparse

import numpy as np
import pandas as pd
import networkx as nx

from .filematrix import FileMatrix


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
        self.fm = fm

    def _compute_absolute_positions(self):
        fm_df = self.fm.data_frame
        df = self.df
        T = self.minimum_spanning_tree

        fm_df['Xs'] = 0
        fm_df['Ys'] = 0
        for edge in nx.dfs_edges(T, source=fm_df.iloc[0].name):
            edge_data = T.get_edge_data(*edge)
            if df.loc[edge_data['label']]['axis'] == 2:
                stride_y = fm_df.ix[edge[1], 'xsize']
                key_Y = 'X'

                key_Xs = 'Ys'
                key_Ys = 'Xs'
            else:
                stride_y = fm_df.ix[edge[1], 'ysize']
                key_Y = 'Y'

                key_Xs = 'Xs'
                key_Ys = 'Ys'

            sign_y = (1 if fm_df.ix[edge[1], key_Y] > fm_df.ix[edge[0], key_Y]
                      else -1)

            # absolute stitched positions
            Xs = (fm_df.ix[edge[0], key_Xs] + df.loc[edge_data['label']]['dx'])
            Ys = (fm_df.ix[edge[0], key_Ys]
                  + sign_y * (stride_y - df.loc[edge_data['label']]['dy']))

            fm_df.ix[edge[1], key_Xs] = Xs
            fm_df.ix[edge[1], key_Ys] = Ys

        fm_df['Xs'] -= fm_df['Xs'].min()
        fm_df['Xs'] = np.rint(fm_df['Xs']).astype(np.int64)

        fm_df['Ys'] -= fm_df['Ys'].min()
        fm_df['Ys'] = np.rint(fm_df['Ys']).astype(np.int64)

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


if __name__ == '__main__':
    main()
