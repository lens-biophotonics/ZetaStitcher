import sys
import os.path
import argparse

import numpy as np
import pandas as pd
import networkx as nx

import skimage.external.tifffile as tiff

from .filematrix import FileMatrix
from .inputfile import InputFile


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

        for key in ['Xs', 'Ys', 'Zs']:
            fm_df[key] -= fm_df[key].min()

    def _fuse(self, a_roi, b_roi, dest):
        """Fuse two overlapping regions.

        Fuses `a_roi` and `b_roi` applying a sinusoidal smoothing. All
        parameters must have equal shapes.

        Parameters
        ----------
        a_roi : :class:`numpy.ndarray`
        b_roi : :class:`numpy.ndarray`
        dest : :class:`numpy.ndarray`
        """
        if a_roi.shape != b_roi.shape or a_roi.shape != dest.shape:
            raise ValueError(
                'ROI shapes must be equal. a: {}, b: {}, dest: {}'.format(
                    a_roi.shape, b_roi.shape, dest.shape))

        output_height = a_roi.shape[1]
        output_width = a_roi.shape[2]

        rad = np.linspace(0.0, np.pi, output_height, dtype=np.float32)

        alpha = (np.cos(rad) + 1) / 2
        alpha = np.tile(alpha, [output_width])
        alpha = np.reshape(alpha, [output_width, output_height])
        alpha = np.transpose(alpha)

        dest[:] = a_roi * alpha + b_roi * (1 - alpha)

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
        for group in self.fm.tiles_along_Y:
            if group.iloc[0]['X'] != 160000:
                continue

            m = group.min()
            M = group.max()

            stripe_width = int(
                np.rint(M['Xs'] - m['Xs'] + group.iloc[-1]['xsize']))
            stripe_height = int(
                np.rint(M['Ys'] - m['Ys'] + group.iloc[-1]['ysize']))

            print(stripe_height, stripe_width)

            output_array = np.zeros((1, stripe_height, stripe_width),
                                    dtype=np.float32)

            tile_generator = group.itertuples()
            atile = next(tile_generator)
            a = InputFile(atile.Index)


            z_frame = 1500

            dy_prev_i = 0
            current_y_i = 0
            while True:
                try:
                    btile = next(tile_generator)

                    oy_to = atile.Ys + atile.ysize
                    dx = btile.Xs - atile.Xs
                    dy = oy_to - btile.Ys
                    dz = btile.Zs - atile.Zs
                except StopIteration:
                    dy = 0
                    break
                finally:
                    alayer = a.layer(int(np.rint(z_frame)), dtype=np.float32)

                    # add first part
                    ox_from = atile.Xs - m['Xs']
                    ox_to = ox_from + atile.xsize
                    oy_to = atile.Ys - m['Ys'] + atile.ysize - dy

                    ox_from_i = int(np.rint(ox_from))
                    ox_width_i = int(np.rint(ox_to - ox_from))
                    ox_to_i = ox_from_i + ox_width_i

                    oy_height_i = int(np.rint(oy_to - current_y_i))
                    oy_to_i = current_y_i + oy_height_i

                    ay_from_i = dy_prev_i
                    ay_to_i = ay_from_i + oy_height_i

                    output_array[0, current_y_i:oy_to_i, ox_from_i:ox_to_i] = \
                        alayer[0, ay_from_i:ay_to_i, :]

                    current_y_i = oy_to_i
                print(atile)
                print(btile)
                print('fusing {} and {}'.format(atile.Index, btile.Index))

                b = InputFile(btile.Index)
                # the updated zframe will be reused in the next loop with
                # alayer
                z_frame = z_frame - dz
                blayer = b.layer(int(np.rint(z_frame)), dtype=np.float32)

                # for the moment consider a and b to have same width
                fused_width = alayer.shape[2] - abs(dx)

                ax_min = 0
                if dx > 0:
                    ax_min = dx

                bx_min = 0
                if dx < 0:
                    bx_min = -dx

                fused_width_i = int(np.rint(fused_width))
                fused_height_i = int(np.rint(dy))

                ax_min_i = int(np.rint(ax_min))
                ax_max_i = ax_min_i + fused_width_i

                bx_min_i = int(np.rint(bx_min))
                bx_max_i = bx_min_i + fused_width_i

                a_roi = alayer[:, -fused_height_i:, ax_min_i:ax_max_i]
                b_roi = blayer[:, 0:fused_height_i, bx_min_i:bx_max_i]

                # add fused part
                ox_from_i = int(np.rint(btile.Xs - m['Xs']))
                ox_to_i = ox_from_i + fused_width_i
                oy_to_i = current_y_i + fused_height_i

                self._fuse(a_roi, b_roi,
                           output_array[:, current_y_i:oy_to_i,
                                        ox_from_i:ox_to_i])
                current_y_i = oy_to_i

                a.close()
                a = b
                atile = btile
                dy_prev_i = fused_height_i
                print('===============')

            b.close()
            tiff.imsave('/mnt/data/temp/stitch/output.tiff', output_array)


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
