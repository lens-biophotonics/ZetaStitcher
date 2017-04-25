"""Parse input file names."""

import os
import re
import math
import logging

import numpy as np
import pandas as pd
import networkx as nx

from .inputfile import InputFile


logger = logging.getLogger('FileMatrix')


def parse_file_name(file_name):
    """Parse fields (stage coordinates) contained in `file_name`.

    Parameters
    ----------
    file_name : str
                The string to be parsed

    Returns
    -------
    x, y, z : int
        The parsed stage coordinates.
    """
    file_name = os.path.basename(file_name)
    m = re.search('^.*x_(\d+).*y_(\d+).*z_(\d+).*', file_name)
    if m is None:
        m = re.search('^(\d+)_(\d+)_(\d+)', file_name)
    if m is None:
        raise ValueError('Invalid name {}'.format(file_name))

    fields = []
    for i in range(1, 4):
        fields.append(int(m.group(i)))

    return fields


class FileMatrix:
    """Data structures for a matrix of input files."""
    def __init__(self, directory=None):
        self.dir = directory

        self.data_frame = None
        """A :class:`pandas.DataFrame` object. Contains the following
        columns: `X`, `Y`, `Z`, `Z_end`, `xsize`, `ysize`, 'nfrms`,
        `filename`."""
        self.stitch_data_frame = None

        self._ascending_tiles_x = True
        self._ascending_tiles_y = True

        if os.path.isdir(directory):
            self.load_dir(directory)
        elif os.path.isfile(directory):
            self.load_json(directory)

    def load_dir(self, dir=None):
        """Look for files in `dir` recursively and populate data structures.

        Parameters
        ----------
        dir : path
        """
        if dir is None:
            dir = self.dir

        if dir is None:
            return

        flist = []

        for root, dirs, files in os.walk(dir):
            if os.path.basename(root):
                try:
                    self.parse_and_append(root, flist)
                    continue
                except (RuntimeError, ValueError) as e:
                    logger.error(e.args[0])

            for f in files:
                try:
                    self.parse_and_append(os.path.join(root, f), flist)
                    continue
                except (RuntimeError, ValueError) as e:
                    logger.error(e.args[0])

        self._load_from_flist(flist)

    def load_json(self, fname):
        with open(fname, 'r') as f:
            df = pd.read_json(f.read(), orient='records')

        a = np.concatenate((df['aname'].unique(), df['bname'].unique()))
        a = np.unique(a)

        flist = []
        self.dir, _ = os.path.split(fname)
        for el in a:
            self.parse_and_append(el, flist)

        temp = df[df['axis'] == 2]
        a_fields = parse_file_name(temp.iloc[0].aname)
        b_fields = parse_file_name(temp.iloc[0].bname)
        if a_fields[0] > b_fields[0]:
            self.ascending_tiles_x = False
        else:
            self.ascending_tiles_x = True

        temp = df[df['axis'] == 1]
        a_fields = parse_file_name(temp.iloc[0].aname)
        b_fields = parse_file_name(temp.iloc[0].bname)
        if a_fields[1] > b_fields[1]:
            self.ascending_tiles_y = False
        else:
            self.ascending_tiles_y = True

        self._load_from_flist(flist)
        self.stitch_data_frame = df
        self._compute_absolute_positions()
        self._compute_overlaps()

    def _load_from_flist(self, flist):
        data = {'X': flist[0::7], 'Y': flist[1::7], 'Z': flist[2::7],
                'nfrms': flist[3::7], 'ysize': flist[4::7],
                'xsize': flist[5::7], 'filename': flist[6::7]}
        df = pd.DataFrame(data)
        df = df.sort_values(['Z', 'Y', 'X'])
        df['Z_end'] = df['Z'] + df['nfrms']

        self.data_frame = df.set_index('filename')

    def parse_and_append(self, name, flist):
        try:
            fields = parse_file_name(name)
            with InputFile(os.path.join(self.dir, name)) as infile:
                fields.append(infile.nfrms)
                fields.append(infile.ysize)
                fields.append(infile.xsize)
            flist += fields
            flist.append(name)
        except (RuntimeError, ValueError) as e:
            logger.error(e.args[0])

    def _compute_absolute_positions(self):
        fm_df = self.data_frame
        df = self.stitch_data_frame
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

        fm_df[['Xs', 'Ys', 'Zs']] = fm_df[['Xs', 'Ys', 'Zs']].astype(int)

        fm_df['Xs_end'] = fm_df['Xs'] + fm_df['xsize']
        fm_df['Ys_end'] = fm_df['Ys'] + fm_df['ysize']
        fm_df['Zs_end'] = fm_df['Zs'] + fm_df['nfrms']

    def _compute_overlaps(self):
        fm_df = self.data_frame

        # tiles along Y
        ty = fm_df.sort_values(
            ['Z', 'Y', 'X'], ascending=self.ascending_tiles_y).groupby('X')
        fm_df['overlap_top'] = (
            ty['Ys_end'].shift() - ty['Ys'].shift(0)).fillna(0).astype(int)
        fm_df['overlap_bottom'] = (
            ty['Ys_end'].shift(0) - ty['Ys'].shift(-1)).fillna(0).astype(int)

        # tiles along X
        tx = fm_df.sort_values(
            ['Z', 'X', 'Y'], ascending=self.ascending_tiles_x).groupby('Y')
        fm_df['overlap_left'] = (
            tx['Xs_end'].shift() - tx['Xs'].shift(0)).fillna(0).astype(int)
        fm_df['overlap_right'] = (
            tx['Xs_end'].shift(0) - tx['Xs'].shift(-1)).fillna(0).astype(int)

        cols = ['overlap_top', 'overlap_bottom', 'overlap_left',
                'overlap_right']

        fm_df[cols] = fm_df[cols].applymap(lambda x: x if x > 0 else 0)

    @property
    def minimum_spanning_tree(self):
        G = nx.Graph()
        df = self.stitch_data_frame
        G.add_edges_from(((
            u, v, {'weight': w}) for u, v, w in
            np.c_[df['aname'], df['bname'], 1 - df['score']]))
        T = nx.minimum_spanning_tree(G)
        return T

    @property
    def slices(self):
        """A slice is a group of tiles that share at least a `z` frame.

        Returns
        -------
        comp : generator
            A generator of graphs, one for each connected component of G,
            where G is the graph of tiles connected by at least a `z` frame.
        """
        G = nx.Graph()
        for index, row in self.data_frame.iterrows():
            G.add_node(index)

        for index, row in self.data_frame.iterrows():
            view = self.data_frame[
                (self.data_frame['Z'] <= row['Z'])
                & (self.data_frame['Z_end'] >= row['Z_end'])
            ]
            pairs = zip(view.index.values[::1], view.index.values[1::1])
            G.add_edges_from(pairs)
            G.add_edge((view.index.values[0]), view.index.values[-1])

        return nx.connected_component_subgraphs(G)

    @property
    def ascending_tiles_x(self):
        return self._ascending_tiles_x

    @ascending_tiles_x.setter
    def ascending_tiles_x(self, value):
        self._ascending_tiles_x = value

    @property
    def ascending_tiles_y(self):
        return self._ascending_tiles_y

    @ascending_tiles_y.setter
    def ascending_tiles_y(self, value):
        self._ascending_tiles_y = value

    @property
    def tiles_along_dir(self):
        """Groups of tiles to be stitched along a given direction.

        You need to send to this generator a tuple containing:
            - a list for sorting the :class:`pandas.DataFrame`, such as \
            :code:`['Z', 'Y', 'X']`

            - an axis for grouping, such as :code:`'Y'`

        Yields
        -------
        :class:`pandas.DataFrame`
            A group of tiles
        """
        for s in self.slices:
            got = yield
            view = self.data_frame.loc[s.nodes()].sort_values(
                got[0], ascending=got[1]).groupby(got[2])
            for name, group in view:
                yield group

    @property
    def tiles_along_X(self):
        """Groups of tiles to be stitched along `X`.

        Equivalent to :attr:`~tiles_along_dir` having sent the following
        tuple: :code:`(['Z', 'X', 'Y'], 'Y')`

        Yields
        -------
        :class:`pandas.DataFrame`
            A group of tiles
        """
        g = self.tiles_along_dir
        next(g)
        yield g.send((['Z', 'X', 'Y'], self.ascending_tiles_x, 'Y'))
        yield from g

    @property
    def tiles_along_Y(self):
        """Groups of tiles to be stitched along `Y`.

        Equivalent to :attr:`~tiles_along_dir` having sent the following
        tuple: :code:`(['Z', 'Y', 'X'], 'X')`

        Yields
        -------
        :class:`pandas.DataFrame`
            A group of tiles
        """
        g = self.tiles_along_dir
        next(g)
        yield g.send((['Z', 'Y', 'X'], self.ascending_tiles_y, 'X'))
        yield from g

    @property
    def full_height(self):
        def rint(x):
            return 0 if math.isnan(x) else int(round(x))

        def gen():
            for group in self.tiles_along_Y:
                overlap_sum = (
                    group['ysize'] - group['Ys'].diff().apply(rint)).sum()
                overlap_sum -= group.iloc[0]['ysize']
                yield rint(
                    group['ysize'].sum() - overlap_sum + group.iloc[0]['Ys'])

        return max(x for x in gen())

    @property
    def full_thickness(self):
        return int(round(
            (self.data_frame['Zs'] + self.data_frame['nfrms']).max()))
