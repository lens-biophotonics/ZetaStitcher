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

        self.overlap_n = None
        self.overlap_s = None
        self.overlap_e = None
        self.overlap_w = None
        self.overlap_nw = None
        self.overlap_ne = None
        self.overlap_sw = None
        self.overlap_se = None

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
            raise

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
        def comp_diff(dest, row_name, row, other_name):
            temp_other_name = other_name
            temp = row
            if fm_df.loc[other_name, 'Z'] < temp['Z'] or fm_df.loc[
                         other_name, 'Y'] < temp['Y'] or fm_df.loc[
                         other_name, 'X'] < temp['X']:
                temp_other_name = row_name
                temp = fm_df.loc[other_name]

            if fm_df.loc[temp_other_name, 'Zs'] > temp['Zs_end'] or fm_df.loc[
                temp_other_name, 'Ys'] > temp['Ys_end'] or fm_df.loc[
                temp_other_name, 'Xs'] > temp['Xs_end']:
                cols_to_zero(dest, row_name)
                return

            dest.loc[row_name, 'Z_from'] = max(fm_df.loc[other_name, 'Zs'],
                                               row['Zs']) - row['Zs']
            dest.loc[row_name, 'Z_to'] = min(fm_df.loc[other_name, 'Zs_end'],
                                             row['Zs_end']) - row['Zs']

            dest.loc[row_name, 'Y_from'] = max(fm_df.loc[other_name, 'Ys'],
                                               row['Ys']) - row['Ys']
            dest.loc[row_name, 'Y_to'] = min(fm_df.loc[other_name, 'Ys_end'],
                                             row['Ys_end']) - row['Ys']

            dest.loc[row_name, 'X_from'] = max(fm_df.loc[other_name, 'Xs'],
                                               row['Xs']) - row['Xs']
            dest.loc[row_name, 'X_to'] = min(fm_df.loc[other_name, 'Xs_end'],
                                             row['Xs_end']) - row['Xs']

        def cols_to_zero(dest, row_name):
            cols = ['Z_from', 'Z_to', 'Y_from', 'Y_to', 'X_from', 'X_to']
            for c in cols:
                dest.loc[row_name, c] = 0

        fm_df = self.data_frame
        sdf = self.stitch_data_frame

        overlap_n = pd.DataFrame()
        overlap_s = pd.DataFrame()
        overlap_e = pd.DataFrame()
        overlap_w = pd.DataFrame()
        overlap_nw = pd.DataFrame()
        overlap_ne = pd.DataFrame()
        overlap_sw = pd.DataFrame()
        overlap_se = pd.DataFrame()

        tmp_df_b = sdf.reset_index().set_index(['bname', 'axis'])
        tmp_df_a = sdf.reset_index().set_index(['filename', 'axis'])

        for index, row in fm_df.iterrows():
            # north
            try:
                parent = tmp_df_b.loc[(row.name, 1), 'filename']
                comp_diff(overlap_n, row.name, row, parent)
            except KeyError:
                cols_to_zero(overlap_n, row.name)

            # south
            try:
                parent = tmp_df_a.loc[(row.name, 1), 'bname']
                comp_diff(overlap_s, row.name, row, parent)
            except KeyError:
                cols_to_zero(overlap_s, row.name)

            # east
            try:
                parent = tmp_df_a.loc[(row.name, 2), 'bname']
                comp_diff(overlap_e, row.name, row, parent)
            except KeyError:
                cols_to_zero(overlap_e, row.name)

            # west
            try:
                parent = tmp_df_b.loc[(row.name, 2), 'filename']
                comp_diff(overlap_w, row.name, row, parent)
            except KeyError:
                cols_to_zero(overlap_w, row.name)

            # north-west
            try:
                other_name = tmp_df_b.loc[
                    (row.name, 2), 'filename']  # one step W
                other_name = tmp_df_b.loc[
                    (other_name, 1), 'filename']  # one step N
                comp_diff(overlap_nw, row.name, row, other_name)
            except KeyError:
                cols_to_zero(overlap_nw, row.name)

            # north-east
            try:
                other_name = tmp_df_a.loc[(row.name, 2), 'bname']  # one step E
                other_name = tmp_df_b.loc[
                    (other_name, 1), 'filename']  # one step N
                comp_diff(overlap_ne, row.name, row, other_name)
            except KeyError:
                cols_to_zero(overlap_ne, row.name)

            # south-west
            try:
                other_name = tmp_df_b.loc[
                    (row.name, 2), 'filename']  # one step W
                other_name = tmp_df_a.loc[
                    (other_name, 1), 'bname']  # one step S
                comp_diff(overlap_sw, row.name, row, other_name)
            except KeyError:
                cols_to_zero(overlap_sw, row.name)

            # south-east
            try:
                other_name = tmp_df_a.loc[(row.name, 2), 'bname']  # one step E
                other_name = tmp_df_a.loc[
                    (other_name, 1), 'bname']  # one step S
                comp_diff(overlap_se, row.name, row, other_name)
            except KeyError:
                cols_to_zero(overlap_se, row.name)

        self.overlap_n = overlap_n.astype(int)
        self.overlap_s = overlap_s.astype(int)
        self.overlap_e = overlap_e.astype(int)
        self.overlap_w = overlap_w.astype(int)
        self.overlap_nw = overlap_nw.astype(int)
        self.overlap_ne = overlap_ne.astype(int)
        self.overlap_sw = overlap_sw.astype(int)
        self.overlap_se = overlap_se.astype(int)

    def overlaps(self, tile_name):
        return [x.loc[tile_name] for x in (
            self.overlap_n, self.overlap_s, self.overlap_w, self.overlap_e,
            self.overlap_nw, self.overlap_ne, self.overlap_sw,
            self.overlap_se)]

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
    def full_width(self):
        return self.data_frame['Xs_end'].max()

    @property
    def full_height(self):
        return self.data_frame['Ys_end'].max()

    @property
    def full_thickness(self):
        return self.data_frame['Zs_end'].max()
