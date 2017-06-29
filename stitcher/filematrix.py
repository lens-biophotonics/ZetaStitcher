"""Parse input file names, compute tile coordinates."""

import os
import re
import logging

import json
import yaml

import numpy as np
import pandas as pd
import networkx as nx

from .inputfile import InputFile
from .global_optimization import absolute_position_global_optimization


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
        columns: `X`, `Y`, `Z`, `Z_end`, `xsize`, `ysize`, `nfrms`,
        `filename`."""
        self.stitch_data_frame = None

        self.ascending_tiles_x = True
        self.ascending_tiles_y = True

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
            self.load_yaml(directory)

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

    def load_yaml(self, fname):
        with open(fname, 'r') as f:
            y = yaml.load(f)

        attrs = ['ascending_tiles_x', 'ascending_tiles_y']

        for attr in attrs:
            setattr(self, attr, y['xcorr-options'][attr])

        df = pd.DataFrame(y['xcorr'])

        a = np.concatenate((df['aname'].unique(), df['bname'].unique()))
        a = np.unique(a)

        flist = []
        self.dir, _ = os.path.split(fname)
        for el in a:
            self.parse_and_append(el, flist)

        self._load_from_flist(flist)

        new_columns = df.columns.values
        new_columns[0] = 'filename'
        df.columns = new_columns
        self.stitch_data_frame = df.set_index('filename')

        fm_df = self.data_frame
        keys = ['X', 'Y', 'Z']
        fm_df[keys] -= fm_df[keys].min()

        if not self.ascending_tiles_x:
            fm_df['X'] = (fm_df['X'] - fm_df['X'].max()).abs()

        if not self.ascending_tiles_y:
            fm_df['Y'] = (fm_df['Y'] - fm_df['Y'].max()).abs()

        self._compute_shift_vectors()
        abs_keys = ['Xs', 'Ys', 'Zs', 'Xs_end', 'Ys_end', 'Zs_end']
        abs_yaml_key = 'absolute_positions'
        if abs_yaml_key in y:
            df = pd.DataFrame(y[abs_yaml_key]).set_index('filename')
            fm_df[abs_keys] = df[abs_keys]
        else:
            self._compute_absolute_positions_initial_guess()
            absolute_position_global_optimization(self.data_frame,
                                                  self.stitch_data_frame,
                                                  y['xcorr-options'])
            with open(fname, 'a') as f:
                df = fm_df[abs_keys].reset_index()
                yaml.dump(
                    {
                        abs_yaml_key: json.loads(df.to_json(orient='records'))
                    }, f, default_flow_style=False)
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

    def save_to_yaml(self, filename, mode):
        j = json.loads(self.data_frame.reset_index().to_json(orient='records'))
        with open(filename, mode) as f:
            yaml.dump({'filematrix': j}, f, default_flow_style=False)

    def _compute_shift_vectors(self):
        sdf = self.stitch_data_frame
        fm_df = self.data_frame

        sdf['px'] = 0
        sdf['py'] = 0
        sdf['pz'] = 0

        idx = sdf['axis'] == 1
        fm_df_idx = idx[idx].index
        sdf.loc[fm_df_idx, 'px'] = sdf.loc[idx, 'dx']
        sdf.loc[fm_df_idx, 'py'] = fm_df.loc[fm_df_idx, 'ysize'] - sdf.loc[
            idx, 'dy']
        sdf.loc[fm_df_idx, 'pz'] = sdf.loc[idx, 'dz']

        idx = sdf['axis'] == 2
        fm_df_idx = idx[idx].index
        sdf.loc[idx, 'px'] = fm_df.loc[fm_df_idx, 'xsize'] - sdf.loc[idx, 'dy']
        sdf.loc[idx, 'py'] = -sdf.loc[idx, 'dx']
        sdf.loc[idx, 'pz'] = sdf.loc[idx, 'dz']

    def _compute_absolute_positions_initial_guess(self):
        sdf = self.stitch_data_frame.reset_index()
        fm_df = self.data_frame

        G = nx.Graph()
        G.add_edges_from(((
            u, v, {'weight': w}) for u, v, w in
            np.c_[sdf['filename'], sdf['bname'], 1 - sdf['score']]))

        keys = ['Xs', 'Ys', 'Zs']
        for k in keys:
            fm_df[k] = 0

        sdf = sdf.reset_index().set_index('bname')

        cond = (fm_df['X'] == 0) & (fm_df['Y'] == 0) & (fm_df['Z'] == 0)
        top_left_corner = fm_df[cond].index[0]
        for edge in nx.bfs_edges(G, source=top_left_corner):
            btile = edge[1]
            parents = fm_df.loc[sdf.loc[[btile], 'filename']]
            temp_sdf = sdf.loc[[btile]].set_index('filename')
            temp = pd.DataFrame()
            temp['Xs'] = parents['Xs'] + temp_sdf['px']
            temp['Ys'] = parents['Ys'] + temp_sdf['py']
            temp['Zs'] = parents['Zs'] + temp_sdf['pz']
            temp['score'] = temp_sdf['score']
            fm_df.loc[btile, keys] = temp.apply(
                lambda x: np.average(x, weights=temp.loc[x.index, 'score']))

        fm_df[keys] -= fm_df[keys].min()
        fm_df[keys] = fm_df[keys].apply(np.round).astype(int)

        fm_df['Xs_end'] = fm_df['Xs'] + fm_df['xsize']
        fm_df['Ys_end'] = fm_df['Ys'] + fm_df['ysize']
        fm_df['Zs_end'] = fm_df['Zs'] + fm_df['nfrms']

    def _compute_overlaps(self):
        def comp_diff(dest, row_name, row, other_name):
            other = fm_df.loc[other_name]
            temp = pd.Series()
            temp['Z_from'] = max(other['Zs'], row['Zs']) - row['Zs']
            temp['Z_to'] = min(other['Zs_end'], row['Zs_end']) - row['Zs']

            temp['Y_from'] = max(other['Ys'], row['Ys']) - row['Ys']
            temp['Y_to'] = min(other['Ys_end'], row['Ys_end']) - row['Ys']

            temp['X_from'] = max(other['Xs'], row['Xs']) - row['Xs']
            temp['X_to'] = min(other['Xs_end'], row['Xs_end']) - row['Xs']

            cols_to_zero(dest, row_name)

            if temp['Z_from'] > temp['Z_to'] or temp['Y_from'] > temp['Y_to'] \
                    or temp['X_from'] > temp['X_to']:
                pass
            else:
                dest.loc[row_name] = temp

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
        df = pd.DataFrame()
        names = ['n', 's', 'e', 'w', 'nw', 'ne', 'sw', 'se']
        for n in names:
            name = 'overlap_' + n
            overlap = getattr(self, name)
            df[n] = overlap.loc[tile_name]
        return df.transpose()

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
