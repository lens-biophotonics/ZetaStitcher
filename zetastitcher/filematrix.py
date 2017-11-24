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

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    m = re.search('^.*x_([-]?\d+).*y_([-]?\d+).*z_([-]?\d+).*', file_name)
    if m is None:
        m = re.search('^.*x_([-]?\d+).*y_([-]?\d+)', file_name)
    if m is None:
        m = re.search('^([-]?\d+)_([-]?\d+)_([-]?\d+)', file_name)
    if m is None:
        m = re.search('^([-]?\d+)_([-]?\d+)', file_name)
    if m is None:
        raise ValueError('Invalid name {}'.format(file_name))

    fields = []
    for i in range(1, 4):
        try:
            fields.append(int(m.group(i)))
        except IndexError:
            fields.append(0)

    logger.info('adding {} \tX={} Y={} Z={}'.format(file_name, *fields))
    return fields


class FileMatrix:
    """Data structures for a matrix of input files."""
    def __init__(self, input_path=None, ascending_tiles_x=True,
                 ascending_tiles_y=True):
        """
        Construct a FileMatrix object from a directory path or a .yml file
        produced by the stitcher. Tile ordering parameters need to be
        specified only if constructing from a directory, otherwise they are
        ignored.

        Parameters
        ----------
        input_path : str
                     input path (directory) or file (.yml)
        ascending_tiles_x : bool
            whether tiles are supposed to be read in ascending X order
        ascending_tiles_y : bool
            whether tiles are supposed to be read in ascending Y order
        """
        self.input_path = input_path

        self.data_frame = None
        """A :class:`pandas.DataFrame` object. Contains the following
        columns: `X`, `Y`, `Z`, `Z_end`, `xsize`, `ysize`, `nfrms`,
        `filename`."""

        self.ascending_tiles_x = ascending_tiles_x
        self.ascending_tiles_y = ascending_tiles_y

        self.name_array = None

        if input_path is None:
            return

        if os.path.isdir(input_path):
            self.load_dir(input_path)
        elif os.path.isfile(input_path):
            self.load_yaml(input_path)

    def load_dir(self, dir=None):
        """Look for files in `dir` recursively and populate data structures.

        Parameters
        ----------
        dir : path
        """
        if dir is None:
            dir = self.input_path

        if dir is None:
            return

        flist = []

        for root, dirs, files in os.walk(dir, followlinks=True):
            if os.path.basename(root):
                try:
                    self.parse_and_append(root, flist)
                    continue
                except (RuntimeError, ValueError):
                    pass

            for f in files:
                try:
                    self.parse_and_append(os.path.join(root, f), flist)
                    continue
                except (RuntimeError, ValueError):
                    pass

        data = {'X': flist[0::7], 'Y': flist[1::7], 'Z': flist[2::7],
                'nfrms': flist[3::7], 'ysize': flist[4::7],
                'xsize': flist[5::7], 'filename': flist[6::7]}
        df = pd.DataFrame(data)
        df = df.sort_values(['Z', 'Y', 'X'])

        if not self.ascending_tiles_x:
            df['X'] = (df['X'] - df['X'].max()).abs()

        if not self.ascending_tiles_y:
            df['Y'] = (df['Y'] - df['Y'].max()).abs()

        self.data_frame = df.set_index('filename')
        self.process_data_frame()

    def load_yaml(self, fname):
        logger.info('loading {}'.format(fname))
        with open(fname, 'r') as f:
            y = yaml.load(f)

        self.data_frame = pd.DataFrame(y['filematrix']).set_index('filename')
        self.data_frame = self.data_frame.sort_values(['Z', 'Y', 'X'])

        self.process_data_frame()
        self.input_path = fname

    def process_data_frame(self):
        df = self.data_frame

        xsize = df['X'].unique().size
        ysize = df['Y'].unique().size
        n_of_files = len(df.index)

        if xsize * ysize != n_of_files:
            msg = 'Mosaic is {}x{} tiles, but there are {} files!'.format(
                xsize, ysize, n_of_files)
            raise ValueError(msg)

        keys = ['X', 'Y', 'Z']
        df[keys] -= df[keys].min()

        cols = df.columns
        if 'Xs' in cols and 'Ys' in cols and 'Zs' in cols:
            for key in ['Xs', 'Ys', 'Zs']:
                df[key] -= df[key].min()

        df = df.sort_values(['Z', 'Y', 'X'])
        self.compute_end_pos()
        self.name_array = np.array(df.index.values).reshape(self.Ny, self.Nx)

    def parse_and_append(self, name, flist):
        try:
            fields = parse_file_name(name)
            with InputFile(name) as infile:
                fields.append(infile.nfrms)
                fields.append(infile.ysize)
                fields.append(infile.xsize)
            flist += fields
            flist.append(name)
        except (RuntimeError, ValueError):
            raise

    def save_to_yaml(self, filename, mode):
        keys = ['X', 'Y', 'Z', 'nfrms', 'xsize', 'ysize']
        abs_keys = ['Xs', 'Ys', 'Zs']
        for k in abs_keys:
            if k in self.data_frame.columns:
                keys.append(k)
        df = self.data_frame[keys].reset_index()
        j = json.loads(df.to_json(orient='records'))

        if mode == 'update':
            with open(filename, 'r') as f:
                y = yaml.load(f)

            y['filematrix'] = j

            logger.info('updating {}'.format(filename))
            with open(filename, 'w') as f:
                yaml.dump(y, f, default_flow_style=False)
        else:
            logger.info('writing {}'.format(filename))
            with open(filename, mode) as f:
                yaml.dump({'filematrix': j}, f, default_flow_style=False)

    def clear_absolute_positions(self):
        keys = ['Xs', 'Ys', 'Zs', 'Xs_end', 'Ys_end', 'Zs_end']
        for k in keys:
            try:
                del self.data_frame[k]
            except KeyError:
                pass

    def compute_end_pos(self):
        df = self.data_frame

        keys = ['X', 'Y', 'Z']
        sizes = ['xsize', 'ysize', 'nfrms']

        cols = df.columns
        if 'Xs' in cols and 'Ys' in cols and 'Zs' in cols:
            keys += ['Xs', 'Ys', 'Zs']
            sizes *= 2

        keys_end = [k + '_end' for k in keys]

        for ke, k, s in zip(keys_end, keys, sizes):
            df[ke] = df[k] + df[s]

    def compute_nominal_positions(self, px_size_z, px_size_xy):
        df = self.data_frame

        df['Xs'] = (df['X'] // px_size_xy).astype(np.int)
        df['Ys'] = (df['Y'] // px_size_xy).astype(np.int)

        df['Zs'] = (df['Z'] // px_size_z).astype(np.int)

        self.compute_end_pos()

    @property
    def Nx(self):
        return self.data_frame['X'].unique().size

    @property
    def Ny(self):
        return self.data_frame['Y'].unique().size

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
                got[0], ascending=True).groupby(got[1])
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
        yield g.send((['Z', 'X', 'Y'], 'Y'))
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
        yield g.send((['Z', 'Y', 'X'], 'X'))
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
