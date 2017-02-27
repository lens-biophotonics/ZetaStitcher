"""Parse input file names."""

import os
import re
import logging

import pandas as pd
import networkx as nx

import dcimg


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
    m = re.search('^x_(\d+).*y_(\d+).*z_(\d+).*zstep_(\d+)', file_name)
    if m is None:
        m = re.search('^(\d+)_(\d+)_(\d+)', file_name)
    if m is None:
        raise RuntimeError('Invalid file name {}'.format(file_name))

    fields = []
    for i in range(1, 4):
        fields.append(int(m.group(i)))

    return fields


class FileMatrix:
    """Data structures for a matrix of input files."""
    def __init__(self, directory):
        self.dir = directory

        self.data_frame = None
        """A :py:class:`pandas.DataFrame` object. Contains the following
        columns: `X`, `Y`, `Z`, `Z_end`."""

        self.load_dir(directory)

    def load_dir(self, dir=None):
        """Look for files in `dir` recursively and populate data structures.

        Parameters
        ----------
        dir : path
        """
        if dir is None:
            dir = self.dir

        input_files = []
        list = []

        for root, dirs, files in os.walk(dir):
            for f in files:
                input_files.append(os.path.join(root, f))

        for i, f in enumerate(input_files):
            try:
                fields = parse_file_name(f)
                with dcimg.DCIMGFile(f) as dc:
                    fields.append(dc.nfrms)
                list += fields
                list.append(f)
            except (RuntimeError, ValueError) as e:
                logger.error(e.args[0])

        data = {'X': list[0::5], 'Y': list[1::5], 'Z': list[2::5],
                'nfrms': list[3::5], 'filename': list[4::5]}
        df = pd.DataFrame(data)
        df = df.sort_values(['Z', 'Y', 'X'])
        df['Z_end'] = df['Z'] + df['nfrms']

        self.data_frame = df

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
