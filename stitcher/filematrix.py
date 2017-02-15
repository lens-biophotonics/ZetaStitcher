"""Parse input file names."""

import os
import re
import logging

import pandas as pd


logger = logging.getLogger('FileMatrix')


def parse_input_file(file_name):
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
        """A :py:class:`pandas.DataFrame` object. Contains four columns: `X`,
        `Y`, `Z` and `filename`."""

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
                list += parse_input_file(f)
                list.append(f)
            except RuntimeError as e:
                logger.error(e.args[0])

        data = {'X': list[0::4], 'Y': list[1::4], 'Z': list[2::4],
                'filename': list[3::4]}
        df = pd.DataFrame(data)
        self.data_frame = df.sort_values(['Z', 'Y', 'X'])
