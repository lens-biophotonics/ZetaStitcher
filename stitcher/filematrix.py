"""Parse input file names."""

import os
import re
import logging

import numpy as np


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

        self.input_files = None
        """List of found files"""

        self.stage_coords = None
        """Numpy array with nominal stage coordinates, as parsed from file
        names. Shape is (*n*, 3) where *n* is the length of
        :py:attr:`~.input_files`."""

        self.load_dir(directory)

    def load_dir(self, dir=None):
        """Look for files in `dir` and its subfolders and populate data
        structures.

        Parameters
        ----------
        dir : path
        """
        if dir is None:
            dir = self.dir

        input_files = []

        for root, dirs, files in os.walk(dir):
            for f in files:
                input_files.append(os.path.join(root, f))

        stage_coords = np.zeros((len(input_files), 3), dtype=np.uint32)

        self.input_files = []
        for i, f in enumerate(input_files):
            try:
                stage_coords[i, :] = parse_input_file(f)
                self.input_files.append(f)
            except RuntimeError as e:
                logger.error(e.args[0])

        self.stage_coords = stage_coords
