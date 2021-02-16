"""API to query an arbitrary region in the stitched volume."""

import math
import os.path
import logging
import threading

from queue import Queue
from functools import lru_cache

import numpy as np

from zetastitcher.align.filematrix import FileMatrix
from zetastitcher.io.inputfile import InputFile
from zetastitcher.fuse.overlaps import Overlaps
from zetastitcher.fuse.fuse import fuse_queue

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def to_dtype(x, dtype):
    if x.dtype == dtype:
        return x
    if np.issubdtype(dtype, np.integer):
        np.rint(x, x)
    return x.astype(dtype, copy=False)


class VirtualFusedVolume:
    """An API to query arbitrary regions in the stitched volume.

    Example usage:

    >>> vfv = VirtualFusedVolume('stitch.yml')
    >>> vfv.shape
    (208, 3, 4533, 2487)

    Axis order is ZCYX.

    >>> subvolume = vfv[40, ..., 1000:1500, 1800:2400]

    Use `peek` to get a list of tiles that would be accessed for a given query:

    >>> vfv.peek[40, ..., 1000:1500, 1800:2400]
    [
        ('0000_0000.tiff', (slice(40, 41, 1), slice(1000, 1500, 1), slice(1800, 2048, 1))),
        ('0100_0000.tiff', (slice(40, 41, 1), slice(1000, 1500, 1), slice(0, 453, 1))),
    ]
    """
    def __init__(self, file_or_matrix):
        if isinstance(file_or_matrix, str):
            self.path, _ = os.path.split(file_or_matrix)
            self.fm = FileMatrix(file_or_matrix)
        else:
            self.fm = file_or_matrix
            self.path, _ = os.path.split(self.fm.input_path)

        absolute_positions_found = True
        abs_keys = ['Xs', 'Ys', 'Zs']
        for k in abs_keys:
            if not k in self.fm.data_frame.columns:
                absolute_positions_found = False
                break

        if absolute_positions_found:
            self.ov = Overlaps(self.fm)
        else:
            self.fm.compute_nominal_positions(1, 1)
            self.ov = None

        self._debug = False

        infile = os.path.join(self.path, self.fm.data_frame.iloc[0].name)
        with InputFile(infile) as f:
            self.temp_shape = list(f.shape)
            self.dtype = f.dtype
            self.nchannels = f.nchannels

        self.squeeze_enabled = True

    @property
    def overlay_debug_enabled(self):
        """Whether to overlay debug information (tile edges and numbers).

        Defaults to `False`."""
        return self._debug

    @overlay_debug_enabled.setter
    def overlay_debug_enabled(self, value):
        self._debug = value

    @property
    @lru_cache()
    def shape(self):
        """Shape of the whole stitched volume.

        Axis order is ZCYX.
        """
        thickness = self.fm.full_thickness

        infile = os.path.join(self.path, self.fm.data_frame.iloc[0].name)
        with InputFile(infile) as f:
            output_shape = list(f.shape)

        output_shape[0] = thickness
        output_shape[-2] = self.fm.full_height
        output_shape[-1] = self.fm.full_width

        return tuple(output_shape)

    def _compute_vars(self, item):
        # ensure item is a tuple
        if isinstance(item, list):
            item = tuple(item)
        else:
            item = np.index_exp[item]

        # ensure all items are slice objects
        myitem = []
        flip_axis = []
        for i in item:
            if isinstance(i, int):
                start = i
                stop = i + 1
                step = 1
            elif i is Ellipsis:
                for _ in range(0, len(self.shape) - len(item) + 1):
                    myitem.append(slice(0, self.shape[len(myitem)], 1))
                    flip_axis.append(1)
                continue
            elif isinstance(i, slice):
                start = i.start
                stop = i.stop
                step = i.step if i.step is not None else 1
            else:
                raise TypeError("Invalid type: {}".format(type(i)))

            curr_max = self.shape[len(myitem)]
            if start is None:
                start = 0 if step > 0 else curr_max
            elif start < 0:
                start += curr_max
                if stop is not None:
                    stop += curr_max
            elif start > curr_max:
                start = curr_max

            if stop is None:
                stop = curr_max if step > 0 else 0
            elif stop < 0:
                stop += curr_max
            elif stop > curr_max:
                stop = curr_max

            if step < 0:
                start += abs(step)
                stop += abs(step)

            flip_axis.append(1 if step > 0 else -1)

            if start < stop:
                myitem.append(slice(start, stop, abs(step)))
            else:
                myitem.append(slice(stop, start, abs(step)))

        for _ in range(0, len(self.shape) - len(myitem)):
            myitem.append(slice(0, self.shape[len(myitem)], 1))

        if len(myitem) != len(self.shape):
            raise IndexError('Too many indices for array')

        zmin = myitem[0].start
        zmax = myitem[0].stop

        ymin = myitem[-2].start
        ymax = myitem[-2].stop

        xmin = myitem[-1].start
        xmax = myitem[-1].stop

        X_min = np.array([myitem[i].start for i in [0, -2, -1]])
        X_stop = np.array([myitem[i].stop for i in [0, -2, -1]])
        steps = np.array([myitem[i].step for i in [0, -2, -1]])

        df = self.fm.data_frame
        df = df[
            (df['Zs_end'] > zmin) & (df['Zs'] < zmax)
            & (df['Ys_end'] > ymin) & (df['Ys'] < ymax)
            & (df['Xs_end'] > xmin) & (df['Xs'] < xmax)
        ]

        return df, myitem, X_min, X_stop, steps, flip_axis

    @property
    def peek(self):
        class WrapperClass:
            def __init__(self, obj):
                self.obj = obj

            def __getitem__(self, item):
                df, myitem, X_min, X_stop, steps, flip_axis = self.obj._compute_vars(item)

                return [(index, tuple(sl)) for index, _, sl in self.obj._my_gen(df, X_min, X_stop, steps, myitem[:])]

        return WrapperClass(self)

    def __getitem__(self, item):
        df, myitem, X_min, X_stop, steps, flip_axis = self._compute_vars(item)

        if self.ov is None or df.shape[0] == 1:
            dtype = self.dtype
        else:
            dtype = np.float32

        output_shape = [
            (it.stop - it.start) / it.step for it in myitem
        ]
        output_shape = list(map(math.ceil, output_shape))
        if 0 in output_shape:
            return np.array([], dtype=self.dtype)

        fused = np.zeros(output_shape, dtype=dtype)

        q = Queue(maxsize=20)

        t = threading.Thread(
            target=fuse_queue,
            args=(q, fused, self.temp_shape[-2::]),
            kwargs={
                'debug': self._debug,
            }
        )
        t.start()

        for index, Xs, sl in self._my_gen(df, X_min, X_stop, steps, myitem[:]):
            logger.info('loading {}\t{}'.format(index, sl))
            with InputFile(os.path.join(self.path, index)) as f:
                f.squeeze = False
                sl_a = np.copy(f[tuple(sl)]).astype(dtype)

            z_from = sl[0].start
            z_to = sl[0].stop

            x_from = np.array([sl[i].start for i in [0, -2, -1]])

            Top_left = Xs + x_from
            top_left = (Top_left - X_min) // steps

            if self.ov is None or df.shape[0] == 1:
                overlaps = None
            else:
                overlaps = self.ov[index]
                overlaps = overlaps.loc[
                    (overlaps['Z_from'] <= z_to) & (overlaps['Z_to'] >= z_from)
                    ].copy()

                overlaps['Z_from'] -= z_from
                overlaps['Z_to'] -= z_from
                overlaps['Z_to'] /= abs(sl[0].step)
                overlaps['Z_to'] = overlaps['Z_to'].apply(np.round).astype(int)

                overlaps.loc[overlaps['Z_from'] < 0, 'Z_from'] = 0

            q.put([sl_a, index, z_from, tuple(sl), top_left, overlaps])

        q.put(None)  # close queue

        t.join()  # wait for fuse thread to finish

        fused = to_dtype(fused, self.dtype)

        ie = tuple([slice(None, None, flip) for flip in flip_axis])
        fused = fused[ie]

        if self.squeeze_enabled:
            return np.squeeze(fused)
        return fused

    @staticmethod
    def _my_gen(df, X_min, X_stop, steps, sl):
        for row in df.itertuples():
            index = row.Index
            Xs = np.array([row.Zs, row.Ys, row.Xs])
            xsize = np.array([row.nfrms, row.ysize, row.xsize])

            xto = X_stop - Xs
            xto[xto > xsize] = xsize[xto > xsize]
            xfrom = X_min - Xs
            xfrom[xfrom < 0] = 0
            xfrom = xfrom + (Xs + xfrom - X_min) % steps

            for i in [0, -2, -1]:
                sl[i] = slice(xfrom[i], xto[i], steps[i])

            yield index, Xs, sl
