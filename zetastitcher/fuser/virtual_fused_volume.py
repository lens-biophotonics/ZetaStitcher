"""API to query an arbitrary region in the stitched volume."""

import os.path
import logging
import threading

from queue import Queue
from functools import lru_cache

import numpy as np

from .overlaps import Overlaps
from ..inputfile import InputFile
from ..filematrix import FileMatrix
from .fuse import fuse_queue, to_dtype

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class VirtualFusedVolume:
    """An API to query arbitrary regions in the stitched volume.

    Example usage:

    >>> vfv = VirtualFusedVolume('stitch.yml')
    >>> vfv.shape
    (208, 3, 4533, 2487)

    Axis order is ZCYX.

    >>> subvolume = vfv[40, ..., 1000:1500, 2000:2400]

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

    def __getitem__(self, item):

        item = np.index_exp[item]  # ensure item is a tuple

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

        output_shape = [
            (it.stop - it.start) // it.step for it in myitem
        ]
        if 0 in output_shape:
            return np.array([], dtype=self.dtype)

        dtype = np.float32 if self.ov is not None else self.dtype
        fused = np.zeros(output_shape, dtype=dtype)

        q = Queue(maxsize=20)

        t = threading.Thread(
            target=fuse_queue,
            args=(q, fused, self.temp_shape[-2::], self._debug))
        t.start()

        sl = myitem[:]

        X_min = np.array([myitem[i].start for i in [0, -2, -1]])
        X_stop = np.array([myitem[i].stop for i in [0, -2, -1]])
        steps = np.array([myitem[i].step for i in [0, -2, -1]])

        df = self.fm.data_frame
        df = df[
            (df['Zs_end'] > zmin) & (df['Zs'] < zmax)
            & (df['Ys_end'] > ymin) & (df['Ys'] < ymax)
            & (df['Xs_end'] > xmin) & (df['Xs'] < xmax)
        ]

        if df.shape[0] == 1:
            index = df.iloc[0].name
            with InputFile(os.path.join(self.path, index)) as f:
                try:
                    f.wrapper.deep_copy_enabled = True
                except AttributeError:
                    pass
                logger.info('opening {}\t{}'.format(index, sl))
                sl_a = f[item]
            return sl_a

        for index, row in df.iterrows():
            Xs = np.array([row.Zs, row.Ys, row.Xs])
            xsize = np.array([row.nfrms, row.ysize, row.xsize])

            xto = X_stop - Xs
            xto[xto > xsize] = xsize[xto > xsize]
            xfrom = X_min - Xs
            xfrom[xfrom < 0] = 0
            xfrom = xfrom + (Xs + xfrom - X_min) % steps

            for i in [0, -2, -1]:
                sl[i] = slice(xfrom[i], xto[i], steps[i])

            z_from = sl[0].start
            z_to = sl[0].stop

            x_from = np.array([sl[i].start for i in [0, -2, -1]])

            logger.info('opening {}\t{}'.format(index, sl))
            with InputFile(os.path.join(self.path, index)) as f:
                sl_a = np.copy(f[tuple(sl)]).astype(dtype)
            sl_a.shape = ([1 for _ in range(0, len(sl) - len(sl_a.shape))]
                          + list(sl_a.shape))

            Top_left = Xs + x_from
            top_left = (Top_left - X_min) // steps

            if self.ov is None:
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

        q.put([None, None, None, None, None, None])  # close queue

        t.join()  # wait for fuse thread to finish

        fused = to_dtype(fused, self.dtype)

        ie = [slice(None, None, flip) for flip in flip_axis]
        fused = fused[ie]

        return np.squeeze(fused)
