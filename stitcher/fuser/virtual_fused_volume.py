import os.path
import logging
import threading

from queue import Queue
from functools import lru_cache

import numpy as np

from .overlaps import Overlaps
from ..inputfile import InputFile
from ..filematrix import FileMatrix
from .fuse import fuse_queue

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class VirtualFusedVolume:
    def __init__(self, file_or_matrix):
        if isinstance(file_or_matrix, str):
            self.path, _ = os.path.split(file_or_matrix)
            self.fm = FileMatrix(file_or_matrix)
        else:
            self.fm = file_or_matrix
            self.path, _ = os.path.split(self.fm.input_path)

        abs_keys = ['Xs', 'Ys', 'Zs']
        for k in abs_keys:
            if not k in self.fm.data_frame.columns:
                raise ValueError('Absolute coordinates not found')

        self.ov = Overlaps(self.fm)

        infile = os.path.join(self.path, self.fm.data_frame.iloc[0].name)
        with InputFile(infile) as f:
            self.temp_shape = list(f.shape)

    @property
    @lru_cache()
    def shape(self):
        thickness = self.fm.full_thickness

        infile = os.path.join(self.path, self.fm.data_frame.iloc[0].name)
        with InputFile(infile) as f:
            output_shape = list(f.shape)

        output_shape[0] = thickness
        output_shape[-2] = self.fm.full_height
        output_shape[-1] = self.fm.full_width

        return output_shape

    def __getitem__(self, item):

        item = np.index_exp[item]  # ensure item is a tuple

        # ensure all items are slice objects
        myitem = []
        for i in item:
            if isinstance(i, int):
                start = i
                stop = i + 1
                step = 1
            elif i is Ellipsis:
                for _ in range(0, len(self.shape) - len(item) + 1):
                    myitem.append(slice(0, self.shape[len(myitem)], 1))
                continue
            elif isinstance(i, slice):
                start = i.start
                stop = i.stop
                step = i.step if i.step is not None else 1
            else:
                raise TypeError("Invalid type: {}".format(type(i)))

            if start is None:
                start = 0 if step > 0 else self.shape[len(myitem)]
            elif start < 0:
                start += self.shape[len(myitem)]

            if stop is None:
                stop = self.shape[len(myitem)] if step > 0 else 0
            elif stop < 0:
                stop += self.shape[len(myitem)]

            myitem.append(slice(start, stop, step))

        for _ in range(0, len(self.shape) - len(myitem)):
            myitem.append(slice(0, self.shape[len(myitem)], 1))

        zmin = myitem[0].start
        zmax = myitem[0].stop

        ymin = myitem[-2].start
        ymax = myitem[-2].stop

        xmin = myitem[-1].start
        xmax = myitem[-1].stop

        df = self.fm.data_frame
        df = df[
            (df['Zs_end'] > zmin) & (df['Zs'] <= zmax)
            & (df['Ys_end'] > ymin) & (df['Ys'] <= ymax)
            & (df['Xs_end'] > xmin) & (df['Xs'] <= xmax)
        ]

        output_shape = [(it.stop - it.start) // abs(it.step) for it in myitem]

        fused = np.zeros(output_shape, dtype=np.float32)

        q = Queue(maxsize=20)

        t = threading.Thread(target=fuse_queue,
                             args=(q, fused, self.temp_shape[-2::], True))
        t.start()

        sl = myitem[:]

        for index, row in df.iterrows():
            for start_key, size_key, i in zip(
                ['Zs', 'Ys', 'Xs'],
                ['nfrms', 'ysize', 'xsize'],
                [0, -2, -1],
            ):
                xto = myitem[i].stop - getattr(row, start_key)
                if xto > getattr(row, size_key):
                    xto = getattr(row, size_key)

                xfrom = myitem[i].start - getattr(row, start_key)
                if xfrom < 0:
                    xfrom = 0

                sl[i] = slice(xfrom, xto, sl[i].step)

            with InputFile(os.path.join(self.path, index)) as f:
                logger.info('opening {}\t{}'.format(index, sl))
                sl_a = np.copy(f[tuple(sl)]).astype(np.float32)

            z_from = sl[0].start
            z_to = sl[0].stop

            y_from = sl[-2].start
            x_from = sl[-1].start

            top_left = [row.Zs + z_from - zmin,
                        row.Ys + y_from - ymin,
                        row.Xs + x_from - xmin]
            overlaps = self.ov[index].copy()
            overlaps = overlaps.loc[
                (overlaps['Z_from'] <= z_to) & (overlaps['Z_to'] >= z_from)
                ]

            overlaps['Z_from'] -= z_from
            overlaps['Z_to'] -= z_from

            overlaps.loc[overlaps['Z_from'] < 0, 'Z_from'] = 0

            q.put([sl_a, index, z_from, tuple(sl), top_left, overlaps])

        q.put([None, None, None, None, None, None])  # close queue

        t.join()  # wait for fuse thread to finish
