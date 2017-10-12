import glob
import os.path

import numpy as np
import skimage.external.tifffile as tiff


class TiffWrapper(object):
    def __init__(self, file_name=None):
        self.file_name = file_name

        self.tfile = None
        self.flist = None
        self.glob_mode = False

        if file_name is not None:
            self.open()

    @property
    def nfrms(self):
        nfrms = len(self.tfile.pages)
        if self.glob_mode:
            nfrms *= len(self.flist)
        return nfrms

    @property
    def xsize(self):
        return self.tfile.pages[0].image_width

    @property
    def ysize(self):
        return self.tfile.pages[0].image_length

    @property
    def axes(self):
        return self.tfile.pages[0].axes

    @property
    def nchannels(self):
        if self.axes == 'YXS':
            return self.tfile.pages[0]._shape[-1]
        elif self.axes == 'SYX':
            return self.tfile.pages[0]._shape[1]

    @property
    def dtype(self):
        return np.dtype(self.tfile.pages[0].dtype)

    @property
    def shape(self):
        s = self.tfile.pages[0]._shape[-3:]
        return (self.nfrms,) + tuple(s[i] for i in [-1, -3, -2])

    def open(self, file_name=None):
        if file_name is not None:
            self.file_name = file_name

        if os.path.isdir(self.file_name):
            self.glob_mode = True
            flist = []
            flist += glob.glob(os.path.join(self.file_name, '*.tif*'))
            flist += glob.glob(os.path.join(self.file_name, '*.TIF*'))
            flist = sorted(flist)
            fname = flist[0]
            self.flist = flist
        else:
            self.glob_mode = False
            fname = self.file_name

        self.tfile = tiff.TiffFile(fname)
        setattr(self, 'close', getattr(self.tfile, 'close'))

    def zslice(self, start_frame, end_frame=None, dtype=None, copy=True):
        if end_frame is None:
            end_frame = start_frame + 1

        if not self.glob_mode:
            a = self.tfile.asarray(slice(start_frame, end_frame))
        else:
            frames_per_file = self.nfrms // len(self.flist)
            start_file = start_frame // frames_per_file
            end_file = end_frame // frames_per_file
            a = tiff.imread(self.flist[start_file:end_file], pattern='')

        if end_frame - start_frame == 1:
            a = np.expand_dims(a, axis=0)

        if self.axes == 'SYX':
            a = np.moveaxis(a, 1, -1)

        if dtype is None:
            return a
        return a.astype(dtype)

    def __getitem__(self, item):
        item = np.index_exp[tuple(item)]  # ensure item is a tuple

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

            myitem.append(slice(start, stop, step))

        for _ in range(0, len(self.shape) - len(myitem)):
            myitem.append(slice(0, self.shape[len(myitem)], 1))

        a = self.zslice(myitem[0].start, myitem[0].stop)

        a = np.rollaxis(a, -1, -3)

        myitem[0] = slice(0, None, myitem[0].step)

        a = a[myitem]
        return np.squeeze(a)
