from pathlib import Path

import numpy as np
import tifffile as tiff

from zetastitcher.io.inputfile_mixin import InputFileMixin


class TiffWrapper(InputFileMixin):
    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path

        self.tfile = None
        self.flist = None
        self.glob_mode = False
        self.axes = None

        if file_path is not None:
            self.file_path = Path(file_path)
            self.open()

    def open(self, file_path=None):
        if file_path is not None:
            self.file_path = Path(file_path)

        if self.file_path.is_dir():
            self.glob_mode = True
            flist = []
            flist += list(Path(self.file_path).glob('*.tif*'))
            flist += list(Path(self.file_path).glob('*.TIF*'))
            flist = sorted(flist)
            fname = flist[0]
            self.flist = flist
        else:
            self.glob_mode = False
            fname = self.file_path

        self.tfile = tiff.TiffFile(str(fname))
        setattr(self, 'close', getattr(self.tfile, 'close'))

        self.dtype = np.dtype(self.tfile.pages[0].dtype)
        self.xsize = self.tfile.pages[0].imagewidth
        self.ysize = self.tfile.pages[0].imagelength

        axes = self.tfile.series[0].axes

        if 'C' in axes:
            self.nchannels = self.tfile.series[0].shape[axes.index('C')]
        elif 'S' in axes:
            self.nchannels = self.tfile.series[0].shape[axes.index('S')]

        if 'Z' in axes:
            nfrms = self.tfile.series[0].shape[axes.index('Z')]
        elif not 'C' in axes:
            nfrms = len(self.tfile.pages)
        else:
            nfrms = 1
        if self.glob_mode:
            nfrms *= len(self.flist)

        self.nfrms = nfrms
        self.axes = axes

    def zslice(self, arg1, arg2=None, step=1, dtype=None, copy=True):
        myslice = self._args_to_slice(arg1, arg2, step)

        if self.glob_mode:
            flist = self.flist[myslice]
            if not flist:
                return np.array([])
            a = tiff.imread(list(map(str, flist)))
        else:
            mma = self.tfile.asarray(out='memmap')
            a = mma

            for letter in 'ZQ':
                if letter in self.axes:
                    # take only first element of left-most axes (e.g. time)
                    mma = mma[tuple(0 for _ in range(self.axes.index(letter)))]
                    if len(mma.shape) >= 3:
                        a = mma[myslice]
                    break

            if copy:
                a = np.copy(a)

        if 'C' in self.axes:
            a = np.moveaxis(a, self.axes.index('C') - len(self.axes), -1)

        if dtype is None:
            return a
        return a.astype(dtype)
