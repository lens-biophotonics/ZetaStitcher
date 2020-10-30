from pathlib import Path

import numpy as np
import tifffile as tiff


class TiffWrapper(object):
    def __init__(self, file_path=None):
        self.file_path = file_path

        self.tfile = None
        self.flist = None
        self.glob_mode = False

        if file_path is not None:
            self.file_path = Path(file_path)
            self.open()

    @property
    def nfrms(self):
        if self.axes.startswith('IYX'):
            nfrms = self.tfile.pages[0]._shape[0]
        else:
            nfrms = len(self.tfile.pages)
        if self.glob_mode:
            nfrms *= len(self.flist)
        return nfrms

    @property
    def xsize(self):
        return self.tfile.pages[0].imagewidth

    @property
    def ysize(self):
        return self.tfile.pages[0].imagelength

    @property
    def axes(self):
        return self.tfile.pages[0].axes

    @property
    def nchannels(self):
        if self.axes.startswith('YX') or self.axes.startswith('IYX'):
            return self.tfile.pages[0].shaped[-1]
        elif self.axes == 'SYX':
            return self.tfile.pages[0].shaped[1]
        return 1

    @property
    def dtype(self):
        return np.dtype(self.tfile.pages[0].dtype)

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

    def zslice(self, start_frame, end_frame=None, dtype=None, copy=True):
        if end_frame is None:
            end_frame = start_frame + 1

        if not self.glob_mode:
            if len(self.tfile.pages) == 1 and self.nfrms > 1:
                a = self.tfile.asarray(0, out='memmap')
                a = a[slice(start_frame, end_frame)]
            else:
                a = self.tfile.asarray(slice(start_frame, end_frame),
                                       out=None if copy else 'memmap')
        else:
            frames_per_file = self.nfrms // len(self.flist)
            start_file = start_frame // frames_per_file
            end_file = end_frame // frames_per_file
            a = tiff.imread(list(map(str, self.flist[start_file:end_file])),
                            pattern='')

        if self.axes == 'SYX':
            a = np.moveaxis(a, 1, -1)

        if dtype is None:
            return a
        return a.astype(dtype)
