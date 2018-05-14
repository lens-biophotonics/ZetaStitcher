import os.path
import zipfile

import imageio


class ZipWrapper(object):
    def __init__(self, file_name=None):
        self.file_name = file_name

        self.zf = None
        self.file_name_fmt = ''
        self.xsize = None
        self.ysize = None
        self.nfrms = None
        self.dtype = None
        self.nchannels = 1

        self.open()

    @property
    def shape(self):
        if self.nchannels > 1:
            return self.nfrms, self.nchannels, self.ysize, self.xsize
        else:
            return self.nfrms, self.ysize, self.xsize

    def open(self, file_name=None):
        if file_name is not None:
            self.file_name = file_name

        self.zf = zipfile.ZipFile(self.file_name, mode='r')
        names = self.zf.namelist()

        im = imageio.imread(self.zf.read(names[0]))

        self.xsize = im.shape[-1]
        self.ysize = im.shape[-2]
        self.nfrms = len(names)
        self.dtype = im.dtype
        if len(im.shape) > 2:
            self.nchannels = im.shape[0]
        fname, ext = os.path.splitext(names[0])
        self.file_name_fmt = '{:0' + str(len(fname)) + '}' + ext

    def frame(self, index, dtype=None, copy=None):
        a = imageio.imread(self.zf.read(self.file_name_fmt.format(index)))

        if dtype is not None:
            a = a.astype(dtype)
        return a
