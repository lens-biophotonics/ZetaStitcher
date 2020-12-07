import gc
import sys
import ctypes
import zipfile
import functools
import concurrent.futures
import multiprocessing.heap as mph
from pathlib import Path

import imageio
import numpy as np

from zetastitcher.io.inputfile_mixin import InputFileMixin


def get_typecodes():
    ct = ctypes
    simple_types = [
        ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
        ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
        ct.c_float, ct.c_double,
    ]

    return {np.dtype(ctype).str: ctype for ctype in simple_types}


e = concurrent.futures.ProcessPoolExecutor()


@functools.lru_cache(2000)
def work(fname, internal_fname, dtype=None):
    return e.submit(imread_wrapper, fname, internal_fname, dtype)


def imread_wrapper(fname, internal_fname, dtype=None):
    zf = zipfile.ZipFile(str(fname), mode='r')
    a = imageio.imread(zf.read(internal_fname))
    if dtype is not None:
        a = a.astype(dtype)
    return a


class ZipWrapper(InputFileMixin):
    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path

        self.zf = None
        self.file_name_fmt = ''

        if self.file_path is not None:
            self.file_path = Path(self.file_path)
            self.open()

    def open(self, file_path=None):
        if file_path is not None:
            self.file_path = Path(file_path)

        self.zf = zipfile.ZipFile(str(self.file_path), mode='r')
        setattr(self, 'close', getattr(self.zf, 'close'))
        names = self.zf.namelist()

        im = imread_wrapper(self.file_path, names[0])

        self.xsize = im.shape[-1]
        self.ysize = im.shape[-2]
        self.nfrms = len(names)
        self.dtype = im.dtype

        if len(im.shape) > 2:
            self.nchannels = im.shape[0]
        fname, ext = Path(names[0]).stem, Path(names[0]).suffix
        self.file_name_fmt = '{:0' + str(len(fname)) + '}' + ext

    def frame(self, index, dtype=None, copy=None):
        a = imageio.imread(self.zf.read(self.file_name_fmt.format(index)))

        if dtype is not None:
            a = a.astype(dtype)
        return a

    def zslice(self, start_frame, end_frame=None, dtype=None, copy=None):
        if dtype is None:
            dtype = self.dtype

        s = list(self.shape)
        s[0] = end_frame - start_frame

        out = np.zeros(s, dtype)

        # NEW ##################################

        my_futures = []

        for c in range(s[0]):
            internal_fname = self.file_name_fmt.format(start_frame + c)
            fut = work(self.file_path, internal_fname, dtype)
            my_futures.append(fut)

        for c, fut in zip(range(s[0]), my_futures):
            out[c] = fut.result(None)

        print(work.cache_info())

        # force release of shared memory for Python < 3.8
        if sys.version_info < (3, 8):
            mph.BufferWrapper._heap = mph.Heap()
            gc.collect()

        return out
