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
        self.names = None

        if self.file_path is not None:
            self.file_path = Path(self.file_path)
            self.open()

    def open(self, file_path=None):
        if file_path is not None:
            self.file_path = Path(file_path)

        self.zf = zipfile.ZipFile(str(self.file_path), mode='r')
        setattr(self, 'close', getattr(self.zf, 'close'))
        names = self.zf.namelist()
        names.sort()

        im = imread_wrapper(self.file_path, names[0])

        self.xsize = im.shape[-1]
        self.ysize = im.shape[-2]
        self.nfrms = len(names)
        self.dtype = im.dtype

        if len(im.shape) > 2:
            self.nchannels = im.shape[0]

        self.names = names

    def frame(self, index, dtype=None, copy=None):
        a = imageio.imread(self.zf.read(self.names[index]))

        if dtype is not None:
            a = a.astype(dtype)
        return a

    def zslice(self, arg1, arg2=None, step=None, dtype=None, copy=None):
        if dtype is None:
            dtype = self.dtype

        s = list(self.shape)
        zlist = list(self._args_to_range(arg1, arg2, step))
        s[0] = len(zlist)

        out = np.zeros(s, dtype)

        # NEW ##################################

        my_futures = []

        for z in zlist:
            fut = work(self.file_path, self.names[z], dtype)
            my_futures.append(fut)

        for z, fut in zip(range(s[0]), my_futures):
            out[z] = fut.result(None)

        # force release of shared memory for Python < 3.8
        if sys.version_info < (3, 8):
            mph.BufferWrapper._heap = mph.Heap()
            gc.collect()

        return out
