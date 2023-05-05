"""
This module provides `ZipWrapper`, a class for accessing a stack of images
stored in a .zip file one frame per file. Files inside the .zip archive must
will be accessed in alphabetical order.

For faster access when using compressed formats (such as JPEG2000), a cache can
optionally be anabled using `set_cache()`.
"""

import os
import gc
import sys
import ctypes
import zipfile
from pathlib import Path
from cachetools import LRUCache
from functools import cached_property


from imagecodecs import imread

import numpy as np

from zetastitcher.io.inputfile_mixin import InputFileMixin

# disable cache
_cache = LRUCache(maxsize=0)
_cache.hits = 0
_cache.misses = 0

try:
    numthreads = int(os.environ['OMP_NUM_THREADS'])
except:
    numthreads = None


def set_cache(cache):
    """
    Set a cache for `ZipWrapper.zslice`.

    Example:

    .. code-block:: python

       import zetastitcher.io.zipwrapper as zw
       from cachetools import LRUCache
       zw.set_cache(LRUCache(maxsize=32))

    Parameters
    ----------
    cache : `cachetools:cachetools.Cache`
        The cache instance to use.
    """
    global _cache
    _cache = cache
    _cache.hits = 0
    _cache.misses = 0


def get_typecodes():
    ct = ctypes
    simple_types = [
        ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
        ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
        ct.c_float, ct.c_double,
    ]

    return {np.dtype(ctype).str: ctype for ctype in simple_types}


def imread_wrapper(fname, internal_fname, dtype=None):
    cache_key = f'{fname}__{internal_fname}__{dtype}'
    cached = _cache.get(cache_key)
    if cached is not None:
        _cache.hits += 1
        return cached
    _cache.misses += 1
    zf = zipfile.ZipFile(str(fname), mode='r')
    a = imread(zf.read(internal_fname), numthreads=numthreads)
    if dtype is not None:
        a = a.astype(dtype)
    if _cache.maxsize > 0:
        _cache[cache_key] = a
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

        self.nfrms = len(names)
        # self.dtype = im.dtype
        #
        # if len(im.shape) > 2:
        #     self.nchannels = im.shape[0]

        self.names = names

    @cached_property
    def xsize(self):
        im = imread_wrapper(self.file_path, self.names[0])
        return im.shape[-1]

    @ cached_property
    def ysize(self):
        im = imread_wrapper(self.file_path, self.names[0])
        return im.shape[-2]

    @ cached_property
    def dtype(self):
        im = imread_wrapper(self.file_path, self.names[0])
        return im.dtype

    @ cached_property
    def nchannels(self):
        im = imread_wrapper(self.file_path, self.names[0])
        if len(im.shape) > 2:
            return im.shape[0]
        return 1

    def frame(self, index, dtype=None, copy=None):
        a = imread(self.zf.read(self.names[index]), numthreads=numthreads)

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

        for i, z in zip(range(s[0]), zlist):
            out[i] = imread_wrapper(self.file_path, self.names[z], dtype)

        return out
