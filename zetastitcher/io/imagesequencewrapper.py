"""
This module provides `ImageSequenceWrapper`, a class for accessing a stack of images
stored in a directory one frame per file.

For faster access when using compressed formats (such as JPEG2000), a cache can
optionally be anabled using `set_cache()`.
"""

import gc
import sys
import ctypes
import zipfile
import concurrent.futures
from pathlib import Path
from cachetools import LRUCache
from typing import Union

import imageio
import numpy as np

from zetastitcher.io.inputfile_mixin import InputFileMixin

# disable cache
_cache = LRUCache(maxsize=0)
_cache.hits = 0
_cache.misses = 0


def set_cache(cache):
    """
    Set a cache for `ImageSequenceWrapper.zslice`.

    Example:

    .. code-block:: python

       import zetastitcher.io.imagesequencewrapper as isw
       from cachetools import LRUCache
       isw.set_cache(LRUCache(maxsize=32))

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


def imread_wrapper(frame_fpath: Union[Path, str] = None, dtype=None):
    a = imageio.imread(str(frame_fpath))
    if dtype is not None:
        a = a.astype(dtype)
    return a


class ImageSequenceWrapper(InputFileMixin):
    def __init__(self, file_path: Union[Path, str]=None):
        super().__init__()
        self.file_path = file_path
        self.names = None
        self.frame_fpaths = None

        if self.file_path is not None:
            self.file_path = Path(self.file_path)
            self.open()

    def open(self, file_path: Union[Path, str] = None):
        if file_path is not None:
            self.file_path = Path(file_path)
        else:
            raise ValueError('file_path must be specified')
        
        self.frame_fpaths = [fpath for fpath in file_path.glob("*.*")]
        self.frame_fpaths.sort(key=lambda fpath: int(fpath.name.split(".")[0]))

        self.names = [str(fpath.name) for fpath in self.frame_fpaths]
        self.extensions = [fpath.suffix for fpath in self.frame_fpaths]

        if len(set(self.extensions)) > 1:
            raise ValueError("All files in an ImageSequence must have the same extension")

        im = imread_wrapper(self.frame_fpaths[0])
        self.nfrms = len(self.frame_fpaths)
        self.xsize = im.shape[-1]
        self.ysize = im.shape[-2]
        self.dtype = im.dtype

        if len(im.shape) > 2:
            self.nchannels = im.shape[0]

    def frame(self, index, dtype=None, copy=None):
        a = imageio.imread(self.frame_fpaths[index])

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

        e = concurrent.futures.ProcessPoolExecutor()

        for i, z in zip(range(s[0]), zlist):
            cache_key = f'{self.file_path}__{str(self.frame_fpaths[z].name)}'
            cached = _cache.get(cache_key)

            if cached is not None:
                _cache.hits += 1
                out[i] = cached
            else:
                _cache.misses += 1
                fut = e.submit(imread_wrapper, self.frame_fpaths[z], dtype)
                my_futures.append((i, cache_key, fut))

        for z, key, fut in my_futures:
            if _cache.maxsize > 0:
                _cache[key] = fut.result(None)
            out[z] = fut.result(None)

        # force release of shared memory for Python < 3.8
        if sys.version_info < (3, 8):
            import multiprocessing.heap as mph
            mph.BufferWrapper._heap = mph.Heap()
            gc.collect()

        return out
