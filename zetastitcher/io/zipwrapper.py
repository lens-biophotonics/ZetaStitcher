import gc
import sys
import ctypes
import os.path
import zipfile
import threading
import multiprocessing as mp
from os import cpu_count

import imageio
import numpy as np


def get_typecodes():
    ct = ctypes
    simple_types = [
        ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
        ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
        ct.c_float, ct.c_double,
    ]

    return {np.dtype(ctype).str: ctype for ctype in simple_types}


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

    def zslice(self, start_frame, end_frame=None, dtype=None, copy=None):
        def worker(outarr):
            while True:
                got = input_q.get()
                if got is None:
                    return

                data, c, i = got

                a = imageio.imread(data)
                if dtype is not None:
                    a = a.astype(dtype)

                outarr[i] = a
                output_q.put((c, i))

        def process_output_queue():
            while True:
                got = output_q.get()
                if got is None:
                    break
                c, i = got
                out[c] = a[i]
                available_q.put(i)

        def populate_input_queue():
            for c in range(s[0]):
                i = available_q.get()
                data = self.zf.read(self.file_name_fmt.format(start_frame + c))
                input_q.put((data, c, i))
            for _ in range(n_of_procs):
                input_q.put(None)

        if dtype is None:
            dtype = self.dtype

        maxsize = cpu_count()
        available_q = mp.Queue(maxsize=maxsize)
        input_q = mp.Queue(maxsize=maxsize)
        output_q = mp.Queue()

        poq = threading.Thread(target=process_output_queue)
        poq.start()

        procs = []
        s = list(self.shape)
        s[0] = end_frame - start_frame

        out = np.zeros(s, dtype)

        temp_s = list(self.shape)
        temp_s[0] = maxsize

        tc = get_typecodes()
        typestr = tc[dtype.str]

        m_array = mp.Array(typestr, int(np.prod(temp_s)))
        a = np.frombuffer(m_array.get_obj(), dtype=typestr).reshape(temp_s)

        n_of_procs = min(maxsize, s[0])

        for p in range(n_of_procs):
            wp = mp.Process(target=worker, args=(a,))
            wp.start()
            procs.append(wp)

            available_q.put(p)

        piq = threading.Thread(target=populate_input_queue)
        piq.start()
        piq.join()

        for p in procs:
            p.join()
            del p

        output_q.put(None)
        poq.join()

        del a
        del m_array

        # force release of shared memory for Python < 3.8
        if sys.version_info < (3, 8):
            mp.heap.BufferWrapper._heap = mp.heap.Heap()
            gc.collect()

        return out
