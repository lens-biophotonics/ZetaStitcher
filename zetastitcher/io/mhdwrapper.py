import os
import mmap

import numpy as np


class MHDWrapper(object):
    def __init__(self, file_name=None):
        self.file_name = file_name
        self.raw_file_name = None

        self.nfrms = -1
        self.xsize = -1
        self.ysize = -1
        self.dtype = None

        self.a = None
        self.mm = None
        self.file = None

        if file_name is not None:
            self.open()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, item):
        return self.a[item]

    @staticmethod
    def element_type_to_type_fmt(et):
        if not et.startswith('MET_'):
            raise ValueError('Invalid ElementType')

        _, eltype = et.split('MET_')

        type_fmt = None
        if eltype.endswith('CHAR'):
            byte_depth = 1
        elif eltype.endswith('SHORT'):
            byte_depth = 2
        elif eltype.endswith('INT'):
            byte_depth = 4
        elif eltype.endswith('LONG'):
            byte_depth = 8
        elif eltype == 'FLOAT':
            type_fmt = 'f'
            byte_depth = 4
        elif eltype == 'DOUBLE':
            type_fmt = 'f'
            byte_depth = 8
        else:
            raise ValueError('Invalid ElementType')

        if type_fmt is None:
            if eltype.startswith('U'):
                type_fmt = 'u'
            else:
                type_fmt = 'i'

        return '{}{}'.format(type_fmt, byte_depth)

    @property
    def nchannels(self):
        return 1

    @property
    def shape(self):
        if self.nchannels > 1:
            return self.nfrms, self.nchannels, self.ysize, self.xsize
        else:
            return self.nfrms, self.ysize, self.xsize

    def open(self, file_name=None):
        if file_name is not None:
            self.file_name = file_name

        d = {}

        with open(self.file_name, 'r') as f:
            try:
                for line in f.readlines():
                    k, v = map(str.strip, line.split('='))
                    d[k] = v
            except UnicodeDecodeError:
                raise ValueError('Invalid file type')

        if d['ObjectType'] != 'Image':
            raise ValueError('Invalid ObjectType')
        if d['BinaryData'] != 'True':
            raise ValueError('BinaryData must be True')

        type_fmt = self.element_type_to_type_fmt(d['ElementType'])

        byte_order = ''
        if 'BinaryDataByteOrderMSB' in d:
            if d['BinaryDataByteOrderMSB'] == 'True':
                byte_order = '>'
            elif d['BinaryDataByteOrderMSB'] == 'False':
                byte_order = '<'
            else:
                raise ValueError('Invalid BinaryDataByteOrderMSB')

        type_str = '{}{}'.format(byte_order, type_fmt)
        self.dtype = np.dtype(type_str)
        byte_depth = self.dtype.itemsize

        dims = d['DimSize'].split(' ')
        if len(dims) != int(d['NDims']) != 3:
            raise ValueError('Invalid number of dimensions')

        self.xsize, self.ysize, self.nfrms = map(int, dims)

        strides = [
            self.ysize * self.xsize * byte_depth,
            self.xsize * byte_depth,
            byte_depth
        ]

        rawfile = os.path.join(os.path.dirname(self.file_name), d['ElementDataFile'])
        self.file = open(rawfile, 'r')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)
        self.a = np.ndarray(self.shape, self.dtype, self.mm, 0, strides)

    def close(self):
        if self.mm is not None:
            self.mm.close()
        del self.mm
        self.mm = None
        if self.file is not None:
            self.file.close()

    def zslice(self, start_frame, end_frame=None, dtype=None, copy=True):
        a = self.__getitem__(slice(start_frame, end_frame))
        if dtype is not None:
            return a.astype(dtype)
        if copy:
            a = np.copy(a)
        return a
