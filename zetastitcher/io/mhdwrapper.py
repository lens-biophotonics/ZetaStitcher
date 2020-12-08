import mmap
from pathlib import Path

import numpy as np

from zetastitcher.io.inputfile_mixin import InputFileMixin


class MHDWrapper(InputFileMixin):
    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path
        self.raw_file_name = None

        self.a = None
        self.mm = None
        self.file = None
        self.nchannels = 1

        if file_path is not None:
            self.file_path = Path(self.file_path)
            self.open()

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

    def open(self, file_name=None):
        if file_name is not None:
            self.file_path = file_name

        d = {}

        with self.file_path.open('r') as f:
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

        rawfile = self.file_path.parent.joinpath(d['ElementDataFile'])
        self.file = rawfile.open('r')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)
        self.a = np.ndarray(self.shape, self.dtype, self.mm, 0, strides)
        setattr(self, '__getitem__', getattr(self.a, '__getitem__'))

    def close(self):
        if self.mm is not None:
            self.mm.close()
        del self.mm
        self.mm = None
        if self.file is not None:
            self.file.close()

    def zslice(self, arg1, arg2=None, step=None, dtype=None, copy=True):
        a = self.a[self._args_to_slice(arg1, arg2, step)]
        if dtype is not None:
            return a.astype(dtype)
        if copy:
            a = np.copy(a)
        return a
