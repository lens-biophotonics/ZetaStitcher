import json
import mimetypes
import subprocess as sp
from pathlib import Path

import numpy as np

from zetastitcher.io.inputfile_mixin import InputFileMixin


class FFMPEGWrapper(InputFileMixin):
    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path

        self.proc = None
        self.pix_fmt = None

        self._probed_dict = None

        if file_path is not None:
            self.file_path = Path(self.file_path)
            self.open()

    @property
    def shape(self):
        s = (self.nfrms, self.ysize, self.xsize)
        if self.nchannels > 1:
            s = s + (self.nchannels,)
        return s

    def open(self, file_path=None):
        if file_path is not None:
            self.file_path = Path(file_path)

        mtype = mimetypes.guess_type(self.file_path)[0]
        if mtype is None or not mtype.startswith('video'):
            raise ValueError(f'Invalid MIME type:{mtype}')

        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-i', str(self.file_path),
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            '-select_streams', 'v:0'
        ]

        pipe = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        self._probed_dict = json.loads(pipe.stdout.read().decode('utf-8'))
        pipe.communicate()

        if len(self._probed_dict) == 0 \
            or 'mp4' not in self._probed_dict['format']['format_name']:
                raise ValueError('Unrecognized format for FFMPEGWrapper')

        self.nfrms = int(self._probed_dict['streams'][0]['nb_frames'])
        self.xsize = self._probed_dict['streams'][0]['width']
        self.ysize = self._probed_dict['streams'][0]['height']
        pix_fmt = self._probed_dict['streams'][0]['pix_fmt']

        if 'gray' in pix_fmt or 'mono' in pix_fmt or 'pal8' in pix_fmt:
            self.nchannels = 1
        elif 'yuva' in pix_fmt:
            self.nchannels = 4
        elif 'yuv' in pix_fmt or 'rgb' in pix_fmt or 'gbr' in pix_fmt or \
                'bgr' in pix_fmt:
            self.nchannels = 3
        self.pix_fmt = pix_fmt
        self.dtype = np.dtype(np.uint8)

    def zslice(self, arg1, arg2=None, step=None, dtype=None, copy=True):
        myslice = self._args_to_slice(arg1, arg2, step)
        norm_slice = self._normalize_slice(myslice)

        if norm_slice.start >= norm_slice.stop:
            return np.array([])

        command = [
            'ffmpeg',
            '-i', str(self.file_path),
            '-f', 'image2pipe',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'gray' if 'gray' in self.pix_fmt else 'rgb24',
            '-'
        ]
        pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8)

        i = 0
        frame_shape = (1, self.ysize, self.xsize, self.nchannels)
        frame_bytes = np.prod(frame_shape) * self.dtype.itemsize

        while i < norm_slice.start:  # skip initial frames
            pipe.stdout.read(frame_bytes)
            i += 1

        shape = list(frame_shape)
        shape[0] = len(self._slice_to_range(norm_slice))

        a = np.zeros(shape, dtype=self.dtype)

        z = 0
        while i < norm_slice.stop:
            raw = pipe.stdout.read(frame_bytes)
            a[z] = np.frombuffer(raw, dtype=self.dtype).reshape(frame_shape)[0]
            for _ in range(norm_slice.step - 1):  # skip frames
                pipe.stdout.read(frame_bytes)
            i += norm_slice.step
            z += 1

        pipe.communicate()

        if myslice.step and myslice.step < 0:
            a = a[::-1]

        a = np.squeeze(a)

        if dtype is None:
            return a
        return a.astype(dtype)
