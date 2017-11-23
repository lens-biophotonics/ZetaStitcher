import json
import subprocess as sp

import numpy as np


class FFMPEGWrapper(object):
    def __init__(self, file_name=None):
        self.file_name = file_name

        self.proc = None

        self._probed_dict = None

        if file_name is not None:
            self.open()

    @property
    def nfrms(self):
        return int(self._probed_dict['streams'][0]['nb_frames'])

    @property
    def xsize(self):
        return self._probed_dict['streams'][0]['width']

    @property
    def ysize(self):
        return self._probed_dict['streams'][0]['height']

    @property
    def pix_fmt(self):
        return self._probed_dict['streams'][0]['pix_fmt']

    @property
    def nchannels(self):
        pix_fmt = self.pix_fmt
        if 'gray' in pix_fmt or 'mono' in pix_fmt or 'pal8' in pix_fmt:
            return 1
        elif 'yuva' in pix_fmt:
            return 4
        elif 'yuv' in pix_fmt or 'rgb' in pix_fmt or 'gbr' in pix_fmt or \
                'bgr' in pix_fmt:
            return 3

    @property
    def dtype(self):
        return np.uint8

    @property
    def shape(self):
        s = (self.nfrms, self.ysize, self.xsize)
        if self.nchannels > 1:
            s = s + (self.nchannels,)
        return s

    def open(self, file_name=None):
        if file_name is not None:
            self.file_name = file_name

        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-i', self.file_name,
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


    def zslice(self, start_frame, end_frame=None, dtype=None, copy=True):
        if end_frame is None:
            end_frame = start_frame + 1

        command = [
            'ffmpeg',
            '-i', self.file_name,
            '-f', 'image2pipe',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'gray' if 'gray' in self.pix_fmt else 'rgb24',
            '-'
        ]
        pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8)

        dt = np.dtype(self.dtype)

        i = 0
        while i < start_frame:
            pipe.stdout.read(self.ysize * self.xsize * dt.itemsize)
            i += 1

        shape = list(self.shape)
        shape[0] = end_frame - start_frame

        a = np.zeros(shape, dtype=self.dtype)

        shape[0] = 1

        while i < end_frame:
            raw = pipe.stdout.read(np.prod(shape) * dt.itemsize)
            a[i - start_frame] = \
                np.fromstring(raw, dtype=self.dtype).reshape(shape)
            i += 1

        pipe.communicate()

        if dtype is None:
            return a
        return a.astype(dtype)
