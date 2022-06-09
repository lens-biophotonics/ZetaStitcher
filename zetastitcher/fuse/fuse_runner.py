import math
import logging
import os.path

from functools import lru_cache

import psutil
import numpy as np
import tifffile as tiff

from zetastitcher import VirtualFusedVolume


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FuseRunner(object):
    def __init__(self, file_matrix):
        self.fm = file_matrix  #: :class:`FileMatrix`
        self.path, _ = os.path.split(self.fm.input_path)

        self.zmin = 0
        self.zmax = None
        self.downsample_xy = None
        self.output_filename = None
        self.channel = None
        self.compression = 'zlib'

        self._is_multichannel = None

        self.vfv = VirtualFusedVolume(file_matrix)

    @property
    @lru_cache()
    def dtype(self):
        return self.vfv.dtype

    @property
    def debug(self):
        return self.vfv.overlay_debug_enabled

    @debug.setter
    def debug(self, value):
        self.vfv.overlay_debug_enabled = value

    @property
    def is_multichannel(self):
        if self.channel is not None:
            return False

        return self.vfv.nchannels > 1

    @property
    def output_shape(self):
        output_shape = list(self.vfv.shape)

        thickness = output_shape[0]
        if self.zmax is not None:
            thickness -= (thickness - self.zmax)
        thickness -= self.zmin

        output_shape[0] = thickness

        if self.downsample_xy:
            output_shape[-2] /= self.downsample_xy
            output_shape[-1] /= self.downsample_xy

        if self.vfv.nchannels > 1 and not self.is_multichannel:
            del output_shape[1]

        return tuple(map(math.ceil, output_shape))

    def run(self):
        total_byte_size = (np.prod(self.output_shape) * self.dtype.itemsize).item()
        bigtiff = total_byte_size > 2**31 - 1

        ram = psutil.virtual_memory().available

        # size in bytes of an xy plane (including channels) (float32)
        xy_size = (np.prod(self.output_shape[1::]) * 4).item()
        n_frames_in_ram = int(ram / xy_size / 2.5)

        n_loops = self.output_shape[0] // n_frames_in_ram

        partial_thickness = [n_frames_in_ram for _ in range(0, n_loops)]
        remainder = self.output_shape[0] % n_frames_in_ram
        if remainder:
            partial_thickness += [remainder]

        try:
            os.remove(self.output_filename)
        except FileNotFoundError:
            pass

        for thickness in partial_thickness:
            self.zmax = self.zmin + thickness
            ie = [
                slice(self.zmin, self.zmax),
                Ellipsis if self.channel is None else self.channel,
            ]
            if self.downsample_xy:
                ie += [slice(None, None, self.downsample_xy)] * 2

            fused = self.vfv[ie]

            if self.is_multichannel:
                fused = np.moveaxis(fused, -3, -1)

            logger.info('saving output to {}'.format(self.output_filename))
            tiff.imwrite(self.output_filename, fused, append=True, bigtiff=bigtiff, compression=self.compression)

            self.zmin += thickness
