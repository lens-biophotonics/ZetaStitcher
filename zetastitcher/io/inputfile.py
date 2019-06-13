import os.path

import numpy as np

try:
    import dcimg
except ImportError:
    pass

from .ffmpeg_wrapper import FFMPEGWrapper
from .tiffwrapper import TiffWrapper
from .zipwrapper import ZipWrapper
from .mhdwrapper import MHDWrapper

from zipfile import BadZipFile


class InputFile(object):
    def __init__(self, file_name=None):
        self.file_name = file_name
        self.wrapper = None
        self._channel = -1
        self.nchannels = 1

        self.nfrms = None

        if file_name is not None:
            self.open()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, item):
        try:
            return self.wrapper.__getitem__(item)
        except AttributeError:
            pass

        item = np.index_exp[item]  # ensure item is a tuple
        myitem = list(item)

        # ensure all items are slice objects
        # myitem = []
        if isinstance(item[0], int):
            start = item[0]
            stop = item[0] + 1
            step = 1
        elif item[0] is Ellipsis:
            start = 0
            stop = self.shape[0]
            step = 1
        elif isinstance(item[0], slice):
            start = item[0].start
            stop = item[0].stop
            step = item[0].step if item[0].step is not None else 1

            curr_max = self.shape[0]
            if start is None:
                start = 0 if step > 0 else curr_max
            elif start < 0:
                start += curr_max
            elif start > curr_max:
                start = curr_max

            if stop is None:
                stop = curr_max if step > 0 else 0
            elif stop < 0:
                stop += curr_max
            elif stop > curr_max:
                stop = curr_max

            if step < 0:
                temp = start
                start = stop
                stop = temp
                if start > 0:
                    start += abs(step) - (start - stop) % abs(step)
                stop += 1
        else:
            raise TypeError("Invalid type: {}".format(type(item[0])))

        a = self.zslice(start, stop)
        if step < 0:
            a = a[::-1]
        a = a[::abs(step)]

        if item[0] is not Ellipsis:
            myitem[0] = slice(None, None, 1)
        a = a[tuple(myitem)]

        return np.squeeze(a)

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if not self.nchannels > 1:
            return
        self._channel = value

    @property
    def file(self):
        return self.wrapper

    @file.setter
    def file(self, value):
        self.wrapper = value
        self._setattrs()

    @property
    def shape(self):
        """Shape of the whole image stack.

        Returns
        -------
        tuple
            (:attr:`nfrms`, `channels`, :attr:`ysize`, :attr:`xsize`) where
            `channels` is the number of color channels in the image. If
            :attr:`channel` is set or if there is only one channel,
            the `channels` dimension is squeezed.
        """
        s = [self.nfrms, self.ysize, self.xsize]
        if self.nchannels != 1 and self.channel == -1:
            s.insert(1, self.nchannels)
        return tuple(s)

    @property
    def file_size(self):
        return os.path.getsize(self.file_name)

    def open(self, file_name=None):
        if file_name is not None:
            self.file_name = file_name

        self._open()
        self._setattrs()

    def _open(self):
        if not os.path.exists(self.file_name):
            raise FileNotFoundError(self.file_name)

        try:
            self.wrapper = TiffWrapper(self.file_name)
            return
        except ValueError:
            pass

        try:
            self.wrapper = dcimg.DCIMGFile(self.file_name)
            return
        except (NameError, ValueError):
            pass

        try:
            self.wrapper = ZipWrapper(self.file_name)
            return
        except (NameError, BadZipFile):
            pass

        try:
            self.wrapper = MHDWrapper(self.file_name)
            return
        except (ValueError, IndexError):
            pass

        try:
            self.wrapper = FFMPEGWrapper(self.file_name)
            return
        except (ValueError, FileNotFoundError):
            pass

        raise ValueError('Unsupported file type')

    def close(self):
        try:
            self.wrapper.close()
        except AttributeError:
            pass

    def _setattrs(self):
        l = ['nfrms', 'xsize', 'ysize', 'nchannels', 'dtype']

        for a in l:
            try:
                setattr(self, a, getattr(self.wrapper, a))
            except AttributeError:
                pass

    def zslice(self, start_frame, end_frame=None, dtype=None, copy=True):
        """Return a slice, i.e. a substack of frames.

        Parameters
        ----------
        start_frame : int
            first frame to select
        end_frame : int
            last frame to select (noninclusive). If None, defaults to
            :code:`start_frame + 1`
        dtype

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array of the original type or of `dtype`, if specified. The
            shape of the array is (`end_frame` - `start_frame`,
            :attr:`channels` :attr:`ysize`, :attr:`xsize`, :attr:`channels`)
            where `channels` is the number of color channels in the image. If
            :attr:`channel` is set or if there is only one channel, the
            `channels` dimension is squeezed.
        """
        ok = callable(getattr(self.wrapper, 'zslice'))
        if ok:
            a = self.wrapper.zslice(start_frame, end_frame, dtype, copy)
        else:
            s = list(self.shape)
            s[0] = end_frame - start_frame
            a = np.zeros(s, dtype=self.dtype)
            for i in range(start_frame, end_frame):
                a[i - start_frame] = self.wrapper.frame(i)

        if len(a.shape) < len(self.shape):
            a = np.expand_dims(a, axis=0)

        if self.channel == -2:
            a = np.sum(a, axis=-1)
        elif self.channel != -1:
            a = a[..., self.channel]
        elif self.nchannels > 1:
            a = np.moveaxis(a, -1, -3)

        return a

    def zslice_idx(self, index, frames_per_slice=1, dtype=None, copy=True):
        """Return a slice, i.e. a substack of frames, by index.

        Parameters
        ----------
        index : int
            slice index
        frames_per_slice : int
            number of frames per slice
        dtype

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array, see :func:`slice`.
        """
        start_frame = index * frames_per_slice
        end_frame = start_frame + frames_per_slice
        return self.zslice(start_frame, end_frame, dtype, copy)

    def whole(self, dtype=None, copy=True):
        """Convenience function to retrieve the whole stack.

        Equivalent to call :func:`slice_idx` with `index` = 0 and
        `frames_per_slice` = :attr:`nfrms`

        Parameters
        ----------
        dtype

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array, see :func:`slice`.
        """
        return self.zslice_idx(0, self.nfrms, dtype, copy)

    def frame(self, index, dtype=None, copy=True):
        """Convenience function to retrieve a single slice.

        Same as calling :func:`slice` and squeezing.

        Parameters
        ----------
        index : int
            slice index
        dtype

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array, see :func:`slice`.
        """
        try:
            a = self.wrapper.frame(index, dtype=dtype, copy=copy)
        except AttributeError:
            a = self.zslice_idx(index, dtype=dtype, copy=copy)
        return np.squeeze(a)
