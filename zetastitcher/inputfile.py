import os.path

import numpy as np

try:
    import dcimg
except ImportError:
    pass

from .ffmpeg_wrapper import FFMPEGWrapper
from .tiffwrapper import TiffWrapper


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
        item = np.index_exp[item]  # ensure item is a tuple

        # ensure all items are slice objects
        myitem = []
        for i in item:
            if isinstance(i, int):
                start = i
                stop = i + 1
                step = 1
            elif i is Ellipsis:
                for _ in range(0, len(self.shape) - len(item) + 1):
                    myitem.append(slice(0, self.shape[len(myitem)], 1))
                continue
            elif isinstance(i, slice):
                start = i.start
                stop = i.stop
                step = i.step if i.step is not None else 1
            else:
                raise TypeError("Invalid type: {}".format(type(i)))

            curr_max = self.shape[len(myitem)]
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
                start += abs(step)
                stop += abs(step)

            myitem.append(slice(start, stop, step))

        for _ in range(0, len(self.shape) - len(myitem)):
            myitem.append(slice(0, self.shape[len(myitem)], 1))

        try:
            return self.wrapper.__getitem__(myitem)
        except AttributeError:
            pass

        a = self.zslice(myitem[0].start, myitem[0].stop)

        myitem[0] = slice(0, None, myitem[0].step)
        a = a[myitem]

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
            (:attr:`nfrms`, `channels`, :attr:`xsize`, :attr:`ysize`) where
            `channels` is the number of color channels in the image. If
            :attr:`channel` is set or if there is only one channel,
            the `channels` dimension is squeezed.
        """
        s = [self.nfrms, self.xsize, self.ysize]
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
        try:
            self.wrapper = TiffWrapper(self.file_name)
            return
        except (FileNotFoundError, ValueError, IsADirectoryError):
            pass

        try:
            self.wrapper = dcimg.DCIMGFile(self.file_name)
            self.wrapper.retrieve_first_4_pixels = False
            return
        except (NameError, FileNotFoundError, ValueError, IsADirectoryError):
            pass

        try:
            self.wrapper = FFMPEGWrapper(self.file_name)
            return
        except ValueError:
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
        l = self.wrapper.zslice(start_frame, end_frame, dtype, copy)
        if self.channel == -2:
            l = np.sum(l, axis=-1)
        elif self.channel != -1:
            l = l[..., self.channel]
        elif self.nchannels > 1:
            l = np.moveaxis(l, -1, -3)

        return l

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
        return np.squeeze(self.zslice_idx(index, dtype=dtype, copy=copy))
