from pathlib import Path

import numpy as np

try:
    import dcimg
except ImportError:
    pass

from .ffmpeg_wrapper import FFMPEGWrapper
from .tiffwrapper import TiffWrapper
from tifffile import TiffFileError
from .zipwrapper import ZipWrapper
from .mhdwrapper import MHDWrapper

from zipfile import BadZipFile

from zetastitcher.io.inputfile_mixin import InputFileMixin


class InputFile(InputFileMixin):
    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path
        self.wrapper = None
        self._channel = -1
        self.squeeze = True

        self.nfrms = None

        if file_path is not None:
            self.file_path = Path(self.file_path)
            self.open()

    def __repr__(self):
        return '<InputFile file_path="{}" shape={} dtype={}>'.format(
            self.file_path, self.shape, self.dtype)

    def __getitem__(self, item):
        item = np.index_exp[item]  # ensure item is a tuple
        myitem = list(item)

        # ensure all items are slice objects
        # myitem = []
        if isinstance(item[0], int):
            start = item[0]
            stop = item[0] + 1
            step = None
        elif item[0] is Ellipsis:
            start = None
            stop = self.shape[0]
            step = None
        elif isinstance(item[0], slice):
            start = item[0].start
            stop = item[0].stop
            step = item[0].step
        else:
            raise TypeError("Invalid type: {}".format(type(item[0])))

        a = self.zslice(start, stop, step)

        if not a.size:
            return a

        if item[0] is not Ellipsis:
            myitem[0] = slice(None)

        a = a[tuple(myitem)]

        if self.squeeze:
            a = np.squeeze(a)
        return a

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
        s = [self.nfrms, self.nchannels, self.ysize, self.xsize]
        if self.nchannels == 1 or self.channel != -1:
            del s[1]
        return tuple(s)

    @property
    def file_size(self):
        return self.path.stat().st_size

    def open(self, file_path=None):
        if file_path is not None:
            self.file_path = Path(file_path)

        self.path = Path(self.file_path)

        self._open()
        self._setattrs()

    def _open(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        try:
            self.wrapper = TiffWrapper(self.file_path)
            return
        except (ValueError, IndexError, TiffFileError):
            pass

        try:
            self.wrapper = dcimg.DCIMGFile(self.file_path)
            return
        except (NameError, ValueError, IsADirectoryError):
            pass

        try:
            self.wrapper = ZipWrapper(self.file_path)
            return
        except (AttributeError, NameError, IsADirectoryError, BadZipFile):
            pass

        try:
            self.wrapper = MHDWrapper(self.file_path)
            return
        except (ValueError, IndexError, IsADirectoryError):
            pass

        try:
            self.wrapper = FFMPEGWrapper(self.file_path)
            return
        except (ValueError, FileNotFoundError):
            pass

        from .pims_wrapper import PimsWrapper
        try:
            self.wrapper = PimsWrapper(self.file_path)
            return
        except:
            pass

        raise ValueError('Unsupported file type')

    def _setattrs(self):
        l = ['nfrms', 'xsize', 'ysize', 'nchannels', 'dtype', 'close',
             '__getitem__']

        for a in l:
            try:
                setattr(self, a, getattr(self.wrapper, a))
            except AttributeError:
                pass

    def zslice(self, arg1, arg2=None, step=None, dtype=None, copy=True):
        """Return a slice, i.e. a substack of frames.

        Parameters
        ----------
        arg1 : int
            Mandatory argument, will be passed to `python:slice`
            If arg2 and step are both None, it will be passed as `slice(arg1)`,
            i.e. it would act as the stop argument.
        arg2 : int
            If not null, will be passed as `slice(arg1, arg2, step)`
        step : int
            If not null, will be passed as `slice(arg1, arg2, step)`
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
        myslice = slice(arg1, arg2, step)
        if arg2 is None and step is None:
            myslice = slice(arg1)

        if dtype is None:
            dtype = self.dtype

        ok = False
        try:
            ok = callable(getattr(self.wrapper, 'zslice'))
        except AttributeError:
            pass

        if ok:
            a = self.wrapper.zslice(myslice.start, myslice.stop, myslice.step, dtype, copy)
        else:
            myrange = list(self._args_to_range(arg1, arg2, step))
            s = list(self.shape)
            s[0] = len(myrange)
            a = np.zeros(s, dtype=dtype)
            z = 0
            for i in myrange:
                a[z] = self.wrapper.frame(i)
                z += 1

        if self.channel == -2:
            a = np.sum(a, axis=-1)
        elif self.channel != -1:
            a = a[..., self.channel]
        elif self.nchannels > 1 and a.ndim >= 3:
            a = np.moveaxis(a, -1, -3)

        if len(a.shape) < len(self.shape):
            a = np.expand_dims(a, axis=0)  # add dummy Z dimension

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
        return self.zslice(start_frame, end_frame, 1, dtype, copy)

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
