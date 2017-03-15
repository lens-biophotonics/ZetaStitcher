import numpy as np

import dcimg

from . import tiffwrapper as tw


class InputFile(object):
    def __init__(self, file_name=None):
        self.file_name = file_name
        self.wrapper = None
        self._channel = -1
        self.channels = 1

        if file_name is not None:
            self.open()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if isinstance(self.wrapper, dcimg.DCIMGFile):
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
            (:attr:`nfrms`, :attr:`xsize`, :attr:`ysize`, `channels`) where
            `channels` is the number of color channels in the image. If
            :attr:`channel` is set or if there is only one channel,
            the `channels` dimension is squeezed.
        """
        s = (self.nfrms, self.xsize, self.ysize)
        if self.channels != 1 and self.channel == -1:
            s = s + (self.channels,)
        return s

    def open(self, file_name=None):
        if file_name is not None:
            self.file_name = file_name

        self._open()
        self._setattrs()

    def _open(self):
        try:
            self.wrapper = dcimg.DCIMGFile(self.file_name)
            return
        except ValueError:
            pass

        try:
            self.wrapper = tw.TiffWrapper(self.file_name)
            return
        except ValueError:
            pass

        raise ValueError('Unsupported file type')

    def _setattrs(self):
        l = ['nfrms', 'xsize', 'ysize', 'channels', 'close']

        for a in l:
            try:
                setattr(self, a, getattr(self.wrapper, a))
            except AttributeError:
                pass

    def layer(self, start_frame, end_frame, dtype=None):
        """Return a layer, i.e a stack of frames.

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
            shape of the array is (`end_frame` - `start_frame`, :attr:`ysize`,
            :attr:`xsize`, :attr:`channels`) where `channels` is the number of
            color channels in the image. If :attr:`channel` is set or if
            there is only one channel, the `channels` dimension is squeezed.
        """
        l = self.wrapper.layer(start_frame, end_frame, dtype)
        if self.channel != -1:
            l = l[..., self.channel]
        return l

    def layer_idx(self, index, frames_per_layer=1, dtype=None):
        """Return a layer, i.e a stack of frames, by index.

        Parameters
        ----------
        index : int
            layer index
        frames_per_layer : int
            number of frames per layer
        dtype

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array, see :func:`layer`.
        """
        start_frame = index * frames_per_layer
        end_frame = start_frame + frames_per_layer
        return self.layer(start_frame, end_frame, dtype)

    def whole(self, dtype=None):
        """Convenience function to retrieve the whole stack.

        Equivalent to call :func:`layer_idx` with `index` = 0 and
        `frames_per_layer` = :attr:`nfrms`

        Parameters
        ----------
        dtype

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array, see :func:`layer`.
        """
        return self.layer_idx(0, self.nfrms, dtype)

    def frame(self, index, dtype=None):
        """Convenience function to retrieve a single layer.

        Same as calling :func:`layer` and squeezing.

        Parameters
        ----------
        index : int
            layer index
        dtype

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array, see :func:`layer`.
        """
        return np.squeeze(self.layer_idx(index), dtype)
