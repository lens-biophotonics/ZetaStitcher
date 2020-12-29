import numpy as np


class InputFileMixin:
    def __init__(self):
        self.path = None
        self.xsize = None
        self.ysize = None
        self.nfrms = None
        self.dtype = None
        self.nchannels = 1

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass

    @property
    def shape(self):
        """Shape of the whole image stack.

        Returns
        -------
        tuple
            (:attr:`nfrms`, :attr:`ysize`, :attr:`xsize`, `channels`) where
            `channels` is the number of color channels in the image. If
            :attr:`channel` is set or if there is only one channel,
            the `channels` dimension is squeezed.
        """
        s = [self.nfrms, self.ysize, self.xsize, self.nchannels]
        if self.nchannels == 1:
            s = s[:-1]
        return tuple(s)

    @property
    def file_size(self):
        return self.path.stat().st_size

    @property
    def array_size(self):
        """Size in bytes of the whole virtual array"""
        return np.prod(self.shape).item() * self.dtype.itemsize

    @property
    def frame_size(self):
        """Size in bytes of a single frame"""
        return self.xsize * self.ysize * self.nchannels * self.dtype.itemsize

    def _normalize_slice(self, myslice):
        """
        Return the `python:slice` that would return the same elements but using
        only positive numbers in `start`, `stop` and `step`.
        """
        if myslice is Ellipsis:
            myslice = slice(None)
        start = myslice.start
        stop = myslice.stop
        step = 1 if myslice.step is None else myslice.step

        curr_max = self.nfrms

        if start and start < 0:
            start += curr_max

        if stop and stop < 0:
            stop += curr_max

        if step > 0:
            if start is None:
                start = 0
            if stop is None:
                stop = curr_max
        else:
            step *= -1

            if start is None:
                start = curr_max

            if stop is None:
                stop = -1

            mod = (start - stop) % step

            temp = start
            start = stop
            stop = temp

            start += mod if mod else 1
            stop += 1

        if stop > curr_max:
            stop = curr_max

        return slice(start, stop, step)

    @staticmethod
    def _args_to_slice(arg1, arg2=None, step=None):
        myslice = slice(arg1, arg2, step)
        if arg2 is None and step is None:
            myslice = slice(arg1)
        return myslice

    @staticmethod
    def _args_to_range(arg1, arg2=None, step=None):
        myrange = range(0)
        myslice = InputFileMixin._args_to_slice(arg1, arg2, step)
        if myslice.stop:
            myrange = range(myslice.stop)
        if myslice.start and myslice.stop:
            myrange = range(myslice.start, myslice.stop)
        if myslice.start and myslice.stop and myslice.step:
            myrange = range(myslice.start, myslice.stop, myslice.step)
        return myrange

    @staticmethod
    def _slice_to_range(myslice):
        return InputFileMixin._args_to_range(myslice.start, myslice.stop, myslice.step)
