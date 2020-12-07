class InputFileMixin:
    def __init__(self):
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
