import skimage.external.tifffile as tiff


class TiffWrapper(object):
    def __init__(self, file_name=None):
        self.file_name = file_name

        self.file = None

        if file_name is not None:
            self.open()

    @property
    def nfrms(self):
        return len(self.file.pages)

    @property
    def xsize(self):
        return self.file.pages[0].image_width

    @property
    def ysize(self):
        return self.file.pages[0].image_length

    @property
    def channels(self):
        try:
            return self.file.pages[0].shape[2]
        except IndexError:
            return 1

    def open(self, file_name=None):
        if file_name is not None:
            self.file_name = file_name

        self.file = tiff.TiffFile(self.file_name)
        setattr(self, 'close', getattr(self.file, 'close'))

    def layer(self, start_frame, end_frame=None, dtype=None):
        if end_frame is None:
            end_frame = start_frame + 1
        a = self.file.asarray(range(start_frame, end_frame), memmap=True)

        if dtype is None:
            return a
        return a.astype(dtype)
