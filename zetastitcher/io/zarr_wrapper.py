import zarr

from zetastitcher.io.inputfile_mixin import InputFileMixin


class ZarrWrapper(InputFileMixin):
    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path
        self.zarr_file = None
        self.open(file_path)

    def open(self, file_path):
        z = zarr.open(file_path, mode='r')
        self.zarr_file = next(z.arrays())[1]

        self.nfrms = self.zarr_file.shape[0]
        self.ysize = self.zarr_file.shape[-2]
        self.xsize = self.zarr_file.shape[-1]
        self.dtype = self.zarr_file.dtype

    def frame(self, index, dtype=None):
        a = self.zarr_file[index]
        if dtype is not None:
            a = a.astype(dtype)
        return a
