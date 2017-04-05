import numpy as np


def to_dtype(x, dtype):
    x = np.rint(x) if np.issubdtype(dtype, np.integer) else x
    return x.astype(dtype, copy=False)


def fuse(a_roi, b_roi, dest):
    """Fuse two overlapping regions.

    Fuses `a_roi` and `b_roi` applying a sinusoidal smoothing. All
    parameters must have equal shapes.

    Parameters
    ----------
    a_roi : :class:`numpy.ndarray`
    b_roi : :class:`numpy.ndarray`
    dest : :class:`numpy.ndarray`
    """
    if a_roi.shape != b_roi.shape or a_roi.shape != dest.shape:
        raise ValueError(
            'ROI shapes must be equal. a: {}, b: {}, dest: {}'.format(
                a_roi.shape, b_roi.shape, dest.shape))

    dtype = a_roi.dtype
    a_roi = a_roi.astype(np.float32, copy=False)
    b_roi = b_roi.astype(np.float32, copy=False)

    output_height = a_roi.shape[1]
    output_width = a_roi.shape[2]

    rad = np.linspace(0.0, np.pi, output_height, dtype=np.float32)

    alpha = (np.cos(rad) + 1) / 2
    alpha = np.tile(alpha, [output_width])
    alpha = np.reshape(alpha, [output_width, output_height])
    alpha = np.transpose(alpha)

    dest[:] = to_dtype(a_roi * alpha + b_roi * (1 - alpha), dtype)
