import pyfftw
import numpy as np


def normxcorr2_fftw(aslice, bframe):
    """Compute normalized cross correlation using fftw.

    .. DANGER::
        Input arrays might be overwritten by in-place transforms!

    Parameters
    ----------
    aslice : :class:`numpy.ndarray`
    bframe : :class:`numpy.ndarray`

    Returns
    -------
    :class:`numpy.ndarray`
    """
    ashape = aslice.shape
    b_old_shape = bframe.shape

    if not (aslice.dtype == np.float32):
        aslice = aslice.astype(np.float32)
    if not (bframe.dtype == np.float32):
        bframe = bframe.astype(np.float32)

    out_height = ashape[1] - b_old_shape[1] + 1
    out_width = ashape[2] - b_old_shape[2] + 1

    sums_b = np.sum(bframe)
    sums_b2 = np.sum(np.square(bframe))

    b1 = np.ones_like(bframe)
    bframe = np.pad(bframe, ((0, 0),
                             (0, ashape[1] - b_old_shape[1]),
                             (0, ashape[2] - b_old_shape[2])), 'constant')
    b1 = np.pad(b1, ((0, 0),
                     (0, ashape[1] - b_old_shape[1]),
                     (0, ashape[2] - b_old_shape[2] + 2)), 'constant')
    b1_real_input = b1[..., :ashape[2]]
    b1_complex_output = b1.view('complex64')

    # pad for in-place transform
    aslice = np.pad(aslice, ((0, 0), (0, 0), (0, 2)), 'constant')

    a_real_input = aslice[..., :ashape[2]]
    a_complex_output = aslice.view('complex64')

    fft_a2 = pyfftw.empty_aligned(a_complex_output.shape, dtype='complex64')
    fft_object = pyfftw.FFTW(np.square(a_real_input), fft_a2, axes=(1, 2),
                             flags=['FFTW_ESTIMATE'])
    fft_object.execute()

    # will overwrite aslice
    fft_object = pyfftw.FFTW(a_real_input, a_complex_output, axes=(1, 2),
                             flags=['FFTW_ESTIMATE'])
    fft_object.execute()

    # pad for in-place transform
    bframe = np.pad(bframe, ((0, 0), (0, 0), (0, 2)), 'constant')

    b_real_input = bframe[..., :ashape[2]]
    b_real_input[:] = bframe[..., :ashape[2]]
    b_complex_output = bframe.view('complex64')
    fft_object = pyfftw.FFTW(b_real_input, b_complex_output, axes=(1, 2),
                             flags=['FFTW_ESTIMATE'])
    fft_object.execute()

    fft_object = pyfftw.FFTW(
        b1_real_input, b1_complex_output, axes=(1, 2), flags=['FFTW_ESTIMATE'])
    fft_object.execute()

    # fftw performs unscaled transforms, therefore we need to rescale by the
    # frame area
    a_frame_area = np.array(ashape[1] * ashape[2], dtype=np.float32)

    conv = pyfftw.empty_aligned(ashape, dtype='float32')
    fft_object = pyfftw.FFTW(
        a_complex_output * np.conj(b_complex_output), conv, axes=(1, 2),
        flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD')
    fft_object.execute()
    conv = conv[:, :out_height, :out_width] / a_frame_area

    fft_b1_conj = np.conj(b1_complex_output)
    sums_a = pyfftw.empty_aligned(ashape, dtype='float32')
    fft_object = pyfftw.FFTW(
        a_complex_output * fft_b1_conj, sums_a, axes=(1, 2),
        flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD')
    fft_object.execute()
    sums_a = sums_a[:, :out_height, :out_width] / a_frame_area

    sums_a2 = pyfftw.empty_aligned(ashape, dtype='float32')
    fft_object = pyfftw.FFTW(
        fft_a2 * fft_b1_conj, sums_a2, axes=(1, 2),
        flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD')
    fft_object.execute()
    sums_a2 = sums_a2[:, :out_height, :out_width] / a_frame_area

    A = np.array(b_old_shape[1] * b_old_shape[2], dtype=np.float32)

    num = conv - sums_b * sums_a / A
    denom = np.sqrt(np.abs(
        (sums_a2 - np.square(sums_a) / A) * (sums_b2 - np.square(sums_b) / A)))
    denom[denom == 0] = 1

    normxcorr = num / denom

    return normxcorr


def normxcorr2(aslice, bframe):
    ashape = aslice.shape
    bshape = bframe.shape

    out_height = ashape[1] - bshape[1] + 1
    out_width = ashape[2] - bshape[2] + 1

    b1 = np.ones_like(bframe)

    fft_a = np.fft.fft2(aslice)
    fft_a2 = np.fft.fft2(np.square(aslice))
    fft_b = np.fft.fft2(bframe, s=[ashape[1], ashape[2]])
    fft_b1 = np.fft.fft2(b1, s=[ashape[1], ashape[2]])

    conv = np.fft.ifft2(fft_a * np.conj(fft_b))
    conv = np.real(conv[:, :out_height, :out_width])
    sums_a = np.fft.ifft2(fft_a * np.conj(fft_b1))
    sums_a = np.real(sums_a[:, :out_height, :out_width])
    sums_a2 = np.fft.ifft2(fft_a2 * np.conj(fft_b1))
    sums_a2 = np.real(sums_a2[:, :out_height, :out_width])

    sums_b = np.sum(bframe)
    sums_b2 = np.sum(np.square(bframe))

    A = np.array(bshape[1] * bshape[2])

    num = conv - sums_b * sums_a / A
    denom = np.sqrt(
        (sums_a2 - np.square(sums_a) / A) * (sums_b2 - np.square(sums_b) / A))

    normxcorr = num / denom

    return normxcorr
