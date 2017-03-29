import pyfftw
import numpy as np

from . import inputfile


def normxcorr2_fftw(alayer, blayer):
    ashape = alayer.shape
    b_old_shape = blayer.shape

    out_height = ashape[1] - b_old_shape[1] + 1
    out_width = ashape[2] - b_old_shape[2] + 1

    sums_b = np.sum(blayer)
    sums_b2 = np.sum(np.square(blayer))

    b1 = np.ones_like(blayer)
    blayer = np.pad(blayer, ((0, 0),
                             (0, ashape[1] - b_old_shape[1]),
                             (0, ashape[2] - b_old_shape[2])), 'constant')
    b1 = np.pad(b1, ((0, 0),
                     (0, ashape[1] - b_old_shape[1]),
                     (0, ashape[2] - b_old_shape[2] + 2)), 'constant')
    b1_real_input = b1[..., :ashape[2]]
    b1_complex_output = b1.view('complex64')
    bshape = blayer.shape

    a_full = pyfftw.empty_aligned(
        (ashape[0], ashape[1], 2 * (ashape[2] // 2 + 1)), dtype='float32')
    a_real_input = a_full[..., :ashape[2]]
    a_real_input[:] = alayer[:]
    a_complex_output = a_full.view('complex64')
    fft_object = pyfftw.FFTW(a_real_input, a_complex_output, axes=(1, 2),
                             flags=['FFTW_ESTIMATE'])
    fft_object.execute()

    fft_a2 = pyfftw.empty_aligned(a_complex_output.shape, dtype='complex64')
    fft_object = pyfftw.FFTW(np.square(alayer), fft_a2, axes=(1, 2),
                             flags=['FFTW_ESTIMATE'])
    fft_object.execute()

    b_full = pyfftw.empty_aligned(
        (bshape[0], bshape[1], 2 * (bshape[2] // 2 + 1)), dtype='float32')
    b_real_input = b_full[..., :ashape[2]]
    b_real_input[:] = blayer[:]
    b_complex_output = b_full.view('complex64')
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
    denom = np.sqrt(
        (sums_a2 - np.square(sums_a) / A) * (sums_b2 - np.square(sums_b) / A))

    normxcorr = num / denom

    return normxcorr


def normxcorr2(alayer, blayer):
    ashape = alayer.shape
    bshape = blayer.shape

    out_height = ashape[1] - bshape[1] + 1
    out_width = ashape[2] - bshape[2] + 1

    b1 = np.ones_like(blayer)

    fft_a = np.fft.fft2(alayer)
    fft_a2 = np.fft.fft2(np.square(alayer))
    fft_b = np.fft.fft2(blayer, s=[ashape[1], ashape[2]])
    fft_b1 = np.fft.fft2(b1, s=[ashape[1], ashape[2]])

    conv = np.fft.ifft2(fft_a * np.conj(fft_b))
    conv = np.real(conv[:, :out_height, :out_width])
    sums_a = np.fft.ifft2(fft_a * np.conj(fft_b1))
    sums_a = np.real(sums_a[:, :out_height, :out_width])
    sums_a2 = np.fft.ifft2(fft_a2 * np.conj(fft_b1))
    sums_a2 = np.real(sums_a2[:, :out_height, :out_width])

    sums_b = np.sum(blayer)
    sums_b2 = np.sum(np.square(blayer))

    A = np.array(bshape[1] * bshape[2])

    num = conv - sums_b * sums_a / A
    denom = np.sqrt(
        (sums_a2 - np.square(sums_a) / A) * (sums_b2 - np.square(sums_b) / A))

    normxcorr = num / denom

    return normxcorr


def stitch(aname, bname, z_frame, axis, overlap, max_shift_z=20,
           max_shift_x=20):
    """Compute optimal shift between adjacent tiles.

    Two 3D tiles are compared at the specified frame index to find their best
    alignment. The following conventions are used:

    * Z is the direction along the stack height,
    * (X, Y) is the frame plane,
    * Y is the direction along which frames are supposed to overlap,
    * X is the direction orthogonal to Y in the frame plane (X, Y).

    Parameters
    ----------
    aname : str
        Input file name.
    bname : str
        Input file name.
    z_frame : int
        Index of frame used for alignment.
    axis : int
        Perform stitching along this axis. Following the convention above, this
        axis will be called Y. The following values can be used:

        1. stitch vertically
        2. stitch horizontally
    overlap : int
        Overlap height in px along the stitching axis (Y).
    max_shift_z : int
        Maximum allowed shift in px along the stack height (Z).
    max_shift_x : int
        Maximum allowed lateral shift in px.

    Returns
    -------
    tuple
        Optimal shifts and overlap score computed by means of normalized
        cross correlation: (`z`, `y`, `x`, `score`).
    """
    a = inputfile.InputFile(aname)
    b = inputfile.InputFile(bname)

    a.channel = 1
    b.channel = 1

    z_min = z_frame - max_shift_z
    z_max = z_frame + max_shift_z + 1

    alayer = a.layer(z_min, z_max, dtype=np.float32)
    if axis == 2:
        alayer = np.rot90(alayer, axes=(1, 2))
    alayer = alayer[:, -overlap:, :]

    blayer = b.layer_idx(z_frame, dtype=np.float32)
    if axis == 2:
        blayer = np.rot90(blayer, axes=(1, 2))
    blayer = blayer[:, 0:overlap, :]

    half_max_shift_x = max_shift_x // 2

    blayer = blayer[:, 0:int(overlap * 0.5),
                    half_max_shift_x:-half_max_shift_x]

    xcorr = normxcorr2_fftw(alayer, blayer)

    shift = list(np.unravel_index(np.argmax(xcorr), xcorr.shape))
    score = xcorr[tuple(shift)]

    print('shift: ' + str(shift))
    shift[0] -= max_shift_z
    shift[1] = overlap - shift[1]
    shift[2] -= half_max_shift_x

    print('max @ {}: {}, score: {:.3}'.format(z_frame, shift, score))
