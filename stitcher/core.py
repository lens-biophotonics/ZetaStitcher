from os import environ

import numpy as np
import tensorflow as tf

from scipy import signal
from scipy.ndimage.filters import maximum_filter
from skimage.restoration import denoise_tv_bregman

from dcimg import DCIMGFile


gpu_options = tf.GPUOptions()
try:
    gpu_options.per_process_gpu_memory_fraction = \
        float(environ['TF_GPU_PER_PROCESS_GPU_MEMORY_FRACTION'])
except (KeyError, ValueError):
    pass

sess_config = tf.ConfigProto(gpu_options=gpu_options)


def hamming_window(wz, wy, wx):
    """A 3D Hamming window.

    Parameters
    ----------
    wz : int
        Width along Z.
    wy : int
        Width along Y.
    wx
        Width along X.

    Returns
    -------
    :class:`numpy.ndarray`.
    """
    gx = signal.hamming(wx)
    gy = signal.hamming(wy)
    gz = signal.hamming(wz)

    f = np.outer(gx, gy)
    f = np.outer(f, gz)

    f = np.reshape(f, (wx, wy, wz))
    f = f.T

    return f


def phase_corr_op(ashape, bshape, filter_shape):
    """Construct a TensorFlow op to compute phase correlation.

    Parameters
    ----------
    ashape : tuple of ints
        Shape of input array.
    bshape : tuple of ints
        Shape of input array.
    filter_shape : tuple
        Shape of filter array.

    Returns
    -------
    phase_corr : tf.Operation
        The op to be run to compute phase correlation. When running the op,
        values for the following placeholders must be fed:
        `input/a_placeholder:0`, `input/b_placeholder:0`,
        `input/filter_placeholder:0`.
    """
    with tf.name_scope('input'):
        aph = tf.placeholder(dtype=tf.uint16, shape=ashape,
                             name='a_placeholder')
        bph = tf.placeholder(dtype=tf.uint16, shape=bshape,
                             name='b_placeholder')

        my_filter_t = tf.placeholder(dtype=tf.float32, shape=filter_shape,
                                     name='filter_placeholder')

        at = tf.to_float(aph)
        bt = tf.to_float(bph)

    with tf.name_scope('normalize'):
        at /= tf.reduce_mean(at)
        bt /= tf.reduce_mean(bt)

    with tf.name_scope('window_filter'):
        at = at * my_filter_t
        bt = bt * my_filter_t

    with tf.name_scope('FFT'):
        ac = tf.cast(at, tf.complex64, name='to_complex')
        bc = tf.cast(bt, tf.complex64, name='to_complex')

        aft = tf.fft3d(ac)
        bft = tf.fft3d(bc)

    with tf.name_scope('cross_power_spectrum'):
        prod = aft * tf.conj(bft)
        prodnorm = tf.abs(prod, name='norm')
        ratio = prod / tf.cast(prodnorm, tf.complex64, name='to_complex')

    with tf.name_scope('phase_correlation'):
        phase_corr = tf.real(tf.ifft3d(ratio))

    return phase_corr


def find_phase_corr_max(phase_corr):
    """Find maximum in phase correlation map.

    Parameters
    ----------
    phase_corr : :class:`numpy.ndarray`.

    Returns
    -------
    tuple
        Coordinates of maximum: (`z`, `y`, `x`).
    """
    temp = phase_corr.view(np.ma.MaskedArray)
    temp[:, 100:, :] = np.ma.masked
    temp[:, 0:100:, 100:-100] = np.ma.masked

    argmax = np.argmax(temp)

    dz, dy, dx = np.unravel_index(argmax, temp.shape)

    data = np.squeeze(denoise_tv_bregman(phase_corr[dz, :], weight=0.1))
    data = data.view(np.ma.MaskedArray)
    data.mask = temp.mask
    data_max = maximum_filter(data, 512)
    mask = (data != data_max)

    data = data.view(np.ma.MaskedArray)
    data.mask = mask
    argmax = np.argmax(data)

    dy, dx = np.unravel_index(argmax, data.shape)

    if dx > phase_corr.shape[2] / 2:
        dx -= phase_corr.shape[2]

    return dz, dy, dx


def xcorr2d_op(ashape, bshape):
    """Construct a TensorFlow op to compute 2D cross-correlation.

    Parameters
    ----------
    ashape : tuple of ints
        Shape of input array.
    bshape : tuple of ints
        Shape of input array.

    Returns
    -------
    tf.Operation
        The op to be run to compute 2D cross-correlation. When running the op,
        values for the following placeholders must be fed:
        `input/a_placeholder:0`, `input/b_placeholder:0`.
    """
    with tf.name_scope('input'):
        at = tf.placeholder(dtype=tf.float32, shape=ashape,
                            name='a_placeholder')
        bt = tf.placeholder(dtype=tf.float32, shape=bshape,
                            name='b_placeholder')

        at = tf.expand_dims(at, 0)  # add dummy batch dimension
        at = tf.expand_dims(at, -1)  # add dummy in_channels dimension

        bt = tf.expand_dims(bt, -1)  # add dummy in_channels dimension
        bt = tf.expand_dims(bt, -1)  # add dummy out_channels dimension

    with tf.name_scope('normalize'):
        at /= tf.reduce_mean(at)
        bt /= tf.reduce_mean(bt)

    conv = tf.nn.conv2d(at, bt, padding='SAME',
                        strides=list((1,) + bshape + (1,)))
    conv = tf.squeeze(conv)

    return conv


def overlap_score(alayer, blayer, dz, dy, dx):
    """Compute overlap score between two layers at the given shifts.

    The score is computed as the 2D cross-correlation in the first
    overlapping plane of the input layers.

    Parameters
    ----------
    alayer : 3D class:`numpy.ndarray`.
    blayer : 3D class:`numpy.ndarray`.
    dz : int
        Shift along Z.
    dy : int
        Shift along Y.
    dx : int
        Shift along X.

    Returns
    -------
    score : int
    """
    if dz < 0:
        az = 0
        bz = -dz
    else:
        az = dz
        bz = 0

    aframe = np.squeeze(alayer[az, :])
    bframe = np.squeeze(blayer[bz, :])

    # for the moment consider a and b to have same width
    roi_width = aframe.shape[1] - abs(dx)

    aframe_roi = aframe[-dy:, :]
    bframe_roi = bframe[0:dy, :]

    ax_min = 0
    if dx > 0:
        ax_min = dx
    ax_max = ax_min + roi_width

    bx_min = 0
    bx_max = roi_width

    if dx < 0:
        bx_min = -dx
        bx_max += abs(dx)

    aframe_roi = aframe_roi[-dy:, ax_min:ax_max].astype(np.float32)
    bframe_roi = bframe_roi[:dy, bx_min:bx_max].astype(np.float32)

    score = xcorr2d_op(aframe_roi.shape, bframe_roi.shape)

    with tf.Session(config=sess_config) as sess:
        score = sess.run(score, feed_dict={
            'input/a_placeholder:0': aframe_roi,
            'input/b_placeholder:0': bframe_roi})
    tf.reset_default_graph()

    return score


def stitch(aname, bname, z_min, z_max, overlap, axis=1):
    """Compute optimal shift between adjacent tiles.

    Parameters
    ----------
    aname : str
        Input file name.
    bname : str
        Input file name.
    z_min : int
        Minimum frame index.
    z_max : int
        Maximum frame index.
    overlap : int
        Overlap height in px along the stitching axis
    axis : int
        1 = stitch along Y
        2 = stitch along X

    Returns
    -------
    tuple
        Optimal shifts and overlap score as given by :func:`overlap_score`:
        (`z`, `y`, `x`, `score`).
    """
    a = DCIMGFile(aname)
    b = DCIMGFile(bname)

    alayer = a.layer(z_min, z_max)
    if axis == 2:
        alayer = np.rot90(alayer, axes=(1, 2))
    alayer = alayer[:, -overlap:, :]

    blayer = b.layer(z_min, z_max)
    if axis == 2:
        blayer = np.rot90(blayer, axes=(1, 2))
    blayer = blayer[:, 0:overlap, :]

    my_filter = hamming_window(*alayer.shape).astype(np.float32)

    phase_corr = phase_corr_op(alayer.shape, blayer.shape, my_filter.shape)

    with tf.Session(config=sess_config) as sess:
        phase_corr = sess.run(phase_corr, feed_dict={
            'input/a_placeholder:0': alayer,
            'input/b_placeholder:0': blayer,
            'input/filter_placeholder:0': my_filter})
    tf.reset_default_graph()

    dz, dy, dx = find_phase_corr_max(phase_corr)
    dy = overlap - dy

    score = overlap_score(alayer, blayer, dz, dy, dx)

    print('phase_corr.shape = {}'.format(phase_corr.shape))
    print('dx = {}, dy = {}, dz = {} @ {}, axis = {}, score = {:e}'.format(
        dx, dy, dz, z_min, axis, score))

    a.close()
    b.close()

    return dz, dy, dx, score
