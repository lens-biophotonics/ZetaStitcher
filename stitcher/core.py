from os import environ

import numpy as np
import tensorflow as tf

from scipy import signal
from scipy.ndimage.filters import maximum_filter
from skimage.restoration import denoise_tv_bregman

from dcimg import DCIMGFile

from . import ops


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


def find_phase_corr_max(phase_corr, max_shift_x=100, max_shift_y=100):
    """Find maximum in phase correlation map.

    Parameters
    ----------
    phase_corr : :class:`numpy.ndarray`.
    max_shift_x : int
        Maximum allowed shift along X in px
    max_shift_y : int
        Maximum allowed shift along Y in px.

    Returns
    -------
    tuple
        Coordinates of maximum: (`z`, `y`, `x`).
    """
    temp = phase_corr.view(np.ma.MaskedArray)
    temp[:, max_shift_y:, :] = np.ma.masked
    temp[:, :max_shift_y, max_shift_x:-max_shift_x] = np.ma.masked

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

    score = ops.xcorr2d_op(aframe_roi.shape, bframe_roi.shape)

    with tf.Session(config=sess_config) as sess:
        score = sess.run(score, feed_dict={
            'input/a_placeholder:0': aframe_roi,
            'input/b_placeholder:0': bframe_roi})
    tf.reset_default_graph()

    return score


def stitch(aname, bname, z_min, z_max, overlap, axis=1, max_shift_x=100,
           max_shift_y=100):
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
    max_shift_x : int
        Maximum allowed shift along X in px
    max_shift_y : int
        Maximum allowed shift along Y in px.

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

    phase_corr = ops.phase_corr_op(alayer.shape, blayer.shape, my_filter.shape)

    with tf.Session(config=sess_config) as sess:
        phase_corr = sess.run(phase_corr, feed_dict={
            'input/a_placeholder:0': alayer,
            'input/b_placeholder:0': blayer,
            'input/filter_placeholder:0': my_filter})
    tf.reset_default_graph()

    dz, dy, dx = find_phase_corr_max(phase_corr, max_shift_x, max_shift_y)
    dy = overlap - dy

    score = overlap_score(alayer, blayer, dz, dy, dx)

    print('dx = {}, dy = {}, dz = {} @ {}, axis = {}, score = {:e}'.format(
        dx, dy, dz, z_min, axis, score))

    a.close()
    b.close()

    return dz, dy, dx, score
