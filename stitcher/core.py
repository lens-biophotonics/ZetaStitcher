import numpy as np
import tensorflow as tf

from scipy import signal
from scipy.ndimage.filters import maximum_filter
from skimage.restoration import denoise_tv_bregman

from dcimg import DCIMGFile


def window_filter(wz, wy, wx):
    gx = signal.hamming(wx)
    gy = signal.hamming(wy)
    gz = signal.hamming(wz)

    f = np.outer(gx, gy)
    f = np.outer(f, gz)

    f = np.reshape(f, (wx, wy, wz))
    f = f.T

    return f


def phase_corr_op(ashape, bshape, filter_shape):
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


def find_shift(phase_corr):
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


def conv2d_op(ashape, bshape):
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

    score = conv2d_op(aframe_roi.shape, bframe_roi.shape)

    with tf.Session() as sess:
        score = sess.run(score, feed_dict={
            'input/a_placeholder:0': aframe_roi,
            'input/b_placeholder:0': bframe_roi})
    tf.reset_default_graph()

    return score


def stitch(aname, bname, bottom, top, overlap, axis=1):
    a = DCIMGFile(aname)
    b = DCIMGFile(bname)

    alayer = a.layer(bottom, top)
    if axis == 2:
        alayer = np.rot90(alayer, axes=(1, 2))
    alayer = alayer[:, -overlap:, :]

    blayer = b.layer(bottom, top)
    if axis == 2:
        blayer = np.rot90(blayer, axes=(1, 2))
    blayer = blayer[:, 0:overlap, :]

    my_filter = window_filter(*alayer.shape).astype(np.float32)

    phase_corr = phase_corr_op(alayer.shape, blayer.shape, my_filter.shape)

    with tf.Session() as sess:
        phase_corr = sess.run(phase_corr, feed_dict={
            'input/a_placeholder:0': alayer,
            'input/b_placeholder:0': blayer,
            'input/filter_placeholder:0': my_filter})
    tf.reset_default_graph()

    dz, dy, dx = find_shift(phase_corr)
    dy = overlap - dy

    score = overlap_score(alayer, blayer, dz, dy, dx)

    print('phase_corr.shape = {}'.format(phase_corr.shape))
    print('dx = {}, dy = {}, dz = {} @ {}, axis = {}, score = {:e}'.format(
        dx, dy, dz, bottom, axis, score))

    a.close()
    b.close()

    return dz, dy, dx, score
