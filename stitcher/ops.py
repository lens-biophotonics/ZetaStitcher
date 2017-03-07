import tensorflow as tf


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

    with tf.name_scope('subtract_mean'):
        at -= tf.reduce_mean(at)
        bt -= tf.reduce_mean(bt)

    conv = tf.nn.conv2d(at, bt, padding='VALID',
                        strides=[1, 1, 1, 1])
    conv = tf.squeeze(conv)

    return conv


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

    with tf.name_scope('subtract_mean'):
        at -= tf.reduce_mean(at)
        bt -= tf.reduce_mean(bt)

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
