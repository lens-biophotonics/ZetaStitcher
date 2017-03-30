import tensorflow as tf


def xcorr3d(ashape, bshape):
    """Construct a TensorFlow op to compute 3D cross-correlation.

    Parameters
    ----------
    ashape : tuple of ints
        Shape of input array.
    bshape : tuple of ints
        Shape of input array.

    Returns
    -------
    tf.Operation
        The op to be run to compute 3D cross-correlation. When running the op,
        values for the following placeholders must be fed:
        `input/a_placeholder:0`, `input/b_placeholder:0`.
    """
    with tf.name_scope('input'):
        at = tf.placeholder(dtype=tf.float32, shape=ashape, name='a_ph')
        bt = tf.placeholder(dtype=tf.float32, shape=bshape, name='b_ph')

    with tf.name_scope('subtract_mean'):
        at -= tf.reduce_mean(at)
        bt -= tf.reduce_mean(bt)

    with tf.name_scope('expand_dims'):
        at = tf.expand_dims(at, 0)  # add dummy batch dimension
        at = tf.expand_dims(at, -1)  # add dummy in_channels dimension

        bt = tf.expand_dims(bt, -1)  # add dummy in_channels dimension
        bt = tf.expand_dims(bt, -1)  # add dummy out_channels dimension

    conv = tf.nn.conv3d(at, bt, padding='VALID', strides=[1, 1, 1, 1, 1])
    conv = tf.squeeze(conv)

    return conv


def phase_corr(ashape, bshape, filter_shape=None):
    """Construct a TensorFlow op to compute phase correlation.

    Parameters
    ----------
    ashape : tuple of ints
        Shape of input array.
    bshape : tuple of ints
        Shape of input array.
    filter_shape : tuple
        Shape of filter array. Optional. If not given, the window filter is
        not applied.

    Returns
    -------
    phase_corr : tf.Operation
        The op to be run to compute phase correlation. When running the op,
        values for the following placeholders must be fed:
        `input/a_ph:0`, `input/b_ph:0`, `input/filter_ph:0`.
    """
    my_filter_t = None

    with tf.name_scope('input'):
        aph = tf.placeholder(dtype=tf.uint16, shape=ashape, name='a_ph')
        bph = tf.placeholder(dtype=tf.uint16, shape=bshape, name='b_ph')

        if filter_shape is not None:
            my_filter_t = tf.placeholder(dtype=tf.float32, shape=filter_shape,
                                         name='filter_ph')

        at = tf.to_float(aph)
        bt = tf.to_float(bph)

    with tf.name_scope('subtract_mean'):
        at -= tf.reduce_mean(at)
        bt -= tf.reduce_mean(bt)

    if filter_shape is not None:
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
        ifft = tf.ifft3d(ratio)
        phase_corr = tf.square(tf.real(ifft) + tf.square(tf.imag(ifft)))
        phase_corr = tf.sqrt(phase_corr)

    return phase_corr


def normxcorr2(ashape, bshape):
    """Construct a Tensorflow op to compute normalized cross correlation (2D).

    At the moment this op does not perform as fast as expected.

    Parameters
    ----------
    ashape : tuple
        must be len(ashape) = 3
    bshape :
        must be len(bshape) = 3 and bshape[0] = 1

    Returns
    -------
    tf.Operation
        The op to be run to compute normalized 2D cross-correlation. When
        running the op, values for the following placeholders must be fed:
        `input/a_ph:0`, `input/b_ph:0`.
    """
    with tf.name_scope('input'):
        at = tf.placeholder(dtype=tf.float32, shape=ashape, name='a_ph')
        bt = tf.placeholder(dtype=tf.float32, shape=bshape, name='b_ph')

    A = tf.constant(bshape[1] * bshape[2], dtype=tf.float32)  # area
    sums_b = tf.reduce_sum(bt)  # scalar
    sums_b2 = tf.reduce_sum(tf.square(bt))  # scalar

    with tf.name_scope('expand_dims'):
        at = tf.expand_dims(at, 0)  # add dummy batch dimension
        at = tf.expand_dims(at, -1)  # add dummy in_channels dimension

        bt = tf.expand_dims(bt, -1)  # add dummy in_channels dimension
        bt = tf.expand_dims(bt, -1)  # add dummy out_channels dimension

    b1 = tf.ones(bshape + (1, 1))

    # conv3d is faster than batches of conv2d
    sums_a = tf.nn.conv3d(at, b1, strides=[1, 1, 1, 1, 1], padding='VALID',
                          name='sums_a')
    sums_a2 = tf.nn.conv3d(tf.square(at), b1, strides=[1, 1, 1, 1, 1],
                           padding='VALID', name='sums_a2')
    conv = tf.nn.conv3d(at, bt, strides=[1, 1, 1, 1, 1], padding='VALID',
                        name='myconv')

    num = conv - sums_b * sums_a / A
    denom = tf.sqrt(
        (sums_a2 - tf.square(sums_a) / A) * (sums_b2 - tf.square(sums_b) / A))

    normxcorr = tf.squeeze(num / denom)
    return normxcorr
