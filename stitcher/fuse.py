import numpy as np


def to_dtype(x, dtype):
    x = np.rint(x) if np.issubdtype(dtype, np.integer) else x
    return x.astype(dtype, copy=False)


def fuse(a_roi, b_roi):
    """Fuse two overlapping regions.

    Fuses `a_roi` and `b_roi` along Y applying a sinusoidal smoothing. The
    passed arrays must have equal shapes.

    Parameters
    ----------
    a_roi : :class:`numpy.ndarray`
    b_roi : :class:`numpy.ndarray`
    """
    if a_roi.shape != b_roi.shape:
        raise ValueError(
            'ROI shapes must be equal. a: {}, b: {}'.format(
                a_roi.shape, b_roi.shape))

    dtype = a_roi.dtype
    a_roi = a_roi.astype(np.float32, copy=False)
    b_roi = b_roi.astype(np.float32, copy=False)

    output_height = a_roi.shape[-2]

    rad = np.linspace(0.0, np.pi, output_height, dtype=np.float32)
    alpha = (np.cos(rad) + 1) / 2
    alpha = alpha[:, np.newaxis]

    fused = to_dtype(a_roi * alpha + b_roi * (1 - alpha), dtype)
    return fused


def fuse_queue(q, stripe_width=None, stripe_thickness=None):
    """Fuses a queue of images along Y, optionally applying padding.

    Parameters
    ----------
    q : :py:class:`queue.Queue`
        A queue containing elements in the form (`layer`, `pos`) where
        `layer` is a :class:`numpy.ndarray` and `pos` is a list specifying
        the image position in the form [`Z`, `Y`, `X`]. If `stripe_width` is
        not specified, the size of the last dimension should be equal for all
        images, and `X` should be set to 0. The same applies with
        `stripe_thickness` and `Z`.
    stripe_width : int
        If specified, input images can have different shapes in X and will be
        padded according to their position `X`.
    stripe_thickness : int
        If specified, input images can have different shapes in Z and will be
        padded according to their position `Z`.

    Returns
    -------
    output_stripe : :class:`numpy.ndarray`
        The fused stripe.
    """
    def pad(layer, top_left):
        Zs = top_left[0]
        Xs = top_left[2]
        pad_left = int(round(Xs))
        pad_right = stripe_width - pad_left - layer.shape[-1]

        pad_top = int(round(Zs))
        pad_bottom = stripe_thickness - pad_top - layer.shape[0]

        pad_tuples = list((0, 0) for i in layer.shape)
        pad_tuples[0] = (pad_top, pad_bottom)
        pad_tuples[-1] = (pad_left, pad_right)

        return np.pad(layer, pad_tuples, 'constant')

    alayer, pos = q.get()
    q.task_done()

    if stripe_width is None:
        stripe_width = alayer.shape[-1]

    if stripe_thickness is None:
        stripe_thickness = alayer.shape[0]

    ostripe_shape = list(alayer.shape)
    ostripe_shape[0] = stripe_thickness
    ostripe_shape[-2] = 0
    ostripe_shape[-1] = stripe_width
    output_stripe = np.zeros(ostripe_shape, dtype=alayer.dtype)

    fused_height_prev = 0

    prev_Ys_end = pos[1] + alayer.shape[-2]
    alayer = pad(alayer, pos)
    while True:
        # add first part
        blayer, pos = q.get()

        if pos is None:
            fused_height = 0
        else:
            Zs = pos[0]
            Ys = pos[1]
            Xs = pos[2]
            fused_height = int(round(prev_Ys_end - Ys))

        oy_height = alayer.shape[-2] - fused_height_prev - fused_height
        if oy_height <= 0:
            raise ValueError('oy_height must be positive')
        ay_from = fused_height_prev
        ay_to = round(ay_from + oy_height)

        output_stripe = np.append(
            output_stripe, alayer[..., ay_from:ay_to, :], axis=-2)

        if blayer is None:
            break
        blayer = pad(blayer, [Zs, 0, Xs])


        a_roi = alayer[..., -fused_height:, :]
        b_roi = blayer[..., 0:fused_height, :]

        # add fused part
        fused = fuse(a_roi, b_roi)
        output_stripe = np.append(output_stripe, fused, axis=-2)
        q.task_done()

        alayer = blayer
        fused_height_prev = fused_height
        prev_Ys_end = Ys + blayer.shape[-2]

    return output_stripe
