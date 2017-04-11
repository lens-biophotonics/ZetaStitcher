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
    output_width = a_roi.shape[-1]

    rad = np.linspace(0.0, np.pi, output_height, dtype=np.float32)

    alpha = (np.cos(rad) + 1) / 2
    alpha = np.tile(alpha, [output_width])
    alpha = np.reshape(alpha, [output_width, output_height])
    alpha = np.transpose(alpha)

    fused = to_dtype(a_roi * alpha + b_roi * (1 - alpha), dtype)
    return fused


def fuse_queue(q, stripe_width=None):
    alayer, pos = q.get()
    q.task_done()

    if stripe_width is None:
        stripe_width = alayer.shape[-1]

    ostripe_shape = list(alayer.shape)
    ostripe_shape[-2] = 0
    ostripe_shape[-1] = stripe_width
    output_stripe = np.zeros(ostripe_shape, dtype=alayer.dtype)

    fused_height_prev = 0

    prev_Ys_end = pos[1] + alayer.shape[-2]
    prev_Xs = pos[2]
    while True:
        # add first part
        blayer, pos = q.get()

        if pos is None:
            fused_height = 0
        else:
            Ys = pos[1]
            Xs = pos[2]
            fused_height = int(round(prev_Ys_end - Ys))

        oy_height = alayer.shape[-2] - fused_height_prev - fused_height
        if oy_height <= 0:
            raise ValueError('oy_height must be positive')
        ay_from = fused_height_prev
        ay_to = round(ay_from + oy_height)

        pad_left = int(round(prev_Xs))
        pad_right = stripe_width - pad_left - alayer.shape[-1]

        pad_tuple = tuple((0, 0) for i in alayer.shape)
        pad_tuple = pad_tuple[:-1] + ((pad_left, pad_right), )
        first = np.pad(alayer[..., ay_from:ay_to, :], pad_tuple,
                       mode='constant')

        output_stripe = np.append(output_stripe, first, axis=-2)

        if blayer is None:
            break

        dx = int(round(Xs - prev_Xs))

        left = max(prev_Xs, Xs)
        right = min(prev_Xs + alayer.shape[-1], Xs + blayer.shape[-1])

        roi_width = int(round(right - left))

        if dx > 0:
            bx_from = 0
            ax_from = dx
        else:
            ax_from = 0
            bx_from = -dx

        ax_to = ax_from + roi_width
        bx_to = bx_from + roi_width

        a_roi = alayer[..., -fused_height:, ax_from:ax_to]
        b_roi = blayer[..., 0:fused_height, bx_from:bx_to]

        pad_left = int(round(max(Xs, prev_Xs)))
        pad_right = stripe_width - pad_left - a_roi.shape[-1]

        # add fused part
        fused = fuse(a_roi, b_roi)
        pad_tuple = tuple((0, 0) for i in alayer.shape)
        pad_tuple = pad_tuple[:-1] + ((pad_left, pad_right), )
        fused = np.pad(fused, pad_tuple, mode='constant')
        output_stripe = np.append(output_stripe, fused, axis=-2)
        q.task_done()

        alayer = blayer
        fused_height_prev = fused_height
        prev_Ys_end = Ys + blayer.shape[-2]
        prev_Xs = Xs

    return output_stripe
