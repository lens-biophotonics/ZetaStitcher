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


def fuse_queue(q, stripe_shape, dest_queue=None):
    stripe_height = stripe_shape[0]
    stripe_width = stripe_shape[1]

    alayer, Ys = q.get()
    q.task_done()

    output_stripe = np.zeros((1, stripe_height, stripe_width),
                             dtype=alayer.dtype)

    fused_height_prev = 0
    current_y = 0
    prev_Ys_end = Ys + alayer.shape[1]
    while True:
        # add first part
        blayer, Ys = q.get()

        if Ys is None:
            fused_height = 0
        else:
            fused_height = round(prev_Ys_end - Ys)

        oy_height = alayer.shape[1] - fused_height_prev - fused_height
        oy_to = current_y + oy_height

        ay_from = fused_height_prev
        ay_to = ay_from + oy_height

        output_stripe[0, current_y:oy_to, :] = alayer[0, ay_from:ay_to, :]

        current_y = oy_to

        if blayer is None:
            diff = stripe_height - current_y
            assert diff <=1
            if diff:
                output_stripe = output_stripe[:, :-diff, :]
            break

        a_roi = alayer[:, -fused_height:, :]
        b_roi = blayer[:, 0:fused_height, :]

        # add fused part
        oy_to = current_y + fused_height

        fuse(a_roi, b_roi, output_stripe[:, current_y:oy_to, :])
        q.task_done()

        current_y = oy_to
        alayer = blayer
        fused_height_prev = fused_height
        prev_Ys_end = Ys + blayer.shape[1]


    if dest_queue is None:
        return output_stripe

    dest_queue.put(output_stripe)
