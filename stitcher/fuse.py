import numpy as np


def to_dtype(x, dtype):
    x = np.rint(x) if np.issubdtype(dtype, np.integer) else x
    return x.astype(dtype, copy=False)


def fuse(a_roi, b_roi):
    """Fuse two overlapping regions.

    Fuses `a_roi` and `b_roi` applying a sinusoidal smoothing. All
    parameters must have equal shapes.

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

    output_height = a_roi.shape[1]
    output_width = a_roi.shape[2]

    rad = np.linspace(0.0, np.pi, output_height, dtype=np.float32)

    alpha = (np.cos(rad) + 1) / 2
    alpha = np.tile(alpha, [output_width])
    alpha = np.reshape(alpha, [output_width, output_height])
    alpha = np.transpose(alpha)

    fused = to_dtype(a_roi * alpha + b_roi * (1 - alpha), dtype)
    return fused


def fuse_queue(q, axis=1, dest_queue=None):
    alayer, stripe_top_Ys = q.get()
    q.task_done()
    if axis == 2:
        alayer = np.rot90(alayer, axes=(1, 2))

    output_stripe = np.zeros((1, 0, alayer.shape[2]), dtype=alayer.dtype)

    fused_height_prev = 0
    prev_Ys_end = stripe_top_Ys + alayer.shape[1]
    while True:
        # add first part
        blayer, Ys = q.get()

        if axis == 2:
            blayer = np.rot90(blayer, axes=(1, 2))

        if Ys is None:
            fused_height = 0
        else:
            fused_height = round(prev_Ys_end - Ys)

        oy_height = alayer.shape[1] - fused_height_prev - fused_height
        ay_from = fused_height_prev
        ay_to = ay_from + oy_height

        output_stripe = np.append(output_stripe, alayer[:, ay_from:ay_to, :],
                                  axis=1)

        if blayer is None:
            break

        a_roi = alayer[:, -fused_height:, :]
        b_roi = blayer[:, 0:fused_height, :]

        # add fused part
        output_stripe = np.append(output_stripe, fuse(a_roi, b_roi), axis=1)
        q.task_done()

        alayer = blayer
        fused_height_prev = fused_height
        prev_Ys_end = Ys + blayer.shape[1]


    if dest_queue is None:
        return output_stripe

    dest_queue.put(output_stripe)
