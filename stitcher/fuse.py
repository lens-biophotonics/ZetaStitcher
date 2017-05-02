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


def pad(layer, top_left, stripe_shape):
    Zs = top_left[0]
    Xs = top_left[2]
    pad_left = int(round(Xs))
    pad_right = stripe_shape[-1] - pad_left - layer.shape[-1]

    pad_top = int(round(Zs))
    pad_bottom = stripe_shape[0] - pad_top - layer.shape[0]

    pad_tuples = list((0, 0) for i in layer.shape)
    pad_tuples[0] = (pad_top, pad_bottom)
    pad_tuples[-1] = (pad_left, pad_right)

    return np.pad(layer, pad_tuples, 'constant')


def alpha(overlap):
    rad = np.linspace(0.0, np.pi, overlap, dtype=np.float32)
    alpha = (np.cos(rad) + 1) / 2
    return alpha


def fuse_queue(q, stripe_shape, dtype):
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
    stripe : :class:`numpy.ndarray`
        The fused stripe.
    """
    stripe = np.zeros(stripe_shape, dtype=dtype)

    while True:
        layer, pos, overlap = q.get()

        if layer is None:
            break

        # apply alpha (top, bottom, left, right)
        layer = np.swapaxes(layer, -1, -2)
        if overlap[0]:
            layer[..., :overlap[0]] = to_dtype(
                layer[..., :overlap[0]] * (1 - alpha(overlap[0])), dtype)

        if overlap[1]:
            layer[..., -overlap[1]:] = to_dtype(
                layer[..., -overlap[1]:] * alpha(overlap[1]), dtype)
        layer = np.swapaxes(layer, -1, -2)

        if overlap[2]:
            layer[..., :overlap[2]] = to_dtype(
                layer[..., :overlap[2]] * (1 - alpha(overlap[2])), dtype)
        if overlap[3]:
            layer[..., -overlap[3]:] = to_dtype(
                layer[..., -overlap[3]:] * alpha(overlap[3]), dtype)

        layer[..., -2:, :] = 255
        layer[..., -2:] = 255

        z_from = pos[0]
        z_to = z_from + layer.shape[0]

        y_from = pos[1]
        y_to = y_from + layer.shape[-2]

        x_from = pos[2]
        x_to = x_from + layer.shape[-1]

        stripe[z_from:z_to, ..., y_from:y_to, x_from:x_to] += layer

        q.task_done()

    return stripe
