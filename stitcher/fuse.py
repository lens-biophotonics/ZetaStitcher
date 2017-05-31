import numpy as np


def flatten(my_list):
    return [item for sublist in my_list for item in sublist]


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


def fuse_queue(q, output_shape):
    """Fuses a queue of images along Y, optionally applying padding.

    Parameters
    ----------
    q : :py:class:`queue.Queue`
        A queue containing elements in the form ``[layer, top_left, overlaps]``
        where `layer` is a :class:`numpy.ndarray`, `top_left` is a list
        specifying the image position in the form ``[Z, Y, X]``, `overlaps`
        is a list in the form ``[top, bottom, left, right]`` specifying
        overlaps with adjacent tiles.
    output_shape : tuple of ints
        Final shape of the fused queue.

    Returns
    -------
    stripe : :class:`numpy.ndarray`
        The fused stripe.
    """

    dtype = q.queue[0][0].dtype

    stripe = np.zeros(output_shape, dtype=dtype)

    while True:
        layer, pos, overlaps = q.get()

        if layer is None:
            break

        z_from = pos[0]
        z_to = z_from + layer.shape[0]

        y_from = pos[1]
        y_to = y_from + layer.shape[-2]

        x_from = pos[2]
        x_to = x_from + layer.shape[-1]

        z = np.array(flatten(overlaps[['Z_from', 'Z_to']].values))
        z = np.unique(z)
        z = np.sort(z)

        height = layer.shape[-2]
        width = layer.shape[-1]

        grid = np.vstack(
            np.meshgrid(np.linspace(1, height, height),
                        np.linspace(1, width, width))).reshape(2, -1).T
        xy_weights = np.array(
            [min(x, y, width - x, height - y) ** 0.5 for y, x in grid])
        xy_weights = xy_weights.reshape(width, -1).T

        for z_from, z_to in zip(z, z[1::]):
            sums = np.copy(xy_weights)
            condition = (
                (overlaps['Z_from'] <= z_from) & (z_to <= (overlaps['Z_to'])))

            for _, row in overlaps[condition].iterrows():
                width = row.X_to - row.X_from
                height = row.Y_to - row.Y_from
                area = width * height
                if not area:
                    continue

                grid = np.vstack(
                    np.meshgrid(np.linspace(1, height, height),
                                np.linspace(1, width, width))).reshape(2, -1).T
                distance_from_border = [min(x, y) ** 0.5 for y, x in grid]
                w = np.array(distance_from_border).reshape(width, -1).T

                if row.X_from == 0:
                    w = np.fliplr(w)
                if row.Y_from == 0:
                    w = np.flipud(w)

                xy_index = np.index_exp[row.Y_from:row.Y_to,
                                        row.X_from:row.X_to]
                sums[xy_index] += w

            layer[:z_to - z_from, ...] *= (xy_weights / sums)

        layer[..., -2:, :] = 255
        layer[..., -2:] = 255

        stripe_roi_index = np.index_exp[
                           z_from:z_to, ..., y_from:y_to, x_from:x_to]
        stripe[stripe_roi_index] += layer

        q.task_done()

    return stripe
