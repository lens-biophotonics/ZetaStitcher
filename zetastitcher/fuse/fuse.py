import re
import math
import numpy as np

from functools import lru_cache

from .lcd_numbers import numbers, canvas_shape


def flatten(my_list):
    return [item for sublist in my_list for item in sublist]


@lru_cache()
def squircle_alpha(height, width):
    squircle = np.zeros((height, width))
    ratio = width / height
    a = math.ceil(width / 2)
    b = math.ceil(height / 2)
    grid = np.vstack(np.meshgrid(np.linspace(0, b - 1, b),
                                 np.linspace(0, a - 1, a))).reshape(2, -1).T
    grid = grid.astype(np.int)
    N = max(a, b)
    ps = np.logspace(np.log10(2), np.log10(50), N)  # exponents
    # ps = np.ones(N) * 2
    alpha = np.linspace(0, 1, N)

    if a > b:
        dra = a / N
        ras = np.arange(0, a, dra) + 1
        rbs = ras / ratio
        drb = dra / ratio
    else:
        drb = b / N
        rbs = np.arange(0, b, drb) + 1
        ras = rbs * ratio
        dra = drb * ratio

    start_y = b - 1 if height % 2 else b
    start_x = a - 1 if width % 2 else a

    for y, x in grid:
        j = x / dra
        k = y / drb
        i = int(max(j, k))
        count = -1  # 0-based
        for n, p, ra, rb in zip(range(0, N - i), ps[i:], ras[i:], rbs[i:]):
            count += 1

            constant = math.pow(x / ra, p) + math.pow(y / rb, p)

            if constant < 1:
                break

        ii = i + count
        squircle[start_y + y, start_x + x] = alpha[ii] ** 2

    stop_y_up = b - 1 if height % 2 else b
    start_y_down = b
    squircle[:stop_y_up, start_x:] = np.flipud(squircle[start_y_down:, start_x:])

    stop_x_left = a - 1 if width % 2 else a
    start_x_right = a
    squircle[:, :stop_x_left] = np.fliplr(squircle[:, start_x_right:])

    squircle = 1 - squircle

    return squircle


def fuse_queue(q, dest, frame_shape, debug=False):
    """Fuse a queue of images along Y, optionally applying padding.

    Parameters
    ----------
    q : :py:class:`queue.Queue`
        A queue containing elements in the form ``[hyperslice, index, zfrom,
        offset_idx, top_left, overlaps]`` where `hyperslice` is a
        `numpy.ndarray`, `index` and `zfrom` are the tile index in the
        pandas dataframe and the starting frame in the original stack (these
        two are used for debugging purposes only), `offset_idx` is a
        tuple of :class:`slice` objects with the slice offset inside a single
        stack, `top_left` is a list specifying the image position in the form
        ``[Z, Y, X]``, `overlaps` is a :class:`pandas.DataFrame` specifying
        overlaps with adjacent tiles.
    frame_shape : tuple
        Shape of a stack plane (XY).
    dest : :class:`numpy.ndarray`
        Destination array.
    debug: bool
        Whether to overlay debug information (tile edges and numbers).
    """

    while True:
        got = q.get()

        if got is None:
            break

        my_slice, index_dbg, zfrom_dbg, sl, pos, overlaps = got

        z_from = pos[0]
        z_to = z_from + my_slice.shape[0]

        y_from = pos[1]
        y_to = y_from + my_slice.shape[-2]

        x_from = pos[2]
        x_to = x_from + my_slice.shape[-1]

        if overlaps is not None:
            z = np.array(flatten(overlaps[['Z_from', 'Z_to']].values))
            z = np.unique(z)
            z = np.sort(z)

            xy_weights = squircle_alpha(*frame_shape)

            z_list = list(zip(z, z[1::]))
            try:
                z_list += [(z[-1], None)]
            except IndexError:
                pass

            for zfrom, zto in z_list:
                sums = np.copy(xy_weights)
                condition = (overlaps['Z_from'] <= zfrom)
                if zto is not None:
                    condition = condition & (zto <= (overlaps['Z_to']))
                else:
                    condition = condition & (overlaps['Z_to'] >= z_to)

                for _, row in overlaps[condition].iterrows():
                    width = row.X_to - row.X_from
                    height = row.Y_to - row.Y_from
                    area = width * height
                    if not area:
                        continue

                    w = squircle_alpha(*frame_shape)[:height, :width]

                    if row.X_from == 0:
                        w = np.fliplr(w)
                    if row.Y_from == 0:
                        w = np.flipud(w)

                    xy_index = np.index_exp[row.Y_from:row.Y_to,
                                            row.X_from:row.X_to]
                    sums[xy_index] += w

                if zto is None:
                    slice_index = np.index_exp[zfrom:, ...]
                else:
                    slice_index = np.index_exp[zfrom:zto, ...]

                with np.errstate(invalid='ignore'):
                    factor = xy_weights / sums

                if sl is not None:
                    factor = factor[sl[-2::]]

                my_slice[slice_index] *= factor

        if debug:
            overlay_debug(my_slice, index_dbg, zfrom_dbg)
            my_slice[..., -2:, :] = 65000
            my_slice[..., -2:] = 65000

        output_roi_index = np.index_exp[z_from:z_to, ..., y_from:y_to,
                                        x_from:x_to]
        dest[output_roi_index] += my_slice

        q.task_done()


def overlay_debug(slice, index, z_from):
    cx = slice.shape[-1] // 2
    cy = slice.shape[-2] // 2 + 10
    x = cx - cx // 2
    for xstr in re.findall(r'\d+', index):
        for l in xstr:
            x_end = x + canvas_shape[1]
            try:
                slice[..., cy:cy + canvas_shape[0], x:x_end] = \
                    numbers[int(l)]
            except ValueError:
                break
            x = x_end + canvas_shape[1] // 2
        x = x_end + canvas_shape[1]

    cy += int(canvas_shape[0] * 1.4)
    for f in range(0, slice.shape[0]):
        x = cx
        xstr = str(z_from + f)
        for l in xstr:
            x_end = x + canvas_shape[1]
            ie = np.index_exp[f, ..., cy:cy + canvas_shape[0], x:x_end]
            if len(slice.shape) <= 3:
                ie = tuple(list(ie)[1::])
            try:
                slice[ie] = numbers[int(l)]
            except ValueError:
                break
            x = x_end + canvas_shape[1] // 2
