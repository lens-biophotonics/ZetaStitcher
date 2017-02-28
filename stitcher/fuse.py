import numpy as np
import skimage.external.tifffile as tiff

from dcimg import DCIMGFile


def fuse(fname_1, fname_2, shift, zplane, axis=1):
    dx = shift[0]
    dy = shift[1]
    dz = shift[2]

    a = DCIMGFile(fname_1)
    b = DCIMGFile(fname_2)

    aframe = a.frame(zplane, dtype=np.float32)
    bframe = b.frame(zplane + dz, dtype=np.float32)

    if axis == 2:
        aframe = np.rot90(aframe)
        bframe = np.rot90(bframe)

    output_height = aframe.shape[0] + bframe.shape[0] - dy
    # for the moment consider a and b to have same width
    output_width = aframe.shape[1] - abs(dx)

    aframe_roi = aframe[-dy:, :]
    bframe_roi = bframe[0:dy, :]

    ax_min = 0
    if dx > 0:
        ax_min = dx
    ax_max = ax_min + output_width

    bx_min = 0
    bx_max = output_width

    if dx < 0:
        bx_min = -dx
        bx_max += abs(dx)


    aframe_roi = aframe_roi[:, ax_min:ax_max]
    bframe_roi = bframe_roi[:, bx_min:bx_max]

    plateau_size = dy // 10

    plateau_size = 0

    rad = np.linspace(0.0, np.pi, dy - 2 * plateau_size, dtype=np.float32)

    rad = np.append(np.zeros(plateau_size, dtype=np.float32), rad)
    rad = np.append(rad, np.full(plateau_size, np.pi, dtype=np.float32))

    alpha = (np.cos(rad) + 1) / 2
    alpha = np.tile(alpha, [output_width])
    alpha = np.reshape(alpha, [output_width, dy])
    alpha = np.transpose(alpha)

    fused = aframe_roi * alpha + bframe_roi * (1 - alpha)

    output = np.zeros((output_height, output_width), dtype=np.float32)

    output[0:aframe.shape[1] - dy, :] = (
        aframe[0:aframe.shape[1] - dy, ax_min:ax_max])

    output[aframe.shape[1] - dy:aframe.shape[1], :] = fused

    output[aframe.shape[1]:output_height, :] = (
        bframe[dy:bframe.shape[1], bx_min:bx_max])

    a.close()
    b.close()

    return output
