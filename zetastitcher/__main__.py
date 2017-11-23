import sys
import argparse

import numpy as np
import skimage.external.tifffile as tiff

from .normxcorr import normxcorr2_fftw
from .inputfile import InputFile


def to_dtype(x, dtype):
    x = np.rint(x) if np.issubdtype(dtype, np.integer) else x
    return x.astype(dtype, copy=False)


def stitch(aname, bname, z_frame, axis, overlap, max_shift_z=20,
           max_shift_y=150, max_shift_x=20):
    """Compute optimal shift between adjacent tiles.

    Two 3D tiles are compared at the specified frame index to find their best
    alignment. The following naming conventions are used:

    * Z is the direction along the stack height,
    * (X, Y) is the frame plane,
    * Y is the direction along which frames are supposed to overlap,
    * X is the direction orthogonal to Y in the frame plane (X, Y).

    Parameters
    ----------
    aname : str
        Input file name.
    bname : str
        Input file name.
    z_frame : int
        Index of frame used for alignment.
    axis : int
        Perform stitching along this axis. Following the convention above, this
        axis will be called Y. The following values can be used:

        1. stitch vertically
        2. stitch horizontally
    overlap : int
        Overlap height in px along the stitching axis (Y).
    max_shift_z : int
        Maximum allowed shift in px along the stack height (Z).
    max_shift_y : int
        Maximum allowed shift in px *of the overlapping area* along (Y).
    max_shift_x : int
        Maximum allowed lateral shift in px.

    Returns
    -------
    tuple
        Optimal shifts and overlap score computed by means of normalized
        cross correlation: [`dz`, `dy`, `dx`, `score`].
    """
    a = InputFile(aname)
    b = InputFile(bname)

    a.channel = -2
    b.channel = -2

    z_min = z_frame - max_shift_z
    z_max = z_frame + max_shift_z + 1

    aslice = a.zslice(z_min, z_max)
    if axis == 2:
        aslice = np.rot90(aslice, axes=(-1, -2))
    a_roi = aslice[..., -(overlap + max_shift_y):, :]

    bframe = b.zslice_idx(z_frame)
    if axis == 2:
        bframe = np.rot90(bframe, axes=(-1, -2))
    b_roi = bframe[..., :overlap - max_shift_y, max_shift_x:-max_shift_x]

    tiff.imsave('aslice.tiff', a_roi.astype(np.float32))
    tiff.imsave('bframe.tiff', b_roi.astype(np.float32))

    xcorr = normxcorr2_fftw(a_roi, b_roi)
    tiff.imsave('xcorr.tiff', xcorr.astype(np.float32))

    shift = list(np.unravel_index(np.argmax(xcorr), xcorr.shape))
    score = xcorr[tuple(shift)]

    print('shift: ' + str(shift))
    z_a = shift[0]
    shift[0] -= max_shift_z
    shift[1] = overlap + max_shift_y - shift[1]
    shift[2] -= max_shift_x

    print('max @ {}: {}, score: {:.3}'.format(z_frame, shift, score))

    a_roi = aslice[z_a, -shift[1]:, ...]
    b_roi = np.squeeze(bframe)[0:shift[1], ...]

    dx = shift[2]
    if dx > 0:
        a_roi = np.pad(a_roi, ((0, 0), (0, dx)), mode='constant')
        b_roi = np.pad(b_roi, ((0, 0), (dx, 0)), mode='constant')
    else:
        dx *= -1
        a_roi = np.pad(a_roi, ((0, 0), (dx, 0)), mode='constant')
        b_roi = np.pad(b_roi, ((0, 0), (0, dx)), mode='constant')

    rad = np.linspace(0.0, np.pi, a_roi.shape[0], dtype=np.float32)
    alpha = (np.cos(rad) + 1) / 2
    alpha = alpha[:, np.newaxis]

    fused = to_dtype(a_roi * alpha + b_roi * (1 - alpha), aslice.dtype)
    if axis == 2:
        fused = np.rot90(fused, axes=(-1, -2), k=3)

    tiff.imsave('fused.tiff', fused.astype(np.float32))
    return shift + [score]


def main():
    parser = argparse.ArgumentParser(
        description='Stitch a pair of files',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_file1', help='input file (.dcimg)')
    parser.add_argument('input_file2', help='input file (.dcimg)')

    parser.add_argument('-f', type=int, dest='frame', required=True,
                        help='frame of input_file2')
    parser.add_argument('-a', type=int, help='axis (1=Y, 2=X)', dest='axis',
                        choices=(1, 2), default=1)
    parser.add_argument('--overlap', type=int, default=100, help='overlap')
    parser.add_argument('--Mz', type=int, default=20, help='Max shift along Z')
    parser.add_argument('--My', type=int, default=100,
                        help='Max shift along Y')
    parser.add_argument('--Mx', type=int, default=20,
                        help='Max shift along X (lateral)')

    args = parser.parse_args(sys.argv[1:])

    stitch(args.input_file1, args.input_file2, z_frame=args.frame,
           overlap=args.overlap, axis=args.axis, max_shift_z=args.Mz,
           max_shift_y=args.My, max_shift_x=args.Mx)


if __name__ == '__main__':
    main()
