import sys
import argparse

import numpy as np

from .normxcorr import normxcorr2_fftw
from .inputfile import InputFile

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

    a.channel = 1
    b.channel = 1

    z_min = z_frame - max_shift_z
    z_max = z_frame + max_shift_z + 1

    alayer = a.layer(z_min, z_max)
    if axis == 2:
        alayer = np.rot90(alayer, axes=(1, 2))
    alayer = alayer[:, -overlap:, :]

    blayer = b.layer_idx(z_frame)
    if axis == 2:
        blayer = np.rot90(blayer, axes=(1, 2))
    blayer = blayer[:, 0:overlap, :]

    half_max_shift_x = max_shift_x // 2

    blayer = blayer[:, :-max_shift_y, half_max_shift_x:-half_max_shift_x]

    xcorr = normxcorr2_fftw(alayer, blayer)

    shift = list(np.unravel_index(np.argmax(xcorr), xcorr.shape))
    score = xcorr[tuple(shift)]

    print('shift: ' + str(shift))
    shift[0] -= max_shift_z
    shift[1] = overlap - shift[1]
    shift[2] -= half_max_shift_x

    print('max @ {}: {}, score: {:.3}'.format(z_frame, shift, score))

    return shift + [score]


def main():
    parser = argparse.ArgumentParser(
        description='Stitch a pair of files',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_file1', help='input file (.dcimg)')
    parser.add_argument('input_file2', help='input file (.dcimg)')

    parser.add_argument('-f', type=int, help='frame', dest='frame',
                        required=True)
    parser.add_argument('-a', type=int, help='axis (1=Y, 2=X)', dest='axis',
                        choices=(1, 2), default=1)
    parser.add_argument('--overlap', type=int, help='overlap', default=600)

    args = parser.parse_args(sys.argv[1:])

    stitch(args.input_file1, args.input_file2, z_frame=args.frame,
           overlap=args.overlap, axis=args.axis)


if __name__ == '__main__':
    main()
