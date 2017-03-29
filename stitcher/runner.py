import sys
import os.path
import argparse

import colored
from colored import stylize

from .core import stitch
from .filematrix import FileMatrix


def main():
    parser = argparse.ArgumentParser(
        description='''
Stitch tiles in a folder.

The following naming conventions are used:
* Z is the direction along the stack height,
* (X, Y) is the frame plane,
* Y is the direction along which frames are supposed to overlap,
* X is the direction orthogonal to Y in the frame plane (X, Y).

Unless otherwise stated, all values are expected in px.
''',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('input_folder', help='input folder')

    group = parser.add_argument_group('maximum shifts')
    group.add_argument('--mz', type=int, default=20, dest='max_dz',
                       help='maximum allowed shift along Z')

    group.add_argument('--my', type=int, default=150, dest='max_dy',
                       help='maximum allowed shift along Y (the stitching '
                            'axis)')

    group.add_argument('--mx', type=int, default=20, dest='max_dx',
                       help='maximum allowed shift along X (lateral shift)')

    group = parser.add_argument_group('overlaps')
    group.add_argument('--overlap-h', type=int, default=600, metavar='OH',
                       help='overlap along the horizontal axis')

    group.add_argument('--overlap-v', type=int, default=600, metavar='OV',
                       help='overlap along the vertical axis')

    group = parser.add_argument_group(
        'multiple sampling along Z',
        description='Measure the optimal shift at different heights around '
                    'the center of the stack')
    group1 = group.add_mutually_exclusive_group()
    group1.add_argument('-a', action='store_true',
                        help='take the average result weighted by the score',
                        dest='average')

    group1.add_argument('-m', action='store_true',
                        help='take the result with the maximum score',
                        dest='maximum')

    group.add_argument('--z-samples', type=int, default=1, metavar='ZSAMP',
                       help='number of samples to take along Z')

    group.add_argument('--stride', type=int, default=200,
                       help='stride used for multiple Z sampling')

    arg = parser.parse_args(sys.argv[1:])

    fm = FileMatrix(arg.input_folder)
    fm.ascending_tiles_X = True
    fm.ascending_tiles_Y = False

    group_generators = [fm.tiles_along_X, fm.tiles_along_Y]
    stitch_axis = [2, 1]
    overlap = [arg.overlap_h, arg.overlap_v]

    for group_generator, axis, overlap in zip(
            group_generators, stitch_axis, overlap):
        for group in group_generator:

            tile_generator = group.itertuples()

            atile = next(tile_generator)

            for btile in tile_generator:
                temp = map(os.path.basename,
                           (atile.filename, btile.filename))
                temp = list(temp)
                print(
                    stylize('stitching {} {}, axis = {}'.format(*temp, axis),
                            colored.fg('magenta')))
                central_frame = atile.nfrms // 2
                start_frame = (central_frame
                               - (arg.z_samples // 2 * arg.stride)
                               + (0 if arg.z_samples % 2 else arg.stride // 2))
                for i in range(0, arg.z_samples):
                    z_frame = start_frame + i * arg.stride
                    stitch(
                        atile.filename, btile.filename, z_frame, axis=axis,
                        overlap=overlap, max_shift_z=arg.max_dz,
                        max_shift_y=arg.max_dy, max_shift_x=arg.max_dx)

                atile = btile

            print('===================================')


if __name__ == '__main__':
    main()
