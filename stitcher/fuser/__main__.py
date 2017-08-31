import sys
import os.path
import argparse

import yaml

from ..version import full_version

from . import absolute_positions
from .fuse_runner import FuseRunner
from ..filematrix import FileMatrix
from .xcorr_filematrix import XcorrFileMatrix
from .global_optimization import absolute_position_global_optimization


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fuse stitched tiles in a folder.',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>\n'
               'Version: {}'.format(full_version),
        formatter_class=CustomFormatter)

    parser.add_argument('input_file', help='input file (.yml) or folder')

    group = parser.add_argument_group('output')
    group.add_argument('-o', type=str, default='fused.tiff',
                       dest='output_filename', help='output file name')

    group.add_argument('-d', dest='debug', action='store_true',
                       help='overlay debug info')

    group.add_argument('--zmin', type=float, default=0)
    group.add_argument('--zmax', type=float, default=None, help='noninclusive')

    group = parser.add_argument_group('absolute positions')
    me_group = group.add_mutually_exclusive_group()
    me_group.add_argument('-a', dest='abs_mode', action='store_const',
                          const='weighted_average',
                          help='instead of maximum score, take the average '
                               'result weighted by the score')

    me_group.add_argument('-n', dest='abs_mode', action='store_const',
                          const='nominal_positions',
                          help='use nominal stage positions')

    group.add_argument('-f', action='store_true', default=False,
                       dest='force_recomputation',
                       help='force recomputation of absolute positions')

    group.add_argument('--no-global', action='store_true',
                       help='do not perform global optimization (where '
                            'applicable)')

    group = parser.add_argument_group('tile ordering (option -n only)')
    group.add_argument('--iX', action='store_true', dest='invert_x',
                       help='invert tile ordering along X')
    group.add_argument('--iY', action='store_true', dest='invert_y',
                       help='invert tile ordering along Y')

    group = parser.add_argument_group(
        'pixel size', 'If specified, the corresponding options can be '
                      'expressed in your custom units.')
    group.add_argument('--px-size-xy', type=float,
                       help='pixel size in the (X, Y) plane')
    group.add_argument('--px-size-z', type=float,
                       help='pixel size in the Z direction')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if os.path.isdir(args.input_file):
        temp = os.path.join(args.input_file, 'stitch.yml')
        if os.path.exists(temp):
            args.input_file = temp

    asc_keys = ['ascending_tiles_x', 'ascending_tiles_y']
    for k in asc_keys:
        setattr(args, k, None)

    # replace None args with values found in yml file
    old_abs_mode = None
    if os.path.isfile(args.input_file):
        with open(args.input_file, 'r') as f:
            y = yaml.load(f)
            try:
                old_abs_mode = y['fuser-options']['abs_mode']
            except KeyError:
                pass
        keys = ['px_size_z', 'px_size_xy'] + asc_keys
        for k in keys:
            if getattr(args, k) is None:
                try:
                    setattr(args, k, y['xcorr-options'][k])
                except KeyError:
                    pass

    if args.abs_mode is None:
        args.abs_mode = old_abs_mode
    if args.abs_mode is None:
        args.abs_mode = 'maximum_score'

    if not os.path.isfile(args.input_file):
        if args.abs_mode != 'nominal_positions':
            sys.exit("No stitch file specified or found. Please specify input "
                     "file or run with -n.")

    for k in ['x', 'y']:
        temp_k = 'ascending_tiles_' + k
        if getattr(args, temp_k, None) is None:
            setattr(args, temp_k, not getattr(args, 'invert_' + k))

    attrs = ['px_size_z', 'px_size_xy']
    for a in attrs:
        if getattr(args, a, None) is None:
            if args.abs_mode == 'nominal_positions':
                sys.exit("px sizes need to be specified when using option -n")
            else:
                setattr(args, a, 1)

    args.zmin = int(round(args.zmin / args.px_size_z))
    if args.zmax is not None:
        args.zmax = int(round(args.zmax / args.px_size_z))

    # =========================================================================
    # init FileMatrix
    # =========================================================================
    fm = FileMatrix(args.input_file,
                    ascending_tiles_x=args.ascending_tiles_x,
                    ascending_tiles_y=args.ascending_tiles_y)

    if args.force_recomputation or args.abs_mode != old_abs_mode:
        fm.clear_absolute_positions()

    cols = fm.data_frame.columns
    if args.abs_mode == 'nominal_positions':
        fm.compute_nominal_positions(args.px_size_z, args.px_size_xy)
    elif 'Xs' in cols and 'Ys' in cols and 'Zs' in cols:
        pass
    else:
        xcorr_fm = XcorrFileMatrix()
        xcorr_fm.load_yaml(fm.input_path)
        compute_average = \
            True if args.abs_mode == 'weighted_average' else False
        xcorr_fm.aggregate_results(compute_average=compute_average)

        sdf = xcorr_fm.stitch_data_frame

        absolute_positions.compute_shift_vectors(fm.data_frame, sdf)
        absolute_positions.compute_initial_guess(fm.data_frame, sdf)

        if not args.no_global:
            absolute_position_global_optimization(fm.data_frame, sdf,
                                                  xcorr_fm.xcorr_options)

    # =========================================================================
    # init FuseRunner
    # =========================================================================
    fr = FuseRunner(fm)

    keys = ['zmin', 'zmax', 'output_filename', 'debug']

    for k in keys:
        setattr(fr, k, getattr(args, k))

    fr.run()

    if os.path.isfile(args.input_file):
        fm.save_to_yaml(args.input_file, 'update')

        with open(args.input_file, 'r') as f:
            y = yaml.load(f)
        fr_options = {}
        keys = ['abs_mode']
        for k in keys:
            fr_options[k] = getattr(args, k)
        y['fuser-options'] = fr_options

        with open(args.input_file, 'w') as f:
            yaml.dump(y, f, default_flow_style=False)


if __name__ == '__main__':
    main()
