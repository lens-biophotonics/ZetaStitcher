import sys
import os.path
import logging
import argparse

import yaml
import coloredlogs

from ..version import __version__

from . import absolute_positions
from .fuse_runner import FuseRunner
from ..filematrix import FileMatrix
from .xcorr_filematrix import XcorrFileMatrix
from .global_optimization import absolute_position_global_optimization

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', fmt='%(levelname)s [%(name)s]: %(message)s')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fuse stitched tiles in a folder.',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>\n'
               'Version: {}'.format(__version__),
        formatter_class=CustomFormatter)

    parser.add_argument(
        'yml_file',
        help='.yml file produced by stitch align. It will also be used for '
             'saving absolute coordinates. If a directory is specified'
             'instead of a file, uses a file named "stitch.yml" if present. '
             'If the file does not exist, it will be created (only where '
             'applicable: see option -s).')

    group = parser.add_argument_group('output')
    group.add_argument('-o', type=str, dest='output_filename',
                       help='output file name. If not specified, no tiff '
                            'output is produced, only absoulute coordinates '
                            'are computed.')

    group.add_argument('-d', dest='debug', action='store_true',
                       help='overlay debug info')

    group.add_argument('-c', type=int, default=-1, dest='channel',
                       help='channel')

    group.add_argument('--zmin', type=float, default=0)
    group.add_argument('--zmax', type=float, default=None, help='noninclusive')

    group = parser.add_argument_group(
        'absolute positions', 'by default, absolute positions are computed by '
                              'taking the maximum score in cross correlations')
    me_group = group.add_mutually_exclusive_group()
    me_group.add_argument('-m', dest='abs_mode', action='store_const',
                          const='maximum_score',
                          help='take the maximum score in cross correlations '
                               '(default)')

    me_group.add_argument('-a', dest='abs_mode', action='store_const',
                          const='weighted_average',
                          help='take the average result weighted by the score')

    me_group.add_argument('-s', dest='abs_mode', action='store_const',
                          const='nominal_positions',
                          help='use nominal stage positions')

    group.add_argument('-f', action='store_true', default=False,
                       dest='force_recomputation',
                       help='force recomputation of absolute positions')

    group.add_argument('--no-global', action='store_true',
                       help='do not perform global optimization (where '
                            'applicable)')

    group = parser.add_argument_group('tile ordering (option -s only)')
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

    if os.path.isdir(args.yml_file):
        temp = os.path.join(args.yml_file, 'stitch.yml')
        if os.path.exists(temp):
            args.yml_file = temp

    asc_keys = ['ascending_tiles_x', 'ascending_tiles_y']
    for k in asc_keys:
        setattr(args, k, None)

    # replace None args with values found in yml file
    old_abs_mode = None
    if os.path.isfile(args.yml_file):
        with open(args.yml_file, 'r') as f:
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

    using_old_abs_mode = False
    if args.abs_mode is None:
        args.abs_mode = old_abs_mode
        using_old_abs_mode = True
    if args.abs_mode is None:
        args.abs_mode = 'maximum_score'

    if not os.path.isfile(args.yml_file):
        if args.abs_mode != 'nominal_positions':
            logger.error("No stitch file specified or found. Please specify "
                         "input file or run with -s.")
            sys.exit(1)

    for k in ['x', 'y']:
        temp_k = 'ascending_tiles_' + k
        if getattr(args, temp_k, None) is None:
            setattr(args, temp_k, not getattr(args, 'invert_' + k))

    attrs = ['px_size_z', 'px_size_xy']
    for a in attrs:
        if getattr(args, a, None) is None:
            if args.abs_mode == 'nominal_positions' and not using_old_abs_mode:
                logger.error(
                    "px sizes need to be specified when using option -s")
                sys.exit(1)
            else:
                setattr(args, a, 1)

    args.zmin = int(round(args.zmin / args.px_size_z))
    if args.zmax is not None:
        args.zmax = int(round(args.zmax / args.px_size_z))

    # =========================================================================
    # init FileMatrix
    # =========================================================================
    fm = FileMatrix(args.yml_file,
                    ascending_tiles_x=args.ascending_tiles_x,
                    ascending_tiles_y=args.ascending_tiles_y)

    if args.force_recomputation or args.abs_mode != old_abs_mode:
        fm.clear_absolute_positions()

    cols = fm.data_frame.columns
    logger.info('absolute positions mode: {}'.format(args.abs_mode))
    if 'Xs' in cols and 'Ys' in cols and 'Zs' in cols:
        logger.info('using absolute positions from {}'.format(args.yml_file))
    elif args.abs_mode == 'nominal_positions':
        fm.compute_nominal_positions(args.px_size_z, args.px_size_xy)
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
    if args.output_filename is not None:
        fr = FuseRunner(fm)

        keys = ['zmin', 'zmax', 'output_filename', 'debug', 'channel']

        for k in keys:
            setattr(fr, k, getattr(args, k))

        logger.info("invert X: {}, invert Y: {}".format(
            not args.ascending_tiles_x, not args.ascending_tiles_y))
        logger.info("output shape: {}".format(fr.output_shape))
        logger.info("voxel size (ZYX): {} * {} * {}".format(
            args.px_size_z, args.px_size_xy, args.px_size_xy))

        fr.run()
    else:
        logger.warning("No output file specified.")

    if os.path.isdir(args.yml_file):
        args.yml_file = os.path.join(args.yml_file, 'stitch.yml')
        fm.save_to_yaml(args.yml_file, 'w')
    else:
        fm.save_to_yaml(args.yml_file, 'update')

    with open(args.yml_file, 'r') as f:
        y = yaml.load(f)
    fr_options = {}
    keys = ['abs_mode', 'px_size_xy', 'px_size_z', 'ascending_tiles_x',
            'ascending_tiles_y']
    for k in keys:
        fr_options[k] = getattr(args, k)
    y['fuser-options'] = fr_options

    with open(args.yml_file, 'w') as f:
        yaml.dump(y, f, default_flow_style=False)


if __name__ == '__main__':
    main()
