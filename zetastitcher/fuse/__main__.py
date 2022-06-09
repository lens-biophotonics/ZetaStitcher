import sys
import os.path
import logging
import argparse

import yaml
import humanize
import coloredlogs

from ..version import __version__

import numpy as np

from . import absolute_positions
from .fuse_runner import FuseRunner
from zetastitcher.align.filematrix import FileMatrix
from zetastitcher.align.xcorr_filematrix import XcorrFileMatrix

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


ABS_MODE_NOMINAL_POSITIONS = 'nominal_positions'
ABS_MODE_MAXIMUM_SCORE = 'maximum_score'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fuse stitched tiles in a folder.',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>\n'
               'Version: {}'.format(__version__),
        formatter_class=CustomFormatter)

    parser.add_argument(
        'yml_file',
        help='.yml file produced by stitch align. It will also be used for '
             'saving absolute coordinates, unless option -w is specified. If a '
             'directory is specified instead of a file, uses a file named '
             '"stitch.yml"')

    group = parser.add_argument_group('output')
    group.add_argument('-o', type=str, dest='output_filename',
                       help='output file name. If not specified, no tiff '
                            'output is produced, only absolute coordinates '
                            'are computed.')

    group.add_argument('-w', type=str, dest='yml_out_file',
                       help='save data to a different .yml file')

    group.add_argument('-d', dest='debug', action='store_true',
                       help='overlay debug info')

    group.add_argument('-c', '--ch', type=int, dest='channel', help='channel')

    group.add_argument('--compression', type=str, default='zlib', help='int or string')

    group.add_argument('--downsample-xy', metavar='S', type=int, required=False,
                       help='downsample XY plane by factor S')

    group.add_argument('--zmin', type=float, default=0,
                       help='start frame (in your units)')
    me_group = group.add_mutually_exclusive_group()
    me_group.add_argument('--zmax', type=float,
                          help='end frame (noninclusive, in your units)')
    me_group.add_argument('--nz', type=int, help='number of z frames')

    group.add_argument('-s', dest='abs_mode', action='store_const',
                       const=ABS_MODE_NOMINAL_POSITIONS,
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


def preprocess_and_check_args(args):
    px_attrs = ['px_size_z', 'px_size_xy']
    if args.abs_mode == ABS_MODE_NOMINAL_POSITIONS:
        for a in px_attrs:
            if getattr(args, a, None) is None:
                logger.error(
                    "px sizes need to be specified when using option -s")
                sys.exit(1)

        for k in ['x', 'y']:
            setattr(args, 'ascending_tiles_' + k, not getattr(args,
                                                              'invert_' + k))
    else:
        if os.path.isdir(args.yml_file):
            args.yml_file = os.path.join(args.yml_file, 'stitch.yml')
        if not os.path.isfile(args.yml_file):
            logger.error(
                "No stitch file specified or found. Please specify input file "
                "or run with -s.")
            sys.exit(1)

        old_abs_mode = None
        # replace None args with values found in yml file
        if os.path.isfile(args.yml_file):
            with open(args.yml_file, 'r') as f:
                y = yaml.safe_load(f)
                try:
                    old_abs_mode = y['fuser-options']['abs_mode']
                except KeyError:
                    pass
            keys = ['px_size_z', 'px_size_xy', 'ascending_tiles_x',
                    'ascending_tiles_y']
            for k in keys:
                try:
                    attr = getattr(args, k)
                except AttributeError:
                    attr = None
                if attr is None:
                    try:
                        setattr(args, k, y['xcorr-options'][k])
                    except KeyError:
                        pass

        if args.abs_mode is None:
            args.abs_mode = old_abs_mode
        if args.abs_mode is None:
            args.abs_mode = ABS_MODE_MAXIMUM_SCORE

        if args.abs_mode != old_abs_mode:
            args.force_recomputation = True
        for a in px_attrs:
            if a not in args:
                setattr(args, a, 1)

    args.zmin = int(round(args.zmin / args.px_size_z))
    if args.zmax is not None:
        args.zmax = int(round(args.zmax / args.px_size_z))
    elif args.nz is not None:
        args.zmax = args.zmin + args.nz

    try:
        args.compression = int(args.compression)
    except ValueError:
        pass


def compute_absolute_positions(args, fm):
    xcorr_fm = XcorrFileMatrix.from_yaml(fm.input_path)
    xcorr_fm.aggregate_results()

    sdf = xcorr_fm.stitch_data_frame

    absolute_positions.compute_shift_vectors(fm.data_frame, sdf)

    if not args.no_global:
        absolute_positions.global_optimization(fm.data_frame, xcorr_fm)


def append_fuser_options_to_yaml(yml_out_file, args):
    with open(yml_out_file, 'r') as f:
        y = yaml.safe_load(f)
    fr_options = {}
    keys = ['px_size_xy', 'px_size_z']
    if args.abs_mode == ABS_MODE_NOMINAL_POSITIONS:
        keys += ['ascending_tiles_x', 'ascending_tiles_y']
    else:
        keys += ['abs_mode']
    for k in keys:
        fr_options[k] = getattr(args, k)
    y['fuser-options'] = fr_options

    with open(yml_out_file, 'w') as f:
        yaml.dump(y, f, default_flow_style=False)


def main():
    args = parse_args()
    preprocess_and_check_args(args)

    logger.info("invert X: {}, invert Y: {}".format(
        not args.ascending_tiles_x, not args.ascending_tiles_y))
    logger.info("voxel size (ZYX): {} * {} * {}".format(
        args.px_size_z, args.px_size_xy, args.px_size_xy))

    # =========================================================================
    # init FileMatrix
    # =========================================================================
    fm = FileMatrix(args.yml_file,
                    ascending_tiles_x=args.ascending_tiles_x,
                    ascending_tiles_y=args.ascending_tiles_y)

    if args.force_recomputation:
        fm.clear_absolute_positions()

    cols = fm.data_frame.columns

    if 'Xs' in cols and 'Ys' in cols and 'Zs' in cols:
        logger.info('using absolute positions from {}'.format(args.yml_file))
    else:
        if args.abs_mode == ABS_MODE_NOMINAL_POSITIONS:
            logger.info('absolute positions mode: using nominal stage coordinates')
            fm.compute_nominal_positions(args.px_size_z, args.px_size_xy)
            yml_out_file = args.yml_out_file
        else:
            if not os.access(args.yml_file, os.W_OK):
                raise ValueError('cannot write to {}'.format(args.yml_file))
            compute_absolute_positions(args, fm)
            yml_out_file = args.yml_file

        if yml_out_file is None:
            logger.warning('No .yml output file specified (-w option)')
        else:
            if os.path.isfile(yml_out_file):
                fm.save_to_yaml(yml_out_file, 'update')
            else:
                fm.save_to_yaml(yml_out_file, 'w')
            append_fuser_options_to_yaml(yml_out_file, args)

    # =========================================================================
    # init FuseRunner
    # =========================================================================
    fr = FuseRunner(fm)
    bytes_human = humanize.naturalsize(np.prod(fr.output_shape) * fr.dtype.itemsize, binary=True)
    logger.info(f'fused shape, whole volume: {fr.output_shape}, {bytes_human}')
    if args.output_filename is not None:

        keys = ['zmin', 'zmax', 'output_filename', 'debug', 'channel',
                'compression', 'downsample_xy']

        for k in keys:
            setattr(fr, k, getattr(args, k))

        bytes_human = humanize.naturalsize(np.prod(fr.output_shape) * fr.dtype.itemsize, binary=True)
        logger.info(f'output shape: {fr.output_shape}, {bytes_human}')

        fr.run()
    else:
        logger.warning("No output file specified.")


if __name__ == '__main__':
    main()
