import sys
import os.path
import argparse

import yaml

from ..version import full_version

from .fuse_runner import FuseRunner
from ..filematrix import FileMatrix


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

    parser.add_argument('-o', type=str, default='fused.tiff',
                        dest='output_filename', help='output file name')

    group = parser.add_argument_group('absolute positions')
    me_group = group.add_mutually_exclusive_group()
    me_group.add_argument('-a', action='store_true', dest='compute_average',
                          help='instead of maximum score, take the average '
                               'result weighted by the score')

    me_group.add_argument('-n', action='store_true',
                          dest='use_nominal_positions',
                          help='use nominal stage positions')

    me_group.add_argument('-f', action='store_true',
                          dest='force_recomputation',
                          help='force recomputation of absolute positions')

    parser.add_argument('-d', dest='debug', action='store_true',
                        help='overlay debug info')

    group = parser.add_argument_group(
        'pixel size', 'If specified, the corresponding options can be '
                      'expressed in your custom units.')
    group.add_argument('--px-size-xy', type=float,
                       help='pixel size in the (X, Y) plane')
    group.add_argument('--px-size-z', type=float,
                       help='pixel size in the Z direction')

    parser.add_argument('--zmin', type=float, default=0)
    parser.add_argument('--zmax', type=float, default=None,
                        help='noninclusive')

    args = parser.parse_args()

    return args


def main():
    old_options = None
    args = parse_args()

    # replace None args with values found in yml file
    if os.path.isfile(args.input_file):
        with open(args.input_file, 'r') as f:
            y = yaml.load(f)
        old_options = y['fuse_runner_options']
        keys = ['px_size_z', 'px_size_xy']
        for k in keys:
            if getattr(args, k) is None:
                try:
                    setattr(args, k, y['xcorr-options'][k])
                except KeyError:
                    pass

    if args.px_size_z is None:
            args.px_size_z = 1
    if args.px_size_xy is None:
            args.px_size_xy = 1

    args.zmin = int(round(args.zmin / args.px_size_z))
    if args.zmax is not None:
        args.zmax = int(round(args.zmax / args.px_size_z))

    # checks
    if os.path.isdir(args.input_file):
        temp = os.path.join(args.input_file, 'stitch.yml')
        if not os.path.exists(temp):
            if not args.use_nominal_positions:
                sys.exit("No stitch file specified or found. Please specify "
                         "input file or run with -n.")
        else:
            args.input_file = temp

    # =========================================================================

    # init FileMatrix
    fm = FileMatrix(args.input_file)
    if old_options and old_options['compute_average'] != args.compute_average \
            or args.force_recomputation or args.use_nominal_positions:
        fm.clear_absolute_positions()
    # TODO: set fm options

    # init FuseRunner
    fr = FuseRunner(fm)

    keys = ['zmin', 'zmax', 'output_filename', 'debug', 'compute_average',
            'use_nominal_positions']
    for k in keys:
        setattr(fr, k, getattr(args, k))

    fr.run()

    if os.path.isfile(args.input_file):
        fm.save_to_yaml(args.input_file, 'update')

        with open(args.input_file, 'r') as f:
            y = yaml.load(f)
        fr_options = {}
        keys = ['compute_average']
        for k in keys:
            fr_options[k] = getattr(args, k)
        y['fuse_runner_options'] = fr_options

        with open(args.input_file, 'w') as f:
            yaml.dump(y, f, default_flow_style=False)


if __name__ == '__main__':
    main()
