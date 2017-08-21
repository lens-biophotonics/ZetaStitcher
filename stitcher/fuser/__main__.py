import os.path
import argparse

import yaml

from ..version import full_version

from .fuse_runner import FuseRunner


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

    parser.add_argument('-a', action='store_true', dest='compute_average',
                        help='instead of maximum score, take the average '
                             'result weighted by the score')

    parser.add_argument('-d', dest='debug', action='store_true',
                        help='overlay debug info')

    parser.add_argument('--px-size-z', type=float, default=1,
                        help='pixel size in the Z direction. If specified, '
                             'the corresponding options can be expressed in '
                             'your custom untis.')

    parser.add_argument('--zmin', type=float, default=0)
    parser.add_argument('--zmax', type=float, default=None,
                        help='noninclusive')

    args = parser.parse_args()

    args.zmin = int(round(args.zmin / args.px_size_z))
    if args.zmax is not None:
        args.zmax = int(round(args.zmax / args.px_size_z))

    return args


def main():
    arg = parse_args()

    if os.path.isdir(arg.input_file):
        input_file = os.path.join(arg.input_file, 'stitch.yml')
    else:
        input_file = arg.input_file

    with open(input_file, 'r') as f:
        y = yaml.load(f)

    try:
        old_options = y['fuse_runner_options']
    except KeyError:
        old_options = None

    fr = FuseRunner(input_file)

    keys = ['zmin', 'zmax', 'output_filename', 'debug', 'compute_average']
    for k in keys:
        setattr(fr, k, getattr(arg, k))

    fr._load_df()
    if old_options and old_options['compute_average'] != arg.compute_average:
        fr.clear_absolute_positions()

    fr.run()

    with open(input_file, 'r') as f:
        y = yaml.load(f)
    fr_options = {}
    keys = ['compute_average']
    for k in keys:
        fr_options[k] = getattr(arg, k)
    y['fuse_runner_options'] = fr_options

    with open(input_file, 'w') as f:
        yaml.dump(y, f, default_flow_style=False)


if __name__ == '__main__':
    main()
