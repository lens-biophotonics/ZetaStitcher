import logging
import argparse
from pathlib import Path

import yaml
import coloredlogs

import numpy as np

from zetastitcher import VirtualFusedVolume, FileMatrix
from zetastitcher.align.xcorr_filematrix import XcorrFileMatrix


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downscale coordinates in a stitch.yml file.',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=CustomFormatter)

    parser.add_argument('input_file', help='input yml file')
    parser.add_argument('output_file', help='output yml file')

    parser.add_argument('--xy-divide-by', type=float)
    parser.add_argument('--z-divide-by', type=float)
    parser.add_argument('--ext', type=str, default='.tiff',
                        help='replace tile extension (e.g. .tiff)')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    fm = FileMatrix(args.input_file)
    df = fm.data_frame

    df['Xs'] /= args.xy_divide_by
    df['Ys'] /= args.xy_divide_by
    df['Zs'] /= args.z_divide_by

    df['xsize'] /= args.xy_divide_by
    df['ysize'] /= args.xy_divide_by
    df['nfrms'] /= args.z_divide_by

    keys = ['Zs', 'Ys', 'Xs', 'xsize', 'ysize', 'nfrms']
    df[keys] = df[keys].apply(np.round).astype(int)

    df['Xs_end'] = df['Xs'] + df['xsize']
    df['Ys_end'] = df['Ys'] + df['ysize']
    df['Zs_end'] = df['Zs'] + df['nfrms']

    xcorr_fm = XcorrFileMatrix.from_yaml(fm.input_path)

    xcorr_fm.xcorr_options['px_size_xy'] *= args.xy_divide_by
    xcorr_fm.xcorr_options['px_size_z'] *= args.z_divide_by
    xcorr_fm.xcorr_options['max_dx'] /= args.xy_divide_by
    xcorr_fm.xcorr_options['max_dy'] /= args.xy_divide_by
    xcorr_fm.xcorr_options['max_dz'] /= args.z_divide_by
    xcorr_fm.xcorr_options['overlap_h'] /= args.xy_divide_by
    xcorr_fm.xcorr_options['overlap_v'] /= args.xy_divide_by
    xcorr_fm.xcorr_options['z_stride'] /= args.z_divide_by

    fm.save_to_yaml(args.output_file, 'w')

    with open(args.input_file, 'r') as f:
        y = yaml.load(f)

    with open(args.output_file, 'a') as f:
        yaml.dump(
            {
                'fuser-options': y['fuser-options'],
                'xcorr-options': xcorr_fm.xcorr_options,
            }, f, default_flow_style=False)

    with open(args.output_file, 'r') as f:
        file_content = f.read()

    ext = args.ext
    if not ext.startswith('.'):
        ext = '.' + ext
    old_ext = Path(df.index[0]).suffix
    if old_ext:
        file_content = file_content.replace(old_ext, ext)

    with open(args.output_file, 'w') as f:
        f.write(file_content)

    vfv = VirtualFusedVolume(args.output_file)
    logger.info(f'final shape: {vfv.shape}')


if __name__ == '__main__':
    main()
