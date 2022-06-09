import os
import sys
import time
import queue
import logging
import argparse
import threading
import concurrent.futures
from datetime import timedelta

import json
import yaml

import coloredlogs

import numpy as np
import pandas as pd

from zetastitcher.io.inputfile import InputFile
from zetastitcher.align.filematrix import FileMatrix
from zetastitcher.align.xcorr_filematrix import XcorrFileMatrix
from zetastitcher.align.dog import align_dog
from zetastitcher.fuse import absolute_positions
from zetastitcher.fuse.__main__ import ABS_MODE_MAXIMUM_SCORE

from zetastitcher.version import __version__


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='''
Stitch tiles in a folder.

The following naming conventions are used:
* Z is the direction along the stack height,
* (X, Y) is the frame plane,
* y is the direction along which frames are supposed to overlap,
* x is the direction orthogonal to y in the frame plane (X, Y).

Unless otherwise stated, all values are expected in px.
    ''',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>\n'
               'Version: {}'.format(__version__),
        formatter_class=CustomFormatter)

    parser.add_argument('input_folder', help='input folder')
    parser.add_argument('-o', type=str, default='stitch.yml', dest='output_file', help='output file')
    parser.add_argument('-c', '--ch', type=int, dest='channel', help='color channel')
    parser.add_argument('-j', type=int, dest='n_of_workers',
                        help='number of parallel jobs (defaults to number of system cores)')
    parser.add_argument('-r', action='store_true', dest='recursive', help='recursively look for files')
    parser.add_argument('-e', action='store_true', dest='equal_shape',
                        help='consider tiles of identical shape (results in slightly faster loading)')

    group = parser.add_argument_group(
        'pixel size', 'If specified, the corresponding options can be '
                      'expressed in your custom units.')
    group.add_argument('--px-size-xy', type=float, default=1,
                       help='pixel size in the (X, Y) plane')
    group.add_argument('--px-size-z', type=float, default=1,
                       help='pixel size in the Z direction')

    group = parser.add_argument_group('maximum shifts')
    group.add_argument('--dz', type=float, required=True, dest='max_dz',
                       help='maximum allowed shift along Z')

    group.add_argument('--dy', type=float, required=True, dest='max_dy',
                       help='maximum allowed shift along y (the stitching '
                            'axis) relative to the nominal overlap')

    group.add_argument('--dx', type=float, required=True, dest='max_dx',
                       help='maximum allowed shift along x (lateral shift)')

    group = parser.add_argument_group('overlaps')
    group.add_argument('--overlap', type=float, help='nominal overlap, H & V')
    group.add_argument('--overlap-h', type=float, metavar='OH',
                       help='nominal overlap along the horizontal axis')

    group.add_argument('--overlap-v', type=float, metavar='OV',
                       help='nominal overlap along the vertical axis')

    group = parser.add_argument_group(
        'multiple sampling along Z',
        description='Measure the optimal shift at different heights around '
                    'the center of the stack')

    group.add_argument('--z-samples', type=int, default=1, metavar='ZSAMP',
                       help='number of samples to take along Z')

    group.add_argument('--z-stride', type=float, default=None,
                       help='stride used for multiple Z sampling')

    group = parser.add_argument_group('tile ordering')
    group.add_argument('--iX', action='store_true', dest='invert_x',
                       help='invert tile ordering along X')
    group.add_argument('--iY', action='store_true', dest='invert_y',
                       help='invert tile ordering along Y')

    args = parser.parse_args()

    if args.overlap is None:
        if args.overlap_h is None or args.overlap_v is None:
            logger.error('Missing overlap options. '
                         'Use --overlap, or both --overlap-h and --overlap-v')
            sys.exit(1)
    elif args.overlap_h is not None or args.overlap_v is not None:
        logger.error('Incompatible options: --overlap and --overlap-h, '
                     '--overlap-v')
        sys.exit(1)
    else:
        setattr(args, 'overlap_h', args.overlap)
        setattr(args, 'overlap_v', args.overlap)

    args.max_dx = int(round(args.max_dx / args.px_size_xy))
    args.max_dy = int(round(args.max_dy / args.px_size_xy))
    args.max_dz = int(round(args.max_dz / args.px_size_z))
    if args.z_stride is not None:
        args.z_stride = int(round(args.z_stride / args.px_size_z))

    args.overlap_v = int(round(args.overlap_v / args.px_size_xy))
    args.overlap_h = int(round(args.overlap_h / args.px_size_xy))

    if args.max_dy > args.overlap_h or args.max_dy > args.overlap_v:
        raise ValueError("dy can't be larger than overlap")

    setattr(args, 'ascending_tiles_x', not args.invert_x)
    setattr(args, 'ascending_tiles_y', not args.invert_y)

    return args


def worker(item, overlap_dict, channel, max_dz, max_dy, max_dx):
    aname = item['aname']
    bname = item['bname']
    z_frame = item['z_frame']
    axis = item['axis']
    overlap = overlap_dict[axis]

    a = InputFile(aname)
    b = InputFile(bname)

    z_min = z_frame - max_dz
    z_max = z_frame + max_dz + 1

    aslice = a.zslice(z_min, z_max, copy=True)
    if a.nchannels > 1:
        if channel is not None:
            aslice = aslice[:, channel]
        else:
            aslice = np.sum(aslice.astype(np.float32), axis=1)
    if axis == 2:
        aslice = np.rot90(aslice, axes=(-1, -2))
    aslice = aslice[..., -(overlap + max_dy):, :]

    bframe = b.zslice_idx(z_frame, copy=True)
    if b.nchannels > 1:
        if channel is not None:
            bframe = bframe[:, channel]
        else:
            bframe = np.sum(bframe.astype(np.float32), axis=1)
    if axis == 2:
        bframe = np.rot90(bframe, axes=(-1, -2))
    bframe = bframe[..., :overlap - max_dy, :]

    aslice = aslice.astype(np.float32)
    bframe = bframe.astype(np.float32)

    output_shape = np.array(aslice.shape) + np.array((0, 0, 2 * max_dx)) - np.array(bframe.shape) + 1
    output_shape[0] = aslice.shape[0]
    xcorr = np.zeros(output_shape)

    for i in range(xcorr.shape[0]):
        cc, max_loc = align_dog(aslice[i], bframe[0], 0, max_dx)
        xcorr[i] = cc

    shift = list(np.unravel_index(np.argmax(xcorr), xcorr.shape))
    score = xcorr[tuple(shift)]
    if score < 0 or score > 1:
        score = 0

    item['score'] = score
    item['dz'] = shift[0]
    item['dy'] = shift[1]
    item['dx'] = shift[2]

    return item


class Runner(object):
    def __init__(self):
        self.channel = None
        self.processing_list = None
        self.fut_q = None
        self.output_q = None
        self.input_folder = None
        self.output_file = None
        self.z_samples = None
        self.z_stride = None
        self.overlap_v = None
        self.overlap_h = None
        self.max_dx = None
        self.max_dy = None
        self.max_dz = None
        self.ascending_tiles_x = True
        self.ascending_tiles_y = True
        self.recursive = False
        self.equal_shape = False
        self.df = None
        self.fm = None
        self.px_size_xy = 1
        self.px_size_z = 1
        self.n_of_workers = None

    @property
    def overlap_dict(self):
        return {1: self.overlap_v, 2: self.overlap_h}

    def initialize_list(self):
        fm = FileMatrix(self.input_folder, self.ascending_tiles_x, self.ascending_tiles_y,
                        recursive=self.recursive, equal_shape=self.equal_shape)
        self.fm = fm

        if self.z_samples > 1 and self.z_stride is None:
            self.z_stride = ((fm.data_frame.iloc[0].nfrms - self.max_dz * self.z_samples) // self.z_samples).item()

        if self.z_stride is None:
            self.z_stride = 0

        stitch_X = {
            'axis': 2,
            'sort': ['Z', 'X', 'Y'],
            'groupby': 'Y'
        }

        stitch_Y = {
            'axis': 1,
            'sort': ['Z', 'Y', 'X'],
            'groupby': 'X'
        }

        mylist = []

        for s in fm.slices():
            df = fm.data_frame.loc[list(s.nodes())]
            for stitch_config in [stitch_X, stitch_Y]:
                view = df.sort_values(stitch_config['sort'],
                                      ascending=True)
                view = view.groupby(stitch_config['groupby'])

                for name, group in view:
                    Z_from = int(max(group['Z']))
                    Z_to = int(min(group['Z_end']))
                    tiles = list(group.itertuples())
                    atile = tiles[0]
                    for btile in tiles[1::]:
                        central_frame = (Z_to - Z_from) // 2
                        start_frame = (
                            central_frame
                            - (self.z_samples // 2 * self.z_stride)
                            + (0 if self.z_samples % 2 else self.z_stride // 2)
                        )
                        for i in range(0, self.z_samples):
                            mylist.append({
                                'aname': atile.Index,
                                'bname': btile.Index,
                                'z_frame': start_frame + i * self.z_stride,
                                'axis': stitch_config['axis'],
                            })
                        atile = btile
        self.processing_list = mylist

    def keep_filling_fut_queue(self):
        e = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_of_workers)

        for item in self.processing_list:
            self.fut_q.put(e.submit(worker, item, self.overlap_dict, self.channel, self.max_dz, self.max_dy, self.max_dx))

        self.fut_q.put(None)

    def output_worker(self):
        i = 1
        while True:
            fut = self.fut_q.get()
            if fut is None:
                self.fut_q.task_done()
                break

            item = fut.result()

            progress = 100 * i / len(self.processing_list)
            aname = item['aname']
            bname = item['bname']
            z_frame = item['z_frame']
            shift = [item['dz'], item['dy'], item['dx']]
            score = item['score']
            logger.info(f'{progress:.0f}%\t{aname}\t{bname}\t{z_frame}\t{shift}\t{score:.3f}')

            self.output_q.put(item)
            self.fut_q.task_done()

            i += 1

    def run(self):
        out_dir = os.path.dirname(os.path.abspath(self.output_file))
        if not os.access(out_dir, os.W_OK):
            raise ValueError('cannot write to {}'.format(self.output_file))

        self.initialize_list()
        self.fut_q = queue.Queue()
        self.output_q = queue.Queue()

        t = threading.Thread(target=self.output_worker)
        t.start()

        self.keep_filling_fut_queue()

        # block until all tasks are done
        self.fut_q.join()

        df = pd.DataFrame(self.output_q.queue)
        self.df = df

        xcorr_fm = XcorrFileMatrix.from_data(self.xcorr_options, self.df)
        xcorr_fm.aggregate_results()

        sdf = xcorr_fm.stitch_data_frame

        absolute_positions.compute_shift_vectors(self.fm.data_frame, sdf)
        absolute_positions.global_optimization(self.fm.data_frame, xcorr_fm)

        self.save_results_to_file()

        cols = ['score', 'dz', 'dy', 'dx']
        print(df[cols].describe())

    @property
    def xcorr_options(self):
        attrs = ['max_dx', 'max_dy', 'max_dz', 'overlap_v', 'overlap_h',
                 'ascending_tiles_x', 'ascending_tiles_y', 'px_size_xy',
                 'px_size_z', 'z_samples', 'z_stride']

        options = {}
        for attr in attrs:
            options[attr] = getattr(self, attr)

        return options

    def save_results_to_file(self):
        self.fm.save_to_yaml(self.output_file, 'w')

        with open(self.output_file, 'a') as f:
            yaml.dump(
                {
                    'xcorr-options': self.xcorr_options,
                    'xcorr': json.loads(self.df.to_json(orient='records')),
                    'fuser-options': {'abs_mode': ABS_MODE_MAXIMUM_SCORE},
                }, f, default_flow_style=False)


def main():
    arg = parse_args()

    r = Runner()

    keys = ['input_folder', 'output_file', 'channel', 'max_dx', 'max_dy',
            'max_dz', 'z_samples', 'z_stride', 'overlap_v', 'overlap_h',
            'ascending_tiles_x', 'ascending_tiles_y', 'px_size_xy',
            'px_size_z', 'n_of_workers', 'recursive', 'equal_shape']

    for key in keys:
        setattr(r, key, getattr(arg, key))

    t = time.time()
    r.run()
    elapsed = timedelta(seconds=time.time() - t)
    logger.info(f'elapsed  time: {elapsed}')


if __name__ == '__main__':
    main()
