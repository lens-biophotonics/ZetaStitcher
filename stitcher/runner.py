import queue
import argparse
import threading

import json
import yaml

import numpy as np
import pandas as pd

from .normxcorr import normxcorr2_fftw
from .filematrix import FileMatrix
from .inputfile import InputFile


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
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('input_folder', help='input folder')
    parser.add_argument('-o', type=str, default='stitch.yml',
                        dest='output_file', help='output file')
    parser.add_argument('-c', type=str, default='s', dest='channel',
                        choices=['r', 'g', 'b', 's'], help='color channel')
    parser.add_argument('-n', type=int, default=1,
                        help='number of parallel threads to use')

    group = parser.add_argument_group('maximum shifts')
    group.add_argument('--dz', type=int, required=True, dest='max_dz',
                       help='maximum allowed shift along Z')

    group.add_argument('--dy', type=int, required=True, dest='max_dy',
                       help='maximum allowed shift along y (the stitching '
                            'axis)')

    group.add_argument('--dx', type=int, required=True, dest='max_dx',
                       help='maximum allowed shift along x (lateral shift)')

    group = parser.add_argument_group('overlaps')
    group.add_argument('--overlap-h', type=int, required=True, metavar='OH',
                       help='overlap along the horizontal axis')

    group.add_argument('--overlap-v', type=int, required=True, metavar='OV',
                       help='overlap along the vertical axis')

    group = parser.add_argument_group(
        'multiple sampling along Z',
        description='Measure the optimal shift at different heights around '
                    'the center of the stack')

    group.add_argument('--z-samples', type=int, default=1, metavar='ZSAMP',
                       help='number of samples to take along Z')

    group.add_argument('--z-stride', type=int, default=200,
                       help='stride used for multiple Z sampling')

    group = parser.add_argument_group('tile ordering')
    group.add_argument('--iX', action='store_true', dest='invert_x',
                       help='invert tile ordering along X')
    group.add_argument('--iY', action='store_true', dest='invert_y',
                       help='invert tile ordering along Y')

    args = parser.parse_args()

    channels = {
        's': -2,  # sum
        'r': 0,
        'g': 1,
        'b': 2
    }

    args.channel = channels[args.channel]

    setattr(args, 'ascending_tiles_x', not args.invert_x)
    setattr(args, 'ascending_tiles_y', not args.invert_y)

    return args


class Runner(object):
    def __init__(self):
        self.channel = None
        self.q = None
        self.output_q = None
        self.data_queue = None
        self.initial_queue_length = None
        self.input_folder = None
        self.output_file = None
        self.z_samples = None
        self.z_stride = None
        self.overlap_v = None
        self.overlap_h = None
        self.max_dx = None
        self.max_dy = None
        self.max_dz = None
        self.compute_average = False
        self.ascending_tiles_x = True
        self.ascending_tiles_y = True
        self.df = None
        self.fm = None

    @property
    def overlap_dict(self):
        return {1: self.overlap_v, 2: self.overlap_h}

    def initialize_queue(self):
        fm = FileMatrix(self.input_folder)
        fm.ascending_tiles_x = self.ascending_tiles_x
        fm.ascending_tiles_y = self.ascending_tiles_y
        self.fm = fm

        stitch_X = {
            'axis': 2,
            'ascending': self.ascending_tiles_x,
            'sort': ['Z', 'X', 'Y'],
            'groupby': 'Y'
        }

        stitch_Y = {
            'axis': 1,
            'ascending': self.ascending_tiles_y,
            'sort': ['Z', 'Y', 'X'],
            'groupby': 'X'
        }

        q = queue.Queue()

        for s in fm.slices:
            df = fm.data_frame.loc[s.nodes()]
            for stitch_config in [stitch_X, stitch_Y]:
                view = df.sort_values(stitch_config['sort'],
                                      ascending=stitch_config['ascending'])
                view = view.groupby(stitch_config['groupby'])

                for name, group in view:
                    Z_from = max(group['Z'])
                    Z_to = min(group['Z_end'])
                    tiles = list(group.itertuples())
                    atile = tiles[0]
                    for btile in tiles[1::]:
                        central_frame = (Z_to - Z_from) // 2
                        start_frame = (
                            central_frame
                            - (self.z_samples // 2 * self.z_stride)
                            + (0 if self.z_samples % 2 else self.z_stride // 2))
                        for i in range(0, self.z_samples):
                            z_frame = start_frame + i * self.z_stride
                            params_dict = {
                                'aname': atile.Index,
                                'bname': btile.Index,
                                'z_frame': z_frame,
                                'axis': stitch_config['axis'],
                            }
                            q.put(params_dict)
                        atile = btile
        self.q = q

    def worker(self, initial_queue_length):
        while True:
            item = self.data_queue.get()
            if item is None:
                break
            try:
                aname = item[0]
                bname = item[1]
                axis = item[2]
                aslice = item[3]
                bframe = item[4]
                z_frame = item[5]

                xcorr = normxcorr2_fftw(aslice, bframe)

                shift = list(np.unravel_index(np.argmax(xcorr), xcorr.shape))
                score = xcorr[tuple(shift)]
                if score < 0 or score > 1:
                    score = 0

                print('{progress:.2f}%\t{aname}\t{bname}\t{z_frame}\t'
                      '{shift}\t{score}'.format(
                       progress=(
                           100 * (1 - self.q.qsize() / initial_queue_length)),
                       aname=aname, bname=bname, z_frame=z_frame, shift=shift,
                       score=score))
                self.output_q.put(
                    [aname, bname, axis, z_frame] + shift + [score])
            finally:
                self.data_queue.task_done()
                self.q.task_done()

    def keep_filling_data_queue(self):
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            aname = item['aname']
            bname = item['bname']
            z_frame = item['z_frame']
            axis = item['axis']
            overlap = self.overlap_dict[axis]

            a = InputFile(aname)
            b = InputFile(bname)

            a.channel = self.channel
            b.channel = self.channel

            z_min = z_frame - self.max_dz
            z_max = z_frame + self.max_dz + 1

            aslice = a.slice(z_min, z_max, copy=True)
            if axis == 2:
                aslice = np.rot90(aslice, axes=(-1, -2))
            aslice = aslice[..., -overlap:, :]

            bframe = b.slice_idx(z_frame, copy=True)
            if axis == 2:
                bframe = np.rot90(bframe, axes=(-1, -2))
            bframe = bframe[..., 0:overlap, :]

            bframe = bframe[..., :-self.max_dy, self.max_dx:-self.max_dx]

            aslice = aslice.astype(np.float32)
            bframe = bframe.astype(np.float32)

            self.data_queue.put([aname, bname, axis, aslice, bframe, z_frame])

    def run(self):
        self.initialize_queue()
        self.data_queue = queue.Queue(maxsize=int(arg.n * 2))
        self.output_q = queue.Queue()
        threads = []
        for i in range(arg.n):
            t = threading.Thread(target=self.worker, args=(self.q.qsize(),))
            t.start()
            threads.append(t)

        self.keep_filling_data_queue()

        # block until all tasks are done
        self.data_queue.join()

        # stop workers
        for i in range(arg.n):
            self.data_queue.put(None)
        for t in threads:
            t.join()

        df = pd.DataFrame(list(self.output_q.queue))
        df.columns = ['aname', 'bname', 'axis', 'z_frame', 'dz', 'dy', 'dx',
                      'score']
        self.df = df

        self.save_results_to_file()

        print(df[['dx', 'dy', 'dz', 'score']].describe())

    def save_results_to_file(self):
        self.fm.save_to_yaml(self.output_file, 'w')

        attrs = ['max_dx', 'max_dy', 'max_dz', 'overlap_v', 'overlap_h',
                 'ascending_tiles_x', 'ascending_tiles_y']

        options = {}
        for attr in attrs:
            options[attr] = getattr(self, attr)

        with open(self.output_file, 'a') as f:
            yaml.dump(
                {
                    'xcorr-options': options,
                    'xcorr': json.loads(self.df.to_json(orient='records'))
                }, f, default_flow_style=False)


if __name__ == '__main__':
    arg = parse_args()

    r = Runner()

    keys = ['input_folder', 'output_file', 'channel', 'max_dx', 'max_dy',
            'max_dz', 'z_samples', 'z_stride', 'overlap_v', 'overlap_h',
            'ascending_tiles_x', 'ascending_tiles_y']
    for key in keys:
        setattr(r, key, getattr(arg, key))

    r.run()
