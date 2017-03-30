import sys
import queue
import os.path
import argparse
import threading

import numpy as np

import colored
from colored import stylize

from .core import normxcorr2_fftw
from .filematrix import FileMatrix
from .inputfile import InputFile


def parse_args():
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
                    'the center of the stack, then take the result with the '
                    'maximum score')
    group.add_argument('-a', action='store_true',
                       help='instead of maximum score, take the average '
                            'result weighted by the score',
                       dest='average')

    group.add_argument('--z-samples', type=int, default=1, metavar='ZSAMP',
                       help='number of samples to take along Z')

    group.add_argument('--stride', type=int, default=200,
                       help='stride used for multiple Z sampling')

    parser.add_argument('-n', type=int, default=1,
                        help='number of parallel threads to use')

    return parser.parse_args(sys.argv[1:])


def build_queue(arg):
    fm = FileMatrix(arg.input_folder)
    fm.ascending_tiles_X = True
    fm.ascending_tiles_Y = False

    group_generators = [fm.tiles_along_X, fm.tiles_along_Y]
    stitch_axis = [2, 1]
    overlap = [arg.overlap_h, arg.overlap_v]

    q = queue.Queue()

    for group_generator, axis, overlap in zip(
                group_generators, stitch_axis, overlap):
        for group in group_generator:

            tile_generator = group.itertuples()

            atile = next(tile_generator)

            for btile in tile_generator:
                central_frame = atile.nfrms // 2
                start_frame = (central_frame
                               - (arg.z_samples // 2 * arg.stride)
                               + (0 if arg.z_samples % 2 else arg.stride // 2))
                for i in range(0, arg.z_samples):
                    z_frame = start_frame + i * arg.stride
                    params_dict = {
                        'aname': atile.filename,
                        'bname': btile.filename,
                        'z_frame': z_frame,
                        'axis': axis,
                        'overlap': overlap,
                        'max_shift_z': arg.max_dz,
                        'max_shift_y': arg.max_dy,
                        'max_shift_x': arg.max_dx
                    }
                    q.put(params_dict)
                atile = btile
    return q


def main():
    def worker():
        while True:
            item = data_queue.get()
            if item is None:
                break
            try:
                aname = item[0]
                bname = item[1]
                axis = item[2]
                alayer = item[3]
                blayer = item[4]

                xcorr = normxcorr2_fftw(alayer, blayer)

                shift = list(np.unravel_index(np.argmax(xcorr), xcorr.shape))
                score = xcorr[tuple(shift)]

                print(100 * (1 - q.qsize() / initial_queue_length))
                output_q.put([aname, bname, axis] + shift + [score])
            finally:
                data_queue.task_done()
                q.task_done()

    arg = parse_args()

    q = build_queue(arg)
    initial_queue_length = q.qsize()
    data_queue = queue.Queue(maxsize=int(arg.n * 2))
    output_q = queue.Queue()
    threads = []
    for i in range(arg.n):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        aname = item['aname']
        bname = item['bname']
        max_shift_z = item['max_shift_z']
        max_shift_y = item['max_shift_y']
        max_shift_x = item['max_shift_x']
        z_frame = item['z_frame']
        axis = item['axis']
        overlap = item['overlap']

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

        alayer = alayer.astype(np.float32)
        blayer = blayer.astype(np.float32)

        data_queue.put([aname, bname, axis, alayer, blayer])

    # block until all tasks are done
    data_queue.join()

    # stop workers
    for i in range(arg.n):
        data_queue.put(None)
    for t in threads:
        t.join()

    for el in list(output_q.queue):
        print(el)

if __name__ == '__main__':
    main()

    # print(
    #     stylize('stitching {} {}, axis = {}'.format(*temp, axis),
    #             colored.fg('magenta')))

    # results.append(ret)
    # print('results: ' + str(results))
    # results = np.array(results)
    # if arg.average:
    #     results = np.average(
    #         results, axis=0, weights=results[:, 3])
    # else:
    #     results = results[np.argmax(results[:, 3]), :]
    # print('results: ' + str(results))
