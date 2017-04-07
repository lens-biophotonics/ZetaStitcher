import sys
import queue
import argparse
import threading

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

    group.add_argument('--z-stride', type=int, default=200,
                       help='stride used for multiple Z sampling')

    parser.add_argument('-n', type=int, default=1,
                        help='number of parallel threads to use')

    return parser.parse_args(sys.argv[1:])


def build_queue(input_folder, z_samples, z_stride):
    fm = FileMatrix(input_folder)
    fm.ascending_tiles_X = True
    fm.ascending_tiles_Y = False

    group_generators = [fm.tiles_along_X, fm.tiles_along_Y]
    stitch_axis = [2, 1]

    q = queue.Queue()

    for group_generator, axis in zip(group_generators, stitch_axis):
        for group in group_generator:

            tile_generator = group.itertuples()

            atile = next(tile_generator)

            for btile in tile_generator:
                central_frame = atile.nfrms // 2
                start_frame = (central_frame
                               - (z_samples // 2 * z_stride)
                               + (0 if z_samples % 2 else z_stride // 2))
                for i in range(0, z_samples):
                    z_frame = start_frame + i * z_stride
                    params_dict = {
                        'aname': atile.filename,
                        'bname': btile.filename,
                        'z_frame': z_frame,
                        'axis': axis,
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
                z_frame = item[5]

                xcorr = normxcorr2_fftw(alayer, blayer)

                shift = list(np.unravel_index(np.argmax(xcorr), xcorr.shape))
                score = xcorr[tuple(shift)]

                print('{progress:.2f}%\t{aname}\t{bname}\t{z_frame}\t'
                      '{shift}\t{score}'.format(
                       progress=100 * (1 - q.qsize() / initial_queue_length),
                       aname=aname, bname=bname, z_frame=z_frame, shift=shift,
                       score=score))
                output_q.put([aname, bname, axis] + shift + [score])
            finally:
                data_queue.task_done()
                q.task_done()

    def keep_filling_data_queue():
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            aname = item['aname']
            bname = item['bname']
            z_frame = item['z_frame']
            axis = item['axis']
            overlap = overlap_dict[axis]

            a = InputFile(aname)
            b = InputFile(bname)

            a.channel = 1
            b.channel = 1

            z_min = z_frame - arg.max_dz
            z_max = z_frame + arg.max_dz + 1

            alayer = a.layer(z_min, z_max)
            if axis == 2:
                alayer = np.rot90(alayer, axes=(1, 2))
            alayer = alayer[:, -overlap:, :]

            blayer = b.layer_idx(z_frame)
            if axis == 2:
                blayer = np.rot90(blayer, axes=(1, 2))
            blayer = blayer[:, 0:overlap, :]

            blayer = blayer[
                :, :-arg.max_dy, half_max_shift_x:-half_max_shift_x]

            alayer = alayer.astype(np.float32)
            blayer = blayer.astype(np.float32)

            data_queue.put([aname, bname, axis, alayer, blayer, z_frame])

    def aggregate_results():
        df = pd.DataFrame(list(output_q.queue))
        df.columns = ['aname', 'bname', 'axis', 'dz', 'dy', 'dx', 'score']

        if arg.average:
            view = df.groupby(['aname', 'bname', 'axis']).agg(
                lambda x: np.average(x, weights=df.loc[x.index, 'score']))
        else:
            view = df.groupby(['aname', 'bname', 'axis']).agg(
                lambda x: df.loc[np.argmax(df.loc[x.index, 'score']), x.name])

        view = view.reset_index()

        view.dz -= arg.max_dz
        for a in [1, 2]:
            indexes = (view['axis'] == a)
            view.loc[indexes, 'dy'] = overlap_dict[a] - view.loc[indexes, 'dy']
        view.dx -= half_max_shift_x

        return view

    arg = parse_args()
    overlap_dict = {1: arg.overlap_v, 2: arg.overlap_h}
    half_max_shift_x = arg.max_dx // 2

    q = build_queue(arg.input_folder, arg.z_samples, arg.z_stride)
    initial_queue_length = q.qsize()
    data_queue = queue.Queue(maxsize=int(arg.n * 2))
    output_q = queue.Queue()
    threads = []
    for i in range(arg.n):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    keep_filling_data_queue()

    # block until all tasks are done
    data_queue.join()

    # stop workers
    for i in range(arg.n):
        data_queue.put(None)
    for t in threads:
        t.join()

    view = aggregate_results()

    print(view)

    with open('stitch.json', 'w') as f:
        f.write(view.to_json(orient='records'))


if __name__ == '__main__':
    main()
