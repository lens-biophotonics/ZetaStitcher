import sys
import argparse

from math import floor

import numpy as np
import tensorflow as tf

from scipy import signal

from dcimg import DCIMGFile


def main():
    parser = argparse.ArgumentParser(
        description='Stitch a pair of files',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_file1', help='input file (.dcimg)')
    parser.add_argument('input_file2', help='input file (.dcimg)')

    parser.add_argument('-f', type=int, help='frame', dest='frame',
                        required=True)
    parser.add_argument('--overlap', type=int, help='overlap', default=600)

    args = parser.parse_args(sys.argv[1:])

    stitch(args.input_file1, args.input_file2, args.frame,
           args.frame + 5, args.overlap)


def window_filter(wz, wy, wx):
    gx = signal.hamming(wx)
    gy = signal.hamming(wy)
    gz = signal.hamming(wz)

    f = np.outer(gx, gy)
    f = np.outer(f, gz)

    f = np.reshape(f, (wx, wy, wz))
    f = f.T

    return f


def stitch(aname, bname, bottom, top, overlap):
    a = DCIMGFile(aname)
    b = DCIMGFile(bname)

    awhole = a.whole()
    awhole = awhole[bottom:top, -overlap:, :]

    bwhole = b.whole()
    bwhole = bwhole[bottom:top, 0:overlap, :]

    my_filter = window_filter(*awhole.shape).astype(np.float32)

    aph = tf.placeholder(dtype=tf.uint16, shape=awhole.shape)
    bph = tf.placeholder(dtype=tf.uint16, shape=bwhole.shape)

    at = tf.to_float(aph)
    bt = tf.to_float(bph)

    my_filter_t = tf.placeholder(dtype=tf.float32, shape=my_filter.shape)

    at = at * my_filter_t
    bt = bt * my_filter_t

    ac = tf.complex(at, tf.zeros_like(at))
    bc = tf.complex(bt, tf.zeros_like(bt))

    aft = tf.fft3d(ac)
    bft = tf.fft3d(bc)

    prod = aft * tf.conj(bft)
    real = tf.real(prod)
    imag = tf.imag(prod)
    prod = tf.complex(tf.to_double(real), tf.to_double(imag))
    prodnorm = tf.abs(prod)
    zl = tf.zeros_like(prodnorm)
    denom = tf.complex(prodnorm, zl)
    ratio = prod / denom
    real = tf.real(ratio)
    imag = tf.imag(ratio)

    ratio = tf.complex(tf.to_float(real), tf.to_float(imag))

    phase_corr = tf.abs(tf.real(tf.ifft3d(ratio)))

    with tf.Session() as sess:
        phase_corr = sess.run(phase_corr, feed_dict={aph: awhole, bph: bwhole,
                                                     my_filter_t: my_filter})
    tf.reset_default_graph()

    argmax = np.argmax(phase_corr)

    frame_size = phase_corr.shape[1] * phase_corr.shape[2]

    dz = floor(argmax / frame_size)

    dy = floor((argmax - dz * frame_size) / phase_corr.shape[2])

    dx = argmax - dz * frame_size - dy * phase_corr.shape[2]
    if dx > phase_corr.shape[2] / 2:
        dx -= phase_corr.shape[2]

    print('phase_corr.shape = {}'.format(phase_corr.shape))
    print('dx = {}, dy = {}, dz = {}'.format(dx, dy, dz))

    a.close()
    b.close()


if __name__ == '__main__':
    main()
