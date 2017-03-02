import sys
import argparse

from stitcher import stitch


def main():
    parser = argparse.ArgumentParser(
        description='Stitch a pair of files',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_file1', help='input file (.dcimg)')
    parser.add_argument('input_file2', help='input file (.dcimg)')

    parser.add_argument('-f', type=int, help='frame', dest='frame',
                        required=True)
    parser.add_argument('-a', type=int, help='axis (1=Y, 2=X)', dest='axis',
                        choices=(1, 2), default=1)
    parser.add_argument('--overlap', type=int, help='overlap', default=600)

    args = parser.parse_args(sys.argv[1:])

    stitch(args.input_file1, args.input_file2, args.frame,
           args.frame + 5, args.overlap, axis=args.axis)


if __name__ == '__main__':
    main()
