import sys
sys.path.append('.')
sys.path.append('..')
import argparse

def parse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Discover frequent motifs \
            of variable length in time series.')
    parser.add_argument(
        '--min-sup', default=0.5, type=float, nargs='?',
        help='set minimum support \
            (default: %(default)s)')
    parser.add_argument(
        '-w', default=8, type=int, nargs='?',
        help='set Piecewise Aggregate Approximation window size \
            (default: %(default)s)')
    parser.add_argument(
        '-a', default=7, type=int, nargs='?',
        help='set Symbolic Aggregate Approximation alphabet size \
            (default: %(default)s)')
    parser.add_argument(
        '-l', default=3, type=int, nargs='?',
        help='minimal length of patterns to mine \
            (default: %(default)s)')
    parser.add_argument(
        '--topk', default=0, type=int, nargs='?',
        help='how many patterns to find the motifs of, finds all if 0 \
            (default: %(default)s)')
    parser.add_argument(
        '--plot', action='store_true',
        help='plot the original timeseries and the motifs')
    args = parser.parse_args()

    return args