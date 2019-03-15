from argparse import ArgumentParser, Namespace
from pathlib import Path

import preprocessing.sets.data_splits as splits

DEFAULT_LOGPATH: Path = Path("log.txt")


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Split projects into groups')
    PARSER.add_argument('datadir', type=Path,
                        help='directory with projects')
    PARSER.add_argument('outdir', nargs='?', type=Path,
                        help='directory to save output files in')
    PARSER.add_argument('-s', '--splitpath', type=Path,
                        help="path to file specifying group sizes as "
                        "(name, min_fraction, max_fraction)")
    PARSER.add_argument('-l', '--logpath', default=None, type=Path,
                        help='file to save splits in, defaults to log.txt')
    PARSER.add_argument('-r', '--read_logfile', action='store_true',
                        help='read splits from log file')
    ARGS: Namespace = PARSER.parse_args()

    # pylint: disable=invalid-name
    OUTDIR = ARGS.outdir if ARGS.outdir else ARGS.datadir

    if ARGS.read_logfile:
        if not ARGS.logpath:
            print("You have to specify logpath "
                  "when using the -r, --read_logfile option")
        else:
            splits.from_logfile(ARGS.datadir, OUTDIR, ARGS.logpath)
    else:
        if not ARGS.splits:
            print("You have to specify splitpath "
                  "when not using the -r, --read_logfile option")
        else:
            splits.create_new(ARGS.splitpath, ARGS.datadir, OUTDIR,
                              ARGS.logpath or DEFAULT_LOGPATH)
