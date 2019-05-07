#!/usr/bin/python3
from pathlib import Path

from funcy import some, cut_suffix

from deeplearn.analysis.util import RecordMode, redirect_stdout
from deeplearn.analysis.cases import Cases

PROJPATH = Path("/home/acalc79/synced/part-ii-project/")
OUTPATH = PROJPATH.joinpath("out")
DATAPATH = PROJPATH.joinpath("data", "sets")
LOGPATH = PROJPATH.joinpath("logs")

NETNAME = "contextnet-1-f-c-april"
DATANAME = "context-1-f-c-very-fine"

LOGFILE = LOGPATH.joinpath(NETNAME, "run0", "cases-log.txt")


def main(dataset_path: Path, net_path: Path):
    elems = dataset_path.stem.split('-')
    mode = elems[0]
    ctx_size = (int(elems[1]) if mode in (RecordMode.OCCURENCES.value,
                                          RecordMode.CONTEXT.value)
                else 0)
    pred_path = some(
        net_path.joinpath("run0").glob("test_predictions_epoch*.csv"))
    assert pred_path, [p.stem for p in net_path.joinpath("run0").iterdir()]
    cases = Cases(dataset_path.joinpath("test.csv"), pred_path,
                  dataset_path.joinpath("vocab.csv"), mode, ctx_size)
    for option in sorted(cases.options, key=lambda f: f.__name__):
        option()
        print()


if __name__ == '__main__':
    redirect_stdout(LOGFILE, main,
                    DATAPATH.joinpath(DATANAME), OUTPATH.joinpath(NETNAME))
