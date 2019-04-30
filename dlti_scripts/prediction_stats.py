from pathlib import Path
import re
from typing import Optional, Iterable

from funcy import re_test, re_tester, some, second, compose, re_find, ilen

from deeplearn.analysis.prediction_stats import PredictionStats
from deeplearn.analysis.util import RecordMode, redirect_stdout

OUTPATH = Path("/home/acalc79/synced/part-ii-project/out/")
DATAPATH = Path("/home/acalc79/synced/part-ii-project/data/sets/")
SUMMARY = True


def get_corr(outpaths: Iterable[Path], path: Path) -> Optional[Path]:
    return some(compose(re_tester(path.stem), str), outpaths)


def single_main(mode, ctx_size, outpath, datapath):
    assert outpath.is_dir() and datapath.is_dir()
    vocab_path = datapath.joinpath("vocab.csv")
    data_paths = list(p for p in datapath.glob("*.csv")
                      if not re_test("train|val|vocab", p.stem))
    out_paths = tuple(outpath.glob("*.csv"))
    paths = list(filter(second, ((p, get_corr(out_paths, p))
                                 for p in data_paths)))
    pred_stats = PredictionStats(paths, mode, ctx_size,
                                 vocab_path, topk=(3, 5))
    if SUMMARY:
        pred_stats.show_summary(threshold=50)
    else:
        pred_stats.show_stats()


MODES = [m.value for m in RecordMode.__members__.values()]
FLAGS = ('f', 'c', 'very', 'fine')


def main(outpath, datapath):
    for net_path in filter(Path.is_dir, outpath.glob("*")):
        eval_path = net_path.joinpath("run_eval.sh")
        if not eval_path.is_file():
            continue
        net_name = re_find(r"DATA=.*?/([a-z0-9\-]+)\"?(?:\n|$)",
                           eval_path.read_text())
        components = re.findall(r"[a-z0-9]+", net_name)
        assert len([c for c in components if c in MODES]) == 1, components
        mode = some(c for c in components if c in MODES)
        if mode != 'identifier':
            assert ilen(filter(re_tester(r"^\d$"), components)) == 1,\
                f"{components}, {eval_path}"
            ctx_size = int(some(re_tester(r"^\d$"), components))
        else:
            ctx_size = None
        flags = [c for c in components if c in FLAGS]
        dataname = f"{mode}"
        if ctx_size is not None:
            dataname += f"-{ctx_size}"
        if flags:
            dataname += "-" + "-".join(map(str, flags))
        for run_path in filter(Path.is_dir, net_path.glob("*")):
            logpath = run_path.joinpath(
                f"test-{'summary' if SUMMARY else 'stat'}.txt")
            logging_main = redirect_stdout(logpath)(single_main)
            logging_main(mode, ctx_size or 0, run_path,
                         datapath.joinpath(dataname))


if __name__ == '__main__':
    main(OUTPATH, DATAPATH)
