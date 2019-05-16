#!/usr/bin/python3
from pathlib import Path
import re
from typing import Optional, Iterable

from funcy import (re_test, re_tester, some, second,
                   compose, re_find, ilen, partial)

from deeplearn.analysis.prediction_stats import PredictionStats
from deeplearn.analysis.util import redirect_stdout

OUTPATH = Path("/home/acalc79/synced/part-ii-project/out/")
DATAPATH = Path("/home/acalc79/synced/part-ii-project/data/sets/")


def get_corr(outpaths: Iterable[Path], path: Path) -> Optional[Path]:
    return some(compose(re_tester(path.stem), str), outpaths)


def single_main(mode, ctx_size, outpath, datapath, summary):
    msg = f"Outpath: {outpath},\nDatapath: {datapath}"
    assert outpath.is_dir() and datapath.is_dir(), msg
    vocab_path = datapath.joinpath("vocab.csv")
    data_paths = list(p for p in datapath.glob("*.csv")
                      if not re_test("train|val|vocab", p.stem))
    out_paths = tuple(outpath.glob("*.csv"))
    paths = list(filter(second, ((p, get_corr(out_paths, p))
                                 for p in data_paths)))
    pred_stats = PredictionStats(paths, mode, ctx_size,
                                 vocab_path, topk=(3, 5))
    if summary:
        pred_stats.show_summary(threshold=50)
    else:
        pred_stats.show_stats()


FLAGS = ('f', 'c', 'very', 'fine')
NETMAP = {'charcnn': 'identifier',
          'contextnet': 'context',
          'occurencenet': 'occurence',
          'baseline': 'identifier'}


def main(outpath, datapath):
    for net_path in filter(Path.is_dir, outpath.glob("*")):
        net_name = re_find(r"([a-z0-9\-]+)-[a-z]*$", net_path.stem)
        if not net_name:
            continue
        components = re.findall(r"[a-z0-9]+", net_name)
        assert len([c for c in components if c in NETMAP]) == 1, components
        mode = some(NETMAP[c] for c in components if c in NETMAP)
        if mode != 'identifier':
            assert ilen(filter(re_tester(r"^\d$"), components)) == 1,\
                f"{components}, {net_path}"
            ctx_size = int(some(re_tester(r"^\d$"), components))
        else:
            ctx_size = None
        flags = [c for c in components if c in FLAGS]
        dataname = f"{mode}"
        if ctx_size is not None:
            dataname += f"-{ctx_size}"
        if flags:
            dataname += "-" + "-".join(map(str, flags))
        dataname = f"{dataname}-very-fine"
        for run_path in filter(Path.is_dir, net_path.glob("*")):

            def getfun(category: str, **args):
                return redirect_stdout(
                    run_path.joinpath(f"test-{category}.txt"),
                    partial(single_main, **args))
            for fun in (getfun('summary', summary=True),
                        getfun('stat', summary=False)):
                fun(mode, ctx_size or 0, run_path, datapath.joinpath(dataname))


if __name__ == '__main__':
    main(OUTPATH, DATAPATH)
