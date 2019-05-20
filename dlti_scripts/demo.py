#!/usr/bin/python3
import json
from pathlib import Path
import time
from typing import MutableMapping, Iterable, Tuple, List
import os

from funcy import post_processing, first, second, cat, lmap
from parso.python.tree import Name
import tensorflow as tf

from deeplearn.analysis.predictions import RecordWithPrediction
from deeplearn.analysis.util import RecordMode, read_unlabeled_dataset
from deeplearn.netmaker import make_charcnn, make_contextnet, make_occurencenet
from deeplearn.modules.model_trainer import ModelTrainer
from preprocessing.core.syntactic.extract import (
    get_all_untyped_syntactic_contexts)
from preprocessing.util import csv_write, csv_read, augment_except

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

OUTPATH = Path("/home/acalc79/synced/part-ii-project/out")
DATAPATH = Path("/home/acalc79/synced/part-ii-project/data/sets")
NETS = {"charcnn": ("charcnn-f-april",
                    "identifier-f-very-fine",
                    make_charcnn),
        "contextnet": ("contextnet-2-f-c-april",
                       "context-2-f-c-very-fine",
                       make_contextnet),
        "occurencenet": ("occurencenet-2-f-c-april",
                         "occurence-2-f-c-very-fine",
                         make_occurencenet)}
DEMODIR = Path("/home/acalc79/code/demo")
WORKDIR = DEMODIR.joinpath("tmp")
MODES = {'charcnn': RecordMode.IDENTIFIER,
         'contextnet': RecordMode.CONTEXT,
         'occurencenet': RecordMode.OCCURENCES}


@post_processing(dict)
def loadnets() -> Iterable[Tuple[str, ModelTrainer]]:
    for netname, (netdir, datadir, netmaker) in NETS.items():
        out_path = OUTPATH.joinpath(netdir)
        data_path = DATAPATH.joinpath(datadir)
        params = json.loads(out_path.joinpath('params.json').read_text())
        net = netmaker(params, data_path, 64,
                       out_dir=out_path, run_name="run0")
        net.load_weights("weights_{epoch}-optimizer")
        yield (netname, net)


def extract_file(source_path: Path):
    print(f"Extracting {source_path}")
    occurences: List[Iterable[Tuple[Name, Iterable[str]]]]
    occurences = get_all_untyped_syntactic_contexts(source_path, 2, True)
    # save_record_mapping
    csv_write(WORKDIR.joinpath(f"{source_path.stem}-names.csv"),
              ((name.value, name.line, name.column)
               for name in (min(map(first, contexts),
                                key=lambda nam: (nam.line, nam.column))
                            for contexts in occurences)))
    identifier_data_path\
        = WORKDIR.joinpath(f"{source_path.stem}-charcnn-data.csv")
    csv_write(identifier_data_path,
              ([nam.value] for nam, _ in map(first, occurences)))
    context_data_path\
        = WORKDIR.joinpath(f"{source_path.stem}-contextnet-data.csv")
    csv_write(context_data_path,
              map(second,
                  (min(contexts, key=lambda t: (t[0].line, t[0].column))
                   for contexts in occurences)))
    occurence_data_path\
        = WORKDIR.joinpath(f"{source_path.stem}-occurencenet-data.csv")
    csv_write(occurence_data_path,
              (cat(map(second, contexts)) for contexts in occurences))


@augment_except('netname')
def analyse_file(source_path: Path, out_path: Path,
                 net: ModelTrainer, netname: str) -> None:
    print(f"Analysing with {netname}")
    data_path = WORKDIR.joinpath(f"{source_path.stem}-{netname}-data.csv")
    predictions = lmap(list, net.full_true_predictions(data_path))
    records = map(RecordWithPrediction.from_record,
                  read_unlabeled_dataset(MODES[netname], 2, data_path),
                  predictions)
    names = csv_read(WORKDIR.joinpath(f"{source_path.stem}-names.csv"))
    out_path.write_text(''.join(
        f"{name} at ({line}, {col}): {rec.top_k()}\n"
        for (name, line, col), rec in zip(names, records)))


def main() -> None:
    nets = loadnets()
    print("Networks loaded")
    processed: MutableMapping[Path, float] = {}
    while True:
        source_files = [path for path in DEMODIR.glob("*.py")
                        if not path.stem.startswith('.')
                        if not 'flycheck' in path.stem]
        with_time = ((path, path.stat().st_mtime) for path in source_files)
        new_source_files = {path: mtime
                            for path, mtime in with_time
                            if mtime > processed.get(path, float('-inf'))}
        if not new_source_files:
            print(f"Alive, no new files, known files: {processed},"
                  f" all files: {source_files}")
        for source_file in new_source_files:
            for netname, net in nets.items():  # pylint: disable=no-member
                out_name = f"{source_file.stem}-{netname}.pyi"
                out_path = source_file.parent.joinpath(out_name)
                extract_file(source_file)
                analyse_file(source_file, out_path, net, netname)
        processed.update(new_source_files)
        time.sleep(10)


if __name__ == '__main__':
    main()
