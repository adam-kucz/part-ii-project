import csv
import json
from pathlib import Path
import shutil
from typing import Iterable, List, Mapping, Tuple
import unittest

__all__ = ['PROJDIR', 'DATADIR', 'TestWithOutDir',
           'csv_read', 'csv_write', 'for_all_cases']

PROJDIR = Path("/home/acalc79/synced/part-ii-project")
DATADIR = PROJDIR.joinpath("data")
UNITTEST_DIR = PROJDIR.joinpath("dlti_scripts", "test", "unit_tests")
TMP_OUT = PROJDIR.joinpath("dlti_scripts", "test", "tmp")


# TODO: put all versions of csv_read/write in one location
# (requires solving import issues)
def csv_read(path: Path) -> List[List]:
    with path.open(newline='') as csvfile:
        return list(csv.reader(csvfile))


def csv_write(path: Path, rows: Iterable[List]):
    with path.open(mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(rows)


def for_all_cases(script_name: str,
                  filename_format: str = "test_case_{}.json",
                  as_json: bool = True):
    def call_on_all_cases(test_func):
        def test_full(self):
            cases = test_data_for(script_name, filename_format)
            for i, path in enumerate(cases):
                with self.subTest(test=i, case=path.name):
                    if as_json:
                        params = json.loads(path.read_text())
                        if isinstance(params, Mapping):
                            test_func(self, **params)
                        else:
                            test_func(self, params)
                    else:
                        test_func(self, path)
        return test_full
    return call_on_all_cases


def test_data_for(
        script_name: str, filename_format: str) -> Iterable[Tuple[Path, Path]]:
    location = Path(script_name.parent)
    print(location)
    return []


class TestWithOutDir(unittest.TestCase):
    def __init__(self, *args, out: Path = TMP_OUT, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.out: Path = out

    def setUp(self):
        if self.out.exists():
            if self.out.is_dir():
                shutil.rmtree(self.out.resolve())
            else:
                self.out.unlink()
