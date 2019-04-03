import csv
import json
from pathlib import Path
import shutil
from typing import (Any, Callable, Hashable, Iterable, List, Mapping, TypeVar)
import unittest

from pytrie import Trie
from multiset import Multiset

__all__ = ['PROJDIR', 'DATADIR', 'TestWithOutDir',
           'as_test_path', 'csv_read', 'csv_write', 'for_all_cases']

PROJDIR = Path("/home/acalc79/synced/part-ii-project")
DATADIR = PROJDIR.joinpath("data")
UNITTEST_DIR = PROJDIR.joinpath("dlti_scripts", "dlti_tests", "unit_tests")
TMP_OUT = PROJDIR.joinpath("dlti_scripts", "dlti_tests", "tmp")

A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T')


# TODO: put all versions of csv_read/write in one location
# (requires solving import issues)
def csv_read(path: Path,
             constructor: Callable[[Iterable[List[str]]], T] = list) -> T:
    with path.open(newline='') as csvfile:
        return constructor(csv.reader(csvfile))


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


def test_data_for(script_name: str, filename_format: str) -> Iterable[Path]:
    i = 0
    while True:
        path = Path(script_name).parent.joinpath(filename_format.format(i))
        if not path.exists():
            break
        yield path
        i += 1


def as_test_path(location: str):
    return UNITTEST_DIR.joinpath(location)


class RichTestCase(unittest.TestCase):
    def assertEssentiallyEqual(self, actual: Any, expected: Any):
        func1, func2 = map(lambda n: self.KNOWN.get(type(n), None),
                           (actual, expected))
        if func1 and func2 and func1 is func2:
            func1(self, actual, expected)
        elif (not isinstance(actual, Iterable)
                and not isinstance(expected, Iterable)):
            self.assertEqual(actual, expected)
        else:
            raise ValueError("Unknown iterables: ({}, {})"
                             .format(type(actual), type(expected)))

    def assertSameOrdered(self, actual: Iterable[T], expected: Iterable[T]):
        actual_list = list(actual)
        expected_list = list(expected)
        common_len = min(len(actual_list), len(expected_list))
        for i in range(common_len):
            with self.subTest(i=i):
                self.assertEssentiallyEqual(actual_list[i], expected_list[i])
        self.assertFalse(
            actual_list[common_len:],
            msg=("\n\nFound unexpected elements:\n{}".format(actual_list)))
        self.assertFalse(
            expected_list[common_len:],
            msg=("\n\nExpected additional elements:\n{}".format(actual_list)))

    def assertSameElements(self, actual: Iterable[Hashable],
                           expected: Iterable[Hashable]):
        actual_set = Multiset(actual)
        expected_set = Multiset(expected)
        for elem in actual_set:
            self.assertTrue(elem in expected_set,
                            msg="\nFound unexpected {}".format(elem))
            expected_set.remove(elem, 1)
        self.assertFalse(
            expected_set,
            msg=("\n\nExpected additional elements:\n{}".format(expected_set)))

    def assertSameMapping(self, actual: Mapping[A, B],
                          expected: Mapping[A, B]):
        actual, expected = dict(actual), dict(expected)
        for key in set.intersection(set(actual.keys()), set(expected.keys())):
            with self.subTest(key=key):
                self.assertEssentiallyEqual(actual[key], expected[key])
            del actual[key], expected[key]
        self.assertFalse(actual, msg=("\n\nFound unexpected mappings:\n{}"
                                      .format(actual)))
        self.assertFalse(expected, msg=("\n\nExpected additional mappings:\n{}"
                                        .format(expected)))

    KNOWN: Mapping[type, Callable[['RichTestCase', Any, Any], None]]\
        = {set: assertSameElements,
           Multiset: assertSameElements,
           dict: assertSameMapping,
           Trie: assertSameMapping,
           list: assertSameOrdered,
           tuple: assertSameOrdered,
           range: assertSameOrdered,
           str: unittest.TestCase.assertEqual}


class TestWithOutDir(RichTestCase):
    def __init__(self, *args, out: Path = TMP_OUT, **kwargs):
        super().__init__(*args, **kwargs)
        self.out: Path = out

    def setUp(self):
        if self.out.exists():
            if self.out.is_dir():
                shutil.rmtree(self.out.resolve())
            else:
                self.out.unlink()
